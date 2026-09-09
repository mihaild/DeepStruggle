# High-Speed Vectorized Tournament Runner & Elo Calculator for Twilight Struggle AI.

import os
import sys
import time
import json
from typing import List, Dict, Any, Tuple, Optional, Sequence, Union
import numpy as np
import numpy.typing as npt
import torch

import ts_engine as ts
from tools.lib.player_agent import PlayerAgent, NeuralAgent, HeuristicAgent, RandomAgent, load_agent, resolve_device
from tools.lib.tournament_evaluator import classify_game_ending_reason


def categorize_flat_action_detailed(action_idx: int) -> str:
    """Categorizes a flat 212-dim action into human-readable semantic categories."""
    if action_idx == 211:
        return "CONFIRM_DONE / PASS"
    elif action_idx < 110:
        card_id = action_idx + 1
        scoring_ids = {1, 2, 3, 37, 38, 39, 81}
        card_type = "Scoring Card" if card_id in scoring_ids else "Normal Card"
        return f"SELECT_CARD ({card_type})"
    elif action_idx < 114:
        modes = ["EVENT", "OPS", "SPACE", "PASS"]
        return f"SELECT_PLAY_MODE ({modes[action_idx - 110]})"
    elif action_idx < 116:
        timings = ["OPS_FIRST", "EVENT_FIRST"]
        return f"CHOOSE_TIMING_BRANCH ({timings[action_idx - 114]})"
    elif action_idx < 119:
        op_modes = ["INFLUENCE", "COUP", "REALIGN"]
        return f"SELECT_OP_MODE ({op_modes[action_idx - 116]})"
    elif action_idx < 203:
        return "POINT_NODE (Country)"
    elif action_idx < 211:
        return f"CHOOSE_BRANCH ({action_idx - 203})"
    else:
        return "UNKNOWN"


def _obs_for(
    agent: NeuralAgent,
    runner: ts.VectorizedBatchRunner,
    obs_batch: npt.NDArray[np.float32],
    indices: npt.NDArray[np.int64],
    d_players: npt.NDArray[np.int64],
    mixed: bool,
) -> npt.NDArray[np.float32]:
    """The observations `agent` should see for `indices`.

    Same-layout matchups read the runner's batched buffer directly. Mixed ones re-extract from the
    states at the agent's own layout, because a model reads fixed slices and would silently
    misread a buffer of the wrong width rather than fail.
    """
    if not mixed:
        return obs_batch[indices]
    layout = str(getattr(agent, "layout", "legacy"))
    flags = int(getattr(agent, "obs_flags", 0))
    rows = []
    for idx in indices:
        st = runner.get_state(int(idx))
        rows.append(np.asarray(
            ts.extract_observation(st, ts.Player(int(d_players[idx])), layout=layout,
                                   flags=flags),
            dtype=np.float32))
    return np.stack(rows) if rows else np.zeros((0, getattr(agent, "obs_size", 4293)),
                                                dtype=np.float32)


def _assert_width(agent: NeuralAgent, obs: npt.NDArray[np.float32]) -> None:
    """A model silently misreads an observation of the wrong width; say so instead."""
    want = getattr(agent, "obs_size", None)
    if want is not None and obs.shape[1] != want:
        raise ValueError(
            f"{agent.name} expects an observation of width {want} but the runner produced "
            f"{obs.shape[1]}. Slicing hides this: the network would read the wrong regions and "
            f"play badly rather than fail.")

class BatchMatchRunner:
    """Runs 2 * games_per_side games between two PlayerAgents in parallel via C++ VectorizedBatchRunner."""

    @staticmethod
    def play_parallel_matchup(
        agent_a: PlayerAgent,
        agent_b: PlayerAgent,
        games_per_side: int = 1000,
        batch_chunk_size: int = 1000,
        base_seed: int = 10000,
        device: Optional[Union[torch.device, str]] = None,
        max_steps: int = 2500,
        temperature: float = 0.1,
        deterministic: Optional[bool] = None,
        start_states: Optional[Sequence["ts.GameState"]] = None,
        track_choices: bool = False,
        log_games_file: Optional[str] = None,
        auto_advance: bool = False,
    ) -> Dict[str, Any]:
        """Play a matchup batched. Action selection matches NeuralAgent.select_action.

        temperature/deterministic are the same contract as the one-game-at-a-time path in
        TournamentEvaluator.play_matchup: sampling at the given temperature unless it is low
        enough to be indistinguishable from an argmax. This used to be hard-coded to
        deterministic=True here while the sequential path sampled at 0.1, so the two
        disagreed on win rate (0.450 vs 0.610 on one 100-game matchup) and could not be
        swapped for one another.
        """
        dev = resolve_device(device)
        greedy = (temperature <= 0.05) if deterministic is None else deterministic

        # One runner emits one observation layout, so where the two agents disagree about which
        # they want, at most one of them can be served from its batched buffer. Such a pairing is
        # not a shape error the runner would raise on -- a model reads fixed slices, so it misreads
        # a wrong-width observation silently and just plays badly -- so mixed matchups take the
        # per-state path in _obs_for below, and the runner's own layout stops mattering.
        # The engine flags count as part of the layout here: two agents at the same width but
        # different flags see different observations, and one runner emits only one of them.
        layouts = {(getattr(a, "layout", "legacy"), int(getattr(a, "obs_flags", 0)))
                   for a in (agent_a, agent_b) if getattr(a, "model", None) is not None}
        mixed_layouts = len(layouts) > 1
        runner_layout, runner_flags = ("legacy", 0) if mixed_layouts else (
            layouts.pop() if layouts else ("legacy", 0))

        # Resume from supplied positions instead of dealing fresh games. Each position is
        # played twice with the sides swapped, which is the same pairing the seeded path
        # uses: both copies resume from one pre-deal state, so they draw the same cards and
        # deal luck cancels between the halves rather than adding variance to the result.
        if start_states is not None:
            games_per_side = len(start_states)
        if games_per_side <= 0:
            # Fail with the reason rather than a ZeroDivisionError from a chunk size of zero
            # further down. Callers that mean "no evaluation" should not call at all.
            raise ValueError(
                f"games_per_side must be positive, got {games_per_side}; "
                f"to skip evaluation, do not call play_parallel_matchup")
        total_games = games_per_side * 2
        half_per_chunk = min(games_per_side, batch_chunk_size // 2)
        chunk_size = half_per_chunk * 2

        a_wins = 0
        b_wins = 0
        draws = 0

        a_us_wins = 0
        a_us_losses = 0
        a_us_draws = 0

        a_ussr_wins = 0
        a_ussr_losses = 0
        a_ussr_draws = 0

        all_steps = []
        all_turns = []
        all_vps = []

        causes_loss_us: Dict[str, int] = {}
        causes_loss_ussr: Dict[str, int] = {}
        causes_all: Dict[str, int] = {}

        total_micro_us = 0
        total_micro_ussr = 0
        single_choice_us = 0
        single_choice_ussr = 0

        category_counts_us: Dict[str, int] = {}
        category_counts_ussr: Dict[str, int] = {}

        if log_games_file:
            os.makedirs(os.path.dirname(os.path.abspath(log_games_file)), exist_ok=True)
            with open(log_games_file, "w", encoding="utf-8") as f_init:
                pass

        t0 = time.time()
        num_chunks = (total_games + chunk_size - 1) // chunk_size

        for chunk_idx in range(num_chunks):
            cur_half = min(games_per_side - (chunk_idx * half_per_chunk), half_per_chunk)
            if cur_half <= 0:
                break
            cur_games = cur_half * 2
            seed_start = base_seed + (chunk_idx * chunk_size)

            runner = ts.VectorizedBatchRunner(cur_games, seed_start, runner_layout,
                                              runner_flags)
            # Paired deals: env i and env i + cur_half are the same matchup with the sides
            # swapped, so give them the same seed and therefore the same shuffle. Deal luck
            # then cancels between the halves rather than adding variance to the result.
            if start_states is not None:
                offset = chunk_idx * half_per_chunk
                for i in range(cur_half):
                    pos = start_states[offset + i]
                    runner.set_state(i, pos)
                    runner.set_state(i + cur_half, pos)
            else:
                for i in range(cur_half):
                    paired_seed = seed_start + i
                    runner.reset_game(i, paired_seed)
                    runner.reset_game(i + cur_half, paired_seed)
            runner.refresh_all()
            active = np.ones(cur_games, dtype=bool)
            steps = 0

            # Temporary arrays for current chunk
            chunk_utils = np.zeros(cur_games, dtype=np.float32)
            chunk_vps = np.zeros(cur_games, dtype=np.int32)
            chunk_turns = np.zeros(cur_games, dtype=np.int32)
            chunk_steps = np.zeros(cur_games, dtype=np.int32)
            chunk_causes = [""] * cur_games

            chunk_ussr_total = np.zeros(cur_games, dtype=np.int32)
            chunk_ussr_single = np.zeros(cur_games, dtype=np.int32)
            chunk_us_total = np.zeros(cur_games, dtype=np.int32)
            chunk_us_single = np.zeros(cur_games, dtype=np.int32)

            while np.any(active) and steps < max_steps:
                obs = runner.get_observations()
                masks = runner.get_action_masks()
                d_players = np.array(runner.get_decision_players())
                terms = np.array(runner.get_terminals())

                newly_finished = active & terms
                if np.any(newly_finished):
                    for idx in np.where(newly_finished)[0]:
                        st = runner.get_state(int(idx))
                        chunk_utils[idx] = float(ts.Engine.get_terminal_utility(st))
                        chunk_vps[idx] = int(st.victory_points)
                        chunk_turns[idx] = int(st.turn)
                        chunk_steps[idx] = steps
                        chunk_causes[idx] = classify_game_ending_reason(st)
                    active = active & (~terms)

                if not np.any(active):
                    break

                if track_choices:
                    active_indices = np.where(active)[0]
                    if len(active_indices) > 0:
                        active_d_players = d_players[active_indices]
                        valid_counts = np.count_nonzero(masks[active_indices], axis=1)

                        is_ussr = (active_d_players == -1)
                        is_us = (active_d_players == 1)

                        ussr_idxs = active_indices[is_ussr]
                        us_idxs = active_indices[is_us]

                        ussr_vcounts = valid_counts[is_ussr]
                        us_vcounts = valid_counts[is_us]

                        chunk_ussr_total[ussr_idxs] += 1
                        chunk_us_total[us_idxs] += 1

                        ussr_single_mask = (ussr_vcounts == 1)
                        us_single_mask = (us_vcounts == 1)

                        chunk_ussr_single[ussr_idxs[ussr_single_mask]] += 1
                        chunk_us_single[us_idxs[us_single_mask]] += 1

                        for s_idx in ussr_idxs[ussr_single_mask]:
                            act_idx = int(np.argmax(masks[s_idx]))
                            cat = categorize_flat_action_detailed(act_idx)
                            category_counts_ussr[cat] = category_counts_ussr.get(cat, 0) + 1

                        for s_idx in us_idxs[us_single_mask]:
                            act_idx = int(np.argmax(masks[s_idx]))
                            cat = categorize_flat_action_detailed(act_idx)
                            category_counts_us[cat] = category_counts_us.get(cat, 0) + 1

                actions = np.zeros(cur_games, dtype=np.int32)

                # Partition: env 0..cur_half-1 -> (A is USSR, B is US); env cur_half..cur_games-1 -> (B is USSR, A is US)
                is_a_turn = active & (
                    ((np.arange(cur_games) < cur_half) & (d_players == -1)) |
                    ((np.arange(cur_games) >= cur_half) & (d_players == 1))
                )
                is_b_turn = active & (
                    ((np.arange(cur_games) < cur_half) & (d_players == 1)) |
                    ((np.arange(cur_games) >= cur_half) & (d_players == -1))
                )

                # Agent A Action Selection
                if np.any(is_a_turn):
                    a_indices = np.where(is_a_turn)[0]
                    if isinstance(agent_a, NeuralAgent):
                        a_obs = _obs_for(agent_a, runner, obs, a_indices, d_players,
                                         mixed_layouts)
                        _assert_width(agent_a, a_obs)
                        obs_t = torch.from_numpy(a_obs).float().to(dev)
                        mask_t = torch.from_numpy(masks[a_indices]).to(dev)
                        with torch.no_grad():
                            act_t, _, _, _, _ = agent_a.model.sample_action(obs_t, mask_t, temperature=temperature, deterministic=greedy)
                        actions[a_indices] = act_t.cpu().numpy()
                    elif hasattr(agent_a, "select_action"):
                        for idx in a_indices:
                            st = runner.get_state(int(idx))
                            actions[idx] = agent_a.select_action(st, ts.Player(int(d_players[idx])), temperature=temperature)
                    else:  # RandomAgent, or anything without a state-based interface
                        for idx in a_indices:
                            leg = np.where(masks[idx] > 0)[0]
                            actions[idx] = np.random.choice(leg) if len(leg) > 0 else 0

                # Agent B Action Selection
                if np.any(is_b_turn):
                    b_indices = np.where(is_b_turn)[0]
                    if isinstance(agent_b, NeuralAgent):
                        b_obs = _obs_for(agent_b, runner, obs, b_indices, d_players,
                                         mixed_layouts)
                        _assert_width(agent_b, b_obs)
                        obs_t = torch.from_numpy(b_obs).float().to(dev)
                        mask_t = torch.from_numpy(masks[b_indices]).to(dev)
                        with torch.no_grad():
                            act_t, _, _, _, _ = agent_b.model.sample_action(obs_t, mask_t, temperature=temperature, deterministic=greedy)
                        actions[b_indices] = act_t.cpu().numpy()
                    elif hasattr(agent_b, "select_action"):
                        for idx in b_indices:
                            st = runner.get_state(int(idx))
                            actions[idx] = agent_b.select_action(st, ts.Player(int(d_players[idx])), temperature=temperature)
                    else:  # RandomAgent, or anything without a state-based interface
                        for idx in b_indices:
                            leg = np.where(masks[idx] > 0)[0]
                            actions[idx] = np.random.choice(leg) if len(leg) > 0 else 0

                runner.step_flat_all(actions.tolist(), auto_advance=auto_advance)
                steps += 1

            # Accumulate Chunk Results
            for idx in range(cur_games):
                a_is_ussr = (idx < cur_half)
                term_util = chunk_utils[idx]
                vp = chunk_vps[idx]
                turn = chunk_turns[idx]
                reason = chunk_causes[idx] or "Early Termination"

                causes_all[reason] = causes_all.get(reason, 0) + 1
                all_steps.append(chunk_steps[idx])
                all_turns.append(turn)

                vp_for_a = -vp if a_is_ussr else vp
                all_vps.append(vp_for_a)

                a_won = (term_util > 0 and not a_is_ussr) or (term_util < 0 and a_is_ussr)
                b_won = (term_util < 0 and not a_is_ussr) or (term_util > 0 and a_is_ussr)

                if a_won:
                    a_wins += 1
                    if a_is_ussr:
                        a_ussr_wins += 1
                    else:
                        a_us_wins += 1
                elif b_won:
                    b_wins += 1
                    clean_reason = reason
                    if a_is_ussr:
                        a_ussr_losses += 1
                        causes_loss_ussr[clean_reason] = causes_loss_ussr.get(clean_reason, 0) + 1
                    else:
                        a_us_losses += 1
                        causes_loss_us[clean_reason] = causes_loss_us.get(clean_reason, 0) + 1
                else:
                    draws += 1
                    if a_is_ussr:
                        a_ussr_draws += 1
                    else:
                        a_us_draws += 1

            if track_choices:
                total_micro_ussr += int(np.sum(chunk_ussr_total))
                total_micro_us += int(np.sum(chunk_us_total))
                single_choice_ussr += int(np.sum(chunk_ussr_single))
                single_choice_us += int(np.sum(chunk_us_single))

            if log_games_file:
                with open(log_games_file, "a", encoding="utf-8") as f_log:
                    for idx in range(cur_games):
                        a_is_ussr = (idx < cur_half)
                        ussr_agent = agent_a.name if a_is_ussr else agent_b.name
                        us_agent = agent_b.name if a_is_ussr else agent_a.name
                        term_util = chunk_utils[idx]
                        winner = "USSR" if term_util < 0 else ("US" if term_util > 0 else "DRAW")

                        g_idx = chunk_idx * chunk_size + idx + 1
                        m_ussr_tot = int(chunk_ussr_total[idx])
                        m_ussr_sgl = int(chunk_ussr_single[idx])
                        m_us_tot = int(chunk_us_total[idx])
                        m_us_sgl = int(chunk_us_single[idx])
                        m_all_tot = m_ussr_tot + m_us_tot
                        m_all_sgl = m_ussr_sgl + m_us_sgl

                        entry = {
                            "game_index": g_idx,
                            "seed": seed_start + (idx % cur_half),
                            "ussr_agent": ussr_agent,
                            "us_agent": us_agent,
                            "winner": winner,
                            "victory_points": int(chunk_vps[idx]),
                            "turn": int(chunk_turns[idx]),
                            "steps": int(chunk_steps[idx]),
                            "cause": chunk_causes[idx] or "Early Termination",
                            "ussr_total_micro_actions": m_ussr_tot,
                            "ussr_single_choice_micro_actions": m_ussr_sgl,
                            "ussr_single_choice_pct": round(m_ussr_sgl / max(1, m_ussr_tot) * 100.0, 2),
                            "us_total_micro_actions": m_us_tot,
                            "us_single_choice_micro_actions": m_us_sgl,
                            "us_single_choice_pct": round(m_us_sgl / max(1, m_us_tot) * 100.0, 2),
                            "total_micro_actions": m_all_tot,
                            "total_single_choice_micro_actions": m_all_sgl,
                            "total_single_choice_pct": round(m_all_sgl / max(1, m_all_tot) * 100.0, 2),
                        }
                        f_log.write(json.dumps(entry) + "\n")

        elapsed = time.time() - t0

        res: Dict[str, Any] = {
            "agent_a": agent_a.name,
            "agent_b": agent_b.name,
            "total_games": total_games,
            "games_per_side": games_per_side,
            "a_wins": a_wins,
            "b_wins": b_wins,
            "draws": draws,
            "win_rate_a": float(a_wins / max(1, total_games)),
            "win_rate_b": float(b_wins / max(1, total_games)),
            "a_wins_as_us": a_us_wins,
            "a_losses_as_us": a_us_losses,
            "a_draws_as_us": a_us_draws,
            "win_rate_a_as_us": float(a_us_wins / max(1, games_per_side)),
            "a_wins_as_ussr": a_ussr_wins,
            "a_losses_as_ussr": a_ussr_losses,
            "a_draws_as_ussr": a_ussr_draws,
            "win_rate_a_as_ussr": float(a_ussr_wins / max(1, games_per_side)),
            "avg_steps": float(np.mean(all_steps)) if all_steps else 0.0,
            "avg_turn": float(np.mean(all_turns)) if all_turns else 0.0,
            "avg_vp_margin_a": float(np.mean(all_vps)) if all_vps else 0.0,
            "causes_loss_us": causes_loss_us,
            "causes_loss_ussr": causes_loss_ussr,
            "causes_all": causes_all,
            "elapsed_seconds": elapsed,
        }

        if track_choices:
            tot_us = total_micro_us
            tot_ussr = total_micro_ussr
            tot_combined = tot_us + tot_ussr
            single_comb = single_choice_us + single_choice_ussr

            res["choice_stats"] = {
                "us_total_micro_actions": tot_us,
                "us_single_choice_micro_actions": single_choice_us,
                "us_single_choice_pct": float((single_choice_us / max(1, tot_us)) * 100.0),
                "ussr_total_micro_actions": tot_ussr,
                "ussr_single_choice_micro_actions": single_choice_ussr,
                "ussr_single_choice_pct": float((single_choice_ussr / max(1, tot_ussr)) * 100.0),
                "overall_total_micro_actions": tot_combined,
                "overall_single_choice_micro_actions": single_comb,
                "overall_single_choice_pct": float((single_comb / max(1, tot_combined)) * 100.0),
                "avg_per_game": {
                    "us_total": float(tot_us / max(1, total_games)),
                    "us_single": float(single_choice_us / max(1, total_games)),
                    "ussr_total": float(tot_ussr / max(1, total_games)),
                    "ussr_single": float(single_choice_ussr / max(1, total_games)),
                    "combined_total": float(tot_combined / max(1, total_games)),
                    "combined_single": float(single_comb / max(1, total_games)),
                },
                "category_counts_us": category_counts_us,
                "category_counts_ussr": category_counts_ussr,
            }

        return res


def compute_mle_elo(
    model_names: List[str],
    win_matrix: np.ndarray,
    total_matrix: np.ndarray,
    anchor_model: str = "HeuristicBot",
    anchor_elo: float = 1500.0,
    iterations: int = 1000,
    lr: float = 0.01,
) -> Dict[str, float]:
    """Computes Bradley-Terry Maximum Likelihood Elo ratings anchored to a reference model."""
    M = len(model_names)
    ratings = np.ones(M, dtype=np.float64) * 1500.0

    for _ in range(iterations):
        grad = np.zeros(M, dtype=np.float64)
        for i in range(M):
            for j in range(M):
                if i == j or total_matrix[i, j] == 0:
                    continue
                w_ij = win_matrix[i, j]
                w_ji = win_matrix[j, i]
                d_ij = total_matrix[i, j] - w_ij - w_ji
                # Win = 1.0, Draw = 0.5
                score_i = w_ij + 0.5 * d_ij
                n_ij = total_matrix[i, j]

                gamma_i = np.exp(ratings[i] / 400.0 * np.log(10))
                gamma_j = np.exp(ratings[j] / 400.0 * np.log(10))
                e_ij = gamma_i / (gamma_i + gamma_j)

                grad[i] += (score_i - n_ij * e_ij)

        # Update unanchored ratings
        ratings += lr * grad

        # Re-anchor to baseline
        if anchor_model in model_names:
            anchor_idx = model_names.index(anchor_model)
            ratings += (anchor_elo - ratings[anchor_idx])

    return {model_names[i]: float(ratings[i]) for i in range(M)}
