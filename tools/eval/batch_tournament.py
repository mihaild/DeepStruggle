# High-Speed Vectorized Tournament Runner & Elo Calculator for Twilight Struggle AI.

import os
import sys
import time
import json
from typing import List, Dict, Any, Tuple, Optional
import numpy as np
import torch

import ts_engine as ts
from tools.eval.player_agent import PlayerAgent, NeuralAgent, HeuristicAgent, RandomAgent, load_agent, resolve_device
from tools.eval.tournament_evaluator import classify_game_ending_reason


class BatchMatchRunner:
    """Runs 2 * games_per_side games between two PlayerAgents in parallel via C++ VectorizedBatchRunner."""

    @staticmethod
    def play_parallel_matchup(
        agent_a: PlayerAgent,
        agent_b: PlayerAgent,
        games_per_side: int = 1000,
        batch_chunk_size: int = 1000,
        base_seed: int = 10000,
        device: Optional[torch.device] = None,
        max_steps: int = 2500,
    ) -> Dict[str, Any]:
        dev = resolve_device(device)
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

        t0 = time.time()
        num_chunks = (total_games + chunk_size - 1) // chunk_size

        for chunk_idx in range(num_chunks):
            cur_half = min(games_per_side - (chunk_idx * half_per_chunk), half_per_chunk)
            if cur_half <= 0:
                break
            cur_games = cur_half * 2
            seed_start = base_seed + (chunk_idx * chunk_size)

            runner = ts.VectorizedBatchRunner(cur_games, seed_start)
            active = np.ones(cur_games, dtype=bool)
            steps = 0

            # Temporary arrays for current chunk
            chunk_utils = np.zeros(cur_games, dtype=np.float32)
            chunk_vps = np.zeros(cur_games, dtype=np.int32)
            chunk_turns = np.zeros(cur_games, dtype=np.int32)
            chunk_steps = np.zeros(cur_games, dtype=np.int32)
            chunk_causes = [""] * cur_games

            while np.any(active) and steps < max_steps:
                obs = runner.get_observations()
                masks = runner.get_action_masks()
                d_players = np.array(runner.get_decision_players())
                terms = np.array(runner.get_terminals())

                newly_finished = active & terms
                if np.any(newly_finished):
                    for idx in np.where(newly_finished)[0]:
                        st = runner.get_state(idx)
                        chunk_utils[idx] = float(ts.Engine.get_terminal_utility(st))
                        chunk_vps[idx] = int(st.victory_points)
                        chunk_turns[idx] = int(st.turn)
                        chunk_steps[idx] = steps
                        chunk_causes[idx] = classify_game_ending_reason(st)
                    active = active & (~terms)

                if not np.any(active):
                    break

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
                        obs_t = torch.from_numpy(obs[a_indices]).float().to(dev)
                        mask_t = torch.from_numpy(masks[a_indices]).to(dev)
                        with torch.no_grad():
                            act_t, _, _, _, _ = agent_a.model.sample_action(obs_t, mask_t, temperature=0.1, deterministic=True)
                        actions[a_indices] = act_t.cpu().numpy()
                    elif isinstance(agent_a, HeuristicAgent):
                        for idx in a_indices:
                            st = runner.get_state(idx)
                            actions[idx] = agent_a.select_action(st, ts.Player(d_players[idx]))
                    else:  # Random
                        for idx in a_indices:
                            leg = np.where(masks[idx] > 0)[0]
                            actions[idx] = np.random.choice(leg) if len(leg) > 0 else 0

                # Agent B Action Selection
                if np.any(is_b_turn):
                    b_indices = np.where(is_b_turn)[0]
                    if isinstance(agent_b, NeuralAgent):
                        obs_t = torch.from_numpy(obs[b_indices]).float().to(dev)
                        mask_t = torch.from_numpy(masks[b_indices]).to(dev)
                        with torch.no_grad():
                            act_t, _, _, _, _ = agent_b.model.sample_action(obs_t, mask_t, temperature=0.1, deterministic=True)
                        actions[b_indices] = act_t.cpu().numpy()
                    elif isinstance(agent_b, HeuristicAgent):
                        for idx in b_indices:
                            st = runner.get_state(idx)
                            actions[idx] = agent_b.select_action(st, ts.Player(d_players[idx]))
                    else:  # Random
                        for idx in b_indices:
                            leg = np.where(masks[idx] > 0)[0]
                            actions[idx] = np.random.choice(leg) if len(leg) > 0 else 0

                runner.step_flat_all(actions)
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
                    clean_reason = "Held scoring" if reason.startswith("Held scoring") else reason
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

        elapsed = time.time() - t0

        return {
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
