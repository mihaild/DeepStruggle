# High-Speed Vectorized Tournament Runner & Elo Calculator for Twilight Struggle AI.

import os
import time
import json
from dataclasses import dataclass, field
from typing import (Any, Callable, Dict, List, NotRequired, Optional, Sequence, Tuple,
                    TypedDict, Union)
import numpy as np
import numpy.typing as npt
import torch

import ts_engine as ts
from bindings.action_encoder import ActionEncoder
from tools.lib.game_step import IllegalActionError
from tools.lib.leaderboard import _Binomial, _newton
from tools.lib.openings import ScriptedSetupOverride
from tools.lib.scripted_rules import SCRIPTS
from bindings.ts_env import CANONICAL_LEARNED, agent_view, give_run_priorities, refuse_learned_order
from tools.lib.player_agent import BatchActor, BatchSelector, NeuralAgent, PlayerAgent, resolve_device
from ai.game_length import ply as game_ply
from ai.search.turn_pricing import Candidate, price, priceable
from tools.lib.game_ending import classify_game_ending_reason


#: The scoring cards' ids, from the engine's card table rather than a hand-kept list (which had
#: listed 39, Arms Race, and missed 79, Africa Scoring).
SCORING_CARD_IDS: frozenset[int] = frozenset(
    c for c in range(1, 111) if bool(ts.CardData.get_card_info(c)["is_scoring"]))



def _action_view_code(agent: Any) -> int:
    """The runner's action view for an agent: 0 E4, 1 merged (E4.1, P23), 2 macro (P42)."""
    v = agent_view(agent)
    return 1 if v.merged_influence else 2 if v.macro_influence else 0

def categorize_flat_action_detailed(action_idx: int) -> str:
    """Categorizes a flat action into human-readable semantic categories."""
    if action_idx == ActionEncoder.CONFIRM_DONE_INDEX:
        return "CONFIRM_DONE / PASS"
    elif action_idx < 110:
        card_id = action_idx + 1
        card_type = "Scoring Card" if card_id in SCORING_CARD_IDS else "Normal Card"
        return f"SELECT_CARD ({card_type})"
    elif action_idx < 114:
        modes = ["EVENT", "OPS", "SPACE", "PASS"]
        return f"SELECT_PLAY_MODE ({modes[action_idx - 110]})"
    elif action_idx < 116:
        timings = ["OPS_FIRST", "EVENT_FIRST"]
        return f"CHOOSE_TIMING_BRANCH ({timings[action_idx - 114]})"
    elif action_idx < ActionEncoder.NODE_OFFSET:
        op_modes = ["INFLUENCE", "COUP", "REALIGN"]
        return f"SELECT_OP_MODE ({op_modes[action_idx - 116]})"
    elif action_idx < ActionEncoder.BRANCH_OFFSET:
        return "POINT_NODE (Country)"
    elif action_idx < ActionEncoder.CONFIRM_DONE_INDEX:
        return f"CHOOSE_BRANCH ({action_idx - ActionEncoder.BRANCH_OFFSET})"
    else:
        return "UNKNOWN"



def _assert_width(agent: Union[NeuralAgent, BatchActor], obs: npt.NDArray[np.float32]) -> None:
    """A model silently misreads an observation of the wrong width; say so instead.

    With the view spec the runner's rows can be wider than an agent's view -- each is written in
    its decider's feature set and zero-padded to the widest set in the match -- so the agent reads
    the first `obs_size` floats of its own rows. Anything narrower than that is an error."""
    if obs.shape[1] < agent.obs_size:
        raise ValueError(
            f"{agent.name} expects an observation of width {agent.obs_size} but the runner produced "
            f"{obs.shape[1]}: its view was not set on the runner (set_obs_features).")


def _agent_features(agent: Any) -> int:
    """The observation feature set a seated agent reads (`bindings.ts_env.agent_view`): a
    NeuralAgent's or OnnxAgent's declared set, a searcher's network's, the base for a bot (which
    acts on the state and never reads the runner's rows)."""
    return agent_view(agent).features


def _opponent_cannot_pay_cmc(st: "ts.GameState") -> bool:
    """Cuban Missile Crisis headlined now could not be paid off by the decider's opponent as the
    board stands: the USSR needs 2 Influence in Cuba, the US 2 in West Germany or in Turkey."""
    if st.ctx().decision_player == ts.Player.US:
        return st.get_country(71).ussr_influence < 2
    return st.get_country(7).us_influence < 2 and st.get_country(33).us_influence < 2


#: Conditions a `headline:<card>+<name>:` rule can carry, by name.
HEADLINE_CONDITIONS: Dict[str, Callable[["ts.GameState"], bool]] = {
    "oppcantpay": _opponent_cannot_pay_cmc,
}


def apply_script(name: str, indices: npt.NDArray[np.int64], masks: npt.NDArray[Any],
                 runner: "ts.VectorizedBatchRunner", actions: npt.NDArray[np.int32]) -> None:
    fn = SCRIPTS[name]
    for idx in indices:
        i = int(idx)
        forced = fn(runner.get_state(i), masks[i])
        if forced is not None:
            actions[i] = forced


def apply_headline_rule(card: int, indices: npt.NDArray[np.int64], masks: npt.NDArray[Any],
                        runner: "ts.VectorizedBatchRunner", actions: npt.NDArray[np.int32],
                        condition: Optional[str] = None) -> None:
    """`headline:<card>:` agents: at a headline card choice where `card` is legal (so in hand),
    the choice is `card` -- when `condition` names one, only where it holds. Card c is flat slot
    c - 1."""
    when = HEADLINE_CONDITIONS[condition] if condition else None
    slot = card - 1
    for idx in indices:
        i = int(idx)
        if not masks[i][slot]:
            continue
        st = runner.get_state(i)
        if st.current_phase == ts.Phase.HEADLINE and st.ctx().decision_type == ts.DecisionType.SELECT_CARD \
                and (when is None or when(st)):
            actions[i] = slot


def apply_pricing(agent: PlayerAgent, indices: npt.NDArray[np.int64], runner: "ts.VectorizedBatchRunner",
                  actions: npt.NDArray[np.int32], dev: torch.device) -> None:
    """`priced:<worlds>:<z>:` agents: at every small decision (`ai.search.turn_pricing.priceable`),
    every option is priced to the end of the turn over <worlds> worlds by the agent's own network, and
    the agent's move is replaced by the best option when that beats it by >= <z> standard errors of
    the paired difference -- the B4' verdict played instead of trained on. One batch over all games
    waiting on the agent. Counts are kept on the agent (`pricing_stats`) and printed at exit."""
    worlds, z_min = getattr(agent, "pricing")
    model = getattr(agent, "model")
    stats: Dict[str, int] = getattr(agent, "pricing_stats")
    cands: List[Candidate] = []
    rows: List[int] = []
    for idx in indices:
        i = int(idx)
        st = runner.get_state(i)
        if not priceable(st, 12):
            continue
        legal = [int(a) for a in np.flatnonzero(ActionEncoder.get_legal_mask(st))]
        cands.append(Candidate(game=-1, choice=-1, state=st, options=legal, probs=[], taken=int(actions[i])))
        rows.append(i)
    stats["decisions"] += len(indices)
    if not cands:
        return
    price(model, dev, cands, int(worlds), int(np.random.randint(1, 1 << 30)))
    for i, c in zip(rows, cands):
        v = c.values
        if v is None or v.shape[1] < 4 or c.taken not in c.options:
            continue
        stats["priced"] += 1
        t = c.options.index(c.taken)
        b = int(np.argmax(v.mean(axis=1)))
        if b == t:
            continue
        d = v[b] - v[t]
        se = float(d.std(ddof=1) / np.sqrt(len(d)))
        if se > 0 and float(d.mean()) / se >= float(z_min):
            actions[i] = c.options[b]
            stats["switched"] += 1


def _observers(agents: Sequence[Any]) -> List[Any]:
    """The agents that follow every decision of their games (`observe`, `new_games`: a belief root,
    ai/search/belief.py), each once however many seats it holds."""
    seen: Dict[int, Any] = {}
    for ag in agents:
        if hasattr(ag, "observe") and id(ag) not in seen:
            seen[id(ag)] = ag
    return list(seen.values())


def _report_decisions(observer: Any, rows: npt.NDArray[np.int64], runner: "ts.VectorizedBatchRunner",
                      actions: npt.NDArray[np.int32], d_players: npt.NDArray[Any]) -> None:
    """Show `observer` the decisions about to be applied in its games `rows`: each position before the
    move, the move, and who made it."""
    rows = rows[d_players[rows] != 0]
    if len(rows):
        observer.observe([int(i) for i in rows], [runner.get_state(int(i)) for i in rows],
                         [int(a) for a in actions[rows]], [int(p) for p in d_players[rows]])


def _choose_actions(
    agent: PlayerAgent,
    indices: npt.NDArray[np.int64],
    obs: npt.NDArray[np.float32],
    masks: npt.NDArray[Any],
    d_players: npt.NDArray[Any],
    runner: "ts.VectorizedBatchRunner",
    temperature: float,
    greedy: bool,
    dev: torch.device,
    actions: npt.NDArray[np.int32],
) -> None:
    """Fill `actions[indices]` with `agent`'s choices, by the cheapest interface it offers."""
    if isinstance(agent, NeuralAgent):
        sel_obs = np.asarray(obs[indices])
        _assert_width(agent, sel_obs)
        sel_obs = sel_obs[:, :agent.obs_size]
        obs_t = torch.from_numpy(sel_obs).float().to(dev)
        mask_t = torch.from_numpy(masks[indices]).to(dev)
        with torch.no_grad():
            act_t, _, _, _, _ = agent.model.sample_action(obs_t, mask_t, temperature=temperature,
                                                          deterministic=greedy)
            if agent.obs_features & CANONICAL_LEARNED:
                # P43 V3: the runs opening here take their order from this network's own logits.
                logits_t, _, _ = agent.model(obs_t, mask_t)
                full = np.zeros((len(actions), logits_t.shape[1]), dtype=np.float32)
                full[indices] = logits_t.float().cpu().numpy()
                give_run_priorities(runner, full, rows=indices)
        actions[indices] = act_t.cpu().numpy()
    elif isinstance(agent, BatchActor):
        refuse_learned_order(getattr(agent, "obs_features", 0), "an ONNX agent")
        # A network outside torch (OnnxAgent): the same observations and masks, one call.
        sel_obs = np.asarray(obs[indices])
        _assert_width(agent, sel_obs)
        actions[indices] = agent.act_batch(sel_obs[:, :agent.obs_size], masks[indices],
                                           temperature, greedy)
    elif isinstance(agent, BatchSelector):
        refuse_learned_order(getattr(agent, "obs_features", 0), "a searcher")
        # A searcher pays for batching: one call over every game waiting on it,
        # rather than one call per game. Measured at 64 simulations, a batch of
        # 256 roots runs at 103 decisions/s against roughly 0.4/s one at a time.
        sel_states = [runner.get_state(int(idx)) for idx in indices]
        games_hook = getattr(agent, "select_actions_games", None)
        picks = (games_hook([int(i) for i in indices], sel_states) if games_hook is not None
                 else agent.select_actions_batch(sel_states))
        for idx, a in zip(indices, picks):
            actions[idx] = a
    else:
        for idx in indices:
            st = runner.get_state(int(idx))
            actions[idx] = agent.select_action(st, ts.Player(int(d_players[idx])),
                                               temperature=temperature)
    headline_card = getattr(agent, "headline_card", None)
    if headline_card is not None:
        apply_headline_rule(int(headline_card), indices, masks, runner, actions,
                            getattr(agent, "headline_condition", None))
    script = getattr(agent, "script", None)
    if script is not None:
        apply_script(str(script), indices, masks, runner, actions)
    if getattr(agent, "pricing", None) is not None:
        apply_pricing(agent, indices, runner, actions, dev)


class ChoiceStats(TypedDict):
    """The `--track-choices` block of a matchup result: how many micro-actions had one choice."""

    us_total_micro_actions: int
    us_single_choice_micro_actions: int
    us_single_choice_pct: float
    ussr_total_micro_actions: int
    ussr_single_choice_micro_actions: int
    ussr_single_choice_pct: float
    overall_total_micro_actions: int
    overall_single_choice_micro_actions: int
    overall_single_choice_pct: float
    avg_per_game: Dict[str, float]
    category_counts_us: Dict[str, int]
    category_counts_ussr: Dict[str, int]


class MatchupResult(TypedDict):
    """What one matchup reports: agent A's record from both seats, with averages and causes.
    `play_parallel_matchup`, `play_packed_matchups` and `merge_matchup_results` all build it
    through `_matchup_result`, so the three cannot disagree on a field."""

    agent_a: str
    agent_b: str
    total_games: int
    games_per_side: int
    a_wins: int
    b_wins: int
    draws: int
    win_rate_a: float
    win_rate_b: float
    a_wins_as_us: int
    a_losses_as_us: int
    a_draws_as_us: int
    win_rate_a_as_us: float
    a_wins_as_ussr: int
    a_losses_as_ussr: int
    a_draws_as_ussr: int
    win_rate_a_as_ussr: float
    avg_steps: float
    avg_turn: float
    avg_ply: float
    avg_vp_margin_a: float
    causes_loss_us: Dict[str, int]
    causes_loss_ussr: Dict[str, int]
    causes_all: Dict[str, int]
    elapsed_seconds: float
    choice_stats: NotRequired[ChoiceStats]


def _add_counts(into: Dict[str, int], more: Dict[str, int]) -> None:
    for k, v in more.items():
        into[k] = into.get(k, 0) + v


@dataclass
class MatchupTally:
    """Agent A's wins, losses and draws in each seat, with the causes of the losses."""

    a_wins: int = 0
    b_wins: int = 0
    draws: int = 0
    a_wins_as_us: int = 0
    a_losses_as_us: int = 0
    a_draws_as_us: int = 0
    a_wins_as_ussr: int = 0
    a_losses_as_ussr: int = 0
    a_draws_as_ussr: int = 0
    causes_loss_us: Dict[str, int] = field(default_factory=dict)
    causes_loss_ussr: Dict[str, int] = field(default_factory=dict)
    causes_all: Dict[str, int] = field(default_factory=dict)

    def record(self, utility: float, a_is_ussr: bool, reason: str) -> None:
        """One finished game; `utility` is the terminal utility from the US side."""
        self.causes_all[reason] = self.causes_all.get(reason, 0) + 1
        a_won = (utility > 0 and not a_is_ussr) or (utility < 0 and a_is_ussr)
        b_won = (utility < 0 and not a_is_ussr) or (utility > 0 and a_is_ussr)
        if a_won:
            self.a_wins += 1
            if a_is_ussr:
                self.a_wins_as_ussr += 1
            else:
                self.a_wins_as_us += 1
        elif b_won:
            self.b_wins += 1
            if a_is_ussr:
                self.a_losses_as_ussr += 1
                self.causes_loss_ussr[reason] = self.causes_loss_ussr.get(reason, 0) + 1
            else:
                self.a_losses_as_us += 1
                self.causes_loss_us[reason] = self.causes_loss_us.get(reason, 0) + 1
        else:
            self.draws += 1
            if a_is_ussr:
                self.a_draws_as_ussr += 1
            else:
                self.a_draws_as_us += 1

    def add(self, other: "MatchupTally") -> None:
        self.a_wins += other.a_wins
        self.b_wins += other.b_wins
        self.draws += other.draws
        self.a_wins_as_us += other.a_wins_as_us
        self.a_losses_as_us += other.a_losses_as_us
        self.a_draws_as_us += other.a_draws_as_us
        self.a_wins_as_ussr += other.a_wins_as_ussr
        self.a_losses_as_ussr += other.a_losses_as_ussr
        self.a_draws_as_ussr += other.a_draws_as_ussr
        _add_counts(self.causes_loss_us, other.causes_loss_us)
        _add_counts(self.causes_loss_ussr, other.causes_loss_ussr)
        _add_counts(self.causes_all, other.causes_all)

    @classmethod
    def from_result(cls, res: MatchupResult) -> "MatchupTally":
        return cls(a_wins=res["a_wins"], b_wins=res["b_wins"], draws=res["draws"],
                   a_wins_as_us=res["a_wins_as_us"], a_losses_as_us=res["a_losses_as_us"],
                   a_draws_as_us=res["a_draws_as_us"], a_wins_as_ussr=res["a_wins_as_ussr"],
                   a_losses_as_ussr=res["a_losses_as_ussr"],
                   a_draws_as_ussr=res["a_draws_as_ussr"],
                   causes_loss_us=dict(res["causes_loss_us"]),
                   causes_loss_ussr=dict(res["causes_loss_ussr"]),
                   causes_all=dict(res["causes_all"]))


def _matchup_result(
    agent_a: str, agent_b: str, n_pairs: int, tally: MatchupTally, *,
    avg_steps: float, avg_turn: float, avg_ply: float, avg_vp_margin_a: float,
    elapsed_seconds: float, choice_stats: Optional[ChoiceStats] = None,
) -> MatchupResult:
    """A matchup's result from its counts. `n_pairs` is the number of deals played, each from both
    seats, so the record holds `2 * n_pairs` games."""
    total = n_pairs * 2
    res: MatchupResult = {
        "agent_a": agent_a,
        "agent_b": agent_b,
        "total_games": total,
        "games_per_side": n_pairs,
        "a_wins": tally.a_wins,
        "b_wins": tally.b_wins,
        "draws": tally.draws,
        "win_rate_a": float(tally.a_wins / max(1, total)),
        "win_rate_b": float(tally.b_wins / max(1, total)),
        "a_wins_as_us": tally.a_wins_as_us,
        "a_losses_as_us": tally.a_losses_as_us,
        "a_draws_as_us": tally.a_draws_as_us,
        "win_rate_a_as_us": float(tally.a_wins_as_us / max(1, n_pairs)),
        "a_wins_as_ussr": tally.a_wins_as_ussr,
        "a_losses_as_ussr": tally.a_losses_as_ussr,
        "a_draws_as_ussr": tally.a_draws_as_ussr,
        "win_rate_a_as_ussr": float(tally.a_wins_as_ussr / max(1, n_pairs)),
        "avg_steps": avg_steps,
        "avg_turn": avg_turn,
        "avg_ply": avg_ply,
        "avg_vp_margin_a": avg_vp_margin_a,
        "causes_loss_us": tally.causes_loss_us,
        "causes_loss_ussr": tally.causes_loss_ussr,
        "causes_all": tally.causes_all,
        "elapsed_seconds": elapsed_seconds,
    }
    if choice_stats is not None:
        res["choice_stats"] = choice_stats
    return res


def _choice_stats(
    tot_us: int, single_us: int, tot_ussr: int, single_ussr: int, total_games: int,
    category_counts_us: Dict[str, int], category_counts_ussr: Dict[str, int],
) -> ChoiceStats:
    """The `choice_stats` block of a matchup result, from its raw counts."""
    tot_combined = tot_us + tot_ussr
    single_comb = single_us + single_ussr
    return {
        "us_total_micro_actions": tot_us,
        "us_single_choice_micro_actions": single_us,
        "us_single_choice_pct": float((single_us / max(1, tot_us)) * 100.0),
        "ussr_total_micro_actions": tot_ussr,
        "ussr_single_choice_micro_actions": single_ussr,
        "ussr_single_choice_pct": float((single_ussr / max(1, tot_ussr)) * 100.0),
        "overall_total_micro_actions": tot_combined,
        "overall_single_choice_micro_actions": single_comb,
        "overall_single_choice_pct": float((single_comb / max(1, tot_combined)) * 100.0),
        "avg_per_game": {
            "us_total": float(tot_us / max(1, total_games)),
            "us_single": float(single_us / max(1, total_games)),
            "ussr_total": float(tot_ussr / max(1, total_games)),
            "ussr_single": float(single_ussr / max(1, total_games)),
            "combined_total": float(tot_combined / max(1, total_games)),
            "combined_single": float(single_comb / max(1, total_games)),
        },
        "category_counts_us": category_counts_us,
        "category_counts_ussr": category_counts_ussr,
    }


def merge_matchup_results(parts: Sequence[MatchupResult]) -> MatchupResult:
    """One matchup result from the results of disjoint sets of its game pairs.

    Counts add; the per-game averages are weighted by each part's game count, so the merge equals
    what one call over every pair would have reported. `elapsed_seconds` is the sum over parts --
    compute time, not wall clock, which the caller measures itself.
    """
    if not parts:
        raise ValueError("merge_matchup_results needs at least one part")
    names = {(p["agent_a"], p["agent_b"]) for p in parts}
    if len(names) != 1:
        raise ValueError(f"parts come from different matchups: {sorted(names)}")
    tally = MatchupTally()
    for p in parts:
        tally.add(MatchupTally.from_result(p))
    total = sum(p["total_games"] for p in parts)

    def weighted_mean(value: Callable[[MatchupResult], float]) -> float:
        return float(sum(value(p) * p["total_games"] for p in parts) / max(1, total))

    stats = [p["choice_stats"] for p in parts if "choice_stats" in p]
    choice: Optional[ChoiceStats] = None
    if stats:
        cat_us: Dict[str, int] = {}
        cat_ussr: Dict[str, int] = {}
        for st in stats:
            _add_counts(cat_us, st["category_counts_us"])
            _add_counts(cat_ussr, st["category_counts_ussr"])
        choice = _choice_stats(
            sum(st["us_total_micro_actions"] for st in stats),
            sum(st["us_single_choice_micro_actions"] for st in stats),
            sum(st["ussr_total_micro_actions"] for st in stats),
            sum(st["ussr_single_choice_micro_actions"] for st in stats),
            total, cat_us, cat_ussr)
    return _matchup_result(
        parts[0]["agent_a"], parts[0]["agent_b"], sum(p["games_per_side"] for p in parts), tally,
        avg_steps=weighted_mean(lambda p: p["avg_steps"]),
        avg_turn=weighted_mean(lambda p: p["avg_turn"]),
        avg_ply=weighted_mean(lambda p: p["avg_ply"]),
        avg_vp_margin_a=weighted_mean(lambda p: p["avg_vp_margin_a"]),
        elapsed_seconds=float(sum(p["elapsed_seconds"] for p in parts)),
        choice_stats=choice)


@dataclass(frozen=True)
class _Chunking:
    """How a matchup's game pairs are cut into engine batches. A pair's seed and game number are
    fixed by its index in the FULL matchup, whatever subset of pairs a call plays, so a split
    matchup plays the deals the unsplit one always has."""

    games_per_side: int
    half_per_chunk: int     # pairs per batch; the batch holds each pair twice, seats swapped

    @classmethod
    def of(cls, games_per_side: int, batch_chunk_size: int) -> "_Chunking":
        return cls(games_per_side, min(games_per_side, batch_chunk_size // 2))

    @property
    def chunk_size(self) -> int:
        return self.half_per_chunk * 2

    def pair_seed(self, base_seed: int, k: int) -> int:
        """The deal of pair `k`, played from both seats."""
        return base_seed + (k // self.half_per_chunk) * self.chunk_size + (k % self.half_per_chunk)

    def game_index(self, k: int, second_half: bool) -> int:
        """1-based, as the unsplit loop numbers games: chunk by chunk, first half then second."""
        c = k // self.half_per_chunk
        full_half = min(self.games_per_side - c * self.half_per_chunk, self.half_per_chunk)
        return (c * self.chunk_size + (k % self.half_per_chunk)
                + (full_half if second_half else 0) + 1)


def _raise_unfinished(active: npt.NDArray[np.bool_], steps: int, max_steps: int, what: str) -> None:
    """Called before each step's decisions, once the step's finished games are counted: a game still
    running when `max_steps` steps have been played is a stuck agent or engine, not a draw. Counting
    it as one (the old "Early Termination", utility 0) quietly moved win rates. Raise instead."""
    if steps >= max_steps and np.any(active):
        raise RuntimeError(
            f"{int(np.count_nonzero(active))} game(s) of {what} still running after {steps} steps "
            f"(max_steps={max_steps}): an agent or the engine is stuck, or max_steps is too low for "
            f"this setting (auto_advance off needs more). Not counted as draws.")


def _agent_temperature(agent: PlayerAgent, temperature: float,
                       deterministic: Optional[bool]) -> Tuple[float, bool]:
    """(temperature, greedy) for one agent. An agent may pin its own temperature
    (`temp:<T>:` in load_agent), which overrides the matchup-wide one: that is what makes a
    same-weights temperature comparison expressible at all. Greedy is an explicit `deterministic`,
    else a temperature too low to differ from an argmax."""
    pinned = getattr(agent, "temperature", None)
    t = temperature if pinned is None else float(pinned)
    return t, (t <= 0.05) if deterministic is None else deterministic


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
        auto_advance: bool = True,
        pairs: Optional[Sequence[int]] = None,
    ) -> MatchupResult:
        """Play a matchup batched. Action selection matches NeuralAgent.select_action.

        temperature/deterministic are the same contract as NeuralAgent.select_action: sampling
        at the given temperature unless it is low enough to be indistinguishable from an argmax.
        This used to be hard-coded to deterministic=True here while a sequential path sampled at
        0.1, so the two disagreed on win rate (0.450 vs 0.610 on one 100-game matchup).

        `pairs` plays only those game pairs (indices into range(games_per_side)), each with the
        deal it has in the full matchup. That is what lets a matchup be split across processes
        (tools/lib/parallel_tournament.py) and merged with `merge_matchup_results` without
        changing which games are played. The result then counts only those pairs.
        """
        dev = resolve_device(device)
        temp_a, greedy_a = _agent_temperature(agent_a, temperature, deterministic)
        temp_b, greedy_b = _agent_temperature(agent_b, temperature, deterministic)

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
        chunking = _Chunking.of(games_per_side, batch_chunk_size)
        half_per_chunk = chunking.half_per_chunk

        ks = list(range(games_per_side)) if pairs is None else [int(k) for k in pairs]
        bad = [k for k in ks if not 0 <= k < games_per_side]
        if bad or len(set(ks)) != len(ks) or not ks:
            raise ValueError(f"pairs must be distinct indices in range({games_per_side}), got {ks}")
        n_pairs = len(ks)
        tally = MatchupTally()
        all_steps = []
        all_turns = []
        # Length in plies alongside turns. A turn number cannot separate a game abandoned
        # at turn 7 AR1 from one that ran to turn 7 AR7, and it reads 11 for a game that
        # went the distance because finish_end_turn increments before testing its bound.
        all_plies = []
        all_vps = []

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

        for lo in range(0, n_pairs, half_per_chunk):
            chunk_ks = ks[lo:lo + half_per_chunk]
            cur_half = len(chunk_ks)
            cur_games = cur_half * 2

            runner = ts.VectorizedBatchRunner(cur_games, chunking.pair_seed(base_seed, chunk_ks[0]))
            # Paired deals: env i and env i + cur_half are the same matchup with the sides
            # swapped, so give them the same seed and therefore the same shuffle. Deal luck
            # then cancels between the halves rather than adding variance to the result.
            if start_states is not None:
                for i, k in enumerate(chunk_ks):
                    pos = start_states[k]
                    runner.set_state(i, pos)
                    runner.set_state(i + cur_half, pos)
            else:
                for i, k in enumerate(chunk_ks):
                    paired_seed = chunking.pair_seed(base_seed, k)
                    runner.reset_game(i, paired_seed)
                    runner.reset_game(i + cur_half, paired_seed)
            runner.refresh_all()
            # P23 / E4.1: each side decides in its own agent's action view. Envs < cur_half seat
            # A as USSR and B as US; the rest the reverse. Only neural agents trained in the
            # merged view use it -- bots and searchers act in E4 terms.
            mv_a = _action_view_code(agent_a)
            mv_b = _action_view_code(agent_b)
            if mv_a or mv_b:
                first = np.arange(cur_games) < cur_half
                runner.set_action_views([int(x) for x in np.where(first, mv_b, mv_a)],
                                        [int(x) for x in np.where(first, mv_a, mv_b)])
            # The view spec's observation half, set the same way: each seat in its agent's view.
            fa, fb = _agent_features(agent_a), _agent_features(agent_b)
            if fa or fb:
                first = np.arange(cur_games) < cur_half
                runner.set_obs_features([int(x) for x in np.where(first, fb, fa)],
                                        [int(x) for x in np.where(first, fa, fb)])
            active = np.ones(cur_games, dtype=bool)
            steps = 0
            observers = _observers([agent_a, agent_b])
            for ob in observers:
                ob.new_games()
            open_a = getattr(agent_a, "forced_opening", None)
            open_b = getattr(agent_b, "forced_opening", None)
            setup_override = ScriptedSetupOverride(cur_games) if (open_a or open_b) else None

            # Temporary arrays for current chunk
            chunk_utils = np.zeros(cur_games, dtype=np.float32)
            chunk_vps = np.zeros(cur_games, dtype=np.int32)
            chunk_turns = np.zeros(cur_games, dtype=np.int32)
            chunk_plies = np.zeros(cur_games, dtype=np.int32)
            chunk_steps = np.zeros(cur_games, dtype=np.int32)
            chunk_causes = [""] * cur_games

            chunk_ussr_total = np.zeros(cur_games, dtype=np.int32)
            chunk_ussr_single = np.zeros(cur_games, dtype=np.int32)
            chunk_us_total = np.zeros(cur_games, dtype=np.int32)
            chunk_us_single = np.zeros(cur_games, dtype=np.int32)

            while np.any(active):
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
                        chunk_plies[idx] = game_ply(int(st.turn), int(st.action_round),
                                                    st.phasing_player == ts.Player.US,
                                                    headline_stage=int(st.headline_stage))
                        chunk_steps[idx] = steps
                        chunk_causes[idx] = classify_game_ending_reason(st)
                    active = active & (~terms)

                if not np.any(active):
                    break
                _raise_unfinished(active, steps, max_steps, f"{agent_a.name} vs {agent_b.name}")

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

                if np.any(is_a_turn):
                    _choose_actions(agent_a, np.where(is_a_turn)[0], obs, masks, d_players,
                                    runner, temp_a, greedy_a, dev, actions)
                if np.any(is_b_turn):
                    _choose_actions(agent_b, np.where(is_b_turn)[0], obs, masks, d_players,
                                    runner, temp_b, greedy_b, dev, actions)

                if setup_override is not None:
                    if open_a:
                        setup_override.apply(actions, obs, masks, d_players, np.where(is_a_turn)[0], open_a)
                    if open_b:
                        setup_override.apply(actions, obs, masks, d_players, np.where(is_b_turn)[0], open_b)
                for ob in observers:
                    _report_decisions(ob, np.where(active)[0], runner, actions, d_players)
                step_results = runner.step_flat_all(actions.tolist(),
                                                    auto_advance=auto_advance)
                # 0 = refused. A refused action here means a game silently did not advance and the
                # matchup's win rate is being computed over a position nobody played.
                if 0 in step_results:
                    bad = step_results.index(0)
                    raise IllegalActionError(
                        f"engine refused flat action {int(actions[bad])} in game {bad} of "
                        f"{len(actions)}; {step_results.count(0)} of the batch refused")
                steps += 1

            # Accumulate Chunk Results
            for idx in range(cur_games):
                a_is_ussr = (idx < cur_half)
                term_util = chunk_utils[idx]
                vp = chunk_vps[idx]
                turn = chunk_turns[idx]
                reason = chunk_causes[idx]

                tally.record(float(term_util), a_is_ussr, reason)
                all_steps.append(chunk_steps[idx])
                all_turns.append(turn)
                all_plies.append(chunk_plies[idx])

                all_vps.append(-vp if a_is_ussr else vp)

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

                        k = chunk_ks[idx % cur_half]
                        g_idx = chunking.game_index(k, not a_is_ussr)
                        m_ussr_tot = int(chunk_ussr_total[idx])
                        m_ussr_sgl = int(chunk_ussr_single[idx])
                        m_us_tot = int(chunk_us_total[idx])
                        m_us_sgl = int(chunk_us_single[idx])
                        m_all_tot = m_ussr_tot + m_us_tot
                        m_all_sgl = m_ussr_sgl + m_us_sgl

                        entry = {
                            "game_index": g_idx,
                            "seed": chunking.pair_seed(base_seed, k),
                            "ussr_agent": ussr_agent,
                            "us_agent": us_agent,
                            "winner": winner,
                            "victory_points": int(chunk_vps[idx]),
                            "turn": int(chunk_turns[idx]),
                            "ply": int(chunk_plies[idx]),
                            "steps": int(chunk_steps[idx]),
                            "cause": chunk_causes[idx],
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

        choice: Optional[ChoiceStats] = None
        if track_choices:
            choice = _choice_stats(
                total_micro_us, single_choice_us, total_micro_ussr, single_choice_ussr,
                n_pairs * 2, category_counts_us, category_counts_ussr)
        return _matchup_result(
            agent_a.name, agent_b.name, n_pairs, tally,
            avg_steps=float(np.mean(all_steps)) if all_steps else 0.0,
            avg_turn=float(np.mean(all_turns)) if all_turns else 0.0,
            avg_ply=float(np.mean(all_plies)) if all_plies else 0.0,
            avg_vp_margin_a=float(np.mean(all_vps)) if all_vps else 0.0,
            elapsed_seconds=time.time() - t0, choice_stats=choice)

    @staticmethod
    def play_packed_matchups(
        pairs: Sequence[Tuple[PlayerAgent, PlayerAgent]],
        games_per_side: int = 1000,
        batch_chunk_size: int = 1000,
        base_seed: int = 10000,
        device: Optional[Union[torch.device, str]] = None,
        max_steps: int = 2500,
        temperature: float = 0.1,
        deterministic: Optional[bool] = None,
        auto_advance: bool = True,
    ) -> List[MatchupResult]:
        """Several matchups in ONE engine batch, one forward per agent per step over all its rows.

        `play_parallel_matchup` plays one pairing at a time with ~2 x games_per_side games, so a
        field spends its time in a Python loop around forwards of ~100 positions -- the GPU sat at
        ~35%. Here K pairings share a VectorizedBatchRunner and every agent's positions across
        all of them go through one forward per step.

        Each game gets exactly the seed `play_parallel_matchup` would give it (the same chunked
        seed formula, the same seat-swapped copy), and the result dicts have the same fields. So
        bots and greedy agents reproduce `play_parallel_matchup` game for game, up to float
        rounding that a different batch composition can introduce; sampled agents reproduce it in
        distribution only, since the random stream is consumed in a different order. Choice
        tracking and game logs are not supported here: callers use the one-pair path for those.
        """
        dev = resolve_device(device)
        if games_per_side <= 0:
            raise ValueError(f"games_per_side must be positive, got {games_per_side}")
        gps = int(games_per_side)
        # The seed play_parallel_matchup gives game k of a pairing (k < games_per_side), both seats.
        chunking = _Chunking.of(gps, batch_chunk_size)
        seeds = [chunking.pair_seed(base_seed, k) for k in range(gps)]

        P = len(pairs)
        n = P * 2 * gps
        runner = ts.VectorizedBatchRunner(n, base_seed)
        a_is_ussr = np.zeros(n, dtype=bool)          # env seats pair p's agent A as USSR
        for p in range(P):
            off = p * 2 * gps
            for k in range(gps):
                runner.reset_game(off + k, seeds[k])
                runner.reset_game(off + gps + k, seeds[k])
            a_is_ussr[off:off + gps] = True
        runner.refresh_all()

        agents: List[PlayerAgent] = []
        index: Dict[int, int] = {}
        for a, b in pairs:
            for ag in (a, b):
                if id(ag) not in index:
                    index[id(ag)] = len(agents)
                    agents.append(ag)
        a_idx = np.repeat(np.array([index[id(a)] for a, _ in pairs]), 2 * gps)
        b_idx = np.repeat(np.array([index[id(b)] for _, b in pairs]), 2 * gps)
        ussr_agent = np.where(a_is_ussr, a_idx, b_idx)
        us_agent = np.where(a_is_ussr, b_idx, a_idx)

        mv = np.array([_action_view_code(ag) for ag in agents], dtype=np.int64)
        if mv.any():
            runner.set_action_views([int(x) for x in mv[us_agent]], [int(x) for x in mv[ussr_agent]])
        fv = np.array([_agent_features(ag) for ag in agents], dtype=np.int64)
        if fv.any():
            runner.set_obs_features([int(x) for x in fv[us_agent]], [int(x) for x in fv[ussr_agent]])

        temps = [_agent_temperature(ag, temperature, deterministic) for ag in agents]
        observers = _observers(agents)
        for ob in observers:
            ob.new_games()
        openings = [getattr(ag, "forced_opening", None) for ag in agents]
        setup_override = ScriptedSetupOverride(n) if any(openings) else None
        active = np.ones(n, dtype=bool)
        utils = np.zeros(n, dtype=np.float32)
        vps = np.zeros(n, dtype=np.int32)
        turns = np.zeros(n, dtype=np.int32)
        plies = np.zeros(n, dtype=np.int32)
        fin_steps = np.zeros(n, dtype=np.int32)
        causes = [""] * n
        t0 = time.time()
        steps = 0
        while np.any(active):
            obs = runner.get_observations()
            masks = runner.get_action_masks()
            d_players = np.array(runner.get_decision_players())
            terms = np.array(runner.get_terminals())
            newly = active & terms
            for idx in np.where(newly)[0]:
                st = runner.get_state(int(idx))
                utils[idx] = float(ts.Engine.get_terminal_utility(st))
                vps[idx] = int(st.victory_points)
                turns[idx] = int(st.turn)
                plies[idx] = game_ply(int(st.turn), int(st.action_round),
                                      st.phasing_player == ts.Player.US,
                                      headline_stage=int(st.headline_stage))
                fin_steps[idx] = steps
                causes[idx] = classify_game_ending_reason(st)
            active = active & (~terms)
            if not np.any(active):
                break
            _raise_unfinished(active, steps, max_steps, ", ".join(f"{a.name} vs {b.name}" for a, b in pairs))

            actor = np.where(d_players == -1, ussr_agent, us_agent)
            actions = np.zeros(n, dtype=np.int32)
            # Neural agents: the step's observations go to the device ONCE and are split there;
            # every agent's actions land in one device tensor, read back with one sync per step.
            # (Row-by-row CPU gathers and a copy + sync per agent were most of the loop's time.)
            nn_rows = active & np.array([isinstance(agents[x], NeuralAgent) for x in range(len(agents))])[actor]
            obs_dev = mask_dev = act_dev = None
            if nn_rows.any():
                obs_dev = torch.from_numpy(np.asarray(obs)).to(dev).float()
                mask_dev = torch.from_numpy(np.asarray(masks)).to(dev)
                act_dev = torch.zeros(n, dtype=torch.long, device=dev)
            for ai, agent in enumerate(agents):
                rows = np.where(active & (actor == ai))[0]
                if len(rows) == 0:
                    continue
                t_ag, g_ag = temps[ai]
                if isinstance(agent, NeuralAgent):
                    assert obs_dev is not None and mask_dev is not None and act_dev is not None
                    want = agent.obs_size
                    if obs_dev.shape[1] < want:
                        _assert_width(agent, np.asarray(obs[rows[:1]]))
                    rows_t = torch.from_numpy(rows).to(dev)
                    with torch.no_grad():
                        sel_obs = obs_dev.index_select(0, rows_t)[:, :want]
                        sel_mask = mask_dev.index_select(0, rows_t)
                        act_t, _, _, _, _ = agent.model.sample_action(
                            sel_obs, sel_mask, temperature=t_ag, deterministic=g_ag)
                        if agent.obs_features & CANONICAL_LEARNED:
                            # P43 V3: the runs opening here take their order from this network's own logits.
                            logits_t, _, _ = agent.model(sel_obs, sel_mask)
                            full = np.zeros((n, logits_t.shape[1]), dtype=np.float32)
                            full[rows] = logits_t.float().cpu().numpy()
                            give_run_priorities(runner, full, rows=rows)
                    act_dev.index_copy_(0, rows_t, act_t.long())
                else:
                    _choose_actions(agent, rows, obs, masks, d_players, runner, t_ag, g_ag, dev,
                                    actions)
            if act_dev is not None:
                nn_act = act_dev.cpu().numpy().astype(np.int32)
                actions = np.where(nn_rows, nn_act, actions)
            if setup_override is not None:
                for ai, op in enumerate(openings):
                    if op:
                        setup_override.apply(actions, obs, masks, d_players,
                                             np.where(active & (actor == ai))[0], op)
            for ob in observers:
                ai = index[id(ob)]
                _report_decisions(ob, np.where(active & ((a_idx == ai) | (b_idx == ai)))[0], runner, actions, d_players)
            step_results = runner.step_flat_all(actions.tolist(), auto_advance=auto_advance)
            if 0 in step_results:
                bad = step_results.index(0)
                raise IllegalActionError(
                    f"engine refused flat action {int(actions[bad])} in game {bad} of "
                    f"{len(actions)}; {step_results.count(0)} of the batch refused")
            steps += 1
        elapsed = time.time() - t0

        out: List[MatchupResult] = []
        for p, (agent_a, agent_b) in enumerate(pairs):
            off = p * 2 * gps
            tally = MatchupTally()
            v_margin = []
            for idx in range(off, off + 2 * gps):
                ussr_is_a = bool(a_is_ussr[idx])
                tally.record(float(utils[idx]), ussr_is_a, causes[idx])
                v_margin.append(-vps[idx] if ussr_is_a else vps[idx])
            sl = slice(off, off + 2 * gps)
            out.append(_matchup_result(
                agent_a.name, agent_b.name, gps, tally,
                avg_steps=float(np.mean(fin_steps[sl])), avg_turn=float(np.mean(turns[sl])),
                avg_ply=float(np.mean(plies[sl])), avg_vp_margin_a=float(np.mean(v_margin)),
                # The pack's wall time is shared; each pairing reports its share.
                elapsed_seconds=elapsed / max(1, P)))
        return out


def compute_mle_elo(
    model_names: List[str],
    win_matrix: np.ndarray,
    total_matrix: np.ndarray,
    anchor_model: str = "HeuristicBot",
    anchor_elo: float = 1500.0,
) -> Dict[str, float]:
    """Bradley-Terry maximum-likelihood Elo ratings, `anchor_model` held at `anchor_elo`.

    `win_matrix[i, j]` is i's wins over j and `total_matrix[i, j]` (symmetric) their games; a game
    neither won is a draw, half a win. The fit is the leaderboard's (`tools.lib.leaderboard._newton`:
    Newton steps on the binomial log-likelihood to convergence, with the leaderboard's weak prior of
    sd PRIOR_SD centred on the anchor's rating), seats pooled. It replaced 1,000 fixed gradient steps
    of size 0.01, which had not converged on a small tournament -- a few hundred games move the
    ratings by only ~0.01 x games per step -- so the reported gaps depended on the step budget. The
    prior keeps an unbeaten or winless entrant finite instead of drifting with the iteration count.
    Without the anchor among the entrants every rating is free and the prior alone sets the level.
    """
    names = [str(n) for n in model_names]
    if len(set(names)) != len(names):
        raise ValueError(f"duplicate entrant names: {names}")
    obs: List[_Binomial] = []
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            n_ij = int(total_matrix[i, j])
            if n_ij == 0:
                continue
            w_ij, w_ji = float(win_matrix[i, j]), float(win_matrix[j, i])
            if w_ij + w_ji > n_ij:
                raise ValueError(f"{names[i]} vs {names[j]}: {w_ij + w_ji:g} wins in {n_ij} games")
            obs.append(_Binomial(names[i], names[j], 1, w_ij + 0.5 * (n_ij - w_ij - w_ji), n_ij))
    fixed = {anchor_model: float(anchor_elo)} if anchor_model in names else {}
    free = [n for n in names if n not in fixed]
    fit = _newton(free, fixed, obs, fit_seat=False, seat=0.0, prior_mean=float(anchor_elo))
    return {n: float(fixed[n]) if n in fixed else float(fit.ratings[n]) for n in names}
