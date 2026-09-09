"""Measures the human ts-replayer corpus on this engine: game arc, forced wins, critic calibration.

Every other arc measurement in `research/experiments.md` (4.5-4.7) was taken on games our own
agents generated, which leaves one question unanswerable: the missing US late-war recovery could
be a property of the engine or a property of the agents. Human games are the control that
separates them -- the same engine, driven by decisions no policy of ours produced.

Three measurements, each mirroring an existing agent-side one so the rows can be laid side by
side:

* **Arc** (against 4.6) -- mean `victory_points` at each turn boundary, US win rate, and win
  rate split by how long the game ran.
* **Forced wins** (against 4.2/4.4) -- at each human decision, whether the mask held an action
  that immediately ends the game in the mover's favour, whether the human took it, and, for
  declines, whether the decliner won anyway.
* **Critic calibration** (against 4.2) -- the value head's `v_win` on human positions bucketed
  against the mover's actual result, which tests the "systematically optimistic" finding against
  ground truth on positions the agent never generates.

**How the live state is observed.** `convert_game` returns observations, masks and actions but
not the `GameState` they came from, and the forced-win check needs the state itself (it clones
and steps it). The converter calls `ts.extract_observation` exactly once per emitted sample, at
the one site that appends to `Conversion.samples`, so wrapping that function -- see
`_observing_conversion` -- yields the live state at each human decision in lockstep with the
samples list, without editing the converter. The wrapper also keeps a reference to the single
`GameState` the converter builds, which is never rebound, so after conversion it holds the final
position and therefore the game's result.

**Nothing is inferred where the log is silent.** A game that fails to convert, a fragment with no
outcome, a decision at a chance node: each is excluded and counted, never filled in. Turn
populations shift as short games drop out (the 4.6 caveat), so N is reported at every turn rather
than conditioned away.
"""

from __future__ import annotations

import gzip
import json
import os
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterator, List, Optional, Sequence, Set, Tuple

import numpy as np
import ts_engine as ts

from ai.eval.safety import classify_legal_actions
from tools.lib.ts_replayer_convert import Conversion, convert_game

# Predicted win probability for a batch of (observation, mask) pairs, in the mover's frame.
ValueFn = Callable[[np.ndarray, np.ndarray], np.ndarray]

US = 1
USSR = -1

# What a decision node turned out to be, recorded per emitted sample.
NODE_DECISION = "decision"
NODE_CHANCE = "chance"
NODE_NO_LEGAL = "no_legal"


@dataclass
class DecisionRecord:
    """One human decision, with what the position offered and what the human did."""

    turn: int
    victory_points: int          # US-positive, the engine's convention
    mover: int                   # +1 US, -1 USSR
    action: int
    legal_actions: int
    win_available: bool
    win_taken: bool
    v_win: Optional[float] = None   # critic's prediction for the mover, if a model was given


@dataclass
class GameMeasurement:
    """Everything measurable about one corpus game."""

    replay_id: int
    skipped: Optional[str] = None
    failed: Optional[str] = None
    # The cached file holds no turns, so there is no game here to measure.
    empty_log: bool = False
    truncated: bool = False
    game_ended: bool = False
    # +1.0 if the US won, -1.0 if the USSR won, 0.0 a draw; None where the log stops first.
    us_utility: Optional[float] = None
    final_turn: int = 0
    samples: int = 0
    # Turn T -> victory_points at the end of turn T. See `_turn_boundaries`.
    vp_at_turn_end: Dict[int, int] = field(default_factory=dict)
    decisions: List[DecisionRecord] = field(default_factory=list)
    # Emitted samples this measurement could not classify, and why.
    excluded_chance_nodes: int = 0
    excluded_no_legal: int = 0

    @property
    def usable(self) -> bool:
        return self.skipped is None and self.failed is None and not self.empty_log


@dataclass
class CorpusMeasurement:
    games: List[GameMeasurement] = field(default_factory=list)
    load_errors: List[Tuple[str, str]] = field(default_factory=list)

    @property
    def usable_games(self) -> List[GameMeasurement]:
        return [g for g in self.games if g.usable]

    @property
    def finished_games(self) -> List[GameMeasurement]:
        return [g for g in self.usable_games if g.game_ended and g.us_utility is not None]


# ---------------------------------------------------------------------------------------
# Observing a conversion
# ---------------------------------------------------------------------------------------


@dataclass
class _Observation:
    """State snapshots taken at each emitted decision, in `Conversion.samples` order."""

    state_ref: List[ts.GameState] = field(default_factory=list)
    node_kind: List[str] = field(default_factory=list)
    turns: List[int] = field(default_factory=list)
    vps: List[int] = field(default_factory=list)
    movers: List[int] = field(default_factory=list)
    win_actions: List[Set[int]] = field(default_factory=list)
    legal_counts: List[int] = field(default_factory=list)


@contextmanager
def _observing_conversion(obs: _Observation) -> Iterator[None]:
    """Wraps `ts.extract_observation` for the duration of one conversion.

    The converter's single call site for it is the line that appends to `Conversion.samples`,
    so one wrapper call corresponds to exactly one sample and the two lists stay aligned. The
    conversion may later drop a trailing turn (`_rewind_to_turn_start` deletes a suffix of
    `samples` when a turn's hand list turns out to be a fragment); truncating this record to
    `len(conv.samples)` afterwards reproduces that deletion exactly.
    """
    original = ts.extract_observation

    def observe(state: ts.GameState, perspective: ts.Player,
                layout: str = "legacy") -> Any:
        # Named `perspective` to match the binding it replaces: a caller may pass it
        # by keyword, and a wrapper with a different parameter name would break there
        # and nowhere else.
        mover = perspective
        if not obs.state_ref:
            obs.state_ref.append(state)
        ctx = state.ctx()
        wins: Set[int] = set()
        legal = 0
        if ctx.decision_type == ts.DecisionType.ROLL_DIE:
            # No choice is being made at a chance node, so nothing about it is the human's.
            kind = NODE_CHANCE
        else:
            kinds = classify_legal_actions(state, mover)
            legal = len(kinds)
            if legal:
                kind = NODE_DECISION
                wins = {a for a, k in kinds.items() if k == "win"}
            else:
                kind = NODE_NO_LEGAL
        obs.node_kind.append(kind)
        obs.win_actions.append(wins)
        obs.legal_counts.append(legal)
        obs.turns.append(int(state.turn))
        obs.vps.append(int(state.victory_points))
        obs.movers.append(US if mover == ts.Player.US else USSR)
        return original(state, mover, layout)

    ts.extract_observation = observe
    try:
        yield
    finally:
        ts.extract_observation = original


def _turn_boundaries(turns: Sequence[int], vps: Sequence[int],
                     ended: bool, final_vp: int, final_turn: int) -> Dict[int, int]:
    """Victory points at the end of each turn the game actually completed.

    The end of turn T is read as the first decision of turn T+1: that is after T's cleanup,
    which is where the required-military-operations penalty lands, so reading the last decision
    *inside* T instead would miss VP the turn awarded. A game that terminated during turn T
    completed no boundary after it, but its result is the position at that point, so the final
    score is recorded against T.
    """
    out: Dict[int, int] = {}
    seen: Optional[int] = None
    for turn, vp in zip(turns, vps):
        if seen is None:
            seen = turn
            continue
        if turn > seen:
            out[turn - 1] = vp
            seen = turn
    if ended and final_turn > 0:
        out[final_turn] = final_vp
    return out


def measure_game(game: Dict[str, Any], value_fn: Optional[ValueFn] = None) -> GameMeasurement:
    """Converts one corpus game and records its arc, forced-win decisions and critic values."""
    replay_id = int(game.get("replay_id", -1))
    if not game.get("all_turns"):
        # Nine cached files in the corpus hold a game shell with no turns at all -- empty
        # `all_turns`, `unique_turns`, `hands` and `stats`. There is nothing to convert, and
        # calling that a conversion failure would blame the converter for a missing download.
        return GameMeasurement(replay_id=replay_id, empty_log=True)
    obs = _Observation()
    with _observing_conversion(obs):
        conv: Conversion = convert_game(game)

    out = GameMeasurement(replay_id=replay_id)
    if conv.skipped is not None:
        out.skipped = conv.skipped
        return out
    if conv.failure is not None:
        out.failed = str(conv.failure)
        return out

    n = len(conv.samples)
    out.truncated = conv.truncated_at is not None
    out.samples = n

    if not obs.state_ref:
        # The conversion emitted nothing at all, so there is no position to read.
        out.failed = "conversion emitted no decisions"
        return out
    final_state = obs.state_ref[0]
    out.final_turn = int(final_state.turn)
    final_vp = int(final_state.victory_points)
    out.game_ended = conv.game_ended
    if out.game_ended:
        out.us_utility = float(ts.Engine.get_terminal_utility(final_state))

    turns = obs.turns[:n]
    vps = obs.vps[:n]
    # A terminal position sits one turn past the last one played whenever the game ran to final
    # scoring, so the score belongs to the turn the last decision was made in.
    scored_turn = min(out.final_turn, turns[-1]) if turns else out.final_turn
    out.vp_at_turn_end = _turn_boundaries(turns, vps, out.game_ended, final_vp, scored_turn)

    values: Optional[np.ndarray] = None
    if value_fn is not None and n:
        obs_batch = np.stack([np.asarray(s[0], dtype=np.float32) for s in conv.samples])
        mask_batch = np.stack([np.asarray(s[1]) for s in conv.samples])
        values = value_fn(obs_batch, mask_batch)

    for i in range(n):
        kind = obs.node_kind[i]
        if kind == NODE_CHANCE:
            out.excluded_chance_nodes += 1
            continue
        if kind == NODE_NO_LEGAL:
            out.excluded_no_legal += 1
            continue
        wins = obs.win_actions[i]
        action = int(conv.samples[i][2])
        out.decisions.append(DecisionRecord(
            turn=turns[i],
            victory_points=vps[i],
            mover=obs.movers[i],
            action=action,
            legal_actions=obs.legal_counts[i],
            win_available=bool(wins),
            win_taken=action in wins,
            v_win=float(values[i]) if values is not None else None,
        ))
    return out


def load_game(path: str) -> Dict[str, Any]:
    with gzip.open(path, "rt") as fh:
        loaded = json.load(fh)
    return dict(loaded)


def measure_corpus(paths: Sequence[str], value_fn: Optional[ValueFn] = None,
                   progress: bool = True) -> CorpusMeasurement:
    """Measures every game given. A game that will not load or convert is counted, not skipped."""
    result = CorpusMeasurement()
    for i, path in enumerate(paths):
        try:
            game = load_game(path)
        except Exception as exc:                       # a corrupt cache file is not a result
            result.load_errors.append((path, f"{type(exc).__name__}: {exc}"))
            continue
        try:
            result.games.append(measure_game(game, value_fn))
        except Exception as exc:
            result.games.append(GameMeasurement(
                replay_id=int(game.get("replay_id", -1)),
                failed=f"{type(exc).__name__}: {exc}"))
        if progress and (i + 1) % 10 == 0:
            print(f"  {i + 1}/{len(paths)} games", flush=True)
    return result


# ---------------------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------------------


@dataclass
class ArcSummary:
    finished: int = 0
    fragments: int = 0
    us_wins: int = 0
    ussr_wins: int = 0
    draws: int = 0
    # turn -> (N, mean VP)
    vp_by_turn: Dict[int, Tuple[int, float]] = field(default_factory=dict)
    # ending-turn bucket -> (N, US win rate)
    win_rate_by_length: Dict[str, Tuple[int, float]] = field(default_factory=dict)

    @property
    def us_win_rate(self) -> float:
        decided = self.us_wins + self.ussr_wins + self.draws
        return self.us_wins / decided if decided else float("nan")


LENGTH_BUCKETS: Tuple[Tuple[str, int, int], ...] = (
    ("ends turns 1-4", 1, 4),
    ("ends turns 5-6", 5, 6),
    ("ends turns 7-10", 7, 10),
)


def summarize_arc(m: CorpusMeasurement, turns: Sequence[int] = tuple(range(2, 11))) -> ArcSummary:
    out = ArcSummary()
    usable = m.usable_games
    finished = m.finished_games
    out.finished = len(finished)
    out.fragments = len(usable) - len(finished)
    for g in finished:
        util = g.us_utility if g.us_utility is not None else 0.0
        if util > 0:
            out.us_wins += 1
        elif util < 0:
            out.ussr_wins += 1
        else:
            out.draws += 1

    for t in turns:
        vals = [float(g.vp_at_turn_end[t]) for g in usable if t in g.vp_at_turn_end]
        out.vp_by_turn[t] = (len(vals), float(np.mean(vals)) if vals else float("nan"))

    for label, lo, hi in LENGTH_BUCKETS:
        bucket = [g for g in finished if lo <= _ending_turn(g) <= hi]
        wins = sum(1 for g in bucket
                   if g.us_utility is not None and g.us_utility > 0)
        out.win_rate_by_length[label] = (
            len(bucket), wins / len(bucket) if bucket else float("nan"))
    return out


def _ending_turn(g: GameMeasurement) -> int:
    """The turn a finished game ended on -- the turn its last decision was made in."""
    if g.decisions:
        return max(d.turn for d in g.decisions)
    return g.final_turn


@dataclass
class ForcedWinSummary:
    decisions: int = 0
    opportunities: int = 0
    taken: int = 0
    declines: int = 0
    # Declines in games whose outcome the log settles.
    declines_resolved: int = 0
    declines_then_won: int = 0
    declines_unresolved: int = 0
    games_with_opportunity: int = 0
    by_side: Dict[int, Tuple[int, int]] = field(default_factory=dict)   # side -> (opps, taken)

    @property
    def take_rate(self) -> float:
        return self.taken / self.opportunities if self.opportunities else float("nan")

    @property
    def cost_of_declining(self) -> float:
        """Share of declines whose game the decliner then lost -- the 4.4 number."""
        if not self.declines_resolved:
            return float("nan")
        return 1.0 - self.declines_then_won / self.declines_resolved


def summarize_forced_wins(m: CorpusMeasurement) -> ForcedWinSummary:
    out = ForcedWinSummary()
    per_side: Dict[int, List[int]] = {US: [0, 0], USSR: [0, 0]}
    for g in m.usable_games:
        resolved = g.game_ended and g.us_utility is not None
        had = False
        for d in g.decisions:
            out.decisions += 1
            if not d.win_available:
                continue
            had = True
            out.opportunities += 1
            per_side[d.mover][0] += 1
            if d.win_taken:
                out.taken += 1
                per_side[d.mover][1] += 1
                continue
            out.declines += 1
            if not resolved:
                out.declines_unresolved += 1
                continue
            out.declines_resolved += 1
            util = g.us_utility if g.us_utility is not None else 0.0
            if util * d.mover > 0:
                out.declines_then_won += 1
        if had:
            out.games_with_opportunity += 1
    out.by_side = {side: (v[0], v[1]) for side, v in per_side.items()}
    return out


@dataclass
class CalibrationBin:
    lo: float
    hi: float
    n: int
    mean_predicted: float
    mean_actual: float


DEFAULT_BINS: Tuple[float, ...] = (-1.0, -0.75, -0.5, -0.25, 0.0, 0.25, 0.5, 0.75, 1.0)


def summarize_calibration(m: CorpusMeasurement,
                          edges: Sequence[float] = DEFAULT_BINS) -> List[CalibrationBin]:
    """Predicted v_win against the mover's actual result, over finished games only.

    A fragment has no outcome, so its positions cannot be scored against one and are left out
    rather than assigned a result the log never states.
    """
    preds: List[float] = []
    actuals: List[float] = []
    for g in m.finished_games:
        util = g.us_utility if g.us_utility is not None else 0.0
        for d in g.decisions:
            if d.v_win is None:
                continue
            preds.append(d.v_win)
            actuals.append(util * d.mover)
    p = np.asarray(preds, dtype=np.float64)
    a = np.asarray(actuals, dtype=np.float64)
    out: List[CalibrationBin] = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        last = hi == edges[-1]
        sel = (p >= lo) & ((p <= hi) if last else (p < hi))
        n = int(sel.sum())
        out.append(CalibrationBin(
            lo=float(lo), hi=float(hi), n=n,
            mean_predicted=float(p[sel].mean()) if n else float("nan"),
            mean_actual=float(a[sel].mean()) if n else float("nan")))
    return out


# ---------------------------------------------------------------------------------------
# Critic
# ---------------------------------------------------------------------------------------


def make_value_fn(checkpoint: str, device: Optional[str] = None,
                  batch_size: int = 512) -> Tuple[ValueFn, str]:
    """Builds a batched `v_win` evaluator from a checkpoint, and returns it with its name."""
    import torch

    from tools.lib.player_agent import NeuralAgent, resolve_device

    dev = resolve_device(device)
    agent = NeuralAgent.from_checkpoint(checkpoint, device=dev)
    model = agent.model
    model.eval()

    def value_fn(obs: np.ndarray, mask: np.ndarray) -> np.ndarray:
        out: List[np.ndarray] = []
        with torch.no_grad():
            for start in range(0, obs.shape[0], batch_size):
                o = torch.from_numpy(obs[start:start + batch_size]).to(dev)
                mk = torch.from_numpy(mask[start:start + batch_size]).to(dev)
                _, v_win, _ = model.forward(o, mk)
                out.append(v_win.squeeze(-1).float().cpu().numpy())
        return np.concatenate(out) if out else np.zeros(0, dtype=np.float32)

    return value_fn, agent.name


def corpus_paths(directory: str) -> List[str]:
    names = sorted((f for f in os.listdir(directory) if f.endswith(".json.gz")),
                   key=lambda f: int(f.split(".")[0]))
    return [os.path.join(directory, f) for f in names]


# ---------------------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------------------


def _pct(x: float) -> str:
    return "n/a" if np.isnan(x) else f"{100.0 * x:.1f}%"


def _num(x: float, places: int = 2) -> str:
    return "n/a" if np.isnan(x) else f"{x:+.{places}f}"


def render_report(m: CorpusMeasurement, checkpoint: Optional[str], model_name: Optional[str],
                  turns: Sequence[int] = tuple(range(2, 11))) -> str:
    """Renders every measurement as markdown, exclusions included."""
    arc = summarize_arc(m, turns)
    fw = summarize_forced_wins(m)
    lines: List[str] = []
    add = lines.append

    add("# E2 -- the human corpus on this engine")
    add("")
    add("Measured by `ai/eval/human_corpus.py` over every game in "
        "`data/datasets/ts_replayer/`. Read alongside `research/experiments.md` 4.2, 4.4, "
        "4.5 and 4.6, whose definitions and table shapes these follow.")
    add("")

    # -- corpus accounting ---------------------------------------------------------------
    usable = m.usable_games
    skipped = [g for g in m.games if g.skipped is not None]
    failed = [g for g in m.games if g.failed is not None]
    empty = [g for g in m.games if g.empty_log]
    add("## 0. Corpus accounting")
    add("")
    add("| | count |")
    add("|:---|---:|")
    add(f"| files read | {len(m.games) + len(m.load_errors)} |")
    add(f"| files that would not load | {len(m.load_errors)} |")
    add(f"| cached files holding no turns at all | {len(empty)} |")
    add(f"| games skipped by the converter | {len(skipped)} |")
    add(f"| games whose conversion failed | {len(failed)} |")
    add(f"| games converted | {len(usable)} |")
    add(f"| ... of which reached a terminal state (`game_ended`) | {arc.finished} |")
    add(f"| ... of which are fragments (log stops first) | {arc.fragments} |")
    add(f"| decisions classified | {fw.decisions} |")
    add(f"| emitted samples excluded: chance nodes | "
        f"{sum(g.excluded_chance_nodes for g in usable)} |")
    add(f"| emitted samples excluded: no legal action in mask | "
        f"{sum(g.excluded_no_legal for g in usable)} |")
    add("")
    if empty:
        add(f"- empty cached files, replays "
            f"{', '.join(str(g.replay_id) for g in empty)}: no turns recorded")
    for g in skipped:
        add(f"- skipped, replay {g.replay_id}: {g.skipped}")
    for g in failed:
        add(f"- failed, replay {g.replay_id}: {g.failed}")
    for path, err in m.load_errors:
        add(f"- load error, {path}: {err}")
    add("")

    # -- 1. arc --------------------------------------------------------------------------
    add("## 1. Human game arc")
    add("")
    add(f"**US win rate over finished games: {_pct(arc.us_win_rate)}** "
        f"({arc.us_wins} US / {arc.ussr_wins} USSR / {arc.draws} draws, "
        f"N = {arc.finished} finished; {arc.fragments} fragments have no outcome and are "
        f"excluded from this rate).")
    add("")
    add("Mean victory_points at each turn boundary (US-positive), against 4.6's agent rows:")
    add("")
    add("| turn | " + " | ".join(str(t) for t in turns) + " |")
    add("|:---|" + "---:|" * len(turns))
    add("| control (4.6) | +0.15 | -1.02 | -1.87 | -3.03 | -3.58 | -3.91 | -3.11 | -2.49 | -4.47 |")
    add("| K=40 (4.6) | +0.01 | -0.91 | -2.26 | -1.96 | -2.20 | -2.31 | -1.95 | -1.68 | -2.18 |")
    add("| **human** | " + " | ".join(_num(arc.vp_by_turn[t][1]) for t in turns) + " |")
    add("| human N | " + " | ".join(str(arc.vp_by_turn[t][0]) for t in turns) + " |")
    add("")
    add("US win rate by how long the game ran (finished games only):")
    add("")
    add("| | " + " | ".join(label for label, _, _ in LENGTH_BUCKETS) + " |")
    add("|:---|" + "---:|" * len(LENGTH_BUCKETS))
    add("| control (4.6) | 44.8% | -- | 39.5% |")
    add("| K=40 (4.6) | 38.7% | -- | 42.0% |")
    add("| **human** | " + " | ".join(
        _pct(arc.win_rate_by_length[label][1]) for label, _, _ in LENGTH_BUCKETS) + " |")
    add("| human N | " + " | ".join(
        str(arc.win_rate_by_length[label][0]) for label, _, _ in LENGTH_BUCKETS) + " |")
    add("")

    # -- 2. forced wins ------------------------------------------------------------------
    add("## 2. Human forced-win behaviour")
    add("")
    add("| | value |")
    add("|:---|---:|")
    add(f"| decisions classified | {fw.decisions} |")
    add(f"| instant-win opportunities | {fw.opportunities} |")
    add(f"| share of all decisions | {100.0 * fw.opportunities / fw.decisions:.3f}% |"
        if fw.decisions else "| share of all decisions | n/a |")
    add(f"| games with at least one | {fw.games_with_opportunity} of {len(usable)} "
        f"({_pct(fw.games_with_opportunity / len(usable)) if usable else 'n/a'}) |")
    add(f"| taken | {fw.taken} |")
    add(f"| **take rate** | **{_pct(fw.take_rate)}** |")
    add(f"| US opportunities / taken | {fw.by_side[US][0]} / {fw.by_side[US][1]} |")
    add(f"| USSR opportunities / taken | {fw.by_side[USSR][0]} / {fw.by_side[USSR][1]} |")
    add(f"| declines | {fw.declines} |")
    add(f"| declines in games with a settled outcome | {fw.declines_resolved} |")
    add(f"| ... decliner won anyway | {fw.declines_then_won} |")
    add(f"| **cost of declining** | **{_pct(fw.cost_of_declining)}** |")
    add(f"| declines in fragments (outcome unknown, excluded) | {fw.declines_unresolved} |")
    add("")

    # -- 3. calibration ------------------------------------------------------------------
    add("## 3. Critic calibration on human positions")
    add("")
    if checkpoint is None:
        add("Not measured: no checkpoint given.")
        add("")
        return "\n".join(lines)
    bins = summarize_calibration(m)
    scored = sum(b.n for b in bins)
    add(f"Checkpoint `{checkpoint}` (`{model_name}`), value head `v_win`, over "
        f"{scored} human decisions in the {arc.finished} finished games. "
        f"Actual outcome is +1 if the mover won that game, -1 if they lost.")
    add("")
    add("| v_win bin | N | mean predicted | mean actual | gap (pred - actual) |")
    add("|:---|---:|---:|---:|---:|")
    for b in bins:
        gap = b.mean_predicted - b.mean_actual
        add(f"| [{b.lo:+.2f}, {b.hi:+.2f}{')' if b.hi != bins[-1].hi else ']'} | {b.n} | "
            f"{_num(b.mean_predicted)} | {_num(b.mean_actual)} | {_num(gap)} |")
    tot_p = sum(b.mean_predicted * b.n for b in bins if b.n) / scored if scored else float("nan")
    tot_a = sum(b.mean_actual * b.n for b in bins if b.n) / scored if scored else float("nan")
    add(f"| **all** | {scored} | {_num(tot_p)} | {_num(tot_a)} | {_num(tot_p - tot_a)} |")
    add("")
    return "\n".join(lines)


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--corpus-dir", default="data/datasets/ts_replayer")
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--device", default=None)
    ap.add_argument("--limit", type=int, default=0, help="0 measures every game")
    ap.add_argument("--output", default=None)
    args = ap.parse_args()

    paths = corpus_paths(args.corpus_dir)
    if args.limit:
        paths = paths[:args.limit]
    value_fn: Optional[ValueFn] = None
    model_name: Optional[str] = None
    if args.checkpoint:
        value_fn, model_name = make_value_fn(args.checkpoint, args.device)
    m = measure_corpus(paths, value_fn)
    report = render_report(m, args.checkpoint, model_name)
    if args.output:
        with open(args.output, "w") as fh:
            fh.write(report + "\n")
        print(f"wrote {args.output}")
    else:
        print(report)


if __name__ == "__main__":
    main()
