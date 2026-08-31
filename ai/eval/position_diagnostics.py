"""Diagnostics for the *positions* self-play produces, not just the results.

Win rate is a poor progress signal for this agent: it plateaus while the underlying play
stays incoherent. These metrics look at the board instead, and they moved first when
anything was actually wrong.

Two findings motivate them, measured on checkpoints/dec_prio_off/snapshot_final:

* Roughly eight of the 29 battlegrounds sit completely untouched from turn 8 onwards, and
  the count stops falling -- the agent gives up contesting the board rather than slowly
  getting to it. Crucially the *same* battlegrounds are always the empty ones: Algeria
  97.9%, Saudi Arabia 95.4%, Libya 91.0%, India 78.0%, versus Iraq 2.2%, South Africa
  2.5%, Poland 2.9%. The always-contested ones are exactly those seeded by the opening
  setup. The agent fights where it was placed and never opens a new front.

* Only about a fifth of games reach turn 10, so the Late War deck is nearly unplayed.

A "salvageable" position is one worth resuming from: neither side is far enough ahead that
the game is decided, and there is something left to play for.
"""

from __future__ import annotations

import collections
import statistics
from typing import Any, Callable, Dict, List, Optional

import ts_engine as ts

REGIONS = (
    ts.Region.EUROPE, ts.Region.ASIA, ts.Region.MIDDLE_EAST,
    ts.Region.AFRICA, ts.Region.CENTRAL_AMERICA, ts.Region.SOUTH_AMERICA,
)

BATTLEGROUNDS: List[int] = [
    cid for cid in range(84) if ts.MapData.get_country_info(cid)["battleground"]
]

# Defaults for is_salvageable. max_region_sum is deliberately strict: at turn 8 only ~10%
# of self-play positions pass it, which is itself the diagnosis rather than a tuning
# problem -- see profile_self_play for the per-turn breakdown.
MAX_ABS_VP = 10
MAX_REGION_SUM = 20


def region_score_sums(state: ts.GameState) -> tuple[int, int]:
    """(US, USSR) totals if every region were scored right now.

    Scoring.evaluate_region does not mutate, so this reuses the engine's own scoring
    rather than reimplementing presence/domination/control in Python.
    """
    us = ussr = 0
    for region in REGIONS:
        summary = ts.Scoring.evaluate_region(state, region, False)
        us += int(summary.us_score)
        ussr += int(summary.ussr_score)
    return us, ussr


def empty_battlegrounds(state: ts.GameState) -> List[int]:
    """Battlegrounds with no influence from either side."""
    out = []
    for cid in BATTLEGROUNDS:
        c = state.get_country(cid)
        if int(c.us_influence) == 0 and int(c.ussr_influence) == 0:
            out.append(cid)
    return out


def is_salvageable(
    state: ts.GameState,
    max_abs_vp: int = MAX_ABS_VP,
    max_region_sum: int = MAX_REGION_SUM,
) -> bool:
    """Is this position worth resuming from -- undecided, with game left to play?

    A decided position is worse than useless as a start state: with terminal-only reward
    every action there returns the same value, so it contributes no gradient at all while
    still consuming rollout budget.
    """
    if ts.Engine.is_terminal(state):
        return False
    if abs(int(state.victory_points)) > max_abs_vp:
        return False
    us, ussr = region_score_sums(state)
    return us <= max_region_sum and ussr <= max_region_sum


def _drain(state: ts.GameState) -> None:
    while (not ts.Engine.is_terminal(state)
           and state.ctx().decision_player == ts.Player.NONE
           and state.ctx().decision_type == ts.DecisionType.ROLL_DIE):
        ts.Engine.step(state, ts.MicroAction(ts.DecisionType.ROLL_DIE, 0, 0, 0))


def profile_self_play(
    select_action: Callable[[ts.GameState, Any], Optional[int]],
    num_games: int = 100,
    base_seed: int = 810_000,
    max_steps: int = 3000,
) -> Dict[str, Any]:
    """Play games and describe the positions they pass through.

    select_action(state, player) -> flat action index, matching PlayerAgent.select_action.
    """
    reach: collections.Counter = collections.Counter()
    salvageable: collections.Counter = collections.Counter()
    empty_bgs: Dict[int, List[int]] = collections.defaultdict(list)
    per_bg_empty: collections.Counter = collections.Counter()
    late_samples = 0
    us_scores: Dict[int, List[int]] = collections.defaultdict(list)
    ussr_scores: Dict[int, List[int]] = collections.defaultdict(list)
    final_turns: List[int] = []

    for i in range(num_games):
        state = ts.GameState()
        ts.Engine.init_game(state, base_seed + i)
        seen: set = set()
        for _ in range(max_steps):
            _drain(state)
            if ts.Engine.is_terminal(state):
                break
            ctx = state.ctx()
            turn = int(state.turn)
            if turn not in seen:
                seen.add(turn)
                reach[turn] += 1
                if is_salvageable(state):
                    salvageable[turn] += 1
                empty = empty_battlegrounds(state)
                empty_bgs[turn].append(len(empty))
                us, ussr = region_score_sums(state)
                us_scores[turn].append(us)
                ussr_scores[turn].append(ussr)
                if turn >= 6:
                    late_samples += 1
                    for cid in empty:
                        per_bg_empty[cid] += 1
            action = select_action(state, ctx.decision_player)
            if action is None:
                break
            ts.Engine.step_flat(state, int(action))
        final_turns.append(int(state.turn))

    def mean(xs: List[int]) -> float:
        return statistics.mean(xs) if xs else 0.0

    per_turn = {
        t: {
            "reached_frac": reach[t] / num_games,
            "salvageable_frac": salvageable[t] / num_games,
            "salvageable_given_reached": salvageable[t] / reach[t] if reach[t] else 0.0,
            "mean_empty_battlegrounds": mean(empty_bgs[t]),
            "mean_us_region_score": mean(us_scores[t]),
            "mean_ussr_region_score": mean(ussr_scores[t]),
        }
        for t in sorted(reach)
    }

    return {
        "num_games": num_games,
        "mean_final_turn": mean(final_turns),
        "per_turn": per_turn,
        "empty_battleground_rate_late": {
            ts.MapData.get_country_info(cid)["name"]: per_bg_empty[cid] / late_samples
            for cid in sorted(per_bg_empty, key=lambda c: -per_bg_empty[c])
        } if late_samples else {},
        "scalars": scalar_metrics(per_turn, mean(final_turns)),
    }


def scalar_metrics(per_turn: Dict[int, Dict[str, float]], mean_final_turn: float) -> Dict[str, float]:
    """The handful worth logging every evaluation.

    empty_battlegrounds_turn8 is the sharpest of these: it is a direct read on whether the
    agent has started contesting the board, and it moves long before win rate does.
    """
    def at(turn: int, key: str) -> float:
        return per_turn.get(turn, {}).get(key, 0.0)

    return {
        "diag/mean_final_turn": mean_final_turn,
        "diag/frac_reaching_turn9": at(9, "reached_frac"),
        "diag/empty_battlegrounds_turn8": at(8, "mean_empty_battlegrounds"),
        "diag/empty_battlegrounds_turn5": at(5, "mean_empty_battlegrounds"),
        "diag/salvageable_frac_turn6": at(6, "salvageable_frac"),
        "diag/salvageable_given_reached_turn6": at(6, "salvageable_given_reached"),
    }


def format_report(profile: Dict[str, Any], top_battlegrounds: int = 10) -> str:
    """Human-readable summary for the snapshot report."""
    lines = [
        f"positions over {profile['num_games']} self-play games "
        f"(mean final turn {profile['mean_final_turn']:.2f})",
        "",
        f"{'turn':>4} {'reached':>9} {'salvageable':>12} {'empty BGs':>10} "
        f"{'US score':>9} {'USSR score':>11}",
    ]
    for turn, row in profile["per_turn"].items():
        lines.append(
            f"{turn:>4} {100*row['reached_frac']:>8.1f}% {100*row['salvageable_frac']:>11.1f}% "
            f"{row['mean_empty_battlegrounds']:>10.2f} {row['mean_us_region_score']:>9.1f} "
            f"{row['mean_ussr_region_score']:>11.1f}"
        )
    rates = profile.get("empty_battleground_rate_late") or {}
    if rates:
        lines += ["", "most-neglected battlegrounds (turn >= 6):"]
        for name, rate in list(rates.items())[:top_battlegrounds]:
            lines.append(f"  {name:<20} empty in {100*rate:5.1f}% of positions")
    return "\n".join(lines)
