"""Which observation features carry information, named one by one.

`input_ablation` asks whether the *network* uses a slice. This asks the prior question: whether the
*engine* ever puts anything there. A feature the engine never writes is a constant the network
learns a bias from, and it is invisible -- the observation is memset to zero first, so a field
nobody writes reads as a legitimate 0.0 forever.

That is not hypothetical. The 512-float action history was exactly this for the whole life of the
project (`ActionHistoryBuffer::record()` was never called), and three of the four turn-aggregate
families are the same: the observation reads ops_spent_by_region, headlines_played and
space_attempts, and nothing in the engine writes any of them, while realignments_by_region -- which
the engine does write -- is never read.

Reports, per named feature, whether it ever varies across a sample of real positions, and its
range. A feature that is constant across thousands of positions from real games is either dead or
carries no signal, and the two are worth telling apart before a network is asked to use it.

    PYTHONPATH=.:build/release python -m ai.eval.feature_audit --games 300
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import List, Sequence, TypedDict

import numpy as np
import numpy.typing as npt

import ts_engine as ts
from bindings.action_encoder import ActionEncoder


@dataclass
class Feature:
    """One named span of the observation."""

    name: str
    start: int
    width: int
    note: str = ""


def v22_features() -> List[Feature]:
    """The v2.2 observation, named. Offsets follow `engine/src/observation.cpp`."""
    board = 84 * 26
    cards = 110 * 14
    g = board + cards
    feats: List[Feature] = [
        Feature("board", 0, board, "84 countries x 26"),
        Feature("cards", board, cards, "110 cards x 14"),
        # Global block, named individually -- this is where a silent gap would hide.
        Feature("global/my_vp", g + 0, 1),
        Feature("global/defcon", g + 1, 1),
        Feature("global/my_mil_ops", g + 2, 1),
        Feature("global/opp_mil_ops", g + 3, 1),
        Feature("global/my_space", g + 4, 1),
        Feature("global/opp_space", g + 5, 1),
        Feature("global/turn", g + 6, 1),
        Feature("global/action_round", g + 7, 1),
        Feature("global/i_am_phasing", g + 8, 1),
        Feature("global/phase", g + 9, 1),
        Feature("global/i_hold_china", g + 10, 1),
        Feature("global/china_playable", g + 11, 1),
        Feature("global/persistent_effects", g + 12, 45, "one float per effect bit"),
        Feature("global/defcon_dropped_to_2", g + 57, 1),
        Feature("global/ctx_stack_depth", g + 58, 1),
        Feature("global/my_space_turns_used", g + 59, 1),
        Feature("global/opp_space_turns_used", g + 60, 1),
        Feature("global/i_am_us", g + 61, 1),
        Feature("global/draw_pile_count", g + 62, 1),
        Feature("global/discard_pile_count", g + 63, 1),
        Feature("global/region_vp", g + 64, 6),
        Feature("global/opp_hand_count", g + 70, 1),
        Feature("global/my_hand_count", g + 71, 1),
        # The decision context: what is being asked right now, all pure functions of state.
        Feature("ctx/decision_type", g + 72, 8, "one-hot NONE..ROLL_DIE"),
        Feature("ctx/op_mode", g + 80, 3, "one-hot INFLUENCE/COUP/REALIGN"),
        Feature("ctx/remaining_steps", g + 83, 1, "points left in this play"),
        Feature("ctx/pending_ops_value", g + 84, 1, "effective Ops of the card being spent"),
        Feature("ctx/max_per_country", g + 85, 1),
        Feature("ctx/allow_early_stop", g + 86, 1),
        Feature("ctx/timing_ops_first", g + 87, 1),
        Feature("ctx/timing_event_first", g + 88, 1),
        Feature("ctx/event_granted_ops", g + 89, 1),
        Feature("ctx/suppress_op_event", g + 90, 1),
        Feature("ctx/temp_card_count", g + 91, 1),
    ]
    return feats


def collect(num_games: int, seed: int = 4242, layout: str = "v2.2") -> npt.NDArray[np.float32]:
    """Observations from self-play games driven by the legal-action distribution.

    Random play reaches turn ~2.7 and would leave every late-game feature constant by accident, so
    positions are taken from games played to a terminal state with a bias toward continuing: the
    point is coverage of the state space, not quality of play.
    """
    rows: List[npt.NDArray[np.float32]] = []
    rng = np.random.default_rng(seed)
    for g in range(num_games):
        state = ts.GameState()
        ts.Engine.init_game(state, int(seed + g))
        steps = 0
        while not ts.Engine.is_terminal(state) and steps < 4000:
            steps += 1
            player = state.ctx().decision_player
            if player == ts.Player.NONE:
                player = state.phasing_player
            rows.append(np.asarray(
                ts.extract_observation(state, player, layout=layout), dtype=np.float32))
            mask = ActionEncoder.get_legal_mask(state)
            legal = np.flatnonzero(np.asarray(mask))
            if legal.size == 0:
                break
            ts.Engine.step_flat(state, int(rng.choice(legal)))
    return np.stack(rows) if rows else np.zeros((0, 1), dtype=np.float32)


class AuditRow(TypedDict):
    """One feature's verdict. Typed rather than Dict[str, object] so callers can do arithmetic
    on the numbers without casting every access."""

    name: str
    width: int
    constant_cols: int
    dead: bool
    min: float
    max: float
    note: str


def audit(obs: npt.NDArray[np.float32], feats: Sequence[Feature]) -> List[AuditRow]:
    out: List[AuditRow] = []
    for f in feats:
        block = obs[:, f.start:f.start + f.width]
        col_const = [bool(np.all(block[:, c] == block[0, c])) for c in range(block.shape[1])]
        out.append({
            "name": f.name,
            "width": f.width,
            "constant_cols": int(sum(col_const)),
            "dead": bool(all(col_const)),
            "min": float(block.min()) if block.size else 0.0,
            "max": float(block.max()) if block.size else 0.0,
            "note": f.note,
        })
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--games", type=int, default=300)
    ap.add_argument("--seed", type=int, default=4242)
    ap.add_argument("--layout", default="v2.2", choices=["legacy", "v2.1", "v2.2"],
                    help="which observation layout to audit")
    a = ap.parse_args()

    obs = collect(a.games, a.seed, layout=a.layout)
    print(f"{obs.shape[0]:,} positions from {a.games} games, observation width {obs.shape[1]}\n")
    rows = audit(obs, v22_features())

    print(f"{'feature':38} {'width':>6} {'const':>7} {'min':>8} {'max':>8}  note")
    dead: List[str] = []
    for r in rows:
        flag = "  <-- DEAD" if r["dead"] else ""
        if r["dead"]:
            dead.append(r["name"])
        print(f"{r['name']:38} {r['width']:>6} {r['constant_cols']:>7} "
              f"{r['min']:>8.3f} {r['max']:>8.3f}  {r['note']}{flag}")

    total = sum(r["width"] for r in rows)
    dead_w = sum(r["width"] for r in rows if r["dead"])
    print(f"\n{dead_w} of {total} floats are constant across every position sampled "
          f"({100.0 * dead_w / max(total, 1):.1f}%)")
    if dead:
        print("dead features: " + ", ".join(dead))


if __name__ == "__main__":
    main()
