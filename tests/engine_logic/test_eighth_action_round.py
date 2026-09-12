"""The eighth Action Round belongs only to the player who earned it.

max_ar is a property of the turn, so raising it to 8 for one side used to hand the extra
round to both. North Sea Oil ("US may play 8 action rounds this turn") therefore gave the
USSR a free action round as well, and so did a Space Station only the opponent had
reached. Seen in review_91004, review_91009 and review_91012, where the USSR took an AR8 it
was never owed.
"""

from typing import List, Tuple

import pytest
import ts_engine as ts

from bot import ExploratoryBot


def _entitled(st: ts.GameState, player: ts.Player) -> bool:
    """North Sea Oil (US only), or a Space Station the opponent has not matched."""
    us, ussr = int(st.us_space_track), int(st.ussr_space_track)
    if player == ts.Player.US:
        return st.has_flag(ts.EffectBits.NORTH_SEA_OIL_ACTIVE) or (us >= 8 > ussr)
    return ussr >= 8 > us


def _ar8_plays(seed: int) -> List[Tuple[int, str, bool]]:
    """Every genuine AR8 card selection in one game: (turn, player, was_entitled)."""
    st = ts.GameState()
    ts.Engine.init_game(st, seed)
    bots = {"US": ExploratoryBot("US", rng_seed=seed * 2 + 1),
            "USSR": ExploratoryBot("USSR", rng_seed=seed * 2 + 2)}
    found: List[Tuple[int, str, bool]] = []

    for _ in range(3000):
        if ts.Engine.is_terminal(st):
            break
        ctx = st.ctx()
        if (ctx.decision_player == ts.Player.NONE
                and ctx.decision_type == ts.DecisionType.ROLL_DIE):
            ts.Engine.step(st, ts.MicroAction(ts.DecisionType.ROLL_DIE, 0, 0, 0))
            continue
        # resolving_card == 0 is what separates a fresh action-round card choice from a
        # card prompt raised while the turn is being wound up. end_turn leaves
        # action_round at 8 and the phase at ACTION_ROUND while it runs, so the Space Walk
        # held-card discard looks identical without this guard.
        if (int(st.action_round) == 8
                and st.current_phase == ts.Phase.ACTION_ROUND
                and ctx.decision_type == ts.DecisionType.SELECT_CARD
                and int(ctx.resolving_card) == 0
                and ctx.decision_player == st.phasing_player):
            p = st.phasing_player
            found.append((int(st.turn), "US" if p == ts.Player.US else "USSR",
                          _entitled(st, p)))
        d = st.to_dict()
        legal = d.get("legal_actions", {})
        a = bots[legal.get("decision_player")].select_action(d, legal)
        if a is None:
            break
        ts.Engine.step(st, ts.MicroAction(ts.DecisionType(a["decision_type"]),
                                          a.get("primary_id", 0), a.get("secondary_id", 0),
                                          a.get("flags", 0)))
    return found


@pytest.mark.parametrize("lo,hi", [(1, 400)])
def test_no_unearned_eighth_action_round(lo: int, hi: int) -> None:
    """Across many games, nobody may take an AR8 without the entitlement for it."""
    unearned = []
    total = 0
    for seed in range(lo, hi + 1):
        for turn, player, ok in _ar8_plays(seed):
            total += 1
            if not ok:
                unearned.append((seed, turn, player))

    assert total > 0, "no AR8 occurred at all; the test would pass vacuously"
    assert not unearned, (
        f"{len(unearned)} of {total} eighth Action Rounds were taken by a player not "
        f"entitled to one, e.g. {unearned[:5]}"
    )
