"""Ops budgets when a bonus is conditional on where the Ops are spent.

Two cards grant Ops only if *all* of them are spent in a region: the China Card (+1 if all in
Asia) and Vietnam Revolts (+1 to the USSR if all in Southeast Asia). The engine models this by
handing out the optimistic total up front and clawing it back on the first placement outside
the region.

The claw-back rewrites `remaining_steps` onto a lower base while leaving `pending_ops_value`
at the optimistic figure, and the next claw-back computes
`total_spent = pending_ops_value - remaining_steps` against that stale figure. So a second
placement outside the region is charged twice, and the player silently loses an Op.

Found while reconstructing a human game: replay 100, turn 1, USSR AR6. The USSR played Five
Year Plan for 3 Ops into South Korea, India and Thailand. The engine allowed only two of the
three.
"""

import pytest
import ts_engine as ts

from tools.lib.ts_replayer_parse import country_id

VIETNAM_REVOLTS_ACTIVE = 1 << 9
CHINA_CARD = 6
def _ussr_card_with_ops(ops: int) -> int:
    """A USSR-associated card of the given Ops value, so no opponent-event timing is involved."""
    for c in range(1, 111):
        i = ts.CardData.get_card_info(c)
        if str(i["side"]) == "USSR" and int(i["ops"]) == ops and not i["is_scoring"]:
            return c
    raise AssertionError(f"no USSR card with {ops} ops")

# Southeast Asia / Asia / elsewhere, all placeable by the USSR from its opening position.
SE_ASIA = "Thailand"
ASIA_NOT_SE = "India"
NOT_ASIA = "Iran"


def _ussr_ops_state(card: int, vietnam: bool) -> ts.GameState:
    """USSR to move, holding `card`, with a foothold in Asia to place from."""
    st = ts.GameState()
    ts.Engine.init_game(st, 4242)
    while st.current_phase == ts.Phase.SETUP:
        import numpy as np
        legal = np.flatnonzero(np.asarray(ts.ActionMask.generate_flat_mask(st)))
        ts.Engine.step_flat(st, int(legal[0]))

    # Play out the headline, or the card under test would be spent as a headline event
    # rather than for Ops, and we would measure the wrong thing entirely.
    import numpy as np
    for _ in range(40):
        if st.current_phase == ts.Phase.ACTION_ROUND:
            break
        legal = np.flatnonzero(np.asarray(ts.ActionMask.generate_flat_mask(st)))
        if len(legal) == 0:
            break
        ts.Engine.step_flat(st, int(legal[0]))
    assert st.current_phase == ts.Phase.ACTION_ROUND, "fixture never reached an action round"

    # A foothold next to each target so placement is legal everywhere we test.
    for nm, ussr in ((SE_ASIA, 1), (ASIA_NOT_SE, 1), (NOT_ASIA, 1)):
        cid = country_id(nm)
        assert cid is not None
        st.set_country(cid, 0, ussr)
    if vietnam:
        st.set_flag(VIETNAM_REVOLTS_ACTIVE)
    if card == CHINA_CARD:
        st.china_card_holder = ts.Player.USSR
        st.china_card_playable = True
    else:
        st.set_card_location(card, ts.hand_of(ts.Player.USSR))
    return st


def _remaining_after(st: ts.GameState, places) -> "list[int]":
    """Play the pending card for Ops/Influence, placing in `places`; record Ops left."""
    import numpy as np
    out = []
    for nm in places:
        cid = country_id(nm)
        assert cid is not None
        mask = np.asarray(ts.ActionMask.generate_flat_mask(st))
        legal = np.flatnonzero(mask)
        act = None
        for a in legal:
            ma = ts.ActionMask.decode_flat_action(st, int(a))
            if (int(ma.decision_type) == int(ts.DecisionType.POINT_NODE)
                    and int(ma.primary_id) == cid):
                act = int(a)
                break
        assert act is not None, f"{nm} not a legal placement (ops left {int(st.ctx().remaining_steps)})"
        ts.Engine.step_flat(st, act)
        out.append(int(st.ctx().remaining_steps))
    return out


def _drive_to_placement(st: ts.GameState, card: int) -> None:
    """Select the card, play it for Ops, choose Influence."""
    import numpy as np
    from ai.eval.positions import PLAY_MODE_ACTION
    for _ in range(8):
        ctx = st.ctx()
        mask = np.asarray(ts.ActionMask.generate_flat_mask(st))
        legal = np.flatnonzero(mask)
        if ctx.decision_type == ts.DecisionType.POINT_NODE:
            return
        act = None
        if ctx.decision_type == ts.DecisionType.SELECT_CARD:
            for a in legal:
                ma = ts.ActionMask.decode_flat_action(st, int(a))
                if int(ma.primary_id) == card:
                    act = int(a)
        elif ctx.decision_type == ts.DecisionType.SELECT_PLAY_MODE:
            act = PLAY_MODE_ACTION["ops"]
        elif ctx.decision_type == ts.DecisionType.CHOOSE_TIMING_BRANCH:
            # Playing an opponent's card for Ops: resolve the event first, as the logs do.
            for a in legal:
                ma = ts.ActionMask.decode_flat_action(st, int(a))
                if int(ma.primary_id) == 1:
                    act = int(a)
        elif ctx.decision_type == ts.DecisionType.SELECT_OP_MODE:
            for a in legal:
                ma = ts.ActionMask.decode_flat_action(st, int(a))
                if int(ma.primary_id) == int(ts.OpMode.INFLUENCE):
                    act = int(a)
        if act is None or not mask[act]:
            act = int(legal[0])
        ts.Engine.step_flat(st, act)
    raise AssertionError("never reached a placement decision")


def test_vietnam_revolts_two_placements_outside_southeast_asia() -> None:
    """3-Ops card, +1 only if all in SE Asia. Two non-SE placements must leave 1 Op."""
    card = _ussr_card_with_ops(3)
    st = _ussr_ops_state(card, vietnam=True)
    _drive_to_placement(st, card)
    assert int(st.ctx().pending_ops_value) == 4, "SE-Asia bonus should be offered up front"
    left = _remaining_after(st, [ASIA_NOT_SE, NOT_ASIA])
    assert left == [2, 1], (
        f"after two non-SE placements of a 3-Ops card the USSR should have 1 Op left, "
        f"got {left}"
    )


def test_china_card_two_placements_outside_asia() -> None:
    """China Card is 4 Ops, +1 only if all in Asia. Two non-Asia placements leave 2."""
    st = _ussr_ops_state(CHINA_CARD, vietnam=False)
    _drive_to_placement(st, CHINA_CARD)
    left = _remaining_after(st, [NOT_ASIA, NOT_ASIA])
    assert left[-1] == 2, (
        f"a 4-Ops China Card with two non-Asia placements should leave 2 Ops, got {left}"
    )


def test_china_card_and_vietnam_revolts_together() -> None:
    """Both bonuses at once: each is lost the moment its region condition breaks.

    6 Ops on offer (4 base, +1 China-in-Asia, +1 Vietnam-in-SE-Asia). Spending in SE Asia
    keeps both; stepping out to Asia loses the Vietnam Op; stepping outside Asia loses the
    China Op as well.
    """
    st = _ussr_ops_state(CHINA_CARD, vietnam=True)
    _drive_to_placement(st, CHINA_CARD)
    assert int(st.ctx().pending_ops_value) == 6, (
        f"both bonuses should be offered up front, got "
        f"{int(st.ctx().pending_ops_value)}"
    )
    left = _remaining_after(st, [SE_ASIA, ASIA_NOT_SE, NOT_ASIA])
    assert left == [5, 3, 1], (
        f"expected 5 left after SE Asia, 3 after Asia, 1 after leaving Asia; got {left}"
    )


def test_leaving_asia_directly_forfeits_both_bonuses() -> None:
    """Stepping straight out of Asia loses the China and Vietnam Ops together.

    6 on offer. One Op in Southeast Asia keeps both, leaving 5. The next Op outside Asia
    breaks both conditions at once, so the budget falls to the plain 4 with 2 spent -- 2 left,
    not the 3 that would remain if only the Vietnam bonus had lapsed.
    """
    st = _ussr_ops_state(CHINA_CARD, vietnam=True)
    _drive_to_placement(st, CHINA_CARD)
    assert int(st.ctx().pending_ops_value) == 6
    left = _remaining_after(st, [SE_ASIA, NOT_ASIA])
    assert left == [5, 2], (
        f"expected 5 left after Southeast Asia and 2 after leaving Asia entirely, got {left}"
    )


def test_vietnam_revolts_un_intervention_thailand_vs_controlled_italy() -> None:
    """Vietnam Revolts active; US controls Italy and Thailand; USSR has influence in Austria and Vietnam.

    USSR plays UN Intervention (1 Op) for operations, selecting influence placement.
    - Thailand is in Southeast Asia, so the Vietnam Revolts bonus applies (1 base + 1 bonus = 2 Ops).
      Placing in US-controlled Thailand costs 2 Ops, so USSR can afford it (Thailand is legal).
    - Italy is outside Southeast Asia, so the Vietnam Revolts bonus does not apply (1 Op plain).
      Placing in US-controlled Italy costs 2 Ops, so USSR cannot afford it (Italy is NOT legal).
    """
    import numpy as np

    st = ts.GameState()
    ts.Engine.init_game(st, 4242)
    st.defcon = 5
    st.current_phase = ts.Phase.ACTION_ROUND
    st.action_round = 1
    st.phasing_player = ts.Player.USSR
    st.set_flag(VIETNAM_REVOLTS_ACTIVE)

    italy_id = country_id("Italy")
    thailand_id = country_id("Thailand")
    austria_id = country_id("Austria")
    vietnam_id = country_id("Vietnam")
    assert italy_id is not None and thailand_id is not None
    assert austria_id is not None and vietnam_id is not None

    # US controls Italy (stability 2) and Thailand (stability 2)
    st.set_country(italy_id, 2, 0)
    st.set_country(thailand_id, 2, 0)

    # USSR has influence in Austria (adjacent to Italy) and Vietnam (adjacent to Thailand)
    st.set_country(austria_id, 0, 1)
    st.set_country(vietnam_id, 0, 1)

    un_intervention = 32
    st.set_card_location(un_intervention, ts.hand_of(ts.Player.USSR))
    st.ctx().decision_player = ts.Player.USSR
    st.ctx().decision_type = ts.DecisionType.SELECT_CARD

    # Play UN Intervention for Operations -> Influence
    ts.Engine.step(st, ts.MicroAction(ts.DecisionType.SELECT_CARD, un_intervention, 0, 0))
    ts.Engine.step(st, ts.MicroAction(ts.DecisionType.SELECT_PLAY_MODE, int(ts.Resolution.OPS_INFLUENCE), 0, 0))
    ts.Engine.step(st, ts.MicroAction(ts.DecisionType.SELECT_OP_MODE, int(ts.OpMode.INFLUENCE), 0, 0))

    assert st.ctx().decision_type == ts.DecisionType.POINT_NODE
    # Initial optimistic offer is 2 Ops (1 base + 1 SE Asia)
    assert int(st.ctx().pending_ops_value) == 2
    assert int(st.ctx().remaining_steps) == 2

    # Query action mask
    mask = np.asarray(ts.ActionMask.generate_flat_mask(st))
    assert mask[119 + thailand_id] == 1, (
        "Thailand (SE Asia) should be legal because Vietnam Revolts bonus gives 2 Ops to pay cost 2"
    )
    assert mask[119 + italy_id] == 0, (
        "Italy (Europe, cost 2) must NOT be legal because non-SE placements only have 1 Op"
    )

    # Placing influence into Thailand succeeds and consumes all 2 Ops
    ts.Engine.step_flat(st, 119 + thailand_id)
    assert st.get_country(thailand_id).ussr_influence == 1
    assert int(st.ctx().remaining_steps) == 0
