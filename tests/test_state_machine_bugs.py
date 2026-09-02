"""Tests exposing defects in engine/src/state_machine.cpp.

These tests reproduce three distinct bugs in state_machine.cpp:
1. Realignments double-deduct conditional Ops budgets and fail to follow the ladder logic
   fixed for influence placement in commit b716ddf (lines 973-990).
2. China Card coup in an Asian country outside Southeast Asia erroneously retains the
   Vietnam Revolts +1 Op bonus (lines 845-855).
3. advance_after_ops pops only a single context frame when an event-granted Op finishes,
   leaking the stack and dropping the phasing player's paid-for Ops on nested EVENT_FIRST
   plays (lines 228-234).
"""

import pytest
import ts_engine as ts

from tools.lib.ts_replayer_parse import country_id

VIETNAM_REVOLTS_ACTIVE = 1 << 9
CHINA_CARD = 6
FIVE_YEAR_PLAN = 5
CIA_CREATED = 26
CANADA = 0


def _cid(name: str) -> int:
    res = country_id(name)
    assert res is not None, f"unknown country: {name}"
    return res


def _find_ussr_card_with_ops(ops: int) -> int:
    for c in range(1, 111):
        info = ts.CardData.get_card_info(c)
        if str(info["side"]) == "USSR" and int(info["ops"]) == ops and not info["is_scoring"]:
            return c
    raise AssertionError(f"no USSR card with {ops} ops found")


# ======================================================================================
# Bug 1: Realignment conditional Ops budget double-deduction & ladder failure
# ======================================================================================

@pytest.mark.xfail(strict=True, reason="state_machine.cpp:983-990 double-deducts conditional ops during realignments")
def test_realignment_vietnam_revolts_double_charges_outside_southeast_asia() -> None:
    """A 3-Ops card played for realignments outside SE Asia under Vietnam Revolts must give 3 Ops.

    Under Vietnam Revolts, a 3-Ops card is granted 4 Ops up front. When the player realigns
    outside Southeast Asia, the +1 bonus is forfeited, leaving the plain 3 Ops.
    With 2 realignments spent (e.g. 1 in India, 1 in Iran), the player should have 3 - 2 = 1 Op
    left for a 3rd realignment.

    BUG in state_machine.cpp:983-990:
    pending_ops_value is never updated during realignments, and total_spent is computed as
    (pending_ops_value - remaining_steps) against the stale initial value 4.
    On the 2nd realignment in Iran, total_spent = 4 - 1 = 3 >= non_se_base (3), so remaining_steps
    is set to 0. The player gets only 2 realignments instead of 3.
    """
    card = _find_ussr_card_with_ops(3)
    st = ts.GameState()
    ts.Engine.init_game(st, 4242)
    st.current_phase = ts.Phase.ACTION_ROUND
    st.action_round = 1
    st.phasing_player = ts.Player.USSR
    st.set_flag(VIETNAM_REVOLTS_ACTIVE)

    # Footholds for legal realignments
    for nm in ("India", "Iran", "Thailand"):
        st.set_country(_cid(nm), 2, 2)

    st.set_card_location(card, ts.CardLocation.HAND_USSR)
    st.ctx().decision_player = ts.Player.USSR
    st.ctx().decision_type = ts.DecisionType.SELECT_CARD

    ts.Engine.step(st, ts.MicroAction(ts.DecisionType.SELECT_CARD, card, 0, 0))
    ts.Engine.step(st, ts.MicroAction(ts.DecisionType.SELECT_PLAY_MODE, int(ts.PlayMode.OPS), 0, 0))
    ts.Engine.step(st, ts.MicroAction(ts.DecisionType.SELECT_OP_MODE, int(ts.OpMode.REALIGN), 0, 0))

    assert int(st.ctx().pending_ops_value) == 4, "initial budget should be 4 (3 base + 1 SE Asia)"

    # Realign 1 in India (Asia, not Southeast Asia -> forfeits Vietnam bonus, 3 budget - 1 spent = 2 left)
    ts.Engine.step(st, ts.MicroAction(ts.DecisionType.POINT_NODE, _cid("India"), 0, 0))
    ts.Engine.step(st, ts.MicroAction(ts.DecisionType.ROLL_DIE, 0, 1, 1))
    assert int(st.ctx().remaining_steps) == 2, "after 1st realign, should have 2 steps left"

    # Realign 2 in Iran (Middle East -> non-Asia, 3 budget - 2 spent = 1 left)
    ts.Engine.step(st, ts.MicroAction(ts.DecisionType.POINT_NODE, _cid("Iran"), 0, 0))
    ts.Engine.step(st, ts.MicroAction(ts.DecisionType.ROLL_DIE, 0, 1, 1))

    # FAILS HERE: st.ctx().remaining_steps == 0 instead of 1 because of double-deduction
    assert int(st.ctx().remaining_steps) == 1, (
        f"after 2 realignments of a 3-Ops card, USSR should have 1 Op left, "
        f"but remaining_steps was wiped to {st.ctx().remaining_steps}"
    )


@pytest.mark.xfail(strict=True, reason="state_machine.cpp:975-990 strips China Card Asia bonus on realignments in Asia")
def test_realignment_china_card_under_vietnam_revolts_in_asia_strips_asia_bonus() -> None:
    """The China Card (4 base + 1 Asia) under Vietnam Revolts (+1 SE Asia) realigning in India.

    Initial offer is 6 Ops (4 base + 1 China-in-Asia + 1 Vietnam-in-SE-Asia).
    When the USSR realigns in India (which is in Asia, but NOT in Southeast Asia):
    - The Vietnam Revolts bonus (+1) should be forfeited.
    - The China Card Asia bonus (+1) MUST BE KEPT because India IS IN ASIA.
    Budget in Asia is 4 base + 1 Asia = 5 Ops.
    After 1 realignment in India, remaining_steps should be 5 - 1 = 4.
    After 2 realignments in India, remaining_steps should be 5 - 2 = 3.

    BUG in state_machine.cpp:975-990:
    Line 975 checks `op_card == CHINA_CARD && c_info.region != Region::ASIA`.
    For India, c_info.region is ASIA, so the condition is FALSE.
    Execution enters the `else if` on line 983:
    `non_se_base = Operations::get_effective_ops(..., Region::NONE_REGION)`.
    For China Card with Region::NONE_REGION, non_se_base is 4 (both bonuses stripped!).
    remaining_steps drops to 4 - 1 = 3 after Realign 1, and drops to 0 after Realign 2!
    The China Card yields only 2 realignments in Asia instead of 5!
    """
    st = ts.GameState()
    ts.Engine.init_game(st, 4242)
    st.current_phase = ts.Phase.ACTION_ROUND
    st.action_round = 1
    st.phasing_player = ts.Player.USSR
    st.set_flag(VIETNAM_REVOLTS_ACTIVE)
    st.china_card_holder = ts.Player.USSR
    st.china_card_playable = True

    for nm in ("India", "Pakistan"):
        st.set_country(_cid(nm), 2, 2)

    st.ctx().decision_player = ts.Player.USSR
    st.ctx().decision_type = ts.DecisionType.SELECT_CARD

    ts.Engine.step(st, ts.MicroAction(ts.DecisionType.SELECT_CARD, CHINA_CARD, 0, 0))
    ts.Engine.step(st, ts.MicroAction(ts.DecisionType.SELECT_PLAY_MODE, int(ts.PlayMode.OPS), 0, 0))
    ts.Engine.step(st, ts.MicroAction(ts.DecisionType.SELECT_OP_MODE, int(ts.OpMode.REALIGN), 0, 0))

    assert int(st.ctx().pending_ops_value) == 6, "initial offer should be 6 Ops"

    # Realign 1 in India (in Asia): budget should be 5, remaining_steps should be 4
    ts.Engine.step(st, ts.MicroAction(ts.DecisionType.POINT_NODE, _cid("India"), 0, 0))
    ts.Engine.step(st, ts.MicroAction(ts.DecisionType.ROLL_DIE, 0, 1, 1))

    # FAILS HERE: remaining_steps is 3 instead of 4 (the Asia bonus was stripped inside Asia!)
    assert int(st.ctx().remaining_steps) == 4, (
        f"China Card realigning in India (Asia) should retain the Asia bonus (budget 5, 1 spent -> 4 left), "
        f"but remaining_steps became {st.ctx().remaining_steps}"
    )


# ======================================================================================
# Bug 2: Coup with China Card in non-SE Asia improperly gets Vietnam Revolts bonus
# ======================================================================================

@pytest.mark.xfail(strict=True, reason="state_machine.cpp:845-855 awards Vietnam Revolts bonus on China Card coup outside SE Asia")
def test_coup_china_card_in_non_se_asia_must_not_receive_vietnam_revolts_bonus() -> None:
    """A USSR coup in Pakistan with China Card under Vietnam Revolts must have 5 Ops, not 6.

    Pakistan is in Asia (Region::ASIA), but NOT in Southeast Asia (in_southeast_asia == False).
    Vietnam Revolts text: 'All Operations cards played by the USSR have their value increased by 1,
    provided that all Operations are conducted in Southeast Asia.'
    China Card text: '+1 Operation value if all Operations are conducted in Asia.'
    Therefore, a coup in Pakistan qualifies for the China Card Asia bonus (4 + 1 = 5 Ops),
    but does NOT qualify for the Vietnam Revolts bonus.

    BUG in state_machine.cpp:845-855:
    When op_card == CHINA_CARD, coup_ops is set by calling get_effective_ops with c_info.region (ASIA).
    Inside get_modified_ops, target_region == Region::ASIA triggers VIETNAM_REVOLTS_ACTIVE.
    Consequently, coup_ops is computed as 6 Ops instead of 5 Ops.
    """
    st = ts.GameState()
    ts.Engine.init_game(st, 4242)
    st.defcon = 5
    st.current_phase = ts.Phase.ACTION_ROUND
    st.action_round = 1
    st.phasing_player = ts.Player.USSR
    st.set_flag(VIETNAM_REVOLTS_ACTIVE)
    st.china_card_holder = ts.Player.USSR
    st.china_card_playable = True

    pak_id = _cid("Pakistan")
    st.set_country(pak_id, 2, 0)

    st.ctx().decision_player = ts.Player.USSR
    st.ctx().decision_type = ts.DecisionType.SELECT_CARD

    ts.Engine.step(st, ts.MicroAction(ts.DecisionType.SELECT_CARD, CHINA_CARD, 0, 0))
    ts.Engine.step(st, ts.MicroAction(ts.DecisionType.SELECT_PLAY_MODE, int(ts.PlayMode.OPS), 0, 0))
    ts.Engine.step(st, ts.MicroAction(ts.DecisionType.SELECT_OP_MODE, int(ts.OpMode.COUP), 0, 0))

    # Point to Pakistan to execute the coup
    ts.Engine.step(st, ts.MicroAction(ts.DecisionType.POINT_NODE, pak_id, 0, 0))

    # FAILS HERE: pending_ops_value is 6 instead of 5
    assert int(st.ctx().pending_ops_value) == 5, (
        f"Coup in Pakistan (non-SE Asia) should have 5 Ops (4 China base + 1 Asia bonus), "
        f"but got {st.ctx().pending_ops_value} (Vietnam Revolts bonus was improperly awarded outside SE Asia)"
    )


# ======================================================================================
# Bug 3: advance_after_ops incomplete frame unwind on nested EVENT_FIRST Ops
# ======================================================================================

def test_nested_event_granting_ops_under_event_first_must_unwind_to_player_ops() -> None:
    """When an event-granted Op finishes, advance_after_ops must unwind to the player's Ops.

    Scenario:
    1. USSR plays Five Year Plan for Ops on USSR Action Round 1, choosing EVENT_FIRST.
       A frame is pushed: Depth 0 = USSR waiting for SELECT_OP_MODE (3 Ops).
    2. Five Year Plan event executes: it randomly discards a card from USSR hand.
       The discarded card is CIA Created (a US event).
    3. Five Year Plan pushes context for CIA Created: Depth 2.
    4. CIA Created gives the US 1 Op. US spends that 1 Op on influence in Canada.
    5. When the US finishes placing influence in Canada, advance_after_ops is called.

    BUG in state_machine.cpp:228-234:
    advance_after_ops executes:
        if (state.ctx_stack_depth > 0) {
            state.pop_context();
            if (state.ctx().decision_type != DecisionType::SELECT_OP_MODE) {
                advance_after_action_round(state);
            }
            return;
        }
    It pops ONCE from Depth 2 to Depth 1 (Five Year Plan's frame).
    In Five Year Plan's frame, decision_type is NOT SELECT_OP_MODE.
    So it calls advance_after_action_round(state)!
    The action round immediately ends, phasing_player advances to US,
    and USSR completely loses the 3 Ops it paid for at Depth 0!
    """
    st = ts.GameState()
    ts.Engine.init_game(st, 12345)
    st.defcon = 5
    st.current_phase = ts.Phase.ACTION_ROUND
    st.action_round = 1
    st.phasing_player = ts.Player.USSR
    st.ctx().decision_player = ts.Player.USSR
    st.ctx().decision_type = ts.DecisionType.SELECT_CARD

    # Isolate hand to exactly Five Year Plan and CIA Created
    for i in range(1, 111):
        if st.get_card_location(i) == ts.CardLocation.HAND_USSR:
            st.set_card_location(i, ts.CardLocation.DISCARD_PILE)

    st.set_card_location(FIVE_YEAR_PLAN, ts.CardLocation.HAND_USSR)
    st.set_card_location(CIA_CREATED, ts.CardLocation.HAND_USSR)

    # USSR plays Five Year Plan for Ops, EVENT_FIRST
    ts.Engine.step(st, ts.MicroAction(ts.DecisionType.SELECT_CARD, FIVE_YEAR_PLAN, 0, 0))
    ts.Engine.step(st, ts.MicroAction(ts.DecisionType.SELECT_PLAY_MODE, int(ts.PlayMode.OPS), 0, 0))
    ts.Engine.step(st, ts.MicroAction(ts.DecisionType.CHOOSE_TIMING_BRANCH, 1, 0, 0))  # EVENT_FIRST

    # US receives 1 Op from CIA Created event and places influence in Canada
    assert st.ctx().decision_player == ts.Player.US
    assert st.ctx().decision_type == ts.DecisionType.SELECT_OP_MODE
    ts.Engine.step(st, ts.MicroAction(ts.DecisionType.SELECT_OP_MODE, int(ts.OpMode.INFLUENCE), 0, 0))
    ts.Engine.step(st, ts.MicroAction(ts.DecisionType.POINT_NODE, CANADA, 0, 0))

    # After US finishes the event's Op, control MUST return to USSR to spend Five Year Plan's 3 Ops
    # FAILS HERE: phasing_player is Player.US, decision_type is SELECT_CARD, and stack depth is 1
    assert st.phasing_player == ts.Player.USSR, (
        f"Phasing player should still be USSR, but became {st.phasing_player}"
    )
    assert st.ctx().decision_player == ts.Player.USSR, (
        f"Decision player should be USSR, but got {st.ctx().decision_player}"
    )
    assert st.ctx().decision_type == ts.DecisionType.SELECT_OP_MODE, (
        f"Decision type should be SELECT_OP_MODE for USSR's Ops, but got {st.ctx().decision_type}"
    )
    assert int(st.ctx().pending_ops_value) == 3, (
        f"USSR should have 3 Ops to spend from Five Year Plan, but got {st.ctx().pending_ops_value}"
    )
