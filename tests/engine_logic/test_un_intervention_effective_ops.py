"""UN Intervention spends the named card's *effective* Ops, not its printed value.

The card is used for Operations exactly as if it had been played for them, so every modifier
applies: Vietnam Revolts turns a printed 1 into 2 while the USSR stays in Southeast Asia, and
the budget ladder withdraws the bonus if they leave. The engine granted only the printed value,
so at turn 2 AR4 of ts-replayer game 108 the USSR named CIA Created under Vietnam Revolts and
placed influence in two Southeast Asian countries where the engine could afford only one.
"""
import numpy as np
import ts_engine as ts

UN_INTERVENTION = 32
CIA_CREATED = 26          # US-associated, printed 1 Op
LAOS_CAMBODIA = 35
THAILAND = 36
VIETNAM = 37
CANADA = 0


def _ussr_action_round() -> ts.GameState:
    state = ts.GameState()
    ts.Engine.init_game(state, 12345)
    while (not ts.Engine.is_terminal(state)
           and state.ctx().decision_type == ts.DecisionType.ROLL_DIE):
        mask = np.asarray(ts.ActionMask.generate_flat_mask(state))
        ts.Engine.step_flat(state, int(np.flatnonzero(mask)[0]))
    state.current_phase = ts.Phase.ACTION_ROUND
    state.action_round = 1
    state.phasing_player = ts.Player.USSR
    state.set_flag(ts.EffectBits.VIETNAM_REVOLTS_ACTIVE)
    # Vietnam Revolts puts the USSR into Vietnam, which is what makes neighbouring
    # Laos/Cambodia a legal placement in the first place.
    state.set_country(VIETNAM, 0, 2)
    for c in range(1, 111):
        if ts.in_hand_of(state.get_card_location(c), ts.Player.USSR):
            state.set_card_location(c, ts.CardLocation.DISCARD_PILE)
    state.set_card_location(UN_INTERVENTION, ts.hand_of(ts.Player.USSR))
    state.set_card_location(CIA_CREATED, ts.hand_of(ts.Player.USSR))
    state.ctx().decision_player = ts.Player.USSR
    state.ctx().decision_type = ts.DecisionType.SELECT_CARD
    return state


def _name_cia_created(state: ts.GameState) -> None:
    ts.Engine.step(state, ts.MicroAction(ts.DecisionType.SELECT_CARD, UN_INTERVENTION, 0, 0))
    ts.Engine.step(state, ts.MicroAction(
        ts.DecisionType.SELECT_PLAY_MODE, int(ts.Resolution.EVENT), 0, 0))
    ts.Engine.step(state, ts.MicroAction(ts.DecisionType.SELECT_CARD, CIA_CREATED, 0, 0))


def test_vietnam_revolts_raises_the_named_cards_ops() -> None:
    state = _ussr_action_round()
    _name_cia_created(state)
    assert state.ctx().decision_type == ts.DecisionType.SELECT_OP_MODE
    assert int(state.ctx().pending_ops_value) == 2, (
        "a printed 1 Op card is worth 2 to the USSR while Vietnam Revolts is in play")


def test_two_southeast_asian_placements_are_affordable() -> None:
    state = _ussr_action_round()
    _name_cia_created(state)
    ts.Engine.step(state, ts.MicroAction(
        ts.DecisionType.SELECT_OP_MODE, int(ts.OpMode.INFLUENCE), 0, 0))
    ts.Engine.step(state, ts.MicroAction(ts.DecisionType.POINT_NODE, LAOS_CAMBODIA, 0, 0))
    ts.Engine.step(state, ts.MicroAction(ts.DecisionType.POINT_NODE, THAILAND, 0, 0))
    assert int(state.get_country(LAOS_CAMBODIA).ussr_influence) == 1
    assert int(state.get_country(THAILAND).ussr_influence) == 1, (
        "both Southeast Asian placements fit inside the raised budget")


def test_leaving_southeast_asia_withdraws_the_bonus() -> None:
    """The ladder still applies: step outside and the extra Op is gone."""
    state = _ussr_action_round()
    _name_cia_created(state)
    ts.Engine.step(state, ts.MicroAction(
        ts.DecisionType.SELECT_OP_MODE, int(ts.OpMode.INFLUENCE), 0, 0))
    state.set_country(CANADA, 0, 5)          # so a placement there is legal at all
    ts.Engine.step(state, ts.MicroAction(ts.DecisionType.POINT_NODE, CANADA, 0, 0))
    assert int(state.ctx().remaining_steps) == 0, (
        "leaving Southeast Asia forfeits the Vietnam Revolts bonus")


def test_without_vietnam_revolts_only_the_printed_op() -> None:
    state = _ussr_action_round()
    state.clear_flag(ts.EffectBits.VIETNAM_REVOLTS_ACTIVE)
    _name_cia_created(state)
    assert int(state.ctx().pending_ops_value) == 1
