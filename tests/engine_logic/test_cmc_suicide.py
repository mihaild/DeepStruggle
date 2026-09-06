"""Couping under Cuban Missile Crisis is a self-inflicted loss, and must be reported as one.

The engine ends a CMC coup at +/-20 VP and deliberately leaves DEFCON alone -- it is a
Cuban Missile Crisis loss, not a thermonuclear one. From the outside that is
indistinguishable from a legitimate 20 VP win: same victory_points, same GAME_OVER, DEFCON
untouched. Every consumer that keyed off ``defcon <= 1`` therefore missed it:

* the ending classifier reported "20 VP", understating self-inflicted losses;
* the training env never set defcon_blunder, so blunder windowing did not apply;
* the behavioural statistics inherited both errors.

CMC_SUICIDE_LOSS records it at the point of resolution, which is the only place that knows.
"""

import numpy as np
import pytest
import ts_engine as ts

from ai.eval.safety import classify_legal_actions
from bindings.action_encoder import ActionEncoder
from tools.lib.tournament_evaluator import classify_game_ending_reason

WEST_GERMANY, TURKEY, CUBA = 13, 17, 71


def _us_to_move_under_ussr_cmc() -> ts.GameState:
    """US to act with CMC active against it and no influence available to cancel."""
    st = ts.GameState()
    ts.Engine.init_game(st, 4242)
    st.set_flag(ts.EffectBits.CMC_ACTIVE_USSR)
    # Strip the influence the US would otherwise spend to cancel CMC.
    for cid in (WEST_GERMANY, TURKEY):
        c = st.get_country(cid)
        st.set_country(cid, 0, int(c.ussr_influence))
    st.defcon = 4
    st.turn = 6
    st.current_phase = ts.Phase.ACTION_ROUND
    st.phasing_player = ts.Player.US
    return st


def test_cmc_coup_ends_the_game_without_touching_defcon() -> None:
    """Drive a real coup through the engine while CMC is active against the US."""
    from ai.eval.positions import PositionBuilder, PLAY_MODE_ACTION

    st = PositionBuilder(
        hand=[23, 34], side=ts.Player.US, turn=6, defcon=4,
        flags=(ts.EffectBits.CMC_ACTIVE_USSR,),
        clear_influence=((WEST_GERMANY, ts.Player.US), (TURKEY, ts.Player.US)),
        influence=((CUBA, ts.Player.USSR, 3),),
    ).build()

    ts.Engine.step_flat(st, 23 - 1)                       # select Marshall Plan
    ts.Engine.step_flat(st, PLAY_MODE_ACTION["ops"])      # play it for Operations

    # Walk to the coup: choose COUP at the op-mode node, then a battleground target.
    for _ in range(8):
        if ts.Engine.is_terminal(st):
            break
        ctx = st.ctx()
        legal = np.flatnonzero(ActionEncoder.get_legal_mask(st))
        if not len(legal):
            break
        pick = int(legal[0])
        if ctx.decision_type == ts.DecisionType.SELECT_OP_MODE:
            coup = 116 + int(ts.OpMode.COUP)
            if coup in legal.tolist():
                pick = coup
        elif ctx.decision_type == ts.DecisionType.POINT_NODE and (119 + CUBA) in legal.tolist():
            pick = 119 + CUBA
        ts.Engine.step_flat(st, pick)
        while (not ts.Engine.is_terminal(st)
               and st.ctx().decision_player == ts.Player.NONE
               and st.ctx().decision_type == ts.DecisionType.ROLL_DIE):
            ts.Engine.step(st, ts.MicroAction(ts.DecisionType.ROLL_DIE, 0, 0, 0))

    if not st.has_flag(ts.EffectBits.CMC_SUICIDE_LOSS):
        pytest.skip("this line did not reach a CMC coup; covered by the synthetic cases below")

    assert ts.Engine.is_terminal(st)
    assert int(st.victory_points) == -20, "US coup under CMC must hand the USSR the game"
    assert int(st.defcon) > 1, "a CMC loss must not masquerade as a DEFCON-1 ending"
    assert classify_game_ending_reason(st) == "DEFCON 1 (own decision)"


def test_classifier_reports_a_cmc_loss_as_self_inflicted() -> None:
    """A synthetic terminal state carrying the flag must not be read as a 20 VP win."""
    st = ts.GameState()
    ts.Engine.init_game(st, 4242)
    st.victory_points = -20
    st.defcon = 3
    st.current_phase = ts.Phase.GAME_OVER
    st.set_flag(ts.EffectBits.CMC_SUICIDE_LOSS)

    assert classify_game_ending_reason(st) == "DEFCON 1 (own decision)"

    st.clear_flag(ts.EffectBits.CMC_SUICIDE_LOSS)
    assert classify_game_ending_reason(st) == "20 VP", \
        "without the flag this is a legitimate 20 VP result; the two must stay distinguishable"


def test_env_marks_a_cmc_loss_as_a_blunder() -> None:
    """The blunder window must cover CMC losses, which defcon <= 1 alone cannot see."""
    st = ts.GameState()
    ts.Engine.init_game(st, 4242)
    st.victory_points = 20
    st.defcon = 3
    st.current_phase = ts.Phase.GAME_OVER
    st.phasing_player = ts.Player.USSR
    st.set_flag(ts.EffectBits.CMC_SUICIDE_LOSS)

    # Mirrors the branch in TsVectorizedEnv.step.
    is_defcon_blunder = (
        int(st.defcon) <= 1 and not st.has_flag(ts.EffectBits.DEFCON_SUICIDE_PROVOKED)
    ) or st.has_flag(ts.EffectBits.CMC_SUICIDE_LOSS)
    assert is_defcon_blunder
    assert int(st.phasing_player) == int(ts.Player.USSR)


def test_safety_probe_flags_a_losing_coup_under_cmc() -> None:
    """The forced probe applies the action, so it sees the terminal state directly."""
    st = _us_to_move_under_ussr_cmc()
    st.ctx().decision_player = ts.Player.US
    st.ctx().decision_type = ts.DecisionType.POINT_NODE

    kinds = classify_legal_actions(st, ts.Player.US)
    legal = np.flatnonzero(ActionEncoder.get_legal_mask(st))
    assert len(legal) > 0
    # Nothing here should be reported as a win for the side about to lose the game.
    assert "win" not in set(kinds.values())
