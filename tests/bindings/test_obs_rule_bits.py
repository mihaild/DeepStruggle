"""ENG-9 as a rule bit: obs_features::OPP_EVENT_GATED, played per seat.

A network trained before ENG-9 learned the rule where an opponent's card offered its Event even
when the Event could not fire (it fizzled into an Ops play), and chooses its Ops differently without
it. So the rule a seat plays is the rule bits of the feature set its model is observed in: the
runner seats them from its per-side sets, `legal_mask` seats the decider it builds a mask for, and a
seat nobody seated plays feature set 0's rules -- the rule from before ENG-9 -- as the runner's
default and every model from before the bit do.
"""

import numpy as np
import ts_engine as ts

from bindings.ts_env import legal_mask

NATO = 21
EVENT = 110              # flat slot of Resolution::EVENT
GATED = ts.OBS_FEATURE_OPP_EVENT_GATED
REVEAL = ts.OBS_FEATURE_HEADLINE_REVEAL


def _nato_in_ussr_hands() -> ts.GameState:
    """The USSR choosing how to play NATO, neither Warsaw Pact Formed nor the Marshall Plan played."""
    s = ts.GameState()
    ts.Engine.init_game(s, 42)
    s.set_card_location(NATO, ts.hand_of(ts.Player.USSR))
    s.current_phase = ts.Phase.ACTION_ROUND
    s.action_round = 1
    s.phasing_player = ts.Player.USSR
    s.ctx().decision_player = ts.Player.USSR
    s.ctx().decision_type = ts.DecisionType.SELECT_PLAY_MODE
    s.ctx().pending_op_card = NATO
    return s


def test_the_bit_is_a_rule_bit_of_no_width() -> None:
    assert GATED & ts.OBS_FEATURES_RULES
    assert GATED & ts.OBS_FEATURES_ALL
    assert ts.obs_size_for(GATED) == ts.obs_size_for(0)


def test_an_unseated_state_plays_the_pre_fix_rule() -> None:
    s = _nato_in_ussr_hands()
    assert not s.seat_plays_rule(ts.Player.US, GATED) and not s.seat_plays_rule(ts.Player.USSR, GATED)
    assert legal_mask(s, False, None)[EVENT] == 1


def test_legal_mask_seats_the_decider_in_its_models_rules() -> None:
    s = _nato_in_ussr_hands()
    assert legal_mask(s, False, REVEAL | GATED)[EVENT] == 0  # a network trained with the bit
    assert s.seat_plays_rule(ts.Player.USSR, GATED)
    assert not s.seat_plays_rule(ts.Player.US, GATED)        # only the decider was seated
    assert legal_mask(s, False, None)[EVENT] == 0            # None leaves the seat as it is
    assert legal_mask(s, False, REVEAL)[EVENT] == 1          # an E8 network without it
    # The step follows the seat: what the mask withholds, the engine refuses.
    legal_mask(s, False, GATED)
    assert not ts.Engine.try_step_flat(s, EVENT)


def test_the_runner_seats_each_side_from_its_feature_set() -> None:
    r = ts.VectorizedBatchRunner(2, 7)
    # Feature set 0 is every network from before the first rule bit: the pre-fix rule.
    assert not r.get_state(0).seat_plays_rule(ts.Player.US, GATED)
    r.set_obs_features([GATED, 0], [0, REVEAL | GATED])
    assert r.get_state(0).seat_plays_rule(ts.Player.US, GATED)
    assert not r.get_state(0).seat_plays_rule(ts.Player.USSR, GATED)
    assert not r.get_state(1).seat_plays_rule(ts.Player.US, GATED)
    assert r.get_state(1).seat_plays_rule(ts.Player.USSR, GATED)
    # A reset and a written position keep the env's seats, whoever seated the position.
    r.reset_game(0, 99)
    assert r.get_state(0).seat_plays_rule(ts.Player.US, GATED)
    assert not r.get_state(0).seat_plays_rule(ts.Player.USSR, GATED)
    r.set_state(1, _nato_in_ussr_hands())
    assert r.get_state(1).seat_plays_rule(ts.Player.USSR, GATED)


def test_the_runners_mask_is_its_deciders_rule() -> None:
    r = ts.VectorizedBatchRunner(2, 7)
    r.set_obs_features([0, 0], [0, GATED])      # env 0's USSR legacy, env 1's current
    r.set_state(0, _nato_in_ussr_hands())
    r.set_state(1, _nato_in_ussr_hands())
    r.refresh_all()
    masks = np.asarray(r.get_action_masks()).reshape(2, -1)
    assert masks[0, EVENT] == 1
    assert masks[1, EVENT] == 0
