"""The marginal-value target must be something a copied input cannot produce.

The observation already carries the live net VP differential per region, so "can the trunk
recover regional score" is a question about copying six floats. The counterfactual -- what would
this region be worth if I placed one influence *here* -- is not in the observation and is not a
smooth function of it: regional scoring is a stack of thresholds, so the answer is zero almost
everywhere and several VP at the point that crosses one.

These tests pin that property. If the target ever became dense or smooth, the probe would stop
distinguishing a network that computed the rules from one that reads the number.
"""
from __future__ import annotations

import numpy as np
import pytest

import ts_engine as ts
from ai.eval.marginal_scoring import marginal_placement_value
from ai.eval.positions import PositionBuilder

EUROPE = [c for c in range(84)
          if str(ts.MapData.get_country_info(c)["region"]) == str(ts.Region.EUROPE)]


def _state(**kw):
    return PositionBuilder(hand=(1, 2, 3), side=ts.Player.USSR, turn=5, defcon=3, **kw).build()


def test_the_target_is_zero_almost_everywhere() -> None:
    """A dense target would mean the probe is measuring something smooth, not the thresholds."""
    v = marginal_placement_value(_state(), ts.Player.USSR)
    assert v.shape == (84,)
    nonzero = float((np.abs(v) > 1e-9).mean())
    assert nonzero < 0.5, f"{nonzero:.0%} of placements change the score; expected a sparse target"


def test_some_placement_somewhere_is_worth_something() -> None:
    """The mirror check: an all-zero target would make the probe vacuous."""
    v = marginal_placement_value(_state(), ts.Player.USSR)
    assert float(np.abs(v).max()) > 0.0


def test_the_influence_that_takes_control_is_worth_more_than_the_one_after_it() -> None:
    """The threshold, stated directly. Poland has stability 3, so from 2 USSR influence in an
    empty country the third takes control and the fourth does nothing."""
    poland = next(c for c in range(84)
                  if str(ts.MapData.get_country_info(c)["name"]) == "Poland")
    at_two = _state(influence=((poland, ts.Player.USSR, 2),))
    at_three = _state(influence=((poland, ts.Player.USSR, 3),))

    taking = marginal_placement_value(at_two, ts.Player.USSR)[poland]
    piling_on = marginal_placement_value(at_three, ts.Player.USSR)[poland]
    assert taking > piling_on, (
        f"taking control scored {taking} and adding beyond it {piling_on}; "
        "the target is not capturing the control threshold")


def test_the_answer_key_does_not_disturb_the_state_it_probes() -> None:
    """Every counterfactual runs on a clone. If it did not, probing would corrupt the rollout."""
    st = _state(influence=((EUROPE[0], ts.Player.USSR, 2),))
    before = [(st.get_country(c).us_influence, st.get_country(c).ussr_influence)
              for c in range(84)]
    marginal_placement_value(st, ts.Player.USSR)
    after = [(st.get_country(c).us_influence, st.get_country(c).ussr_influence)
             for c in range(84)]
    assert before == after


@pytest.mark.parametrize("player", [ts.Player.US, ts.Player.USSR])
def test_the_target_is_signed_from_the_acting_players_side(player: ts.Player) -> None:
    """Placing your own influence can never reduce your own region differential."""
    v = marginal_placement_value(_state(), player)
    assert float(v.min()) >= 0.0, \
        "adding one's own influence scored negative; the perspective sign is wrong"
