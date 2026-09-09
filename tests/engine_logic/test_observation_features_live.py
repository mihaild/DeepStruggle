"""Every observation feature the engine claims to expose actually varies.

A field nobody writes is invisible: the observation is memset to zero first, so it reads as a
legitimate 0.0 for the life of the project. That is what the 512-float action history was, and
what three of the four turn-aggregate families still are.

This is a characterization test. It does not assert the dead set is empty -- it is not -- it
asserts the dead set is *exactly* the one that has been diagnosed, so a newly-added feature that
nobody wires up fails here rather than being discovered a year later by an ablation.
"""

from typing import Set

import pytest

from ai.eval.feature_audit import audit, collect, v21_features

#: Diagnosed dead, with the reason. Shrink this list by wiring the writer, never by adding to it.
KNOWN_DEAD: Set[str] = {
    # observation.cpp writes global_features[0..71] of 76.
    "global/UNWRITTEN_TAIL",
    # Read by observation.cpp, written by nothing in engine/src. The engine does maintain
    # realignments_by_region (ops.cpp:412), which the observation in turn never reads.
    "turn_agg/ops_spent_by_region_mine",
    "turn_agg/ops_spent_by_region_opp",
    "turn_agg/headlines_played",
    "turn_agg/space_attempts",
    # observation.cpp writes turn_aggregates[0..27] of 32.
    "turn_agg/UNWRITTEN_TAIL",
}

#: Features that need a broad state sample before they vary; excluded from the live check because
#: 40 games of uniform-random play is not guaranteed to reach them. They are covered by the
#: full-size audit, not here.
_SAMPLE_SENSITIVE: Set[str] = {"board", "cards"}


@pytest.fixture(scope="module")
def rows():
    obs = collect(num_games=40, seed=99)
    assert obs.shape[0] > 3000, f"only {obs.shape[0]} positions; the sample is too thin to judge"
    return audit(obs, v21_features())


def test_the_dead_set_is_exactly_what_has_been_diagnosed(rows) -> None:
    dead = {str(r["name"]) for r in rows if r["dead"]} - _SAMPLE_SENSITIVE
    assert dead == KNOWN_DEAD, (
        f"observation dead-feature set changed.\n"
        f"  newly dead (nobody writes these): {sorted(dead - KNOWN_DEAD)}\n"
        f"  revived (update KNOWN_DEAD):      {sorted(KNOWN_DEAD - dead)}")


def test_the_model_is_told_the_turn_and_the_action_round(rows) -> None:
    """Both are single floats in the global block and both must move across a game."""
    by_name = {str(r["name"]): r for r in rows}
    turn = by_name["global/turn"]
    ar = by_name["global/action_round"]
    assert not turn["dead"] and float(turn["max"]) > float(turn["min"])
    assert not ar["dead"] and float(ar["max"]) > float(ar["min"])
    # turn is normalised by 10. How far the sample actually gets is a property of the sample --
    # 40 games of uniform-random play reach turn 9, not 10 -- so this checks the feature spans a
    # real range rather than pinning a ceiling the sample size decides.
    assert float(turn["max"]) >= 0.8, "the sample never got past turn 8; widen it before judging"
    assert float(turn["min"]) <= 0.2


def test_the_phase_and_stack_depth_reach_the_model(rows) -> None:
    """Which decision is being asked, not just where on the board it lands."""
    by_name = {str(r["name"]): r for r in rows}
    for name in ("global/phase", "global/ctx_stack_depth", "global/i_am_phasing",
                 "active_player"):
        assert not by_name[name]["dead"], f"{name} never varies"


def test_the_side_and_hand_sizes_reach_the_model(rows) -> None:
    by_name = {str(r["name"]): r for r in rows}
    for name in ("global/i_am_us", "global/my_hand_count", "global/opp_hand_count",
                 "global/draw_pile_count", "global/discard_pile_count"):
        assert not by_name[name]["dead"], f"{name} never varies"


def test_coups_by_region_is_the_one_live_turn_aggregate(rows) -> None:
    by_name = {str(r["name"]): r for r in rows}
    assert not by_name["turn_agg/coups_by_region_mine"]["dead"]
    assert not by_name["turn_agg/coups_by_region_opp"]["dead"]
