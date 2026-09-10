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

from ai.eval.feature_audit import audit, collect, v22_features

#: Diagnosed dead, with the reason. Empty for v2.2, and it should stay that way: every float in
#: the observation is now written by something and varies on real positions. A name appearing here
#: means a feature was added and never wired up.
KNOWN_DEAD: Set[str] = set()

#: Features that need a broad state sample before they vary; excluded from the live check because
#: 40 games of uniform-random play is not guaranteed to reach them. They are covered by the
#: full-size audit, not here.
#:
#: chernobyl_region is the clear case: Chernobyl is a mid-war card and random play ends around
#: turn 3, so the sample cannot reach it. Reported dead here it would be a false alarm -- the
#: 150-game audit run at layout v2.2 finds it live. This is the same sampling trap that made an
#: earlier redundancy pass report nine dead effect bits that were merely unreachable.
_SAMPLE_SENSITIVE: Set[str] = {"board", "cards", "ctx/chernobyl_region"}


@pytest.fixture(scope="module")
def rows():
    # 120, not 40. How deep a uniform-random sample gets is a property of the sample and of the
    # decision stream, not of the observation, so it moves whenever the engine legitimately
    # changes. Starred cards now reach the discard pile instead of leaving the game, which puts
    # more cards -- scoring cards among them -- back in circulation, and random play loses to a
    # held scoring card sooner: 40 games stopped reaching turn 8 and the turn-span check below
    # failed. Widening is what that check's own message asks for, and is the honest fix; lowering
    # the bar would weaken the assertion to match the sample rather than the other way round.
    obs = collect(num_games=120, seed=99)
    assert obs.shape[0] > 3000, f"only {obs.shape[0]} positions; the sample is too thin to judge"
    return audit(obs, v22_features())


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
    for name in ("global/phase", "global/ctx_stack_depth", "global/i_am_phasing"):
        assert not by_name[name]["dead"], f"{name} never varies"


def test_the_side_and_hand_sizes_reach_the_model(rows) -> None:
    by_name = {str(r["name"]): r for r in rows}
    for name in ("global/i_am_us", "global/my_hand_count", "global/opp_hand_count",
                 "global/draw_pile_count", "global/discard_pile_count"):
        assert not by_name[name]["dead"], f"{name} never varies"


def test_the_decision_context_reaches_the_model(rows) -> None:
    """v2.2's whole point: the network is told what it is being asked, not just the board.

    Before this the observation carried one field of DecisionContext (node_counts, as board
    feature 25) plus ctx_stack_depth, so mid-play it was asked to place a point without being
    told which card it was spending or how many points remained.
    """
    by_name = {str(r["name"]): r for r in rows}
    for name in ("ctx/decision_type", "ctx/op_mode", "ctx/remaining_steps",
                 "ctx/pending_ops_value", "ctx/allow_early_stop", "ctx/timing_ops_first",
                 "ctx/timing_event_first"):
        assert name in by_name, f"{name} is missing from the observation"
        assert not by_name[name]["dead"], f"{name} never varies"


def test_no_turn_history_block_remains(rows) -> None:
    """Dropped rather than completed: three of its four families had no writer, and the network
    never sliced any of them. A partial turn history is worse than none."""
    assert not [r for r in rows if str(r["name"]).startswith("turn_agg/")]
