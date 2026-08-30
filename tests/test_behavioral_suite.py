"""Tests for the behavioural test harness.

The harness is only trustworthy if a malformed position fails loudly rather than quietly
passing, and if an assertion the policy never answered is not scored as a failure.
"""

from typing import List

import numpy as np
import pytest
import ts_engine as ts

from ai.eval.behavioral_suite import (
    BehavioralTest,
    NeverArgmax,
    Prefer,
    run_suite,
)
from ai.eval.claims import build_claims
from ai.eval.positions import (
    PLAY_MODE_ACTION,
    PositionBuilder,
    card_action,
    legal_mask,
    legal_play_modes,
    step_to_play_mode,
)

NATO = 21
MARSHALL_PLAN = 23
COMECON = 14


def _uniform_policy(state: ts.GameState, obs: np.ndarray, mask: np.ndarray) -> np.ndarray:
    legal = mask.astype(np.float64)
    return legal / legal.sum()


def _point_mass(action: int):
    def fn(state: ts.GameState, obs: np.ndarray, mask: np.ndarray) -> np.ndarray:
        p = np.zeros_like(mask, dtype=np.float64)
        p[action] = 1.0
        return p
    return fn


# --- position construction -------------------------------------------------------------

def test_builder_deals_exactly_the_requested_hand() -> None:
    st = PositionBuilder(hand=[NATO, MARSHALL_PLAN], side=ts.Player.US).build()
    selectable = [int(i) + 1 for i in np.flatnonzero(legal_mask(st)) if i < 110]
    assert sorted(selectable) == [NATO, MARSHALL_PLAN]


def test_builder_sets_influence_on_the_named_side_only() -> None:
    FRANCE = 8
    st = PositionBuilder(hand=[NATO], side=ts.Player.US,
                         influence=[(FRANCE, ts.Player.USSR, 5)],
                         clear_influence=[(FRANCE, ts.Player.US)]).build()
    c = st.get_country(FRANCE)
    assert int(c.ussr_influence) == 5
    assert int(c.us_influence) == 0


def test_step_to_play_mode_reaches_the_play_mode_node() -> None:
    st = PositionBuilder(hand=[NATO, MARSHALL_PLAN], side=ts.Player.US,
                         flags=[ts.EffectBits.WARSAW_PACT_PLAYED]).build()
    pm = step_to_play_mode(st, NATO)
    assert pm.ctx().decision_type == ts.DecisionType.SELECT_PLAY_MODE
    assert "event" in legal_play_modes(pm) and "ops" in legal_play_modes(pm)


def test_step_to_play_mode_rejects_a_card_not_in_hand() -> None:
    st = PositionBuilder(hand=[NATO], side=ts.Player.US).build()
    with pytest.raises(ValueError, match="not selectable"):
        step_to_play_mode(st, MARSHALL_PLAN)


# --- assertions -------------------------------------------------------------------------

def test_never_argmax_fails_when_the_action_is_the_top_choice() -> None:
    probs = np.zeros(212); probs[PLAY_MODE_ACTION["event"]] = 0.9; probs[PLAY_MODE_ACTION["ops"]] = 0.1
    out = NeverArgmax(PLAY_MODE_ACTION["event"], "event").check(probs)
    assert not out.passed and "IS the top choice" in out.detail


def test_never_argmax_fails_on_the_probability_ceiling_even_when_not_top() -> None:
    probs = np.zeros(212); probs[PLAY_MODE_ACTION["ops"]] = 0.6; probs[PLAY_MODE_ACTION["event"]] = 0.4
    out = NeverArgmax(PLAY_MODE_ACTION["event"], "event", max_prob=0.2).check(probs)
    assert not out.passed


def test_never_argmax_passes_when_the_action_is_rare_and_not_top() -> None:
    probs = np.zeros(212); probs[PLAY_MODE_ACTION["ops"]] = 0.95; probs[PLAY_MODE_ACTION["event"]] = 0.05
    assert NeverArgmax(PLAY_MODE_ACTION["event"], "event").check(probs).passed


def test_prefer_is_inconclusive_when_neither_option_has_mass() -> None:
    """A deterministic bot picking a third action says nothing about this ordering."""
    probs = np.zeros(212); probs[5] = 1.0
    out = Prefer(card_action(NATO), card_action(MARSHALL_PLAN), "NATO", "MP").check(probs)
    assert out.inconclusive and not out.passed


def test_prefer_orders_by_probability_mass() -> None:
    probs = np.zeros(212)
    probs[card_action(NATO)] = 0.7
    probs[card_action(MARSHALL_PLAN)] = 0.3
    assert Prefer(card_action(NATO), card_action(MARSHALL_PLAN), "NATO", "MP").check(probs).passed
    assert not Prefer(card_action(MARSHALL_PLAN), card_action(NATO), "MP", "NATO").check(probs).passed


# --- suite mechanics ---------------------------------------------------------------------

def test_inconclusive_results_are_excluded_from_the_pass_rate() -> None:
    tests: List[BehavioralTest] = [
        BehavioralTest(
            claim_id="probe-inconclusive",
            description="ordering the policy never answers",
            builder=PositionBuilder(hand=[NATO, MARSHALL_PLAN], side=ts.Player.USSR),
            assertion=Prefer(card_action(COMECON), card_action(5), "Comecon", "FYP"),
            tier="N",
        ),
    ]
    res = run_suite(_uniform_policy, tests)
    assert res.results[0].inconclusive
    assert res.pass_rate == 0.0          # nothing scored
    assert res.by_tier() == {}           # and no tier claims a rate


def test_a_malformed_position_is_reported_as_an_error_not_a_pass() -> None:
    tests = [
        BehavioralTest(
            claim_id="probe-broken",
            description="selects a card that is not in hand",
            builder=PositionBuilder(hand=[NATO], side=ts.Player.US),
            assertion=NeverArgmax(PLAY_MODE_ACTION["event"], "event"),
            play_card_first=MARSHALL_PLAN,
        ),
    ]
    res = run_suite(_uniform_policy, tests)
    assert res.results[0].error is not None
    assert not res.results[0].passed


def test_every_encoded_claim_builds_a_legal_position() -> None:
    """Guards against a claim silently rotting when engine ids or rules change."""
    for t in build_claims():
        state = t.position()
        mask = legal_mask(state)
        assert mask.sum() > 0, f"{t.claim_id}: position has no legal actions"


def test_suite_detects_a_policy_that_always_commits_the_blunder() -> None:
    blunder = PLAY_MODE_ACTION["event"]
    tests = [t for t in build_claims() if isinstance(t.assertion, NeverArgmax)
             and t.assertion.action == blunder]
    assert tests, "expected some never-play-for-event claims"
    res = run_suite(_point_mass(blunder), tests)
    assert res.pass_rate == 0.0
