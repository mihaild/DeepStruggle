"""bindings/belief.hpp's tracker on hand-built records: the record leaves out a decision when a card the
opponent held then is in the decider's hand now (the count test alone passes when that matches a deck
discard), a record that loses decisions keeps their evidence in the temperature weights, a job not yet
weighed deals no stale worlds, the bindings refuse what would crash, and the Python view of a box-4
headline is the tracker's."""
from __future__ import annotations

from typing import Tuple

import numpy as np
import pytest

import ts_engine as ts
from ai.search.belief import Decision, Now, card_locations, left_since, record
from tests.bindings.test_rollout_batch import _box4_headline

US_UNKNOWN = ts.CardLocation.HAND_US_UNKNOWN
USSR_UNKNOWN = ts.CardLocation.HAND_USSR_UNKNOWN


def _us_to_move(seed: int = 4) -> ts.GameState:
    """An action-round position with the US to decide and a hidden USSR hand of at least two cards."""
    runner = ts.VectorizedBatchRunner(1, seed)
    rng = np.random.default_rng(seed)
    for _ in range(2000):
        st = runner.get_state(0)
        if runner.get_terminals()[0]:
            seed += 1000
            runner = ts.VectorizedBatchRunner(1, seed)
            continue
        if (st.current_phase == ts.Phase.ACTION_ROUND and st.ctx().decision_player == ts.Player.US
                and int(st.turn) >= 2 and sum(st.get_card_location(c) == USSR_UNKNOWN for c in range(1, 111)) >= 2
                and not np.isin(card_locations(st), [int(ts.CardLocation.PEEKED_TEMP),
                                                     int(ts.CardLocation.HEADLINE_COMMITTED)]).any()):
            return st
        m = np.asarray(runner.get_action_masks())[0]
        runner.step_flat_all([int(rng.choice(np.flatnonzero(m)))], True)
    raise AssertionError("no such position")


def _cards(st: ts.GameState, loc: ts.CardLocation) -> list:
    return [c for c in range(1, 111) if st.get_card_location(c) == loc]


def _move(st: ts.GameState, card: int, loc: ts.CardLocation) -> ts.GameState:
    out = st.clone()
    out.set_card_location(card, loc)
    return out


def _gift_and_deck_discard(st: ts.GameState) -> Tuple[ts.GameState, int, int]:
    """`st` after a USSR card went to the US's hand (a Missile Envy gift) and a deck card to the discard
    (an Our Man in Tehran discard): the USSR's hand shrank by one and one card left the pool."""
    gift = _cards(st, USSR_UNKNOWN)[0]
    tehran = _cards(st, ts.CardLocation.DRAW_DECK)[0]
    now = _move(_move(st, gift, ts.CardLocation.HAND_US_KNOWN), tehran, ts.CardLocation.DISCARD_PILE)
    return now, gift, tehran


def test_a_gift_matched_by_a_deck_discard_takes_the_decision_off_the_record() -> None:
    st = _us_to_move()
    now_state, gift, tehran = _gift_and_deck_discard(st)
    tr = ts.BeliefTracker(16, True, 1)
    tr.observe(0, st, ts.Player.USSR, 0)
    assert tr.prepare([0], [now_state]) == [True]
    # Before the fix: one decision, with the Tehran discard dealt into the USSR's hand at d and the gift not.
    assert tr.job_count() == 0
    d = Decision(st, 2, 0, card_locations(st))
    assert left_since(d, Now.of(now_state, 1)) is None
    # Without the gift the same discard fails the count test, as it always did; with nothing moved it passes.
    assert left_since(d, Now.of(_move(st, tehran, ts.CardLocation.DISCARD_PILE), 1)) is None
    left = left_since(d, Now.of(st, 1))
    assert left is not None and not left.any()


def test_a_played_card_stays_on_the_record() -> None:
    st = _us_to_move()
    played = _cards(st, USSR_UNKNOWN)[0]
    now_state = _move(st, played, ts.CardLocation.DISCARD_PILE)
    d = Decision(st, 2, 0, card_locations(st))
    left = left_since(d, Now.of(now_state, 1))
    assert left is not None and np.flatnonzero(left).tolist() == [played - 1]
    tr = ts.BeliefTracker(16, True, 1)
    tr.observe(0, st, ts.Player.USSR, 0)
    tr.prepare([0], [now_state])
    assert tr.job_count() == 1 and tr.job_decisions(0) == 1
    assert np.flatnonzero(tr.rows_patch(0)[2][0]).tolist() == [played - 1]


def _set(tr: ts.BeliefTracker, favour_first: bool) -> None:
    R = sum(tr.job_decisions(j) for j in range(tr.job_count()))
    lp = np.zeros((len(ts.BeliefTracker.temperatures()), R, tr.particles), dtype=np.float32)
    if favour_first:
        lp[1:] = -8.0
    tr.set_log_likelihoods(lp)


def test_evidence_of_decisions_dropped_from_the_record_stays_in_the_weights() -> None:
    """A card entering the pool (a reshuffle) drops the turn's earlier decisions from the record. Their
    evidence must stay in the temperature weights, not be overwritten by the new record's."""
    st = _us_to_move()
    tr = ts.BeliefTracker(16, True, 2)
    tr.observe(0, st, ts.Player.USSR, 0)
    tr.prepare([0], [st])
    assert tr.job_count() == 1
    _set(tr, favour_first=True)
    w = np.asarray(tr.temperature_weights(0))
    assert w[0] > 0.99
    # A card of the US's hand goes back to the deck: the first decision drops, a second one is recorded.
    back = _cards(st, US_UNKNOWN)[0] if _cards(st, US_UNKNOWN) else _cards(st, ts.CardLocation.HAND_US_KNOWN)[0]
    after = _move(st, back, ts.CardLocation.DRAW_DECK)
    tr.observe(0, after, ts.Player.USSR, 1)
    assert tr.prepare([0], [after]) == [True]
    assert tr.job_count() == 1 and tr.job_decisions(0) == 1
    assert tr.rows_patch(0)[4].tolist() == [1]
    _set(tr, favour_first=False)          # the new decision says nothing about the temperature
    w = np.asarray(tr.temperature_weights(0))
    assert w[0] > 0.99, w                 # before the fix: uniform, the first decision's evidence lost


def test_a_job_not_yet_weighed_deals_no_stale_worlds() -> None:
    st = _us_to_move()
    tr = ts.BeliefTracker(16, True, 3)
    tr.observe(0, st, ts.Player.USSR, 0)
    tr.prepare([0], [st])
    _set(tr, favour_first=False)
    assert tr.worlds(0, 4) is not None
    played = _cards(st, USSR_UNKNOWN)[0]
    later = _move(st, played, ts.CardLocation.DISCARD_PILE)
    tr.observe(0, later, ts.Player.USSR, 2)
    tr.prepare([0], [later])
    assert tr.job_count() == 1
    assert tr.worlds(0, 4) is None        # the earlier position's hands, until set_log_likelihoods
    _set(tr, favour_first=False)
    hands = tr.worlds(0, 4)
    assert hands is not None and not hands[:, played - 1].any()


def test_the_bindings_refuse_what_would_crash() -> None:
    with pytest.raises(ValueError):
        ts.BeliefTracker(0, True, 0)
    st = _us_to_move()
    tr = ts.BeliefTracker(4, True, 0)
    with pytest.raises((TypeError, ValueError)):
        tr.prepare([0], [None])           # type: ignore[list-item]
    tr.observe(0, st, ts.Player.USSR, 0)
    tr.prepare([0], [st])
    with pytest.raises(ValueError):
        tr.rows_patch(1 << 30)
    lp = np.full((len(ts.BeliefTracker.temperatures()), 1, 4), np.nan, dtype=np.float32)
    with pytest.raises(ValueError):
        tr.set_log_likelihoods(lp)


def test_the_python_view_of_a_box4_headline_is_the_trackers() -> None:
    """Space Race box 4 shows the US the USSR's headline: the tracker with `reveal` conditions on the USSR's
    choice, and so must the probes' Python record (Now.of with reveal). The record's decision is a headline
    choice at this same position (its filter reads the phase, the stage and the decision type)."""
    st = _box4_headline(5)
    assert ts.revealed_headline(st, ts.Player.US) != 0
    log = [Decision(st, 2, 0, card_locations(st))]
    for reveal in (False, True):
        tr = ts.BeliefTracker(8, True, 1, reveal)
        tr.observe(0, st, ts.Player.USSR, 0)
        assert tr.prepare([0], [st]) == [True]
        want = tr.job_decisions(0) if tr.job_count() else 0
        assert want == (1 if reveal else 0)
        assert len(record(log, Now.of(st, 1, reveal), True)) == want
