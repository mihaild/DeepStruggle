"""bindings/belief.hpp's hands are consistent with every recorded decision (audit S-1, 2026-11-22).

A trap's scoring-card escape is legal only for a hand holding no card of 2+ Ops, which almost no uniform hand
of seven cards is: the tracker must still draw only such hands. Over self-play games, both sides, every
belief: the opponent's real hand is consistent with the record (the legality check never excludes the truth)
and every particle with weight is consistent (no inconsistent hand is weighed). An impossible record falls
back to the uniform prior with a diagnostic naming the position, or raises when strict. And the record's own
edges: a card back in the deck through Our Man in Tehran's reshuffle takes its decision off the record, and a
decision still on the record after a fold keeps weighing the hands but not the temperatures."""
from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np
import pytest

import ts_engine as ts
from ai.search.belief import Decision, Now, card_locations, left_since
from bindings.action_encoder import ActionEncoder
from tools.lib.game_step import drain_chance

US_UNKNOWN = ts.CardLocation.HAND_US_UNKNOWN
USSR_UNKNOWN = ts.CardLocation.HAND_USSR_UNKNOWN
DECK = ts.CardLocation.DRAW_DECK
TRANSIENT = [int(ts.CardLocation.PEEKED_TEMP), int(ts.CardLocation.HEADLINE_COMMITTED)]
TEMPS = len(ts.BeliefTracker.temperatures())


def _cards(st: ts.GameState, loc: ts.CardLocation) -> List[int]:
    return [c for c in range(1, 111) if st.get_card_location(c) == loc]


def _info(c: int) -> Tuple[int, bool]:
    i = ts.CardData.get_card_info(c)
    return int(i["ops"]), bool(i["is_scoring"])


def _mask(st: ts.GameState) -> np.ndarray:
    return np.asarray(ActionEncoder.get_legal_mask(st))


def _legal(st: ts.GameState) -> int:
    return int(np.flatnonzero(_mask(st))[0])


def _weigh(tr: ts.BeliefTracker, rng: Optional[np.random.Generator] = None) -> None:
    """A stub policy's log-likelihoods for every job: zeros, or random finite values."""
    R = sum(tr.job_decisions(j) for j in range(tr.job_count()))
    if rng is None:
        lp = np.zeros((TEMPS, R, tr.particles), dtype=np.float32)
    else:
        lp = np.log(rng.uniform(0.01, 1.0, size=(TEMPS, R, tr.particles))).astype(np.float32)
    tr.set_log_likelihoods(lp)


# ---- a trap's scoring-card escape --------------------------------------------------------------

def _trapped_scoring_play(seed: int) -> Tuple[ts.GameState, int, np.ndarray]:
    """The US at its first action round of a turn >= 4, trapped in Quagmire, holding one scoring card and seven
    1-Op cards (no Ops modifier in play): its only legal move is the scoring card. Returns the position, the
    move and the US's hand after it (110 bytes)."""
    rng = np.random.default_rng(seed)
    st = ts.GameState()
    ts.Engine.init_game(st, seed)
    drain_chance(st)
    modifiers = [ts.EffectBits.CONTAINMENT_ACTIVE, ts.EffectBits.BREZHNEV_DOCTRINE_ACTIVE,
                 ts.EffectBits.PURGE_US_ACTIVE, ts.EffectBits.PURGE_USSR_ACTIVE]
    for _ in range(200000):
        if ts.Engine.is_terminal(st) or not _mask(st).any():
            seed += 1000
            ts.Engine.init_game(st, seed)
            drain_chance(st)
            continue
        if (st.current_phase == ts.Phase.ACTION_ROUND and int(st.turn) >= 4 and int(st.action_round) == 1
                and st.ctx().decision_player == ts.Player.US and st.ctx().decision_type == ts.DecisionType.SELECT_CARD
                and int(st.ctx().resolving_card) == 0 and not _cards(st, ts.CardLocation.HAND_US_KNOWN)
                and not np.isin(card_locations(st), TRANSIENT).any() and not any(st.has_flag(f) for f in modifiers)):
            pool = _cards(st, DECK) + _cards(st, US_UNKNOWN)
            ones = [c for c in pool if _info(c) == (1, False)]
            scoring = [c for c in pool if _info(c)[1]]
            if len(ones) >= 7 and scoring:
                d = st.clone()
                for c in _cards(d, US_UNKNOWN):
                    d.set_card_location(c, DECK)
                hand = [scoring[0]] + ones[:7]
                for c in hand:
                    d.set_card_location(c, US_UNKNOWN)
                d.set_flag(ts.EffectBits.QUAGMIRE_ACTIVE)
                m = _mask(d)
                if np.flatnonzero(m).tolist() == [scoring[0] - 1]:
                    after = np.zeros(110, dtype=np.uint8)
                    after[np.asarray(ones[:7]) - 1] = 1
                    return d, scoring[0] - 1, after
        ts.Engine.step_flat(st, int(rng.choice(np.flatnonzero(_mask(st)))))
        drain_chance(st)
    raise AssertionError("no such position")


def test_a_trap_escape_with_a_scoring_card_draws_only_hands_without_two_ops() -> None:
    d, action, true_hand = _trapped_scoring_play(21)
    scoring = action + 1
    now = d.clone()
    now.set_card_location(scoring, ts.CardLocation.DISCARD_PILE)
    now.ctx().decision_player = ts.Player.USSR
    pool = (card_locations(now) == int(DECK)) | (card_locations(now) == int(US_UNKNOWN))
    # The uniform prior almost never allows the move: the sampler before 2026-11-22 weighed such hands at -1e9.
    rng = np.random.default_rng(0)
    uniform = np.zeros((2048, 110), dtype=np.uint8)
    for k in range(2048):
        uniform[k, rng.choice(np.flatnonzero(pool), 7, replace=False)] = 1
    held = uniform.copy()
    held[:, scoring - 1] = 1
    assert ts.hypothesis_masks(d, ts.Player.USSR, held)[:, action].mean() < 0.05

    tr = ts.BeliefTracker(2048, True, 7, strict=True)
    tr.observe(0, d, ts.Player.US, action)
    assert tr.prepare([0], [now]) == [True] and tr.job_count() == 1 and tr.exhausted() == []
    assert tr.consistent(0, true_hand)
    hands = np.asarray(tr.job_hands(0))
    assert np.all(hands.sum(1) == 7) and not hands[:, ~pool].any()
    for h in np.unique(hands, axis=0):
        cards = np.flatnonzero(h) + 1
        assert all(_info(int(c))[0] <= 1 or _info(int(c))[1] for c in cards), cards   # no card of 2+ Ops
        assert tr.consistent(0, h)
    w = np.exp(np.asarray(tr.job_log_weights(0)))
    assert w.sum() == pytest.approx(1.0) and np.isfinite(tr.job_log_mass(0)) and tr.job_log_mass(0) < np.log(0.05)
    _weigh(tr)
    mix = np.asarray(tr.mixture(0))
    assert mix.sum() == pytest.approx(1.0, abs=1e-5) and (mix > 0).all()
    assert tr.worlds(0, 16) is not None
    assert tr.stats()["exhausted"] == 0 and tr.stats()["sampled"] == 1


# ---- every hand, over self-play ------------------------------------------------------------------

def _true_hand(st: ts.GameState, me: ts.Player) -> np.ndarray:
    opp = USSR_UNKNOWN if me == ts.Player.US else US_UNKNOWN
    return (card_locations(st) == int(opp)).astype(np.uint8)


def _check_position(tr: ts.BeliefTracker, p: int, st: ts.GameState, me: ts.Player, every_particle: bool,
                    revealed: int) -> None:
    truth = _true_hand(st, me)
    if revealed:
        truth[revealed - 1] = 1
    assert tr.consistent(p, truth), f"the real hand fails the record at turn {st.turn} AR {st.action_round}"
    if every_particle:
        hands = np.asarray(tr.hands(p))
        mix = np.asarray(tr.mixture(p))
        for k in np.unique(np.flatnonzero(mix > 0)):
            assert tr.consistent(p, hands[k])


def _self_play(n_games: int, seed: int, particles: int, auto_advance: bool, reveal: bool) -> Tuple[int, int, int]:
    """Random self-play, every decision of both sides prepared for its decider and checked. Returns the
    positions believed, the positions with a record, and the positions the prior alone did not fill."""
    tr = ts.BeliefTracker(particles, True, seed, reveal, auto_advanced=auto_advance, strict=True)
    rng = np.random.default_rng(seed)
    states = []
    for g in range(n_games):
        st = ts.GameState()
        ts.Engine.init_game(st, seed + g)
        drain_chance(st)
        if auto_advance:
            ts.Engine.auto_advance_step(st, 128)
        states.append(st)
    n = 0
    checked = 0
    while True:
        live = [g for g, st in enumerate(states) if not ts.Engine.is_terminal(st)
                and st.ctx().decision_player in (ts.Player.US, ts.Player.USSR)]
        if not live:
            break
        ask = [g for g in live if not np.isin(card_locations(states[g]), TRANSIENT).any()]
        has = tr.prepare(ask, [states[g] for g in ask])
        if tr.job_count():
            _weigh(tr, rng)
        for p, g in enumerate(ask):
            if has[p]:
                st = states[g]
                me = st.ctx().decision_player
                revealed = int(ts.revealed_headline(st, me)) if reveal else 0
                _check_position(tr, p, st, me, n % 7 == 0, revealed)
                n += 1
        checked += 1
        for g in live:
            st = states[g]
            a = int(rng.choice(np.flatnonzero(_mask(st))))
            tr.observe(g, st, st.ctx().decision_player, a)
            ts.Engine.step_flat(st, a, auto_advance)
            drain_chance(st)
    st_ = tr.stats()
    assert st_["exhausted"] == 0
    return n, int(st_["sampled"]), int(st_["prior_short"])


def test_every_belief_over_self_play_is_consistent_with_the_record_and_holds_the_real_hand() -> None:
    n, sampled, short = _self_play(24, 101, 128, auto_advance=True, reveal=True)
    assert n > 2500 and sampled > 800


def test_the_same_holds_for_a_stream_that_asks_every_decision() -> None:
    n, sampled, _ = _self_play(12, 202, 64, auto_advance=False, reveal=False)
    assert n > 1000 and sampled > 300


# ---- an impossible record ------------------------------------------------------------------------

def _us_to_move(seed: int = 4) -> ts.GameState:
    runner = ts.VectorizedBatchRunner(1, seed)
    rng = np.random.default_rng(seed)
    for _ in range(20000):
        st = runner.get_state(0)
        if runner.get_terminals()[0]:
            seed += 1000
            runner = ts.VectorizedBatchRunner(1, seed)
            continue
        if (st.current_phase == ts.Phase.ACTION_ROUND and st.ctx().decision_player == ts.Player.US
                and int(st.turn) >= 2 and len(_cards(st, USSR_UNKNOWN)) >= 2
                and not np.isin(card_locations(st), TRANSIENT).any()):
            return st
        runner.step_flat_all([int(rng.choice(np.flatnonzero(np.asarray(runner.get_action_masks())[0])))], True)
    raise AssertionError("no such position")


def _impossible(st: ts.GameState) -> int:
    """A move no USSR hand makes legal at `st` (the US's card choice): a country slot."""
    held = np.zeros((64, 110), dtype=np.uint8)
    rng = np.random.default_rng(1)
    pool = (card_locations(st) == int(DECK)) | (card_locations(st) == int(USSR_UNKNOWN))
    n = len(_cards(st, USSR_UNKNOWN))
    for k in range(64):
        held[k, rng.choice(np.flatnonzero(pool), n, replace=False)] = 1
    never = ~ts.hypothesis_masks(st, ts.Player.US, held).any(0).astype(bool)
    return int([a for a in np.flatnonzero(never) if a >= 112][0])


def test_an_impossible_record_falls_back_to_the_prior_with_a_diagnostic_or_raises_when_strict() -> None:
    st = _us_to_move()
    bad = _impossible(st)
    tr = ts.BeliefTracker(32, True, 3)
    tr.observe(0, st, ts.Player.USSR, bad)
    assert tr.prepare([0], [st]) == [True]
    (p, why), = tr.exhausted()
    assert p == 0 and f"turn {int(st.turn)} AR {int(st.action_round)}" in why and f"action {bad}" in why
    assert tr.job_count() == 0
    mix = np.asarray(tr.mixture(0))
    assert np.allclose(mix, 1.0 / 32)                       # uniform over the prior's hands
    assert tr.worlds(0, 8) is not None                      # and the root still plays
    assert tr.stats()["exhausted"] == 1
    strict = ts.BeliefTracker(32, True, 3, strict=True)
    strict.observe(0, st, ts.Player.USSR, bad)
    with pytest.raises(RuntimeError, match="no USSR hand"):
        strict.prepare([0], [st])


def test_a_non_finite_log_likelihood_is_refused() -> None:
    st = _us_to_move()
    for bad in (np.inf, -np.inf, np.nan):
        tr = ts.BeliefTracker(8, True, 0)
        tr.observe(0, st, ts.Player.USSR, _legal(st))
        tr.prepare([0], [st])
        lp = np.zeros((TEMPS, 1, 8), dtype=np.float32)
        lp[2, 0, 3] = bad
        with pytest.raises(ValueError, match="not finite"):
            tr.set_log_likelihoods(lp)
        assert tr.worlds(0, 2) is None                      # nothing weighed


# ---- the record's edges --------------------------------------------------------------------------

def test_a_card_reshuffled_into_the_deck_takes_its_decision_off_the_record() -> None:
    """Our Man in Tehran reshuffles the discard pile into the deck: a card the opponent held at the decision,
    played since, is in the deck now, and a deck card has gone to the discard. The count test alone passes
    (one card gone from the hand, one from the deck) and would deal the discarded deck card into the hand then."""
    st = _us_to_move(6)
    played = _cards(st, USSR_UNKNOWN)[0]
    discarded = _cards(st, DECK)[0]
    now = st.clone()
    now.set_card_location(played, DECK)
    now.set_card_location(discarded, ts.CardLocation.DISCARD_PILE)
    d = Decision(st, 2, _legal(st), card_locations(st))
    assert left_since(d, Now.of(now, 1)) is None
    tr = ts.BeliefTracker(16, True, 1)
    tr.observe(0, st, ts.Player.USSR, _legal(st))
    assert tr.prepare([0], [now]) == [True] and tr.job_count() == 0


def test_a_decision_still_on_the_record_after_a_fold_weighs_the_hands_not_the_temperatures() -> None:
    """Two decisions, then the first leaves the record (a card the opponent played between them reaches the
    decider's hand: SALT Negotiations) and the second stays. The fold puts the turn's evidence into the finished
    turns' and spends both; the second must still weigh the hands."""
    st = _us_to_move(8)
    x = _cards(st, USSR_UNKNOWN)[0]
    st1 = st.clone()
    st1.set_card_location(x, ts.CardLocation.DISCARD_PILE)
    tr = ts.BeliefTracker(64, True, 2)
    tr.observe(0, st, ts.Player.USSR, _legal(st))
    tr.observe(0, st1, ts.Player.USSR, _legal(st1))
    tr.prepare([0], [st1])
    assert tr.job_count() == 1 and tr.job_decisions(0) == 2
    lp = np.zeros((TEMPS, 2, 64), dtype=np.float32)
    lp[1:] = -8.0                                             # T = 1 explains the opponent best
    tr.set_log_likelihoods(lp)
    w1 = np.asarray(tr.temperature_weights(0))
    assert w1[0] > 0.99
    now = st1.clone()
    now.set_card_location(x, ts.CardLocation.HAND_US_KNOWN)
    assert tr.prepare([0], [now]) == [True]
    # Before the fix: no job, the second decision dropped from the hand posterior with the first.
    assert tr.job_count() == 1 and tr.job_decisions(0) == 1
    assert tr.rows_patch(0)[4].tolist() == [_legal(st1)]
    lp = np.full((TEMPS, 1, 64), -30.0, dtype=np.float32)
    lp[:, 0, 5] = 0.0                                         # the decision picks hand 5 out at every temperature
    lp[:5, 0, 5] = -20.0                                      # and would put the temperatures' weight on T = 8
    tr.set_log_likelihoods(lp)
    hands = np.asarray(tr.hands(0))
    mix = np.asarray(tr.mixture(0))
    same = np.all(hands == hands[5], axis=1)
    assert mix[same].sum() > 0.99                             # it weighs the hands
    assert np.allclose(np.asarray(tr.temperature_weights(0)), w1, atol=1e-6)   # not the temperatures again
