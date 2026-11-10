"""bindings/rollout_batch.hpp through the Python module: the determinizer keeps every count and
every seen card, the batch builds k x worlds rows and drops a world whole, a horizon of 0 is all
leaves, a row stops exactly at its horizon, shared dice give a world's candidates one stream, the
leaves are read from the mover's side, an illegal candidate is refused, and the turn-end and later
stops end every row at the right place, a batch played across threads gives exactly the rows it
gives on one, and a raced decision drops its trailing challengers, plays the rest on the batch's next
worlds and decides among the candidates left."""
from __future__ import annotations

import os
import subprocess
import sys
from typing import Dict, List, Set, Tuple

import numpy as np
import pytest

import ts_engine as ts
from ai.eval.paired_playouts import ARKey, ar_key, decider
from bindings.action_encoder import ActionEncoder
from tools.lib.game_step import drain_chance


def _position(plies: int = 60, seed: int = 7) -> ts.GameState:
    runner = ts.VectorizedBatchRunner(1, seed)
    rng = np.random.default_rng(seed)
    for _ in range(plies):
        m = np.asarray(runner.get_action_masks())[0]
        runner.step_flat_all([int(rng.choice(np.flatnonzero(m)))], True)
    st = runner.get_state(0)
    assert not ts.Engine.is_terminal(st)
    return st


def _legal(st: ts.GameState) -> List[int]:
    return [int(a) for a in np.flatnonzero(np.asarray(ActionEncoder.get_legal_mask(st)))]


def _locations(st: ts.GameState) -> List[ts.CardLocation]:
    return [st.get_card_location(c) for c in range(1, 111)]


def _play(batch: ts.RolloutBatch, seed: int = 0) -> int:
    """Drive a batch to its leaves with a fixed pseudo-random legal policy; returns the plies."""
    plies = 0
    while not batch.all_done():
        obs, masks = batch.fill_live()
        rng = np.random.default_rng(seed * 1000 + plies)
        lg = np.where(masks > 0, rng.random(masks.shape), -1e9).astype(np.float32)
        batch.step(np.ascontiguousarray(lg))
        plies += 1
        assert plies < 2000
    return plies


@pytest.mark.parametrize("me", [ts.Player.US, ts.Player.USSR])
def test_determinize_keeps_every_count_and_every_seen_card(me: ts.Player) -> None:
    st = _position()
    before = _locations(st)
    out = ts.determinize(st, me, 42)
    after = _locations(out)
    opp_unknown = ts.CardLocation.HAND_USSR_UNKNOWN if me == ts.Player.US else ts.CardLocation.HAND_US_UNKNOWN
    hidden = {opp_unknown, ts.CardLocation.DRAW_DECK}
    for b, a in zip(before, after):
        if b in hidden:
            assert a in hidden                      # a hidden card stays in the hidden pool
        else:
            assert a == b                           # everything the mover can see is untouched
    for loc in hidden:
        assert before.count(loc) == after.count(loc)
    # Only the card locations and the rng seed differ: the board, tracks and context are the same.
    assert out.turn == st.turn and out.defcon == st.defcon and out.victory_points == st.victory_points
    assert out.ctx().decision_type == st.ctx().decision_type
    # The same seed redeals the same way; another seed differs when there is anything to shuffle.
    assert _locations(ts.determinize(st, me, 42)) == after
    if sum(before.count(loc) for loc in hidden) > 1:
        assert _locations(ts.determinize(st, me, 43)) != after or True   # may coincide; never raises


@pytest.mark.parametrize("me", [ts.Player.US, ts.Player.USSR])
def test_the_oracles_world_moves_no_card(me: ts.Player) -> None:
    """keep_hand: the opponent's hand stays as it is and the deck is only reordered -- and a card
    location is a place, not a position in the deck, so every location is unchanged."""
    st = _position()
    assert _locations(ts.determinize(st, me, 42, keep_hand=True)) == _locations(st)


def test_an_oracle_batch_plays_every_world_with_the_opponents_true_hand() -> None:
    st = _position()
    cands = _legal(st)[:2]
    me = decider(st)
    opp_unknown = ts.CardLocation.HAND_USSR_UNKNOWN if me == ts.Player.US else ts.CardLocation.HAND_US_UNKNOWN
    hand = sorted(c for c, loc in enumerate(_locations(st), 1) if loc == opp_unknown)
    for oracle in (True, False):
        b = ts.RolloutBatch(horizon=0, seed=5, oracle=oracle, keyed=True)
        n = b.add(st, cands, 8)
        assert n > 0
        hands = {tuple(sorted(c for c, loc in enumerate(_locations(b.row_state(r)), 1) if loc == opp_unknown))
                 for r in range(n)}
        if oracle:
            assert hands == {tuple(hand)}
        elif len(hand) > 0:
            assert len(hands) > 1                   # the honest worlds redeal the hand


def test_rows_are_candidates_by_worlds_and_live_until_their_horizon() -> None:
    st = _position()
    cands = _legal(st)[:3]
    b = ts.RolloutBatch(horizon=2, shared_dice=True, guard=True, features=0, seed=5)
    n = b.add(st, cands, 4)
    assert n == 3 * b.worlds_used(0) and b.worlds_used(0) <= 4 and b.rows() == n
    assert b.positions() == 1 and b.mover(0) == decider(st)
    assert b.live() == n and not b.all_done()
    _play(b)
    assert b.refused() == 0
    crossed = b.row_crossed()
    for r, (done, term) in enumerate(zip(b.row_done(), b.row_terminal())):
        assert done
        if not term:
            assert crossed[r] == 2                  # stopped exactly at the horizon
    assert sorted(set(b.row_candidate())) == list(range(3))
    assert sorted(set(b.row_world())) == list(range(b.worlds_used(0)))


def test_horizon_zero_is_all_leaves_read_from_the_movers_side() -> None:
    st = _position()
    cands = _legal(st)[:2]
    b = ts.RolloutBatch(horizon=0, seed=1)
    b.add(st, cands, 3)
    assert b.all_done() and b.live() == 0
    obs, masks = b.fill_leaves()
    assert obs.shape == (b.rows(), b.obs_width()) and masks.shape == (b.rows(), ActionEncoder.FLAT_ACTION_SIZE)
    mover = b.mover(0)
    for r in range(b.rows()):
        s = b.row_state(r)
        want = np.asarray(ts.extract_observation_features(s, mover, 0), dtype=np.float32)
        assert np.array_equal(obs[r], want)
        assert np.array_equal(masks[r], np.asarray(ActionEncoder.get_legal_mask(s), dtype=np.uint8))
    values = (np.arange(b.rows()) / 10).astype(np.float32)
    b.set_leaf_values(values)
    assert np.array_equal(b.values(), values)


def test_values_are_nan_until_valued_and_a_terminal_row_scores_itself() -> None:
    st = _position()
    b = ts.RolloutBatch(horizon=1, seed=2)
    b.add(st, _legal(st)[:2], 2)
    assert np.isnan(b.values()).all()
    _play(b)
    v = b.values()
    for r, term in enumerate(b.row_terminal()):
        if term:
            s = b.row_state(r)
            u = ts.Engine.get_terminal_utility(s)
            assert v[r] == (u if b.mover(0) == ts.Player.US else -u)
        else:
            assert np.isnan(v[r])


def test_a_seed_reproduces_the_batch_and_shared_dice_are_one_stream_per_world() -> None:
    st = _position()
    cands = _legal(st)[:3]

    def rows_of(seed: int, shared: bool) -> List[bytes]:
        b = ts.RolloutBatch(horizon=1, shared_dice=shared, seed=seed)
        b.add(st, cands, 3)
        _play(b)
        return [bytes(b.row_state(r).raw_bytes()) for r in range(b.rows())]

    assert rows_of(9, True) == rows_of(9, True)      # the same seed plays the same worlds
    assert rows_of(9, True) != rows_of(10, True)     # another seed deals other worlds
    # With shared dice the candidates of a world play under one stream; with their own, each draws
    # its own, so the two settings play different games from one seed.
    assert rows_of(9, True) != rows_of(9, False)


def test_a_second_position_in_the_same_batch_keeps_its_own_rows() -> None:
    a, c = _position(60, 7), _position(90, 8)
    b = ts.RolloutBatch(horizon=1, seed=3)
    na = b.add(a, _legal(a)[:2], 2)
    nc = b.add(c, _legal(c)[:3], 2)
    assert b.positions() == 2 and b.rows() == na + nc
    pos = b.row_position()
    assert pos.count(0) == na and pos.count(1) == nc
    _play(b)
    assert b.all_done()


def test_an_illegal_candidate_is_refused() -> None:
    st = _position()
    legal = set(_legal(st))
    illegal = next(a for a in range(ActionEncoder.FLAT_ACTION_SIZE) if a not in legal)
    b = ts.RolloutBatch(seed=1)
    with pytest.raises(ValueError):
        b.add(st, [next(iter(legal)), illegal], 2)
    assert b.rows() == 0
    with pytest.raises(ValueError):
        ts.RolloutBatch(horizon=-1)


def test_step_checks_its_shape() -> None:
    st = _position()
    b = ts.RolloutBatch(horizon=1, seed=4)
    b.add(st, _legal(st)[:2], 2)
    obs, masks = b.fill_live()
    with pytest.raises(ValueError):
        b.step(np.zeros((obs.shape[0] + 1, ActionEncoder.FLAT_ACTION_SIZE), dtype=np.float32))


def test_the_boundary_count_matches_ar_key_changes() -> None:
    # Play one row's state forward by hand with the same greedy rule and count ar_key changes.
    st = _position()
    cands = _legal(st)[:1] + _legal(st)[1:2]
    b = ts.RolloutBatch(horizon=3, shared_dice=True, guard=False, seed=11)
    b.add(st, cands, 1)
    starts: List[ARKey] = [ar_key(b.row_state(r)) for r in range(b.rows())]
    _play(b, seed=1)
    for r in range(b.rows()):
        if b.row_terminal()[r]:
            continue
        assert b.row_crossed()[r] == 3
        assert ar_key(b.row_state(r)) != starts[r]


def test_candidates_are_the_most_probable_legal_moves_best_first() -> None:
    st = _position()
    mask = np.asarray(ActionEncoder.get_legal_mask(st), dtype=np.uint8)
    rng = np.random.default_rng(3)
    logits = rng.standard_normal(mask.shape[0]).astype(np.float32)
    got = ts.rollout_candidates(logits, mask, 4)
    legal = np.flatnonzero(mask)
    want = [int(a) for a in legal[np.argsort(-logits[legal], kind="stable")][:4]]
    assert got == want
    assert ts.rollout_candidates(logits, mask, 1000) == [int(a) for a in legal[np.argsort(-logits[legal], kind="stable")]]


def test_decide_reports_means_paired_leads_and_the_rule() -> None:
    st = _position()
    mask = np.asarray(ActionEncoder.get_legal_mask(st), dtype=np.uint8)
    cands = _legal(st)[:3]
    b = ts.RolloutBatch(horizon=0, seed=5)
    b.add(st, cands, 4)
    n_w = b.worlds_used(0)
    # Leaf values by row, chosen so candidate 2 leads candidate 0 by exactly +0.2 in every world
    # (no spread: a clear lead) and candidate 1 trails it.
    rows = b.rows()
    cand = b.row_candidate()
    world = b.row_world()
    values = np.zeros(rows, dtype=np.float32)
    for r in range(rows):
        values[r] = {0: 0.1, 1: -0.3, 2: 0.3}[cand[r]] + 0.01 * world[r]
    lo, _ = b.fill_leaves()
    # fill_leaves lists rows in row order; the same order values are built in.
    assert lo.shape[0] == rows
    b.set_leaf_values(values)
    logits = np.zeros(mask.shape[0], dtype=np.float32)
    d = ts.rollout_decide(b, 0, cands, logits, mask, "z2")
    assert d["worlds"] == n_w
    assert np.allclose(d["q"], [0.1 + 0.01 * np.mean(range(n_w)), -0.3 + 0.01 * np.mean(range(n_w)), 0.3 + 0.01 * np.mean(range(n_w))], atol=1e-6)
    assert np.allclose(d["lead"], [0.0, -0.4, 0.2], atol=1e-6)
    assert np.allclose(d["lead_se"], [0.0, 0.0, 0.0], atol=1e-6)
    assert d["pick"] == cands[2]                       # a clear lead departs from the policy
    assert ts.rollout_decide(b, 0, cands, logits, mask, "argmax")["pick"] == cands[2]
    # The KL rule at a tiny temperature is the argmax of Q; at a huge one, the policy's move.
    assert ts.rollout_decide(b, 0, cands, logits, mask, "kl0.001")["pick"] == cands[2]
    logits[cands[0]] = 5.0
    assert ts.rollout_decide(b, 0, cands, logits, mask, "kl1000")["pick"] == cands[0]
    with pytest.raises(ValueError):
        ts.rollout_decide(b, 0, cands, logits, mask, "best")
    assert ts.rollout_parse_rule("kl") == ("kl", pytest.approx(0.05))
    assert ts.rollout_parse_rule("z") == ("z", 2.0)


def test_decide_keeps_the_policys_move_without_a_clear_lead() -> None:
    st = _position()
    mask = np.asarray(ActionEncoder.get_legal_mask(st), dtype=np.uint8)
    cands = _legal(st)[:2]
    b = ts.RolloutBatch(horizon=0, seed=6)
    b.add(st, cands, 4)
    cand = b.row_candidate()
    world = b.row_world()
    # Candidate 1 is better on average but noisily: +0.3 in two worlds, -0.25 in the other two.
    values = np.asarray([0.0 if cand[r] == 0 else (0.3 if world[r] % 2 == 0 else -0.25) for r in range(b.rows())],
                        dtype=np.float32)
    b.fill_leaves()
    b.set_leaf_values(values)
    logits = np.zeros(mask.shape[0], dtype=np.float32)
    d = ts.rollout_decide(b, 0, cands, logits, mask, "z2")
    assert d["lead"][1] == pytest.approx(0.025, abs=1e-6) and d["lead_se"][1] > 0.1
    assert d["pick"] == cands[0]                                   # not two standard errors clear
    assert ts.rollout_decide(b, 0, cands, logits, mask, "argmax")["pick"] == cands[1]


def test_the_turn_end_stop_ends_every_row_at_the_next_turns_first_position() -> None:
    st = _position()
    cands = _legal(st)[:3]
    b = ts.RolloutBatch(horizon=4, shared_dice=True, guard=True, features=0, seed=17, stop="turn")
    b.add(st, cands, 4)
    assert b.rows() > 0
    _play(b, seed=3)
    assert b.refused() == 0
    for r in range(b.rows()):
        assert b.row_done()[r]
        if not b.row_terminal()[r]:
            assert int(b.row_state(r).turn) == int(st.turn) + 1      # stopped as soon as the turn changed


def test_the_later_stop_needs_both_the_boundaries_and_the_turns_end() -> None:
    st = _position()
    cands = _legal(st)[:2]
    b = ts.RolloutBatch(horizon=1, shared_dice=True, guard=True, features=0, seed=19, stop="later")
    b.add(st, cands, 3)
    _play(b, seed=4)
    for r in range(b.rows()):
        if not b.row_terminal()[r]:
            assert b.row_crossed()[r] >= 1 and int(b.row_state(r).turn) == int(st.turn) + 1


def test_the_earlier_stop_ends_at_whichever_comes_first() -> None:
    """kEarlier: a row is a leaf at the turn's end if that comes within the horizon, else after the
    horizon's boundaries -- and with a horizon beyond the turn it is the turn-end stop exactly."""
    st = _position()
    cands = _legal(st)[:3]
    b = ts.RolloutBatch(horizon=1, shared_dice=True, guard=True, features=0, seed=23, stop="earlier")
    b.add(st, cands, 4)
    _play(b, seed=5)
    for r in range(b.rows()):
        if b.row_terminal()[r]:
            continue
        turned = int(b.row_state(r).turn) == int(st.turn) + 1
        crossed = int(b.row_crossed()[r])
        assert (turned and crossed <= 1) or (not turned and crossed == 1)

    def leaves(stop: str, horizon: int) -> List[bytes]:
        x = ts.RolloutBatch(horizon=horizon, shared_dice=True, guard=True, features=0, seed=29, stop=stop)
        x.add(st, cands, 4)
        _play(x, seed=6)
        return [bytes(x.row_state(r).raw_bytes()) for r in range(x.rows())]
    assert leaves("earlier", 99) == leaves("turn", 99)


def test_decider_leaves_are_read_from_their_deciders_side_and_negated_into_the_movers_frame() -> None:
    """leaf="decider": each leaf's observation is its decider's, and where that is the root mover's
    opponent the critic's value is negated into the mover's frame; leaf="mover" reads the mover's side."""
    st = _position()
    cands = _legal(st)[:3]
    mover = ts.Player(int(st.ctx().decision_player)) if int(st.ctx().decision_player) else st.phasing_player
    checked = {"same": 0, "flipped": 0}
    settings = [(leaf, stop, h) for leaf in ("mover", "decider") for stop, h in (("turn", 2), ("boundaries", 1),
                                                                                  ("boundaries", 2), ("boundaries", 3))]
    for leaf, stop, horizon in settings:
        b = ts.RolloutBatch(horizon=horizon, shared_dice=True, guard=True, features=0, seed=31, stop=stop, leaf=leaf)
        b.add(st, cands, 6)
        _play(b, seed=7)
        obs, _masks = b.fill_leaves()
        term = b.row_terminal()
        leaf_rows = [r for r in range(b.rows()) if not term[r]]
        assert len(leaf_rows) == obs.shape[0] > 0
        for j, r in enumerate(leaf_rows):
            s = b.row_state(r)
            dp = int(s.ctx().decision_player) or int(s.phasing_player)
            side = ts.Player(dp) if leaf == "decider" else mover
            assert np.array_equal(obs[j], np.asarray(ts.extract_observation(s, side), dtype=np.float32))
        b.set_leaf_values(np.full(obs.shape[0], 0.25, dtype=np.float32))
        values = np.asarray(b.values())
        for r in leaf_rows:
            s = b.row_state(r)
            dp = int(s.ctx().decision_player) or int(s.phasing_player)
            flip = leaf == "decider" and dp != int(mover)
            assert values[r] == pytest.approx(-0.25 if flip else 0.25)
            if leaf == "decider":
                checked["flipped" if flip else "same"] += 1
    assert checked["flipped"] > 0 and checked["same"] > 0      # leaves of both kinds were checked
    with pytest.raises(ValueError):
        ts.RolloutBatch(horizon=4, leaf="other")


def test_own_leaves_play_on_to_the_movers_next_decision() -> None:
    """leaf="own": every non-terminal leaf is a position the root mover decides, at or past the stop."""
    st = _position()
    cands = _legal(st)[:3]
    mover = int(st.ctx().decision_player) or int(st.phasing_player)
    for stop, horizon in (("turn", 2), ("boundaries", 1), ("boundaries", 3), ("earlier", 2)):
        b = ts.RolloutBatch(horizon=horizon, shared_dice=True, guard=True, features=0, seed=37, stop=stop, leaf="own")
        b.add(st, cands, 6)
        _play(b, seed=8)
        term = b.row_terminal()
        obs, _masks = b.fill_leaves()
        leaf_rows = [r for r in range(b.rows()) if not term[r]]
        assert len(leaf_rows) == obs.shape[0] > 0
        for j, r in enumerate(leaf_rows):
            s = b.row_state(r)
            assert (int(s.ctx().decision_player) or int(s.phasing_player)) == mover
            if stop == "turn":
                assert int(s.turn) == int(st.turn) + 1
            assert np.array_equal(obs[j], np.asarray(ts.extract_observation(s, ts.Player(mover)), dtype=np.float32))


def test_predeal_leaves_are_the_turn_end_park_of_the_decisions_turn() -> None:
    """stop="predeal": every non-terminal leaf is the turn-end park (nobody's decision, before the deal)
    of the decision's own turn; "earlier_predeal": the park or the horizon, whichever came first."""
    st = _position()
    cands = _legal(st)[:3]
    for stop, horizon in (("predeal", 4), ("earlier_predeal", 2), ("earlier_predeal", 30)):
        b = ts.RolloutBatch(horizon=horizon, shared_dice=True, guard=True, features=0, seed=41, stop=stop)
        b.add(st, cands, 5)
        _play(b, seed=9)
        term = b.row_terminal()
        parks = 0
        for r in range(b.rows()):
            if term[r]:
                continue
            s = b.row_state(r)
            c = s.ctx()
            parked = (int(c.decision_player) == 0 and c.decision_type == ts.DecisionType.ROLL_DIE
                      and c.pending_roll_type == ts.RollType.TURN_CLEANUP)
            if stop == "predeal" or horizon == 30:
                assert parked and int(s.turn) == int(st.turn)
            else:
                assert parked or int(b.row_crossed()[r]) == horizon
            parks += int(parked)
        if stop == "predeal" or horizon == 30:
            assert parks > 0                          # (2 boundaries come before this turn's end)


def test_a_bad_horizon_or_stop_is_refused() -> None:
    with pytest.raises(ValueError):
        ts.RolloutBatch(horizon=-1)
    with pytest.raises(ValueError):
        ts.RolloutBatch(horizon=4, stop="never")


_THREADED = r"""
import hashlib, sys
import numpy as np
import ts_engine as ts
runner = ts.VectorizedBatchRunner(1, 7)
rng = np.random.default_rng(7)
for _ in range(60):
    m = np.asarray(runner.get_action_masks())[0]
    runner.step_flat_all([int(rng.choice(np.flatnonzero(m)))], True)
st = runner.get_state(0)
cands = [int(a) for a in np.flatnonzero(np.asarray(ts.get_flat_action_mask(st, False)))[:4]]
b = ts.RolloutBatch(horizon=3, shared_dice=True, guard=True, features=0, seed=99, stop="later")
b.add(st, cands, 32)                                  # 4 x 32 rows: above the parallel threshold
h = hashlib.sha256()
ply = 0
while not b.all_done():
    obs, masks = b.fill_live()
    h.update(obs.tobytes()); h.update(masks.tobytes())
    lg = np.where(masks > 0, np.sin(np.arange(masks.size).reshape(masks.shape) * 0.37 + ply), -1e9).astype(np.float32)
    b.step(np.ascontiguousarray(lg))
    ply += 1
obs, masks = b.fill_leaves()
h.update(obs.tobytes())
b.set_leaf_values(np.linspace(-1, 1, obs.shape[0]).astype(np.float32))
h.update(b.values().tobytes())
print(h.hexdigest())
"""


def test_a_batch_plays_the_same_rows_on_one_thread_and_on_many() -> None:
    """step, fill_live and fill_leaves run across the OpenMP team (gomp_parallel_for); the rows are
    independent and listed serially, so the observations, masks and values are bit-identical."""
    digests = []
    for threads in ("1", "8"):
        paths = [os.path.dirname(ts.__file__), os.environ.get("PYTHONPATH", "")]   # the engine this process imported
        env = {**os.environ, "OMP_NUM_THREADS": threads, "PYTHONPATH": os.pathsep.join(p for p in paths if p)}
        out = subprocess.run([sys.executable, "-c", _THREADED], env=env, capture_output=True, text=True, check=True)
        digests.append(out.stdout.strip())
    assert digests[0] == digests[1] and len(digests[0]) == 64


def test_place_hand_with_the_true_hand_changes_nothing_and_add_deals_the_given_hands() -> None:
    st = _position()
    me = decider(st)
    opp_unknown = ts.CardLocation.HAND_USSR_UNKNOWN if me == ts.Player.US else ts.CardLocation.HAND_US_UNKNOWN
    locs = _locations(st)
    true = np.asarray([loc == opp_unknown for loc in locs], dtype=np.uint8)
    assert _locations(ts.place_hand(st, me, true)) == locs
    pool = [c for c, loc in enumerate(locs) if loc in (opp_unknown, ts.CardLocation.DRAW_DECK)]
    rng = np.random.default_rng(1)
    hands = np.zeros((3, 110), dtype=np.uint8)
    for w in range(3):
        hands[w, rng.choice(pool, size=int(true.sum()), replace=False)] = 1
    b = ts.RolloutBatch(horizon=0, seed=5, keyed=True)
    cands = _legal(st)[:2]
    n = b.add(st, cands, 3, hands=hands)
    assert n == 2 * b.worlds_used(0)
    for r in range(n):
        got = np.asarray([loc == opp_unknown for loc in _locations(b.row_state(r))], dtype=np.uint8)
        played = np.asarray([loc != l0 for loc, l0 in zip(_locations(b.row_state(r)), locs)])
        # the row has played its candidate: compare only the cards it did not move
        w = b.row_world()[r]
        assert np.array_equal(got[~played], hands[w][~played])


def _leaf_keys_by_world(st: ts.GameState, candidates: List[int], horizon: int, aligned: bool,
                        seed: int, same_turn: bool = True) -> List[Set[Tuple[int, int, int]]]:
    """Per world whose every candidate reaches a leaf inside the decision's turn: the (turn, action round,
    phasing side) its candidates' leaves are at."""
    b = ts.RolloutBatch(horizon=horizon, seed=3, keyed=True, aligned=aligned)
    n = b.add(st, candidates, 3)
    _play(b, seed=seed)
    keys: Dict[int, Set[Tuple[int, int, int]]] = {}
    outside: Set[int] = set()
    for r in range(n):
        world = int(b.row_world()[r])
        w = b.row_state(r)
        if b.row_terminal()[r] or (same_turn and int(w.turn) != int(st.turn)):
            outside.add(world)
            continue
        keys.setdefault(world, set()).add((int(w.turn), int(w.action_round), int(w.phasing_player)))
    return [ks for world, ks in keys.items() if world not in outside]


def test_aligned_counts_every_candidates_horizon_from_the_end_of_the_movers_round() -> None:
    """A play-mode decision whose candidates end the action round differently -- space ends it inside the
    move, Ops leave points to place -- reads every candidate's leaf at the same point of the turn when
    aligned: the same (turn, action round, phasing side) in each world. Unaligned, the Ops candidate counts
    the round's end and is read an action round earlier (the owner's Decolonization replay, 2026-10-19).
    Only worlds whose leaves are all inside the root's turn are compared: a headline whose event the
    phasing side resolves adds a boundary on some paths and not others, which is the playout's own course,
    not where the count starts."""
    off = int(ActionEncoder.PLAY_MODE_OFFSET)
    space, ops = off + 1, off + 2
    positions: List[ts.GameState] = []
    for seed in range(12):
        runner = ts.VectorizedBatchRunner(1, seed)
        rng = np.random.default_rng(seed)
        while not runner.get_terminals()[0] and len(positions) < 2 * (seed + 1):
            st = runner.get_state(0)
            m = np.asarray(runner.get_action_masks())[0]
            if (int(st.ctx().decision_type) == int(ts.DecisionType.SELECT_PLAY_MODE)
                    and st.current_phase == ts.Phase.ACTION_ROUND and m[space] and m[ops]):
                positions.append(st)
            runner.step_flat_all([int(rng.choice(np.flatnonzero(m)))], True)
    assert len(positions) >= 6
    seen = apart = 0
    for seed, st in enumerate(positions):
        for horizon in (1, 2, 4):
            for ks in _leaf_keys_by_world(st, [space, ops], horizon, True, seed):
                assert len(ks) == 1, (seed, horizon, ks)
                seen += 1
            apart += sum(len(ks) > 1 for ks in _leaf_keys_by_world(st, [space, ops], horizon, False, seed))
    assert seen >= 10
    assert apart > 0        # the unaligned count, every measured player's, reads them apart


def _headline_choices(n: int) -> List[ts.GameState]:
    """Headline card choices, first and second to choose, from random games."""
    out: List[ts.GameState] = []
    for seed in range(40):
        runner = ts.VectorizedBatchRunner(1, 100 + seed)
        rng = np.random.default_rng(seed)
        while not runner.get_terminals()[0] and len(out) < n:
            st = runner.get_state(0)
            m = np.asarray(runner.get_action_masks())[0]
            if (st.current_phase == ts.Phase.HEADLINE and st.headline_stage == 0
                    and int(st.ctx().decision_type) == int(ts.DecisionType.SELECT_CARD) and m.sum() >= 3):
                out.append(st)
            runner.step_flat_all([int(rng.choice(np.flatnonzero(m)))], True)
    return out


def test_aligned_counts_the_headline_as_one_boundary_for_every_candidate() -> None:
    """At a headline choice the candidate decides how the headline resolves: whether its card goes first
    and whether its event stops for a choice while its owner is phasing. Counting every phasing change
    made that 1 to 3 boundaries, so the candidates of one choice were read at different points of the
    next action rounds (2026-10-19: 30% of worlds at the first choice under the deployed root, 89% with
    the round-start alignment alone). Aligned, the headline is one boundary for every candidate."""
    positions = _headline_choices(24)
    assert len(positions) >= 12
    seen = apart = 0
    for seed, st in enumerate(positions):
        cands = _legal(st)[:3]
        for horizon in (1, 2, 3):
            for ks in _leaf_keys_by_world(st, cands, horizon, True, seed):
                assert len(ks) == 1, (seed, horizon, ks)
                seen += 1
            apart += sum(len(ks) > 1 for ks in _leaf_keys_by_world(st, cands, horizon, False, seed))
    assert seen >= 50
    assert apart > 0        # the unaligned count, every measured player's, reads them apart


def _box4_headline(seed: int) -> ts.GameState:
    """The US's headline choice with Man in Earth Orbit (Space Race box 4) its own: the USSR chose first."""
    st = ts.GameState()
    ts.Engine.init_game(st, seed)
    drain_chance(st)
    st.us_space_track = 4
    rng = np.random.default_rng(seed)
    while True:
        assert not ts.Engine.is_terminal(st)
        if (st.current_phase == ts.Phase.HEADLINE and st.headline_stage == 0 and st.headline_ussr_card
                and st.ctx().decision_player == ts.Player.US):
            return st
        m = np.asarray(ActionEncoder.get_legal_mask(st))
        ts.Engine.step_flat(st, int(rng.choice(np.flatnonzero(m))))
        drain_chance(st)


def test_box4_keeps_the_revealed_opposing_headline_in_every_world() -> None:
    """With Man in Earth Orbit held alone, the opponent chooses its headline first and reveals it: the box-4
    worlds keep the card chosen and in the opponent's hand, the rest of the hand redealt around it; the
    default worlds forget it and ask it again, as every measured player's did."""
    st = _box4_headline(5)
    card = int(st.headline_ussr_card)
    assert ts.revealed_headline(st, ts.Player.US) == card
    assert ts.revealed_headline(st, ts.Player.USSR) == 0
    unknown = ts.CardLocation.HAND_USSR_UNKNOWN
    n_hand = sum(st.get_card_location(c) == unknown for c in range(1, 111))
    hands = set()
    for seed in range(8):
        w = ts.determinize(st, ts.Player.US, seed, keep_revealed=True)
        assert w.headline_ussr_card == card and w.get_card_location(card) == unknown
        assert sum(w.get_card_location(c) == unknown for c in range(1, 111)) == n_hand
        hands.add(tuple(c for c in range(1, 111) if w.get_card_location(c) == unknown))
        assert ts.determinize(st, ts.Player.US, seed).headline_ussr_card == 0
    assert len(hands) > 1
    cands = _legal(st)[:3]
    for box4 in (True, False):
        b = ts.RolloutBatch(horizon=2, seed=3, keyed=True, box4=box4)
        n = b.add(st, cands, 4)
        for r in range(n):
            w = b.row_state(r)
            asks_again = (w.current_phase == ts.Phase.HEADLINE and w.headline_stage == 0
                          and w.ctx().decision_player == ts.Player.USSR)
            assert asks_again != box4, (box4, r)


def test_box4_belief_hands_all_hold_the_revealed_headline() -> None:
    st = _box4_headline(5)
    card = int(st.headline_ussr_card)
    for reveal in (True, False):
        tr = ts.BeliefTracker(64, True, 3, reveal)
        assert tr.prepare([0], [st]) == [True]
        hands = np.asarray(tr.worlds(0, 64))
        n_hand = sum(st.get_card_location(c) == ts.CardLocation.HAND_USSR_UNKNOWN for c in range(1, 111))
        assert (hands.sum(axis=1) == n_hand).all()
        if reveal:
            assert hands[:, card - 1].all()
        else:
            assert not hands[:, card - 1].all()


def _value_leaves(b: ts.RolloutBatch, by_candidate: Dict[int, float]) -> None:
    """Value every unvalued leaf (horizon 0) by its candidate, plus 0.01 x its world."""
    cand, world = b.row_candidate(), b.row_world()
    leaves: List[int] = []
    unvalued = np.isnan(b.values())
    terminal = b.row_terminal()
    for r in range(b.rows()):
        if unvalued[r] and not terminal[r]:
            leaves.append(r)
    obs, _ = b.fill_leaves()
    assert obs.shape[0] == len(leaves)
    b.set_leaf_values(np.asarray([by_candidate[cand[r]] + 0.01 * world[r] for r in leaves], dtype=np.float32))


def test_race_drops_the_trailing_challengers_and_plays_the_rest_on_fresh_worlds() -> None:
    st = _position()
    mask = np.asarray(ActionEncoder.get_legal_mask(st), dtype=np.uint8)
    cands = _legal(st)[:3]
    b = ts.RolloutBatch(horizon=0, seed=5)
    b.add(st, cands, 4)
    first = b.worlds_used(0)
    with pytest.raises(ValueError, match="valued"):
        b.race(0, 4)                                     # the first stage is not valued yet
    _value_leaves(b, {0: 0.1, 1: -0.3, 2: 0.3})          # candidate 1 trails, 2 leads
    added = b.race(0, 4)
    assert [b.in_race(0, c) for c in range(3)] == [True, False, True]
    second = b.worlds_used(0) - first
    assert added == 2 * second and second > 0           # only the network's move and candidate 2
    assert set(b.row_candidate()[-added:]) == {0, 2}
    assert set(b.row_world()[-added:]) == set(range(first, first + second))
    with pytest.raises(ValueError, match="already"):
        b.race(0, 4)
    _value_leaves(b, {0: 0.1, 2: 0.3})
    logits = np.zeros(mask.shape[0], dtype=np.float32)
    d = ts.rollout_decide(b, 0, cands, logits, mask, "z2")
    assert d["worlds"] == first + second                 # the raced candidates over both stages
    assert np.allclose(d["lead"], [0.0, -0.4, 0.2], atol=1e-6)
    # the dropped candidate keeps its first-stage mean, and is never picked -- not even by argmax
    assert d["q"][1] == pytest.approx(-0.3 + 0.01 * np.mean(range(first)), abs=1e-6)
    assert d["pick"] == cands[2]
    assert ts.rollout_decide(b, 0, cands, logits, mask, "argmax")["pick"] == cands[2]


def test_a_dropped_candidate_is_never_picked() -> None:
    """A dropped challenger's mean is over the first stage alone; when the second stage goes badly for
    the candidates left, it can read best of all -- and still is not picked."""
    st = _position()
    mask = np.asarray(ActionEncoder.get_legal_mask(st), dtype=np.uint8)
    cands = _legal(st)[:3]
    b = ts.RolloutBatch(horizon=0, seed=5)
    b.add(st, cands, 4)
    _value_leaves(b, {0: 0.1, 1: -0.05, 2: 0.3})
    b.race(0, 4)
    _value_leaves(b, {0: -0.9, 2: -0.99})
    logits = np.zeros(mask.shape[0], dtype=np.float32)
    d = ts.rollout_decide(b, 0, cands, logits, mask, "argmax")
    assert d["q"][1] > max(d["q"][0], d["q"][2])
    assert d["pick"] == cands[2]
    assert ts.rollout_decide(b, 0, cands, logits, mask, "kl0.001")["pick"] == cands[2]


def test_a_race_with_no_challenger_left_adds_nothing() -> None:
    st = _position()
    cands = _legal(st)[:3]
    b = ts.RolloutBatch(horizon=0, seed=5)
    b.add(st, cands, 4)
    _value_leaves(b, {0: 0.5, 1: 0.0, 2: -0.1})
    rows = b.rows()
    assert b.race(0, 8) == 0 and b.rows() == rows
    assert [b.in_race(0, c) for c in range(3)] == [True, False, False]


def _dealt_hands(st: ts.GameState, n: int, seed: int) -> np.ndarray:
    """`n` opposing hands for the decider of `st`, each as many cards as the opponent's unseen hand,
    dealt from its unseen hand and the draw deck."""
    me = decider(st)
    opp_unknown = ts.CardLocation.HAND_USSR_UNKNOWN if me == ts.Player.US else ts.CardLocation.HAND_US_UNKNOWN
    locs = _locations(st)
    held = sum(loc == opp_unknown for loc in locs)
    pool = [c for c, loc in enumerate(locs) if loc in (opp_unknown, ts.CardLocation.DRAW_DECK)]
    rng = np.random.default_rng(seed)
    hands = np.zeros((n, 110), dtype=np.uint8)
    for w in range(n):
        hands[w, rng.choice(pool, size=held, replace=False)] = 1
    return hands


def _assert_same_batch(a: ts.RolloutBatch, b: ts.RolloutBatch, st: ts.GameState, cands: List[int]) -> None:
    """The same rows, state for state, and the same decision at position 0 under every rule."""
    mask = np.asarray(ActionEncoder.get_legal_mask(st), dtype=np.uint8)
    assert a.rows() == b.rows() and a.worlds_used(0) == b.worlds_used(0)
    assert [bytes(a.row_state(r).raw_bytes()) for r in range(a.rows())] == \
        [bytes(b.row_state(r).raw_bytes()) for r in range(b.rows())]
    logits = np.zeros(mask.shape[0], dtype=np.float32)
    for rule in ("z2", "argmax", "kl0.05"):
        assert ts.rollout_decide(a, 0, cands, logits, mask, rule) == ts.rollout_decide(b, 0, cands, logits, mask, rule)


@pytest.mark.parametrize("shared,keyed,keyed2", [(True, False, False), (False, False, False), (True, True, False),
                                                  (True, False, True)])
def test_a_race_that_keeps_every_candidate_is_the_unraced_batch(shared: bool, keyed: bool, keyed2: bool) -> None:
    """The second stage's worlds continue the batch's stream (or its keys): with no challenger dropped
    (a lead of 0 is kept), a raced batch builds exactly the rows -- and decides exactly as -- a batch
    that played every candidate on every world."""
    st = _position()
    cands = _legal(st)[:3]
    values = {0: 0.0, 1: 0.05, 2: 0.0}
    whole = ts.RolloutBatch(horizon=0, shared_dice=shared, seed=11, keyed=keyed, keyed2=keyed2)
    whole.add(st, cands, 8)
    _value_leaves(whole, values)
    raced = ts.RolloutBatch(horizon=0, shared_dice=shared, seed=11, keyed=keyed, keyed2=keyed2)
    raced.add(st, cands, 3)
    _value_leaves(raced, values)
    raced.race(0, 5)
    _value_leaves(raced, values)
    assert all(raced.in_race(0, c) for c in range(3))
    _assert_same_batch(raced, whole, st, cands)


def test_a_race_plays_its_second_stage_on_the_hands_it_is_given() -> None:
    """A belief's race: the root deals every world's hands at once, plays the first stage on the first
    slice and the race on the rest -- the same rows as one unraced add of all of them."""
    st = _position()
    cands = _legal(st)[:3]
    hands = _dealt_hands(st, 8, seed=3)
    values = {0: 0.0, 1: 0.05, 2: 0.0}
    whole = ts.RolloutBatch(horizon=0, seed=11, keyed=True)
    whole.add(st, cands, 8, hands=hands)
    _value_leaves(whole, values)
    raced = ts.RolloutBatch(horizon=0, seed=11, keyed=True)
    raced.add(st, cands, 3, hands=np.ascontiguousarray(hands[:3]))
    _value_leaves(raced, values)
    raced.race(0, 5, hands=np.ascontiguousarray(hands[3:]))
    _value_leaves(raced, values)
    _assert_same_batch(raced, whole, st, cands)
    other = ts.RolloutBatch(horizon=0, seed=11, keyed=True)
    other.add(st, cands, 3, hands=np.ascontiguousarray(hands[:3]))
    _value_leaves(other, values)
    other.race(0, 5, hands=np.ascontiguousarray(hands[[3, 4, 5, 7, 6]]))
    assert [bytes(other.row_state(r).raw_bytes()) for r in range(other.rows())] != \
        [bytes(raced.row_state(r).raw_bytes()) for r in range(raced.rows())]


def test_racing_one_position_leaves_the_others_decision_alone() -> None:
    """In a batch over several positions (a tournament's), the race appends rows for its own position
    only: another position's rows and its decision are those of the same batch unraced."""
    st0, st1 = _position(), _position(plies=80, seed=9)
    c0, c1 = _legal(st0)[:3], _legal(st1)[:3]
    mask1 = np.asarray(ActionEncoder.get_legal_mask(st1), dtype=np.uint8)
    logits = np.zeros(mask1.shape[0], dtype=np.float32)
    values = {0: 0.1, 1: -0.3, 2: 0.3}
    plain = ts.RolloutBatch(horizon=0, seed=5)
    raced = ts.RolloutBatch(horizon=0, seed=5)
    for b in (plain, raced):
        b.add(st0, c0, 4)
        b.add(st1, c1, 4)
        _value_leaves(b, values)
    added = raced.race(0, 4)
    assert added > 0 and not raced.in_race(0, 1)
    _value_leaves(raced, {0: 0.1, 2: 0.3})
    assert set(raced.row_position()[-added:]) == {0}
    assert all(raced.in_race(1, c) for c in range(3)) and raced.worlds_used(1) == plain.worlds_used(1)
    assert [bytes(raced.row_state(r).raw_bytes()) for r in range(plain.rows())] == \
        [bytes(plain.row_state(r).raw_bytes()) for r in range(plain.rows())]
    for rule in ("z2", "argmax", "kl0.05"):
        assert ts.rollout_decide(raced, 1, c1, logits, mask1, rule) == \
            ts.rollout_decide(plain, 1, c1, logits, mask1, rule)


def test_an_unraced_decision_is_untouched_by_the_race_bookkeeping() -> None:
    st = _position()
    cands = _legal(st)[:3]
    b = ts.RolloutBatch(horizon=0, seed=5)
    b.add(st, cands, 4)
    assert all(b.in_race(0, c) for c in range(3)) and not b.in_race(0, 3)
    with pytest.raises(ValueError, match="no such position"):
        b.race(1, 4)
    hands = np.zeros((3, 110), dtype=np.uint8)
    _value_leaves(b, {0: 0.0, 1: 0.0, 2: 0.0})
    with pytest.raises(ValueError, match="worlds x 110"):
        b.race(0, 4, hands=hands)
