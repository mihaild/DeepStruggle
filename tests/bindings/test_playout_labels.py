"""bindings/playout_labels.hpp through the Python module: every world the labeller plays is the
game a step-by-step Python playout of the same world plays under ai/eval/paired_playouts.play_safe's
rule, column for column; the sibling slice's candidates share a world's deal and dice and its leaf
is the mover's next decision; a position's labels do not depend on the batch around it; and raw
state bytes round-trip."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pytest

import ts_engine as ts
from ai.eval.paired_playouts import ar_key, decider, loses_now, play_safe
from bindings.action_encoder import ActionEncoder

_W = np.random.default_rng(5).standard_normal((ts.OBS_SIZE, ActionEncoder.FLAT_ACTION_SIZE))


def _logits(obs: np.ndarray) -> np.ndarray:
    # float64 then rounded: a row's logits do not move with the batch it is computed in.
    return (np.asarray(obs, dtype=np.float64) @ _W).astype(np.float32)


def _act(obs: np.ndarray, masks: np.ndarray) -> np.ndarray:
    lg = np.where(masks > 0, _logits(obs), -np.inf)
    return lg.argmax(axis=1)


def _positions(n: int, seed: int = 3) -> List[ts.GameState]:
    """Decision states from games of a pseudo-random legal policy, a few per game."""
    out: List[ts.GameState] = []
    rng = np.random.default_rng(seed)
    game = 0
    while len(out) < n:
        runner = ts.VectorizedBatchRunner(1, seed * 100 + game)
        game += 1
        stop_at = sorted(rng.choice(np.arange(20, 900), size=3, replace=False).tolist())
        ply = 0
        while stop_at and not runner.get_terminals()[0]:
            if ply == stop_at[0]:
                stop_at.pop(0)
                st = runner.get_state(0)
                if st.turn < 10:
                    out.append(st)
            m = np.asarray(runner.get_action_masks())[0]
            runner.step_flat_all([int(rng.choice(np.flatnonzero(m)))], True)
            ply += 1
    return out[:n]


def _drive(pl: ts.PlayoutLabels) -> None:
    obs = np.zeros((pl.capacity(), pl.obs_width()), dtype=np.float32)
    steps = 0
    while not pl.idle():
        n = pl.fill_live(obs)
        pl.step(np.ascontiguousarray(_logits(obs[:n])))
        steps += 1
        assert steps < 20000


@dataclass(frozen=True)
class _Played:
    """What the labeller records of one world, VP from the mover's side."""
    score: float          #: 1 win, 0.5 draw, 0 loss
    vp: int               #: at the end of the game
    turn_end_vp: int      #: when the base's turn ended (or the game, if sooner)
    turn_end_defcon: int
    final: ts.GameState


def _reference(base: ts.GameState, world: ts.GameState,
               keyed: bool = False, keyed2: bool = False, keyedsym: bool = False) -> _Played:
    """One world played the way play_safe plays it, recording what the labeller records. With `keyed`
    every engine step is keyed first (ts.keyed_key, the world's rng_state its key), as keyed=True does
    (`keyed2`: keyed2's packing; `keyedsym`: keyedsym's)."""
    mover = decider(base)
    key = ar_key(base)
    st = world.clone()
    world_key = int(world.rng_state)
    guarding = True
    turn: Optional[Tuple[int, int]] = None
    while not ts.Engine.is_terminal(st):
        if keyed:
            st = ts.keyed_key(st, world_key, keyed2, keyedsym)
        mask = np.asarray(ActionEncoder.get_legal_mask(st)).copy()
        if guarding:
            if ar_key(st) != key or decider(st) != mover:
                guarding = False
            elif int(st.defcon) <= 2:
                legal = np.flatnonzero(mask)
                safe = [a for a in legal if not loses_now(st, int(a), mover)]
                if safe and len(safe) < len(legal):
                    mask[:] = 0
                    mask[safe] = 1
        obs = np.asarray(ts.extract_observation(st, decider(st)))[None]
        a = int(_act(obs, mask[None])[0])
        ts.Engine.step_flat(st, a)
        if keyed:
            while not ts.Engine.is_terminal(st):
                st = ts.keyed_key(st, world_key, keyed2, keyedsym)
                if ts.Engine.auto_advance_step(st, 1) == 0:
                    break
        else:
            ts.Engine.auto_advance_step(st)
        if turn is None and (ts.Engine.is_terminal(st) or st.turn != base.turn):
            turn = (int(st.victory_points), int(st.defcon))
    assert turn is not None
    sign = 1 if mover == ts.Player.US else -1
    u = float(ts.Engine.get_terminal_utility(st)) * sign
    score = 1.0 if u > 0 else (0.5 if u == 0 else 0.0)
    return _Played(score, sign * int(st.victory_points), sign * turn[0], turn[1], st)


def test_raw_bytes_round_trip() -> None:
    st = _positions(1)[0]
    raw = st.raw_bytes()
    assert len(raw) == ts.state_size()
    assert ts.state_from_raw_bytes(raw).raw_bytes() == raw
    with pytest.raises(ValueError):
        ts.state_from_raw_bytes(raw[:-1])


def test_every_world_is_the_python_playout_of_that_world() -> None:
    positions = _positions(6)
    worlds = 3
    pl = ts.PlayoutLabels(worlds=worlds, capacity=8, seed=11)
    pl.enqueue(positions, list(range(len(positions))))
    _drive(pl)
    got = pl.take_positions()
    assert sorted(got["id"].tolist()) == list(range(len(positions)))
    for row, pid in enumerate(got["id"].tolist()):
        base = positions[pid]
        ref = [_reference(base, pl.world(base, pid, w)) for w in range(worlds)]
        assert got["count"][row].tolist() == [worlds, 0]
        s = got["sums"][row]
        assert s[0] == pytest.approx(sum(r.score for r in ref))
        assert s[2] == pytest.approx(sum(r.vp for r in ref))
        assert s[3] == pytest.approx(sum(r.turn_end_vp for r in ref))
        assert s[4] == pytest.approx(sum(r.turn_end_defcon for r in ref))
        # A game that reached Final Scoring scored all six regions; the region columns are the
        # engine tap's (engine/tests/test_scoring.cpp), here only consistent with how games end.
        finals = sum(1 for r in ref if r.final.turn == 11)
        assert (got["swing_n"][row][:6] >= finals).all()


def test_the_score_agrees_with_play_safe_itself() -> None:
    positions = _positions(4, seed=8)
    pl = ts.PlayoutLabels(worlds=2, capacity=64, seed=2)
    pl.enqueue(positions, list(range(len(positions))))
    _drive(pl)
    got = pl.take_positions()
    by_id = {int(i): float(got["sums"][r][0]) for r, i in enumerate(got["id"].tolist())}
    starts = [pl.world(positions[p], p, w) for p in range(len(positions)) for w in range(2)]
    movers = [decider(positions[p]) for p in range(len(positions)) for _ in range(2)]
    keys = [ar_key(positions[p]) for p in range(len(positions)) for _ in range(2)]
    scores = play_safe(starts, movers, keys, _act, seed=0, features=0)
    for p in range(len(positions)):
        assert by_id[p] == pytest.approx(scores[2 * p] + scores[2 * p + 1])


def test_labels_do_not_depend_on_the_batch_around_them() -> None:
    positions = _positions(8, seed=4)

    def labels(capacity: int, order: List[int]) -> Dict[int, List[float]]:
        pl = ts.PlayoutLabels(worlds=4, capacity=capacity, seed=9)
        pl.enqueue([positions[i] for i in order], order)
        _drive(pl)
        got = pl.take_positions()
        return {int(i): got["sums"][r].tolist() + got["swing_sum"][r].tolist()
                for r, i in enumerate(got["id"].tolist())}

    assert labels(4, list(range(8))) == labels(64, list(reversed(range(8))))


def test_siblings_share_a_worlds_dice_and_keep_the_movers_next_decision() -> None:
    base = _positions(1, seed=6)[0]
    legal = np.flatnonzero(np.asarray(ActionEncoder.get_legal_mask(base)))
    cands = [int(a) for a in legal[:3]]
    pl = ts.PlayoutLabels(worlds=2, capacity=16, seed=1)
    pl.enqueue([base], [7], [cands])
    _drive(pl)
    got = pl.take_siblings()
    assert len(got["id"]) == 6 and set(got["id"].tolist()) == {7}
    mover = decider(base)
    for r in range(6):
        c, w, failed, has_leaf, _ = got["meta"][r].tolist()
        assert failed == 0
        world = pl.world(base, 7, w)
        after = world.clone()
        ts.Engine.step_flat(after, cands[c])
        ts.Engine.auto_advance_step(after)
        if ts.Engine.is_terminal(after):
            assert has_leaf == 0
            continue
        # The leaf is the first position after the candidate at which the mover decides.
        st = after
        while decider(st) != mover:
            mask = np.asarray(ActionEncoder.get_legal_mask(st))
            obs = np.asarray(ts.extract_observation(st, decider(st)))[None]
            ts.Engine.step_flat(st, int(_act(obs, mask[None])[0]))
            ts.Engine.auto_advance_step(st)
            assert not ts.Engine.is_terminal(st) or has_leaf == 0
            if ts.Engine.is_terminal(st):
                break
        if has_leaf:
            assert bytes(got["leaf"][r]) == st.raw_bytes()


def test_a_terminal_position_or_an_illegal_candidate_is_refused() -> None:
    base = _positions(1)[0]
    pl = ts.PlayoutLabels(worlds=2, capacity=16)
    legal = set(np.flatnonzero(np.asarray(ActionEncoder.get_legal_mask(base))).tolist())
    illegal = next(a for a in range(ActionEncoder.FLAT_ACTION_SIZE) if a not in legal)
    with pytest.raises(ValueError):
        pl.enqueue([base], [0], [[illegal]])
    assert pl.queued() == 0


@pytest.mark.parametrize("keyed2,keyedsym", [(False, False), (True, False), (False, True)])
def test_a_keyed_world_is_the_keyed_python_playout_of_it_and_deals_cleanly(keyed2: bool, keyedsym: bool,
                                                                          capfd: pytest.CaptureFixture[str]) -> None:
    """keyed=True (and keyed2=True or keyedsym=True): every world equals a Python playout that keys each engine step the same way, and
    the deals forced from the world's deck order never name a card the deck lacks (the engine would
    report an anomaly) -- across whole games, turn ends and era changes included."""
    positions = _positions(6, seed=12)
    worlds = 3
    pl = ts.PlayoutLabels(worlds=worlds, capacity=8, seed=5, keyed=True, keyed2=keyed2, keyedsym=keyedsym)
    pl.enqueue(positions, list(range(len(positions))))
    _drive(pl)
    got = pl.take_positions()
    for row, pid in enumerate(got["id"].tolist()):
        base = positions[pid]
        ref = [_reference(base, pl.world(base, pid, w), keyed=True, keyed2=keyed2, keyedsym=keyedsym)
               for w in range(worlds)]
        s = got["sums"][row]
        assert s[0] == pytest.approx(sum(r.score for r in ref))
        assert s[2] == pytest.approx(sum(r.vp for r in ref))
        assert s[3] == pytest.approx(sum(r.turn_end_vp for r in ref))
    sib = ts.PlayoutLabels(worlds=8, capacity=64, seed=3, keyed=True, keyed2=keyed2, keyedsym=keyedsym)
    for i, st in enumerate(_positions(4, seed=13)):
        legal = np.flatnonzero(np.asarray(ActionEncoder.get_legal_mask(st)))
        if legal.size >= 2:
            sib.enqueue([st], [i], [[int(legal[0]), int(legal[1])]])
    _drive(sib)
    assert sib.take_siblings()["id"].size > 0
    assert "anomaly" not in capfd.readouterr().err


def test_keying_is_a_function_of_the_situation() -> None:
    """Two copies of one position key to the same stream; another world key, or another situation, does not."""
    st = _positions(1, seed=14)[0]
    a, b = ts.keyed_key(st, 99), ts.keyed_key(st.clone(), 99)
    assert a.rng_state == b.rng_state
    assert ts.keyed_key(st, 98).rng_state != a.rng_state
    other = _positions(1, seed=15)[0]
    assert ts.keyed_key(other, 99).rng_state != a.rng_state



def test_forced_actions_override_the_argmax_only_where_given_and_legal() -> None:
    """step(logits, forced): -1 (or an illegal action) leaves the argmax, so such a run labels exactly
    as an unforced one; listed_info describes the listed rows."""
    positions = _positions(4, seed=21)

    def labels(forced_kind: str) -> List[float]:
        pl = ts.PlayoutLabels(worlds=2, capacity=16, seed=8)
        pl.enqueue(positions, list(range(len(positions))))
        obs = np.zeros((pl.capacity(), pl.obs_width()), dtype=np.float32)
        while not pl.idle():
            n = pl.fill_live(obs)
            info = pl.listed_info()
            assert info.shape == (n, 4) and (info[:, 2] >= info[:, 3]).all()
            forced = np.full(n, -1, dtype=np.int32)
            if forced_kind == "illegal":
                for j in range(n):
                    mask = np.asarray(ActionEncoder.get_legal_mask(pl.listed_state(j)))
                    forced[j] = int(np.flatnonzero(mask == 0)[0])
            pl.step(np.ascontiguousarray(_logits(obs[:n])), None if forced_kind == "none" else forced)
        got = pl.take_positions()
        return sorted(got["sums"][:, 0].tolist())

    assert labels("none") == labels("minus") == labels("illegal")


def test_a_redealt_world_drops_a_forced_deal() -> None:
    """keyed chance forces the next deal at the turn's end from the world's deck order; a world whose
    unseen cards are then redealt (ts.determinize, the search's worlds) cannot honour those names, so
    it deals at random -- the engine would otherwise report a card that is not in the deck."""
    st = _positions(1, seed=31)[0]
    runner_rng = np.random.default_rng(31)
    while not (st.ctx().decision_player == ts.Player.NONE and st.ctx().pending_roll_type == ts.RollType.TURN_CLEANUP):
        assert not ts.Engine.is_terminal(st)
        if st.ctx().decision_player == ts.Player.NONE:
            ts.Engine.auto_advance_step(st, 1)
            continue
        legal = np.flatnonzero(np.asarray(ActionEncoder.get_legal_mask(st)))
        ts.Engine.step_flat(st, int(runner_rng.choice(legal)))
    keyed = ts.keyed_key(st, 5)
    assert keyed.get_forced_deal_remaining(ts.Player.US) + keyed.get_forced_deal_remaining(ts.Player.USSR) > 0
    redealt = ts.determinize(keyed, ts.Player.US, 9)
    assert redealt.get_forced_deal_remaining(ts.Player.US) == 0
    assert redealt.get_forced_deal_remaining(ts.Player.USSR) == 0
