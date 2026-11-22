"""Keyed chance's three packings of a situation (bindings/keyed_chance.hpp).

`keyed` is kept bit for bit, so every keyed player measured with it stays one searcher: there a USSR
player field (-1, widened as a signed byte) hides the turn, action round and staged roll. `keyed2`
narrows each player field to its byte first, so every packed field separates situations, and it is
`keyed` wherever no player field is the USSR. `keyedsym` packs any side as `keyed` packs the USSR, so
the same fields drop out for the US too, and keys the two sides apart with one more word.
"""
from __future__ import annotations

import copy
from typing import Any, Dict, List, Tuple

import numpy as np
import pytest
import ts_engine as ts

FIELDS = [("turn", 5, False), ("action_round", 3, False),
          ("decision_type", int(ts.DecisionType.POINT_NODE), True),
          ("pending_roll", int(ts.RollType.COUP), True), ("roll_target", 24, True)]


def _saved(phasing: int, decider: int, roller: int) -> Dict[str, Any]:
    """A position's save dict, its three player fields given."""
    state = ts.GameState()
    ts.Engine.init_game(state, 1)
    saved: Dict[str, Any] = state.to_save_dict()
    saved.update(turn=4, action_round=1, phasing_player=phasing)
    saved["ctx_stack"][0].update(decision_player=decider, decision_type=int(ts.DecisionType.ROLL_DIE),
                                pending_roll=0, roll_actor=roller, roll_target=25)
    return saved


def _pair(player: int, field: str, value: int, context: bool,
          players: Tuple[int, int, int] | None = None) -> Tuple[ts.GameState, ts.GameState]:
    """Two positions that differ in `field` alone, every player field `player` (or `players`: phasing,
    decision, roll)."""
    saved = _saved(*(players or (player, player, player)))
    changed = copy.deepcopy(saved)
    (changed["ctx_stack"][0] if context else changed)[field] = value
    return ts.state_from_save_dict(saved), ts.state_from_save_dict(changed)


def _key(state: ts.GameState, world: int = 123, keyed2: bool = False, keyedsym: bool = False) -> int:
    return int(ts.keyed_key(state, world, keyed2, keyedsym).rng_state)


#: keyed's and keyed2's keys of _pair's two positions (world key 123), taken before keyedsym existed:
#: (player, field, keyed2, unchanged position's key, changed position's key). Recorded leaderboard players
#: play these packings, so neither may move.
PINNED = [(-1, "turn", False, 2947846877068017199, 2947846877068017199),
          (-1, "turn", True, 8307018449635304717, 3334032705725675623),
          (-1, "roll_target", False, 2947846877068017199, 2947846877068017199),
          (-1, "roll_target", True, 8307018449635304717, 16579197393503854579),
          (1, "turn", False, 3675713027390267116, 13908951002101334841),
          (1, "turn", True, 3675713027390267116, 13908951002101334841),
          (1, "roll_target", False, 3675713027390267116, 17165489052136194444),
          (1, "roll_target", True, 3675713027390267116, 17165489052136194444)]


@pytest.mark.parametrize("player,field,keyed2,before,after", PINNED)
def test_keyed_and_keyed2_key_bit_for_bit_as_before(player: int, field: str, keyed2: bool, before: int,
                                                    after: int) -> None:
    value, context = {f: (v, c) for f, v, c in FIELDS}[field]
    a, b = _pair(player, field, value, context)
    assert (_key(a, keyed2=keyed2), _key(b, keyed2=keyed2)) == (before, after)


@pytest.mark.parametrize("player", [-1, 1])
@pytest.mark.parametrize("field,value,context", FIELDS)
def test_keyedsym_drops_the_same_fields_for_either_side(player: int, field: str, value: int, context: bool) -> None:
    """keyed's USSR collisions, on the US's side as well: with the side in every player field, no field
    above them separates situations -- a US die no more depends on the turn than a USSR one under keyed."""
    a, b = _pair(player, field, value, context)
    assert _key(a, keyedsym=True) == _key(b, keyedsym=True)
    assert _key(a, keyedsym=True) != _key(a, 124, keyedsym=True)
    assert _key(a, keyedsym=True) not in (_key(a), _key(a, keyed2=True))


@pytest.mark.parametrize("field,value", [("turn", 5), ("action_round", 3)])
def test_keyedsym_a_us_turn_or_action_round_keys_alike_as_the_ussrs_does_under_keyed(field: str, value: int) -> None:
    """The owner's case: the US phasing, nobody deciding or rolling -- only the phasing field hides
    anything, and it hides the turn and action round, as the USSR's does under keyed; a context field
    no side's field covers still counts."""
    nobody = (1, 0, 0)
    a, b = _pair(1, field, value, False, nobody)
    assert _key(a, keyedsym=True) == _key(b, keyedsym=True)
    assert _key(a) != _key(b)
    ussr_a, ussr_b = _pair(-1, field, value, False, (-1, 0, 0))
    assert _key(ussr_a) == _key(ussr_b) and _key(ussr_a, keyedsym=True) == _key(ussr_b, keyedsym=True)
    c, d = _pair(1, "roll_target", 24, True, nobody)
    assert _key(c, keyedsym=True) != _key(d, keyedsym=True)


@pytest.mark.parametrize("which", [0, 1, 2])
def test_keyedsym_keys_the_us_and_the_ussr_apart(which: int) -> None:
    """Both sides pack as all ones, so the extra word of player bytes is what tells them apart: a
    situation differing only in one player field's side keys differently."""
    for base in ((1, 1, 1), (-1, -1, -1), (1, 0, 0), (0, -1, 0), (0, 0, 1)):
        if base[which] == 0:
            continue
        flipped = tuple(-p if i == which else p for i, p in enumerate(base))
        a = ts.state_from_save_dict(_saved(*base))
        b = ts.state_from_save_dict(_saved(*flipped))
        assert _key(a, keyedsym=True) != _key(b, keyedsym=True), (base, flipped)


def test_keyed2_and_keyedsym_are_exclusive() -> None:
    state = ts.state_from_save_dict(_saved(1, 1, 1))
    with pytest.raises(ValueError, match="exclusive"):
        ts.keyed_key(state, 1, True, True)
    with pytest.raises(ValueError, match="exclusive"):
        ts.RolloutBatch(keyed2=True, keyedsym=True)
    with pytest.raises(ValueError, match="exclusive"):
        ts.PlayoutLabels(keyed2=True, keyedsym=True)


@pytest.mark.parametrize("player", [-1, 1])
@pytest.mark.parametrize("field,value,context", FIELDS)
def test_keyed2_separates_every_packed_field(player: int, field: str, value: int, context: bool) -> None:
    a, b = _pair(player, field, value, context)
    assert ts.keyed_key(a, 123, True).rng_state != ts.keyed_key(b, 123, True).rng_state
    assert ts.keyed_key(a, 123, True).rng_state != ts.keyed_key(a, 124, True).rng_state


@pytest.mark.parametrize("field,value,context", FIELDS)
def test_keyed_keeps_its_original_packing(field: str, value: int, context: bool) -> None:
    """The original packing: on the US's side every field separates situations and keyed2 keys alike;
    on the USSR's they collide -- the defect keyed2 removes, kept so keyed results stay comparable."""
    a, b = _pair(1, field, value, context)
    assert ts.keyed_key(a, 123).rng_state != ts.keyed_key(b, 123).rng_state
    assert ts.keyed_key(a, 123).rng_state == ts.keyed_key(a, 123, True).rng_state
    a, b = _pair(-1, field, value, context)
    assert ts.keyed_key(a, 123).rng_state == ts.keyed_key(b, 123).rng_state
    assert ts.keyed_key(a, 123).rng_state != ts.keyed_key(a, 123, True).rng_state


def test_keyed2_che_coups_do_not_reuse_one_die_for_different_targets() -> None:
    ids = {ts.MapData.get_country_info(i)["name"]: i for i in range(84)}
    pairs = []
    for world in range(10):
        state = ts.GameState()
        ts.Engine.init_game(state, 1)
        state.current_phase = ts.Phase.ACTION_ROUND
        state.phasing_player = ts.Player.USSR
        state.turn, state.action_round = 4, 1
        state.ctx().pending_op_card = 107
        for name in ("Colombia", "Sudan"):
            state.set_country(ids[name], 2, 0)
        ts.CardHandlers.trigger_event(state, 107, ts.Player.USSR)
        dice = []
        for name in ("Colombia", "Sudan"):
            ts.CardHandlers.handle_event_step(state, ts.MicroAction(ts.DecisionType.POINT_NODE, ids[name], 0, 0))
            state = ts.keyed_key(state, world, True)
            ts.CardHandlers.handle_event_step(state, ts.MicroAction(ts.DecisionType.ROLL_DIE, 0, 0, 0))
            dice.append(state.to_save_dict()["last_die_roll"])
        pairs.append(dice)
    assert any(a != b for a, b in pairs), pairs


def test_the_rollout_batch_keys_with_the_packing_it_was_given() -> None:
    """keyed2=True reaches the batch (and implies keyed): at a USSR decision its rows are keyed apart
    from keyed's, from the same worlds."""
    state = ts.GameState()
    ts.Engine.init_game(state, 3)
    while ts.Engine.auto_advance_step(state, 1):
        pass
    saved: Dict[str, Any] = state.to_save_dict()
    assert saved["ctx_stack"][saved["ctx_stack_depth"]]["decision_player"] == -1
    cands = [int(a) for a in np.flatnonzero(np.asarray(ts.get_flat_action_mask(state, False)))[:2]]
    rows = []
    for keyed, keyed2, keyedsym in ((True, False, False), (False, True, False), (True, True, False),
                                    (False, False, True), (True, False, True)):
        b = ts.RolloutBatch(horizon=0, seed=5, keyed=keyed, keyed2=keyed2, keyedsym=keyedsym)
        n = b.add(state, cands, 2)
        rows.append([b.row_state(r).rng_state for r in range(n)])
    assert rows[0] != rows[1] and rows[1] == rows[2]
    assert rows[3] == rows[4] and rows[3] not in (rows[0], rows[1])
    # keyed's and keyed2's rows as they were before keyedsym existed
    assert cands == [121, 129]
    assert rows[0] == [3780117730282841367, 3780117730282841367, 11422801565828239613, 11422801565828239613]
    assert rows[1] == [213994199633789656, 213994199633789656, 6601279820434675531, 6601279820434675531]


def _midgame(plies: int = 60, seed: int = 7) -> ts.GameState:
    runner = ts.VectorizedBatchRunner(1, seed)
    rng = np.random.default_rng(seed)
    for _ in range(plies):
        m = np.asarray(runner.get_action_masks())[0]
        runner.step_flat_all([int(rng.choice(np.flatnonzero(m)))], True)
    st = runner.get_state(0)
    assert not ts.Engine.is_terminal(st)
    return st


def _played(state: ts.GameState, cands: List[int], keyed2: bool, keyedsym: bool) -> Tuple[List[bytes], np.ndarray]:
    """A deployed-shape batch (stop earlier, leaf own, aligned, box4) played to its leaves by a fixed
    pseudo-random policy and valued by a fixed function of each leaf's observation."""
    b = ts.RolloutBatch(horizon=2, seed=9, stop="earlier", leaf="own", keyed=True, keyed2=keyed2, keyedsym=keyedsym,
                        aligned=True, box4=True)
    b.add(state, cands, 4)
    plies = 0
    while not b.all_done():
        _obs, masks = b.fill_live()
        rng = np.random.default_rng(plies)
        b.step(np.ascontiguousarray(np.where(masks > 0, rng.random(masks.shape), -1e9).astype(np.float32)))
        plies += 1
        assert plies < 4000
    obs, _ = b.fill_leaves()
    b.set_leaf_values(np.ascontiguousarray(np.tanh(np.asarray(obs).sum(axis=1) * 1e-3)).astype(np.float32))
    return [bytes(b.row_state(r).raw_bytes()) for r in range(b.rows())], np.asarray(b.values())


def test_a_keyedsym_rollout_batch_plays_and_is_reproducible() -> None:
    state = _midgame()
    cands = [int(a) for a in np.flatnonzero(np.asarray(ts.get_flat_action_mask(state, False)))[:3]]
    rows, values = _played(state, cands, False, True)
    again_rows, again_values = _played(state, cands, False, True)
    assert rows == again_rows and np.array_equal(values, again_values)
    assert not np.isnan(values).any()
    assert rows != _played(state, cands, False, False)[0]
