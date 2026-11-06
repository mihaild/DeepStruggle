"""The browser workbench's WebAssembly engine is the native engine, bit for bit.

The page plays, displays and saves positions with a WebAssembly build of engine/ + bindings/
(tools/scripts/build_web.sh). It is compiled by a different compiler for a different machine, so
"the same sources" proves nothing by itself: an uninitialised byte, a float contraction or an
integer-width assumption would make the page play a different game while looking right. So the
two builds are run on the same games and compared.

Needs the built engine (web/ui/public/engine/ts_engine.mjs) and node, like the rest of tests/web
needs web/ui/dist. A missing build fails here; it is not skipped.
"""
from __future__ import annotations

import hashlib
import json
import os
import random
import shutil
import subprocess
from typing import Any, Dict, List, Optional

import numpy as np
import pytest
import torch

import ts_engine as ts
from ai.search.belief import _hypothesis_obs
from tests.bindings.test_obs_features import _box4_headline
from tools.lib.engine_fingerprint import fingerprint
from tools.lib.game_step import drain_chance

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MODULE = os.path.join(REPO, "web", "ui", "public", "engine", "ts_engine.mjs")
RUNNER = os.path.join(os.path.dirname(os.path.abspath(__file__)), "wasm_parity.mjs")


def _node(*args: str) -> Dict[str, Any]:
    assert os.path.exists(MODULE), (
        f"no WebAssembly engine at {MODULE}: build it with tools/scripts/build_web.sh --engine")
    node = shutil.which("node")
    assert node, "node is needed to run the WebAssembly engine"
    out = subprocess.run([node, RUNNER, MODULE, *args], capture_output=True, text=True, timeout=600)
    assert out.returncode == 0, out.stderr
    return json.loads(out.stdout.strip().splitlines()[-1])


def test_the_wasm_engine_was_built_from_these_sources() -> None:
    got = _node("selftest", "1")["fingerprint"]
    assert got == fingerprint(), (
        f"web/ui/public/engine was built from other sources ({got[:12]}) than these "
        f"({fingerprint()[:12]}): rebuild it with tools/scripts/build_web.sh --engine")


def test_the_page_shows_a_reveal_model_the_box4_headline_as_python_does(tmp_path) -> None:
    """A model trained with HEADLINE_REVEAL (E8's default) must be shown its own view, which the base
    layout cannot tell apart by width: at the US's box-4 headline choice the page's observation under
    each feature set is Python's, byte for byte, and under HEADLINE_REVEAL it differs from the base
    layout. A bit this engine does not know has no width (the page then refuses the model)."""
    state = _box4_headline(5)
    assert ts.revealed_headline(state, ts.Player.US) != 0
    path = tmp_path / "obs.json"
    path.write_text(json.dumps({"save": state.to_save_json(), "features": [0, *FEATURE_SETS]}))
    wasm = _node("obs", str(path))
    assert "error" not in wasm, wasm.get("error")
    for f in [0, *FEATURE_SETS]:
        for p, side in ((1, ts.Player.US), (-1, ts.Player.USSR)):
            want = hashlib.sha256(np.asarray(ts.extract_observation_features(state, side, f),
                                             dtype=np.float32).tobytes()).hexdigest()
            assert wasm["obs"][f"{f}:{p}"] == want, f"features {f}, side {p}"
    reveal = int(ts.OBS_FEATURE_HEADLINE_REVEAL)
    assert wasm["obs"][f"{reveal}:1"] != wasm["obs"]["0:1"]
    assert wasm["unknown_width"] == 0


def test_whole_games_match_the_native_engine() -> None:
    native_digest, native_steps = ts.selftest_digest(200)
    wasm = _node("selftest", "200")
    assert wasm["steps"] == native_steps
    assert wasm["digest"] == native_digest, (
        "the WebAssembly engine played different games from the native one "
        f"(digest {wasm['digest']:08x} vs {native_digest:08x})")


#: The feature sets a model can read, each checked as the page builds it for such a model.
FEATURE_SETS = [int(ts.OBS_FEATURE_OPS_BUDGET), int(ts.OBS_FEATURE_HEADLINE_REVEAL), int(ts.OBS_FEATURES_ALL)]


@pytest.mark.parametrize("seed", [11, 58])
def test_the_page_sees_exactly_what_python_sees(seed: int, tmp_path) -> None:
    """Display JSON, save JSON, both observations -- the base layout and each feature set's view --
    and the mask, at every step of a game driven through the MicroActions a click sends."""
    state = ts.GameState()
    ts.Engine.init_game(state, seed)
    drain_chance(state)
    rng = random.Random(seed)
    h = hashlib.sha256()

    def record() -> None:
        h.update(state.to_display_json().encode())
        h.update(state.to_save_json().encode())
        for p in (ts.Player.US, ts.Player.USSR):
            h.update(np.asarray(ts.extract_observation(state, p), dtype=np.float32).tobytes())
            for f in FEATURE_SETS:
                h.update(np.asarray(ts.extract_observation_features(state, p, f), dtype=np.float32).tobytes())
        h.update(np.asarray(ts.get_flat_action_mask(state, False), dtype=np.uint8).tobytes())

    record()
    actions: List[List[int]] = []
    while not ts.Engine.is_terminal(state) and len(actions) < 1500:
        legal = [int(i) for i in np.nonzero(ts.get_flat_action_mask(state, False))[0]]
        ma = ts.decode_flat_action(state, rng.choice(legal))
        actions.append([int(ma.decision_type), int(ma.primary_id), int(ma.secondary_id), int(ma.flags)])
        ts.Engine.step(state, ma)
        drain_chance(state)
        record()

    path = tmp_path / "actions.json"
    path.write_text(json.dumps({"seed": seed, "actions": actions, "features": FEATURE_SETS}))
    wasm = _node("walk", str(path))
    assert "error" not in wasm, wasm.get("error")
    assert wasm["steps"] == len(actions)
    assert wasm["digest"] == h.hexdigest(), "the page's engine and Python diverged along this game"
    if ts.Engine.is_terminal(state):
        assert wasm["ending"] == ts.game_ending_reason(state)


@pytest.mark.parametrize("seed", [11, 58])
def test_the_page_expands_an_until_control_move_as_the_engine_does(seed: int, tmp_path) -> None:
    """P42: along a game, the page's macro mask is the native one, and every until-control slot it offers, applied as
    the page applies it (ts_macro_points one-point placements), reaches the position the native expansion reaches."""
    state = ts.GameState()
    ts.Engine.init_game(state, seed)
    drain_chance(state)
    rng = random.Random(seed)
    h = hashlib.sha256()
    base = int(ts.FLAT_BASE_SIZE)
    expansions = 0

    def record() -> None:
        nonlocal expansions
        mask = np.asarray(ts.get_flat_action_mask(state, False, True), dtype=np.uint8)
        h.update(mask.tobytes())
        for idx in np.nonzero(mask[base:])[0] + base:
            after = state.clone()
            ts.Engine.step_flat(after, int(idx), False, False, True)
            h.update(f"{int(idx)}:{after.to_save_json()}".encode())
            expansions += 1

    record()
    actions: List[List[int]] = []
    while not ts.Engine.is_terminal(state) and len(actions) < 1500:
        legal = [int(i) for i in np.nonzero(ts.get_flat_action_mask(state, False))[0]]
        ma = ts.decode_flat_action(state, rng.choice(legal))
        actions.append([int(ma.decision_type), int(ma.primary_id), int(ma.secondary_id), int(ma.flags)])
        ts.Engine.step(state, ma)
        drain_chance(state)
        record()

    path = tmp_path / "actions.json"
    path.write_text(json.dumps({"seed": seed, "actions": actions}))
    wasm = _node("macro", str(path))
    assert "error" not in wasm, wasm.get("error")
    assert expansions > 0, "the walk never offered an until-control slot"
    assert wasm["expansions"] == expansions
    assert wasm["digest"] == h.hexdigest(), "the page's until-control moves and the engine's diverged"


class _PlayOuts:
    """The batch's live rows played to their horizons under a fixed synthetic policy, then their leaves
    valued by a fixed synthetic critic, every observation and mask into `h` -- as wasm_parity.mjs's
    playOuts does. `plies` and `leaves` count across runs (a raced position plays out twice), and the
    policy reads the running ply."""

    def __init__(self, batch: Any, h: Any, act: int) -> None:
        self.batch, self.h = batch, h
        self.a_idx = np.arange(act, dtype=np.int64)
        self.plies = self.leaves = 0

    def run(self) -> None:
        batch, h = self.batch, self.h
        while not batch.all_done():
            obs, masks = batch.fill_live()
            h.update(np.ascontiguousarray(obs, dtype=np.float32).tobytes())
            h.update(np.ascontiguousarray(masks, dtype=np.uint8).tobytes())
            row = (((self.a_idx * 2654435761) % 4294967296 + self.plies * 40503 + 7) % 1000) / 1000.0
            batch.step(np.ascontiguousarray(np.tile(row.astype(np.float32), (obs.shape[0], 1))))
            self.plies += 1
            assert self.plies < 2000
        lobs, _lm = batch.fill_leaves()
        h.update(np.ascontiguousarray(lobs, dtype=np.float32).tobytes())
        n = int(lobs.shape[0])
        batch.set_leaf_values(np.asarray([((j * 31) % 17) / 17 - 0.5 for j in range(n)], dtype=np.float32))
        self.leaves += n


@pytest.mark.parametrize("seed,plies,horizon,stop,leaf,keyed,aligned,race", [
    (21, 70, 2, "boundaries", "mover", False, False, 0), (35, 150, 1, "boundaries", "mover", False, False, 0),
    (47, 40, 0, "boundaries", "mover", False, False, 0), (59, 90, 4, "turn", "mover", False, False, 0),
    (63, 120, 2, "later", "mover", False, False, 0), (71, 130, 3, "earlier", "mover", False, False, 0),
    (59, 90, 4, "turn", "decider", False, False, 0), (77, 110, 2, "boundaries", "decider", False, False, 0),
    (59, 90, 4, "turn", "own", False, False, 0), (84, 100, 3, "earlier", "own", False, False, 0),
    (59, 90, 4, "predeal", "mover", False, False, 0), (93, 80, 3, "earlier_predeal", "own", False, False, 0),
    (84, 100, 3, "earlier", "own", 1, False, 0), (35, 150, 1, "boundaries", "mover", 1, False, 0),
    (84, 100, 3, "earlier", "own", 1, True, 0), (21, 70, 2, "boundaries", "mover", 0, True, 0),
    (84, 100, 3, "earlier", "own", 2, True, 0), (35, 150, 1, "boundaries", "mover", 2, False, 0),
    (84, 100, 3, "earlier", "own", 1, True, 4), (35, 150, 1, "boundaries", "mover", 0, False, 3)])
def test_the_rollout_batch_plays_the_same_worlds_in_both_builds(seed: int, plies: int, horizon: int, stop: str,
                                                                 leaf: str, keyed: int, aligned: bool,
                                                                 race: int, tmp_path) -> None:
    """The rollout root's playouts (bindings/rollout_batch.hpp) under a fixed synthetic policy and
    critic: the same rows, the same worlds, the same observations ply by ply, the same values. `keyed`
    is the page's code: 0 off, 1 keyed, 2 keyed2; `race` > 0 races the position for that many more
    worlds once the first are valued (RolloutBatch.race)."""
    state = ts.GameState()
    ts.Engine.init_game(state, seed)
    drain_chance(state)
    rng = random.Random(seed)
    actions: List[List[int]] = []
    while not ts.Engine.is_terminal(state) and len(actions) < plies:
        legal = [int(i) for i in np.nonzero(ts.get_flat_action_mask(state, False))[0]]
        ma = ts.decode_flat_action(state, rng.choice(legal))
        actions.append([int(ma.decision_type), int(ma.primary_id), int(ma.secondary_id), int(ma.flags)])
        ts.Engine.step(state, ma)
        drain_chance(state)
    assert not ts.Engine.is_terminal(state)
    k, worlds, batch_seed = 3, 4, 987654321
    mask = np.asarray(ts.get_flat_action_mask(state, False))
    cands = [int(a) for a in np.flatnonzero(mask)[:k]]
    act = int(mask.shape[0])

    batch = ts.RolloutBatch(horizon=horizon, shared_dice=True, guard=True, features=0, seed=batch_seed, stop=stop,
                            leaf=leaf, keyed=keyed == 1, keyed2=keyed == 2, aligned=aligned,
                            box4=aligned)
    rows = batch.add(state, cands, worlds)
    h = hashlib.sha256()
    play = _PlayOuts(batch, h, act)
    play.run()
    in_race: Optional[List[int]] = None
    if race:
        rows += batch.race(0, race)
        play.run()
        in_race = [int(batch.in_race(0, c)) for c in range(len(cands))]
    native = {
        "candidates": cands, "rows": rows, "worlds_used": batch.worlds_used(0), "plies": play.plies,
        "leaves": play.leaves, "refused": batch.refused(), "digest": h.hexdigest(), "in_race": in_race,
        "values": [float(v) for v in np.asarray(batch.values(), dtype=np.float32)],
        "table": [[p, w, c, int(d), int(t)] for p, w, c, d, t in zip(
            batch.row_position(), batch.row_world(), batch.row_candidate(), batch.row_done(), batch.row_terminal())],
    }
    path = tmp_path / "rollout.json"
    path.write_text(json.dumps({"seed": seed, "actions": actions, "k": k, "worlds": worlds,
                                "horizon": horizon, "batch_seed": batch_seed, "stop": stop, "leaf": leaf,
                                "keyed": keyed, "aligned": aligned, "box4": aligned, "race": race}))
    # The root's choice from synthetic root logits, under each rule, and the top-3 pick from them.
    root = np.asarray([((a * 7919) % 97) / 97 for a in range(act)], dtype=np.float32)
    top = ts.rollout_candidates(root, np.ascontiguousarray(mask, dtype=np.uint8), 3)
    decisions: Dict[str, Dict[str, Any]] = {}
    for rule in ("argmax", "z2", "kl0.05"):
        d = ts.rollout_decide(batch, 0, cands, root, np.ascontiguousarray(mask, dtype=np.uint8), rule)
        decisions[rule] = {"pick": d["pick"], "worlds": d["worlds"], "q": [float(x) for x in d["q"]],
                           "lead": [float(x) for x in d["lead"]], "lead_se": [float(x) for x in d["lead_se"]]}
    wasm = _node("rollout", str(path))
    assert "error" not in wasm, wasm.get("error")
    for key in ("candidates", "rows", "worlds_used", "plies", "leaves", "refused", "table", "in_race"):
        assert wasm[key] == native[key], key
    assert wasm["top"] == top
    assert wasm["digest"] == native["digest"], "the two builds' playouts diverged"
    assert np.array_equal(np.asarray(wasm["values"], dtype=np.float32), np.asarray(native["values"], dtype=np.float32))
    for rule, want in decisions.items():
        got = wasm["decisions"][rule]
        assert (got["pick"], got["worlds"]) == (want["pick"], want["worlds"]), rule
        for key in ("q", "lead", "lead_se"):
            assert np.allclose(np.asarray(got[key], dtype=np.float32), np.asarray(want[key], dtype=np.float32),
                               rtol=1e-6, atol=1e-7), (rule, key)
    assert rows > 0 and native["refused"] == 0


@pytest.mark.parametrize("seed,plies", [(21, 70), (35, 150), (59, 95), (84, 120)])
def test_the_belief_tracker_is_the_same_in_both_builds(seed: int, plies: int, tmp_path) -> None:
    """The belief over the opponent's hand (bindings/belief.hpp) on one game: the page observes moves
    as clicks and finds their flat actions, Python observes the flat actions; at the last position the
    same recorded decisions, the same hypothesis rows (the page builds them whole, Python on its device
    from rows_patch), and under the same synthetic log-likelihoods the same temperature weights and
    the same dealt hands. Then the page's raced belief search: one draw of every world's hands, the
    first `race` of them for every candidate, the rest for the candidates left (ts_rollout_add_belief
    and ts_rollout_race against RolloutBatch.add and .race with the same slices)."""
    particles, tracker_seed, n_worlds, race, n_cands, horizon, batch_seed = 16, 4242, 8, 3, 3, 2, 13579
    tracker = ts.BeliefTracker(particles, True, tracker_seed)
    state = ts.GameState()
    ts.Engine.init_game(state, seed)
    drain_chance(state)
    rng = random.Random(seed)
    actions: List[List[int]] = []
    while not ts.Engine.is_terminal(state) and len(actions) < plies:
        legal = [int(i) for i in np.nonzero(ts.get_flat_action_mask(state, False))[0]]
        a = rng.choice(legal)
        ma = ts.decode_flat_action(state, a)
        tracker.observe(0, state, state.ctx().decision_player, a)
        actions.append([int(ma.decision_type), int(ma.primary_id), int(ma.secondary_id), int(ma.flags)])
        ts.Engine.step(state, ma)
        drain_chance(state)
    assert not ts.Engine.is_terminal(state)
    has = tracker.prepare([0], [state])[0]
    D = tracker.job_decisions(0) if tracker.job_count() else 0
    h = hashlib.sha256()
    weights: List[float] = []
    worlds = None
    rows_actions: List[int] = []
    if D:
        bases, pools, lefts, job_of, acts, hands, packed = tracker.rows_patch(0)
        rows_actions = [int(x) for x in acts]
        for d in range(D):
            held = torch.from_numpy(hands[0] | lefts[d][None, :])[None]
            obs = _hypothesis_obs(torch.from_numpy(bases[d])[None], held, torch.from_numpy(pools[d]).bool()[None])
            masks = np.unpackbits(packed[d], axis=1)[:, :int(ts.get_flat_action_mask(state, False).shape[0])]
            h.update(np.ascontiguousarray(obs.numpy(), dtype=np.float32).tobytes())
            h.update(np.ascontiguousarray(masks, dtype=np.uint8).tobytes())
        T = len(ts.BeliefTracker.temperatures())
        lp = np.empty((T, D, particles), dtype=np.float32)
        for t in range(T):
            for d in range(D):
                for k in range(particles):
                    lp[t, d, k] = -(((k * 2654435761 + d * 40503 + t * 7) % 1000) / 250)
        tracker.set_log_likelihoods(lp)
        weights = [float(x) for x in tracker.temperature_weights(0)]
        w = tracker.worlds(0, n_worlds)
        worlds = None if w is None else [int(x) for x in np.asarray(w).reshape(-1)]
    raced: Optional[Dict[str, Any]] = None
    if worlds is not None:
        hands = np.asarray(tracker.worlds(0, n_worlds), dtype=np.uint8).reshape(n_worlds, 110)
        mask = np.asarray(ts.get_flat_action_mask(state, False))
        cands = [int(a) for a in np.flatnonzero(mask)[:n_cands]]
        batch = ts.RolloutBatch(horizon=horizon, shared_dice=True, guard=True, features=0, seed=batch_seed)
        rows = batch.add(state, cands, race, hands=np.ascontiguousarray(hands[:race]))
        rh = hashlib.sha256()
        play = _PlayOuts(batch, rh, int(mask.shape[0]))
        play.run()
        rows += batch.race(0, n_worlds - race, hands=np.ascontiguousarray(hands[race:]))
        play.run()
        raced = {"candidates": cands, "rows": rows, "worlds_used": batch.worlds_used(0), "plies": play.plies,
                 "leaves": play.leaves, "digest": rh.hexdigest(),
                 "in_race": [int(batch.in_race(0, c)) for c in range(len(cands))]}
    path = tmp_path / "belief.json"
    path.write_text(json.dumps({"seed": seed, "actions": actions, "particles": particles,
                                "tracker_seed": tracker_seed, "worlds": n_worlds, "race": race, "k": n_cands,
                                "horizon": horizon, "batch_seed": batch_seed}))
    wasm = _node("belief", str(path))
    assert "error" not in wasm, wasm.get("error")
    assert wasm["observed"] == len(actions), "the page found a flat action for every click"
    assert (bool(wasm["has"]), wasm["decisions"]) == (bool(has), D)
    assert wasm["actions"] == rows_actions
    assert wasm["digest"] == h.hexdigest(), "the two builds' hypothesis rows differ"
    assert np.allclose(np.asarray(wasm["weights"], dtype=np.float32), np.asarray(weights, dtype=np.float32), atol=1e-6)
    assert wasm["worlds"] == worlds
    assert wasm["raced"] == raced, "the two builds' raced belief searches differ"
