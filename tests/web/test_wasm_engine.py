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
from typing import Any, Dict, List

import numpy as np
import pytest

import ts_engine as ts
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


def test_whole_games_match_the_native_engine() -> None:
    native_digest, native_steps = ts.selftest_digest(200)
    wasm = _node("selftest", "200")
    assert wasm["steps"] == native_steps
    assert wasm["digest"] == native_digest, (
        "the WebAssembly engine played different games from the native one "
        f"(digest {wasm['digest']:08x} vs {native_digest:08x})")


@pytest.mark.parametrize("seed", [11, 58])
def test_the_page_sees_exactly_what_python_sees(seed: int, tmp_path) -> None:
    """Display JSON, save JSON, both observations and the mask, at every step of a game driven
    through the MicroActions a click sends."""
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
    path.write_text(json.dumps({"seed": seed, "actions": actions}))
    wasm = _node("walk", str(path))
    assert "error" not in wasm, wasm.get("error")
    assert wasm["steps"] == len(actions)
    assert wasm["digest"] == h.hexdigest(), "the page's engine and Python diverged along this game"
    if ts.Engine.is_terminal(state):
        assert wasm["ending"] == ts.game_ending_reason(state)
