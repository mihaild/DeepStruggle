"""A `pos=` link means the same position to the page and to Python.

The page encodes positions itself (web/ui/src/game/position.ts: the engine's save JSON, zlib via
the browser's "deflate" CompressionStream, base64url). Links written by the server-side workbench
used Python's zlib, and a token may be opened by a tool in Python -- so both directions are
checked, on a mid-game position.
"""
from __future__ import annotations

import base64
import json
import random
import zlib

import numpy as np

import ts_engine as ts
from tests.web.js_runner import run_ts
from tools.lib.game_step import drain_chance


def _mid_game() -> ts.GameState:
    s = ts.GameState()
    ts.Engine.init_game(s, 31)
    drain_chance(s)
    rng = random.Random(31)
    for _ in range(180):
        if ts.Engine.is_terminal(s):
            break
        legal = [int(i) for i in np.nonzero(ts.get_flat_action_mask(s, False))[0]]
        ts.Engine.step_flat(s, rng.choice(legal))
        drain_chance(s)
    return s


def _python_token(save_json: str) -> str:
    return base64.urlsafe_b64encode(zlib.compress(save_json.encode(), 9)).decode().rstrip("=")


def _python_decode(token: str) -> str:
    return zlib.decompress(base64.urlsafe_b64decode(token + "=" * (-len(token) % 4))).decode()


def test_tokens_travel_both_ways(tmp_path) -> None:
    s = _mid_game()
    save = s.to_save_json()
    case = tmp_path / "case.json"
    case.write_text(json.dumps({"save_json": save, "python_token": _python_token(save)}))
    out = run_ts("position_interop.ts", str(tmp_path), str(case))

    # The page's token opens in Python, as the same position.
    assert _python_decode(out["page_token"]) == save
    assert ts.state_from_save_json(_python_decode(out["page_token"])).to_save_dict() == s.to_save_dict()
    # A Python (old server) token opens in the page.
    assert out["decoded_python"] == save
    assert out["garbage_refused"]
    assert len(out["page_token"]) < 2000, "a position must fit comfortably in an address bar"
