"""The workbench in a real browser: the game, the model and the link all run in the page.

The engine is the WebAssembly build, the model runs in onnxruntime-web, and the only server is
the local one listing checkpoints and replays -- or, for the GitHub Pages case, a plain static
file server with nothing behind it. Checkpoints are never committed: a random-weight network is
written into a temp tree, which is all a readout-agreement check needs.

Needs web/ui/dist (tools/scripts/build_web.sh) and a Playwright Chromium.
"""
from __future__ import annotations

import functools
import http.server
import json
import os
import socket
import threading
import time
from typing import Any, Dict, Iterator

import numpy as np
import pytest
import torch
import uvicorn
from playwright.sync_api import sync_playwright

import ts_engine as ts
from ai.eval.policy_readout import read_critic, read_policy
from bindings.action_encoder import ActionEncoder
from tools.lib.player_agent import NeuralAgent
from web.server.main import app

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DIST = os.path.join(REPO, "web", "ui", "dist")
RUN = "E9-04-01_20260101_000000"
SNAP = "snapshot_2000000steps.pt"


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return int(s.getsockname()[1])


@pytest.fixture(scope="module")
def checkpoint(tmp_path_factory: pytest.TempPathFactory) -> Dict[str, str]:
    from ai.models.coldwar_net_v2 import create_coldwar_net_v2

    root = tmp_path_factory.mktemp("e2e_checkpoints")
    run = root / RUN
    run.mkdir()
    torch.manual_seed(11)
    torch.save(create_coldwar_net_v2(torch.device("cpu")).state_dict(), run / SNAP)
    (run / "metadata.json").write_text(json.dumps({"merged_influence": False}))
    return {"root": str(root), "path": str(run / SNAP)}


@pytest.fixture(scope="module")
def server(checkpoint: Dict[str, str], tmp_path_factory: pytest.TempPathFactory) -> Iterator[str]:
    assert os.path.exists(os.path.join(DIST, "engine", "ts_engine.mjs")), (
        "web/ui/dist is missing or has no engine: build it with tools/scripts/build_web.sh")
    env = {
        "TS_CHECKPOINTS_DIR": checkpoint["root"],
        "TS_ONNX_CACHE_DIR": str(tmp_path_factory.mktemp("e2e_onnx")),
        "TS_REPLAYS_DIR": str(tmp_path_factory.mktemp("e2e_replays")),
    }
    old = {k: os.environ.get(k) for k in env}
    os.environ.update(env)
    port = _free_port()
    srv = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="error"))
    threading.Thread(target=srv.run, daemon=True).start()
    time.sleep(1.0)
    try:
        yield f"http://127.0.0.1:{port}"
    finally:
        srv.should_exit = True
        for k, v in old.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


@pytest.fixture(scope="module")
def static_site() -> Iterator[str]:
    """The built page on a plain static server with no API -- what GitHub Pages is."""
    handler = functools.partial(http.server.SimpleHTTPRequestHandler, directory=DIST)
    httpd = http.server.ThreadingHTTPServer(("127.0.0.1", _free_port()), handler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    try:
        yield f"http://127.0.0.1:{httpd.server_address[1]}"
    finally:
        httpd.shutdown()


@pytest.fixture(scope="module")
def browser() -> Iterator[Any]:
    with sync_playwright() as p:
        b = p.chromium.launch(headless=True, args=["--no-sandbox", "--disable-setuid-sandbox",
                                                   "--disable-dev-shm-usage", "--disable-gpu"])
        yield b
        b.close()


def _open(browser: Any, url: str) -> Any:
    page = browser.new_page(viewport={"width": 1440, "height": 900})
    errors: list = []
    page.on("pageerror", lambda e: errors.append(str(e)))
    page.errors = errors
    page.goto(url)
    page.wait_for_function("window.__wb && window.__wb.liveState && window.__wb.positionToken", timeout=30000)
    return page


def _pick_local_model(page: Any) -> None:
    page.select_option("#analysis-source-select", "local")
    page.wait_for_function(f"[...document.querySelectorAll('#analysis-run-select option')].some(o => o.value === '{RUN}')")
    page.select_option("#analysis-run-select", RUN)
    page.wait_for_selector("#analysis-body .analysis-choice", timeout=120000)


def test_a_game_is_played_by_the_page_itself(browser: Any, server: str) -> None:
    page = _open(browser, server + "/")
    assert "ENGINE" in page.inner_text("#connection-status")
    assert page.evaluate("window.__wb.state.decision_context.decision_type_name") == "POINT_NODE"
    cid = page.evaluate("window.__wb.state.legal_actions.valid_ids[0]")
    page.click(f".svg-country-node[data-id='{cid}']")
    page.wait_for_function("window.__wb.state.step_index === 1")
    last = page.evaluate("window.__wb.state.action_logs.at(-1)")
    assert "places 1 Influence in" in last["text"] and last["player"] == "USSR"
    assert not page.errors


def test_the_page_reads_the_model_as_python_does(browser: Any, server: str, checkpoint: Dict[str, str]) -> None:
    """The in-browser readout (engine wasm + ONNX + analysis/readout.ts) against read_policy /
    read_critic in Python, with torch, on the same checkpoint and the same position."""
    page = _open(browser, server + "/")
    _pick_local_model(page)
    for _ in range(3):   # a few positions, not just the opening
        page.wait_for_function("window.__wb.liveAnalysis && window.__wb.liveAnalysis.policy")
        a = page.evaluate("window.__wb.liveAnalysis")
        state = ts.state_from_save_json(page.evaluate("window.__wb.engine.saveJson()"))

        model = NeuralAgent.from_checkpoint(checkpoint["path"], device="cpu").model
        p = state.ctx().decision_player
        obs = torch.from_numpy(np.asarray(ts.extract_observation(state, p), dtype=np.float32)).unsqueeze(0)
        mask = torch.from_numpy(np.asarray(ActionEncoder.get_legal_mask(state), dtype=np.uint8)).unsqueeze(0)
        _, pol = read_policy(model, obs, mask, deterministic=True, state=state)
        crit = read_critic(model, state)

        assert a["policy"]["argmax_idx"] == pol["argmax_idx"]
        page_p = {c["idx"]: c["p"] for c in a["choices"]}
        py_p = {e["idx"]: e["p"] for e in pol["top"]}
        assert set(page_p) == set(py_p), "the page and Python disagree about which actions are legal"
        assert max(abs(page_p[i] - py_p[i]) for i in py_p) < 1e-3
        for k in ("v_win_us", "v_win_ussr", "v_vp_us", "v_vp_ussr"):
            assert abs(a["critic"][k] - crit[k]) < 1e-4, k
        n = page.evaluate("window.__wb.state.step_index")
        page.click("#btn-play-favourite")
        page.wait_for_function(f"window.__wb.state.step_index > {n}")
    assert not page.errors


def test_auto_play_undo_and_the_shared_link(browser: Any, server: str) -> None:
    page = _open(browser, server + "/")
    _pick_local_model(page)
    page.select_option("#analysis-autoplay-select", "USSR")
    # The USSR setup is auto-played; the US setup waits for us.
    page.wait_for_function("window.__wb.state.step_index > 0 && window.__wb.state.decision_context.decision_player === 'US'", timeout=60000)
    # The engine's own save, not the link token: the token is encoded asynchronously and can
    # land a moment after the move it describes.
    position = "window.__wb.engine.saveJson()"
    parked = page.evaluate(position)
    page.wait_for_timeout(1000)
    assert page.evaluate(position) == parked, "auto-play must not move for the US"

    history = page.evaluate("history.length")
    for _ in range(40):
        before = page.evaluate(position)
        s0 = page.evaluate("window.__wb.state.step_index")
        page.keyboard.press("f")
        page.wait_for_function(f"window.__wb.state.step_index > {s0}")
        page.wait_for_function("window.__wb.state.is_terminal || window.__wb.state.decision_context.decision_player === 'US'", timeout=60000)
        if page.evaluate("window.__wb.state.step_index") > s0 + 1:
            break
    else:
        pytest.fail("the USSR never got a move to auto-play")
    page.click("#btn-undo-action")
    page.wait_for_function(f"{position} === {json.dumps(before)}", timeout=10000)
    page.wait_for_timeout(1000)
    assert page.evaluate(position) == before, "auto-play replayed the move just taken back"
    assert page.evaluate("history.length") == history, "moves must replace, not push, history"

    page.wait_for_function("new URLSearchParams(location.search).get('pos') === window.__wb.positionToken")
    url = page.url
    assert f"model=local%3A{RUN}" in url and "auto=ussr" in url
    other = _open(browser, url)
    assert other.evaluate("window.__wb.engine.saveJson()") == page.evaluate("window.__wb.engine.saveJson()")
    other.wait_for_selector("#analysis-body .analysis-choice", timeout=60000)
    assert other.input_value("#analysis-snapshot-select") == SNAP
    assert not page.errors and not other.errors


def test_a_dropped_onnx_file_is_a_model(browser: Any, server: str, checkpoint: Dict[str, str], tmp_path) -> None:
    from tools.export_onnx import export

    onnx_path = str(tmp_path / "dropped.onnx")
    export(checkpoint["path"], onnx_path, positions=16)
    page = _open(browser, server + "/")
    page.select_option("#analysis-source-select", "file")
    page.set_input_files("#analysis-file-input", onnx_path)
    page.wait_for_selector("#analysis-body .analysis-choice", timeout=60000)
    assert "E4 view" in page.inner_text("#analysis-model-info")
    page.wait_for_function("new URLSearchParams(location.search).has('pos')")
    assert "model=" not in page.url, "a dropped file has no address to share"


def test_a_file_that_is_not_an_export_is_refused_with_a_reason(browser: Any, server: str, tmp_path) -> None:
    bogus = tmp_path / "bogus.onnx"
    bogus.write_bytes(b"\x08\x07not really a model")
    page = _open(browser, server + "/")
    page.select_option("#analysis-source-select", "file")
    page.set_input_files("#analysis-file-input", str(bogus))
    page.wait_for_selector("#analysis-body .analysis-error", timeout=30000)
    assert page.inner_text("#analysis-status") == "error"


def test_debug_overrides_and_their_undo(browser: Any, server: str) -> None:
    page = _open(browser, server + "/")
    page.click("#btn-toggle-debug")
    page.fill("#dbg-vp", "5")
    page.click("#dbg-apply-vp")
    page.wait_for_function("window.__wb.state.victory_points === 5")
    assert "[DEBUG] Set VP -> 5" in page.evaluate("window.__wb.state.action_logs.at(-1).text")
    page.click("#btn-undo-action")
    page.wait_for_function("window.__wb.state.victory_points === 0")


def test_the_live_game_exports_as_a_replay(browser: Any, server: str) -> None:
    page = _open(browser, server + "/")
    for _ in range(3):
        n = page.evaluate("window.__wb.state.step_index")
        cid = page.evaluate("window.__wb.state.legal_actions.valid_ids[0]")
        page.click(f".svg-country-node[data-id='{cid}']")
        page.wait_for_function(f"window.__wb.state.step_index > {n}")
    with page.expect_download() as dl:
        page.click("#btn-rep-export")
    doc = json.loads(open(dl.value.path()).read())
    assert len(doc["steps"]) == 3
    assert doc["metadata"]["engine_fingerprint"] == page.evaluate("window.__wb.engine.fingerprint")
    assert doc["steps"][-1]["state_snapshot"]["countries"]


def test_the_page_works_with_no_server_behind_it(browser: Any, static_site: str) -> None:
    """GitHub Pages: the same build, no API. The game runs; local sources are marked missing."""
    page = _open(browser, static_site + "/")
    assert "ENGINE" in page.inner_text("#connection-status")
    assert page.evaluate("document.getElementById('analysis-source-local-option').disabled")
    assert page.evaluate("document.getElementById('rep-server-select').classList.contains('hidden')")
    cid = page.evaluate("window.__wb.state.legal_actions.valid_ids[0]")
    page.click(f".svg-country-node[data-id='{cid}']")
    page.wait_for_function("window.__wb.state.step_index === 1")
    assert not page.errors
