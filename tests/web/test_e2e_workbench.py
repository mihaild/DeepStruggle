"""The workbench in a real browser: the game, the model and the link all run in the page.

The engine is the WebAssembly build, the model runs in onnxruntime-web, and the only server is
the local one listing checkpoints and replays -- or, for the GitHub Pages case, a plain static
file server with nothing behind it. Checkpoints are never committed: a random-weight network is
written into a temp tree, which is all a readout-agreement check needs.

Needs web/ui/dist (tools/scripts/build_web.sh) and a Playwright Chromium.
"""
from __future__ import annotations

import functools
import hashlib
import http.server
import json
import os
import threading
from typing import Any, Callable, Dict, Iterator, List, Optional, Set, Tuple
from urllib.parse import parse_qs, quote

import numpy as np
import pytest
import torch
from playwright.sync_api import sync_playwright

import ts_engine as ts
from ai.eval.policy_readout import read_critic, read_policy
from ai.search.rollout_root import RolloutConfig, RolloutRoot
from bindings.action_encoder import ActionEncoder
from tools.lib.player_agent import NeuralAgent
from tools.lib.player_spec import Rollout, spec_from_dict
from tests.web.live_server import serving
from web.server.main import app

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DIST = os.path.join(REPO, "web", "ui", "dist")
RUN = "E9-04-01_20260101_000000"
SNAP = "snapshot_2000000steps.pt"

#: Chromium flag every E2E browser launches with. A link that names no model makes the page list
#: the default Hugging Face repo; a test must never depend on -- or download from -- the real one.
#: Requests a test serves itself with `page.route` are answered before name resolution.
HF_BLOCKED = "--host-resolver-rules=MAP huggingface.co ~NOTFOUND"
HF_TREE = "https://huggingface.co/api/models/mihaild/deepstruggle/tree/main?recursive=true"


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
    try:
        with serving(app) as url:
            yield url
    finally:
        for k, v in old.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


@pytest.fixture(scope="module")
def static_site() -> Iterator[str]:
    """The built page on a plain static server with no API -- what GitHub Pages is."""
    handler = functools.partial(http.server.SimpleHTTPRequestHandler, directory=DIST)
    httpd = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)   # bound, so ready
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    try:
        yield f"http://127.0.0.1:{httpd.server_address[1]}"
    finally:
        httpd.shutdown()


@pytest.fixture(scope="module")
def browser() -> Iterator[Any]:
    with sync_playwright() as p:
        b = p.chromium.launch(headless=True, args=["--no-sandbox", "--disable-setuid-sandbox",
                                                   "--disable-dev-shm-usage", "--disable-gpu",
                                                   # Never the real Hugging Face: a link that names
                                                   # no model lists it for the default one.
                                                   HF_BLOCKED])
        yield b
        b.close()


def _answer_mode_dialog(page: Any, choice: str = "mode-choice-dev") -> None:
    """A fresh open asks how to play (#mode-dialog) over the live board; answer it -- Developer by
    default, what every test here assumes -- so later clicks reach the page. A link that names a
    view, a position or a replay is not asked, and then there is nothing to answer."""
    dialog = page.locator("#mode-dialog")
    if dialog.count() and dialog.is_visible():
        page.click(f"#{choice}")
        page.wait_for_selector("#mode-dialog", state="hidden")


def _open(browser: Any, url: str, setup: Optional[Callable[[Any], None]] = None, context: Any = None) -> Any:
    """A page in a fresh browser context -- or in `context`, to share its storage across pages."""
    page = context.new_page() if context else browser.new_page(viewport={"width": 1440, "height": 900})
    errors: list = []
    page.on("pageerror", lambda e: errors.append(str(e)))
    page.errors = errors
    if setup:
        setup(page)   # e.g. routes, which must be in place before the page's first request
    page.goto(url)
    page.wait_for_function("window.__wb && window.__wb.liveState && window.__wb.positionToken", timeout=30000)
    _answer_mode_dialog(page)
    return page


def _open_model_chooser(page: Any) -> None:
    """The model's source controls and the search radios are the model & search section, which
    the technical bar's model · mode label (#btn-model-choose) opens in #model-mode-dialog; open it
    before driving them."""
    if page.locator("#model-mode-dialog.hidden").count():
        page.click("#btn-model-choose")
        page.wait_for_selector("#model-mode-dialog:not(.hidden)")


def _close_model_chooser(page: Any) -> None:
    """The dialog is modal: close it before clicking the board."""
    if not page.locator("#model-mode-dialog.hidden").count():
        page.click("#model-mode-dialog-close")
        page.wait_for_selector("#model-mode-dialog", state="hidden")


def _set_search(page: Any, search: str) -> None:
    """How the model plays ("rollout" / "policy"), from the model & search dialog: at once."""
    _open_model_chooser(page)
    page.check(f"#mode-opponent-{search}")
    _close_model_chooser(page)


def _search(page: Any) -> str:
    """How the model plays, as the section's radios say (outside the play-mode question, the setting in force)."""
    return "rollout" if page.is_checked("#mode-opponent-rollout") else "policy"


def _pick_local_model(page: Any) -> None:
    _open_model_chooser(page)
    page.select_option("#analysis-source-select", "local")
    page.wait_for_function(f"[...document.querySelectorAll('#analysis-run-select option')].some(o => o.value === '{RUN}')")
    page.select_option("#analysis-run-select", RUN)
    page.wait_for_selector("#analysis-body .analysis-choice", timeout=120000)
    _close_model_chooser(page)


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


def test_the_page_searches_with_the_rollout_root_as_python_does(browser: Any, server: str, checkpoint: Dict[str, str]) -> None:
    """The search select's rollout root: the page's loop over the engine's batch (analysis/rollout.ts,
    in a worker) against the same loop in Python with torch (ai/search/rollout_root.py), on the same
    position, the same candidates, the same worlds and the same seed. The engine's part is identical
    by construction; the networks differ only in onnxruntime's and torch's last bits."""
    spec = "rollout:3:2:1:z2:earlier:own"
    page = _open(browser, server + f"/?search={spec}")
    _pick_local_model(page)
    assert _search(page) == "rollout"
    assert page.inner_text("#model-mode-label-mode") == "Rollout root"
    page.wait_for_function("window.__wb.liveAnalysis && window.__wb.liveAnalysis.search", timeout=120000)
    a = page.evaluate("window.__wb.liveAnalysis")
    s = a["search"]
    assert s["kind"] == "rollout" and s["spec"] == {"k": 3, "worlds": 2, "horizon": 1, "rule": "z2", "stop": "earlier", "leaf": "own",
                                                   "keyed": False, "keyed2": False, "aligned": False, "box4": False,
                                                   "belief": 0, "race": 0}
    assert s["pick"] in s["candidates"] and s["candidates"][0] == a["policy"]["argmax_idx"]
    assert s["worlds"] == 2 and len(s["q"]) == len(s["candidates"]) == 3
    assert parse_qs(page.url.split("?", 1)[1]).get("search") == [spec]
    assert "ROLLOUT ROOT" in page.inner_text("#analysis-body")

    # The same search in Python, from the same position with the same seed.
    state = ts.state_from_save_json(page.evaluate("window.__wb.engine.saveJson()"))
    seed = (page.evaluate("window.__wb.state.step_index") + 1) * 1_000_003
    model = NeuralAgent.from_checkpoint(checkpoint["path"], device="cpu").model
    root = RolloutRoot(model, RolloutConfig(k=3, worlds=2, horizon=1, rule="z2", stop="earlier", leaf_side="own"))
    root.reseed(0)
    # RolloutRoot draws the batch's seed from its own stream; drive the batch directly instead.
    p = state.ctx().decision_player
    obs = np.asarray(ts.extract_observation(state, p), dtype=np.float32)[None]
    mask = np.asarray(ActionEncoder.get_legal_mask(state), dtype=np.uint8)
    with torch.no_grad():
        logits = model(torch.from_numpy(obs), torch.from_numpy(mask[None]))[0].float().numpy()[0]
    cands = ts.rollout_candidates(np.ascontiguousarray(logits), mask, 3)
    assert cands == s["candidates"], "the page and Python rank the root's moves differently"
    batch = ts.RolloutBatch(horizon=1, shared_dice=True, guard=True, features=0, seed=seed, stop="earlier", leaf="own")
    batch.add(state, cands, 2)
    root._playouts(batch)
    d = ts.rollout_decide(batch, 0, cands, np.ascontiguousarray(logits), mask, "z2")
    assert d["worlds"] == s["worlds"]
    # onnxruntime and torch differ in the last bits, and a playout's greedy move can flip on a
    # near-tie, so the values agree to a tolerance rather than exactly.
    assert max(abs(x - y) for x, y in zip(d["q"], s["q"])) < 0.05

    # The favourite is now the search's pick, and Play favourite plays it. The next position's search
    # is held at a gate, so it is still running when the select goes back to the policy -- the order
    # a slow machine produces by itself, built here rather than left to timing.
    page.evaluate("""() => { const r = window.__wb.runner, search = r.search.bind(r);
        r.search = (...args) => search(...args).then(res => new Promise(done => { window.__release = () => done(res); })); }""")
    n = page.evaluate("window.__wb.state.step_index")
    page.click("#btn-play-favourite")
    page.wait_for_function(f"window.__wb.state.step_index > {n}")
    page.wait_for_function("!!(window.__wb.liveAnalysis && window.__wb.liveAnalysis.search_pending) "
                           "&& typeof window.__release === 'function'")
    # Back to the policy: the section goes, and the address no longer names a search.
    _set_search(page, "policy")
    assert page.inner_text("#model-mode-label-mode") == "Policy"
    page.wait_for_function("!new URLSearchParams(location.search).has('search')")
    page.wait_for_function("window.__wb.liveAnalysis && window.__wb.liveAnalysis.policy && !window.__wb.liveAnalysis.search")
    # The search asked for under the old setting answers now: it must not bring its section back.
    page.evaluate("async () => { window.__release(); await new Promise(r => setTimeout(r, 0)); }")
    assert "ROLLOUT ROOT" not in page.inner_text("#analysis-body")
    assert not page.evaluate("window.__wb.liveAnalysis.search")
    assert not page.errors


def test_the_page_searches_with_the_belief_root(browser: Any, server: str) -> None:
    """The leaderboard's belief root (`:keyed:belief<N>`, bindings/belief.hpp) in the page: the search
    worker's tracker is fed the game's moves (rollout_runner.ts) and its searches deal their worlds from
    the belief -- a result carries the temperature weights it settled on -- through Play favourite and an
    undo (which replays the moves into the tracker)."""
    spec = "rollout:2:2:0:z2:earlier:own:keyed:belief16"
    page = _open(browser, server + f"/?search={spec}")
    _pick_local_model(page)
    weighed = 0
    for i in range(14):
        page.wait_for_function("window.__wb.liveAnalysis && (window.__wb.liveAnalysis.search "
                               "|| window.__wb.liveAnalysis.search_error)", timeout=120000)
        a = page.evaluate("window.__wb.liveAnalysis")
        assert not a.get("search_error"), a.get("search_error")
        s = a["search"]
        assert s["spec"]["keyed"] is True and s["spec"]["belief"] == 16
        w = s.get("temperature_weights")
        if w:
            assert len(w) == 6 and abs(sum(w) - 1.0) < 1e-4
            weighed += 1
        n = page.evaluate("window.__wb.state.step_index")
        if i == 8:
            page.click("#btn-undo-action")
            page.wait_for_function(f"window.__wb.state.step_index < {n}")
            continue
        page.click("#btn-play-favourite")
        page.wait_for_function(f"window.__wb.state.step_index > {n}")
    assert weighed > 0, "no search was dealt from the belief"
    assert not page.errors


def test_the_live_readout_highlights_the_side_to_move(browser: Any, server: str) -> None:
    """The critic is trained only on the decider's observation: the live panel shows each side's
    own P(victory) and expected VP, lights the decider's and dims the other's, and follows the
    decider as the game moves from the USSR setup to the US setup."""
    page = _open(browser, server + "/")
    _pick_local_model(page)
    for side in ("USSR", "US"):
        for _ in range(12):
            if page.evaluate("window.__wb.state.decision_context.decision_player") == side:
                break
            n = page.evaluate("window.__wb.state.step_index")
            page.wait_for_function("window.__wb.liveAnalysis && window.__wb.liveAnalysis.policy")
            page.click("#btn-play-favourite")
            page.wait_for_function(f"window.__wb.state.step_index > {n}")
        page.wait_for_function(
            "d => window.__wb.liveAnalysis && window.__wb.liveAnalysis.decision_player === d"
            " && document.querySelector('#analysis-body .value-side.calibrated')?.dataset.side === d",
            arg=side, timeout=60000)
        a = page.evaluate("window.__wb.liveAnalysis")
        body = page.locator("#analysis-body")
        other = "US" if side == "USSR" else "USSR"
        assert body.locator(".value-side.calibrated").count() == 1
        assert body.locator(f".value-side.uncalibrated[data-side='{other}']").count() == 1

        # Each side: its own P(victory) = (1 + v_win)/2 and expected VP = v_vp x 20.
        for s in ("US", "USSR"):
            box = body.locator(f".value-side[data-side='{s}']")
            k = s.lower()
            assert abs(float(box.get_attribute("data-p")) - (1 + a["critic"][f"v_win_{k}"]) / 2) < 1e-5
            vp = a["critic"][f"v_vp_{k}"] * 20
            assert abs(float(box.get_attribute("data-vp")) - vp) < 1e-3
            assert box.locator(".value-side-vp").inner_text() == f"{'+' if vp >= 0 else '−'}{abs(vp):.1f} VP"
        text = body.inner_text().lower()
        assert "trained" not in text and "v_win" not in text, "only the two numbers per side"
    assert not page.errors


def test_a_countrys_probability_never_covers_its_influence(browser: Any, server: str) -> None:
    """The badge sits in the corner right of the influence boxes, inside the country's card."""
    page = _open(browser, server + "/")
    _pick_local_model(page)   # the USSR setup: every Eastern European country is a choice
    page.wait_for_selector(".svg-country-node rect.trace-choice-badge")
    report = page.evaluate("""() => [...document.querySelectorAll('.svg-country-node')]
        .filter(g => g.querySelector('rect.trace-choice-badge'))
        .map(g => {
          const bb = e => { const b = e.getBBox(); return [b.x, b.y, b.x + b.width, b.y + b.height]; };
          return { name: g.dataset.name, badge: bb(g.querySelector('rect.trace-choice-badge')),
                   text: bb(g.querySelector('text.trace-choice-badge')),
                   card: bb(g.querySelector('rect.country-card-bg')),
                   inf: [...g.querySelectorAll('rect.inf-box')].map(bb) };
        })""")
    assert len(report) >= 6, report
    overlap = lambda a, b: a[0] < b[2] and b[0] < a[2] and a[1] < b[3] and b[1] < a[3]
    for c in report:
        b, card = c["badge"], c["card"]
        assert card[0] <= b[0] and b[2] <= card[2] and card[1] <= b[1] and b[3] <= card[3], c
        assert len(c["inf"]) == 2 and not any(overlap(b, i) for i in c["inf"]), c
        # The number itself fits its box: the HTML badge's CSS once reached the SVG text too and
        # drew it at 10 map units, spilling over the influence while the box sat where it should.
        t = c["text"]
        assert b[0] - 0.2 <= t[0] and t[2] <= b[2] + 0.2 and b[1] - 0.2 <= t[1] and t[3] <= b[3] + 0.2, c
    assert not page.errors


def test_the_cards_panel_is_its_own_column_left_of_the_map(browser: Any, server: str) -> None:
    page = _open(browser, server + "/")
    assert page.locator("#cards-sidebar #hands-panel .tabs .tab-btn").count() == 4
    assert page.locator(".sidebar #hands-panel").count() == 0, "the cards left the right rail"
    cards = page.locator("#cards-sidebar").bounding_box()
    board = page.locator("#map-container").bounding_box()
    rail = page.locator(".sidebar").bounding_box()
    assert cards and board and rail
    assert cards["x"] + cards["width"] <= board["x"] + 1 and board["x"] + board["width"] <= rail["x"] + 1

    # The switch still switches; the "all" list names each card and where it is, and nothing more.
    page.click("[data-tab='tab-all-cards']")
    assert page.locator("#tab-all-cards").is_visible() and not page.locator("#tab-ussr-hand").is_visible()
    assert page.locator("#all-cards-list .all-cards-item").count() == 110
    assert page.locator("#all-cards-list .card-meta").count() == 0
    page.click("[data-tab='tab-us-hand']")
    assert page.locator("#us-cards-list .card-item").count() > 0

    # Narrow: one column, map first, no sideways overflow of the board area.
    page.set_viewport_size({"width": 800, "height": 900})
    page.wait_for_timeout(200)
    board = page.locator("#map-container").bounding_box()
    cards = page.locator("#cards-sidebar").bounding_box()
    assert board and cards and cards["y"] >= board["y"] + board["height"] - 1
    assert page.evaluate("(() => { const m = document.querySelector('.main-content');"
                         " return m.scrollWidth <= m.clientWidth + 1; })()")
    assert not page.errors


def test_the_piles_tab_lists_each_pile_as_the_hands_do_by_side_then_ops(browser: Any, static_site: str) -> None:
    """The Piles tab draws the discards and the removed cards as the hand tabs draw a hand --
    packed, no gap between them -- by side (USSR, US, Neutral) and then highest Ops first."""
    page = _open(browser, static_site + "/?model=off")
    with open(os.path.join(REPO, "rules", "cards.json")) as f:
        meta = {c["id"]: c for c in json.load(f)}
    neutral = [i for i, c in meta.items() if c["side"] == "neutral" and i != 6][:2]
    discard = [103, 8, 4, *neutral, 33, 21]
    page.evaluate(f"(() => {{ const wb = window.__wb; wb.cardsView.render({{...wb.shown, discard_pile: {discard}, removed_pile: [] }}); }})()")
    page.click("[data-tab='tab-board-cards']")
    order = {"ussr": 0, "us": 1, "neutral": 2}
    want = [f"#{i} {meta[i]['name']}" for i in sorted(discard, key=lambda i: (order[meta[i]["side"]], -meta[i]["ops"], meta[i]["name"]))]
    rows = page.locator("#discard-cards-list .card-item")
    assert [t.strip() for t in rows.locator(".card-name-single").all_inner_texts()] == want
    assert page.inner_text("#discard-count") == str(len(discard))
    assert page.inner_text("#removed-cards-list").strip() == "Empty"
    boxes = [rows.nth(k).bounding_box() for k in range(rows.count())]
    assert all(a and b and abs(a["y"] + a["height"] - b["y"]) < 1.5 for a, b in zip(boxes, boxes[1:])), "the pile is not packed"
    # No probability badge lands on a pile: only the hands carry data-card-id.
    assert page.locator("#discard-cards-list [data-card-id]").count() == 0
    rows.first.hover()
    assert page.is_visible("#card-tooltip") and meta[33]["name"] in page.inner_text("#card-tooltip")
    assert not page.errors


def test_a_card_named_in_the_log_shows_its_text_on_hover(browser: Any, static_site: str) -> None:
    page = _open(browser, static_site + "/?model=off")
    page.evaluate("""() => { const wb = window.__wb;
        for (let n = 0; n < 5000 && !wb.state.is_terminal && wb.state.current_phase_name !== 'ACTION_ROUND'; n++) {
            const m = wb.engine.mask(false); wb.sendFlatAction(m.findIndex(x => x), 0, true);
        } }""")
    ref = page.locator("#action-log-stream .log-card[data-card-id]").first
    ref.wait_for()
    cid = int(ref.get_attribute("data-card-id") or "0")
    name = page.evaluate(f"window.__wb.engine.cardName({cid})")
    assert name in ref.inner_text()
    ref.hover()
    tip = page.inner_text("#card-tooltip")
    assert page.is_visible("#card-tooltip") and name in tip and "OPS" in tip
    page.mouse.move(5, 5)
    assert not page.is_visible("#card-tooltip")
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
    _open_model_chooser(page)
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
    _open_model_chooser(page)
    page.select_option("#analysis-source-select", "file")
    page.set_input_files("#analysis-file-input", str(bogus))
    page.wait_for_selector("#analysis-body .analysis-error", timeout=30000)
    assert page.inner_text("#analysis-status") == "error"


@pytest.fixture(scope="module")
def onnx_bytes(checkpoint: Dict[str, str], tmp_path_factory: pytest.TempPathFactory) -> bytes:
    from tools.export_onnx import export

    path = str(tmp_path_factory.mktemp("hf_onnx") / "model.onnx")
    export(checkpoint["path"], path, positions=16)
    with open(path, "rb") as f:
        return f.read()


def _fake_hf_repo(page: Any, onnx: bytes, requests: List[str], sha256: Optional[str] = None,
                  default_model: Optional[str] = None, manifest: Optional[Dict[str, Any]] = None) -> None:
    """The default repo, served by the test: two listing pages, the newest upload on the second.

    Every request to huggingface.co is recorded. The listing is paged the way the real API pages
    a tree -- a `Link: <...>; rel="next"` header -- so a page that read only the first page would
    load `old.onnx`. As on the real API, a plain tree entry carries no date: `paths-info` with
    `expand=true` gives each file's last commit. `paths-info` reports every file's LFS sha256 as
    `sha256`, by default the real one of `onnx`. `models.json` is `manifest`, `default.json` names
    `default_model`; each is missing (404, as Hugging Face answers for a file a repo does not
    hold) when not given.
    """
    lfs_oid = sha256 or hashlib.sha256(onnx).hexdigest()
    cors = {"access-control-allow-origin": "*", "access-control-expose-headers": "Link"}
    dates = {"README.md": "2026-09-24T10:00:00.000Z", "old.onnx": "2026-09-01T12:00:00.000Z",
             "runs/new.onnx": "2026-09-25T19:46:55.000Z"}
    first = [{"type": "file", "path": "README.md"}, {"type": "file", "path": "old.onnx"},
             {"type": "directory", "path": "runs"}]
    second = [{"type": "file", "path": "runs/new.onnx"}]

    def tree(route: Any) -> None:
        requests.append(route.request.url)
        if "/paths-info/" in route.request.url:
            form = parse_qs(route.request.post_data or "")
            expand = form.get("expand") == ["true"]
            body = [{"type": "file", "path": p, "lfs": {"oid": lfs_oid, "size": len(onnx)},
                     **({"lastCommit": {"date": dates[p]}} if expand else {})} for p in form["paths"]]
            route.fulfill(status=200, headers=cors, content_type="application/json", body=json.dumps(body))
            return
        more = "cursor=" not in route.request.url
        headers = {**cors, "link": f'<{HF_TREE}&cursor=p2>; rel="next"'} if more else cors
        route.fulfill(status=200, headers=headers, content_type="application/json",
                      body=json.dumps(first if more else second))

    def json_file(route: Any, doc: Optional[Any]) -> None:
        if doc is None:
            route.fulfill(status=404, headers=cors, body="Entry not found")
        else:
            route.fulfill(status=200, headers=cors, content_type="application/json", body=json.dumps(doc))

    def resolve(route: Any) -> None:
        requests.append(route.request.url)
        if route.request.url.endswith("/models.json"):
            json_file(route, manifest)
        elif route.request.url.endswith("/default.json"):
            json_file(route, None if default_model is None else {"model": default_model})
        else:
            route.fulfill(status=200, headers=cors, body=onnx)

    page.route("https://huggingface.co/api/models/**", tree)
    page.route("https://huggingface.co/mihaild/deepstruggle/resolve/**", resolve)


def test_a_link_that_names_no_model_loads_the_newest_upload(browser: Any, static_site: str, onnx_bytes: bytes) -> None:
    """GitHub Pages: no local server, no model in the link -- the newest .onnx in the default repo."""
    requests: List[str] = []
    page = _open(browser, static_site + "/", lambda p: _fake_hf_repo(p, onnx_bytes, requests))
    page.wait_for_selector("#analysis-body .analysis-choice", timeout=60000)
    assert page.input_value("#analysis-source-select") == "hf"
    assert page.input_value("#analysis-hf-repo") == "mihaild/deepstruggle"
    assert page.input_value("#analysis-hf-file") == "runs/new.onnx"
    options = page.eval_on_selector_all("#analysis-hf-file option", "os => os.map(o => o.value).filter(Boolean)")
    assert options == ["runs/new.onnx", "old.onnx"], "newest upload first, and only .onnx files"
    assert requests[-1] == "https://huggingface.co/mihaild/deepstruggle/resolve/main/runs/new.onnx"
    page.wait_for_function("new URLSearchParams(location.search).get('model') === 'hf:mihaild/deepstruggle@main:runs/new.onnx'")
    groups = page.eval_on_selector_all("#analysis-hf-file optgroup", "gs => gs.map(g => g.label)")
    assert groups == ["runs"], "a directory (a published run) is one group; top-level files are not grouped"
    # Without a manifest: the plain tree (no expand, which pages 50 at a time), then one
    # paths-info for the dates of the .onnx files alone.
    listing = [r for r in requests if "/tree/" in r]
    assert len(listing) == 2 and not [r for r in listing if "expand" in r]
    assert not page.errors


def test_a_repo_default_json_names_the_default_model(browser: Any, static_site: str, onnx_bytes: bytes) -> None:
    """A repo of whole published runs has several .onnx files per run, so "newest upload" is
    whatever was uploaded last; `default.json` names the model a bare link gets instead."""
    requests: List[str] = []
    page = _open(browser, static_site + "/",
                 lambda p: _fake_hf_repo(p, onnx_bytes, requests, default_model="old.onnx"))
    page.wait_for_selector("#analysis-body .analysis-choice", timeout=60000)
    assert page.input_value("#analysis-hf-file") == "old.onnx"
    assert requests[-1] == "https://huggingface.co/mihaild/deepstruggle/resolve/main/old.onnx"
    assert not page.errors


def test_a_repo_with_models_json_is_not_listed(browser: Any, static_site: str, onnx_bytes: bytes) -> None:
    """tools/publish_hf.py writes models.json on every publish: the page reads that one file
    instead of listing the tree (one request per 50 files with dates), and takes its default."""
    requests: List[str] = []
    manifest = {"schema": 1, "default": "runs/new.onnx", "models": [
        {"path": "runs/new.onnx", "date": "2026-09-25T19:46:55.000Z", "sha256": "", "size": 0},
        {"path": "runs/mid.onnx", "date": "2026-09-20T00:00:00.000Z", "sha256": "", "size": 0},
        {"path": "old.onnx", "date": "2026-09-01T12:00:00.000Z", "sha256": "", "size": 0}]}
    page = _open(browser, static_site + "/",
                 lambda p: _fake_hf_repo(p, onnx_bytes, requests, manifest=manifest))
    page.wait_for_selector("#analysis-body .analysis-choice", timeout=60000)
    assert page.input_value("#analysis-hf-file") == "runs/new.onnx"
    options = page.eval_on_selector_all("#analysis-hf-file option", "os => os.map(o => o.value).filter(Boolean)")
    assert options == ["runs/new.onnx", "runs/mid.onnx", "old.onnx"]
    assert not [r for r in requests if "/tree/" in r], "a repo with a manifest is never listed"
    assert not [r for r in requests if r.endswith("/default.json")]
    assert not page.errors


def test_the_leaderboard_page_filters_one_fit(browser: Any, static_site: str) -> None:
    """leaderboard.html shows the leaderboard.json that build_web.sh fitted from leaderboard/: the
    newest epoch with games in it (a new engine epoch starts with its anchor alone), the main players
    by default, every player under "All", each with the same rating in both views; a row opens its
    head-to-head results, and the view is kept in the link."""
    with open(os.path.join(DIST, "leaderboard.json")) as f:
        data = json.load(f)
    with open(os.path.join(REPO, "leaderboard", "epochs.json")) as f:
        names = list(json.load(f))
    epoch = [n for n in names if data["epochs"][n]["pairs"]][-1]
    ep = data["epochs"][epoch]
    page = browser.new_page(viewport={"width": 1280, "height": 900})
    errors: List[str] = []
    page.on("pageerror", lambda e: errors.append(str(e)))
    if epoch != names[-1]:
        # An epoch with no games yet still opens when the link names it.
        page.goto(static_site + f"/leaderboard.html?epoch={names[-1]}")
        page.wait_for_selector("#lb-body tr")
        assert page.input_value("#lb-epoch") == names[-1]
    page.goto(static_site + "/leaderboard.html")
    page.wait_for_selector("#lb-body tr")
    assert page.input_value("#lb-epoch") == epoch

    def shown() -> Dict[str, str]:
        rows = page.eval_on_selector_all(
            "#lb-body tr", "rs => rs.map(r => [r.dataset.player, r.querySelector('.lb-elo').textContent])")
        return dict(rows)

    mains = shown()
    assert list(mains) == [r["player"] for r in ep["ratings"] if r["main"]]
    page.check("input[name=lb-view][value=all]")
    every = shown()
    assert list(every) == [r["player"] for r in ep["ratings"]]
    assert all(every[p] == mains[p] for p in mains), "a view filters the fit; it never refits"
    page.click(f"#lb-body tr[data-player='{ep['anchor']}']")
    assert page.inner_text("#lb-detail h2") == ep["anchor"]
    anchor = next(r for r in ep["ratings"] if r["player"] == ep["anchor"])
    assert page.eval_on_selector_all("#lb-detail tbody tr", "rs => rs.length") == anchor["opponents"]
    assert "view=all" in page.url
    # The codes tab: the recipes and architectures of the networks on the board, from the same fit; the
    # selected player's codes link to their rows.
    pid = next(p for p, pl in data["players"].items() if pl["network"] and data["codes"]["by_network"][pl["network"]]["recipe"]
               and p != ep["anchor"] and any(r["player"] == p for r in ep["ratings"]))
    page.click(f"#lb-body tr[data-player='{pid}']")
    recipe = data["codes"]["by_network"][data["players"][pid]["network"]]["recipe"]
    page.click(f"#lb-detail a[data-code='{recipe}']")
    assert page.is_visible("#lb-tab-codes") and not page.is_visible("#lb-tab-ratings")
    assert page.is_visible(f"#code-{recipe}")
    used = [c for c, v in data["codes"]["recipes"].items() if v["networks"]]
    assert page.eval_on_selector_all("#lb-tab-codes table.lb-codes", "ts => ts[0].tBodies[0].rows.length") == len(used)
    page.check("#lb-codes-all")
    assert page.eval_on_selector_all("#lb-tab-codes table.lb-codes", "ts => ts[0].tBodies[0].rows.length") == len(data["codes"]["recipes"])
    assert "tab=codes" in page.url and "codes=all" in page.url
    assert not errors


def test_the_leaderboard_switches_to_the_playdek_estimate(browser: Any, static_site: str) -> None:
    """The estimate comes with the build only if it could read the stats database, so a synthetic one
    is served here. No estimate: no switch. With one: its numbers, in its own order, kept in the link."""
    data, epoch, rated = _leaderboard_data()
    data.pop("playdek", None)
    players = sorted(rated)
    estimate = {p: {"rating": 1000.0 + 10 * k, "se": 40.0, "lo": 900.0, "hi": 1100.0, "games": k, "humans": 1,
                    "w": k, "d": 0, "l": 0} for k, p in enumerate(players)}
    with_estimate = {**data, "playdek": {"as_of": "2026-10-31", "games": 7, "humans": 3, "excluded": {}, "offset": 1.0,
                                         "deviation_sd": 100.0, "reference_epoch": epoch, "players": estimate}}
    served = {"body": json.dumps(data)}
    page = browser.new_page(viewport={"width": 1280, "height": 900})
    errors: List[str] = []
    page.on("pageerror", lambda e: errors.append(str(e)))
    page.route("**/leaderboard.json", lambda route: route.fulfill(content_type="application/json", body=served["body"]))
    page.goto(static_site + f"/leaderboard.html?epoch={epoch}&view=all&scale=playdek")
    page.wait_for_selector("#lb-body tr")
    assert not page.is_visible("#lb-scale-mode")
    assert page.text_content("#lb-th-elo") == "Elo" and "scale=playdek" not in page.url

    served["body"] = json.dumps(with_estimate)
    page.goto(static_site + f"/leaderboard.html?epoch={epoch}&view=all")
    page.wait_for_selector("#lb-body tr")
    assert page.is_visible("#lb-scale-mode") and page.text_content("#lb-th-elo") == "Elo"
    page.check("input[name=lb-mode][value=playdek]")
    assert page.text_content("#lb-th-elo") == "Playdek est."
    rows = page.eval_on_selector_all(
        "#lb-body tr", "rs => rs.map(r => [r.dataset.player, r.querySelector('.lb-elo').textContent])")
    assert rows == [[p, f"{estimate[p]['rating']:.0f}"] for p in reversed(players)]
    assert "7 clean games against 3 people" in page.inner_text("#lb-note")
    assert "scale=playdek" in page.url
    page.click(f"#lb-body tr[data-player='{players[-1]}']")
    assert "Playdek estimate" in page.inner_text("#lb-detail")
    page.reload()
    page.wait_for_selector("#lb-body tr")
    assert page.is_checked("input[name=lb-mode][value=playdek]"), "the scale is kept in the link"
    page.check("input[name=lb-mode][value=anchor]")
    assert page.text_content("#lb-th-elo") == "Elo" and "scale=playdek" not in page.url
    assert not errors


#: The workbench's built-in default search (analysis/rollout_spec.ts DEFAULT_ROLLOUT), spelled out: what a
#: page that cannot read the repo's default plays.
BUILTIN_SEARCH = "rollout:4:32:4:z2:earlier:own:keyed:aligned:box4"


def _leaderboard_data() -> Tuple[Dict[str, Any], str, Set[str]]:
    """leaderboard.json as built, the epoch the page opens on, and the players rated in it."""
    with open(os.path.join(DIST, "leaderboard.json")) as f:
        data = json.load(f)
    with open(os.path.join(REPO, "leaderboard", "epochs.json")) as f:
        names = list(json.load(f))
    epoch = [n for n in names if data["epochs"][n]["pairs"]][-1]
    return data, epoch, {r["player"] for r in data["epochs"][epoch]["ratings"]}


def test_the_leaderboard_tags_the_hugging_face_default_player(browser: Any, static_site: str) -> None:
    """The player that is the repo's default -- its default model played with its default search
    (models.json, read when the page opens) -- carries the HF DEFAULT tag, and only it; with no default
    search the workbench's built-in one stands. A default that is no player tags nothing and the footer
    says so; a repo the page cannot read tags nothing and breaks nothing. Every play link opens its
    player's search."""
    data, epoch, rated = _leaderboard_data()
    players = data["players"]
    pid = next(p for p in sorted(rated) if (players[p]["workbench_search"] or "").startswith("rollout")
               and data["networks"][players[p]["network"]]["hf"])
    net = players[pid]["network"]
    onnx = data["networks"][net]["hf"].removesuffix(".pt") + ".onnx"
    on_net = {p for p in rated if players[p]["network"] == net}

    def opened(manifest: Optional[Dict[str, Any]], status: int = 200) -> Any:
        page = browser.new_page(viewport={"width": 1280, "height": 900})
        errors: List[str] = []
        page.on("pageerror", lambda e: errors.append(str(e)))
        body = json.dumps({"schema": 1, "models": [], **(manifest or {})})
        page.route("https://huggingface.co/mihaild/deepstruggle/resolve/main/models.json",
                   lambda route: route.fulfill(status=status, content_type="application/json", body=body))
        page.goto(static_site + f"/leaderboard.html?epoch={epoch}&view=all")
        page.wait_for_selector("#lb-body tr")
        page.wait_for_function("document.getElementById('lb-footer').textContent.includes('Weights')")
        page.errors = errors  # type: ignore[attr-defined]
        return page

    def tagged(page: Any) -> Set[str]:
        return set(page.eval_on_selector_all("#lb-body tr:has(.lb-tag.hf)", "rs => rs.map(r => r.dataset.player)"))

    page = opened({"default": onnx, "default_search": players[pid]["workbench_search"]})
    page.wait_for_selector("#lb-body .lb-tag.hf")
    assert tagged(page) == {pid}
    assert onnx in page.inner_text("#lb-footer") and players[pid]["workbench_search"] in page.inner_text("#lb-footer")
    page.click(f"#lb-body tr[data-player='{pid}']")
    assert "The default on Hugging Face" in page.inner_text("#lb-detail")
    # Every play link: the player's network and its search as the workbench plays it.
    links = dict(page.eval_on_selector_all(
        "#lb-body tr", "rs => rs.map(r => [r.dataset.player, (r.querySelector('.lb-links a[title*=\"workbench\"]') || {}).href || ''])"))
    for p, href in links.items():
        if not href:
            continue
        q = parse_qs(href.split("?", 1)[1])
        assert q["search"] == [players[p]["workbench_search"] or "policy"], p
        assert q["model"][0].endswith(data["networks"][players[p]["network"]]["hf"].removesuffix(".pt") + ".onnx")
    assert not page.errors
    page.close()

    # The policy as the default search: the network's greedy player.
    page = opened({"default": onnx, "default_search": "policy"})
    greedy = {p for p in on_net if players[p]["workbench_search"] == "policy"}
    if greedy:
        page.wait_for_selector("#lb-body .lb-tag.hf")
    assert tagged(page) == greedy
    page.close()

    # No default search: the built-in one.
    page = opened({"default": onnx})
    builtin = {p for p in on_net if players[p]["workbench_search"] == BUILTIN_SEARCH}
    page.wait_for_function("document.getElementById('lb-footer').textContent.includes('default model')")
    assert tagged(page) == builtin
    page.close()

    page = opened({"default": "_models/not-registered.onnx", "default_search": "policy"})
    page.wait_for_function("document.getElementById('lb-footer').textContent.includes('not a player on the board')")
    assert not tagged(page) and not page.errors
    page.close()

    page = opened(None, status=500)
    assert not tagged(page) and "default model" not in page.inner_text("#lb-footer")
    assert not page.errors
    page.close()


def test_every_leaderboard_search_reads_in_the_workbench_as_python_wrote_it(browser: Any, static_site: str) -> None:
    """tools/lib/player_spec.workbench_search writes each player's search; the page parses it
    (analysis/rollout_spec.ts). Every one on the board must come back as the same spec, field by field
    -- a field the two grammars default differently would otherwise change the player."""
    data, _, _ = _leaderboard_data()
    searches: Dict[str, Dict[str, Any]] = {}
    for pl in data["players"].values():
        ws = pl["workbench_search"]
        if ws and ws.startswith("rollout") and ws not in searches:
            spec = spec_from_dict(pl["inference"])
            assert isinstance(spec, Rollout)
            searches[ws] = {"k": spec.k, "worlds": spec.worlds, "horizon": spec.horizon, "rule": spec.rule,
                            "stop": spec.stop, "leaf": spec.leaf, "keyed": spec.chance == "keyed",
                            "keyed2": spec.chance == "keyed2", "aligned": spec.aligned, "box4": spec.box4, "belief": spec.belief,
                            "race": spec.race}
    assert searches
    for ws, want in searches.items():
        page = _open(browser, static_site + f"/?model=off&view=us&search={quote(ws)}")
        assert page.evaluate("window.__wb.searchSpec") == want, ws
        assert parse_qs(page.url.split("?", 1)[1])["search"] == [ws], "the page writes it back as it came"
        assert not page.errors
        page.close()


def test_a_fresh_open_offers_the_repos_default_search(browser: Any, static_site: str) -> None:
    """models.json's default_search is what a fresh open offers and the search settings show; choosing
    a side plays with it, and the link then names it. (The repo here holds no model: the page says so,
    and the search stands.)"""
    default = "rollout:2:1:1:z2:earlier:own"
    page = browser.new_page(viewport={"width": 1366, "height": 768})
    errors: List[str] = []
    page.on("pageerror", lambda e: errors.append(str(e)))
    body = json.dumps({"schema": 1, "default": None, "default_search": default, "models": []})
    page.route("https://huggingface.co/mihaild/deepstruggle/resolve/main/models.json",
               lambda route: route.fulfill(status=200, content_type="application/json", body=body))
    page.goto(f"{static_site}/")
    page.wait_for_function("window.__wb && window.__wb.liveState && window.__wb.positionToken", timeout=30000)
    page.wait_for_selector("#mode-dialog:not(.hidden)")
    page.wait_for_function(f"document.getElementById('search-settings-code').textContent === '{default}'")
    assert page.is_checked("#mode-opponent-rollout")
    page.click("#mode-choice-us")
    page.wait_for_selector("#mode-dialog", state="hidden")
    page.wait_for_function(f"new URLSearchParams(location.search).get('search') === '{default}'")
    assert not errors
    page.close()


def test_the_search_settings_set_the_search(browser: Any, static_site: str) -> None:
    """The search settings under the radios edit the rollout root's spec: a change is the setting (the
    Rollout root chosen, the link naming it), one that does not parse is refused with a note, and
    Default goes back to the default search."""
    page = _open(browser, static_site + "/?model=off&view=us&search=policy")
    assert page.evaluate("window.__wb.searchSpec") is None
    _open_model_chooser(page)
    page.click("#search-settings summary")
    page.fill("#ss-worlds", "8")
    page.dispatch_event("#ss-worlds", "change")
    eight = BUILTIN_SEARCH.replace(":32:", ":8:")
    page.wait_for_function(f"new URLSearchParams(location.search).get('search') === '{eight}'")
    assert page.is_checked("#mode-opponent-rollout")
    assert page.inner_text("#search-settings-code") == eight
    # Edited settings are kept through a turn on the policy (a link's spec is not: the default comes back).
    page.check("#mode-opponent-policy")
    page.wait_for_function("new URLSearchParams(location.search).get('search') === null")
    assert page.inner_text("#search-settings-code") == eight
    page.check("#mode-opponent-rollout")
    page.wait_for_function(f"new URLSearchParams(location.search).get('search') === '{eight}'")
    page.fill("#ss-rule", "zz")
    page.dispatch_event("#ss-rule", "change")
    assert page.inner_text("#ss-error")
    assert parse_qs(page.url.split("?", 1)[1])["search"] == [eight], "a spec that does not parse changes nothing"
    page.click("#ss-default")
    page.wait_for_function(f"new URLSearchParams(location.search).get('search') === '{BUILTIN_SEARCH}'")
    assert page.input_value("#ss-rule") == "z2"
    _close_model_chooser(page)
    assert not page.errors
    page.close()


def test_the_search_settings_keep_a_raced_search(browser: Any, static_site: str) -> None:
    """A link's `:race<n>` is a field of the settings: editing another field keeps it, a race that is
    not below the worlds is refused with a note, and 0 turns it off."""
    raced = BUILTIN_SEARCH + ":race16"
    page = _open(browser, static_site + f"/?model=off&view=us&search={quote(raced)}")
    _open_model_chooser(page)
    page.click("#search-settings summary")
    assert page.input_value("#ss-race") == "16"
    assert page.inner_text("#search-settings-code") == raced
    page.fill("#ss-horizon", "2")
    page.dispatch_event("#ss-horizon", "change")
    two = raced.replace(":32:4:", ":32:2:")
    page.wait_for_function(f"new URLSearchParams(location.search).get('search') === '{two}'")
    page.fill("#ss-race", "32")
    page.dispatch_event("#ss-race", "change")
    assert page.inner_text("#ss-error")
    assert parse_qs(page.url.split("?", 1)[1])["search"] == [two], "a race of every world changes nothing"
    page.fill("#ss-race", "0")
    page.dispatch_event("#ss-race", "change")
    page.wait_for_function(f"new URLSearchParams(location.search).get('search') === '{two[:-len(':race16')]}'")
    assert page.inner_text("#ss-error") == ""
    _close_model_chooser(page)
    assert not page.errors
    page.close()


def test_a_downloaded_model_is_kept_for_the_next_visit(browser: Any, static_site: str, onnx_bytes: bytes) -> None:
    """Reopening the page loads the same file from the browser, not Hugging Face -- but a file
    the repo has since replaced (another sha256 under the same name) is downloaded again."""
    resolve = "https://huggingface.co/mihaild/deepstruggle/resolve/main/runs/new.onnx"
    context = browser.new_context(viewport={"width": 1440, "height": 900})
    try:
        first: List[str] = []
        page = _open(browser, static_site + "/", lambda p: _fake_hf_repo(p, onnx_bytes, first), context)
        page.wait_for_selector("#analysis-body .analysis-choice", timeout=60000)
        assert first[-1] == resolve
        assert "cached" not in page.inner_text("#analysis-model-info")
        page.wait_for_function("!document.getElementById('analysis-hf-clear').disabled")
        page.close()

        again: List[str] = []
        page = _open(browser, static_site + "/", lambda p: _fake_hf_repo(p, onnx_bytes, again), context)
        page.wait_for_selector("#analysis-body .analysis-choice", timeout=60000)
        assert resolve not in again, "the kept file was downloaded again"
        assert "cached" in page.inner_text("#analysis-model-info")
        page.close()

        replaced: List[str] = []
        page = _open(browser, static_site + "/",
                     lambda p: _fake_hf_repo(p, onnx_bytes, replaced, sha256="0" * 64), context)
        page.wait_for_selector("#analysis-body .analysis-choice", timeout=60000)
        assert replaced[-1] == resolve, "a file the repo replaced must not come from the cache"
        assert "cached" not in page.inner_text("#analysis-model-info")
        # The bytes do not hash to what the repo claimed, so they are not kept under that name.
        assert page.evaluate("caches.open('ts-models-v1').then(c => c.keys()).then(k => k.length)") == 1

        _open_model_chooser(page)
        page.click("#analysis-hf-clear")
        page.wait_for_function("document.getElementById('analysis-hf-clear').disabled")
        assert page.evaluate("caches.has('ts-models-v1')") is False
        assert not page.errors
    finally:
        context.close()


def test_turning_analysis_off_stays_off_in_the_link(browser: Any, static_site: str, onnx_bytes: bytes) -> None:
    requests: List[str] = []
    page = _open(browser, static_site + "/", lambda p: _fake_hf_repo(p, onnx_bytes, requests))
    page.wait_for_selector("#analysis-body .analysis-choice", timeout=60000)
    _open_model_chooser(page)
    page.select_option("#analysis-source-select", "")
    page.wait_for_function("new URLSearchParams(location.search).get('model') === 'off'")
    link = page.url

    again: List[str] = []
    other = _open(browser, link, lambda p: _fake_hf_repo(p, onnx_bytes, again))
    other.wait_for_timeout(1500)
    assert again == [], "a link that says model=off must not list or download the default model"
    assert other.inner_text("#analysis-status") == "off"
    assert other.input_value("#analysis-source-select") == ""
    assert "model=off" in other.url
    assert not page.errors and not other.errors


def test_an_unreachable_default_repo_is_reported_not_fatal(browser: Any, static_site: str) -> None:
    page = _open(browser, static_site + "/")   # huggingface.co does not resolve in these tests
    page.wait_for_selector("#analysis-body .analysis-error", timeout=30000)
    assert "Could not list mihaild/deepstruggle" in page.inner_text("#analysis-body")
    assert page.evaluate("window.__wb.state.decision_context.decision_type_name") == "POINT_NODE"
    assert not page.errors


def _scoring_tags(page: Any) -> Dict[str, str]:
    return page.eval_on_selector_all(
        ".svg-region-badge",
        "gs => Object.fromEntries(gs.map(g => [g.dataset.region, g.querySelector('.svg-region-card-tag').dataset.status]))")


def test_each_region_says_whether_its_scoring_card_can_still_come(browser: Any, static_site: str) -> None:
    page = _open(browser, static_site + "/?model=off")
    assert _scoring_tags(page) == {
        "EUROPE": "live", "ASIA": "live", "MIDDLE EAST": "live",
        "AFRICA": "later", "CENTRAL AMERICA": "later", "SOUTH AMERICA": "later", "SOUTHEAST ASIA": "later",
    }, "at the start the Early War scoring cards are dealt or in the deck; the Mid War ones are not in yet"
    # The state as shown carries the engine's region scores (main.ts sets them), which is what the
    # badges are drawn from; the raw game state does not, and drew no badges at all.
    page.evaluate("""() => {
        const wb = window.__wb, s = structuredClone(wb.shown);
        s.card_locations['2'] = 'DISCARD_PILE';
        s.card_locations['1'] = 'HAND_USSR_UNKNOWN';
        s.card_locations['37'] = 'DRAW_DECK';
        s.card_locations['38'] = 'REMOVED_FROM_GAME';
        wb.mapView.render(s);
    }""")
    tags = _scoring_tags(page)
    assert (tags["EUROPE"], tags["ASIA"], tags["CENTRAL AMERICA"], tags["SOUTHEAST ASIA"]) == ("out", "live", "live", "out")
    assert not page.errors


def test_the_page_says_it_is_unofficial(browser: Any, static_site: str) -> None:
    page = _open(browser, static_site + "/?model=off")
    bar = page.locator("#disclaimer-bar")
    assert bar.is_visible()
    text = bar.inner_text()
    assert "Unofficial" in text and "not affiliated with or endorsed by GMT Games" in text
    assert "Twilight Struggle® is a registered trademark of GMT Games" in text
    assert bar.bounding_box()["y"] == 0, "the first line of the page"


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
