"""The live analysis mode, in a real browser: pick a model, read it, play its favourite, share.

`test_live_analysis.py` proves the server computes and applies the right things; this proves they
reach the screen and that the address bar is a working link -- which only a browser can show.
"""
import json
import os
import socket
import threading
import time
from urllib.parse import parse_qs, urlparse

import pytest
import requests
import torch
import uvicorn
from playwright.sync_api import sync_playwright

from web.server.main import app

RUN = "E9-02-01_20260101_000000"
SNAPSHOT = "snapshot_3000000steps.pt"


def get_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return int(s.getsockname()[1])


@pytest.fixture(scope="module")
def analysis_server(tmp_path_factory):
    from ai.models.coldwar_net_v2 import create_coldwar_net_v2

    root = tmp_path_factory.mktemp("e2e_checkpoints")
    run = root / RUN
    run.mkdir()
    torch.manual_seed(1)
    torch.save(create_coldwar_net_v2(torch.device("cpu")).state_dict(), run / SNAPSHOT)
    (run / "metadata.json").write_text(json.dumps({"merged_influence": False}))

    previous = os.environ.get("TS_CHECKPOINTS_DIR")
    os.environ["TS_CHECKPOINTS_DIR"] = str(root)
    port = get_free_port()
    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="error"))
    threading.Thread(target=server.run, daemon=True).start()
    time.sleep(1.0)
    try:
        yield f"http://127.0.0.1:{port}"
    finally:
        server.should_exit = True
        if previous is None:
            os.environ.pop("TS_CHECKPOINTS_DIR", None)
        else:
            os.environ["TS_CHECKPOINTS_DIR"] = previous


@pytest.fixture(scope="module")
def browser():
    with sync_playwright() as p:
        b = p.chromium.launch(headless=True, args=["--no-sandbox", "--disable-setuid-sandbox",
                                                   "--disable-dev-shm-usage", "--disable-gpu"])
        yield b
        b.close()


def _step_index(page) -> int:
    return int(page.evaluate("window.__wb.state?.step_index ?? -1"))


def test_pick_model_play_favourite_and_share_the_link(browser, analysis_server):
    game_id = "e2e-analysis"
    requests.post(f"{analysis_server}/api/games/new", json={"game_id": game_id, "seed": 31})

    page = browser.new_page(viewport={"width": 1440, "height": 900})
    page.goto(f"{analysis_server}/?game_id={game_id}")
    page.wait_for_function("window.__wb && window.__wb.state && window.__wb.state.position")
    page.wait_for_function("document.querySelectorAll('#analysis-run-select option').length > 1")

    page.select_option("#analysis-run-select", RUN)
    assert page.input_value("#analysis-snapshot-select") == SNAPSHOT, "defaults to the run's latest weights"
    page.wait_for_selector("#analysis-body .analysis-choice", timeout=15000)

    # Setup is an influence placement: every legal country carries its probability, and the
    # critic's verdict is on screen.
    assert page.locator("#ts-map-svg .trace-choice-badge").count() > 1
    assert page.locator("#analysis-body .trace-critic tr.row-us").count() == 1
    assert page.locator("#analysis-body .analysis-value-bar").count() == 1
    assert page.locator(".trace-choice-played").count() >= 1, "the favourite is not marked"

    history_before = page.evaluate("history.length")
    start = _step_index(page)
    page.click("#btn-play-favourite")
    page.wait_for_function(f"window.__wb.state.step_index > {start}")
    page.keyboard.press("f")
    page.wait_for_function(f"window.__wb.state.step_index > {start + 1}")
    page.wait_for_selector("#analysis-body .analysis-choice")

    # The address bar names this exact board and model, and moving did not add history.
    page.wait_for_function("new URLSearchParams(location.search).get('pos') === window.__wb.state.position")
    url = page.url
    q = parse_qs(urlparse(url).query)
    assert q["model"] == [f"{RUN}/{SNAPSHOT}"]
    assert q["game_id"] == [game_id]
    assert page.evaluate("history.length") == history_before, "moves must replace, not push, history"
    shared_pos = q["pos"][0]
    shared_step_board = page.evaluate("JSON.stringify(window.__wb.state.countries)")

    # Someone else starts over on that game; the link still puts the shared board back.
    requests.post(f"{analysis_server}/api/games/new", json={"game_id": game_id, "seed": 999})
    other = browser.new_page(viewport={"width": 1440, "height": 900})
    other.goto(url)
    other.wait_for_function("window.__wb && window.__wb.state && window.__wb.state.position")
    assert other.evaluate("window.__wb.state.position") == shared_pos
    assert other.evaluate("JSON.stringify(window.__wb.state.countries)") == shared_step_board
    # ...with the same model selected and analysing it.
    other.wait_for_selector("#analysis-body .analysis-choice", timeout=15000)
    assert other.input_value("#analysis-run-select") == RUN
    assert other.input_value("#analysis-snapshot-select") == SNAPSHOT

    # Clicking a country by hand still works alongside the analysis.
    before = _step_index(other)
    badge_country = other.locator("#ts-map-svg .svg-country-node:has(.trace-choice-badge)").first
    badge_country.click()
    other.wait_for_function(f"window.__wb.state.step_index > {before}")

    # Turning analysis off clears the board's numbers and the model from the link.
    other.select_option("#analysis-run-select", "")
    other.wait_for_function("document.querySelectorAll('.trace-choice-badge').length === 0")
    other.wait_for_function("!new URLSearchParams(location.search).has('model')")


def _state(page, expr: str):
    return page.evaluate(f"window.__wb.state.{expr}")


def test_auto_play_moves_one_side_and_cancel_unwinds_past_it(browser, analysis_server):
    """auto=ussr: the USSR plays the model's favourite by itself, stops whenever the US is to
    move, and a Cancel takes back the US move together with the USSR replies to it."""
    game_id = "e2e-autoplay"
    requests.post(f"{analysis_server}/api/games/new", json={"game_id": game_id, "seed": 57})
    page = browser.new_page(viewport={"width": 1440, "height": 900})
    page.goto(f"{analysis_server}/?game_id={game_id}&model={RUN}/{SNAPSHOT}&auto=ussr")
    page.wait_for_function("window.__wb && window.__wb.state && window.__wb.state.position")
    assert page.input_value("#analysis-autoplay-select") == "USSR", "auto side restored from the URL"

    # The USSR setup is played without anyone touching it; the US setup then waits for us.
    page.wait_for_function("window.__wb.state.step_index > 0 && "
                           "window.__wb.state.decision_context.decision_player === 'US'", timeout=20000)
    parked = _state(page, "position")
    page.wait_for_timeout(1000)
    assert _state(page, "position") == parked, "auto-play must not move for the US"
    assert parse_qs(urlparse(page.url).query)["auto"] == ["ussr"]

    # Play US favourites by hand until one of them is answered by USSR auto-play.
    before_us_move = None
    for _ in range(40):
        before_us_move = _state(page, "position")
        s0 = _state(page, "step_index")
        page.keyboard.press("f")
        page.wait_for_function(f"window.__wb.state.step_index > {s0}")
        page.wait_for_function("window.__wb.state.is_terminal || "
                               "window.__wb.state.decision_context.decision_player === 'US'", timeout=20000)
        if _state(page, "step_index") > s0 + 1:
            break
    else:
        pytest.fail("the USSR never got a move to auto-play")

    # One Cancel: back to the position before our US move, and it stays there.
    page.click("#btn-undo-action")
    page.wait_for_function(f"window.__wb.state.position === {json.dumps(before_us_move)}", timeout=10000)
    page.wait_for_timeout(1000)
    assert _state(page, "position") == before_us_move, "auto-play replayed the move just taken back"

    # Switching auto-play off drops it from the link.
    page.select_option("#analysis-autoplay-select", "")
    page.wait_for_function("!new URLSearchParams(location.search).has('auto')")
