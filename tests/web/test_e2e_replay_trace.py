"""The workbench must actually show the trace it is handed.

The backend tests prove the numbers are recorded; these prove they reach the screen. Both views
are checked against a replay generated here, so they are driven by what the writer emits today
rather than by a fixture that can drift from it.
"""
import socket
import threading
import time

import pytest
import uvicorn
from playwright.sync_api import sync_playwright

from tests.web.test_e2e_workbench import HF_BLOCKED
from web.server.main import app


def get_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return int(s.getsockname()[1])


@pytest.fixture(scope="module")
def traced_replay_server(tmp_path_factory):
    """A server serving one freshly generated, traced replay."""
    import os

    from ai.models.coldwar_net_v2 import create_coldwar_net_v2
    from tools.lib.self_play import generate_self_play_replay
    from web.server.replay import REPLAYS_DIR_ENV

    target = tmp_path_factory.mktemp("traced_replays")
    previous = os.environ.get(REPLAYS_DIR_ENV)
    os.environ[REPLAYS_DIR_ENV] = str(target)
    try:
        generate_self_play_replay(
            model=create_coldwar_net_v2("cpu"), seed=4242, temperature=0.5,
            game_id="traced_fixture", output_path=str(target / "traced_fixture.tslog.json"),
            device="cpu", verbose=False, max_steps=80)

        port = get_free_port()
        server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port,
                                               log_level="error"))
        threading.Thread(target=server.run, daemon=True).start()
        time.sleep(1.0)
        yield f"http://127.0.0.1:{port}"
        server.should_exit = True
    finally:
        if previous is None:
            os.environ.pop(REPLAYS_DIR_ENV, None)
        else:
            os.environ[REPLAYS_DIR_ENV] = previous


@pytest.fixture(scope="module")
def browser():
    with sync_playwright() as p:
        b = p.chromium.launch(headless=True, args=["--no-sandbox", "--disable-setuid-sandbox",
                                                   "--disable-dev-shm-usage", "--disable-gpu",
                                                   HF_BLOCKED])
        yield b
        b.close()


def test_the_value_ribbon_and_readout_panel_render_for_a_traced_replay(browser,
                                                                       traced_replay_server):
    page = browser.new_page(viewport={"width": 1440, "height": 900})
    page.goto(f"{traced_replay_server}/?replay=traced_fixture.tslog.json")
    page.wait_for_selector("#rep-value-ribbon svg path", timeout=15000)

    ribbon = page.locator("#rep-value-ribbon")
    assert ribbon.is_visible(), "the value ribbon stayed hidden for a replay that has a trace"
    assert ribbon.locator("svg path").count() >= 1, "the ribbon drew no curve"

    # Step forward until the position on screen has a real choice open, and check that every
    # option carries its own number -- on the cards, the HUD buttons or the map.
    for _ in range(60):
        if page.locator(".trace-choice-badge").count() > 1:
            break
        page.click("#btn-rep-next")
    assert page.locator("#trace-panel").is_visible(), "the readout panel never appeared"
    assert page.locator(".trace-choice-badge").count() > 1, (
        "no choice was labelled with its probability in the first 60 steps")
    assert page.locator(".trace-choice-played").count() >= 1, (
        "the move that was actually played is not marked")

    # The critic's prediction for both sides, to five decimals, is the panel's job.
    critic = page.locator("#trace-panel .trace-critic")
    assert critic.count() == 1, "no critic block"
    assert critic.locator("tr.row-us").count() == 1 and critic.locator("tr.row-ussr").count() == 1
    import re as _re
    assert _re.search(r"[+-]\d\.\d{5}", critic.inner_text()), (
        f"expected five-decimal predictions, got: {critic.inner_text()!r}")
    page.close()


def test_the_log_rows_carry_probability_chips(browser, traced_replay_server):
    page = browser.new_page(viewport={"width": 1440, "height": 900})
    page.goto(f"{traced_replay_server}/?replay=traced_fixture.tslog.json")
    page.wait_for_selector("#action-log-stream .log-item", timeout=15000)
    chips = page.locator("#action-log-stream .trace-chip")
    assert chips.count() > 0, "no step in the stream showed a probability chip"
    page.close()
