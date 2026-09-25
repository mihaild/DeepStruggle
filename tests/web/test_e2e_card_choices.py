"""Events that make the US choose cards it does not hold, played by a human in the browser.

Our Man in Tehran shows the US five cards drawn from the deck and lets it discard any of them;
Star Wars and SALT Negotiations pick from the discard pile. None of those cards is in a hand the
page shows, and the HUD used to list no cards for an event's card choice at all -- so the US
could neither see the drawn cards (the display state also reported them as UNAVAILABLE) nor
click any of the offered ones. Each event is played here the way a person does it: click the card
in the US hand, click Event, then choose from what the HUD offers.

Positions are real mid-game positions from the engine, with the event card put into the US hand
and its precondition met, opened in the page through a `pos=` link.
"""
from __future__ import annotations

import base64
import functools
import http.server
import random
import socket
import threading
import zlib
from typing import Any, Callable, Iterator, List

import numpy as np
import pytest
from playwright.sync_api import sync_playwright

import ts_engine as ts
from tests.web.test_e2e_workbench import DIST
from tools.lib.game_step import drain_chance

OUR_MAN_IN_TEHRAN, STAR_WARS, SALT, NATO = 108, 85, 43, 21
US_HAND = (ts.CardLocation.HAND_US_UNKNOWN, ts.CardLocation.HAND_US_KNOWN)


def _us_action_round(seed: int) -> ts.GameState:
    """A position from turn 4 on where the US is choosing which card to play."""
    for s0 in range(seed, seed + 50):
        s = ts.GameState()
        ts.Engine.init_game(s, s0)
        drain_chance(s)
        rng = random.Random(s0)
        for _ in range(3000):
            c = s.ctx()
            if (s.current_phase == ts.Phase.ACTION_ROUND and c.decision_type == ts.DecisionType.SELECT_CARD
                    and c.decision_player == ts.Player.US and c.resolving_card == 0 and s.turn >= 4):
                return s
            if ts.Engine.is_terminal(s):
                break
            legal = [int(i) for i in np.nonzero(ts.get_flat_action_mask(s, False))[0]]
            ts.Engine.step_flat(s, rng.choice(legal))
            drain_chance(s)
    raise AssertionError("no US action round found")


def _give_us(s: ts.GameState, card: int) -> None:
    for i in range(1, 111):
        if s.get_card_location(i) in US_HAND and i not in (card, 6):
            old = s.get_card_location(card)
            s.set_card_location(i, old if old not in US_HAND else ts.CardLocation.DISCARD_PILE)
            s.set_card_location(card, ts.CardLocation.HAND_US_KNOWN)
            return
    raise AssertionError("the US hand has no card to swap")


def _token(s: ts.GameState) -> str:
    return base64.urlsafe_b64encode(zlib.compress(s.to_save_json().encode(), 9)).decode().rstrip("=")


def _position(card: int, tweak: Callable[[ts.GameState], ts.GameState]) -> str:
    s = _us_action_round(7)
    _give_us(s, card)
    return _token(tweak(s))


def _tehran(s: ts.GameState) -> ts.GameState:
    israel = next(i for i in range(84) if ts.MapData.get_country_name(i) == "Israel")
    s.set_country(israel, 6, 0)   # the US must control a Middle East country
    return s


def _space_lead(s: ts.GameState) -> ts.GameState:
    save = s.to_save_dict()
    save.update(us_space_track=3, ussr_space_track=1)   # Star Wars needs the US ahead
    return ts.state_from_save_dict(save)


def _only_untriggerable_discard(s: ts.GameState) -> ts.GameState:
    """Star Wars facing a discard pile whose one non-scoring card the US cannot trigger."""
    for i in range(1, 111):
        if s.get_card_location(i) == ts.CardLocation.DISCARD_PILE:
            s.set_card_location(i, ts.CardLocation.REMOVED_FROM_GAME)
    s.set_card_location(NATO, ts.CardLocation.DISCARD_PILE)
    save = _space_lead(s).to_save_dict()
    save["persistent_effects"] &= ~(int(ts.EffectBits.MARSHALL_PLAN_PLAYED) | int(ts.EffectBits.WARSAW_PACT_PLAYED))
    return ts.state_from_save_dict(save)


@pytest.fixture(scope="module")
def site() -> Iterator[str]:
    """The built page on a plain static server: none of this needs the local API."""
    handler = functools.partial(http.server.SimpleHTTPRequestHandler, directory=DIST)
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("", 0))
        port = sock.getsockname()[1]
    httpd = http.server.ThreadingHTTPServer(("127.0.0.1", port), handler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    try:
        yield f"http://127.0.0.1:{port}"
    finally:
        httpd.shutdown()


@pytest.fixture(scope="module")
def browser() -> Iterator[Any]:
    with sync_playwright() as p:
        b = p.chromium.launch(headless=True, args=["--no-sandbox", "--disable-gpu"])
        yield b
        b.close()


def _play_event_as_us(browser: Any, site: str, token: str, card: int) -> Any:
    page = browser.new_page(viewport={"width": 1440, "height": 1000})
    page.goto(f"{site}/?pos={token}")
    page.wait_for_function("window.__wb && window.__wb.state && window.__wb.state.step_index === 0")
    page.click(".tab-btn[data-tab='tab-us-hand']")
    page.click(f"#tab-us-hand .card-item[data-card-id='{card}']")
    page.wait_for_function("window.__wb.state.decision_context.decision_type_name === 'SELECT_PLAY_MODE'")
    page.click("#decision-body button[data-primary='0']")   # Resolve the Event
    page.wait_for_function(f"window.__wb.state.decision_context.resolving_card === {card}")
    return page


def _choice_ids(page: Any) -> List[int]:
    return [int(x) for x in page.eval_on_selector_all(
        "#decision-body .card-choice", "els => els.map(e => e.getAttribute('data-primary'))")]


def test_our_man_in_tehran_shows_the_drawn_cards_and_discards_them(browser: Any, site: str) -> None:
    page = _play_event_as_us(browser, site, _position(OUR_MAN_IN_TEHRAN, _tehran), OUR_MAN_IN_TEHRAN)
    assert "Our Man in Tehran" in page.inner_text("#decision-body")
    locs = page.evaluate("window.__wb.state.card_locations")
    drawn = sorted(int(k) for k, v in locs.items() if v == "PEEKED_TEMP")
    assert len(drawn) == 5, f"the display state must name the drawn cards (got {drawn})"
    assert sorted(_choice_ids(page)) == drawn, "every drawn card must be on screen and clickable"
    first_name = page.evaluate(f"window.__wb.engine.cardName({drawn[0]})")
    assert first_name in page.inner_text("#decision-body") and "Drawn" in page.inner_text("#decision-body")

    page.click(f"#decision-body .card-choice[data-primary='{drawn[0]}']")
    page.wait_for_function(f"window.__wb.state.card_locations['{drawn[0]}'] === 'DISCARD_PILE'")
    assert sorted(_choice_ids(page)) == drawn[1:], "the discarded card leaves the choice"

    page.click("#decision-body button[data-flags='128']")   # Done: the rest go back
    page.wait_for_function(f"window.__wb.state.decision_context.resolving_card !== {OUR_MAN_IN_TEHRAN}")
    locs = page.evaluate("window.__wb.state.card_locations")
    assert all(locs[str(c)] == "DRAW_DECK" for c in drawn[1:])
    assert locs[str(drawn[0])] == "DISCARD_PILE"


def test_salt_negotiations_takes_a_card_back_from_the_discard_pile(browser: Any, site: str) -> None:
    page = _play_event_as_us(browser, site, _position(SALT, lambda s: s), SALT)
    offered = _choice_ids(page)
    valid = [i for i in page.evaluate("window.__wb.state.legal_actions.valid_ids") if 1 <= i <= 110]
    assert offered and sorted(offered) == sorted(valid)
    assert "Discard pile" in page.inner_text("#decision-body")
    assert page.locator("#decision-body button[data-flags='128']").count() == 1, "SALT lets the US take nothing"
    pick = offered[0]
    page.click(f"#decision-body .card-choice[data-primary='{pick}']")
    page.wait_for_function(f"window.__wb.state.card_locations['{pick}'].startsWith('HAND_US')")


def test_star_wars_plays_an_event_from_the_discard_pile(browser: Any, site: str) -> None:
    page = _play_event_as_us(browser, site, _position(STAR_WARS, _space_lead), STAR_WARS)
    offered = _choice_ids(page)
    valid = [i for i in page.evaluate("window.__wb.state.legal_actions.valid_ids") if 1 <= i <= 110]
    assert offered and sorted(offered) == sorted(valid)
    assert page.locator("#decision-body button[data-flags='128']").count() == 0, "the Star Wars pick is mandatory"
    pick = offered[0]
    n = page.evaluate("window.__wb.state.step_index")
    page.click(f"#decision-body .card-choice[data-primary='{pick}']")
    page.wait_for_function(f"window.__wb.state.step_index > {n}")
    assert page.evaluate("window.__wb.state.decision_context.resolving_card") != STAR_WARS
    name = page.evaluate(f"window.__wb.engine.cardName({pick})")
    assert name in page.inner_text("#action-log-stream"), "the chosen card's event is played"


def test_star_wars_with_nothing_triggerable_can_be_passed_from_the_page(browser: Any, site: str) -> None:
    """The engine offers only a pass here (see the report on its fallback); the page must too."""
    page = _play_event_as_us(browser, site, _position(STAR_WARS, _only_untriggerable_discard), STAR_WARS)
    assert page.evaluate("window.__wb.state.legal_actions.valid_ids") == [0]
    assert _choice_ids(page) == []
    assert "No card can be chosen" in page.inner_text("#decision-body")
    page.click("#decision-body button[data-flags='128']")
    page.wait_for_function(f"window.__wb.state.decision_context.resolving_card !== {STAR_WARS}")
