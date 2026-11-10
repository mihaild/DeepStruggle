"""Playing one side against the model: the page must not show the other side's private cards.

A player's view (`?view=us|ussr`) draws every panel from `redactState` (web/ui/src/game/view.ts),
never from the engine's full display state. These tests drive a game in the page -- both sides by
the auto-play path, `sendFlatAction(i, 0, true)`, so no model is needed -- and check what the
panels were handed (`__wb.shown`) and what the page actually says. With a model loaded, a player's
view shows none of its readout -- no win chances, no move probabilities, no search -- until Show
analysis turns it on, never the readout of the model's own decisions, and searches only what
auto-play plays or the readout shows.

Needs web/ui/dist (tools/scripts/build_web.sh) and a Playwright Chromium.
"""
from __future__ import annotations

import json
from typing import Any, Dict, List

from tests.web.test_e2e_workbench import (RUN, SNAP, _close_model_chooser, _open, _open_model_chooser, _search, _set_search, browser, checkpoint,  # noqa: F401 -- fixtures
                                          server, static_site)

#: Play the first legal action for whoever is to move, until `stop(state)` holds.
_PLAY_UNTIL = """stop => { const wb = window.__wb, until = new Function('s', 'return ' + stop);
    for (let n = 0; n < 5000 && !wb.state.is_terminal && !until(wb.state); n++) {
        const m = wb.engine.mask(false); wb.sendFlatAction(m.findIndex(x => x), 0, true);
    }
    return wb.state.step_index; }"""


def _play_until(page: Any, stop: str) -> None:
    page.evaluate(_PLAY_UNTIL, stop)


def _names(page: Any, cards: List[int]) -> List[str]:
    return [page.evaluate(f"window.__wb.engine.cardName({c})") for c in cards]


def test_the_other_sides_hand_and_the_deck_are_one_unseen_pile(browser: Any, static_site: str) -> None:
    page = _open(browser, f"{static_site}/?model=off")
    _play_until(page, "s.current_phase_name === 'HEADLINE'")
    full: Dict[str, Any] = page.evaluate("window.__wb.state")
    page.evaluate("window.__wb.setView('US')")
    shown: Dict[str, Any] = page.evaluate("window.__wb.shown")

    hidden = [c for c in full["hands"]["USSR"] if full["card_locations"][str(c)] == "HAND_USSR_UNKNOWN"]
    assert hidden, "the USSR's opening hand is not known to the US"
    assert not set(hidden) & set(shown["hands"]["USSR"])
    assert shown["hidden_cards"] == {"US": 0, "USSR": len(hidden)}
    locations = set(shown["card_locations"].values())
    assert not locations & {"HAND_USSR_UNKNOWN", "DRAW_DECK"}
    assert shown["unseen_count"] == full["draw_deck_count"] + len(hidden)
    # Nothing the US cannot know reaches the page, on any tab of the cards panel.
    for tab in ("tab-ussr-hand", "tab-us-hand", "tab-board-cards"):
        page.click(f"button[data-tab='{tab}']")
        text = page.inner_text("body")
        assert [n for n in _names(page, hidden) if n in text] == [], f"a USSR card is named on {tab}"
        if tab == "tab-us-hand":
            assert all(n in text for n in _names(page, full["hands"]["US"]))
        if tab == "tab-ussr-hand":
            assert page.locator(".card-item.card-hidden:visible").count() == len(hidden)
    # The All tab lists every card, the USSR's included -- as the deck-or-hand pile, never the hand.
    page.click("button[data-tab='tab-all-cards']")
    page.select_option("#all-cards-filter", "DRAW_DECK")
    deck = page.inner_text("#tab-all-cards")
    assert all(n in deck for n in _names(page, hidden))
    assert "HAND_USSR" not in deck and "USSR hand" in deck
    assert page.evaluate("new URL(location.href).searchParams.get('view')") == "us"
    assert page.evaluate("[document.getElementById('btn-toggle-debug').disabled, "
                         "document.getElementById('btn-toggle-replay').disabled]") == [True, True]
    assert page.errors == []
    page.close()


def test_the_other_sides_deal_and_face_down_headline_are_not_logged(browser: Any, static_site: str) -> None:
    page = _open(browser, f"{static_site}/?model=off")
    _play_until(page, "s.turn >= 2 && s.current_phase_name === 'HEADLINE'")
    full_logs = json.dumps(page.evaluate("window.__wb.englishLog()"))
    for view, opp in (("US", "USSR"), ("USSR", "US")):
        page.evaluate(f"window.__wb.setView('{view}')")
        logs = json.dumps(page.evaluate("window.__wb.shownEnglishLog()"))
        assert f"-> HAND_{opp}_UNKNOWN" in full_logs
        assert f"-> HAND_{opp}_UNKNOWN" not in logs, f"the {view} view logs the {opp} deal"
        assert f"-> HAND_{view}_UNKNOWN" in logs, f"the {view} view lost its own deal"
        # A card leaving the other hand -- played, discarded, removed -- is public.
        leaving = f"moved: HAND_{opp}_UNKNOWN"
        assert full_logs.count(leaving) > 0 and logs.count(leaving) == full_logs.count(leaving)

    # One side commits its headline; the side still choosing must not see it.
    page.evaluate("window.__wb.setView('DEV')")
    first = page.evaluate("window.__wb.state.decision_context.decision_player")
    _play_until(page, f"s.decision_context.decision_player !== '{first}'")
    second = "USSR" if first == "US" else "US"
    key = "headline_us_card" if first == "US" else "headline_ussr_card"
    card = page.evaluate(f"window.__wb.state.{key}")
    space = page.evaluate("window.__wb.state.space")
    perk = space[second] >= 4 and space[first] < 4   # Man in Earth Orbit: the other side shows first
    assert card and not perk
    page.evaluate(f"window.__wb.setView('{second}')")
    assert page.evaluate(f"window.__wb.shown.{key}") == 0
    log = page.inner_text("#action-log-stream")
    assert _names(page, [card])[0] not in log
    assert "face down" in log
    page.close()


def test_the_other_sides_decision_is_hidden_and_refused(browser: Any, static_site: str) -> None:
    page = _open(browser, f"{static_site}/?model=off&view=ussr")
    # Setup: the USSR places first, so play it out until the US is to move.
    _play_until(page, "s.decision_context.decision_player === 'US'")
    shown = page.evaluate("window.__wb.shown")
    assert shown["view_hidden_decision"] is True
    assert shown["legal_actions"]["valid_ids"] == []
    assert "to move" in page.inner_text("#decision-body")
    before = page.evaluate("window.__wb.state.step_index")
    page.evaluate("window.__wb.sendAction({decision_type: 1, primary_id: "
                  "window.__wb.state.legal_actions.valid_ids[0], secondary_id: 0, flags: 0})")
    page.evaluate("window.__wb.sendFlatAction(window.__wb.engine.mask(false).findIndex(x => x), 0)")
    assert page.evaluate("window.__wb.state.step_index") == before
    page.close()


#: Random legal moves from a fixed stream (an LCG), a new game whenever one ends, until `stop`.
#: Play seeded random games, from seeded deals, until `stop(state)` holds. The page's own new game
#: has a random seed, and its New game button asks how to play instead of dealing, so neither would
#: make the walk the same on every run.
_PLAY_RANDOM_UNTIL = """stop => { const wb = window.__wb, until = new Function('s', 'return ' + stop);
    let x = 12345; const rnd = n => { x = (x * 1103515245 + 12345) % 2147483648; return x % n; };
    let seed = 7000; wb.session.newGame(seed); wb.refresh();
    for (let n = 0; n < 20000 && !until(wb.state); n++) {
        if (wb.state.is_terminal) { wb.session.newGame(++seed); wb.refresh(); continue; }
        const m = wb.engine.mask(false), legal = [];
        m.forEach((ok, i) => { if (ok) legal.push(i); });
        wb.sendFlatAction(legal[rnd(legal.length)], 0, true);
    }
    return until(wb.state); }"""


def test_the_other_sides_last_turn_is_marked_on_the_map(browser: Any, static_site: str) -> None:
    page = _open(browser, f"{static_site}/?model=off&view=us")
    assert page.evaluate(_PLAY_RANDOM_UNTIL,
                         "s.current_phase_name === 'ACTION_ROUND' && s.decision_context.decision_player === 'US' && "
                         "s.turn >= 2 && window.__wb.shown.last_turn && window.__wb.shown.last_turn.countries.length >= 2")
    lt: Dict[str, Any] = page.evaluate("window.__wb.shown.last_turn")
    assert lt["player"] == "USSR" and lt["label"].startswith("T")
    for c in lt["countries"]:
        assert c["us"] or c["ussr"] or c["targeted"], c
    marked = page.evaluate("[...document.querySelectorAll('.svg-country-node.last-turn')].map(g => +g.dataset.id)")
    assert sorted(marked) == sorted(c["id"] for c in lt["countries"])
    summary = page.inner_text("#map-last-turn")
    assert f"USSR's last turn ({lt['label']})" in summary
    assert all(c["name"] in summary for c in lt["countries"])
    # The developer's view marks nothing.
    page.evaluate("window.__wb.setView('DEV')")
    assert page.evaluate("window.__wb.shown.last_turn") is None
    assert page.locator(".svg-country-node.last-turn").count() == 0
    assert not page.locator("#map-last-turn").is_visible()
    assert page.errors == []
    page.close()


def test_new_game_and_the_view_select_fit_a_laptop_screen(browser: Any, static_site: str) -> None:
    page = browser.new_page(viewport={"width": 1366, "height": 768})
    page.goto(f"{static_site}/?model=off")
    page.wait_for_function("window.__wb && window.__wb.liveState && window.__wb.positionToken", timeout=30000)
    # A fresh open asks how to play, over the live board; the answer sets the view.
    page.wait_for_selector("#mode-dialog:not(.hidden)")
    page.click("#mode-choice-us")
    page.wait_for_selector("#mode-dialog", state="hidden")
    assert page.evaluate("window.__wb.shown.view") == "US"
    for el in ("view-select", "btn-new-game", "btn-toggle-debug", "btn-toggle-replay", "btn-model-choose",
               "analysis-visible-toggle", "active-player-badge", "btn-undo-action"):
        box = page.locator(f"#{el}").bounding_box()
        assert box and box["width"] > 0 and box["x"] + box["width"] <= 1366, f"#{el} is off screen"
    _play_until(page, "s.current_phase_name === 'HEADLINE'")
    # New Game asks again; closing the question deals nothing -- and its Escape is not also the
    # page's Escape, which takes back the last move.
    n = page.evaluate("window.__wb.state.step_index")
    assert n > 0
    page.click("#btn-new-game")
    page.wait_for_selector("#mode-dialog:not(.hidden)")
    page.keyboard.press("Escape")
    page.wait_for_selector("#mode-dialog", state="hidden")
    assert page.evaluate("window.__wb.state.step_index") == n
    # Answering it deals a new game in the chosen mode.
    page.click("#btn-new-game")
    page.wait_for_selector("#mode-dialog:not(.hidden)")
    page.click("#mode-choice-ussr")
    page.wait_for_function("window.__wb.state.step_index === 0")
    assert page.evaluate("window.__wb.shown.view") == "USSR"
    assert page.evaluate("document.getElementById('view-select').value") == "USSR"
    page.close()


def test_the_play_mode_question_sets_how_the_model_plays_a_side(browser: Any, static_site: str) -> None:
    page = browser.new_page(viewport={"width": 1366, "height": 768})
    page.goto(f"{static_site}/?model=off")
    page.wait_for_function("window.__wb && window.__wb.liveState && window.__wb.positionToken", timeout=30000)
    # A fresh open offers the strongest player, the rollout root, for playing a side -- in the
    # model & search section, folded under a summary that says so.
    page.wait_for_selector("#mode-dialog:not(.hidden)")
    assert page.evaluate("document.getElementById('mode-model-details').open") is False
    assert not page.is_visible("#mode-opponent-rollout")
    assert page.inner_text("#mode-model-summary") == "no model · Rollout root"
    assert page.is_checked("#mode-opponent-rollout") and not page.is_checked("#mode-opponent-policy")
    page.click("#mode-choice-us")
    page.wait_for_selector("#mode-dialog", state="hidden")
    assert _search(page) == "rollout"
    assert page.inner_text("#model-mode-label-mode") == "Rollout root"
    page.wait_for_function("new URLSearchParams(location.search).get('search') === 'rollout:4:32:4:z2:earlier:own:keyed:aligned:box4'")
    # New Game offers the setting in force, folded again; the policy turns the search off.
    page.click("#btn-new-game")
    page.wait_for_selector("#mode-dialog:not(.hidden)")
    assert page.evaluate("document.getElementById('mode-model-details').open") is False
    assert page.is_checked("#mode-opponent-rollout")
    page.click("#mode-model-details summary")
    page.check("#mode-opponent-policy")
    assert page.inner_text("#mode-model-summary") == "no model · Policy"
    assert page.inner_text("#model-mode-label-mode") == "Rollout root", "the question's choice waits for the mode"
    page.click("#mode-choice-ussr")
    page.wait_for_selector("#mode-dialog", state="hidden")
    assert _search(page) == "policy"
    page.wait_for_function("!new URLSearchParams(location.search).has('search')")
    # The Developer choice leaves the search as it is, unless it was changed in the question;
    # cancelling sets nothing.
    page.click("#btn-new-game")
    page.wait_for_selector("#mode-dialog:not(.hidden)")
    page.click("#mode-choice-dev")
    page.wait_for_selector("#mode-dialog", state="hidden")
    assert _search(page) == "policy"
    page.click("#btn-new-game")
    page.wait_for_selector("#mode-dialog:not(.hidden)")
    page.click("#mode-model-details summary")
    page.check("#mode-opponent-rollout")
    page.keyboard.press("Escape")
    page.wait_for_selector("#mode-dialog", state="hidden")
    assert _search(page) == "policy"
    page.click("#btn-new-game")
    page.wait_for_selector("#mode-dialog:not(.hidden)")
    page.click("#mode-model-details summary")
    page.check("#mode-opponent-rollout")
    page.click("#mode-choice-dev")
    page.wait_for_selector("#mode-dialog", state="hidden")
    assert _search(page) == "rollout"
    # The model & search dialog, from the technical bar's label, sets the search at once.
    _set_search(page, "policy")
    assert page.inner_text("#model-mode-label-mode") == "Policy"
    page.wait_for_function("!new URLSearchParams(location.search).has('search')")
    # Its Escape closes it and takes back no move.
    n = page.evaluate("window.__wb.state.step_index")
    _open_model_chooser(page)
    page.keyboard.press("Escape")
    page.wait_for_selector("#model-mode-dialog", state="hidden")
    assert page.evaluate("window.__wb.state.step_index") == n
    page.close()


def test_the_play_mode_question_describes_the_search_it_would_keep(browser: Any, static_site: str) -> None:
    # A link's own spec is kept by the rollout choice, so that is the one the question describes.
    page = _open(browser, f"{static_site}/?model=off&view=us&search=rollout:2:1:1:z2")
    page.click("#btn-new-game")
    page.wait_for_selector("#mode-dialog:not(.hidden)")
    assert page.text_content("#mode-opponent-rollout-spec") == "its top 2 moves, each played out in 1 sampled world"
    page.click("#mode-choice-us")
    page.wait_for_function("new URLSearchParams(location.search).get('search') === 'rollout:2:1:1:z2:earlier:own'")
    # Back on the policy, it offers the default spec again.
    _set_search(page, "policy")
    page.click("#btn-new-game")
    page.wait_for_selector("#mode-dialog:not(.hidden)")
    assert page.text_content("#mode-opponent-rollout-spec") == "its top 4 moves, each played out in 32 sampled worlds"
    page.close()


def test_show_analysis_follows_the_view_and_the_toggle(browser: Any, static_site: str) -> None:
    page = _open(browser, f"{static_site}/?model=off&view=us")
    # Off in a player's view, on in the Developer view: each view's default, set on a change of view.
    assert not page.is_checked("#analysis-visible-toggle")
    page.select_option("#view-select", "DEV")
    assert page.is_checked("#analysis-visible-toggle")
    page.select_option("#view-select", "USSR")
    assert not page.is_checked("#analysis-visible-toggle")
    # The toggle overrides the default until the view changes again.
    page.check("#analysis-visible-toggle")
    page.evaluate("window.__wb.setView('USSR')")
    assert page.is_checked("#analysis-visible-toggle")
    page.select_option("#view-select", "DEV")
    page.uncheck("#analysis-visible-toggle")
    assert page.evaluate("window.__wb.analysisPanel.readoutShown") is False
    page.select_option("#view-select", "US")
    page.select_option("#view-select", "DEV")
    assert page.evaluate("window.__wb.analysisPanel.readoutShown") is True
    box = page.locator("#analysis-visible-toggle").bounding_box()
    assert box and box["x"] + box["width"] <= 1440
    assert page.errors == []
    page.close()


def test_a_players_view_shows_no_readout_and_no_hints(browser: Any, server: str, checkpoint: Dict[str, str]) -> None:
    # A small search, so the model's own moves are quick; the view names the side, so no question.
    page = _open(browser, server + "/?view=us&search=rollout:2:1:1:z2")
    # Record whose decision every search is run on.
    page.evaluate("""() => { const r = window.__wb.runner, search = r.search.bind(r); window.__searched = [];
        r.search = (...args) => { window.__searched.push(window.__wb.engine.decisionPlayer()); return search(...args); }; }""")
    _open_model_chooser(page)
    page.select_option("#analysis-source-select", "local")
    page.wait_for_function(f"[...document.querySelectorAll('#analysis-run-select option')].some(o => o.value === '{RUN}')")
    page.select_option("#analysis-run-select", RUN)
    _close_model_chooser(page)
    # The USSR (the model) sets up first, by auto-play; then the US -- the player -- is to choose.
    page.wait_for_function("window.__wb.state.decision_context.decision_player === 'US' && window.__wb.liveAnalysis "
                           "&& window.__wb.liveAnalysis.decision_player === 'US' && !!window.__wb.liveAnalysis.policy",
                           timeout=120000)
    body = page.inner_text("#analysis-body")
    assert "You are playing US" in body
    assert page.locator("#analysis-body .analysis-choice").count() == 0, "the model's move list is shown"
    assert "ROLLOUT ROOT" not in body
    assert page.locator(".trace-choice-badge").count() == 0, "the model's probabilities are painted on the board"
    a = page.evaluate("window.__wb.liveAnalysis")
    assert not a.get("search") and not a.get("search_pending"), "the player's own decision was searched"
    searched = page.evaluate("window.__searched")
    assert searched and set(searched) == {-1}, f"searches ran on {searched} -- only the model's side (-1) should be"
    # Neither the button nor the key plays the model's move for the player.
    assert page.evaluate("document.getElementById('btn-play-favourite').disabled") is True
    n = page.evaluate("window.__wb.state.step_index")
    page.keyboard.press("f")
    page.evaluate("window.__wb.analysisPanel.playFavourite()")
    assert page.evaluate("window.__wb.state.step_index") == n
    # Show analysis turns the readout on for the player's own decision -- which is then searched,
    # being on screen -- and off again.
    page.check("#analysis-visible-toggle")
    page.wait_for_selector("#analysis-body .analysis-choice", timeout=60000)
    page.wait_for_function("window.__wb.liveAnalysis && !!window.__wb.liveAnalysis.search", timeout=120000)
    assert page.evaluate("window.__searched").count(1) == 1
    assert page.locator(".trace-choice-badge").count() > 0
    page.uncheck("#analysis-visible-toggle")
    assert page.locator("#analysis-body .analysis-choice").count() == 0
    assert page.locator(".trace-choice-badge").count() == 0
    # The Developer view shows everything again, and keeps the search already made: a change of
    # view never re-reads the position.
    page.evaluate("window.__wb.setView('DEV')")
    page.wait_for_selector("#analysis-body .analysis-choice", timeout=60000)
    assert page.evaluate("window.__wb.liveAnalysis && !!window.__wb.liveAnalysis.search")
    assert page.evaluate("window.__searched").count(1) == 1
    assert page.locator(".trace-choice-badge").count() > 0
    assert page.evaluate("document.getElementById('btn-play-favourite').disabled") is False
    # Playing the USSR makes this the model's decision: its search is already made, so auto-play
    # arms at once with it rather than searching again.
    searched_again, same, armed = page.evaluate("""() => { const wb = window.__wb, n = window.__searched.length,
        a = wb.liveAnalysis; wb.setView('USSR');
        return [window.__searched.length - n, wb.liveAnalysis === a && !!a.search, wb.autoPlayTimer !== null]; }""")
    assert (searched_again, same, armed) == (0, True, True)
    page.wait_for_function(f"window.__wb.state.step_index > {n}", timeout=30000)
    assert page.errors == []
    page.close()


def test_the_play_mode_question_starts_one_search_at_a_time(browser: Any, server: str, checkpoint: Dict[str, str]) -> None:
    # A fresh open, so the question is asked; the model loads while it is open.
    page = browser.new_page(viewport={"width": 1440, "height": 900})
    errors: list = []
    page.on("pageerror", lambda e: errors.append(str(e)))
    page.goto(server + "/?model=off")
    page.wait_for_function("window.__wb && window.__wb.liveState && window.__wb.positionToken", timeout=30000)
    page.wait_for_selector("#mode-dialog:not(.hidden)")
    # Record every search: the position it was asked for, and how many ran at once on the runner.
    page.evaluate("""() => { const wb = window.__wb, r = wb.runner, search = r.search.bind(r), now = r.searchNow.bind(r);
        window.__asked = []; window.__active = 0; window.__most = 0;
        r.search = (...args) => { window.__asked.push(wb.state.step_index); return search(...args); };
        r.searchNow = async (...args) => { window.__most = Math.max(window.__most, ++window.__active);
            try { return await now(...args); } finally { window.__active--; } }; }""")
    page.evaluate(f"window.__wb.loadModel({{source: {{kind: 'local', path: '{RUN}/{SNAP}'}}}})")
    page.wait_for_function("window.__wb.liveAnalysis && !!window.__wb.liveAnalysis.policy", timeout=120000)
    # The rollout root, the default on a fresh open; playing the US, the model sets up the USSR.
    assert page.is_checked("#mode-opponent-rollout")
    page.click("#mode-choice-us")
    page.wait_for_function("window.__wb.state.decision_context.decision_player === 'US' && window.__wb.liveAnalysis "
                           "&& window.__wb.liveAnalysis.decision_player === 'US'", timeout=300000)
    asked = page.evaluate("window.__asked")
    assert asked and len(asked) == len(set(asked)), f"a position was searched twice: {asked}"
    assert page.evaluate("window.__most") == 1
    # The runner itself: a request still waiting when a newer one comes is superseded, never run;
    # one already running finishes, and the newest runs after it.
    outcomes = page.evaluate("""async () => { const wb = window.__wb, r = wb.runner, s = wb.engine.snapshot(),
        spec = {k: 2, worlds: 1, horizon: 1, rule: 'z2', stop: 'earlier', leaf: 'own'}, ask = () => r.search(s, wb.model, wb.modelKey, spec, 7);
        const first = ask();
        await new Promise(res => setTimeout(res, 0));   // the first is running
        const all = await Promise.allSettled([first, ask(), ask()]);
        return all.map(o => o.status === 'fulfilled' ? 'ok' : o.reason.message); }""")
    assert outcomes == ["ok", "superseded by a newer search", "ok"]
    assert page.evaluate("window.__most") == 1
    assert errors == []
    page.close()


def test_a_failed_search_is_reported_in_a_players_view(browser: Any, server: str, checkpoint: Dict[str, str]) -> None:
    page = _open(browser, server + "/?view=us&search=rollout:2:1:1:z2&model=off")
    # Every search fails, and the model is held on its first decision so the panel can be read.
    page.evaluate("""() => { const wb = window.__wb; wb.maybeAutoPlay = () => {};
        wb.runner.search = () => Promise.reject(new Error('the search broke')); }""")
    page.evaluate(f"window.__wb.loadModel({{source: {{kind: 'local', path: '{RUN}/{SNAP}'}}}})")
    page.wait_for_function("window.__wb.liveAnalysis && window.__wb.liveAnalysis.search_error", timeout=120000)
    assert page.evaluate("window.__wb.liveAnalysis.decision_player") == "USSR"
    body = page.inner_text("#analysis-body")
    assert "The model is choosing for USSR" in body
    assert "The rollout root failed, so the model plays its policy's move instead: the search broke" in body
    # It does: the favourite is the policy's argmax.
    assert page.evaluate("window.__wb.analysisPanel.autoPlayMove() === window.__wb.liveAnalysis.policy.argmax_idx")
    assert page.errors == []
    page.close()


def test_a_view_change_drops_a_move_armed_for_the_old_auto_side(browser: Any, server: str, checkpoint: Dict[str, str]) -> None:
    # The Developer view, the model auto-playing the USSR with a quick search.
    page = _open(browser, server + "/?auto=ussr&search=rollout:2:1:1:z2&model=off")
    page.evaluate("window.__wb.loadModel({source: {kind: 'local', path: '%s/%s'}})" % (RUN, SNAP))
    # With the USSR's move armed, hand the USSR to the player: the move is dropped at once, not when
    # some later readout lands -- the armed timer would otherwise race it, and could play for them.
    page.wait_for_function("window.__wb.autoPlayTimer !== null && window.__wb.state.decision_context.decision_player === 'USSR'",
                           timeout=120000)
    n, timer = page.evaluate("() => { const wb = window.__wb; wb.setView('USSR'); return [wb.state.step_index, wb.autoPlayTimer]; }")
    assert timer is None, "the move armed for the old auto side is still armed"
    page.wait_for_timeout(1000)   # well past AUTO_PLAY_DELAY_MS
    assert page.evaluate("window.__wb.state.step_index") == n
    assert page.evaluate("window.__wb.state.decision_context.decision_player") == "USSR"
    assert page.errors == []
    page.close()
