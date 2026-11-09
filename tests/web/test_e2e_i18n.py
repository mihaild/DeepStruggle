"""The page's languages in Chromium (web/ui/src/i18n): a switch redraws in place and changes nothing else.

The pseudo-language `en-XA` stands in for a translation: every message is the English one marked
`⟦…⟧`. Switching mid-game keeps the position, the moves and the undo history while the static
markup and the header follow the language chosen. A link's `?lang=` opens the page in that language
without being remembered, a choice made in the select is remembered (and drops the link's), and the
leaderboard switches the same way with the same rows. The catalogs themselves are test_i18n.py's.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterator

import pytest

from tests.web.test_e2e_card_choices import browser, site  # noqa: F401 (fixtures)

LOCALES = Path(__file__).resolve().parents[2] / "web" / "ui" / "src" / "i18n" / "locales"
PSEUDO = "en-XA"


def message(lang: str, key: str) -> str:
    """`ns:dotted.id` as the page shows it in `lang` (English, or English marked for the pseudo-language)."""
    ns, path = key.split(":", 1)
    node: Any = json.loads((LOCALES / "en" / f"{ns}.json").read_text(encoding="utf-8"))
    for part in path.split("."):
        node = node[part]
    assert isinstance(node, str), key
    return f"⟦{node}⟧" if lang == PSEUDO else node


@pytest.fixture
def page(browser: Any, site: str) -> Iterator[Any]:  # noqa: F811
    ctx = browser.new_context(viewport={"width": 1440, "height": 1000})
    p = ctx.new_page()
    errors: list = []
    p.on("pageerror", lambda e: errors.append(str(e)))
    yield p
    ctx.close()
    assert errors == [], errors


def _workbench(page: Any, site: str, query: str = "") -> None:
    # view=dev names the play mode, so the page does not open on its play-mode question.
    page.goto(f"{site}/?model=off&view=dev{query}")
    page.wait_for_function("window.__wb && window.__wb.state")


def _switch(page: Any, lang: str) -> None:
    page.select_option("select[data-language-select]", lang)
    page.wait_for_function(f"document.documentElement.lang === {json.dumps(lang)}")


def _game(page: Any) -> Dict[str, Any]:
    return page.evaluate("""() => {
      const w = window.__wb;
      return { position: w.engine.saveJson(), steps: w.session.steps.length, undo: w.session.history.length };
    }""")


def test_with_one_language_there_is_no_select(page: Any, site: str) -> None:  # noqa: F811
    _workbench(page, site)
    assert page.evaluate("document.documentElement.lang") == "en"
    assert not page.is_visible("select[data-language-select]")


def test_a_switch_mid_game_keeps_the_game_and_redraws_the_markup(page: Any, site: str) -> None:  # noqa: F811
    _workbench(page, site, f"&lang={PSEUDO}")
    assert page.is_visible("select[data-language-select]"), "the pseudo-language offers the select"
    key = page.get_attribute("#btn-new-game", "data-i18n")
    assert key, "the New Game button names its message"
    assert page.inner_text("#btn-new-game") == message(PSEUDO, key)
    start = _game(page)
    page.evaluate("""() => {
      const w = window.__wb;
      for (let n = 0; n < 3; n++) w.sendFlatAction(w.engine.mask(false).findIndex(Boolean), 0, true);
    }""")
    played = _game(page)
    assert played["position"] != start["position"] and played["steps"] > start["steps"]

    _switch(page, "en")
    assert _game(page) == played
    assert page.inner_text("#btn-new-game") == message("en", key)

    _switch(page, PSEUDO)
    assert _game(page) == played
    assert page.inner_text("#btn-new-game") == message(PSEUDO, key)
    page.click("#btn-undo-action")
    assert _game(page)["steps"] == played["steps"] - 1, "undo still takes back one move"


def test_a_links_language_opens_the_page_and_a_choice_is_kept(page: Any, site: str) -> None:  # noqa: F811
    _workbench(page, site, f"&lang={PSEUDO}")
    assert page.evaluate("document.documentElement.lang") == PSEUDO
    assert page.evaluate("localStorage.getItem('ts.language')") is None, "a link's language is not remembered"

    _switch(page, "en")
    assert "lang=" not in page.url, "a choice drops the link's language, so a reload keeps it"
    page.reload()
    page.wait_for_function("window.__wb && window.__wb.state")
    assert page.evaluate("document.documentElement.lang") == "en"

    _workbench(page, site, f"&lang={PSEUDO}")
    _switch(page, PSEUDO)
    assert "lang=" not in page.url, "a choice is not written into the address"
    _workbench(page, site)
    assert page.evaluate("document.documentElement.lang") == PSEUDO, "the choice is remembered"
    _workbench(page, site, "&lang=en")
    assert page.evaluate("document.documentElement.lang") == "en", "a link's language wins over the remembered one"


def test_a_longer_language_fits_a_phone(browser: Any, site: str) -> None:  # noqa: F811
    ctx = browser.new_context(viewport={"width": 390, "height": 844}, is_mobile=True, has_touch=True)
    p = ctx.new_page()
    _workbench(p, site, f"&lang={PSEUDO}")
    assert p.evaluate("document.documentElement.scrollWidth") <= 390, "nothing wider than the screen"
    ctx.close()


def test_the_leaderboard_switches_in_place(page: Any, site: str) -> None:  # noqa: F811
    page.goto(f"{site}/leaderboard.html?lang={PSEUDO}")
    page.wait_for_function("document.querySelector('#lb-body')?.children.length > 0")
    rows = page.locator("#lb-body tr").count()
    assert page.inner_text("#lb-tab-ratings-btn") == message(PSEUDO, "leaderboard:page.tabRatings")
    assert page.inner_text("#lb-note").startswith(message(PSEUDO, "leaderboard:note.anchor"))
    _switch(page, "en")
    assert page.inner_text("#lb-tab-ratings-btn") == message("en", "leaderboard:page.tabRatings")
    assert page.locator("#lb-body tr").count() == rows
    assert "lang=" not in page.url
