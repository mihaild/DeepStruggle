"""The page's languages (web/ui/src/i18n): the catalogs against each other and the static markup.

English (locales/en/<namespace>.json) is the source: every other language uses its ids, with its
own plural forms and the same `{{parameters}}`, and may leave any message out -- the page shows the
English for it. The `game` namespace holds card, country and region names by id, whose English is
the engine's and the rules', so English has none there. Every `data-i18n*` key in the two pages is
an English id, and the markup's own text is that message, so the page reads the same before the
catalog is applied.
"""
from __future__ import annotations

import json
import re
from html.parser import HTMLParser
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple

import pytest

REPO = Path(__file__).resolve().parents[2]
UI = REPO / "web" / "ui"
LOCALES = UI / "src" / "i18n" / "locales"
LANGUAGES = sorted(p.name for p in LOCALES.iterdir() if p.is_dir())
OTHERS = [lang for lang in LANGUAGES if lang != "en"]
NAMESPACES = sorted(p.stem for p in (LOCALES / "en").glob("*.json"))
#: CLDR's plural categories for cardinals (what Intl.PluralRules gives i18next), by language: a
#: language added to locales/ adds its own here (Chinese or Japanese {"other"}; Russian one, few, many, other).
PLURALS = {"en": {"one", "other"}}
SUFFIX = re.compile(r"_(zero|one|two|few|many|other)$")
PARAM = re.compile(r"\{\{\s*([A-Za-z_]\w*)\s*\}\}")


def flat(tree: Dict, prefix: str = "") -> Iterator[Tuple[str, str]]:
    for key, value in tree.items():
        if isinstance(value, dict):
            yield from flat(value, f"{prefix}{key}.")
        else:
            yield f"{prefix}{key}", value


def catalog(lang: str, ns: str) -> Dict[str, str]:
    return dict(flat(json.loads((LOCALES / lang / f"{ns}.json").read_text(encoding="utf-8"))))


def base(key: str) -> str:
    return SUFFIX.sub("", key)


def english() -> Dict[str, str]:
    """Every English message as `ns:key`, plural keys under their base with the `other` form."""
    out: Dict[str, str] = {}
    for ns in NAMESPACES:
        for key, text in catalog("en", ns).items():
            if not key.endswith(tuple(f"_{c}" for c in PLURALS["en"] - {"other"})):
                out[f"{ns}:{base(key)}"] = text
    return out


def test_every_language_has_the_same_namespaces_and_an_index() -> None:
    for lang in LANGUAGES:
        assert sorted(p.stem for p in (LOCALES / lang).glob("*.json")) == NAMESPACES, lang
        index = (LOCALES / lang / "index.ts").read_text()
        for ns in NAMESPACES:
            assert f'import {ns} from "./{ns}.json";' in index, (lang, ns)


@pytest.mark.parametrize("lang", LANGUAGES)
def test_plural_forms_are_the_languages_own(lang: str) -> None:
    for ns in NAMESPACES:
        keys = catalog(lang, ns)
        for key in keys:
            m = SUFFIX.search(key)
            if m:
                assert m.group(1) in PLURALS[lang], f"{lang} {ns}:{key}: no such plural form in {lang}"
        if lang == "en":
            for key in keys:
                if SUFFIX.search(key):
                    for form in PLURALS["en"]:
                        assert f"{base(key)}_{form}" in keys, f"en {ns}:{key} lacks its _{form} form"


def test_a_translation_is_of_an_english_message_with_its_parameters() -> None:
    en = english()
    for lang, ns in ((lang, ns) for lang in OTHERS for ns in NAMESPACES):
        if ns == "game":
            continue
        for key, text in catalog(lang, ns).items():
            source = en.get(f"{ns}:{base(key)}")
            assert source is not None, f"{lang} {ns}:{key} is not an English message"
            assert text.strip(), f"{lang} {ns}:{key} is empty"
            want = set(PARAM.findall(source))
            got = set(PARAM.findall(text))
            # A plural message may leave its count out: a language with one form may say it once.
            assert got <= want and want - got <= {"count"}, f"{lang} {ns}:{key}: {sorted(got)} != {sorted(want)}"


def test_no_message_is_empty_or_has_a_stray_placeholder() -> None:
    for ns in NAMESPACES:
        for key, text in catalog("en", ns).items():
            assert text.strip(), f"en {ns}:{key} is empty"
            assert not re.search(r"\{\d+\}|(?<!\{)\{[A-Za-z_]+\}(?!\})", text), f"en {ns}:{key}: use {{{{name}}}}"


def test_game_names_are_keyed_by_the_rules_ids() -> None:
    cards = {c["id"] for c in json.loads((REPO / "rules" / "cards.json").read_text())}
    board = json.loads((REPO / "rules" / "map.json").read_text())
    countries = {c["id"] for c in board["countries"]}
    regions = {c["region"] for c in board["countries"]} | {s for c in board["countries"] for s in c.get("subregions", [])}
    assert json.loads((LOCALES / "en" / "game.json").read_text()) == {}, "English names are the engine's"
    for lang in OTHERS:
        game = json.loads((LOCALES / lang / "game.json").read_text(encoding="utf-8"))
        assert set(game) <= {"cards", "countries", "regions"}
        assert {int(k) for k in game.get("cards", {})} <= cards, lang
        assert all(set(v) <= {"name", "description"} for v in game.get("cards", {}).values()), lang
        assert {int(k) for k in game.get("countries", {})} <= countries, lang
        assert set(game.get("regions", {})) <= regions, lang


class _Static(HTMLParser):
    """Each element with a data-i18n* attribute: the key, and the text or attribute it stands for."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.found: List[Tuple[str, str, Optional[str]]] = []
        self._open: List[Tuple[str, str, List[str]]] = []

    def handle_starttag(self, tag: str, attrs: List[Tuple[str, Optional[str]]]) -> None:
        a = dict(attrs)
        for name in ("title", "placeholder", "aria-label"):
            key = a.get(f"data-i18n-{name}")
            if key:
                self.found.append((key, f"@{name}", a.get(name)))
        key = a.get("data-i18n")
        if key:
            self._open.append((tag, key, []))

    def handle_data(self, data: str) -> None:
        for _, _, parts in self._open:
            parts.append(data)

    def handle_endtag(self, tag: str) -> None:
        if self._open and self._open[-1][0] == tag:
            _, key, parts = self._open.pop()
            self.found.append((key, "text", "".join(parts)))


@pytest.mark.parametrize("page", ["index.html", "leaderboard.html"])
def test_the_markup_names_english_messages_and_reads_as_them(page: str) -> None:
    en = english()
    parser = _Static()
    parser.feed((UI / page).read_text(encoding="utf-8"))
    assert parser.found, f"{page} has no data-i18n"
    norm = lambda s: re.sub(r"\s+", " ", s or "").strip()
    for key, where, text in parser.found:
        assert key in en, f"{page}: {key} is not an English message"
        assert norm(text) == norm(en[key]), f"{page} {where} of {key}: {norm(text)!r} != {norm(en[key])!r}"
