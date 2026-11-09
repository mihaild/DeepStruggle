"""The cards the action log names, made hoverable (web/ui/src/game/log.ts lineHtml).

A log line is a Message whose cards are arguments by id; put into HTML, each card argument is a
`span.log-card[data-card-id]`, whose hover shows the card's Ops, side and text. The wrapping must
change nothing else: with the spans taken out and the text unescaped, every heading and line reads
as its English, and the spans are the message's cards, in order, each under its own name. Checked
under node on a two-turn game's grouped log -- and in the pseudo-language, where every line is
marked as translated and its cards are still spans.
"""
from __future__ import annotations

import html
import re
from typing import Any, Dict, List, Tuple

import pytest

from tests.web.js_runner import PUBLIC, run_ts

SPAN = re.compile(r'<span class="log-card" data-card-id="(\d+)">(.*?)</span>')
MARK = re.compile(r"^  (•|🎲|★|✂|⇄|▸) ")


@pytest.fixture(scope="module")
def log(tmp_path_factory: pytest.TempPathFactory) -> Dict[str, Any]:
    return run_ts("log_html.ts", str(tmp_path_factory.mktemp("log_html")), PUBLIC, "6")


def _spans(h: str) -> List[Tuple[int, str]]:
    return [(int(i), html.unescape(inner)) for i, inner in SPAN.findall(h)]


def _strip(h: str) -> str:
    """The line with the spans taken out and its text unescaped (the log escapes what it renders)."""
    return html.unescape(SPAN.sub(lambda m: m.group(2), h))


def test_the_spans_change_nothing_else(log: Dict[str, Any]) -> None:
    for row in log["game"]:
        assert _strip(row["html"]) == row["text"], row


def test_every_card_a_line_names_is_hoverable(log: Dict[str, Any]) -> None:
    wrapped = 0
    for row in log["game"]:
        assert _spans(row["html"]) == [(i, name) for i, name in row["cards"]], row
        wrapped += len(row["cards"])
    assert wrapped > 20, "two turns of play name more cards than that"
    deals = [row for row in log["game"] if re.match(r"^  • (US|USSR) draws \d+ cards?: ", row["text"])]
    assert deals and all(len(_spans(row["html"])) == int(row["text"].split(" draws ")[1].split()[0]) for row in deals), \
        "each card of a deal is hoverable"


def test_a_number_that_is_no_card_is_not_wrapped(log: Dict[str, Any]) -> None:
    """A Space Race box, a step, a die: only a card argument becomes a span."""
    numbered = [row for row in log["game"] if "Box #" in row["text"] or "Step " in row["text"]]
    for row in numbered:
        assert len(_spans(row["html"])) == len(row["cards"]), row


def test_in_the_pseudo_language_every_line_is_translated_and_its_cards_still_hoverable(log: Dict[str, Any]) -> None:
    for row, pseudo in zip(log["game"], log["pseudo"]):
        words = MARK.sub("", _strip(pseudo))
        assert words.startswith("⟦") and words.endswith("⟧"), pseudo
        assert [i for i, _ in _spans(pseudo)] == [i for i, _ in row["cards"]], pseudo
        assert MARK.match(pseudo) is None if not row["mark"] else pseudo.startswith(f"  {row['mark']} ")
