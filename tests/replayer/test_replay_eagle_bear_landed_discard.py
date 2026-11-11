"""A player at box 6 of the space race may discard a held card as the turn ends.

ts-replayer records that where it records the rest of the turn's bookkeeping: on a second copy
of the last entry's header, with a body holding nothing but the discard. The engine asks for it
inside the entry above, as that entry's last action round ends.
"""
from __future__ import annotations

from typing import Dict, List, cast

from tools.lib.ts_replayer_convert import _eagle_bear_landed_discard
from tools.lib.ts_replayer_parse import parse_entry
from tests.replay_corpus import ReplayCorpus

_COLONIAL_REAR_GUARDS = 63


def test_the_discard_is_read_off_the_cleanup_copy(corpus: ReplayCorpus) -> None:
    """Replay 56 turn 9 AR7: the USSR is at box 6 with the US at 5, and discards Colonial Rear
    Guards. The card is named on the copy of the entry, not in the entry itself."""
    raws = cast(List[Dict[str, object]], corpus.game(56)["all_turns"])
    index = next(i for i, r in enumerate(raws)
                 if (parse_entry(r).turn, parse_entry(r).phase, parse_entry(r).player)
                 == (9, "AR7", "US"))
    entry = parse_entry(raws[index])
    assert not entry.discards, "the entry itself records the play and nothing else"
    assert parse_entry(raws[index + 1]).discards == [("USSR", "Colonial Rear Guards")]
    assert _eagle_bear_landed_discard(raws, index, entry) == _COLONIAL_REAR_GUARDS


def test_an_ordinary_entry_names_no_such_discard(corpus: ReplayCorpus) -> None:
    raws = cast(List[Dict[str, object]], corpus.game(56)["all_turns"])
    index = next(i for i, r in enumerate(raws)
                 if (parse_entry(r).turn, parse_entry(r).phase, parse_entry(r).player)
                 == (9, "AR6", "US"))
    assert _eagle_bear_landed_discard(raws, index, parse_entry(raws[index])) is None


def test_replay_56_converts_in_full(corpus: ReplayCorpus) -> None:
    """Declined for want of a card to name, the turn ended with Colonial Rear Guards still in
    hand -- and the cleanup copy was then driven as a play into an engine already dealing turn
    10, which is where the conversion stopped."""
    conv = corpus.converted(56)
    assert conv.failure is None, f"replay 56 stopped at {conv.failure}"
    assert conv.entries_converted == conv.entries_total == 153
