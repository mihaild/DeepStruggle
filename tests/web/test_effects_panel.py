"""The page's "in effect" panel (web/ui/src/cards_view.ts) covers every effect the engine exports.

The display state lists the set flags by the engine's names (bindings/state_json.cpp, CHECK_FLAG),
and the panel shows a flag only when EFFECT_INFO_MAP has an entry for it and the English catalog
(effects.json) has its words. A flag missing from either is silently not shown: The Iron Lady and
North Sea Oil blocked Socialist Governments and OPEC with nothing on the page to say so, and the
Iranian Hostage Crisis entry was keyed under another name than the engine's.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Set

REPO = Path(__file__).resolve().parents[2]

#: Flags the display state carries that are not effects a player plays under: how the game ended,
#: and the Space Race's attempts this turn (shown on the track).
NOT_EFFECTS = {
    "CMC_SUICIDE_LOSS", "DEFCON_SUICIDE_PROVOKED", "EUROPE_CONTROL_WIN",
    "SPACE_US_ATTEMPT_1", "SPACE_US_ATTEMPT_2", "SPACE_USSR_ATTEMPT_1", "SPACE_USSR_ATTEMPT_2",
}


def exported() -> Set[str]:
    source = (REPO / "bindings" / "state_json.cpp").read_text()
    return set(re.findall(r"CHECK_FLAG\((\w+)\);", source))


def panel() -> Set[str]:
    source = (REPO / "web" / "ui" / "src" / "cards_view.ts").read_text()
    block = source.split("export const EFFECT_INFO_MAP", 1)[1].split("\n};", 1)[0]
    return set(re.findall(r"^  (\w+): \{", block, re.M))


def catalog() -> Set[str]:
    return set(json.loads((REPO / "web" / "ui" / "src" / "i18n" / "locales" / "en" / "effects.json").read_text()))


def test_every_exported_effect_is_on_the_panel() -> None:
    flags = exported()
    assert NOT_EFFECTS <= flags, "a listed non-effect is no longer exported"
    assert flags - NOT_EFFECTS == panel()


def test_every_panel_entry_has_its_words() -> None:
    assert panel() == catalog()
