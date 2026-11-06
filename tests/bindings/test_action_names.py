"""Every flat slot has a readable name in replays: the P17 heads (DEFCON value, Chernobyl's region)
used to print as "UnknownAction #214", which hid the choice from anyone reading a replay."""

from __future__ import annotations

import ts_engine as ts
from bindings.action_encoder import REGION_NAMES, ActionEncoder


def test_the_late_slots_have_names() -> None:
    st = ts.GameState()
    names = [ActionEncoder.get_action_name(st, i) for i in range(ActionEncoder.FLAT_ACTION_SIZE)]
    assert not [n for n in names if n.startswith("UnknownAction")]
    assert names[ActionEncoder.DEFCON_VALUE_OFFSET] == "SetDEFCON 1"
    assert names[ActionEncoder.REGION_OFFSET] == "Region #0 (Europe)"
    assert names[ActionEncoder.BASE_ACTION_SIZE - 1] == f"Region #5 ({REGION_NAMES[5]})"
    assert names[ActionEncoder.MACRO_NODE_OFFSET + 7].startswith("UntilControl #7 (")


def test_region_names_follow_the_engine_enum() -> None:
    assert [int(getattr(ts.Region, n)) for n in ("EUROPE", "ASIA", "MIDDLE_EAST", "AFRICA",
                                                   "CENTRAL_AMERICA", "SOUTH_AMERICA")] == list(range(6))
    assert len(REGION_NAMES) == 6
