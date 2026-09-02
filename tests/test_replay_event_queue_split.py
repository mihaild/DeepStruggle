"""Which of an entry's influence lines are targets someone chose, and which are die results.

The converter keeps two target queues. The Ops queue holds what a "Coup"/"Realignment"/"Place
Influence" header covers, and a target taken from it is understood to roll, so the logged
outcome is used to steer the die. The event queue holds an event's own placements, which roll
nothing. Putting a row in the wrong one is silent: the entry still converts, but with a die the
engine chose freely instead of the one the human rolled.
"""
from tools.lib.ts_replayer_convert import event_queue, section_queue
from tools.lib.ts_replayer_parse import Entry, country_id, parse_entry

CHE_TURN_6_AR7 = """Turn 6, US AR7: Che: Realignment (3 Ops):
Target: Cuba
USSR rolls 1 (+1) = 2
US rolls 4 (+3) = 7
USSR -4 in Cuba [0][0]

Event: Che
Coup (2 Ops):
Target: Haiti
SUCCESS: 2 [ + 2 - 2x1 = 2 ]
US -2 in Haiti [1][0]
USSR Military Ops to 5

Coup (2 Ops):
Target: Nicaragua
SUCCESS: 2 [ + 2 - 2x1 = 2 ]
US -2 in Nicaragua [3][0]
USSR Military Ops to 5
"""

MARSHALL_PLAN_TURN_1_AR3 = """Turn 1, USSR AR3: Marshall Plan*: Place Influence (4 Ops):
USSR +4 in France [2][5]

Event: Marshall Plan*
US +1 in Spain/Portugal [1][0]
US +1 in Italy [5][0]
US +1 in Turkey [1][0]
US +1 in Greece [1][0]
US +1 in West Germany [5][0]
US +1 in UK [6][0]
US +1 in Canada [3][0]
Marshall Plan* is now in play.
"""

SOUTH_AFRICAN_UNREST = """Turn 4, US AR7: South African Unrest: Realignment (3 Ops):
Target: South Africa
USSR rolls 6
US rolls 1 (+2) = 3
US -3 in South Africa [0][1]

Event: South African Unrest
USSR +2 in Angola [3][2]
"""


def _entry(text: str, player: str = "US", card: str = "Che") -> Entry:
    return parse_entry({"num": "7", "player": player, "phase": "AR7",
                        "card": card, "text": text})


def _cid(name: str) -> int:
    cid = country_id(name)
    assert cid is not None, f"unknown country {name!r}"
    return cid


def test_a_free_coup_inside_an_event_is_not_an_event_placement() -> None:
    """Replay 113 turn 6 AR7: Che's two coups belong to the Ops queue, headers and all."""
    e = _entry(CHE_TURN_6_AR7)
    haiti, nicaragua = _cid("Haiti"), _cid("Nicaragua")
    assert event_queue(e) == [], (
        "a coup's removed influence is the result of a die, not a target anyone chose")
    coups = [s for s in e.sections if s.mode == "coup"]
    assert [section_queue(s) for s in coups] == [[haiti], [nicaragua]], (
        "each free coup carries its own target")


def test_the_coup_rows_are_not_expanded_by_their_delta() -> None:
    """Two Influence removed from one coup is one target, not two placements."""
    e = _entry(CHE_TURN_6_AR7)
    assert _cid("Haiti") not in event_queue(e)
    assert len(event_queue(e)) == 0


def test_an_events_placements_under_an_ops_header_stay_in_the_event_queue() -> None:
    """Replay 100 turn 1 AR3: the parser hangs every line on the header above it.

    Marshall Plan's seven US placements follow the USSR's own "Place Influence (4 Ops)"
    header, so they are attached to that section though they are not part of it. They are
    still the US's to choose.
    """
    e = _entry(MARSHALL_PLAN_TURN_1_AR3, player="USSR", card="Marshall Plan*")
    q = event_queue(e)
    assert len(q) == 7, f"all seven event placements must be offered, got {q}"
    assert _cid("France") not in q, "the USSR's own Ops are not an event placement"
    for name in ("Spain/Portugal", "Italy", "Turkey", "Greece", "West Germany",
                 "UK", "Canada"):
        assert _cid(name) in q, f"{name} missing from the event queue"


def test_a_placement_into_a_country_the_ops_realigned_still_counts_as_a_placement() -> None:
    """Replay 101 turn 4 AR7: the event places where the Ops rolled, and that placement
    rolls nothing."""
    e = _entry(SOUTH_AFRICAN_UNREST, card="South African Unrest")
    angola, south_africa = _cid("Angola"), _cid("South Africa")
    q = event_queue(e)
    assert q.count(angola) == 2, f"two Influence placed in Angola, got {q}"
    assert south_africa not in q, "the realignment's own removal is not a placement"
