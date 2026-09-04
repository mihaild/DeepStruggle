"""Parse human Twilight Struggle games from the ts-replayer log format, and verify them.

The logs (see tools/download_ts_replayer.py) record each action as a small structured block:

    Turn 1, USSR AR1: Warsaw Pact Formed*: Place Influence (3 Ops):
    USSR +2 in Thailand [0][2]
    USSR +1 in Laos/Cambodia [0][1]

Every influence line carries the *resulting* counts as [US][USSR], and each entry separately
carries the full board, the running score and DEFCON. That redundancy is the point: a parse can
be checked against the log's own arithmetic at every step, so a reconstruction that drifts is
caught immediately instead of silently poisoning a training set. Nothing here trusts the
parser -- `verify_game` re-derives influence, VP and DEFCON and reports every disagreement.

Auxiliary data is extracted too, since it is needed to reconstruct decisions faithfully: coup
rolls with their modifier arithmetic, realignment rolls, war/event die rolls, military ops,
DEFCON transitions and space-race advances.
"""

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import ts_engine as ts

# The log's display names match the engine's for 82 of 84 countries.
_ALIASES = {"uk": "United Kingdom", "dominicanrepublic": "Dominican Rep"}


def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]", "", s.lower())


def _build_country_index() -> Dict[str, int]:
    idx = {}
    for i in range(84):
        idx[_norm(ts.MapData.get_country_info(i)["name"])] = i
    for alias, target in _ALIASES.items():
        idx[alias] = idx[_norm(target)]
    return idx


COUNTRY_INDEX = _build_country_index()


def country_id(name: str) -> Optional[int]:
    return COUNTRY_INDEX.get(_norm(name))


# -- line grammar --------------------------------------------------------------------------

RE_INFLUENCE = re.compile(r"^(US|USSR) ([+-]\d+) in (.+?) \[(\d+)\]\[(\d+)\]$")
RE_MODE = re.compile(r"(Place Influence|Coup|Realignment|Realign|Space Race) \((\d+) Ops\):$")
RE_TARGET = re.compile(r"Target: (.+)$")
RE_WAR = re.compile(r"War in (.+?)\.?$")
RE_REVEALS = re.compile(r"(US|USSR) reveals (.+?)(?: from hand)?\.?$")
RE_RETURNS = re.compile(r"(US|USSR) returns (.+?) to (?:US|USSR)\.?$")
RE_IN_PLAY = re.compile(r"(.+?) is now in play\.")
RE_OUT_OF_PLAY = re.compile(r"(.+?) is no longer in play\.")
RE_COUP_RESULT = re.compile(r"(SUCCESS|FAILURE): (\d+) \[(.*)\] *$")
# "DEFEAT: 2 (-1)  < 4", "VICTORY: 5 >= 3" -- the die a war was decided on, and the
# modifier applied to it. A war with a fixed target rolls inside the event with nothing
# to choose, so this line is the only record of what the humans rolled.
RE_WAR_RESULT = re.compile(
    r"^(VICTORY|DEFEAT): (\d+)(?: \(([+-]?\d+)\))? *(?:>=|<) *(\d+)$")
# The modifier and total are only printed when there is a modifier: "USSR rolls 2" is as
# much a realignment roll as "US rolls 4 (+3) = 7". Both are kept here, in log order, so
# the pair belonging to one realignment stays together.
RE_REALIGN_ROLL = re.compile(r"^(US|USSR) rolls (\d+)(?: \(([+-]\d+)\) = (-?\d+))?$")
RE_DIE = re.compile(r"Die roll: (\d+) -- (Success!|Failed!) \(Needed (\d+) or less\)$")
RE_VP = re.compile(r"(US|USSR) gains (\d+) VP\. Score is (US|USSR) (\d+)\.$")
RE_DEFCON = re.compile(r"DEFCON (degrades|improves) to (\d+)$")
RE_MILOPS = re.compile(r"(US|USSR) Military Ops to (\d+)$")
RE_SPACE = re.compile(r"(US|USSR) advances to (\d+) in the Space Race\.$")
RE_EVENT = re.compile(r"Event: (.+)$")
RE_VP_EVEN = re.compile(r"(US|USSR) gains (\d+) VP\. Score is even\.$")
RE_NO_VP = re.compile(r"No VP awarded\. Score is (?:(US|USSR) (\d+)|even)\.$")
RE_TRAP = re.compile(r"Trap Roll: (\d+) (?:<=|>) (\d+) -- Trap (Escaped|Remains in Effect)$")
RE_BARE_ROLL = re.compile(r"^(US|USSR) rolls (\d+)$")
RE_EFFECT_END = re.compile(r"^(.+) is no longer in play\.$")
RE_HEADLINE = re.compile(r"(US|USSR) Headlines (.+)$")
RE_DISCARD = re.compile(r"(US|USSR) discards? (.+?)\.?$")
RE_PLAYS = re.compile(r"(US|USSR) plays (.+?)\.?$")


@dataclass
class Section:
    """One Ops header and the lines under it, in log order."""
    mode: str
    ops: int
    event: bool                      # header appeared inside the event's own text
    influence: List[Tuple[str, int, int, int, int]] = field(default_factory=list)
    targets: List[int] = field(default_factory=list)


@dataclass
class Entry:
    turn: int
    player: str
    phase: str
    card: Optional[str]
    score: Optional[int]
    defcon: Optional[int]
    mode: Optional[str] = None
    ops: Optional[int] = None
    influence: List[Tuple[str, int, int, int, int]] = field(default_factory=list)
    ops_influence: List[Tuple[str, int, int, int, int]] = field(default_factory=list)
    coup_target: Optional[int] = None
    targets: List[int] = field(default_factory=list)
    # Country a war event is fought in, from "War in India". The engine asks for it as a
    # POINT_NODE while the card resolves, so it belongs to the event queue, not the Ops queue.
    war_targets: List[int] = field(default_factory=list)
    # A second Ops header inside the event section, e.g. Che's free coup for the USSR.
    event_mode: Optional[str] = None
    event_ops: Optional[int] = None
    # Every Ops header in the entry, in order, each owning the lines that follow it. One entry
    # can hold several: CIA Created reveals the USSR hand, gives the US 1 Op to coup with, and
    # only then spends its own Op for the USSR. Keeping a single mode dropped the second.
    sections: List["Section"] = field(default_factory=list)
    # Ongoing effects the entry starts or ends. An effect that should have expired but did not
    # can make later play illegal outright -- a stale Cuban Missile Crisis turns every USSR
    # coup into an instant loss -- so expiry has to be reconciled from the log like the board.
    # The entry's narration verbatim. Where a field cannot say it, the words can: Summit
    # states its winner only when it has one, so the outcome has to be read from the text
    # after "Event: Summit" rather than from the entry's VP lines, which a headline's other
    # card also writes into.
    text: str = ""
    # The opening placement shares this entry with the turn 1 headline in every game in the
    # corpus, and is spread differently from an ordinary placement -- see point_queue.
    setup: bool = False
    in_play: List[str] = field(default_factory=list)
    out_of_play: List[str] = field(default_factory=list)
    # Cards named by a "reveals" line. Usually informational -- Lone Gunman and CIA Created
    # reveal a whole hand -- but where the engine asks which card to hand over, this is the
    # answer: Missile Envy's tie between two 3 Ops cards is settled by what the log reveals.
    revealed: List[Tuple[str, str]] = field(default_factory=list)
    # A card handed over and given back unplayed: "US returns Brezhnev Doctrine* to USSR" is
    # the US declining what Grain Sales offered and taking that card's Ops instead. Without it
    # the choice is invisible and the wrong branch plays the opponent's card.
    returned_card: Optional[str] = None
    coup_roll: Optional[int] = None
    # (die, succeeded) per coup resolved in this entry, in log order. coup_roll above
    # keeps only the last, and an entry can hold several -- Che coups twice.
    coup_rolls: List[Tuple[int, bool]] = field(default_factory=list)
    coup_success: Optional[bool] = None
    realign_rolls: List[Tuple[str, int, int, int]] = field(default_factory=list)
    die_rolls: List[Tuple[int, bool, int]] = field(default_factory=list)
    vp_gains: List[Tuple[str, int, str, int]] = field(default_factory=list)
    defcon_changes: List[Tuple[str, int]] = field(default_factory=list)
    milops: List[Tuple[str, int]] = field(default_factory=list)
    space: List[Tuple[str, int]] = field(default_factory=list)
    events: List[str] = field(default_factory=list)
    trap_rolls: List[Tuple[int, int, bool]] = field(default_factory=list)
    # (die, modifier, won) per war resolved in this entry, in log order.
    war_rolls: List[Tuple[int, int, bool]] = field(default_factory=list)
    bare_rolls: List[Tuple[str, int]] = field(default_factory=list)
    effects_ended: List[str] = field(default_factory=list)
    score_assertions: List[int] = field(default_factory=list)
    headlines: Dict[str, str] = field(default_factory=dict)
    discards: List[Tuple[str, str]] = field(default_factory=list)
    event_first: Optional[bool] = None
    # Card named by another card's event, played inside the same entry: UN Intervention makes
    # you name an opponent card and use it for Ops, and the log prints "USSR plays NORAD*".
    played_card: Optional[str] = None
    unparsed: List[str] = field(default_factory=list)


# Lines that carry no state change and need no interpretation.
_IGNORE = re.compile(
    r"^(SETUP:|.* will play as |Handicap influence|Scenario:|Optional Cards|Time per Player|"
    r"Turn \d+, Cleanup|Turn \d+,|.* is now in play\.|.* reveals |.* Headlines |"
    r"Headline Events Revealed|War in |.*discards?|.*Discard|"
    r"Place Influence:|Space Race|The (US|USSR) |\*RESHUFFLE\*|"
    r"(US|USSR) has no cards to reveal|(US|USSR) plays |(US|USSR) may not |"
    r"(US|USSR) cannot |No effect|Effect:|Note:)")


def _as_int(v) -> Optional[int]:
    """Coerce a log scalar to int.

    The log is inconsistently typed: `defcon` is an int on the setup entry and a string on
    every entry after it. An isinstance(int) check silently dropped all the string ones, so
    DEFCON was never reconciled and drifted -- by turn 2 AR1 of replay 100 the engine sat at
    DEFCON 3 where the game was at 4, which bars coups in Asia and made a legal coup on
    Pakistan look illegal.
    """
    if isinstance(v, bool):
        return None
    if isinstance(v, int):
        return v
    if isinstance(v, str):
        t = v.strip()
        if t.lstrip("-").isdigit():
            return int(t)
    return None


def parse_entry(raw: Dict) -> Entry:
    e = Entry(
        turn=int(raw.get("num", 0)) if str(raw.get("num", "")).isdigit() else 0,
        player=str(raw.get("player", "")),
        phase=str(raw.get("phase", "")),
        card=raw.get("card"),
        score=_as_int(raw.get("score")),
        defcon=_as_int(raw.get("defcon")),
        text=str(raw.get("text", "")),
        setup="SETUP:" in str(raw.get("text", "")),
    )
    in_event = False
    section_open = False
    for line in str(raw.get("text", "")).split("\n"):
        line = line.strip()
        if not line:
            continue
        # Playing an opponent's card for Ops asks which resolves first. The log answers it by
        # line order: whichever of "Event:" or the mode header appears first.
        if e.event_first is None:
            if RE_EVENT.search(line):
                e.event_first = True
            elif RE_MODE.search(line):
                e.event_first = False
        if RE_EVENT.search(line):
            in_event = True
            # ...and it closes whatever section was open. Influence lines are hung on the
            # section header above them, but a new "Event:" starts something that is not part
            # of it: at turn 4's headline of replay 129 the US headlines Junta and coups
            # Panama, and the USSR's Liberation Theology then places an Influence in Panama
            # too. Left attached to the coup's section, that placement looked like part of the
            # coup's own result and was never offered as the decision it is.
            section_open = False

        m = RE_RETURNS.search(line)
        if m:
            e.returned_card = m.group(2).strip()
            continue

        m = RE_REVEALS.search(line)
        if m:
            e.revealed.append((m.group(1), m.group(2).strip()))
            continue

        m = RE_OUT_OF_PLAY.search(line)
        if m:
            e.out_of_play.append(m.group(1).strip())
            continue
        m = RE_IN_PLAY.search(line)
        if m:
            e.in_play.append(m.group(1).strip())
            continue

        m = RE_PLAYS.search(line)
        if m and e.played_card is None:
            e.played_card = m.group(2).strip()

        m = RE_INFLUENCE.match(line)
        if m:
            cid = country_id(m.group(3))
            if cid is None:
                e.unparsed.append(line)
            else:
                rec = (m.group(1), int(m.group(2)), cid,
                       int(m.group(4)), int(m.group(5)))
                e.influence.append(rec)
                if not in_event:
                    e.ops_influence.append(rec)
                if e.sections and section_open:
                    e.sections[-1].influence.append(rec)
            continue
        m = RE_MODE.search(line)
        if m:
            mode = {"Place Influence": "influence", "Coup": "coup",
                    "Realignment": "realign", "Realign": "realign",
                    "Space Race": "space"}[m.group(1)]
            e.sections.append(Section(mode=mode, ops=int(m.group(2)), event=in_event))
            section_open = True
            if e.mode is None:
                # First header is how the card itself was played. A later one belongs to the
                # event -- Che grants the USSR a free coup after the US already spent Che's
                # Ops -- so it must not overwrite the play mode or end the event section.
                e.mode, e.ops = mode, int(m.group(2))
                in_event = False
            else:
                e.event_mode, e.event_ops = mode, int(m.group(2))
            continue
        m = RE_WAR.search(line)
        if m:
            wid = country_id(m.group(1))
            if wid is None:
                e.unparsed.append(line)
            else:
                e.war_targets.append(wid)
            continue

        m = RE_TARGET.search(line)
        if m:
            tid = country_id(m.group(1))
            if tid is None:
                e.unparsed.append(line)
            else:
                e.targets.append(tid)
                if e.sections:
                    e.sections[-1].targets.append(tid)
                if e.coup_target is None:
                    e.coup_target = tid
            continue
        m = RE_COUP_RESULT.search(line)
        if m:
            e.coup_success = (m.group(1) == "SUCCESS")
            e.coup_roll = int(m.group(2))
            e.coup_rolls.append((int(m.group(2)), m.group(1) == "SUCCESS"))
            continue
        m = RE_REALIGN_ROLL.match(line)
        if m:
            _roll = int(m.group(2))
            _mod = int(m.group(3)) if m.group(3) else 0
            e.realign_rolls.append((m.group(1), _roll, _mod,
                                    int(m.group(4)) if m.group(4) else _roll))
            continue
        m = RE_DIE.search(line)
        if m:
            e.die_rolls.append((int(m.group(1)), m.group(2) == "Success!", int(m.group(3))))
            continue
        m = RE_VP.search(line)
        if m:
            e.vp_gains.append((m.group(1), int(m.group(2)), m.group(3), int(m.group(4))))
            continue
        m = RE_DEFCON.search(line)
        if m:
            e.defcon_changes.append((m.group(1), int(m.group(2))))
            continue
        m = RE_MILOPS.search(line)
        if m:
            e.milops.append((m.group(1), int(m.group(2))))
            continue
        m = RE_SPACE.search(line)
        if m:
            e.space.append((m.group(1), int(m.group(2))))
            continue
        m = RE_EVENT.search(line)
        if m:
            e.events.append(m.group(1))
            continue
        m = RE_VP_EVEN.search(line)
        if m:
            e.vp_gains.append((m.group(1), int(m.group(2)), "even", 0))
            e.score_assertions.append(0)
            continue
        m = RE_NO_VP.search(line)
        if m:
            val = int(m.group(2)) if m.group(2) else 0
            e.score_assertions.append(val if m.group(1) == "US" else -val)
            continue
        m = RE_TRAP.search(line)
        if m:
            e.trap_rolls.append((int(m.group(1)), int(m.group(2)),
                                 m.group(3) == "Escaped"))
            continue
        m = RE_WAR_RESULT.match(line)
        if m:
            e.war_rolls.append((int(m.group(2)), int(m.group(3) or 0),
                                m.group(1) == "VICTORY"))
            continue
        m = RE_BARE_ROLL.match(line)
        if m:
            e.bare_rolls.append((m.group(1), int(m.group(2))))
            continue
        m = RE_HEADLINE.search(line)
        if m:
            e.headlines[m.group(1)] = m.group(2).strip()
            continue
        m = RE_DISCARD.search(line)
        if m:
            # A discarded card leaves the hand just as a played one does. Missing these left
            # scoring cards in the reconstructed hand, and the engine correctly ended the game
            # at end of turn for holding one -- replay 100 turn 1 AR6, where Five Year Plan
            # made the USSR discard Mideast Scoring.
            e.discards.append((m.group(1), m.group(2).strip()))
            continue
        m = RE_EFFECT_END.match(line)
        if m:
            e.effects_ended.append(m.group(1))
            continue
        if not _IGNORE.match(line):
            e.unparsed.append(line)
    return e


# -- verification --------------------------------------------------------------------------

def initial_board() -> Dict[int, Tuple[int, int]]:
    """Fixed setup influence, present before either player places anything.

    Starting a running board at zero makes the first influence line for every pre-placed
    country (East Germany, Iran, the UK...) look like a mismatch. It is not; the log is
    counting from the real starting position.
    """
    st = ts.GameState()
    ts.Engine.init_game(st, 1)
    return {c: (int(st.get_country(c).us_influence), int(st.get_country(c).ussr_influence))
            for c in range(84)}


@dataclass
class GameCheck:
    replay_id: int
    entries: int = 0
    influence_lines: int = 0
    influence_mismatch: int = 0
    board_checked: int = 0
    board_mismatch: int = 0
    vp_checked: int = 0
    vp_mismatch: int = 0
    defcon_checked: int = 0
    defcon_mismatch: int = 0
    coup_rolls: int = 0
    realign_rolls: int = 0
    die_rolls: int = 0
    trap_rolls: int = 0
    effects_ended: int = 0
    unparsed_lines: int = 0
    notes: List[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return (self.influence_mismatch == 0 and self.board_mismatch == 0
                and self.vp_mismatch == 0 and self.defcon_mismatch == 0
                and self.unparsed_lines == 0)


def verify_game(game: Dict, max_notes: int = 6) -> Tuple[GameCheck, List[Entry]]:
    """Re-derive influence, VP and DEFCON from the parsed moves and compare to the log."""
    chk = GameCheck(replay_id=int(game.get("replay_id", -1)))
    board: Dict[int, Tuple[int, int]] = initial_board()
    parsed: List[Entry] = []

    for raw in game.get("all_turns", []):
        e = parse_entry(raw)
        parsed.append(e)
        chk.entries += 1
        chk.unparsed_lines += len(e.unparsed)
        if e.unparsed and len(chk.notes) < max_notes:
            chk.notes.append(f"unparsed: {e.unparsed[0][:70]}")
        chk.coup_rolls += 1 if e.coup_roll is not None else 0
        chk.realign_rolls += len(e.realign_rolls)
        chk.die_rolls += len(e.die_rolls) + len(e.bare_rolls)
        chk.trap_rolls += len(e.trap_rolls)
        chk.effects_ended += len(e.effects_ended)

        # 1. every influence line's bracketed result must match our running board
        for side, delta, cid, res_us, res_ussr in e.influence:
            chk.influence_lines += 1
            us, ussr = board.get(cid, (0, 0))
            if side == "US":
                us += delta
            else:
                ussr += delta
            if (us, ussr) != (res_us, res_ussr):
                chk.influence_mismatch += 1
                if len(chk.notes) < max_notes:
                    nm = ts.MapData.get_country_info(cid)["name"]
                    chk.notes.append(
                        f"T{e.turn} {nm}: derived [{us}][{ussr}] vs logged "
                        f"[{res_us}][{res_ussr}]")
                us, ussr = res_us, res_ussr      # resync so one error is not amplified
            board[cid] = (us, ussr)

        # 2. the entry's own board snapshot must agree with the running board
        for key, c in (raw.get("countries") or {}).items():
            cid = country_id(key)
            if cid is None:
                continue
            try:
                lus, lussr = int(c["inflUS"]), int(c["inflUSSR"])
            except (KeyError, TypeError, ValueError):
                continue
            chk.board_checked += 1
            if board.get(cid, (0, 0)) != (lus, lussr):
                chk.board_mismatch += 1
            board[cid] = (lus, lussr)            # the snapshot is authoritative

        # 3. VP: the running score quoted on each VP line must match the entry's score
        quoted = [(0 if ss == "even" else (sv if ss == "US" else -sv))
                  for _, _, ss, sv in e.vp_gains] + e.score_assertions
        if quoted and e.score is not None:
            chk.vp_checked += 1
            if quoted[-1] != e.score:
                chk.vp_mismatch += 1
                if len(chk.notes) < max_notes:
                    chk.notes.append(f"T{e.turn} score: quoted {quoted[-1]} vs entry "
                                     f"{e.score}")

        # 4. DEFCON transitions must land on the entry's DEFCON
        # Only the last transition in an entry should match the entry's DEFCON: an entry can
        # improve and then degrade, and the field records where it ended up.
        if e.defcon_changes and e.defcon is not None:
            chk.defcon_checked += 1
            # On the last action round of a turn the line records the mid-turn change and the
            # field records the value after end-of-turn cleanup, which improves DEFCON by one.
            # Both are right; they describe different moments.
            line_value = e.defcon_changes[-1][1]
            if e.defcon not in (line_value, min(5, line_value + 1)):
                chk.defcon_mismatch += 1
                if len(chk.notes) < max_notes:
                    chk.notes.append(f"T{e.turn} defcon: line says {line_value} "
                                     f"vs entry {e.defcon}")

    return chk, parsed
