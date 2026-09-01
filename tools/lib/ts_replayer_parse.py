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
RE_MODE = re.compile(r"^(Place Influence|Coup|Realignment|Realign) \((\d+) Ops\):$")
RE_TARGET = re.compile(r"^Target: (.+)$")
RE_COUP_RESULT = re.compile(r"(SUCCESS|FAILURE): (\d+) \[(.*)\] *$")
RE_REALIGN_ROLL = re.compile(r"^(US|USSR) rolls (\d+) \(([+-]\d+)\) = (-?\d+)$")
RE_DIE = re.compile(r"Die roll: (\d+) -- (Success!|Failed!) \(Needed (\d+) or less\)$")
RE_VP = re.compile(r"(US|USSR) gains (\d+) VP\. Score is (US|USSR) (\d+)\.$")
RE_DEFCON = re.compile(r"DEFCON (degrades|improves) to (\d+)$")
RE_MILOPS = re.compile(r"(US|USSR) Military Ops to (\d+)$")
RE_SPACE = re.compile(r"(US|USSR) advances to (\d+) in the Space Race\.$")
RE_EVENT = re.compile(r"^Event: (.+)$")
RE_VP_EVEN = re.compile(r"(US|USSR) gains (\d+) VP\. Score is even\.$")
RE_NO_VP = re.compile(r"No VP awarded\. Score is (?:(US|USSR) (\d+)|even)\.$")
RE_TRAP = re.compile(r"Trap Roll: (\d+) (?:<=|>) (\d+) -- Trap (Escaped|Remains in Effect)$")
RE_BARE_ROLL = re.compile(r"^(US|USSR) rolls (\d+)$")
RE_EFFECT_END = re.compile(r"^(.+) is no longer in play\.$")


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
    coup_target: Optional[int] = None
    coup_roll: Optional[int] = None
    coup_success: Optional[bool] = None
    realign_rolls: List[Tuple[str, int, int, int]] = field(default_factory=list)
    die_rolls: List[Tuple[int, bool, int]] = field(default_factory=list)
    vp_gains: List[Tuple[str, int, str, int]] = field(default_factory=list)
    defcon_changes: List[Tuple[str, int]] = field(default_factory=list)
    milops: List[Tuple[str, int]] = field(default_factory=list)
    space: List[Tuple[str, int]] = field(default_factory=list)
    events: List[str] = field(default_factory=list)
    trap_rolls: List[Tuple[int, int, bool]] = field(default_factory=list)
    bare_rolls: List[Tuple[str, int]] = field(default_factory=list)
    effects_ended: List[str] = field(default_factory=list)
    score_assertions: List[int] = field(default_factory=list)
    unparsed: List[str] = field(default_factory=list)


# Lines that carry no state change and need no interpretation.
_IGNORE = re.compile(
    r"^(SETUP:|.* will play as |Handicap influence|Scenario:|Optional Cards|Time per Player|"
    r"Turn \d+, Cleanup|Turn \d+,|.* is now in play\.|.* reveals |.* Headlines |"
    r"Headline Events Revealed|War in |VICTORY|DEFEAT|.*discards?|.*Discard|"
    r"Place Influence:|Space Race|The (US|USSR) |\*RESHUFFLE\*|"
    r"(US|USSR) has no cards to reveal|(US|USSR) plays |(US|USSR) may not |"
    r"(US|USSR) cannot |No effect|Effect:|Note:)")


def parse_entry(raw: Dict) -> Entry:
    e = Entry(
        turn=int(raw.get("num", 0)) if str(raw.get("num", "")).isdigit() else 0,
        player=str(raw.get("player", "")),
        phase=str(raw.get("phase", "")),
        card=raw.get("card"),
        score=raw.get("score") if isinstance(raw.get("score"), int) else None,
        defcon=raw.get("defcon") if isinstance(raw.get("defcon"), int) else None,
    )
    for line in str(raw.get("text", "")).split("\n"):
        line = line.strip()
        if not line:
            continue

        m = RE_INFLUENCE.match(line)
        if m:
            cid = country_id(m.group(3))
            if cid is None:
                e.unparsed.append(line)
            else:
                e.influence.append((m.group(1), int(m.group(2)), cid,
                                    int(m.group(4)), int(m.group(5))))
            continue
        m = RE_MODE.match(line)
        if m:
            e.mode = {"Place Influence": "influence", "Coup": "coup",
                      "Realignment": "realign", "Realign": "realign"}[m.group(1)]
            e.ops = int(m.group(2))
            continue
        m = RE_TARGET.match(line)
        if m:
            e.coup_target = country_id(m.group(1))
            if e.coup_target is None:
                e.unparsed.append(line)
            continue
        m = RE_COUP_RESULT.search(line)
        if m:
            e.coup_success = (m.group(1) == "SUCCESS")
            e.coup_roll = int(m.group(2))
            continue
        m = RE_REALIGN_ROLL.match(line)
        if m:
            e.realign_rolls.append((m.group(1), int(m.group(2)), int(m.group(3)),
                                    int(m.group(4))))
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
        m = RE_EVENT.match(line)
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
        m = RE_BARE_ROLL.match(line)
        if m:
            e.bare_rolls.append((m.group(1), int(m.group(2))))
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
        for _, value in e.defcon_changes:
            chk.defcon_checked += 1
            if e.defcon is not None and value != e.defcon:
                chk.defcon_mismatch += 1

    return chk, parsed
