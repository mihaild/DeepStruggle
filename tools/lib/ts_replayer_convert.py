"""Drive the engine through a logged human game, emitting (observation, action) pairs.

The engine deals from a seed, so a human game cannot be reproduced by seeding: hands are
forced to the logged ones at each turn boundary, and the board is reconciled to the log after
every entry. Die outcomes will differ from the human game -- what behaviour cloning needs is a
faithful *observation* at each decision, not a bit-identical simulation, and reconciling keeps
the observation faithful even when a coup roll goes the other way.

Nothing is approximated. If the engine asks something the log does not determine, or the log
states something the engine will not do, conversion of that game stops with a ConversionFailure
naming the turn and action round -- a dataset is only worth training on if every decision in it
is the one the human actually made, and a plausible substitute is indistinguishable from a real
one once it is in the file. The single approved exception is Our Man in Tehran, whose log lines
record the discards but never the five revealed cards, so the rest of that peek is invented.

Failures are reported with replay id and turn/action round rather than counted, since a
systematic failure in one card's handling looks identical to noise in a summary statistic.
"""

import hashlib
import json
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import ts_engine as ts

from tools.lib.ts_replayer_parse import (Entry, RE_PASSED_ROUND, country_id,
                                        parse_entry)


def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]", "", s.lower())


def _build_cards() -> Tuple[Dict[str, int], Dict[str, int]]:
    idx = {}
    for c in range(1, 111):
        idx[_norm(ts.CardData.get_card_info(c)["name"])] = c
    idx["mideastscoring"] = idx[_norm("Middle East Scoring")]
    # The log truncates this title; match a distinctive prefix.
    prefix = {"asknotwhatyourcountry": idx[_norm("“Ask Not What Your Country Can Do For You…”")]}
    return idx, prefix


CARD_INDEX, CARD_PREFIX = _build_cards()

# Ongoing effects whose expiry the log records and whose lingering presence changes what is
# legal rather than merely what is scored. Both Cuban Missile Crisis bits are cleared together
# because the log names the card, not the side that played it.
_EFFECT_BITS: Dict[str, Tuple[int, ...]] = {
    _norm("Cuban Missile Crisis*"): (ts.EffectBits.CMC_ACTIVE_US,
                                     ts.EffectBits.CMC_ACTIVE_USSR),
    _norm("Bear Trap*"): (ts.EffectBits.BEAR_TRAP_ACTIVE,),
    _norm("Quagmire*"): (ts.EffectBits.QUAGMIRE_ACTIVE,),
}


def card_id(name: Optional[str]) -> Optional[int]:
    if not name:
        return None
    n = _norm(name)
    if n in CARD_INDEX:
        return CARD_INDEX[n]
    for pref, cid in CARD_PREFIX.items():
        if n.startswith(pref):
            return cid
    return None


class ConversionFailure(Exception):
    """A logged entry could not be reproduced exactly.

    The dataset is only worth training on if every decision in it is the one the human made,
    so there is no approximating here: anything the log states and the engine cannot follow
    stops the game's conversion and is reported with its turn and action round. The single
    approved exception is Our Man in Tehran, where the log records the discards but never the
    five revealed cards, so the remainder of the peeked set is invented.
    """

    def __init__(self, mismatch: "Mismatch") -> None:
        super().__init__(str(mismatch))
        self.mismatch = mismatch


@dataclass
class Mismatch:
    replay_id: int
    turn: int
    phase: str
    player: str
    card: Optional[str]
    kind: str
    detail: str

    def __str__(self) -> str:
        return (f"replay {self.replay_id} T{self.turn} {self.phase} {self.player} "
                f"card={self.card!r}: {self.kind} -- {self.detail}")


# Entries whose text the log itself never finishes. The game is converted up to the entry
# before one of these and no further: the entry records a decision whose outcome the log does
# not state, so neither it nor anything after it can be reconstructed, and none of it is
# training data. This is a property of the recording, not a defect in the engine or the driver,
# so it is listed rather than diagnosed again each time.
#
#   replay 60, turn 7 AR6 (USSR) -- the text reads "Coup (4 Ops):" and stops there. No target,
#   no roll, no result, and no further entries in the file.
#
#   replay 133, turn 10 headline -- the last entry in the file, and it holds one line: "USSR
#   Headlines Missile Envy". The US headline is not recorded, nor which of the two 4 Ops cards
#   the US handed over when Missile Envy asked (ABM Treaty or Muslim Revolution), nor anything
#   either event did.
#
#   replay 55, turn 9 AR7 (US) -- the last entry, and it stops one line into Tear Down This
#   Wall: the 3 US Influence it puts into East Germany is recorded, then "Coup (3 Ops):" and
#   nothing. The free coup in Europe that the card grants has no target, no roll and no result,
#   and the engine taking one of its own removed 2 USSR Influence from East Germany -- which
#   the log has standing at 5 since turn 3 AR2 and never moving again.
#
# The entry named is where the unfinished tail begins, and everything from it to the end of the
# file belongs to that same turn. Whichever entry is named, the whole of its turn is given back:
# see _rewind_to_turn_start.
#
# Replay 148 needs no listing -- its last entry ends on a bare "Turn 4, USSR AR2" header, which
# _is_the_record_ending recognises on its own. The player is part of the key because an action round holds an
# entry for each side and only one of them need be cut short: replay 55's USSR half of turn 9
# AR7 is complete and converts.
_KNOWN_INCOMPLETE: Dict[int, Set[Tuple[int, str, str]]] = {
    60: {(7, "AR6", "USSR")},
    133: {(10, "Headline", "both")},
    55: {(9, "AR7", "US")},
}

# Scores the engine and the log disagree on because a choice the engine does not offer was
# made differently, not because either is wrong. The engine's score is replaced by the log's
# and the game carries on. Listed rather than diagnosed again each time, and kept small: a
# disagreement that is not understood belongs in a failure, not here.
#
#   replay 127, turn 5 AR3 -- Asia Scoring with Shuttle Diplomacy in play. The card subtracts
#   one USSR battleground country from the USSR's total, and *which* one is the US player's
#   choice. Taking Japan also costs the USSR the superpower-adjacent Influence there, which is
#   why it is the choice to make; the engine assumes it and scores the USSR 5. This US took a
#   different battleground and left Japan alone, so the log scores the USSR 6, one more than
#   the best play would have allowed. Legal, and not reconstructible until Shuttle Diplomacy's
#   target is a decision the engine asks for.
_KNOWN_SCORE: Dict[int, Dict[Tuple[int, str, str], int]] = {
    127: {(5, "AR3", "USSR"): -5},
}


# The engine's opening handicap is the tournament one: 2 extra US Influence, placed where the
# US already has some. Nine of the corpus's 287 games were played with a different one and are
# not reconstructible without making the handicap a setup parameter -- which is not worth
# doing, since a handicap other than 2 says the players were mismatched and the positions it
# produces are not ones the engine will ever play from.
_STANDARD_HANDICAP = "US +2"
# The corpus writes the opening handicap two ways. Most games state it outright; the rest bid
# for sides, and the winning bid becomes the same extra Influence -- "lkslks bids 1 Influence
# for USSR ... Additional Influence from bidding: US +1". Both are the same thing to the setup,
# and reading only the first left the 8 bidding games that did not land on 2 looking standard:
# at turn 1 of replay 230 the US has 8 Influence to place where the engine offers 9, and the
# handicap stage was left with a placement the log never made.
RE_HANDICAP = re.compile(r"Handicap influence: (\S+ [+-]?\d+)")
RE_BID = re.compile(r"Additional Influence from bidding: (\S+ [+-]?\d+)")


def unsupported_handicap(raws: List[Dict]) -> Optional[str]:
    """The game's handicap, if it is one the engine cannot set up. None when it can."""
    if not raws:
        return None
    text = str(raws[0].get("text", ""))
    m = RE_HANDICAP.search(text) or RE_BID.search(text)
    if m is None or m.group(1) == _STANDARD_HANDICAP:
        return None
    return m.group(1)


def distinct_replays(games: Dict[int, List[Dict]]) -> Dict[int, int]:
    """Map each replay id to the id whose game it duplicates, itself when it is the original.

    The download holds 287 files but only 254 games. Twenty-four of them arrived more than
    once, one of them nine times over (replays 90, 214-218, 254, 298 and 309 are one game),
    and a game counted nine times is nine times the weight in anything trained on it. The
    duplicates are exact -- the same entries, the same text -- so they are found by content
    and not by a list that would have to be maintained.

    The lowest id wins, so the choice does not move when a file is added or removed.
    """
    first: Dict[str, int] = {}
    out: Dict[int, int] = {}
    for rid in sorted(games):
        digest = hashlib.sha256(
            json.dumps(games[rid], sort_keys=True).encode()).hexdigest()
        out[rid] = first.setdefault(digest, rid)
    return out


@dataclass
class Conversion:
    replay_id: int
    samples: List[Tuple[np.ndarray, np.ndarray, int, int]] = field(default_factory=list)
    entries_total: int = 0
    entries_converted: int = 0
    entries_guessed: int = 0
    decisions_emitted: int = 0
    board_resyncs: int = 0
    # Set when the game is not converted at all, with the reason. Distinct from `failure`,
    # which means the conversion was attempted and stopped somewhere.
    skipped: Optional[str] = None
    vp_drift: int = 0
    entries_board_mismatch: int = 0
    hand_misses: int = 0
    # Cards the turn's hand lists gave to the wrong side, corrected from the entries.
    hand_reattributions: int = 0
    # Entries where a listed disagreement (_KNOWN_SCORE) replaced the engine's score.
    scores_forced: int = 0
    # Cards whose event the engine resolved while driving the current entry. Reset per entry.
    events_resolved: Set[int] = field(default_factory=set)
    # (turn, entries converted, samples emitted) as the current turn began.
    turn_started_at: Tuple[int, int, int] = (0, 0, 0)
    # Whether the engine reached a terminal state -- a 20 VP win, DEFCON 1, or final scoring.
    game_ended: bool = False
    first_board_mismatch: Optional[Mismatch] = None
    first_vp_drift: Optional[Mismatch] = None
    # Set when conversion stopped: the entry that could not be reproduced. Entries after it
    # were never attempted, so a Conversion with a failure describes only a prefix of the game.
    failure: Optional[Mismatch] = None
    # Set instead of `failure` where the log simply stops. Everything before it converted; there
    # was nothing after it to convert.
    truncated_at: Optional[Mismatch] = None
    mismatches: List[Mismatch] = field(default_factory=list)


def _reveal_ops_cap(raws, turn: int, side: str) -> Optional[int]:
    """Lowest Ops among cards this side is recorded revealing during the turn.

    Missile Envy takes the opponent's *highest* Ops card, so padding a hand with anything that
    matches or beats what the log says was handed over would change the decision -- either
    making a different card the maximum, or creating a tie the log never records a choice for.
    Padding below this leaves the logged card the only one Missile Envy could have taken.
    """
    cap: Optional[int] = None
    for raw in raws:
        e = parse_entry(raw)
        if e.turn != turn:
            continue
        for rev_side, name in (e.revealed or []):
            if rev_side != side:
                continue
            cid = card_id(name)
            if cid is None:
                continue
            ops = int(ts.CardData.get_card_info(cid)["ops"])
            cap = ops if cap is None else min(cap, ops)
    return cap


_MISSILE_ENVY_REVEAL = re.compile(r"^(US|USSR) reveals (.+?)(?: from hand)?\.?$", re.M)


def _taken_from_hand(e: Entry) -> List[Tuple[str, int]]:
    """Cards this entry moves out of a hand without their owner playing them.

    Two events reach into the opponent's hand and take a card: Missile Envy takes the highest
    Ops one, Grain Sales To Soviets takes one at random. Either way it is gone from that hand
    for the rest of the turn, while the turn's list still names it -- it was theirs when the
    turn began and it became visible while they held it.

    Leaving them in put cards back that had changed sides, and the two show it differently. For
    Missile Envy it is what the opponent is offered next: at turn 5 AR1 of replay 114 the US
    takes Nuclear Test Ban, and the USSR, caught by Bear Trap with nothing left to discard and
    only a scoring card in hand, was still being offered it two action rounds later. For Grain
    Sales it is what the opponent has left at all: at turn 8 AR1 of replay 51 the US draws
    Puppet Governments out of the USSR hand and plays it, and at the USSR's own AR7 -- where the
    log reads "USSR has no cards to discard" -- Five Year Plan found it still sitting there and
    fired its event.
    """
    out: List[Tuple[str, int]] = []
    for marker in ("Event: Missile Envy", "Event: Grain Sales To Soviets"):
        took = _revealed_under(e, marker)
        if took is not None:
            out.append(took)
    return out


def _missile_envy_took(e: Entry) -> Optional[Tuple[str, int]]:
    """The card Missile Envy took, as (the side that lost it, its id), or None."""
    return _revealed_under(e, "Event: Missile Envy")


def _revealed_under(e: Entry, marker: str) -> Optional[Tuple[str, int]]:
    """The first card revealed after `marker`, as (the side that revealed it, its id).

    Read from the line after the event's own header rather than from the entry's reveals as a
    whole, because other things reveal cards and one of them can share the entry. At turn 5 of
    replay 141 the USSR headlines "Lone Gunman", which reveals the entire US hand, against the
    US's Missile Envy -- and taking every reveal as exchanged emptied the US hand outright, so
    the UN Intervention they played two action rounds later was not there to play.
    """
    at = (e.text or "").find(marker)
    if at < 0:
        return None
    m = _MISSILE_ENVY_REVEAL.search(e.text, at)
    if m is None:
        return None
    cid = card_id(m.group(2).strip())
    return (m.group(1), cid) if cid else None


_OP_MODE_NAMES = {0: "influence", 1: "coup", 2: "realignment"}
_PLAY_MODE_NAMES = {0: "event", 1: "ops", 2: "space race", 3: "pass"}


def _name_options(state: ts.GameState, dt: "ts.DecisionType",
                  legal: "np.ndarray") -> List[str]:
    """The choices on offer, named the way the decision itself names them."""
    out: List[str] = []
    for raw_action in legal:
        a = int(raw_action)
        if dt == ts.DecisionType.SELECT_CARD:
            out.append("pass" if a == _PASS
                       else str(ts.CardData.get_card_info(a + 1)["name"]))
            continue
        primary = int(ts.ActionMask.decode_flat_action(state, a).primary_id)
        if dt == ts.DecisionType.POINT_NODE and 0 <= primary < 84:
            out.append(str(ts.MapData.get_country_info(primary)["name"]))
        elif dt == ts.DecisionType.SELECT_OP_MODE:
            out.append(_OP_MODE_NAMES.get(primary, f"mode {primary}"))
        elif dt == ts.DecisionType.SELECT_PLAY_MODE:
            out.append(_PLAY_MODE_NAMES.get(a - 110, f"play mode {a}"))
        else:
            out.append(f"{str(dt).split('.')[-1].lower()} {primary}")
    return out


def event_point_queues(e: Entry) -> Dict[int, List[int]]:
    """The placements each event printed, by the card that printed them.

    A headline resolves two cards and the driver held their placements in one queue, taking
    whichever target the engine would accept. That is fine while the two events want different
    countries and wrong the moment they overlap, because a point one event could not use stays
    in the queue and the next event spends it.

    At turn 8's headline of replay 150 the US's East European Unrest removes 2 USSR Influence
    from each of East Germany, Poland and Yugoslavia -- three decisions for five queued points,
    since the log writes the amount and not the decision -- and the two it did not consume were
    still there when the USSR's The Reformer asked where to place. The Reformer put one of its
    four into Poland, which the log has it never touching, and West Germany finished an
    Influence short.

    Keyed by card id, so the driver can ask for the queue belonging to whatever the engine says
    is resolving.
    """
    out: Dict[int, List[int]] = {}
    for name, rows in (e.influence_by_event or {}).items():
        cid = card_id(name)
        if cid is None:
            continue
        points = out.setdefault(cid, [])
        for _side, delta, country, _u, _s in rows:
            points.extend([country] * abs(int(delta)))
    return out


# Events the log never names, because it records their effect and not their firing.
#
#   NORAD prints only the Influence it places -- "US +1 in Poland" trailing the coup that
#   dropped DEFCON -- and never a header of its own.
_NORAD = 106
_UNNAMED_EVENTS = frozenset({_NORAD})


def _events_the_log_names(e: Entry) -> Set[int]:
    """Every card this entry says was played, revealed, returned or fired.

    Deliberately generous. The check it feeds asks whether the engine fired an event the log
    knows nothing about at all, which is a different and much louder question than whether the
    log names it in the right place: a card the entry mentions anywhere is not "from nowhere".
    """
    named: Set[int] = set()
    for nm in list(e.events or []):
        cid = card_id(nm)
        if cid:
            named.add(cid)
    for nm in (e.headlines or {}).values():
        cid = card_id(nm)
        if cid:
            named.add(cid)
    for nm in (e.card, e.played_card, e.returned_card):
        cid = card_id(nm) if nm else None
        if cid:
            named.add(cid)
    for _side, nm in (e.revealed or []):
        cid = card_id(nm)
        if cid:
            named.add(cid)
    for _side, nm in (e.discards or []):
        cid = card_id(nm)
        if cid:
            named.add(cid)
    # A headline entry's card field is "A & B"; card_id gives nothing for that, and the two
    # halves are in e.headlines already.
    return named


def _reattribute_hands(raws: List[Dict], turn: int,
                       turn_hands: Dict[str, List[int]]) -> int:
    """Give each card to the side the log says played it. Returns how many moved.

    The turn's hand lists and the entries disagree, and where they do it is the entries that
    are right: a list is a summary the interface assembled, while an entry is the play itself,
    narrated as it happened. At turn 6 of replay 111 the lists have Asia Scoring in the USSR's
    hand and ABM Treaty in the US's, and the headline reads "US Headlines Asia Scoring / USSR
    Headlines ABM Treaty" -- the two cards are swapped. A scoring card pays out from the board and
    not from whose hand it came, so nothing was mis-scored; what broke is that neither side
    could play the card the log says they played, and the headline went to whatever else was
    held. Turn 6 diverged from there and the game ran to a US win at 20 VP where the log has
    the USSR ahead by 8.

    Only a card the log names a player as *playing* is moved, which is the one attribution an
    entry states outright. A card merely revealed or discarded during someone's action round
    is not necessarily theirs -- Five Year Plan makes its opponent discard, Missile Envy takes
    from the other hand -- so those are left alone.
    """
    moved = 0
    for raw in raws:
        e = parse_entry(raw)
        if e.turn != turn:
            continue
        owners: List[Tuple[str, Optional[int]]] = [
            (side, card_id(nm)) for side, nm in (e.headlines or {}).items()]
        if e.card and " & " not in e.card and e.player in ("US", "USSR"):
            owners.append((e.player, card_id(e.card)))
        for side, cid in owners:
            if not cid or side not in turn_hands:
                continue
            other = "USSR" if side == "US" else "US"
            if cid in turn_hands[side]:
                continue
            if cid in turn_hands[other]:
                turn_hands[other].remove(cid)
                turn_hands[side].append(cid)
                moved += 1
    return moved


def _pad_hand(state: ts.GameState, held: List[int], size: int, ops_cap: Optional[int],
              taken: set, final_turn: bool = True) -> List[int]:
    """Top a short logged hand up with cards the log does not account for.

    The log records only the cards a player used, so a game that ends mid-turn leaves hands
    that are far too small: at turn 7 of replay 119 the US is credited with one card and the
    USSR with two, where both should hold nine. Playing from a hand of one is not the decision
    the human faced, and Missile Envy in particular reads the whole hand. Scoring cards are
    never used as padding -- holding one at the end of a turn loses the game outright.
    """
    # Only where the log is obviously truncated, which means a game that stopped mid-turn. A
    # turn's list is the cards that became *visible* during it, so a hand one or two short is
    # the ordinary case -- a player who carries a card over to the next turn never reveals it,
    # and 47% of the corpus's lists are exactly one card short. Those are left as the log has
    # them, wrong hand size and all, until there is a heuristic worth trusting for what was
    # held: inventing a card is not free, because the rules read the hand in places that
    # inventing changes -- Blockade and Latin American Debt Crisis ask whether a 3 Ops card is
    # held, and a trap is escaped by playing a 2 Ops card.
    # Only the turn the log stops in. A short list anywhere else is not a truncated record
    # but a turn in which little became visible, and there are ordinary reasons for that: at
    # turn 5 of replay 114 the USSR is caught by Bear Trap, discards two cards and skips four
    # action rounds, so five of nine ever show. Padding it invented four cards with 2 Ops or
    # more, and the trap is escaped by discarding one -- so the engine offered those inventions
    # where the log says the USSR had nothing to discard and had to play their scoring card.
    if not final_turn or len(held) >= size - 2:
        return list(held)
    pool = [c for c in range(1, 111)
            if c not in held and c not in taken
            and state.get_card_location(c) == ts.CardLocation.DRAW_DECK
            and not ts.CardData.get_card_info(c)["is_scoring"]
            and (ops_cap is None or int(ts.CardData.get_card_info(c)["ops"]) < ops_cap)]
    return list(held) + pool[:size - len(held)]


def _set_hand(state: ts.GameState, player: ts.Player, names: List[str]) -> int:
    """Force a player's hand to the logged cards. Returns how many were placed."""
    loc = ts.CardLocation.HAND_US if player == ts.Player.US else ts.CardLocation.HAND_USSR
    # clear the current hand first, so leftovers cannot linger
    for c in range(1, 111):
        if state.get_card_location(c) == loc:
            state.set_card_location(c, ts.CardLocation.DISCARD_PILE)
    placed = 0
    for nm in names:
        cid = card_id(nm)
        if cid:
            state.set_card_location(cid, loc)
            placed += 1
    return placed


def _narrated_score(entry: Entry) -> Optional[int]:
    """The score this entry states in words, US positive, or None if it states none.

    Only the narration -- "US gains 5 VP. Score is US 18." -- is worth asserting against. The
    entry's score field is a running value that lags the narration, disagreeing with it in 531
    of the 6602 places the log states a score, and it goes stale between VP events: at turn 1
    AR1 of replay 166 the log has just narrated the score to even, and the field still reads
    the USSR's 1 from the headline before it.
    """
    # The last one stated, not the first: an entry can narrate the score more than once, and it
    # is the value it leaves behind that the engine has to match. A headline states one per
    # card -- at turn 2 of replay 100 the US takes 2 from Captured Nazi Scientist and the USSR
    # then takes 1 from Europe Scoring, leaving 1, and asserting on the 2 called the engine
    # wrong where it was right.
    score = None
    for _side, _amount, total_side, total in (entry.vp_gains or []):
        if total_side == "US":
            score = int(total)
        elif total_side == "USSR":
            score = -int(total)
        else:                                      # "Score is even."
            score = 0
    return score


def _reconcile_scalars(state: ts.GameState, entry: Entry,
                       take_score: bool = True) -> None:
    """Force VP and DEFCON to the logged values.

    Without this the engine scores from its own board and the two diverge fast: on replay 100
    the engine reached 20 VP -- game over -- at T2 AR1 while the log had the score at 1, which
    is why only a quarter of entries were reachable. VP is the one quantity the log disputes
    with itself (field vs ledger), so which source is used here is recorded per sample.

    take_score is False where the entry's own play carried the game past the score it states.
    An entry that ends a turn is followed by the Military Operations comparison, which the log
    never narrates: at turn 4 AR7 of replay 16 the US takes 3 VP from Alliance For Progress and
    the entry states "Score is USSR 5", then turn 4 ends with the USSR on 5 military ops and
    the US on none against DEFCON 2, so the US owes 2 and the real score is USSR 7. The engine
    reaches that exactly -- and reconciling the next entry from this one's narration put it
    back to 5, so Camp David's VP at turn 5 AR1 landed one short of the log.
    """
    # Exactly, not max(): taking the larger of the two hides a track the reconstruction
    # advanced on its own, which is precisely the error worth catching.
    for side, level in (entry.space or []):
        if side == "US":
            state.us_space_track = int(level)
        else:
            state.ussr_space_track = int(level)
    # Military operations belong to the turn the entry is in, and the engine zeroes them when
    # that turn ends. Reconciling them from an entry of an earlier turn carries the old counts
    # across the boundary and cancels the end-of-turn comparison: at turn 6 of replay 60 the US
    # entered with the 5 they had finished turn 5 on, where the log has them at 0 all turn, so
    # neither side showed a deficit against DEFCON 2 and the 2 VP the USSR was owed never
    # moved. Seven entries later that missing 2 ended the game at 20.
    if entry.turn is None or int(entry.turn) == int(state.turn):
        for side, level in (entry.milops or []):
            if side == "US":
                state.us_mil_ops = int(level)
            else:
                state.ussr_mil_ops = int(level)
    # Clear ongoing effects the log says have ended. An effect that outlives its expiry can
    # make later play illegal outright: a stale Cuban Missile Crisis makes every USSR coup an
    # instant loss, which is how turn 6 AR3 of replay 104 ended the game at DEFCON 3 with the
    # score at 20 while the log has the coup simply succeeding.
    for name in (entry.out_of_play or []):
        for bit in _EFFECT_BITS.get(_norm(name), ()):
            state.persistent_effects &= ~bit

    # The score the entry narrates, in preference to the score field it carries. "US gains 5 VP.
    # Score is US 18." states the score outright; the field lags behind it, disagreeing in 531
    # of the 6602 places the log states one. Forcing the stale field left replay 60 two VP ahead
    # of the real game at turn 7 AR2, so Central America Scoring's 5 VP -- which the engine
    # awards exactly as the log does -- landed on 20 and ended a game that ran to turn 10.
    narrated = None
    for _side, _amount, total_side, total in (entry.vp_gains or []):
        if total_side == "US":
            narrated = int(total)
        elif total_side == "USSR":
            narrated = -int(total)
        else:                                  # "Score is even."
            narrated = 0
    if narrated is not None and take_score:
        state.victory_points = narrated
    if entry.defcon is not None and 1 <= int(entry.defcon) <= 5:
        state.defcon = int(entry.defcon)


_RE_AR = re.compile(r"AR(\d+)")


def _reconcile_turn(state: ts.GameState, entry: Entry, replay_id: int = -1) -> None:
    """Force turn, action round and phasing player to the ones the log names.

    The engine advances these itself, and any entry it could not drive faithfully leaves them
    a step out. From then on every entry is attributed to the wrong player: at turn 2 AR6 of
    replay 114 the log has the US placing influence and the engine had the USSR to move, so
    nothing the entry described was legal. The log states whose action round it is outright.
    """
    if entry.turn:
        state.turn = int(entry.turn)
    m = _RE_AR.search(entry.phase or "")
    if m:
        state.action_round = int(m.group(1))
    if entry.player == "US":
        state.phasing_player = ts.Player.US
    elif entry.player == "USSR":
        state.phasing_player = ts.Player.USSR

    # Every entry must begin with the engine waiting for a card. Anything else means the
    # previous entry did not finish, and continuing from a half-resolved decision would convert
    # a position the humans never played -- so this is a failure, not something to tidy up.
    want_phase = ts.Phase.ACTION_ROUND if m else ts.Phase.HEADLINE
    ctx = state.ctx()
    if ts.Engine.is_terminal(state):
        raise ConversionFailure(Mismatch(
            replay_id, entry.turn, entry.phase, entry.player, entry.card,
            "engine ended the game early",
            f"the log continues but the engine is in {str(state.current_phase).split('.')[-1]} "
            f"at {int(state.victory_points)} VP, DEFCON {int(state.defcon)}"))
    if (ctx.decision_type != ts.DecisionType.SELECT_CARD
            or int(ctx.resolving_card) != 0
            or int(state.ctx_stack_depth) != 0
            or state.current_phase != want_phase):
        raise ConversionFailure(Mismatch(
            replay_id, entry.turn, entry.phase, entry.player, entry.card,
            "previous entry left the engine mid-decision",
            f"expected a card request in {str(want_phase).split('.')[-1]}, found "
            f"{str(ctx.decision_type).split('.')[-1]} in "
            f"{str(state.current_phase).split('.')[-1]} "
            f"(resolving={int(ctx.resolving_card)}, op_card={int(ctx.pending_op_card)}, "
            f"depth={int(state.ctx_stack_depth)})"))
    ctx.decision_player = state.phasing_player


def _reconcile_board(state: ts.GameState, countries: Dict) -> int:
    """Force the board to the logged one. Returns the number of cells corrected."""
    fixed = 0
    for key, c in (countries or {}).items():
        cid = country_id(key)
        if cid is None:
            continue
        try:
            lus, lussr = int(c["inflUS"]), int(c["inflUSSR"])
        except (KeyError, TypeError, ValueError):
            continue
        cur = state.get_country(cid)
        if int(cur.us_influence) != lus or int(cur.ussr_influence) != lussr:
            state.set_country(cid, lus, lussr)
            fixed += 1
    return fixed


def _pending_roll(state: ts.GameState) -> Tuple[int, str]:
    """What the chance node about to resolve is for and whose roll it is, as the engine
    itself classifies them.

    The RollType is kept in the context's temp_cards, which reach Python trimmed to
    temp_card_cnt and so do not include it. Rather than infer the kind from the card -- an
    ordinary Ops coup has no resolving card at all, while Junta's and Che's do -- the node is
    resolved on a throwaway clone and the record it writes is read back. One clone per chance
    node is nothing next to the seed searches this replaces.
    """
    probe = state.clone()
    try:
        ts.Engine.step(probe, ts.MicroAction(ts.DecisionType.ROLL_DIE, 0, 0, 0))
    except Exception:
        return int(ts.RollType.NONE), "NONE"
    record = probe.to_dict()["die_roll"]
    return int(record["type_id"]), str(record["roller"])


_SUMMIT_SCORE = re.compile(r"Score is (?:(US|USSR) (\d+)|even)\.")


def _summit_target(entry: Optional[Entry]) -> Tuple[bool, Optional[int]]:
    """What the log says Summit came to: (Summit is in this entry, the score it left).

    Summit is the one event whose outcome the log states only when there is one. The card
    reads "do not reroll ties", so a tie awards nothing, changes no DEFCON, and moves no
    Influence -- and the log, having nothing to report, prints "Event: Summit" and stops. Five
    of the corpus's twelve Summits end that way, and reading the silence as "unknown" left the
    dice to the engine: at turn 4's headline of replay 109 the reconstruction handed the US the
    2 VP that the log gave to nobody, and the game ran 2 VP adrift into the Southeast Asia
    Scoring two action rounds later.

    The score is read from after the "Event: Summit" line rather than from the entry as a
    whole, because a headline resolves two cards and the other one may score as well.
    """
    text = (entry.text or "") if entry is not None else ""
    marker = text.find("Event: Summit")
    if marker < 0:
        return False, None
    m = _SUMMIT_SCORE.search(text, marker)
    if m is None:
        return True, None                          # narrated nothing: a tie
    if m.group(1) is None:
        return True, 0                             # "Score is even."
    return True, int(m.group(2)) * (1 if m.group(1) == "US" else -1)


def _summit_dice(state: ts.GameState, target: int) -> Tuple[int, int]:
    """The pair of dice that leaves the score where the log leaves it.

    Both dice are handed to the chance node outright rather than searched for in the rng, so
    the outcome is reconstructed and not stumbled upon. The pair is not unique -- what decides
    Summit is the two totals, dice plus regions dominated -- and any pair reaching the logged
    score is as faithful as the next, since the dice themselves are never recorded.
    """
    for us_roll in range(1, 7):
        for ussr_roll in range(1, 7):
            probe = state.clone()
            try:
                ts.Engine.step(probe, ts.MicroAction(
                    ts.DecisionType.ROLL_DIE, us_roll, ussr_roll, 0))
            except Exception:
                continue
            if int(probe.victory_points) == target:
                return us_roll, ussr_roll
    raise RuntimeError(
        f"no Summit dice reach the logged score {target} from {int(state.victory_points)}")


def _drain(state: ts.GameState,
           expected: Optional[Dict[int, Tuple[int, int]]] = None,
           forced_roll: int = 0,
           war_rolls: Optional[List[int]] = None,
           coup_rolls: Optional[List[int]] = None,
           realign_rolls: Optional[List[Tuple[str, int]]] = None,
           want_vp: Optional[int] = None,
           summit: Optional[Tuple[bool, Optional[int]]] = None,
           random_discards: Optional[List[int]] = None) -> None:
    """Resolve chance nodes, steering them to what the log recorded.

    Not every die belongs to a decision. A war with a fixed target -- Korean War, Arab-Israeli
    War -- rolls inside the event with nothing to choose, so there is no action to force the
    outcome through, and the roll came out however the engine's stream said: at turn 4 AR4 of
    replay 101 the USSR won the Korean War in the log and lost it in the reconstruction.

    The log states that die outright ("DEFEAT: 2 (-1)  < 4"), so it is handed to the chance
    node rather than searched for, which is both exact and immune to the guard below. Searching
    the rng for a matching board could not settle a war that shares its entry with a coup: the
    coup picks the seed first and sets seed_settled, which suppresses the search, so at turn 6
    AR4 of replay 121 the US couped Tunisia and the Korean War that followed was left to
    chance -- won in the reconstruction, lost in the log.
    """
    while (not ts.Engine.is_terminal(state)
           and state.ctx().decision_player == ts.Player.NONE
           and state.ctx().decision_type == ts.DecisionType.ROLL_DIE):
        roll = forced_roll
        # Each die goes to the kind of roll it was recorded for. A coup and a war can share an
        # entry -- at turn 6 AR4 of replay 121 the US coups Tunisia and then loses the Korean
        # War -- so the roll type decides, not the order they happen to arrive in.
        second = 0
        kind, roller = _pending_roll(state) if (war_rolls or coup_rolls or realign_rolls
                                                or want_vp is not None
                                                or (summit is not None and summit[0])) \
            else (int(ts.RollType.NONE), "NONE")
        if war_rolls and kind == int(ts.RollType.WAR_EVENT):
            roll = war_rolls.pop(0)
        elif coup_rolls and kind == int(ts.RollType.COUP):
            roll = coup_rolls.pop(0)
        elif realign_rolls and kind == int(ts.RollType.REALIGNMENT):
            # A realignment rolls for both sides at once. The engine reads the acting player's
            # die from primary_id and the opponent's from secondary_id, and says which player
            # is acting, so the pair is matched by side rather than by the order printed.
            pair = {side: die for side, die in realign_rolls[:2]}
            del realign_rolls[:2]
            other = "USSR" if roller == "US" else "US"
            roll, second = pair.get(roller, 0), pair.get(other, 0)
        elif summit is not None and summit[0] and kind == int(ts.RollType.SUMMIT):
            # A tie leaves the score alone, which is what the log's silence records.
            roll, second = _summit_dice(
                state, summit[1] if summit[1] is not None else int(state.victory_points))
        elif want_vp is not None and kind in (int(ts.RollType.OLYMPIC_GAMES),
                                              int(ts.RollType.SUMMIT)):
            # Olympic Games and Summit are decided by dice the log never prints -- it records
            # only who won ("USSR chooses to participate in the Olympics / US gains 2 VP"). The
            # board tells us nothing, since neither moves a single Influence, so the seed is
            # chosen by the score it lands on instead. Left to chance the winner was a coin
            # flip, and losing it puts the 2 VP on the wrong side: at turn 6's headline of
            # replay 245 the US wins the Olympics in the log and the reconstruction gave the
            # USSR the points, a four VP swing from one roll.
            _force_roll(state, expected, want_vp)
        elif expected:
            _force_roll(state, expected)
        # A space race roll is given outright by the log ("Die roll: 5 -- Failed!"), and the
        # chance node takes it directly, so there is nothing to search for. Leaving it to the
        # engine's own stream advanced tracks the humans never advanced -- and since the log
        # only prints a track on success, a wrong one was never corrected afterwards.
        micro = ts.MicroAction(
            ts.DecisionType.ROLL_DIE, roll if 1 <= roll <= 6 else 0,
            second if 1 <= second <= 6 else 0, 0)
        # An event can fire as this node resolves -- Five Year Plan's discard follows the last
        # Op spent, and the last Op of a realignment is spent here -- so the card it draws has
        # to be steered on this step too, not only on the ones the log answers.
        if random_discards and _force_random_discard(state, random_discards, micro=micro):
            del random_discards[:]
        ts.Engine.step(state, micro)


def _force_roll(state: ts.GameState, expected: Optional[Dict[int, Tuple[int, int]]],
                want_vp: Optional[int] = None, tries: int = 400) -> bool:
    """Seed the rng so the pending chance node resolves the way the log says it did.

    Matched on the board where the log gives one, and on the score where it gives that instead:
    Olympic Games and Summit move no Influence at all, so the only trace they leave is who
    gained the points.
    """
    base = int(state.rng_state)
    for k in range(tries):
        cand = (base + (k + 1) * _GOLDEN) % _UINT64
        probe = state.clone()
        probe.rng_state = cand
        try:
            ts.Engine.step(probe, ts.MicroAction(ts.DecisionType.ROLL_DIE, 0, 0, 0))
        except Exception:
            continue
        if expected and not all(
                int(probe.get_country(c).us_influence) == us
                and int(probe.get_country(c).ussr_influence) == ussr
                for c, (us, ussr) in expected.items()):
            continue
        if want_vp is not None and int(probe.victory_points) != want_vp:
            continue
        state.rng_state = cand
        return True
    return False


def _acting(state: ts.GameState) -> ts.Player:
    ctx = state.ctx()
    return ctx.decision_player if ctx.decision_player != ts.Player.NONE else state.phasing_player


_PASS = 211


def _drive_passed_rounds(state: ts.GameState, e: Entry, conv: "Conversion") -> None:
    """Pass the action rounds the log says nobody played.

    A player who has run out of cards skips their action round. The log gives it a header and
    nothing else -- "Turn 5, USSR AR4" at the foot of the entry above -- and writes no entry of
    its own, so the rounds simply go missing from the record. At turn 5 of replay 114 the USSR
    skips four in a row and the next four entries are all the US's, which left the driver
    handing the USSR's cards to the US.

    The pass is driven rather than skipped over, because it is what happened and because what
    follows depends on it: at turn 10 AR7 of replay 105 the US passes the last action round of
    the game, which ends turn 10 and runs the final scoring the log records as 11 VP.
    """
    for _turn, side, _ar in e.passed_rounds:
        if ts.Engine.is_terminal(state):
            return
        want = ts.Player.US if side == "US" else ts.Player.USSR
        ctx = state.ctx()
        # The exact round, not merely the same player asked for a card. The entry before a
        # skipped round can carry the game past it -- at turn 8 AR7 of replay 116 the USSR's
        # play ends the turn -- and the next card request is then turn 9's headline, which is
        # not a round anyone can pass.
        if (state.current_phase != ts.Phase.ACTION_ROUND
                or int(state.turn) != _turn
                or int(state.action_round) != _ar
                or ctx.decision_type != ts.DecisionType.SELECT_CARD
                or ctx.decision_player != want):
            continue
        mask = ts.ActionMask.generate_flat_mask(state)
        if not mask[_PASS]:
            # Passing with cards in hand is illegal, so a log that says a round was skipped
            # and an engine that says it could not be is a disagreement worth hearing about
            # rather than papering over.
            raise ConversionFailure(Mismatch(
                conv.replay_id, e.turn, e.phase, e.player, e.card,
                "logged pass is not legal",
                f"the log skips {side} AR{_ar} of turn {_turn}, and the engine still offers "
                f"{sum(1 for i in range(110) if mask[i])} cards to play"))
        ts.Engine.step_flat(state, _PASS)
        _drain(state)


def _coup_extras(rows: List[Tuple[str, int, int, int, int]],
                 targets: List[int]) -> List[int]:
    """Influence in a coup's own lines that the coup itself cannot explain.

    A coup takes the opponent's Influence away and then puts the couper's down; a realignment
    only takes away. So within one coup the side that *loses* Influence is the victim, and the
    side that gains it in the same country is the couper -- unless it is the victim gaining,
    which no coup does. The only thing that does that is NORAD, which lets the US add 1
    Influence after an action round in which it lost some.

    At turn 6 AR1 of replay 131 the USSR coups Argentina and the US puts its NORAD Influence
    straight back into Argentina; reading that as part of the coup lost the placement. At turn
    6 AR3 of replay 104 the US put it into Poland instead, where being in another country made
    it obvious.
    """
    victim = {cid: side for side, delta, cid, _u, _s in rows
              if int(delta) < 0 and cid in targets}
    out: List[int] = []
    for side, delta, cid, _u, _s in rows:
        if cid not in targets:
            out.extend([cid] * abs(int(delta)))
        elif int(delta) > 0 and victim.get(cid, side) == side:
            # Either the victim gaining in the country just couped, or a gain in a coup that
            # removed nothing at all. A successful coup always removes before it places, so a
            # target with no removal against it is a coup that failed, and nothing it prints
            # afterwards is its: at turn 6 AR3 of replay 150 the USSR's coup on Angola fails,
            # DEFCON drops to 2, and the "US +1 in Angola" that follows is NORAD's.
            out.extend([cid] * abs(int(delta)))
    return out


def point_queue(e: Entry) -> List[int]:
    """Countries the player actually pointed at -- decisions, not consequences.

    Under a coup or realignment the only choice is the target; the influence lines that
    follow are the *result* of the roll. Queuing those as placements made the engine try to
    place influence where the coup had removed it.
    """
    if e.mode == "space":
        return []
    if e.mode in ("coup", "realign"):
        # Influence in a country that is not a target belongs to something else the action
        # round set off, and that is a decision of its own: NORAD lets the US add 1 Influence
        # after an action round in which it lost some, which is the "US +1 in Poland" trailing
        # the USSR's Panama coup at turn 6 AR3 of replay 104.
        # Each coup section is read against its own target, because an entry can hold two
        # coups belonging to two different players: at turn 7 AR6 of replay 184 the US plays
        # "Lone Gunman", the USSR coups Nigeria with the Ops the event grants them, and the US
        # then coups Nigeria back with the card's own. Judging the rows by the entry's player
        # made the USSR's own gain look like someone else's placement.
        runs = ([(s.targets, s.influence) for s in e.sections if s.mode in ("coup", "realign")]
                or [(e.targets, list(e.influence or []))])
        extra: List[int] = []
        for run_targets, rows in runs:
            extra.extend(_coup_extras(rows, run_targets))
        # Rows the sections never claimed -- a placement printed outside any Ops header.
        if e.sections:
            claimed = [rec for _t, rows in runs for rec in rows]
            for rec in (e.influence or []):
                if rec in claimed:
                    claimed.remove(rec)
                else:
                    extra.extend([rec[2]] * abs(int(rec[1])))
        return list(e.targets) + extra

    if e.setup:
        return _setup_point_queue(e)

    q: List[int] = []
    for _side, delta, cid, _u, _s in (e.ops_influence or []):
        q.extend([cid] * abs(int(delta)))
    if not q and e.targets:
        q = list(e.targets)
    return q


def _setup_point_queue(e: Entry) -> List[int]:
    """The opening placement, spread a country at a time rather than a country at a stretch.

    The log states the setup as a total per country -- "US +4 in West Germany", "+3 in France",
    "+2 in Italy" -- and says nothing about the order, because in the game there is none: the
    placement is simultaneous. The order still matters to the reconstruction, because the
    handicap is a second placement and it may only go where that side already has Influence.

    Placing each country's total in one run spends the base allotment before the last country
    is reached. At turn 1 of replay 152 the US has 7 to spread and 2 more from the handicap;
    four into West Germany and three into France used all seven, and Italy -- still empty --
    was not a legal target for the handicap that followed. The reconstruction put those two
    into West Germany and France instead and opened the game two Influence out in three
    countries, which the Socialist Governments headline then removed Influence from.

    Dealing one at a time round the named countries gives every one of them Influence inside
    the base allotment, so the handicap has somewhere to go. Both spreads reach the same board.
    """
    out: List[int] = []
    runs: List[Tuple[str, List[List[int]]]] = []
    for side, delta, cid, _u, _s in (e.ops_influence or []):
        if not runs or runs[-1][0] != side:
            runs.append((side, []))
        counts = runs[-1][1]
        for pair in counts:
            if pair[0] == cid:
                pair[1] += abs(int(delta))
                break
        else:
            counts.append([cid, abs(int(delta))])
    for _side, counts in runs:
        while any(n > 0 for _cid, n in counts):
            for pair in counts:
                if pair[1] > 0:
                    out.append(pair[0])
                    pair[1] -= 1
    return out or list(e.targets)


def section_queue(section) -> List[int]:
    """Countries the player pointed at within one Ops section."""
    if section.mode == "space":
        return []
    if section.mode in ("coup", "realign"):
        return list(section.targets)
    q: List[int] = []
    for _side, delta, cid, _u, _s in section.influence:
        q.extend([cid] * abs(int(delta)))
    return q or list(section.targets)


def section_outcomes(section) -> List[Tuple[int, int, int]]:
    """Per-roll expected board values inside one Ops section, in log order.

    A realignment rolls once per target and prints the running result of each, so every line is
    its own expectation. A coup rolls once and prints a line per side whose influence moved, so
    the country's settled value is the last line naming it. Scoping this to the section matters
    when one entry couples twice at the same country: at turn 4's headline of replay 119 the
    USSR coups Venezuela to [0][2] and the US then coups it to [1][0], and taking the entry's
    final value as the USSR's expectation matched no roll at all -- a USSR coup cannot hand the
    US influence -- so the roll went unforced and the US was left with nothing to coup.
    """
    if section.mode == "realign":
        return [(cid, us, ussr) for _s, _d, cid, us, ussr in section.influence]
    last: Dict[int, Tuple[int, int]] = {}
    for _s, _d, cid, us, ussr in section.influence:
        last[cid] = (us, ussr)
    return [(cid, us, ussr) for cid, (us, ussr) in last.items()]


def event_queue(e: Entry) -> List[int]:
    """Targets the card's event asks the player to choose, in log order.

    Kept apart from the Ops queue on purpose. Neither half can be dropped -- at turn 1 AR5 of
    replay 100 the event's Vietnam influence is automatic and never comes back as a decision,
    while at turn 1 AR3 the USSR plays Marshall Plan for Ops and the US still chooses all seven
    event placements -- but merging them let turn 3 AR4 spend Nasser's Op on an event target.

    What a coup or realignment section removed is left out even when the section sits inside an
    event, because that influence is the *result* of a die rather than a target anyone chose,
    and the section drives the target itself through the Ops queue. Che's free coups print
    their own "Coup (2 Ops):" headers inside the event, so ops_influence -- which holds only
    what falls outside an event -- did not claim them: at turn 6 AR7 of replay 113 both coups
    arrived here instead and were expanded by their influence delta, as if removing two US
    Influence from Haiti were two placements. The event queue answers first while a card is
    resolving, so it supplied the targets, nothing recognised them as coups, and the die was
    never steered -- the engine rolled 3 and 1 against the log's 2 and 2.

    Only rows naming that section's own target count. The parser hangs every influence line on
    the section header above it, so an event's placements are attached to whatever Ops section
    preceded them: at turn 1 AR3 of replay 100 the USSR plays Marshall Plan for Ops and the
    seven US event placements land under its "Place Influence" header, and dropping those left
    the US with nothing to place.
    """
    ops = list(e.ops_influence or [])
    sectioned = [rec for s in (e.sections or []) if s.mode in ("coup", "realign")
                 for rec in s.influence if rec[2] in s.targets]
    q: List[int] = list(e.war_targets or [])
    for rec in (e.influence or []):
        if rec in ops:
            ops.remove(rec)
            continue
        if rec in sectioned:
            sectioned.remove(rec)
            continue
        q.extend([rec[2]] * abs(int(rec[1])))
    return q


# -- driving the engine --------------------------------------------------------------------

from ai.eval.positions import PLAY_MODE_ACTION  # noqa: E402

_OP_MODE = {"influence": ts.OpMode.INFLUENCE, "coup": ts.OpMode.COUP,
            "realign": ts.OpMode.REALIGN}


_GOLDEN = 0x9E3779B97F4A7C15
_UINT64 = 1 << 64


def expected_counts(e, raw: Optional[Dict] = None) -> Dict[int, Tuple[int, int]]:
    """Final [US][USSR] per country named in this entry, from the log's own arithmetic.

    Coup and war targets are added from the entry's board snapshot even when no influence line
    mentions them. A failed coup prints no influence at all, which left force_outcome with
    nothing to match: it accepted the first roll, and at turn 5 AR6 the US coup of SE African
    States succeeded in the engine where the log has it fail.
    """
    out: Dict[int, Tuple[int, int]] = {}
    for _side, _delta, cid, res_us, res_ussr in e.influence:
        out[cid] = (res_us, res_ussr)
    countries = (raw or {}).get("countries") or {}
    for cid in list(e.targets) + list(e.war_targets):
        if cid in out:
            continue
        for key, c in countries.items():
            if country_id(key) != cid:
                continue
            try:
                out[cid] = (int(c["inflUS"]), int(c["inflUSSR"]))
            except (KeyError, TypeError, ValueError):
                pass
            break
    return out


def force_outcome(state: ts.GameState, action: int,
                  expected: Dict[int, Tuple[int, int]], tries: int = 400) -> bool:
    """Search rng_state so that stepping `action` reproduces the logged result.

    Coups, wars and realignments resolve on a die the engine rolls itself, so without this the
    resulting influence differs from the log and the entry fails verification even when the
    decision was parsed perfectly -- the check could no longer tell a parse error from an
    unlucky roll. The engine exposes no die value, so we match the *outcome* instead, which is
    the stronger condition anyway.
    """
    if not expected:
        return True
    base = int(state.rng_state)
    for k in range(tries):
        cand = (base + (k + 1) * _GOLDEN) % _UINT64
        probe = state.clone()
        probe.rng_state = cand
        try:
            ts.Engine.step_flat(probe, int(action))
        except Exception:
            continue
        _drain(probe)
        if all(int(probe.get_country(c).us_influence) == us
               and int(probe.get_country(c).ussr_influence) == ussr
               for c, (us, ussr) in expected.items()):
            # the PRE-step candidate, not probe.rng_state -- stepping has advanced that
            state.rng_state = cand
            return True
    return False


_FIVE_YEAR_PLAN = 5
_UN_INTERVENTION = 32
_GRAIN_SALES = 67
_OUR_MAN_IN_TEHRAN = 108
_MISSILE_ENVY = 49
_STAR_WARS = 85
_CHERNOBYL = 94
_TEAR_DOWN_THIS_WALL = 96
# The two cards whose event grants Ops that may only be spent on a coup or a realignment, and
# whose free action is optional. The engine offers INFLUENCE as the decline.
_FREE_ACTION_CARDS = frozenset({47, _TEAR_DOWN_THIS_WALL})


def _seed_missile_envy_hand(state: ts.GameState, revealed: int, giver: ts.Player) -> None:
    """Make the card the log says was handed over the highest Ops one the giver holds.

    Missile Envy takes the opponent's highest Ops card, so the engine's choice is forced by the
    hand -- and a hand holding one card too many chooses differently. The turn's hand list is
    everything a player held during the turn, including what they picked up part way through:
    at turn 5 of replay 64 the US list contains Red Scare/Purge, 4 Ops, which they did not have
    at AR3 at all -- they retrieved it from the discard pile with SALT Negotiations at AR7. Our
    reconstruction handed it over, where the human handed over Suez Crisis at 3.

    Anything strictly higher than the card the log names was demonstrably not in that hand yet,
    so it is set aside. _apply_hands rebuilds the hand from the tracked list at the next entry,
    so this reaches no further than the event it fixes.
    """
    loc = ts.CardLocation.HAND_US if giver == ts.Player.US else ts.CardLocation.HAND_USSR
    want = int(ts.CardData.get_card_info(revealed)["ops"])
    for c in range(1, 111):
        if c == revealed or state.get_card_location(c) != loc:
            continue
        info = ts.CardData.get_card_info(c)
        if not info["is_scoring"] and int(info["ops"]) > want:
            state.set_card_location(c, ts.CardLocation.DISCARD_PILE)
    if state.get_card_location(revealed) != loc:
        state.set_card_location(revealed, loc)


def _seed_revealed_card(state: ts.GameState, cid: int) -> None:
    """Make the card the engine drew at random be the one the log says was revealed.

    Grain Sales takes a random card from the USSR hand and offers it to the US; the log names
    it ("USSR reveals NATO*"), so the draw is not really a chance node for our purposes.
    """
    if state.get_card_location(cid) != ts.CardLocation.HAND_USSR:
        state.set_card_location(cid, ts.CardLocation.HAND_USSR)
    state.ctx().temp_cards = [cid]


def _seed_peeked_set(state: ts.GameState, discards: List[int], size: int = 5) -> None:
    """Make the engine's peeked set the cards the log implies, padded with plausible keeps.

    Our Man in Tehran lets the US look at the top five cards and discard any of them, but the
    log records only the discards -- never the full five. The discards are known, so the only
    invention is the remainder. We fill with US-associated events, which the US would plainly
    keep, so the reconstruction stays consistent with it having discarded everything else.
    Five Year Plan is excluded: it is the one US card whose event hurts the US, so keeping it
    would not be the obvious choice this padding relies on.

    This does lose a little training signal -- the model never sees which *neutral* events the
    US chose to keep -- but the log cannot tell us that, and the discards themselves are real.
    """
    for c in range(1, 111):
        if state.get_card_location(c) == ts.CardLocation.PEEKED_TEMP:
            state.set_card_location(c, ts.CardLocation.DRAW_DECK)

    # The discards are deck cards wherever the reconstruction currently has them. The log's
    # per-turn hand island lists them as held -- turn 5 credits the US with twelve cards, four
    # more than a mid-war hand -- because the US saw them, so _apply_hands seats them in the
    # hand and they would otherwise be missing from the peek entirely.
    peek: List[int] = []
    for c in discards:
        if c not in peek:
            state.set_card_location(c, ts.CardLocation.DRAW_DECK)
            peek.append(c)
    fillers = [
        c for c in range(1, 111)
        if c not in peek and c != _FIVE_YEAR_PLAN
        and state.get_card_location(c) == ts.CardLocation.DRAW_DECK
        and str(ts.CardData.get_card_info(c)["side"]) == "US"
        and not ts.CardData.get_card_info(c)["is_scoring"]
    ]
    need = max(0, min(size, len(peek) + len(fillers)) - len(peek))
    if need > len(fillers):
        # Loud on purpose: silently peeking a short set would quietly change which cards the
        # US could have discarded, and the mismatch would surface far from its cause.
        raise ValueError(
            f"cannot reconstruct a {size}-card peek: {len(peek)} logged discards are in the "
            f"draw deck but only {len(fillers)} US event cards remain there to pad with")
    peek.extend(fillers[:need])

    for c in peek:
        state.set_card_location(c, ts.CardLocation.PEEKED_TEMP)
    state.ctx().temp_cards = peek
    state.ctx().remaining_steps = len(peek)


def _choose_branch(state: ts.GameState, legal, wanted: List[int],
                   raw: Optional[Dict], e: Optional[Entry] = None,
                   returned_cid: Optional[int] = None) -> Optional[int]:
    """Pick the branch of a two-sided event that leads where the log went.

    Events like Warsaw Pact Formed offer a genuine choice -- remove US influence from Eastern
    Europe, or add USSR influence to it -- and the log records only the consequence. Rather
    than encode each card's branches, try each one on a clone: the right branch is the one that
    then offers the targets the log names, or failing that the one whose board ends up closest
    to the log's own snapshot.
    """
    board = _logged_board(raw)
    best, best_score = None, -1
    for a in legal:
        probe = state.clone()
        try:
            ts.Engine.step_flat(probe, int(a))
        except Exception:
            continue
        _drain(probe)
        score = 0
        # The log says the card was handed back unplayed, so reject any branch that goes on to
        # play it. At turn 4's headline of replay 119 the US returned Brezhnev Doctrine and
        # took Grain Sales' own 2 Ops; the branch that plays the card instead left the engine
        # mid-headline, and every later entry inherited that.
        if returned_cid is not None:
            plays_it = (int(probe.ctx().pending_op_card) == returned_cid
                        or int(probe.ctx().resolving_card) == returned_cid)
            score += 0 if plays_it else 2000
        # Some branches are a numeric setting rather than a target: How I Learned To Stop
        # Worrying picks the new DEFCON, and with no targets to tell the options apart the
        # first legal one -- DEFCON 1 -- ended the game at turn 4's headline of replay 101.
        # Where the log states the score this entry ends on, the branch that reaches it is the
        # branch that was taken. Wargames offers 6 VP to the opponent and an immediate end, or
        # nothing at all, and the two differ only in the score: at turn 8 AR1 of replay 113 the
        # USSR takes the ending and the log reads "US gains 6 VP. Score is USSR 7."
        # ...but only where the log names no targets to judge by. The score compared here is
        # the one the *entry* ends on, which a branch reaches only if it finishes the event: a
        # branch that opens further decisions leaves the probe short of it, and a branch that
        # does nothing at all sails past to the turn end and matches. At turn 1 AR6 of replay
        # 302 the US holds no Eastern European Influence, so Warsaw Pact Formed's removal
        # branch is a no-op that reached the score while the placement branch -- the one the
        # log spells out, five Influence across East Germany, Poland and Bulgaria -- did not.
        # Where the log names targets they are the better evidence, and they decide.
        want_vp = _narrated_score(e) if e is not None else None
        reaches_logged_score = (not wanted and want_vp is not None
                                and int(probe.victory_points) == want_vp)
        if reaches_logged_score:
            score += 4000
        if e is not None and e.defcon is not None:
            if int(probe.defcon) == int(e.defcon):
                score += 500
            # ...but only where the log kept playing. DEFCON 1 is thermonuclear war and the
            # phasing player loses, so it is never the branch to take on a tie -- and yet at
            # turn 9 AR7 of replay 104 it is exactly what happened: the USSR played Star Wars,
            # the US took How I Learned To Stop Worrying out of the discard pile and set DEFCON
            # to 1, and the USSR, as the phasing player, lost. The entry records defcon 1, so
            # penalising every branch that ends the game put the real one out of reach.
            if (ts.Engine.is_terminal(probe) and int(e.defcon) != 1
                    and not reaches_logged_score):
                score -= 5000
        if wanted and not ts.Engine.is_terminal(probe):
            offered = {int(ts.ActionMask.decode_flat_action(probe, int(x)).primary_id)
                       for x in np.flatnonzero(np.asarray(
                           ts.ActionMask.generate_flat_mask(probe)))}
            # How many of the logged targets this branch can reach, not merely whether it
            # reaches one. Warsaw Pact Formed's two branches both offer Romania -- the USSR
            # added influence there and the US had influence to remove -- so a yes/no test
            # tied, and the wrong branch stripped the US instead of reinforcing the USSR.
            score += 1000 * sum(1 for c in set(wanted) if c in offered)
        # Tiebreak on how much of the logged board this branch already reproduces, which
        # settles branches that resolve fully on their own and ask nothing further.
        score += sum(1 for cid, (us, ussr) in board.items()
                     if int(probe.get_country(cid).us_influence) == us
                     and int(probe.get_country(cid).ussr_influence) == ussr)
        if score > best_score:
            best, best_score = int(a), score
    return best


def _logged_board(raw: Optional[Dict]) -> Dict[int, Tuple[int, int]]:
    """The entry's own board snapshot, as country id -> (US, USSR)."""
    out: Dict[int, Tuple[int, int]] = {}
    for key, c in ((raw or {}).get("countries") or {}).items():
        cid = country_id(key)
        if cid is None:
            continue
        try:
            out[cid] = (int(c["inflUS"]), int(c["inflUSSR"]))
        except (KeyError, TypeError, ValueError):
            continue
    return out


def _force_random_discard(state: ts.GameState, cards: List[int],
                          action: Optional[int] = None,
                          micro: Optional["ts.MicroAction"] = None,
                          tries: int = 400) -> bool:
    """Seed the rng so that the coming step discards the card the log says was discarded.

    Five Year Plan discards at random from the USSR hand and, if the card is a US event, plays
    it. Which card comes out therefore decides what happens next, and the log records it -- but
    it is drawn inside the event, so there is no action to steer. At turn 3's headline of
    replay 119 the engine drew Marshall Plan where the log drew Duck and Cover, and seven US
    influence went into Western Europe that the human game never placed.

    The step is not always a decision. Played for Ops first and the event second, the event
    fires the moment the Ops run out -- and the last of three realignments runs out inside its
    own chance node, so the step that discards is one `_drain` takes, not one the log answers.
    At turn 3 AR5 of replay 107 the USSR realigns Angola three times with Five Year Plan and
    the discard was never steered at all: the log drew Nasser, a USSR card that does nothing,
    and the reconstruction drew a US one and played its event for a VP that stayed on the board
    until the Special Relationship two action rounds later asserted against it.

    Only a card the USSR is holding *now* can be the one this step discards. Accepting a card
    that is already gone made every step a match, so the first step offered -- the one that
    plays the card -- claimed the forcing and cleared the queue before the discard happened.
    """
    want = [c for c in cards if state.get_card_location(c) == ts.CardLocation.HAND_USSR]
    if not want:
        return False

    def take(probe: ts.GameState) -> None:
        if micro is not None:
            ts.Engine.step(probe, micro)
        else:
            ts.Engine.step_flat(probe, int(action or 0))

    base = int(state.rng_state)
    for k in range(tries + 1):
        probe = state.clone()
        if k:                       # k == 0 tries the seed the engine already has
            probe.rng_state = (base + k * _GOLDEN) % _UINT64
        try:
            take(probe)
        except Exception:
            continue
        if all(probe.get_card_location(c) != ts.CardLocation.HAND_USSR for c in want):
            if k:
                state.rng_state = (base + k * _GOLDEN) % _UINT64
            return True
    return False


def _find_confirm_done(state: ts.GameState, legal) -> Optional[int]:
    """The 'decline / stop here' action, when the engine offers one."""
    for a in legal:
        if ts.ActionMask.decode_flat_action(state, int(a)).is_confirm_done():
            return int(a)
    return None


def _find(state, legal, want_type, match) -> Optional[int]:
    for a in legal:
        ma = ts.ActionMask.decode_flat_action(state, int(a))
        if int(ma.decision_type) == int(want_type) and match(ma):
            return int(a)
    return None


def _drive_entry(state: ts.GameState, e: Entry, conv: Conversion,
                 raw: Dict, max_steps: int = 300) -> bool:
    """Play one logged entry through the engine, emitting the decisions the log determines."""
    cid_target = card_id(e.card) if e.card and " & " not in e.card else None
    headline_ids = {side: card_id(nm) for side, nm in (e.headlines or {}).items()}
    headline_ids = {k: v for k, v in headline_ids.items() if v}
    pq = point_queue(e)
    evq = event_point_queues(e)
    # Points the log records that belong to no Ops header and no named event: NORAD's, in
    # practice. Asked last, after every queue that can say what a decision is for.
    loose: List[int] = []
    _war_targets = set(e.war_targets or [])
    eq = event_queue(e)
    picked_card = cid_target is None
    # A card whose event makes the player name and use a second card (UN Intervention) asks
    # for two SELECT_CARDs in one entry. Breaking at the second one left the Ops unspent.
    # Only UN Intervention names its second card through a SELECT_CARD. Other cards that put
    # an opponent's card into play stage it themselves (Grain Sales), so treating every
    # "X plays Y" line as a card to select would have the driver play Y a second time.
    second_cid = (card_id(e.played_card)
                  if e.played_card and cid_target == _UN_INTERVENTION else None)
    # Keyed by card, not a single flag: an entry can reach more than one play mode when
    # its event hands the player a second card. At turn 7 AR1 of replay 113 the US plays
    # Grain Sales To Soviets as its Event, is handed the USSR's Nuclear Subs, and plays
    # that for Ops to coup Angola -- two cards, two modes.
    mode_picked_for: set = set()
    discard_queue = [c for c in (card_id(nm) for _side, nm in (e.discards or [])) if c]
    reveal_queue = [c for c in (card_id(nm) for _side, nm in (e.revealed or [])) if c]
    envy_took = _missile_envy_took(e)
    # Cards this entry fires the event of besides its own. Star Wars lets the US take any
    # non-scoring card out of the discard pile and play it as its event, and the log names that
    # card on an "Event:" line of its own: at turn 9 AR7 of replay 104 the USSR plays Star Wars
    # and the US answers with How I Learned To Stop Worrying, which sets DEFCON to 1 and ends
    # the game. Without this the engine's card request had no answer and the entry was reported
    # as undetermined though the log states it outright.
    event_card_queue = [c for c in (card_id(nm) for nm in (e.events or []))
                        if c and c != cid_target and c not in headline_ids.values()]
    # Five Year Plan draws its discard at random from the USSR hand, and the drawn card's event
    # then plays out, so which card comes out changes the whole entry. The log names it.
    #
    # A card need not be the one played to fire: an event can reach another card and set it off.
    # At turn 8 AR3 of replay 158 the US plays Star Wars, which takes Grain Sales To Soviets out
    # of the discard pile and fires it -- and because the entry's own card is Star Wars, Grain
    # Sales' draw from the USSR hand was never steered to the Cuban Missile Crisis the log says
    # it revealed. It drew Blockade instead, whose event strips every US Influence from West
    # Germany. So the test is whether the entry fires the card at all, not whether it played it.
    fired_here = {c for c in (card_id(nm) for nm in (e.events or [])) if c}
    fired_here.update(cid for cid in headline_ids.values() if cid)
    if cid_target:
        fired_here.add(cid_target)
    plays_five_year_plan = _FIVE_YEAR_PLAN in fired_here
    plays_grain_sales = _GRAIN_SALES in fired_here
    returned_cid = card_id(e.returned_card) if e.returned_card else None
    random_discards = list(discard_queue) if plays_five_year_plan else []
    seeded_peek = False
    seeded_reveal = False
    # Missile Envy's exchange is decided by the hand, not by a decision, so the hand has to be
    # right before the event fires rather than steered once it asks.
    if (cid_target == _MISSILE_ENVY or _MISSILE_ENVY in headline_ids.values()) and reveal_queue:
        giver = ts.Player.US if (e.revealed or [("US", "")])[0][0] == "US" else ts.Player.USSR
        _seed_missile_envy_hand(state, reveal_queue[0], giver)
    # The die each war in this entry was decided on, in log order.
    war_roll_queue = [int(r) for r, _mod, _won in (e.war_rolls or [])]
    coup_roll_queue = [int(r) for r, _ok in (e.coup_rolls or [])]
    realign_roll_queue = [(side, int(r)) for side, r, _m, _t in (e.realign_rolls or [])]
    # A war with a fixed target resolves in a chance node, so its outcome has to be steered
    # there rather than at a decision. Only the war's own countries are constrained: the rest
    # of the entry has not happened yet at that point.
    # A war's die is settled by the event's own lines, not the entry's final board. A lost war
    # prints no influence at all and leaves the country exactly as it was, so the expectation is
    # its value now -- at turn 1 AR2 of replay 109 the USSR loses the Korean War and the US then
    # places two influence in South Korea, and demanding the post-placement board matched no
    # roll, leaving the war to chance.
    # Every card the engine resolves the event of, checked at the end against what the log
    # says happened. See _events_the_log_names.
    resolved_here: Set[int] = conv.events_resolved
    resolved_here.clear()
    _ops_rows = list(e.ops_influence or [])
    _event_last: Dict[int, Tuple[int, int]] = {}
    for _rec in (e.influence or []):
        if _rec in _ops_rows:
            _ops_rows.remove(_rec)
        else:
            _event_last[_rec[2]] = (_rec[3], _rec[4])
    war_outcome: Dict[int, Tuple[int, int]] = {}
    for _c in e.war_targets:
        if _c in _event_last:
            war_outcome[_c] = _event_last[_c]
        else:
            _cur = state.get_country(_c)
            war_outcome[_c] = (int(_cur.us_influence), int(_cur.ussr_influence))
    # Only where the entry really holds several. With one section the existing queue already
    # describes it, and re-deriving it per decision only risks disagreeing with itself.
    sections = list(e.sections) if len(e.sections or []) > 1 else []
    if sections:
        # ...and then they are the only authority, because a flat queue cannot say which
        # operation a target belongs to -- or whose it is. At turn 9 AR2 of replay 105 the US
        # plays Ortega Elected in Nicaragua for Ops: the USSR takes the event's free coup
        # against Costa Rica, then the US coups Saharan States with the card's own Ops. Seeded
        # flat, the queue answered the USSR's coup out of its own head, and the first
        # SELECT_OP_MODE then popped the section that coup had already used, leaving the US's
        # coup pointed at Costa Rica -- a country with no USSR Influence, which the US cannot
        # coup at all.
        pq[:] = []
        # ...but a row no section claims is not any operation's, and has to go somewhere. NORAD
        # places 1 US Influence after an action round in which the US lost some, and prints it
        # loose at the foot of the entry. At turn 7 AR6 of replay 184 the USSR coups Nigeria
        # with the Ops "Lone Gunman" grants them and the US coups it back with the card's own,
        # so the entry has two sections; clearing the queue outright threw away the "US +1 in
        # Panama" that followed, and NORAD was left asking which of 32 countries was meant. The
        # queue is asked last of all, so it can never pre-empt a section or an event: offered
        # earlier it answered Che's free coup at turn 5 AR3 of replay 131 with a country Che
        # may not touch, and Ortega's at turn 8 AR2 of replay 123.
        _claimed = [t for s_ in e.sections if s_.mode in ("coup", "realign") for t in s_.targets]
        _loose = point_queue(e)
        for _c in _claimed:
            if _c in _loose:
                _loose.remove(_c)
        loose[:] = _loose
    seed_settled = False
    cur_mode = e.mode
    # The log states the space race die outright, so it is handed to the chance node rather
    # than searched for. Any other roll in a space entry belongs to something else.
    space_roll = int(e.die_rolls[0][0]) if (e.mode == "space" and e.die_rolls) else 0
    # A Quagmire or Bear Trap discard is an entry with no play at all: the log records the card
    # discarded and the die rolled to escape ("Trap Roll: 4 <= 4 -- Trap Escaped"). The engine
    # asks for the card and then rolls, so the logged die is handed to that chance node the
    # same way a space race roll is -- at turn 6 AR1 of replay 111 the USSR escapes Bear Trap.
    if not space_roll and e.trap_rolls:
        space_roll = int(e.trap_rolls[0][0])
    # Realignment only. It rolls once per target and prints the running result of each, so the
    # results have to be consumed in order. A coup rolls once but prints a line per side whose
    # influence moved, where only the last line is the country's settled value.
    # ...and only from the section that did the realigning. At turn 4 AR7 of replay 101 South
    # African Unrest first places USSR influence and then realigns the same countries, so
    # taking the event's line as the roll's expected result made it look already satisfied.
    if e.mode == "realign":
        _rows = list(e.ops_influence or [])
    elif e.event_mode == "realign":
        _ops = list(e.ops_influence or [])
        _rows = []
        for rec in (e.influence or []):
            if rec in _ops:
                _ops.remove(rec)
            else:
                _rows.append(rec)
    else:
        _rows = []
    step_outcomes = [(cid, us, ussr) for _s, _d, cid, us, ussr in _rows]

    for _ in range(max_steps):
        # A die the previous decision already chose a seed for must be left alone: force_outcome
        # picks a seed by draining the coup's own chance node, and re-seeding here threw that
        # away. At turn 4's headline of replay 100 that turned the failed Libya coup into a
        # success, because the war constraint it was re-seeded against was already satisfied.
        _drain(state, None if seed_settled else war_outcome, space_roll,
               war_roll_queue, coup_roll_queue, realign_roll_queue, _narrated_score(e),
               _summit_target(e), random_discards)
        seed_settled = False
        if ts.Engine.is_terminal(state):
            break
        ctx = state.ctx()
        if 1 <= int(ctx.resolving_card) <= 110:
            resolved_here.add(int(ctx.resolving_card))
        dt = ctx.decision_type
        # The shared event queue holds every placement the entry hangs on an "Event:" header.
        # While a card is resolving that the entry does hang placements on, those rows may be
        # its; while a card is resolving that it does not, none of them are, and the queue must
        # not answer for it or stand in the way of the section that should.
        #
        # At turn 5's headline of replay 173 the USSR headlines Che and the US The Voice of
        # America. Che's two free coups are sections and Che places nothing, so the queue held
        # only the Voice of America's four removals -- and it both answered Che's first coup
        # with Uruguay, the head of *those*, and being non-empty stopped the section that holds
        # the real targets from being loaded. The log coups Saharan States first, so the dice
        # went to the wrong countries too: Uruguay took the 1 that Saharan States should have
        # had, and 1 + 3 - 2x2 is 0, a failure where the log records a success.
        # Only where the card has somewhere else to look. Blanking the queue for any card the
        # entry does not name leaves events starved whose placements the log records somewhere
        # the parser hangs elsewhere -- under an Ops header, or before the "Event:" line.
        #
        # Sections already consumed still count as somewhere else: at turn 7's headline of
        # replay 239 the USSR headlines Che and coups Nicaragua and then Guatemala, and by the
        # second coup both sections have been taken and the target is sitting in the Ops queue.
        # Testing for sections *remaining* let the US's Puppet Governments answer it with El
        # Salvador, the head of its own three placements -- so the USSR's 4 Influence went
        # there instead of Guatemala, and Puppet Governments was left one placement short.
        # A war's target is in that queue too -- event_queue starts with them -- and it is the
        # war's own, whatever else the entry hangs on an event. At turn 8's headline of replay
        # 112 the US headlines Indo-Pakistani War against the USSR's Junta, and blanking the
        # queue outright left the war with no country to be fought in.
        # An event the log never names by header is never a key of evq, so its absence there
        # says nothing about it: NORAD prints the Influence it places and no header at all, and
        # its placement is in the shared queue like any other. At turn 9 AR1 of replay 133
        # Independent Reds places in Czechoslovakia and NORAD in Venezuela, and blanking the
        # queue for NORAD sent it to the Czechoslovakia the Ops queue still had a copy of.
        if (int(ctx.resolving_card) and evq
                and int(ctx.resolving_card) not in evq
                and int(ctx.resolving_card) not in _UNNAMED_EVENTS
                and (pq or loose or sections)):
            eq_here: List[int] = [c for c in eq if c in _war_targets]
        else:
            eq_here = eq
        # Before the mask is read, since the mask for a peek is built from that very set.
        if (int(ctx.resolving_card) == _OUR_MAN_IN_TEHRAN and not seeded_peek
                and dt == ts.DecisionType.SELECT_CARD):
            _seed_peeked_set(state, discard_queue)
            seeded_peek = True
            ctx = state.ctx()
        elif (plays_grain_sales and not seeded_reveal and reveal_queue
                and dt == ts.DecisionType.CHOOSE_BRANCH):
            # Grain Sales takes a card from the USSR hand at random and offers it to the US, so
            # the branch decision is about a card the engine chose for itself. Correct it to the
            # one the log reveals before that decision is read. Keying this off resolving_card
            # missed the headline case, where it is never set: at turn 4's headline of replay
            # 119 the engine drew Indo-Pakistani War instead of Brezhnev Doctrine, the US then
            # played *that* for Ops, and the headline never ended.
            _seed_revealed_card(state, reveal_queue[0])
            seeded_reveal = True
            ctx = state.ctx()
        mask = np.asarray(ts.ActionMask.generate_flat_mask(state))
        legal = np.flatnonzero(mask)
        if len(legal) == 0:
            break

        # Entry is done once its intents are spent and the engine wants a new card.
        # A headline entry has no play-mode decision of its own, so it is finished once both
        # cards are chosen and the engine has left the Headline phase. Without this the driver
        # ran on into the next action round and played an extra card.
        if e.headlines:
            # Not merely "out of the Headline phase": a headline can hand its Ops to the other
            # player -- "Lone Gunman" gave the USSR the 1 Op it couped Libya with at turn 4 --
            # and those decisions come after the phase ends. Wait for a fresh card request.
            if (not headline_ids and state.current_phase != ts.Phase.HEADLINE
                    and dt == ts.DecisionType.SELECT_CARD):
                break
        elif (picked_card and second_cid is not None
                and dt == ts.DecisionType.SELECT_CARD
                and _find(state, legal, ts.DecisionType.SELECT_CARD,
                          lambda ma: int(ma.primary_id) == second_cid) is not None):
            pass  # second card of this entry; select it below rather than ending the entry

        elif (picked_card and dt == ts.DecisionType.SELECT_CARD
                and int(ctx.resolving_card) == 0):
            # Only a card request from the engine itself ends the entry. A SELECT_CARD raised
            # while a card is still resolving is that card's own sub-decision -- which of the
            # five peeked cards to discard, say -- and belongs to this entry.
            # Do not require a play mode to have been chosen: a scoring card has no
            # SELECT_PLAY_MODE at all, its event fires on selection. Requiring one meant the
            # driver sailed past the entry into the opponent's action round and played an
            # extra card -- turn 2 AR6 of replay 100 put two stray influence into Canada.
            # Nor an empty queue: leftovers are normal whenever the event placed its own
            # influence, and waiting for them let turn 1 AR5 run two whole cards too far.
            break

        mover = _acting(state)
        chosen: Optional[int] = None
        # Reset per decision: a stale value would misread the next target's provenance.
        target_from_ops = False
        informative = False

        if (dt == ts.DecisionType.SELECT_CARD and picked_card
                and second_cid is not None and not headline_ids):
            chosen = _find(state, legal, ts.DecisionType.SELECT_CARD,
                           lambda ma: int(ma.primary_id) == second_cid)
            if chosen is None:
                # UN Intervention names an opponent-associated card out of the player's own
                # hand, and the log says outright which one. The turn's hand list does not
                # always contain it: at turn 5 AR4 of replay 159 the US plays Quagmire through
                # UN Intervention and Quagmire appears nowhere in the eight cards the log
                # credits them with. The log's own statement is the better evidence, so the
                # card is seated in that hand -- the same forcing a headline card already gets.
                loc = (ts.CardLocation.HAND_US if mover == ts.Player.US
                       else ts.CardLocation.HAND_USSR)
                state.set_card_location(second_cid, loc)
                mask = np.asarray(ts.ActionMask.generate_flat_mask(state))
                legal = np.flatnonzero(mask)
                chosen = _find(state, legal, ts.DecisionType.SELECT_CARD,
                               lambda ma: int(ma.primary_id) == second_cid)
            if chosen is not None:
                second_cid = None
                informative = True

        elif dt == ts.DecisionType.SELECT_CARD and int(ctx.resolving_card) != 0:
            # A card resolving a sub-decision over cards: take the next logged discard, and
            # once they are all spent decline the rest, which returns them to the deck.
            for slot, want_c in enumerate(discard_queue):
                chosen = _find(state, legal, ts.DecisionType.SELECT_CARD,
                               lambda ma, w=want_c: int(ma.primary_id) == w)
                if chosen is not None:
                    discard_queue.pop(slot)
                    informative = True
                    break
            if chosen is None and int(ctx.resolving_card) == _MISSILE_ENVY and envy_took:
                # Missile Envy's own reveal, not whichever reveal the entry printed first. It
                # asks which card to hand over only when the highest Ops cards tie, and the
                # answer is the card named under "Event: Missile Envy" -- at turn 4's headline
                # of replay 14 the entry also carries Grain Sales To Soviets, whose reveal
                # (Marshall Plan) stood at the head of the queue and was handed over instead.
                chosen = _find(state, legal, ts.DecisionType.SELECT_CARD,
                               lambda ma: int(ma.primary_id) == envy_took[1])
                if chosen is not None:
                    if envy_took[1] in reveal_queue:
                        reveal_queue.remove(envy_took[1])
                    informative = True
            if chosen is None:
                # Then whatever the log says was revealed. Where the engine asks which card to
                # hand over -- Missile Envy tying NORAD against Cuban Missile Crisis at both
                # 3 Ops -- the reveal is the answer. Guessing gave away Cuban Missile Crisis
                # and fired its event, after which every USSR coup was an instant loss.
                for slot, want_c in enumerate(reveal_queue):
                    chosen = _find(state, legal, ts.DecisionType.SELECT_CARD,
                                   lambda ma, w=want_c: int(ma.primary_id) == w)
                    if chosen is not None:
                        reveal_queue.pop(slot)
                        informative = True
                        break
            if chosen is None:
                # Then a card the entry goes on to fire the event of, which is how Star Wars
                # records its pick from the discard pile. Only Star Wars reaches into that pile,
                # so only Star Wars may put a card there: at turn 4's headline of replay 234 the
                # entry goes on to fire Nuclear Test Ban, handed over by Missile Envy, and Our
                # Man in Tehran's peek asked for a card first. Answering that peek from this
                # queue moved Nuclear Test Ban out of the US hand and into the discard pile,
                # after which Missile Envy found nothing above 3 Ops to take and the event that
                # was owed never fired.
                for slot, want_c in enumerate(event_card_queue):
                    chosen = _find(state, legal, ts.DecisionType.SELECT_CARD,
                                   lambda ma, w=want_c: int(ma.primary_id) == w)
                    if chosen is None and int(ctx.resolving_card) == _STAR_WARS:
                        # Only hands are forced from the log, so our discard pile holds just
                        # what this reconstruction happened to play; a card the humans had
                        # discarded turns earlier may still be sitting in the draw deck. The
                        # log states it was in the pile, so put it there -- the same forcing a
                        # headline already gets. At turn 9 AR7 of replay 104 How I Learned To
                        # Stop Worrying was never offered for that reason.
                        state.set_card_location(want_c, ts.CardLocation.DISCARD_PILE)
                        mask = np.asarray(ts.ActionMask.generate_flat_mask(state))
                        legal = np.flatnonzero(mask)
                        chosen = _find(state, legal, ts.DecisionType.SELECT_CARD,
                                       lambda ma, w=want_c: int(ma.primary_id) == w)
                    if chosen is not None:
                        event_card_queue.pop(slot)
                        informative = True
                        break
            if chosen is None:
                chosen = _find_confirm_done(state, legal)

        elif dt == ts.DecisionType.SELECT_CARD and headline_ids:
            side = "US" if mover == ts.Player.US else "USSR"
            want_c = headline_ids.get(side)
            if want_c:
                chosen = _find(state, legal, ts.DecisionType.SELECT_CARD,
                               lambda ma: int(ma.primary_id) == want_c)
                if chosen is None:
                    loc = (ts.CardLocation.HAND_US if mover == ts.Player.US
                           else ts.CardLocation.HAND_USSR)
                    state.set_card_location(want_c, loc)
                    mask = np.asarray(ts.ActionMask.generate_flat_mask(state))
                    legal = np.flatnonzero(mask)
                    chosen = _find(state, legal, ts.DecisionType.SELECT_CARD,
                                   lambda ma: int(ma.primary_id) == want_c)
                if chosen is not None:
                    informative = True
                    headline_ids.pop(side, None)
                else:
                    raise ConversionFailure(Mismatch(
                        conv.replay_id, e.turn, e.phase, e.player, e.card,
                        "headline card not selectable",
                        f"{side} headline #{want_c} not legal"))
                    headline_ids.pop(side, None)

        elif dt == ts.DecisionType.SELECT_CARD and cid_target and not picked_card:
            chosen = _find(state, legal, ts.DecisionType.SELECT_CARD,
                           lambda ma: int(ma.primary_id) == cid_target)
            if chosen is None:
                # The engine deals from its own shuffle, so a card the human held may not be
                # in its hand. The log is authoritative about what was held.
                if cid_target == 6:
                    state.china_card_holder = mover
                    state.china_card_playable = True
                else:
                    loc = (ts.CardLocation.HAND_US if mover == ts.Player.US
                           else ts.CardLocation.HAND_USSR)
                    state.set_card_location(cid_target, loc)
                mask = np.asarray(ts.ActionMask.generate_flat_mask(state))
                legal = np.flatnonzero(mask)
                chosen = _find(state, legal, ts.DecisionType.SELECT_CARD,
                               lambda ma: int(ma.primary_id) == cid_target)
            if chosen is not None:
                picked_card, informative = True, True
            else:
                raise ConversionFailure(Mismatch(
                    conv.replay_id, e.turn, e.phase, e.player, e.card,
                    "card not selectable",
                    f"card #{cid_target} not among {len(legal)} legal actions"))
                return False

        elif (dt == ts.DecisionType.SELECT_PLAY_MODE
                and (int(ctx.pending_op_card) or cid_target or 0) not in mode_picked_for):
            # An opponent's card can only be played for Ops -- its event fires on its own,
            # so "Event: X" in the text does not mean the play mode was Event.
            # Ask about the card the engine is actually playing, which need not be the entry's
            # own: at turn 7's headline Grain Sales hands the US the USSR's NATO, and this
            # decision is NATO's. Judging it by the headline entry instead made it look like a
            # neutral card played as an Event, so the US never got to coup with it.
            play_cid = int(ctx.pending_op_card) or cid_target
            side = str(ts.CardData.get_card_info(play_cid)["side"]) if play_cid else "NONE"
            mine = "US" if mover == ts.Player.US else "USSR"
            opponent_card = side not in ("NONE", mine)
            if e.played_card and cid_target == _UN_INTERVENTION:
                # UN Intervention is played as its Event; the Ops that follow are the named
                # card's. Reading the "Place Influence" header as the play mode instead had
                # the engine spend UN Intervention's own Ops and never ask for NORAD.
                want = PLAY_MODE_ACTION["event"]
            elif (play_cid == cid_target and cid_target and not opponent_card
                    and e.event_first and e.events):
                # Only for the entry's own card. A headline entry has none, and the play mode
                # it reaches belongs to a card an event put into play -- NATO, via Grain Sales
                # -- whose Ops header is the real signal.
                # Own or neutral card whose "Event:" line precedes the Ops header: it was
                # played as an Event and the Ops belong to the event (ABM Treaty grants the
                # US 4 Ops). Reading that header as the play mode had the engine spend the
                # card's own Ops, and the coup then never resolved.
                want = PLAY_MODE_ACTION["event"]
            elif e.mode == "space" or (not e.mode and e.space):
                want = PLAY_MODE_ACTION["space"]
            elif e.mode or opponent_card:
                want = PLAY_MODE_ACTION["ops"]
            elif e.space:
                want = PLAY_MODE_ACTION["space"]
            elif e.events:
                want = PLAY_MODE_ACTION["event"]
            else:
                want = None
            if want is None or not mask[want]:
                for fallback in ("ops", "event", "space"):
                    if mask[PLAY_MODE_ACTION[fallback]]:
                        want = PLAY_MODE_ACTION[fallback]
                        break
            if want is not None and mask[want]:
                chosen, informative = want, True
                mode_picked_for.add(int(ctx.pending_op_card) or cid_target or 0)
            else:
                raise ConversionFailure(Mismatch(
                    conv.replay_id, e.turn, e.phase, e.player, e.card,
                    "play mode illegal",
                    f"wanted {'ops' if e.mode else ('space' if e.space else 'event')}, "
                    f"legal modes {[k for k, v in PLAY_MODE_ACTION.items() if mask[v]]}"))
                return False

        elif dt == ts.DecisionType.CHOOSE_TIMING_BRANCH:
            # Playing an opponent's card for Ops asks which resolves first. The log answers by
            # line order: at turn 1 AR5 of replay 100 "Place Influence" precedes "Event:", so
            # that one is Ops first. Always choosing event-first mis-sequenced those entries.
            want_branch = 0 if e.event_first is False else 1
            chosen = _find(state, legal, ts.DecisionType.CHOOSE_TIMING_BRANCH,
                           lambda ma: int(ma.primary_id) == want_branch)
            informative = chosen is not None

        elif (dt == ts.DecisionType.SELECT_OP_MODE
                and int(ctx.pending_op_card) in _FREE_ACTION_CARDS
                and not sections and e.mode not in _OP_MODE):
            # Junta and Tear Down This Wall place first and then *may* make a free coup or
            # realignment, so the engine bars Influence from the Ops their event grants and
            # lets INFLUENCE stand for declining -- see free_action_bars_influence in
            # action_mask.cpp. A log that records the placement and nothing after it is a
            # player who declined: at turn 5 AR6 of replay 174 the US plays Junta for its
            # event, puts 2 Influence into Chile, and makes no coup or realignment at all.
            #
            # Only where the log describes no Ops. Played for Operations rather than for its
            # event, the card spends them and the log prints a header, which the branch below
            # answers from.
            chosen = _find(state, legal, ts.DecisionType.SELECT_OP_MODE,
                           lambda ma: int(ma.primary_id) == int(ts.OpMode.INFLUENCE))
            informative = chosen is not None

        elif dt == ts.DecisionType.SELECT_OP_MODE and (sections or e.mode in _OP_MODE):
            # An entry can hold more than one Ops section, and they are asked for in the order
            # the log prints them: CIA Created reveals the USSR hand, hands the US 1 Op to coup
            # with, and only then spends its own Op for the USSR. Taking the entry's single
            # mode for every such decision lost the second operation entirely.
            # A space race is settled at the play mode, not here, so its section is still at
            # the head of the queue when the next Ops decision arrives and must be stepped
            # over. At turn 6's headline of replay 156 the US headlines Grain Sales To
            # Soviets, draws Brezhnev Doctrine and puts it on the space race; the USSR's Junta
            # then places 2 Influence in Chile and asks how to spend its free action, and the
            # space section answered for it -- "space" is not an Ops mode, so nothing was
            # chosen and the coup on Panama the log records never happened.
            while sections and sections[0].mode not in _OP_MODE:
                sections.pop(0)
            section = sections.pop(0) if sections else None
            mode = section.mode if section is not None else e.mode
            cur_mode = mode
            if section is not None:
                pq[:] = section_queue(section)
                step_outcomes[:] = section_outcomes(section)
            if mode in _OP_MODE:
                om = int(_OP_MODE[mode])
                # primary_id only. INFLUENCE is 0 and secondary_id defaults to 0, so matching
                # either field silently selected the first legal action -- REALIGN -- and the
                # engine then correctly offered a realignment mask, which looked like a mask
                # bug.
                chosen = _find(state, legal, ts.DecisionType.SELECT_OP_MODE,
                               lambda ma: int(ma.primary_id) == om)
                informative = chosen is not None

        elif dt == ts.DecisionType.CHOOSE_BRANCH:
            if int(ctx.resolving_card) == _CHERNOBYL and e.region_choice is not None:
                # The log names the region outright -- "US chooses South America" -- and
                # Chernobyl's branches are the six regions in order. Left to the branch search,
                # which judges a branch by the board and score it reaches, the choice was
                # arbitrary: nothing about Chernobyl moves either. At turn 8 AR2 of replay 161
                # the US chose South America and the engine took Europe, and the USSR's North
                # Sea Oil three action rounds later could place nothing at all -- East Germany
                # and Spain/Portugal are both in Europe, so its 3 Ops went nowhere.
                chosen = _find(state, legal, ts.DecisionType.CHOOSE_BRANCH,
                               lambda ma, r=e.region_choice: int(ma.primary_id) == r)
                informative = chosen is not None
            if chosen is None:
                chosen = _choose_branch(state, legal, eq + pq, raw, e, returned_cid)
                informative = chosen is not None

        elif dt == ts.DecisionType.POINT_NODE and not pq and not eq_here and sections:
            # A further Ops section that the engine never announces with a play mode. Che
            # grants the USSR a second coup when the first removes influence, and offers it
            # straight as another target choice, so waiting for a SELECT_OP_MODE to advance
            # the section left the second coup with no target -- at turn 4 AR4 of replay 103
            # the USSR coups Sudan and then Saharan States under two headers.
            section = sections.pop(0)
            pq[:] = section_queue(section)
            step_outcomes[:] = section_outcomes(section)
            cur_mode = section.mode
            if pq:
                continue
            # A header with nothing under it is a decline, and it has to be answered as one
            # rather than skipped: the next section belongs to the other player. At turn 8 AR2
            # of replay 123 the US plays Ortega Elected in Nicaragua, the USSR has no coup
            # worth making -- "Coup (1 Ops):" and then nothing -- and the US places 2 Influence
            # of its own. Falling through to that section answered the USSR's coup with
            # Nigeria, which at DEFCON 2 is thermonuclear war and lost the USSR the game.
            chosen = _find_confirm_done(state, legal)
            informative = chosen is not None
            if chosen is None:
                continue

        elif dt == ts.DecisionType.POINT_NODE and (pq or eq_here or loose):
            # Ask the queue that matches what the engine is doing: while a card is resolving,
            # these are the event's own placements, otherwise they are the Ops. Within a queue
            # take the first target actually offered rather than insisting on the head, since
            # an event may resolve some of its placements itself and never ask about them.
            in_event = int(ctx.resolving_card) != 0
            # The queue belonging to the card the engine says is resolving comes first, so one
            # event cannot spend a point the log wrote under the other. See event_point_queues.
            own = evq.get(int(ctx.resolving_card)) if in_event else None
            if own:
                order = (own, eq, pq, loose)
            else:
                order = ((eq_here, pq, loose) if in_event else (pq, eq_here, loose))
            for queue in order:
                for slot, want_c in enumerate(queue):
                    chosen = _find(state, legal, ts.DecisionType.POINT_NODE,
                                   lambda ma, w=want_c: int(ma.primary_id) == w)
                    if chosen is not None:
                        queue.pop(slot)
                        if queue is not pq and queue is not eq and want_c in eq:
                            # A per-event queue holds the same points the shared event queue
                            # does, so spending one there has to spend it here as well. Left
                            # in, every placement under an "Event:" header could be made twice.
                            eq.remove(want_c)
                        # Which queue answered decides whether a die follows: the Ops queue
                        # holds coup and realignment targets, the event queue holds placements.
                        target_from_ops = queue is pq
                        informative = True
                        break
                if chosen is not None:
                    break
            if chosen is None and sections:
                # Neither queue can answer, but the entry has more sections to come: this
                # decision belongs to one of them. A headline holds two cards, and the queues
                # do not separate them -- at turn 7's headline of replay 128 the USSR headlines
                # Che and the US headlines Junta, and Che's coup was offered while the event
                # queue still held Junta's two placements into Venezuela, a battleground Che
                # may not touch. Declining there threw away both of Che's coups.
                section = sections.pop(0)
                pq[:] = section_queue(section)
                step_outcomes[:] = section_outcomes(section)
                cur_mode = section.mode
                continue
            if chosen is None:
                # An event can ask for more than the board can give: Suez Crisis removes four
                # US Influence across France, the UK and Israel, and at turn 2 AR6 of replay
                # 105 only three were there to remove. The engine offers the pass; taking it
                # finishes the event and leaves the entry's own Ops still to spend, where
                # abandoning the entry lost them.
                chosen = _find_confirm_done(state, legal)
            if chosen is None:
                names = ", ".join(ts.MapData.get_country_info(c)["name"] for c in (pq + eq))
                raise ConversionFailure(Mismatch(
                    conv.replay_id, e.turn, e.phase, e.player, e.card,
                    "target not legal",
                    f"none of the remaining targets ({names}) are legal; engine "
                    f"(resolving={int(ctx.resolving_card)}, op_card={int(ctx.pending_op_card)}, "
                    f"dt={str(dt).split('.')[-1]}) offers "
                    f"{sorted(ts.MapData.get_country_info(int(p))['name'] for p in (int(ts.ActionMask.decode_flat_action(state, int(a)).primary_id) for a in legal) if 0 <= p < 84)[:12]}"))
                pq.clear()
                eq.clear()
                break

        if chosen is None and dt == ts.DecisionType.POINT_NODE and not pq and not eq:
            # An Ops header with nothing under it means the player declined. Che offers the
            # USSR a coup rather than requiring one, and at turn 4 AR2 the US held no influence
            # in any non-battleground country of the Americas or Africa, so there was nothing
            # worth couping and the USSR passed: "Coup (3 Ops):" and then no target, no result,
            # no military ops line. The engine already allows this, so take the pass.
            chosen = _find_confirm_done(state, legal)
            informative = chosen is not None

        if chosen is None:
            # Named the way the decision names them. Only a POINT_NODE is choosing between
            # countries; an op mode is choosing how to spend the Ops, and a card selection is
            # choosing a card, and running either of those through the country table produced
            # a list of countries nobody was being offered -- "which of 3 options was taken:
            # ['Canada', 'Norway', 'United Kingdom']" for a choice between influence, coup and
            # realignment.
            offered = sorted(_name_options(state, dt, legal))
            raise ConversionFailure(Mismatch(
                conv.replay_id, e.turn, e.phase, e.player, e.card,
                "decision not determined by the log",
                f"{str(dt).split('.')[-1]} for "
                f"{'US' if mover == ts.Player.US else 'USSR'} "
                f"(resolving={int(ctx.resolving_card)}, op_card={int(ctx.pending_op_card)}); "
                f"nothing in the entry says which of {len(offered)} options was taken: "
                f"{offered[:12]}"))

        # Anything the engine settles with a die it rolls itself: coups, realignments, and war
        # events, whose target is chosen the same way but never carried an Ops mode.
        if dt == ts.DecisionType.POINT_NODE and chosen is not None:
            target = int(ts.ActionMask.decode_flat_action(state, chosen).primary_id)
            # Only decisions that actually roll. An influence placement into a country that
            # happens to be a war or coup target rolls nothing, and searching for a seed to
            # match it can never succeed: at turn 1 AR2 of replay 109 the US places two
            # influence in South Korea, the country the USSR had just lost the Korean War in.
            # Which queue answered decides it, not whether a card is resolving: Che's free
            # coups happen inside its own event, while at turn 4 AR7 of replay 101 South
            # African Unrest places influence in Angola -- an ordinary placement -- in the very
            # country the US then realigns. The Ops queue holds coup and realignment targets;
            # the event queue holds placements.
            rolled = ((target_from_ops and cur_mode in ("coup", "realign")
                       and target in e.targets)
                      or (int(ctx.resolving_card) != 0 and target in e.war_targets))
            # Constrain only the country this operation resolves against. The entry's other
            # influence has not happened yet at this point: at turn 2 AR1 of replay 101 the
            # USSR coups Panama and only then does Independent Reds place US influence in
            # Czechoslovakia, so demanding the whole entry's board matched no roll at all and
            # the coup was left to chance -- succeeding where the log has it fail.
            want = expected_counts(e, raw)
            # Realignments roll once per target and the log prints the running result of each,
            # so the country's *final* value is wrong for every roll but the last: at turn 4's
            # headline of replay 102 North Korea goes 10 US to 5 and only then to 1. Consume
            # the logged results in order, falling back to the entry's board for a roll that
            # changed nothing and so printed no influence line.
            outcome = {target: want[target]} if target in want else want
            for slot, (cid_r, us_r, ussr_r) in enumerate(step_outcomes):
                if cid_r == target:
                    outcome = {target: (us_r, ussr_r)}
                    step_outcomes.pop(slot)
                    break
            # A coup whose die the log states needs no search at all: the value is carried on
            # the target choice itself, in secondary_id, which is where the engine looks for a
            # forced roll whether the coup resolves at a chance node or immediately. Che's two
            # coups take the second path -- they resolve inside the event, with no chance node
            # to steer afterwards.
            #
            # Searching for the die could not work anyway once anything else in the entry
            # touches the same country: at turn 4's headline of replay 129 the US headlines
            # Junta and coups Panama, then the USSR's Liberation Theology places an Influence
            # there, so the expectation held Panama's final [1][1] and no roll could produce it.
            if rolled and cur_mode == "realign" and realign_roll_queue:
                pass   # likewise: both dice are given at the realignment's chance node
            elif rolled and cur_mode == "coup" and coup_roll_queue:
                pass   # settled at the chance node the target choice opens
            elif rolled and force_outcome(state, chosen, outcome):
                seed_settled = True
            elif rolled:
                raise ConversionFailure(Mismatch(
                    conv.replay_id, e.turn, e.phase, e.player, e.card,
                    "could not reproduce outcome",
                    f"no rng_state reproduced the logged "
                    f"{e.mode or e.event_mode or 'war'} result"))

        if random_discards:
            # Do this on whichever step actually fires the event; the search accepts the seed
            # the engine already has, so steps that discard nothing cost one clone and pass.
            if _force_random_discard(state, random_discards, action=int(chosen)):
                random_discards = []

        if informative:
            obs = np.asarray(ts.extract_observation(state, mover), dtype=np.float32)
            conv.samples.append((obs, mask.copy(), int(chosen),
                                 1 if mover == ts.Player.US else -1))
            conv.decisions_emitted += 1

        ts.Engine.step_flat(state, int(chosen))

    return True


def _hand_after(turn_hand, played, pending=None):
    """Cards still held: the turn's logged hand, less what is spent and what is not yet held."""
    pending = pending or set()
    return [c for c in turn_hand if c not in played and c not in pending]


_SALT_NEGOTIATIONS = 43
_ASK_NOT = 77


def _sort_key_for_keeping(card: int):
    """The order a hand is assumed to have been dealt in: the better cards first.

    Higher Ops first, and within an Ops value the US and neutral cards before the USSR ones.
    Used only to split a turn's cards into those dealt at the start and those drawn during it,
    where nothing in the log distinguishes them.
    """
    info = ts.CardData.get_card_info(card)
    side = str(info["side"])
    return (-int(info["ops"]), 0 if side in ("US", "NONE") else 1, card)


def _mid_turn_acquisitions(raws, turn: int, side: str, held: List[int]) -> Dict[int, int]:
    """Cards in a turn's list that were picked up during it rather than dealt at its start.

    A turn's list is every card that became visible during the turn, which is not the same as
    the hand it was dealt. Two cards reach a hand mid-turn, and both are recorded:

    SALT Negotiations reclaims a card from the discard pile, so that card demonstrably was not
    dealt -- at turn 5 of replay 64 the US list holds Red Scare/Purge, which they retrieved at
    AR7, and having it four action rounds early gave Missile Envy the wrong card to take.

    "Ask Not What Your Country Can Do For You" discards any number of cards and draws that many
    replacements. What was discarded was in the dealt hand; the replacements were not, and there
    are exactly as many of them as there were discards. Which of the turn's remaining cards they
    are is not recorded, so the cards are ordered best-first and the tail of that order is taken
    as the draws.

    Returns {card: index of the entry that brings it into hand}.
    """
    acquired: Dict[int, int] = {}
    held_set = set(held)

    for idx, raw in enumerate(raws):
        e = parse_entry(raw)
        if e.turn != turn:
            continue
        in_play = {card_id(e.card)} if (e.card and " & " not in e.card) else set()
        in_play |= {card_id(nm) for nm in (e.headlines or {}).values()}

        if _SALT_NEGOTIATIONS in in_play:
            for rev_side, nm in (e.revealed or []):
                c = card_id(nm)
                if c and rev_side == side and c in held_set:
                    acquired[c] = idx

        if _ASK_NOT in in_play:
            discarded = [c for c in (card_id(nm) for sd, nm in (e.discards or [])
                                     if sd == side) if c]
            if not discarded:
                continue
            # Everything this side had already spent was necessarily in the dealt hand, as was
            # everything it discarded here, and Ask Not itself if this side played it.
            pinned = set(discarded) | (in_play & held_set)
            for earlier in raws[:idx]:
                pe = parse_entry(earlier)
                if pe.turn != turn:
                    continue
                if pe.player == side and pe.card and " & " not in pe.card:
                    c = card_id(pe.card)
                    if c:
                        pinned.add(c)
                nm = (pe.headlines or {}).get(side)
                if nm:
                    c = card_id(nm)
                    if c:
                        pinned.add(c)
            candidates = [c for c in held if c not in pinned and c not in acquired]
            candidates.sort(key=_sort_key_for_keeping)
            for c in candidates[len(candidates) - min(len(discarded), len(candidates)):]:
                acquired[c] = idx

    return acquired


def _apply_hands(state, us_cards, ussr_cards) -> None:
    for c in range(1, 111):
        loc = state.get_card_location(c)
        if loc in (ts.CardLocation.HAND_US, ts.CardLocation.HAND_USSR):
            state.set_card_location(c, ts.CardLocation.DISCARD_PILE)
    for c in us_cards:
        state.set_card_location(c, ts.CardLocation.HAND_US)
    for c in ussr_cards:
        state.set_card_location(c, ts.CardLocation.HAND_USSR)


def _board_diff(state, countries) -> str:
    """Which countries disagree with the log, and how -- named, not counted."""
    out = []
    for key, c in (countries or {}).items():
        cid = country_id(key)
        if cid is None:
            continue
        try:
            lus, lussr = int(c["inflUS"]), int(c["inflUSSR"])
        except (KeyError, TypeError, ValueError):
            continue
        cur = state.get_country(cid)
        if int(cur.us_influence) != lus or int(cur.ussr_influence) != lussr:
            out.append(f"{ts.MapData.get_country_info(cid)['name']} "
                       f"engine [{int(cur.us_influence)}][{int(cur.ussr_influence)}] "
                       f"log [{lus}][{lussr}]")
    return "countries differ: " + "; ".join(out)


def _board_matches(state, countries) -> int:
    bad = 0
    for key, c in (countries or {}).items():
        cid = country_id(key)
        if cid is None:
            continue
        try:
            lus, lussr = int(c["inflUS"]), int(c["inflUSSR"])
        except (KeyError, TypeError, ValueError):
            continue
        cur = state.get_country(cid)
        if int(cur.us_influence) != lus or int(cur.ussr_influence) != lussr:
            bad += 1
    return bad


RE_ACTION_ROUND = re.compile(r"^AR(\d+)$")


def unfinished_final_turn(raws: List[Dict]) -> Optional[int]:
    """The turn number the recording stops inside, or None if it plays its last turn out.

    Every turn plays all of its action rounds -- six in turns 1 to 3 and seven after -- unless
    the game ends. A last turn that stops short of its own last action round, with no ending
    recorded, is a recording that stopped rather than a game that finished.

    That matters beyond the entries never written. The turn's hand list is assembled from the
    cards that became visible during it, so a turn cut short lists only the few played before
    the recording stopped -- replay 246's turn 9 reaches AR3 and credits the US with three
    cards and the USSR with six, where a turn 9 hand holds nine. Every decision converted in
    such a turn was driven from a hand the player never held.

    156 of the corpus's 278 games end this way, and the great majority simply stop: the log
    narrates no win, no final scoring and no DEFCON 1.
    """
    if not raws:
        return None
    last_turn = parse_entry(raws[-1]).turn
    if not last_turn:
        return None
    reached = 0
    for raw in raws:
        e = parse_entry(raw)
        if e.turn != last_turn:
            continue
        m = RE_ACTION_ROUND.match(e.phase or "")
        if m:
            reached = max(reached, int(m.group(1)))
    expected = 6 if last_turn <= 3 else 7
    return last_turn if reached < expected else None


def _rewind_to_turn_start(conv: "Conversion", turn: int) -> None:
    """Give back everything converted in the turn the record stops in.

    A log that ends mid-turn does not only lose the entries it never wrote. The turn's hand
    list is assembled from the cards that became visible during it, so a turn cut short lists
    only the few that were played before the recording stopped -- and every decision already
    converted in that turn was driven from a hand that is not the one the player held. The
    board and the score still check out, because the log's own board is what they are checked
    against, but the position the model would learn from is wrong.

    So the whole turn goes, not just the entry that failed. Nothing earlier is touched: those
    turns have complete hand lists.
    """
    started_turn, entries, samples = conv.turn_started_at
    if started_turn != turn:
        return
    del conv.samples[samples:]
    conv.entries_converted = entries


def _is_the_record_ending(m: Mismatch, raws: List[Dict]) -> bool:
    """Is this failure the log running out rather than the reconstruction going wrong?

    Only on the file's very last entry, and only where the engine asked something the log
    never answers. On any earlier entry an unanswered decision means the answer is somewhere we
    are not reading; on the last one it means the recording stopped. Nine of the corpus's games
    end that way -- at turn 3 AR3 of replay 153 the file's final entry is "Event: Arab-Israeli
    War" and not one word more, and at turn 6 AR1 of replay 251 it is "Coup (3 Ops):" with no
    target, roll or result.

    A disagreement is never treated this way, wherever it lands. Fifteen games fail their last
    entry on a pass the log records and the engine will not allow, and three more on a board or
    a score: those are the reconstruction being wrong about something the log does state, and
    they stay failures. The distinction is between the log saying nothing and the log saying
    something else.
    """
    if not raws:
        return False
    # A failure inside a turn the recording stops in is not worth diagnosing: the turn is a
    # fragment and is dropped either way, so whether we could have reproduced it is moot. At
    # turn 9 AR3 of replay 246 the log stops four action rounds short with three cards credited
    # to the US and six to the USSR, and the board it disagrees about is one reached from hands
    # neither player held.
    if unfinished_final_turn(raws) == m.turn:
        return True
    last = parse_entry(raws[-1])
    if (m.turn, m.phase, m.player) != (last.turn, last.phase, last.player):
        return False
    if m.kind == "decision not determined by the log":
        return True
    if m.kind == "logged pass is not legal":
        # A bare "Turn 4, USSR AR2" header at the foot of an entry means a player out of cards
        # skipping their round -- when entries follow it. At the end of the file it means the
        # opposite: the round was announced and never written. Every one of the corpus's 14
        # remaining pass failures is that shape, the header being the last thing in the file
        # with the player still holding cards, and not one occurs mid-file, where a genuine
        # skip is always followed by the other player's entries.
        text = (last.text or "").rstrip()
        matches = list(RE_PASSED_ROUND.finditer(text))
        return bool(matches) and text.endswith(matches[-1].group(0))
    return False


def convert_game(game: Dict) -> Conversion:
    """Rebuild each entry's position from the log, drive it, and verify the outcome.

    Two things make this per-entry rather than a forward simulation of the whole game. Drift
    cannot accumulate: every entry starts from the logged position, so one mis-parsed entry
    does not poison the rest. And the forward step is the *check* -- if the actions we
    extracted are right, replaying them must reproduce the log's next board, so a parse error
    shows up as a board mismatch on that entry instead of passing silently into the dataset.

    Hands are tracked across the turn, since the log records them only per turn: the hand at
    an entry is the turn's logged hand minus what has already been played. The card a player
    plays must be in that tracked hand, which is itself a check on the hand model.
    """
    conv = Conversion(replay_id=int(game.get("replay_id", -1)))
    raws = game.get("all_turns", [])
    hands = game.get("hands", {}) or {}

    handicap = unsupported_handicap(raws)
    if handicap is not None:
        conv.skipped = f"handicap {handicap}: the engine sets up {_STANDARD_HANDICAP}"
        return conv

    state = ts.GameState()
    ts.Engine.init_game(state, 12345)
    _drain(state)

    try:
        _convert_entries(state, raws, hands, conv)
    except ConversionFailure as failure:
        # Reported, not raised on: one unconvertible game should not stop a sweep of hundreds,
        # and the caller decides whether a partial game is usable. conv.failure being set means
        # the entries after it were never converted.
        if _is_the_record_ending(failure.mismatch, raws):
            conv.truncated_at = failure.mismatch
            _rewind_to_turn_start(conv, int(failure.mismatch.turn))
        else:
            conv.failure = failure.mismatch
            conv.mismatches.append(failure.mismatch)
    conv.game_ended = bool(ts.Engine.is_terminal(state))
    if conv.failure is None and conv.truncated_at is None and not conv.game_ended:
        # The log played no ending and stopped mid-turn: that turn is a fragment, whether or
        # not anything in it happened to fail. See unfinished_final_turn.
        unfinished = unfinished_final_turn(raws)
        if unfinished is not None:
            conv.truncated_at = Mismatch(
                conv.replay_id, unfinished, "", "", None,
                "log stops mid-turn",
                "the recording stops before this turn's last action round and records no "
                "ending, so the turn's hand list is a fragment and the turn is not "
                "training data")
            _rewind_to_turn_start(conv, unfinished)
    conv.decisions_emitted = len(conv.samples)
    return conv


def _is_skipped_round(e: Entry) -> bool:
    """An action round the log gives an entry to and nothing else: the player skipped it.

    Distinct from the bare "Turn 5, USSR AR4" header at the foot of another entry, which says
    the same thing about a round that gets no entry of its own. This one has its own header and
    an empty body -- "Turn 9, USSR AR8: :" -- and comes up where a player is granted the eighth
    action round and declines it.

    The engine may already have taken the round away: it passes a player with an empty hand
    without asking. Where it has not -- an eighth round the player could have used -- the pass
    is driven, which is a real decision and worth keeping.
    """
    return (e.phase or "").startswith("AR") and not (
        e.card or e.influence or e.sections or e.targets or e.war_targets or e.events
        or e.headlines or e.space or e.discards or e.revealed or e.mode or e.played_card
        or e.vp_gains)


def _is_turn_end_record(e: Entry, prev: Optional[Entry]) -> bool:
    """The repeated copy of a turn's last entry, which records cleanup rather than a play.

    ts-replayer hangs the end-of-turn bookkeeping on the header of the entry above it, so the
    last action round of every turn appears twice: once with the play, and once with the same
    turn, phase, player and card but a body holding only the effects that expired, the same
    board, and DEFCON improved by one. There are 357 of them across the 287 downloaded games.

    Driven as a play the second copy asked the engine for a card it had already ended the turn
    to give -- 49 of the 182 games that stopped did so exactly there, at the last action round
    of turn 1, under a dozen different card names because the card is only the one copied down
    from the entry above.
    """
    if prev is None:
        return False
    if (e.turn, e.phase, e.player, e.card) != (prev.turn, prev.phase, prev.player, prev.card):
        return False
    return not (e.influence or e.sections or e.targets or e.war_targets or e.events
                or e.headlines or e.space or e.discards or e.revealed or e.mode
                or e.played_card)


def _convert_entries(state: ts.GameState, raws, hands, conv: Conversion) -> None:
    cur_turn = None
    turn_hands = {"US": [], "USSR": []}
    played = {"US": set(), "USSR": set()}
    # Cards this turn's list names that the side did not hold at the start of it, and the entry
    # that puts each into hand. See _mid_turn_acquisitions.
    pending: Dict[str, Dict[int, int]] = {"US": {}, "USSR": {}}
    prev_raw = None
    prev_entry = None
    # Whether driving the previous entry carried the game into a new turn. Its narrated score
    # is then out of date -- see _reconcile_scalars.
    prev_crossed_turn = False
    # Where the turn now being converted began: how many entries had been converted and how
    # many samples emitted. A record that stops mid-turn takes the whole turn with it, so this
    # is what a truncation rewinds to. See _rewind_to_turn_start.
    conv.turn_started_at = (0, 0, 0)
    # The turn the record stops in, which is the only one a short hand list can be blamed on
    # the recording for.
    last_turn = max((int(r.get("num")) for r in raws
                     if str(r.get("num", "")).isdigit()), default=0)

    for index, raw in enumerate(raws):
        e = parse_entry(raw)
        if (e.turn, e.phase, e.player) in _KNOWN_INCOMPLETE.get(conv.replay_id,
                                                                frozenset()):
            conv.truncated_at = Mismatch(
                conv.replay_id, e.turn, e.phase, e.player, e.card,
                "log stops mid-entry",
                "the recording ends here, so this turn and everything after it are not "
                "training data")
            _rewind_to_turn_start(conv, int(e.turn))
            return
        conv.entries_total += 1

        if _is_skipped_round(e):
            # Nothing to play. If the engine still has the round to give, take the pass; if it
            # has already passed the player itself, the round is simply over.
            conv.entries_converted += 1
            m = _RE_AR.search(e.phase or "")
            want = ts.Player.US if e.player == "US" else ts.Player.USSR
            ctx = state.ctx()
            if (not ts.Engine.is_terminal(state)
                    and state.current_phase == ts.Phase.ACTION_ROUND
                    and m and int(state.action_round) == int(m.group(1))
                    and int(state.turn) == int(e.turn)
                    and ctx.decision_type == ts.DecisionType.SELECT_CARD
                    and ctx.decision_player == want):
                mask = ts.ActionMask.generate_flat_mask(state)
                if not mask[_PASS]:
                    raise ConversionFailure(Mismatch(
                        conv.replay_id, e.turn, e.phase, e.player, e.card,
                        "logged pass is not legal",
                        f"the log skips {e.player} {e.phase} of turn {e.turn}, and the engine "
                        f"still offers {sum(1 for i in range(110) if mask[i])} cards to play"))
                ts.Engine.step_flat(state, _PASS)
                _drain(state)
            prev_crossed_turn = False
            prev_raw, prev_entry = raw, e
            continue

        if _is_turn_end_record(e, prev_entry):
            # Nothing to drive: the engine ends the turn itself once both players have taken
            # their last action round. Only the record's own bookkeeping is taken -- the
            # improved DEFCON and the effects that expired.
            conv.entries_converted += 1
            conv.board_resyncs += _reconcile_board(state, raw.get("countries"))
            _reconcile_scalars(state, e)
            # Nothing was driven, so nothing carried the game past a score.
            prev_crossed_turn = False
            prev_raw, prev_entry = raw, e
            continue

        if e.turn and e.turn != cur_turn:
            cur_turn = e.turn
            conv.turn_started_at = (int(e.turn), conv.entries_converted,
                                    len(conv.samples))
            h = hands.get(str(e.turn)) or {}
            turn_hands = {
                "US": [c for c in (card_id(n) for n in h.get("us", [])) if c],
                "USSR": [c for c in (card_id(n) for n in h.get("ussr", [])) if c],
            }
            conv.hand_reattributions += _reattribute_hands(raws, int(e.turn), turn_hands)
            # The log lists only the cards a player used, so a game that stops mid-turn leaves
            # hands far too small to have been the ones played from. Top them up to the size
            # the rules deal, capping Ops below anything the log records that side revealing.
            # A new turn restores each side's space race attempt. The engine resets this when
            # it advances the turn itself, which forcing turn numbers from the log bypasses, so
            # a stale count made a legitimate attempt illegal: at turn 4 AR4 of replay 112 the
            # USSR races with Duck and Cover and the engine offered only Ops modes.


            size = 8 if int(e.turn) <= 3 else 9
            claimed = set(turn_hands["US"]) | set(turn_hands["USSR"])
            for side in ("US", "USSR"):
                padded = _pad_hand(state, turn_hands[side], size,
                                   _reveal_ops_cap(raws, e.turn, side), claimed,
                                   final_turn=int(e.turn) >= last_turn)
                claimed |= set(padded)
                turn_hands[side] = padded
            played = {"US": set(), "USSR": set()}
            pending = {side: _mid_turn_acquisitions(raws, int(e.turn), side, turn_hands[side])
                       for side in ("US", "USSR")}

        # --- rebuild the position this entry was decided from ---
        if prev_raw is not None:
            _reconcile_board(state, prev_raw.get("countries"))
        if prev_entry is not None:
            _reconcile_scalars(state, prev_entry, take_score=not prev_crossed_turn)

        _apply_hands(state,
                     _hand_after(turn_hands["US"], played["US"], set(pending["US"])),
                     _hand_after(turn_hands["USSR"], played["USSR"], set(pending["USSR"])))

        # --- the card(s) this entry uses must be in the tracked hand ---
        for side, nm in (e.headlines or {}).items():
            cid = card_id(nm)
            if cid and cid not in _hand_after(turn_hands[side], played[side]):
                conv.hand_misses += 1
            if cid:
                played[side].add(cid)
        for side, nm in (e.discards or []):
            cid = card_id(nm)
            if cid and side in played:
                played[side].add(cid)
        if e.card and " & " not in e.card:
            cid = card_id(e.card)
            side = "US" if e.player == "US" else "USSR"
            if cid and e.player in ("US", "USSR"):
                if cid not in _hand_after(turn_hands[side], played[side]):
                    conv.hand_misses += 1
                played[side].add(cid)
        # A card spent by being *named* rather than selected leaves the hand just the same. UN
        # Intervention names one of the player's own cards and uses its Ops, and leaving it in
        # the tracked hand meant the USSR still held CIA Created at turn 2 AR6 of replay 108,
        # where the log says they had nothing left: Five Year Plan then discarded it, its event
        # opened a decision nobody answered, and the frame was still open at the next entry.
        # Harmless where the named card was never in that player's hand, since _hand_after
        # only ever removes from the logged hand.
        if e.played_card and e.player in ("US", "USSR"):
            named = card_id(e.played_card)
            if named:
                played[e.player].add(named)
        # Missile Envy takes the opponent's highest Ops card, and the card it takes is gone
        # from that hand for the rest of the turn. The turn's list still names it, because it
        # was theirs when the turn began and it became visible while they held it, so leaving
        # it in put a card back that had changed hands: at turn 5 AR1 of replay 114 the US
        # takes Nuclear Test Ban, and the USSR -- trapped by Bear Trap, with nothing left to
        # discard and only a scoring card in hand -- was still being offered it two action
        # rounds later, which is why they were never allowed to play the scoring card the log
        # says they played.
        for side, cid in _taken_from_hand(e):
            if side in played:
                played[side].add(cid)

        before = len(conv.samples)
        turn_before = int(state.turn)
        if prev_entry is not None:
            # Not on the first entry: that one carries the setup placements, and the engine's
            # own initialisation is the position they belong to.
            _reconcile_turn(state, e, conv.replay_id)
        _drive_entry(state, e, conv, raw)
        _drive_passed_rounds(state, e, conv)

        # --- did the engine fire an event the log knows nothing about? ---
        # An event the log never mentions is an event that did not happen, and what it does to
        # the board is not a near miss but a different game. At turn 8 AR3 of replay 158 the US
        # plays Star Wars, which takes Grain Sales To Soviets out of the discard pile; the
        # engine took Blockade instead, and Blockade removes every US Influence from West
        # Germany -- four of them, where the log has the US untouched there and placing three
        # Influence elsewhere with the Cuban Missile Crisis that Grain Sales drew.
        stray = sorted(conv.events_resolved - _events_the_log_names(e) - _UNNAMED_EVENTS)
        if stray:
            del conv.samples[before:]
            names = ", ".join(str(ts.CardData.get_card_info(c)["name"]) for c in stray)
            raise ConversionFailure(Mismatch(
                conv.replay_id, e.turn, e.phase, e.player, e.card,
                "event the log does not mention",
                f"the engine resolved {names}, which this entry never names"))

        # --- did replaying our parsed actions reproduce the log's board? ---
        bad = _board_matches(state, raw.get("countries"))
        if bad:
            del conv.samples[before:]          # unverified actions are not training data
            raise ConversionFailure(Mismatch(
                conv.replay_id, e.turn, e.phase, e.player, e.card,
                "board mismatch after replay",
                f"{bad} {_board_diff(state, raw.get('countries'))}"))
        # --- and does the score it reached match the log's? ---
        # The one score change the log never narrates is the Military Operations comparison at
        # the end of a turn, which lands between the last action round and the next headline:
        # at turn 1 AR6 of replay 60 the log leaves the US on 5, the comparison adds 2, and the
        # log's next statement of the score -- the turn 2 headline, after the USSR scores 1 --
        # is 6. So the check is skipped on an entry whose play carried the game into a new
        # turn, and applies everywhere else.
        # Not on the first entry: it carries the setup placements, which the engine's own
        # initialisation stands in for, so anything scored there is scored on a board the log
        # has not yet corrected.
        want_score = _narrated_score(e) if prev_entry is not None else None
        # ...or into the end of the game. Final scoring runs when the last action round of turn
        # 10 finishes, and it moves the score by whatever the board is worth without the turn
        # number changing: at turn 10 AR7 of replay 154 the US plays the China Card for
        # Influence, the turn ends, final scoring hands the USSR enough to reach the cap, and
        # the engine stands at -20 against the -16 the log states. That -16 is the score before
        # final scoring, which the log never gets to record.
        # A win the log *does* record is still checked: it narrates the winning score itself,
        # so want_score is 20 and the comparison is against a number the log actually states.
        crossed_turn = (int(state.turn) != turn_before
                        or (ts.Engine.is_terminal(state)
                            and want_score is not None and abs(want_score) < 20))
        if want_score is None and prev_entry is not None and e.score is not None:
            # A score field that has *moved* since the entry before has been brought up to
            # date, and is worth asserting on even where nothing was narrated -- it is where
            # the Military Operations comparison at a turn end finally shows up, the one score
            # change the log never states in words. A field that merely repeats the previous
            # entry's is stale and says nothing.
            if prev_entry.score is not None and int(e.score) != int(prev_entry.score):
                want_score = int(e.score)
                crossed_turn = False
        forced = _KNOWN_SCORE.get(conv.replay_id, {}).get((e.turn, e.phase, e.player))
        if forced is not None:
            # A listed disagreement: the log's score stands and the run continues from it.
            state.victory_points = forced
            conv.scores_forced += 1
            want_score = None
        if want_score is not None and not crossed_turn:
            # The VP track runs from 20 to -20 and the game ends the moment it is reached, so
            # a score past either end is the replayer's arithmetic and not a position. At turn
            # 5 AR7 of replay 109 the USSR takes the Military Operations penalty for 2 and the
            # log reads "Score is USSR 21", where the game was already won at 20.
            want_score = max(-20, min(20, want_score))
            if int(state.victory_points) != want_score:
                del conv.samples[before:]
                raise ConversionFailure(Mismatch(
                    conv.replay_id, e.turn, e.phase, e.player, e.card,
                    "score mismatch after replay",
                    f"engine is at {int(state.victory_points)} VP, the log says {want_score}"))
        # A space race attempt either advanced the track or it did not, and the log says which.
        # Checked here rather than left to drift: the log prints a track only on success, so an
        # advance the reconstruction invented would otherwise never be contradicted.
        if e.mode == "space":
            logged_tracks = {side: level for side, level in (e.space or [])}
            for side, want_track in (("US", logged_tracks.get("US")),
                                     ("USSR", logged_tracks.get("USSR"))):
                got = int(state.us_space_track if side == "US" else state.ussr_space_track)
                if want_track is not None and got != int(want_track):
                    raise ConversionFailure(Mismatch(
                        conv.replay_id, e.turn, e.phase, e.player, e.card,
                        "space race track mismatch",
                        f"{side} is on box {got}, the log says {int(want_track)}"))
        conv.entries_converted += 1

        if e.score is not None and int(state.victory_points) != int(e.score):
            conv.vp_drift += 1

        conv.board_resyncs += _reconcile_board(state, raw.get("countries"))
        # Not the score, where this entry's own play carried the game past it. The Military
        # Operations comparison at a turn end is the one score change the log never narrates.
        _reconcile_scalars(state, e, take_score=not crossed_turn)
        for side in ("US", "USSR"):
            for card, at in list(pending[side].items()):
                if at <= index:
                    del pending[side][card]
        prev_crossed_turn = crossed_turn
        prev_raw = raw
        prev_entry = e
