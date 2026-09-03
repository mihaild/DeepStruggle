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

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import ts_engine as ts

from tools.lib.ts_replayer_parse import Entry, country_id, parse_entry


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


@dataclass
class Conversion:
    replay_id: int
    samples: List[Tuple[np.ndarray, np.ndarray, int, int]] = field(default_factory=list)
    entries_total: int = 0
    entries_converted: int = 0
    entries_guessed: int = 0
    decisions_emitted: int = 0
    board_resyncs: int = 0
    vp_drift: int = 0
    entries_board_mismatch: int = 0
    hand_misses: int = 0
    first_board_mismatch: Optional[Mismatch] = None
    first_vp_drift: Optional[Mismatch] = None
    # Set when conversion stopped: the entry that could not be reproduced. Entries after it
    # were never attempted, so a Conversion with a failure describes only a prefix of the game.
    failure: Optional[Mismatch] = None
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


def _pad_hand(state: ts.GameState, held: List[int], size: int, ops_cap: Optional[int],
              taken: set) -> List[int]:
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
    if len(held) >= size - 2:
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


def _reconcile_scalars(state: ts.GameState, entry: Entry) -> None:
    """Force VP and DEFCON to the logged values.

    Without this the engine scores from its own board and the two diverge fast: on replay 100
    the engine reached 20 VP -- game over -- at T2 AR1 while the log had the score at 1, which
    is why only a quarter of entries were reachable. VP is the one quantity the log disputes
    with itself (field vs ledger), so which source is used here is recorded per sample.
    """
    # Exactly, not max(): taking the larger of the two hides a track the reconstruction
    # advanced on its own, which is precisely the error worth catching.
    for side, level in (entry.space or []):
        if side == "US":
            state.us_space_track = int(level)
        else:
            state.ussr_space_track = int(level)
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

    if entry.score is not None:
        state.victory_points = int(entry.score)
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


def _drain(state: ts.GameState,
           expected: Optional[Dict[int, Tuple[int, int]]] = None,
           forced_roll: int = 0,
           war_rolls: Optional[List[int]] = None,
           coup_rolls: Optional[List[int]] = None,
           realign_rolls: Optional[List[Tuple[str, int]]] = None) -> None:
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
        kind, roller = _pending_roll(state) if (war_rolls or coup_rolls or realign_rolls) \
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
        elif expected:
            _force_roll(state, expected)
        # A space race roll is given outright by the log ("Die roll: 5 -- Failed!"), and the
        # chance node takes it directly, so there is nothing to search for. Leaving it to the
        # engine's own stream advanced tracks the humans never advanced -- and since the log
        # only prints a track on success, a wrong one was never corrected afterwards.
        ts.Engine.step(state, ts.MicroAction(
            ts.DecisionType.ROLL_DIE, roll if 1 <= roll <= 6 else 0,
            second if 1 <= second <= 6 else 0, 0))


def _force_roll(state: ts.GameState, expected: Dict[int, Tuple[int, int]],
                tries: int = 400) -> bool:
    """Seed the rng so the pending chance node resolves the way the log says it did."""
    base = int(state.rng_state)
    for k in range(tries):
        cand = (base + (k + 1) * _GOLDEN) % _UINT64
        probe = state.clone()
        probe.rng_state = cand
        try:
            ts.Engine.step(probe, ts.MicroAction(ts.DecisionType.ROLL_DIE, 0, 0, 0))
        except Exception:
            continue
        if all(int(probe.get_country(c).us_influence) == us
               and int(probe.get_country(c).ussr_influence) == ussr
               for c, (us, ussr) in expected.items()):
            state.rng_state = cand
            return True
    return False


def _acting(state: ts.GameState) -> ts.Player:
    ctx = state.ctx()
    return ctx.decision_player if ctx.decision_player != ts.Player.NONE else state.phasing_player


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
        extra: List[int] = []
        for _side, delta, cid, _u, _s in (e.influence or []):
            if cid not in e.targets:
                extra.extend([cid] * abs(int(delta)))
        return list(e.targets) + extra

    q: List[int] = []
    for _side, delta, cid, _u, _s in (e.ops_influence or []):
        q.extend([cid] * abs(int(delta)))
    if not q and e.targets:
        q = list(e.targets)
    return q


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
        if e is not None and e.defcon is not None:
            if int(probe.defcon) == int(e.defcon):
                score += 500
            # ...but only where the log kept playing. DEFCON 1 is thermonuclear war and the
            # phasing player loses, so it is never the branch to take on a tie -- and yet at
            # turn 9 AR7 of replay 104 it is exactly what happened: the USSR played Star Wars,
            # the US took How I Learned To Stop Worrying out of the discard pile and set DEFCON
            # to 1, and the USSR, as the phasing player, lost. The entry records defcon 1, so
            # penalising every branch that ends the game put the real one out of reach.
            if ts.Engine.is_terminal(probe) and int(e.defcon) != 1:
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


def _force_random_discard(state: ts.GameState, action: int, cards: List[int],
                          tries: int = 400) -> bool:
    """Seed the rng so that stepping `action` discards the card the log says was discarded.

    Five Year Plan discards at random from the USSR hand and, if the card is a US event, plays
    it. Which card comes out therefore decides what happens next, and the log records it -- but
    it is drawn inside the event, so there is no action to steer. At turn 3's headline of
    replay 119 the engine drew Marshall Plan where the log drew Duck and Cover, and seven US
    influence went into Western Europe that the human game never placed.
    """
    def discarded(probe: ts.GameState) -> bool:
        return all(probe.get_card_location(c) not in (ts.CardLocation.HAND_US,
                                                      ts.CardLocation.HAND_USSR)
                   for c in cards)

    base = int(state.rng_state)
    for k in range(tries + 1):
        probe = state.clone()
        if k:                       # k == 0 tries the seed the engine already has
            probe.rng_state = (base + k * _GOLDEN) % _UINT64
        try:
            ts.Engine.step_flat(probe, int(action))
        except Exception:
            continue
        if discarded(probe):
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
    plays_five_year_plan = cid_target == _FIVE_YEAR_PLAN or _FIVE_YEAR_PLAN in headline_ids.values()
    plays_grain_sales = cid_target == _GRAIN_SALES or _GRAIN_SALES in headline_ids.values()
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
               war_roll_queue, coup_roll_queue, realign_roll_queue)
        seed_settled = False
        if ts.Engine.is_terminal(state):
            break
        ctx = state.ctx()
        dt = ctx.decision_type
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
                # records its pick from the discard pile.
                for slot, want_c in enumerate(event_card_queue):
                    chosen = _find(state, legal, ts.DecisionType.SELECT_CARD,
                                   lambda ma, w=want_c: int(ma.primary_id) == w)
                    if chosen is None:
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

        elif dt == ts.DecisionType.SELECT_OP_MODE and (sections or e.mode in _OP_MODE):
            # An entry can hold more than one Ops section, and they are asked for in the order
            # the log prints them: CIA Created reveals the USSR hand, hands the US 1 Op to coup
            # with, and only then spends its own Op for the USSR. Taking the entry's single
            # mode for every such decision lost the second operation entirely.
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
            chosen = _choose_branch(state, legal, eq + pq, raw, e, returned_cid)
            informative = chosen is not None

        elif dt == ts.DecisionType.POINT_NODE and not pq and not eq and sections:
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

        elif dt == ts.DecisionType.POINT_NODE and (pq or eq):
            # Ask the queue that matches what the engine is doing: while a card is resolving,
            # these are the event's own placements, otherwise they are the Ops. Within a queue
            # take the first target actually offered rather than insisting on the head, since
            # an event may resolve some of its placements itself and never ask about them.
            in_event = int(ctx.resolving_card) != 0
            order = (eq, pq) if in_event else (pq, eq)
            for queue in order:
                for slot, want_c in enumerate(queue):
                    chosen = _find(state, legal, ts.DecisionType.POINT_NODE,
                                   lambda ma, w=want_c: int(ma.primary_id) == w)
                    if chosen is not None:
                        queue.pop(slot)
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
            offered = sorted(
                ts.MapData.get_country_info(int(p))["name"] if 0 <= int(p) < 84 else str(int(p))
                for p in (int(ts.ActionMask.decode_flat_action(state, int(a)).primary_id)
                          for a in legal))
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
            if _force_random_discard(state, int(chosen), random_discards):
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

    state = ts.GameState()
    ts.Engine.init_game(state, 12345)
    _drain(state)

    try:
        _convert_entries(state, raws, hands, conv)
    except ConversionFailure as failure:
        # Reported, not raised on: one unconvertible game should not stop a sweep of hundreds,
        # and the caller decides whether a partial game is usable. conv.failure being set means
        # the entries after it were never converted.
        conv.failure = failure.mismatch
        conv.mismatches.append(failure.mismatch)
    conv.decisions_emitted = len(conv.samples)
    return conv


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

    for index, raw in enumerate(raws):
        e = parse_entry(raw)
        conv.entries_total += 1

        if _is_turn_end_record(e, prev_entry):
            # Nothing to drive: the engine ends the turn itself once both players have taken
            # their last action round. Only the record's own bookkeeping is taken -- the
            # improved DEFCON and the effects that expired.
            conv.entries_converted += 1
            conv.board_resyncs += _reconcile_board(state, raw.get("countries"))
            _reconcile_scalars(state, e)
            prev_raw, prev_entry = raw, e
            continue

        if e.turn and e.turn != cur_turn:
            cur_turn = e.turn
            h = hands.get(str(e.turn)) or {}
            turn_hands = {
                "US": [c for c in (card_id(n) for n in h.get("us", [])) if c],
                "USSR": [c for c in (card_id(n) for n in h.get("ussr", [])) if c],
            }
            # The log lists only the cards a player used, so a game that stops mid-turn leaves
            # hands far too small to have been the ones played from. Top them up to the size
            # the rules deal, capping Ops below anything the log records that side revealing.
            # A new turn restores each side's space race attempt. The engine resets this when
            # it advances the turn itself, which forcing turn numbers from the log bypasses, so
            # a stale count made a legitimate attempt illegal: at turn 4 AR4 of replay 112 the
            # USSR races with Duck and Cover and the engine offered only Ops modes.
            state.set_space_turns_used(ts.Player.US, 0)
            state.set_space_turns_used(ts.Player.USSR, 0)

            size = 8 if int(e.turn) <= 3 else 9
            claimed = set(turn_hands["US"]) | set(turn_hands["USSR"])
            for side in ("US", "USSR"):
                padded = _pad_hand(state, turn_hands[side], size,
                                   _reveal_ops_cap(raws, e.turn, side), claimed)
                claimed |= set(padded)
                turn_hands[side] = padded
            played = {"US": set(), "USSR": set()}
            pending = {side: _mid_turn_acquisitions(raws, int(e.turn), side, turn_hands[side])
                       for side in ("US", "USSR")}

        # --- rebuild the position this entry was decided from ---
        if prev_raw is not None:
            _reconcile_board(state, prev_raw.get("countries"))
        if prev_entry is not None:
            _reconcile_scalars(state, prev_entry)
        if state.current_phase == ts.Phase.GAME_OVER:
            state.current_phase = ts.Phase.ACTION_ROUND
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

        before = len(conv.samples)
        if prev_entry is not None:
            # Not on the first entry: that one carries the setup placements, and the engine's
            # own initialisation is the position they belong to.
            _reconcile_turn(state, e, conv.replay_id)
        _drive_entry(state, e, conv, raw)

        # --- did replaying our parsed actions reproduce the log's board? ---
        bad = _board_matches(state, raw.get("countries"))
        if bad:
            del conv.samples[before:]          # unverified actions are not training data
            raise ConversionFailure(Mismatch(
                conv.replay_id, e.turn, e.phase, e.player, e.card,
                "board mismatch after replay",
                f"{bad} {_board_diff(state, raw.get('countries'))}"))
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
        _reconcile_scalars(state, e)
        for side in ("US", "USSR"):
            for card, at in list(pending[side].items()):
                if at <= index:
                    del pending[side][card]
        prev_raw = raw
        prev_entry = e
