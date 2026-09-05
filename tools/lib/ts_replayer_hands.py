"""Reconstruct both players' hands across a whole game, as a constraint problem.

A ts-replayer log records what became *visible*: every card played, discarded or revealed, with
the turn and the side it came from. What it never records is what a player held and did not
spend. The hands it does not state are not free, though -- they are pinned by the rules on
every side:

  * a hand holds exactly 8 cards in the early war and 9 from turn 4, the China Card aside;
  * what a player does not play, they carry into the next turn;
  * a card that has been played is in the discard pile until a reshuffle, and the log prints
    "*RESHUFFLE*" where one happens;
  * a card of a later era is not in the deck yet;
  * a scoring card may not be held past the end of a turn -- holding one loses the game;
  * a hand the log calls empty is empty, and a player who skipped an action round had nothing
    they were allowed to play;
  * Missile Envy takes the highest Ops card in a hand, so nothing higher was in it.

Those constraints leave a choice, so the choice is made by preference rather than by guesswork
spread across half a dozen heuristics: a player carries what is least worth playing, which is
the opponent's small events, and does not sit on their own good cards.

Solved with z3 (MIT). Where z3 is not installed the converter falls back to the older
turn-by-turn borrowing, which is why this module keeps its own entry point.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Set, Tuple

import ts_engine as ts

from tools.lib.ts_replayer_parse import Entry, RE_PASSED_ROUND, parse_entry

try:                                            # pragma: no cover - environment dependent
    import z3
    HAVE_Z3 = True
except ImportError:                             # pragma: no cover - environment dependent
    HAVE_Z3 = False

THE_CHINA_CARD = 6
_ASK_NOT = 77
_CAMBRIDGE_FIVE = 104
_CIA_CREATED = 26
_GRAIN_SALES = 67
_VOICE_OF_AMERICA = 74
_COLONIAL_REAR_GUARDS = 63
_MISSILE_ENVY = 49
_RED_SCARE_PURGE = 31
_ERA_ENTERS = {"0": 1, "1": 4, "2": 8}          # early war, mid war, late war


def _build_country_index() -> Dict[str, int]:
    """The log names countries by slug -- "west_germany" -- and the engine by id."""
    out: Dict[str, int] = {}
    for cid in range(84):
        name = str(ts.MapData.get_country_info(cid)["name"])
        out[name.lower().replace(" ", "_").replace("/", "_").replace("'", "")] = cid
    out["uk"] = out.get("united_kingdom", 1)
    out["dominican_republic"] = out.get("dominican_rep", 73)
    return out


_COUNTRY_BY_SLUG = _build_country_index()


def hand_size(turn: int) -> int:
    """What a turn deals up to: eight in the early war, nine from the mid war on."""
    return 8 if turn <= 3 else 9


class GameFacts:
    """Everything the log states about where cards were, gathered once per game."""

    def __init__(self, raws: List[Dict], hands: Dict, card_id) -> None:
        self.raws = raws
        self.card_id = card_id
        self.last_turn = max((int(r["num"]) for r in raws
                              if str(r.get("num", "")).isdigit()), default=0)
        self.sides = ("US", "USSR")
        # Cards each side is shown spending in a turn: played as a card of their own, played
        # through another card, or discarded out of their hand.
        self.spent: Dict[Tuple[int, str], Set[int]] = {}
        # Cards the log shows changing hands or arriving mid-turn, which need not have been in
        # the hand the turn was dealt.
        self.arrived: Dict[Tuple[int, str], Set[int]] = {}
        # Cards taken *out* of a hand mid-turn by the opponent's event.
        self.taken: Dict[Tuple[int, str], Set[int]] = {}
        self.reshuffled_before: Set[int] = set()
        self.empty_hand: Set[Tuple[int, str]] = set()
        self.skipped: Set[Tuple[int, str]] = set()
        self.trapped: Set[Tuple[int, str]] = set()
        self.red_scare: Set[Tuple[int, str]] = set()
        self.envy_took: Dict[Tuple[int, str], int] = {}
        self.listed: Dict[Tuple[int, str], Set[int]] = {}
        # The most VP a single scoring card took off a side in a turn, and the most any region
        # would take off the USSR if it were scored as the turn opened.
        self.lost_to_scoring: Dict[Tuple[int, str], int] = {}
        self.region_threat: Dict[int, int] = {}
        # How many cards a side drew after the deal, which is how many of the ones they spend
        # need not have been in the hand the turn opened with.
        self.drew_mid_turn: Dict[Tuple[int, str], int] = {}
        # ...and the action round they were drawn in: nothing played before it can be one.
        self.drew_at: Dict[Tuple[int, str], int] = {}
        # Where the log shows the opponent's scoring cards being looked for and none found,
        # against the action round it happened in.
        self.no_scoring_at: Dict[Tuple[int, str], int] = {}
        # The action round a card was played at, where the log places it in one.
        self.played_at: Dict[Tuple[int, str, int], int] = {}
        self._is_last = False
        for turn in range(1, self.last_turn + 1):
            for side in self.sides:
                self.spent[(turn, side)] = set()
                self.arrived[(turn, side)] = set()
                self.taken[(turn, side)] = set()
                lists = (hands or {}).get(str(turn)) or {}
                self.listed[(turn, side)] = {
                    c for c in (card_id(nm) for nm in (lists.get(side.lower()) or [])) if c}
        self._read(raws)
        self._read_board(raws)

    def _read(self, raws: List[Dict]) -> None:
        for index, raw in enumerate(raws):
            e = parse_entry(raw)
            self._is_last = index == len(raws) - 1
            turn = e.turn
            if not turn or turn > self.last_turn:
                continue
            text = e.text or ""
            if "*RESHUFFLE*" in text:
                self.reshuffled_before.add(turn)
            for side, nm in (e.headlines or {}).items():
                cid = self.card_id(nm)
                if cid and cid != THE_CHINA_CARD and side in self.sides:
                    self.spent[(turn, side)].add(cid)
            if e.card and " & " not in e.card and e.player in self.sides:
                cid = self.card_id(e.card)
                if cid and cid != THE_CHINA_CARD:
                    self.spent[(turn, e.player)].add(cid)
            if e.played_card and e.player in self.sides:
                cid = self.card_id(e.played_card)
                if cid and cid != THE_CHINA_CARD:
                    self.spent[(turn, e.player)].add(cid)
            peeked = _peeked_from_the_pile(e, self.card_id)
            for side, nm in (e.discards or []):
                cid = self.card_id(nm)
                if cid and side in self.sides and cid not in peeked:
                    self.spent[(turn, side)].add(cid)
            at = _round_number(e)
            if at is not None and e.player in self.sides:
                here = set()
                if e.card and " & " not in e.card:
                    cid = self.card_id(e.card)
                    if cid:
                        here.add(cid)
                if e.played_card:
                    cid = self.card_id(e.played_card)
                    if cid:
                        here.add(cid)
                for cid in here:
                    self.played_at.setdefault((turn, e.player, cid), at)
            card = self.card_id(e.card) if e.card and " & " not in e.card else None
            if card and bool(ts.CardData.get_card_info(card)["is_scoring"]):
                for gain_side, amount, _total_side, _total in (e.vp_gains or []):
                    for side in self.sides:
                        if gain_side != side:
                            key = (turn, side)
                            self.lost_to_scoring[key] = max(
                                self.lost_to_scoring.get(key, 0), int(amount))
            for side in self.sides:
                if f"{side} has no cards to discard" in text:
                    self.empty_hand.add((turn, side))
                if f"{side} has no cards in hand to reveal" in text:
                    self.empty_hand.add((turn, side))
            self._read_exchanges(e, turn)
            self._read_constraints(e, turn)
            self._read_scoring_reveals(e, turn)

    def _read_exchanges(self, e: Entry, turn: int) -> None:
        """Cards that change hands, or arrive from somewhere other than the turn's deal."""
        text = e.text or ""
        for marker, mover in (("Event: Missile Envy", "envy"),
                              ("Event: Grain Sales To Soviets", "grain"),
                              ("Event: SALT Negotiations*", "salt")):
            at = text.find(marker)
            if at < 0:
                continue
            for raw_line in text[at:].split("\n")[1:]:
                line = raw_line.strip()
                if line.startswith("Event: ") or not line:
                    break
                if " reveals " not in line:
                    continue
                side = line.split(" ", 1)[0]
                name = line.split(" reveals ", 1)[1].replace(" from hand", "").strip(" .")
                cid = self.card_id(name)
                if not cid or side not in self.sides:
                    break
                if mover == "salt":
                    # Reclaimed from the discard pile, so it was not in the dealt hand.
                    self.arrived[(turn, side)].add(cid)
                else:
                    other = "USSR" if side == "US" else "US"
                    self.taken[(turn, side)].add(cid)
                    self.arrived[(turn, other)].add(cid)
                    if mover == "envy":
                        self.envy_took[(turn, side)] = cid
                        # Missile Envy itself changes hands: whoever played it hands it over,
                        # and the receiver may play it again in the same turn.
                        self.arrived[(turn, side)].add(_MISSILE_ENVY)
                break
        # "Ask Not What Your Country Can Do For You" draws exactly as many cards as it
        # discards, and those arrived after the deal. Which of the turn's cards they are the
        # log does not say, so the count is what is known: at most this many of the cards the
        # side spends need not have been in the hand the turn opened with.
        if any(self.card_id(nm) == _ASK_NOT for nm in (e.events or [])):
            side = e.player if e.player in self.sides else None
            drawn = 0
            for s, nm in (e.discards or []):
                if s in self.sides:
                    side = s
                    drawn += 1
            if side and drawn:
                self.drew_mid_turn[(turn, side)] = (
                    self.drew_mid_turn.get((turn, side), 0) + drawn)
                at = _round_number(e)
                self.drew_at[(turn, side)] = min(
                    self.drew_at.get((turn, side), 99), at if at is not None else 0)

    def _read_scoring_reveals(self, e: Entry, turn: int) -> None:
        """Moments the log shows a hand searched for scoring cards and none found.

        The Cambridge Five reads the opponent's scoring cards and lets its player place
        Influence in a region one of them names. Where it resolves with nothing -- no region,
        no placement, or the log's "has no cards to reveal" -- that hand held no scoring card
        then, whatever it played later in the turn. At turn 5 AR2 of ts-replayer game 54 the US
        plays The Cambridge Five for its Operations and the event finds nothing, while the
        model had them holding Southeast Asia Scoring, so the USSR was asked where to place.
        """
        card = self.card_id(e.card) if e.card and " & " not in e.card else None
        headlined = {self.card_id(nm) for nm in (e.headlines or {}).values()}
        if _CAMBRIDGE_FIVE not in ({card} | headlined):
            return
        text = e.text or ""
        looked_at = "USSR" if e.player == "USSR" and card == _CAMBRIDGE_FIVE else "US"
        # Whoever plays it reads the *other* hand.
        looked_at = "US" if e.player == "USSR" else "USSR"
        if e.player not in self.sides:
            looked_at = "US"
        found = ("reveals" in text and "no cards to reveal" not in text) or bool(e.influence)
        if not found:
            at = _round_number(e)
            self.no_scoring_at[(turn, looked_at)] = min(
                self.no_scoring_at.get((turn, looked_at), 99), at if at is not None else 0)

    def _read_board(self, raws: List[Dict]) -> None:
        """What the board threatens as each turn opens, read off the turn's first entry."""
        for raw in raws:
            turn = parse_entry(raw).turn
            if not turn or turn in self.region_threat or turn > self.last_turn:
                continue
            self.region_threat[turn] = _region_threat_to_ussr(raw.get("countries"))

    def _read_constraints(self, e: Entry, turn: int) -> None:
        text = e.text or ""
        for side in self.sides:
            trap = "Quagmire*" if side == "US" else "Bear Trap*"
            if any(nm.strip() == trap for nm in (e.in_play or [])):
                self.trapped.add((turn, side))
            other = "USSR" if side == "US" else "US"
            if self.card_id((e.headlines or {}).get(other) or "") == _RED_SCARE_PURGE:
                self.red_scare.add((turn, side))
            if e.player == other and e.card and self.card_id(e.card) == _RED_SCARE_PURGE:
                self.red_scare.add((turn, side))
        if e.player in self.sides and e.trap_rolls:
            self.trapped.add((turn, e.player))
        # Only a round they had to take. The eighth is granted by North Sea Oil or a Space
        # Station to the one player who earned it and is theirs to decline, so skipping it says
        # nothing about the hand: at turn 9 of ts-replayer game 16 the USSR plays their eight
        # cards and lets the ninth round go, still holding one.
        if e.player in self.sides and _is_skipped_round(e) and _round_number(e) not in (None, 8):
            self.skipped.add((turn, e.player))
        # A bare header inside another entry, at the end of the file, is the recording
        # stopping rather than a player passing.
        if self._is_last:
            return
        for m in RE_PASSED_ROUND.finditer(text):
            if (m.group(2) in self.sides and int(m.group(1)) <= self.last_turn
                    and int(m.group(3)) != 8):
                self.skipped.add((int(m.group(1)), m.group(2)))


def _region_threat_to_ussr(countries: Optional[Dict]) -> int:
    """The most any region would pay the US, scored on this board.

    Five Year Plan discards at random from the USSR hand, so the USSR carries it while a region
    stands to cost them -- the card is worth having in hand where the alternative is worse.
    """
    if not countries:
        return 0
    state = ts.GameState()
    ts.Engine.init_game(state, 1)
    for cid in range(84):
        state.set_country(cid, 0, 0)
    for name, info in countries.items():
        cid = _COUNTRY_BY_SLUG.get(name)
        if cid is None:
            continue
        try:
            state.set_country(cid, int(info.get("inflUS", 0)), int(info.get("inflUSSR", 0)))
        except (TypeError, ValueError):
            continue
    worst = 0
    for region in (ts.Region.EUROPE, ts.Region.ASIA, ts.Region.MIDDLE_EAST, ts.Region.AFRICA,
                   ts.Region.CENTRAL_AMERICA, ts.Region.SOUTH_AMERICA):
        try:
            level = ts.Scoring.evaluate_region(state, region)
        except Exception:
            continue
        worst = max(worst, int(level.net_delta))
    return worst


def _peeked_from_the_pile(e: Entry, card_id) -> Set[int]:
    """Cards an entry names as discarded that were never in a hand.

    Our Man in Tehran turns the top five cards of the discard pile face up and lets the US play
    one; the log writes the other four as "US discards X", which they never held. At turn 5 AR7
    of ts-replayer game 100 that is four of the twelve cards the US would otherwise have had to
    be holding, out of a hand of nine.
    """
    text = e.text or ""
    at = text.find("Event: Our Man in Tehran*")
    if at < 0:
        return set()
    out: Set[int] = set()
    for raw_line in text[at:].split("\n")[1:]:
        line = raw_line.strip()
        if not line or line.startswith("Event: "):
            break
        if " discards " in line:
            cid = card_id(line.split(" discards ", 1)[1].strip(" ."))
            if cid:
                out.add(cid)
    return out


def _round_number(e: Entry) -> Optional[int]:
    """Which action round an entry belongs to, or None for a headline."""
    phase = e.phase or ""
    return int(phase[2:]) if phase.startswith("AR") and phase[2:].isdigit() else None


def _is_skipped_round(e: Entry) -> bool:
    """An action round with a header and nothing under it: the player took no action."""
    return bool((e.phase or "").startswith("AR")) and not (
        e.influence or e.sections or e.targets or e.events or e.headlines or e.space
        or e.discards or e.revealed or e.mode or e.card)


def _enters_deck(cid: int) -> int:
    """The first turn a card can be in the deck at all."""
    return _ERA_ENTERS[str(ts.CardData.get_card_info(cid)["era"])]



# Cards a player is glad to be rid of, and cards they would rather sit on. The base reading is
# that a player carries what is least worth playing -- the opponent's events, small ones first
# -- and spends their own. The rest are moments where that is not what a player would do.
_US_WOULD_HAVE_PLAYED = (_CIA_CREATED, _VOICE_OF_AMERICA, _GRAIN_SALES, _COLONIAL_REAR_GUARDS)
_ASK_NOT = 77
_EAST_EUROPEAN_UNREST = 15
_FIVE_YEAR_PLAN = 5
_LONE_GUNMAN = 62


def _hold_cost(side: str, cid: int, turn: int, facts: "GameFacts") -> int:
    """How unlikely this card is to be the one carried. Lower is likelier.

    The base is what a hand is for: you spend your own events and the neutrals, and you are left
    holding the opponent's, the small ones first because the big ones are worth playing for
    their Operations alone.
    """
    info = ts.CardData.get_card_info(cid)
    card_side = str(info["side"])
    ops = int(info["ops"])
    cost = (0 if card_side not in (side, "NONE") else 40) + ops

    # A reshuffle puts the discard pile back in the deck, so a card held across one comes round
    # again -- and these are the ones their holder least wants to see again.
    if turn + 1 in facts.reshuffled_before:
        if side == "US" and cid in _US_WOULD_HAVE_PLAYED:
            cost += 60
        if side == "USSR" and cid == _LONE_GUNMAN:
            cost += 60

    # East European Unrest is worth least to the USSR by the mid war, and turn 7 is where they
    # would have spent it rather than carried it.
    if side == "USSR" and cid == _EAST_EUROPEAN_UNREST and turn == 7:
        cost += 60

    # The Voice of America is a card the US holds for the late war, where its removals bite.
    if side == "US" and cid == _VOICE_OF_AMERICA and turn == 9:
        cost -= 30

    if side == "USSR" and cid == _FIVE_YEAR_PLAN:
        # Five Year Plan discards at random from the USSR hand, so they carry it while the board
        # threatens them -- and having just been hit for three or more by a scoring card, they
        # would not have been holding the card that does it to them.
        if facts.lost_to_scoring.get((turn, "USSR"), 0) >= 3:
            cost += 60
        elif facts.region_threat.get(turn, 0) >= 3:
            cost -= 30

    # "Ask Not What Your Country Can Do For You" throws away exactly the small cards, so a turn
    # that played it did not end holding one.
    if _ASK_NOT in facts.spent.get((turn, side), set()):
        if (card_side == "NONE" and ops <= 1) or (card_side not in (side, "NONE") and ops <= 2):
            cost += 60

    return max(cost, 1)


def _name(cid: int) -> str:
    return str(ts.CardData.get_card_info(cid)["name"])


def _print_core(hard: List, tags: List[str], ctx=None) -> None:
    """Say which rules cannot all hold, for a log the model rejects."""
    solver = z3.Solver(ctx=ctx)
    solver.set(unsat_core=True)
    for expr, tag in zip(hard, tags):
        solver.assert_and_track(expr, z3.Bool(f"{tag} #{len(solver.assertions())}", ctx))
    if solver.check() == z3.unsat:
        print("  these cannot all hold:")
        for c in solver.unsat_core():
            print("   ", str(c).strip("|").rsplit(" #", 1)[0])


def solve_hands(raws: List[Dict], hands: Dict, card_id,
                rlimit: int = 4_000_000,
                explain: bool = False) -> Optional[Dict[int, Dict[str, List[int]]]]:
    """The cards each side holds as each turn opens, or None where the log allows no hand.

    One boolean per card, turn and side -- "this card is in that hand as the turn opens" -- and
    the rules as constraints over them. Satisfying them is quick; choosing *well* among the
    hands that satisfy them is a weighted MaxSAT problem and is not, so the hard model is
    solved first and the preferences are given a budget on top of it. A hand that merely obeys
    the rules is worth having; an optimal one is not worth minutes.
    """
    if not HAVE_Z3 or not raws:
        return None
    facts = GameFacts(raws, hands, card_id)
    if facts.last_turn < 1:
        return None
    # A context of its own for every game. The same log has to give the same hands every time
    # -- a reconstruction that shifted between runs would put a different game in the training
    # data each time it was built -- and z3 counts its resource limit per context, so sharing
    # one would hand each successive game a smaller budget than the last and a worse hand with
    # it.

    turns = list(range(1, facts.last_turn + 1))
    cards = [c for c in range(1, 111) if c != THE_CHINA_CARD]
    ctx = z3.Context()
    held = {(c, t, s): z3.Bool(f"h_{c}_{t}_{s}", ctx)
            for c in cards for t in turns for s in facts.sides}
    hard: List = []
    tags: List[str] = []

    def keep(expr, tag: str) -> None:
        hard.append(expr)
        tags.append(tag)
    # Variables the rules settle on their own. Preferences are only worth stating about the
    # rest, and stating them about all 2,000 is what made the search hopeless.
    settled: Set[Tuple[int, int, str]] = set()

    def forbid(c: int, t: int, s: str, why: str) -> None:
        keep(z3.Not(held[(c, t, s)]), f"t{t} {s} not {_name(c)}: {why}")
        settled.add((c, t, s))

    def require(c: int, t: int, s: str, why: str) -> None:
        keep(held[(c, t, s)], f"t{t} {s} holds {_name(c)}: {why}")
        settled.add((c, t, s))

    for t in turns:
        for s in facts.sides:
            # A turn deals up to the hand size. On the last turn of a game it can deal fewer:
            # the deck holds 110 cards and the last of them can simply run out, which is how a
            # player comes to skip an action round they were entitled to -- at turn 10 AR7 of
            # ts-replayer game 72 the US has nothing left to play.
            keep(z3.PbLe([(held[(c, t, s)], 1) for c in cards], hand_size(t)),
                 f"t{t} {s} holds no more than {hand_size(t)} cards")
            if t < facts.last_turn:
                keep(z3.PbGe([(held[(c, t, s)], 1) for c in cards], hand_size(t)),
                     f"t{t} {s} holds {hand_size(t)} cards")
        for c in cards:
            keep(z3.Or(z3.Not(held[(c, t, "US")]), z3.Not(held[(c, t, "USSR")])),
                 f"t{t} {_name(c)} is in one hand at a time")
            if _enters_deck(c) > t:
                for s in facts.sides:
                    forbid(c, t, s, "its era has not started")

    for t in turns:
        for s in facts.sides:
            spent = facts.spent[(t, s)]
            arrived = facts.arrived[(t, s)]
            other = "USSR" if s == "US" else "US"
            empty = ((t, s) in facts.empty_hand
                     or ((t, s) in facts.skipped and (t, s) not in facts.trapped))
            ceiling = None
            if (t, s) in facts.skipped and (t, s) in facts.trapped:
                ceiling = 2 if (t, s) in facts.red_scare else 1
            envy = facts.envy_took.get((t, s))
            envy_cap = int(ts.CardData.get_card_info(envy)["ops"]) if envy else None
            drew = facts.drew_mid_turn.get((t, s), 0)
            late: List[int] = []
            # A hand searched for scoring cards and found wanting held none at that point, so
            # nothing they play after it was there when the turn opened.
            searched_at = facts.no_scoring_at.get((t, s))
            for c in cards:
                info = ts.CardData.get_card_info(c)
                scoring = bool(info["is_scoring"])
                drawn_after = (drew and facts.played_at.get((t, s, c), -1)
                               >= facts.drew_at.get((t, s), 99))
                if (envy_cap is not None and drew and int(info["ops"]) > envy_cap
                        and not scoring):
                    # Missile Envy took the highest Ops card there was, so anything bigger
                    # reached this hand after it had read it -- which is possible only because
                    # something drew cards mid-turn. At turn 7's headline of replay 96 "Ask
                    # Not" discards six and draws six before Missile Envy reads the hand at
                    # all, and ABM Treaty at 4 Ops cannot have been sitting in it.
                    forbid(c, t, s, "Missile Envy would have taken it instead")
                    continue
                if c in spent and c not in arrived:
                    if drawn_after:
                        # Some of what they spent arrived after the deal, and the log says how
                        # many but not which -- only that it was not anything they had already
                        # played by then. Let the rest be late arrivals, and hold the count to
                        # what was drawn.
                        late.append(c)
                        continue
                    require(c, t, s, "the log shows them spending it")
                    continue
                if c in facts.taken[(t, s)]:
                    require(c, t, s, "the opponent takes it out of their hand")
                    continue
                if c in facts.spent[(t, other)] and c not in facts.arrived[(t, other)]:
                    forbid(c, t, s, "the other side spends it")
                    continue
                if scoring and c not in spent:
                    forbid(c, t, s, "a scoring card cannot be carried past a turn")
                    continue
                if (scoring and searched_at is not None
                        and facts.played_at.get((t, s, c), 99) > searched_at):
                    forbid(c, t, s, "the opponent looked for scoring cards and found none")
                    continue
                if empty and c not in spent:
                    forbid(c, t, s, "the log says the hand was empty")
                    continue
                if (ceiling is not None and c not in spent
                        and int(info["ops"]) > ceiling):
                    forbid(c, t, s, f"a trap would have taken it (above {ceiling} Ops)")
                    continue
                if envy_cap is not None and int(info["ops"]) > envy_cap:
                    forbid(c, t, s, "Missile Envy would have taken it instead")
                    continue
                if (t + 1 <= facts.last_turn and c not in spent
                        and c not in facts.taken[(t, s)]):
                    # What they do not spend, and nobody takes off them, they carry.
                    keep(z3.Implies(held[(c, t, s)], held[(c, t + 1, s)]),
                         f"t{t} {s} carries {_name(c)} if unspent")

    # Whatever a side drew after the deal, the rest of what they spent was in the hand the turn
    # opened with -- the log gives the count even where it does not name the cards.
    for t in turns:
        for s in facts.sides:
            drew = facts.drew_mid_turn.get((t, s), 0)
            if not drew:
                continue
            late = [c for c in facts.spent[(t, s)]
                    if c not in facts.arrived[(t, s)] and c != THE_CHINA_CARD
                    and facts.played_at.get((t, s, c), -1) >= facts.drew_at.get((t, s), 99)]
            if len(late) > drew:
                keep(z3.PbGe([(held[(c, t, s)], 1) for c in late], len(late) - drew),
                     f"t{t} {s} drew {drew} cards after the deal and held the rest")

    # A card that has been spent is in the discard pile, and only a reshuffle puts it back --
    # the log prints those. A card reclaimed out of the pile starts over.
    for c in cards:
        last_spent = 0
        for t in turns:
            if c in facts.spent[(t, "US")] or c in facts.spent[(t, "USSR")]:
                last_spent = t
                if not any(c in facts.arrived[(t, s)] for s in facts.sides):
                    continue
            if any(c in facts.arrived[(t, s)] for s in facts.sides):
                last_spent = 0
                continue
            if last_spent and not any(last_spent < r <= t for r in facts.reshuffled_before):
                for s in facts.sides:
                    if c not in facts.spent[(t, s)]:
                        forbid(c, t, s, f"spent at turn {last_spent}, no reshuffle since")

    solver = z3.Solver(ctx=ctx)
    solver.add(hard)
    if solver.check() != z3.sat:
        if explain:
            _print_core(hard, tags, ctx)
        return None
    model = solver.model()

    free = [(c, t, s) for c in cards for t in turns for s in facts.sides
            if (c, t, s) not in settled]
    if free:
        opt = z3.Optimize(ctx=ctx)
        # A deterministic budget, not a clock: the same log has to give the same hands whatever
        # else the machine is doing, and a wall-clock timeout does not. Optimising is
        # worth-having rather than must-have -- the hard model is already solved -- so this
        # simply stops the search where it stands.
        opt.set("rlimit", rlimit)
        opt.add(hard)
        for c, t, s in free:
            opt.add_soft(z3.Not(held[(c, t, s)]), weight=_hold_cost(s, c, t, facts))
        # ...and the last turn deals a full hand wherever the deck can still fill one.
        for s in facts.sides:
            opt.add_soft(
                z3.PbGe([(held[(c, facts.last_turn, s)], 1) for c in cards],
                        hand_size(facts.last_turn)), weight=500)
        if opt.check() == z3.sat:
            model = opt.model()

    return {t: {s: [c for c in cards if z3.is_true(model.eval(held[(c, t, s)]))]
                for s in facts.sides}
            for t in turns}
