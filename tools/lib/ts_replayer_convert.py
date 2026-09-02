"""Drive the engine through a logged human game, emitting (observation, action) pairs.

The engine deals from a seed, so a human game cannot be reproduced by seeding: hands are
forced to the logged ones at each turn boundary, and the board is reconciled to the log after
every entry. Die outcomes will differ from the human game -- what behaviour cloning needs is a
faithful *observation* at each decision, not a bit-identical simulation, and reconciling keeps
the observation faithful even when a coup roll goes the other way.

Only decisions the log actually determines are emitted. Where the engine asks something the log
does not record (an event's internal branch, say), a legal action is taken to keep the game
moving but nothing is emitted, and the entry is marked `guessed` so it can be excluded.

Every mismatch is reported with replay id and turn/action round rather than counted, since a
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
    mismatches: List[Mismatch] = field(default_factory=list)


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
    for side, level in (entry.space or []):
        if side == "US":
            state.us_space_track = max(int(state.us_space_track), int(level))
        else:
            state.ussr_space_track = max(int(state.ussr_space_track), int(level))
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


def _reconcile_turn(state: ts.GameState, entry: Entry) -> None:
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
    ctx = state.ctx()
    if ctx.decision_type == ts.DecisionType.SELECT_CARD and int(ctx.resolving_card) == 0:
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


def _drain(state: ts.GameState,
           expected: Optional[Dict[int, Tuple[int, int]]] = None) -> None:
    """Resolve chance nodes, optionally steering them to the outcome the log recorded.

    Not every die belongs to a decision. A war with a fixed target -- Korean War, Arab-Israeli
    War -- rolls inside the event with nothing to choose, so there is no action to force the
    outcome through, and the roll came out however the engine's stream said: at turn 4 AR4 of
    replay 101 the USSR won the Korean War in the log and lost it in the reconstruction.
    """
    while (not ts.Engine.is_terminal(state)
           and state.ctx().decision_player == ts.Player.NONE
           and state.ctx().decision_type == ts.DecisionType.ROLL_DIE):
        if expected:
            _force_roll(state, expected)
        ts.Engine.step(state, ts.MicroAction(ts.DecisionType.ROLL_DIE, 0, 0, 0))


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


def event_queue(e: Entry) -> List[int]:
    """Targets the card's event asks the player to choose, in log order.

    Kept apart from the Ops queue on purpose. Neither half can be dropped -- at turn 1 AR5 of
    replay 100 the event's Vietnam influence is automatic and never comes back as a decision,
    while at turn 1 AR3 the USSR plays Marshall Plan for Ops and the US still chooses all seven
    event placements -- but merging them let turn 3 AR4 spend Nasser's Op on an event target.
    """
    ops = list(e.ops_influence or [])
    q: List[int] = list(e.war_targets or [])
    for rec in (e.influence or []):
        if rec in ops:
            ops.remove(rec)
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
                   raw: Optional[Dict], e: Optional[Entry] = None) -> Optional[int]:
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
        # Some branches are a numeric setting rather than a target: How I Learned To Stop
        # Worrying picks the new DEFCON, and with no targets to tell the options apart the
        # first legal one -- DEFCON 1 -- ended the game at turn 4's headline of replay 101.
        if e is not None and e.defcon is not None:
            if int(probe.defcon) == int(e.defcon):
                score += 500
            if ts.Engine.is_terminal(probe):
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
    picked_mode = False
    guessed = False
    discard_queue = [c for c in (card_id(nm) for _side, nm in (e.discards or [])) if c]
    reveal_queue = [c for c in (card_id(nm) for nm in (e.revealed or [])) if c]
    seeded_peek = False
    seeded_reveal = False
    # A war with a fixed target resolves in a chance node, so its outcome has to be steered
    # there rather than at a decision. Only the war's own countries are constrained: the rest
    # of the entry has not happened yet at that point.
    _want = expected_counts(e, raw)
    war_outcome = {c: _want[c] for c in e.war_targets if c in _want}
    # Only where the entry really holds several. With one section the existing queue already
    # describes it, and re-deriving it per decision only risks disagreeing with itself.
    sections = list(e.sections) if len(e.sections or []) > 1 else []
    seed_settled = False
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
        _drain(state, None if seed_settled else war_outcome)
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
        elif (int(ctx.resolving_card) == _GRAIN_SALES and not seeded_reveal
                and e.played_card is not None):
            revealed = card_id(e.played_card)
            if revealed:
                _seed_revealed_card(state, revealed)
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
                    conv.mismatches.append(Mismatch(
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
                conv.mismatches.append(Mismatch(
                    conv.replay_id, e.turn, e.phase, e.player, e.card,
                    "card not selectable",
                    f"card #{cid_target} not among {len(legal)} legal actions"))
                return False

        elif dt == ts.DecisionType.SELECT_PLAY_MODE and not picked_mode:
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
            elif cid_target and not opponent_card and e.event_first and e.events:
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
                chosen, picked_mode, informative = want, True, True
            else:
                conv.mismatches.append(Mismatch(
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
            if section is not None:
                pq[:] = section_queue(section)
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
            chosen = _choose_branch(state, legal, eq + pq, raw, e)
            informative = chosen is not None

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
                        informative = True
                        break
                if chosen is not None:
                    break
            if chosen is None:
                # An event can ask for more than the board can give: Suez Crisis removes four
                # US Influence across France, the UK and Israel, and at turn 2 AR6 of replay
                # 105 only three were there to remove. The engine offers the pass; taking it
                # finishes the event and leaves the entry's own Ops still to spend, where
                # abandoning the entry lost them.
                chosen = _find_confirm_done(state, legal)
            if chosen is None:
                names = ", ".join(ts.MapData.get_country_info(c)["name"] for c in (pq + eq))
                conv.mismatches.append(Mismatch(
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
            chosen = int(legal[0])
            guessed = True

        # Anything the engine settles with a die it rolls itself: coups, realignments, and war
        # events, whose target is chosen the same way but never carried an Ops mode.
        if dt == ts.DecisionType.POINT_NODE and chosen is not None:
            target = int(ts.ActionMask.decode_flat_action(state, chosen).primary_id)
            rolled = target in e.targets or target in e.war_targets
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
            if rolled and force_outcome(state, chosen, outcome):
                seed_settled = True
            elif rolled:
                conv.mismatches.append(Mismatch(
                    conv.replay_id, e.turn, e.phase, e.player, e.card,
                    "could not reproduce outcome",
                    f"no rng_state reproduced the logged "
                    f"{e.mode or e.event_mode or 'war'} result"))

        if informative:
            obs = np.asarray(ts.extract_observation(state, mover), dtype=np.float32)
            conv.samples.append((obs, mask.copy(), int(chosen),
                                 1 if mover == ts.Player.US else -1))
            conv.decisions_emitted += 1

        ts.Engine.step_flat(state, int(chosen))

    if guessed:
        conv.entries_guessed += 1
    return True


def _hand_after(turn_hand, played):
    """Cards still held: the turn's logged hand minus what has been played so far."""
    return [c for c in turn_hand if c not in played]


def _apply_hands(state, us_cards, ussr_cards) -> None:
    for c in range(1, 111):
        loc = state.get_card_location(c)
        if loc in (ts.CardLocation.HAND_US, ts.CardLocation.HAND_USSR):
            state.set_card_location(c, ts.CardLocation.DISCARD_PILE)
    for c in us_cards:
        state.set_card_location(c, ts.CardLocation.HAND_US)
    for c in ussr_cards:
        state.set_card_location(c, ts.CardLocation.HAND_USSR)


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

    cur_turn = None
    turn_hands = {"US": [], "USSR": []}
    played = {"US": set(), "USSR": set()}
    prev_raw = None
    prev_entry = None

    for raw in raws:
        e = parse_entry(raw)
        conv.entries_total += 1

        if e.turn and e.turn != cur_turn:
            cur_turn = e.turn
            h = hands.get(str(e.turn)) or {}
            turn_hands = {
                "US": [c for c in (card_id(n) for n in h.get("us", [])) if c],
                "USSR": [c for c in (card_id(n) for n in h.get("ussr", [])) if c],
            }
            played = {"US": set(), "USSR": set()}

        # --- rebuild the position this entry was decided from ---
        if prev_raw is not None:
            _reconcile_board(state, prev_raw.get("countries"))
        if prev_entry is not None:
            _reconcile_scalars(state, prev_entry)
        if state.current_phase == ts.Phase.GAME_OVER:
            state.current_phase = ts.Phase.ACTION_ROUND
        _apply_hands(state,
                     _hand_after(turn_hands["US"], played["US"]),
                     _hand_after(turn_hands["USSR"], played["USSR"]))

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

        before = len(conv.samples)
        _reconcile_turn(state, e)
        _drive_entry(state, e, conv, raw)

        # --- did replaying our parsed actions reproduce the log's board? ---
        bad = _board_matches(state, raw.get("countries"))
        if bad == 0:
            conv.entries_converted += 1
        else:
            conv.entries_board_mismatch += 1
            if conv.first_board_mismatch is None:
                conv.first_board_mismatch = Mismatch(
                    conv.replay_id, e.turn, e.phase, e.player, e.card,
                    "board mismatch after replay",
                    f"{bad} countries differ from the log after applying parsed actions")
                conv.mismatches.append(conv.first_board_mismatch)
            del conv.samples[before:]          # unverified actions are not training data
            conv.decisions_emitted -= 0

        if e.score is not None and int(state.victory_points) != int(e.score):
            conv.vp_drift += 1

        conv.board_resyncs += _reconcile_board(state, raw.get("countries"))
        _reconcile_scalars(state, e)
        prev_raw = raw
        prev_entry = e

    conv.decisions_emitted = len(conv.samples)
    return conv
