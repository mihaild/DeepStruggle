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
    if entry.score is not None:
        state.victory_points = int(entry.score)
    if entry.defcon is not None and 1 <= int(entry.defcon) <= 5:
        state.defcon = int(entry.defcon)


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


def _drain(state: ts.GameState) -> None:
    while (not ts.Engine.is_terminal(state)
           and state.ctx().decision_player == ts.Player.NONE
           and state.ctx().decision_type == ts.DecisionType.ROLL_DIE):
        ts.Engine.step(state, ts.MicroAction(ts.DecisionType.ROLL_DIE, 0, 0, 0))


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
        return list(e.targets)

    q: List[int] = []
    for _side, delta, cid, _u, _s in (e.ops_influence or []):
        q.extend([cid] * abs(int(delta)))
    if not q and e.targets:
        q = list(e.targets)
    return q


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


def _reconstruct_target(state: ts.GameState, legal, raw: Optional[Dict]) -> Optional[int]:
    """Pick the legal target whose resolution leaves the board as the log records it.

    Used only where the log omits an operation's target entirely. Dice are searched the same
    way force_outcome does, so a target is accepted only if some roll makes the whole board
    agree -- not merely the target country.
    """
    board = _logged_board(raw)
    if not board:
        return None
    for a in legal:
        ma = ts.ActionMask.decode_flat_action(state, int(a))
        if int(ma.decision_type) != int(ts.DecisionType.POINT_NODE):
            continue
        # force_outcome never steps; on success it leaves the winning seed on `state` itself,
        # which is exactly what the caller needs before it steps this action.
        if force_outcome(state, int(a), board):
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
    second_cid = card_id(e.played_card) if e.played_card else None
    picked_mode = False
    guessed = False

    for _ in range(max_steps):
        _drain(state)
        if ts.Engine.is_terminal(state):
            break
        ctx = state.ctx()
        dt = ctx.decision_type
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

        elif picked_card and dt == ts.DecisionType.SELECT_CARD:
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
            side = str(ts.CardData.get_card_info(cid_target)["side"]) if cid_target else "NONE"
            mine = "US" if mover == ts.Player.US else "USSR"
            opponent_card = side not in ("NONE", mine)
            if e.played_card:
                # UN Intervention is played as its Event; the Ops that follow are the named
                # card's. Reading the "Place Influence" header as the play mode instead had
                # the engine spend UN Intervention's own Ops and never ask for NORAD.
                want = PLAY_MODE_ACTION["event"]
            elif not opponent_card and e.event_first and e.events:
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

        elif dt == ts.DecisionType.SELECT_OP_MODE and e.mode in _OP_MODE:
            om = int(_OP_MODE[e.mode])
            # primary_id only. INFLUENCE is 0 and secondary_id defaults to 0, so matching
            # either field silently selected the first legal action -- REALIGN -- and the
            # engine then correctly offered a realignment mask, which looked like a mask bug.
            chosen = _find(state, legal, ts.DecisionType.SELECT_OP_MODE,
                           lambda ma: int(ma.primary_id) == om)
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
            # The log sometimes prints an Ops header with nothing under it: at turn 4 AR2 Che's
            # free USSR coup is "Coup (3 Ops):" and then nothing -- no target, no result, no
            # military ops line. The board snapshot is still authoritative, so pick whichever
            # legal target reproduces it rather than guessing and inventing influence.
            chosen = _reconstruct_target(state, legal, raw)
            if chosen is not None:
                conv.mismatches.append(Mismatch(
                    conv.replay_id, e.turn, e.phase, e.player, e.card,
                    "log gap: unrecorded operation",
                    f"{e.mode or e.event_mode or 'operation'} has no target in the log; "
                    f"reconstructed the one consistent with the entry's board"))

        if chosen is None:
            chosen = int(legal[0])
            guessed = True

        # Anything the engine settles with a die it rolls itself: coups, realignments, and war
        # events, whose target is chosen the same way but never carried an Ops mode.
        if dt == ts.DecisionType.POINT_NODE and chosen is not None:
            target = int(ts.ActionMask.decode_flat_action(state, chosen).primary_id)
            rolled = target in e.targets or target in e.war_targets
            if rolled and not force_outcome(state, chosen, expected_counts(e, raw)):
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
