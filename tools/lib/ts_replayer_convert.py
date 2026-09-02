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
    """Countries the log says were pointed at, one entry per influence point."""
    q: List[int] = []
    for _side, delta, cid, _u, _s in (e.ops_influence or []):
        q.extend([cid] * abs(int(delta)))
    if e.coup_target is not None:
        q.append(e.coup_target)
    return q


# -- driving the engine --------------------------------------------------------------------

from ai.eval.positions import PLAY_MODE_ACTION  # noqa: E402

_OP_MODE = {"influence": ts.OpMode.INFLUENCE, "coup": ts.OpMode.COUP,
            "realign": ts.OpMode.REALIGN}


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
    picked_card = cid_target is None
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
        if (picked_card and picked_mode and not pq and not headline_ids
                and dt == ts.DecisionType.SELECT_CARD):
            break

        mover = _acting(state)
        chosen: Optional[int] = None
        informative = False

        if dt == ts.DecisionType.SELECT_CARD and headline_ids:
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
            if e.mode or opponent_card:
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

        elif dt == ts.DecisionType.SELECT_OP_MODE and e.mode in _OP_MODE:
            om = int(_OP_MODE[e.mode])
            chosen = _find(state, legal, ts.DecisionType.SELECT_OP_MODE,
                           lambda ma: int(ma.primary_id) == om or int(ma.secondary_id) == om)
            informative = chosen is not None

        elif dt == ts.DecisionType.POINT_NODE and pq:
            want_c = pq[0]
            chosen = _find(state, legal, ts.DecisionType.POINT_NODE,
                           lambda ma: int(ma.primary_id) == want_c)
            if chosen is not None:
                pq.pop(0)
                informative = True
            else:
                nm = ts.MapData.get_country_info(want_c)["name"]
                conv.mismatches.append(Mismatch(
                    conv.replay_id, e.turn, e.phase, e.player, e.card,
                    "target not legal", f"{nm} not among legal point targets"))
                pq.pop(0)

        if chosen is None:
            chosen = int(legal[0])
            guessed = True

        if informative:
            obs = np.asarray(ts.extract_observation(state, mover), dtype=np.float32)
            conv.samples.append((obs, mask.copy(), int(chosen),
                                 1 if mover == ts.Player.US else -1))
            conv.decisions_emitted += 1

        ts.Engine.step_flat(state, int(chosen))

    if guessed:
        conv.entries_guessed += 1
    return True


def convert_game(game: Dict) -> Conversion:
    conv = Conversion(replay_id=int(game.get("replay_id", -1)))
    raws = game.get("all_turns", [])
    hands = game.get("hands", {}) or {}

    state = ts.GameState()
    ts.Engine.init_game(state, 12345)
    _drain(state)

    cur_turn = None
    prev_countries = None
    prev_entry = None
    for raw in raws:
        e = parse_entry(raw)
        if prev_countries is not None:
            _reconcile_board(state, prev_countries)
        if prev_entry is not None:
            _reconcile_scalars(state, prev_entry)
        conv.entries_total += 1
        if e.turn and e.turn != cur_turn:
            cur_turn = e.turn
            h = hands.get(str(e.turn))
            if h:
                _set_hand(state, ts.Player.US, h.get("us", []))
                _set_hand(state, ts.Player.USSR, h.get("ussr", []))
        if ts.Engine.is_terminal(state):
            conv.mismatches.append(Mismatch(
                conv.replay_id, e.turn, e.phase, e.player, e.card,
                "engine ended the game early",
                f"log continues; engine vp={int(state.victory_points)}, "
                f"log score={e.score}"))
            break
        if _drive_entry(state, e, conv, raw):
            conv.entries_converted += 1
        # Compare BEFORE reconciling: engine VP drifting from the log is the earliest and
        # sharpest signal that this entry was reconstructed wrongly.
        if e.score is not None and int(state.victory_points) != int(e.score):
            conv.vp_drift += 1
            if conv.first_vp_drift is None:
                conv.first_vp_drift = Mismatch(
                    conv.replay_id, e.turn, e.phase, e.player, e.card, "VP drift",
                    f"engine {int(state.victory_points)} vs log {int(e.score)}")
                conv.mismatches.append(conv.first_vp_drift)
        conv.board_resyncs += _reconcile_board(state, raw.get("countries"))
        _reconcile_scalars(state, e)
        prev_countries = raw.get("countries")
        prev_entry = e
    return conv
