"""Local decisions with a position-independent right answer.

Most evaluation here has the problem that "the better move" depends on the position, so a
disagreement with the policy is not necessarily an error. These rules do not: they compare
two options that are identical in the dimension that matters (printed Ops) and differ only in
a way that is weakly better in *every* position. A policy that prefers the dominated option is
wrong, with no strategic judgement to argue about.

This measures; it does not train. The rules are never fed to the agent.

Why it is needed: the earlier attempt compared the discard against "the card the policy most
wants to play now", which conflates two different questions -- sometimes the best card to play
is also the right card to discard. Dominance pairs remove that confound.

Rules currently encoded
-----------------------
**Discard (Quagmire / Bear Trap).** At equal printed Ops, discarding an opponent-associated
*recurring* event is never worse than discarding your own or a neutral card. Holding an
opponent's event means eventually triggering it for them; discarding it does not. Your own or
a neutral card of the same Ops could instead have been played for your benefit.

Exceptions, all excluded from the "opponent" side of a pair:
  * **Five Year Plan** (#5) -- the one recurring event whose firing can help its non-owner, so
    the usual reasoning does not hold.
  * **one-time (starred) events** -- discarding those removes them from the game permanently,
    which is a different and usually stronger argument. Left out to keep every pair strictly
    defensible rather than merely usually right.
  * **scoring cards and The China Card** -- special handling, never ordinary discards.

**Space race.** The same relation, applied to what you spend on the space track: do not space
your own or a neutral card while holding a recurring opponent event of equal printed Ops.
Spacing consumes the card without firing its event either way, so consuming the opponent's is
never worse -- it denies them the event and leaves your own card available to play.

Note there is no rule about playing an opponent card for its Event: the engine makes that
illegal (verified over 971 play-mode decisions on opponent cards, EVENT legal in none). An
opponent card is always played for Ops and its event fires on its own.
"""

from typing import Dict, List, Optional, Tuple

import numpy as np
import ts_engine as ts

FIVE_YEAR_PLAN = 5
CHINA_CARD = 6

QUAGMIRE_ACTIVE = 1 << 14
BEAR_TRAP_ACTIVE = 1 << 15


def _side_of(card_id: int) -> str:
    return str(ts.CardData.get_card_info(card_id)["side"])


def _info(card_id: int) -> Dict:
    return ts.CardData.get_card_info(card_id)


def trapped_effect(state: ts.GameState, player: ts.Player) -> Optional[str]:
    """Which trap, if any, binds `player` in this state."""
    eff = int(state.persistent_effects)
    if (eff & QUAGMIRE_ACTIVE) and player == ts.Player.US:
        return "quagmire"
    if (eff & BEAR_TRAP_ACTIVE) and player == ts.Player.USSR:
        return "bear_trap"
    return None


def _eligible_opponent_discard(card_id: int, opponent: str) -> bool:
    info = _info(card_id)
    if card_id in (FIVE_YEAR_PLAN, CHINA_CARD):
        return False
    if info["is_scoring"] or info["one_time"]:
        return False
    return str(info["side"]) == opponent


def _eligible_own_or_neutral(card_id: int, opponent: str) -> bool:
    info = _info(card_id)
    if card_id in (FIVE_YEAR_PLAN, CHINA_CARD):
        return False
    if info["is_scoring"]:
        return False
    return str(info["side"]) != opponent


def discard_dominance_pairs(
    state: ts.GameState,
    player: ts.Player,
    legal_cards: Dict[int, int],
) -> List[Tuple[int, int, int]]:
    """Pairs (better_action, worse_action, ops) among legal discards.

    `legal_cards` maps flat action index -> card id. Returns pairs where discarding
    `better_action` is weakly dominant: same printed Ops, opponent recurring event against an
    own or neutral card.
    """
    opponent = "USSR" if player == ts.Player.US else "US"
    out: List[Tuple[int, int, int]] = []
    for a_act, a_card in legal_cards.items():
        if not _eligible_opponent_discard(a_card, opponent):
            continue
        a_ops = int(_info(a_card)["ops"])
        for b_act, b_card in legal_cards.items():
            if b_act == a_act or not _eligible_own_or_neutral(b_card, opponent):
                continue
            if int(_info(b_card)["ops"]) != a_ops:
                continue
            out.append((int(a_act), int(b_act), a_ops))
    return out


def legal_card_actions(state: ts.GameState) -> Dict[int, int]:
    """Legal SELECT_CARD actions in this state, as {flat action index: card id}."""
    mask = np.asarray(ts.ActionMask.generate_flat_mask(state))
    out: Dict[int, int] = {}
    for a in np.flatnonzero(mask):
        ma = ts.ActionMask.decode_flat_action(state, int(a))
        if int(ma.decision_type) != int(ts.DecisionType.SELECT_CARD):
            continue
        cid = int(ma.primary_id)
        # SELECT_CARD also carries non-card options (pass / no-card); ignore those.
        if 1 <= cid <= 110:
            out[int(a)] = cid
    return out


def hand_cards(state: ts.GameState, player: ts.Player) -> List[int]:
    """Card ids currently held by `player`."""
    want = ts.CardLocation.HAND_US if player == ts.Player.US else ts.CardLocation.HAND_USSR
    return [c for c in range(1, 111) if state.get_card_location(c) == want]


def space_dominance_alternatives(
    state: ts.GameState,
    player: ts.Player,
    spaced_card: int,
) -> List[int]:
    """Recurring opponent events of equal Ops that should have been spaced instead.

    Empty when `spaced_card` is itself an opponent card, or when the hand holds no equal-Ops
    opponent recurring event. A non-empty result is an unambiguous error: spacing the
    opponent's card denies them the event and keeps your own card playable.
    """
    opponent = "USSR" if player == ts.Player.US else "US"
    if not (1 <= spaced_card <= 110):
        return []
    if not _eligible_own_or_neutral(spaced_card, opponent):
        return []
    ops = int(_info(spaced_card)["ops"])
    return [c for c in hand_cards(state, player)
            if c != spaced_card
            and _eligible_opponent_discard(c, opponent)
            and int(_info(c)["ops"]) == ops]


def space_dominance_outcome(
    state: ts.GameState,
    player: ts.Player,
    spaced_card: int,
) -> Optional[bool]:
    """Was this space play right, wrong, or not a test case?

    Returns None when no choice existed, True when the dominant card was spaced, False when a
    dominated one was. Reporting violations as a share of *all* space plays understates the
    error, because most space plays offer no equal-Ops opponent alternative at all; the honest
    denominator is the plays where the choice was actually available.
    """
    opponent = "USSR" if player == ts.Player.US else "US"
    if not (1 <= spaced_card <= 110):
        return None
    ops = int(_info(spaced_card)["ops"])
    hand = [c for c in hand_cards(state, player) if c != spaced_card]

    if _eligible_own_or_neutral(spaced_card, opponent):
        # Wrong choice iff a dominant alternative was actually held.
        alts = [c for c in hand
                if _eligible_opponent_discard(c, opponent) and int(_info(c)["ops"]) == ops]
        return False if alts else None

    if _eligible_opponent_discard(spaced_card, opponent):
        # Right choice, but only counts as a test case if a dominated option was available.
        others = [c for c in hand
                  if _eligible_own_or_neutral(c, opponent) and int(_info(c)["ops"]) == ops]
        return True if others else None

    return None
