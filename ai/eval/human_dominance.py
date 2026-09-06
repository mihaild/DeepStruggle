"""Do the humans in the ts-replayer corpus respect the dominance relations?

`research/experiments.md` §4.8 measures our agents against pairs of options that are identical in
printed Ops and differ only in a way that is weakly better in every position, so preferring the
dominated one is wrong with no judgement to argue about. The control ranks Quagmire pairs **64.5%
wrong -- worse than a coin flip** -- and §4.8's conclusion was that this is "the strongest argument
yet for demonstrations, since one human game shows the reversal that RL needs thousands to notice".

That is a claim about the corpus, and it has never been checked against the corpus. This checks it.

The measure differs from §4.8's by necessity. For a policy you can ask how it *ranks* the two
options, because you have a distribution. For a human you have only the move they made, so the
question is whether the move they chose was the dominated one while a dominant alternative sat in
the same hand at the same Ops. That is the stricter and more meaningful half of §4.8's pair anyway:
"chose a dominated card", not "ranked the pair wrongly".
"""

from __future__ import annotations

import contextlib
import gzip
import json
from dataclasses import dataclass, field
from typing import Any, Dict, Iterator, List, Optional, Tuple

import numpy as np

import ts_engine as ts

from ai.eval.dominance import (discard_dominance_pairs, legal_card_actions,
                               space_dominance_alternatives, trapped_effect)
from tools.lib.corpus_paths import corpus_files
from tools.lib.ts_replayer_convert import Conversion, convert_game

SPACE_MODE = 2


@dataclass
class DominanceTally:
    """Counts over decisions where a dominance relation actually applied."""

    discard_decidable: int = 0
    discard_dominated: int = 0
    space_decidable: int = 0
    space_dominated: int = 0
    by_trap: Dict[str, List[int]] = field(default_factory=dict)  # trap -> [decidable, dominated]
    examples: List[str] = field(default_factory=list)

    def discard_rate(self) -> Optional[float]:
        if not self.discard_decidable:
            return None
        return 100.0 * self.discard_dominated / self.discard_decidable

    def space_rate(self) -> Optional[float]:
        if not self.space_decidable:
            return None
        return 100.0 * self.space_dominated / self.space_decidable


@dataclass
class _Capture:
    states: List[ts.GameState] = field(default_factory=list)
    movers: List[ts.Player] = field(default_factory=list)


@contextlib.contextmanager
def _capturing(cap: _Capture) -> Iterator[None]:
    """Snapshot the position at every decision the converter emits.

    The converter calls `ts.extract_observation` exactly once per emitted sample, at the single
    line that appends to `Conversion.samples`, so wrapping it keeps these lists in step with
    `samples`. `GameState.clone()` is required: the engine mutates one state in place, so a
    stored reference would read back as the end of the game.
    """
    original = ts.extract_observation

    def observe(state: ts.GameState, mover: ts.Player) -> Any:
        cap.states.append(state.clone())
        cap.movers.append(mover)
        return original(state, mover)

    ts.extract_observation = observe
    try:
        yield
    finally:
        ts.extract_observation = original


def _decode(state: ts.GameState, action: int) -> Any:
    return ts.decode_flat_action(state, int(action))


def measure_game(game: Dict[str, Any], tally: DominanceTally, replay_id: int) -> None:
    cap = _Capture()
    with _capturing(cap):
        conv: Conversion = convert_game(game)
    if conv.skipped is not None or conv.failure is not None:
        return

    n = len(conv.samples)
    for i in range(min(n, len(cap.states))):
        state = cap.states[i]
        mover = cap.movers[i]
        chosen = int(conv.samples[i][2])
        ctx = state.ctx()

        # --- discard under a trap -------------------------------------------------------
        # The phase check is not incidental. A headline play is a SELECT_CARD decision too, and
        # the trap flag is set in that state as well, so without it a headline is scored as a
        # trap discard -- and it is scored as an error almost every time, because the rule says
        # "discard the opponent's recurring event" while a headline is where you play your own
        # best card. That accounted for 14 of the 25 errors this first reported: at turn 5 of
        # replay 63 the USSR headlined Che and discarded Duck and Cover at AR1, and only the
        # headline was counted; at turn 6 of replay 71 the USSR headlined Quagmire.
        trap = trapped_effect(state, mover)
        if (trap is not None
                and ctx.decision_type == ts.DecisionType.SELECT_CARD
                and state.current_phase == ts.Phase.ACTION_ROUND):
            legal = legal_card_actions(state)
            pairs = discard_dominance_pairs(state, mover, legal)
            if pairs:
                better = {a for a, _b, _o in pairs}
                worse = {b for _a, b, _o in pairs}
                # Only decidable when the human's move is on one side of a pair; a third card
                # that is in no pair says nothing about the relation.
                if chosen in better or chosen in worse:
                    tally.discard_decidable += 1
                    slot = tally.by_trap.setdefault(trap, [0, 0])
                    slot[0] += 1
                    if chosen in worse and chosen not in better:
                        tally.discard_dominated += 1
                        slot[1] += 1
                        if len(tally.examples) < 12:
                            card = int(_decode(state, chosen).primary_id)
                            name = str(ts.CardData.get_card_info(card)["name"])
                            tally.examples.append(
                                f"replay {replay_id} T{int(state.turn)} {trap}: discarded {name}")

        # --- what was spent on the space race --------------------------------------------
        if ctx.decision_type == ts.DecisionType.SELECT_PLAY_MODE:
            ma = _decode(state, chosen)
            if int(ma.secondary_id) == SPACE_MODE or int(ma.primary_id) == SPACE_MODE:
                card = int(ctx.resolving_card) or int(ctx.pending_op_card)
                if 1 <= card <= 110:
                    # A space play is only decidable when the hand held both kinds of card at
                    # the same Ops, so a choice between them actually existed. Counting every
                    # opponent-card space as a correct decision inflates the denominator with
                    # non-decisions -- the first version of this did, and reported 0 errors in
                    # 265 "decidable" plays because nearly all of them were forced.
                    alts = space_dominance_alternatives(state, mover, card)
                    if alts:
                        # Spaced own/neutral while holding an equal-Ops opponent recurring
                        # event: a real choice, and the wrong side of it.
                        tally.space_decidable += 1
                        tally.space_dominated += 1
                    elif (_spaced_an_opponent_card(state, mover, card)
                          and _had_equal_ops_own_alternative(state, mover, card)):
                        # Spaced the opponent's card while an equal-Ops own/neutral card was
                        # available: the same choice, taken correctly.
                        tally.space_decidable += 1



def _had_equal_ops_own_alternative(state: ts.GameState, mover: ts.Player, card: int) -> bool:
    """Was an own-or-neutral card of the same printed Ops also in hand?

    Without one there was no choice to get right, and the decision does not belong in the
    denominator. This mirrors `space_dominance_alternatives` with the sides reversed.
    """
    from ai.eval.dominance import _eligible_own_or_neutral, hand_cards

    opponent = "USSR" if mover == ts.Player.US else "US"
    ops = int(ts.CardData.get_card_info(card)["ops"])
    return any(c != card
               and _eligible_own_or_neutral(c, opponent)
               and int(ts.CardData.get_card_info(c)["ops"]) == ops
               for c in hand_cards(state, mover))


def _spaced_an_opponent_card(state: ts.GameState, mover: ts.Player, card: int) -> bool:
    """The dominant choice: the card spent on the track was the opponent's."""
    opponent = "USSR" if mover == ts.Player.US else "US"
    return str(ts.CardData.get_card_info(card)["side"]) == opponent


def measure_corpus(limit: Optional[int] = None) -> DominanceTally:
    tally = DominanceTally()
    for path in corpus_files()[:limit]:
        with gzip.open(path, "rt") as fh:
            game = json.load(fh)
        if not game.get("all_turns"):
            continue
        measure_game(game, tally, int(game.get("replay_id", -1)))
    return tally
