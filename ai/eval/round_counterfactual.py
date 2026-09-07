"""One Action Round, played by the human and by the model, scored by the same critic.

§15 compared two ways of playing the *same card*, with the model resolving both branches. That
leaves a confound it cannot remove: when the critic prefers the Ops line it may be pricing the mode
wrongly, or it may be correctly disliking how this model executes the event. De-Stalinization asks
for eight choices and §14.3 shows the model spending them by stripping East Germany.

This removes the confound. The human branch is the board the humans actually produced, taken from
the log through `convert_game`'s `on_entry` seam and already checked against the log's own next
position. The model branch starts from the identical board and lets the model play that Action
Round however it likes -- its own card, its own mode, its own targets. Then both are scored by the
same value head from the same side.

If the critic still prefers the model's round over a human's De-Stalinization, the critic is wrong
about the card and not merely about the model's execution of it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

import ts_engine as ts

from ai.eval.card_probe import _drain, card_name, hand, policy


@dataclass
class RoundPair:
    """One Action Round: the board before it, and the board the human left behind."""

    before: ts.GameState
    after_human: ts.GameState
    mover: ts.Player
    turn: int
    action_round: int
    phase: str
    card: str
    replay_id: int


def capture_rounds(game: Dict, card: int, side: ts.Player,
                   turns: Optional[Sequence[int]] = None) -> List[RoundPair]:
    """Every Action Round in which `side` played `card`, with the board before and after.

    The two states come from different callbacks -- the decision the human faced, and the board
    their whole entry produced -- so they are paired by entry rather than assumed adjacent.
    """
    from tools.lib.ts_replayer_convert import convert_game

    wanted = None if turns is None else {int(t) for t in turns}
    pending: Dict[int, Tuple[ts.GameState, Any]] = {}
    out: List[RoundPair] = []

    def key(entry: Any) -> int:
        return id(entry)

    def on_decision(state: ts.GameState, mover: ts.Player, entry: Any, chosen: int) -> None:
        if mover != side or state.current_phase != ts.Phase.ACTION_ROUND:
            return
        if state.ctx().decision_type != ts.DecisionType.SELECT_CARD:
            return
        if wanted is not None and int(entry.turn) not in wanted:
            return
        if card not in hand(state, side):
            return
        # The first SELECT_CARD of the entry is the one the human faced with a full hand.
        pending.setdefault(key(entry), (state, entry))

    def on_entry(state: ts.GameState, entry: Any) -> None:
        found = pending.pop(key(entry), None)
        if found is None:
            return
        before, e = found
        # Only rounds where the human actually played this card. `pending` also collects rounds
        # where they merely held it, and those have nothing to compare against.
        if card_name(card).lower().split("*")[0] not in str(e.card or "").lower():
            return
        out.append(RoundPair(before, state, side, int(e.turn), int(before.action_round),
                             str(e.phase), str(e.card or ""), -1))

    convert_game(game, on_decision=on_decision, on_entry=on_entry)
    return out


def play_one_round(model: Any, state: ts.GameState, mover: ts.Player, device: Any,
                   max_steps: int = 120) -> Optional[ts.GameState]:
    """Let the model play this Action Round out, greedily, and stop when the round is over.

    "Over" is when the turn or the action-round counter moves on, or the other side is asked for a
    card in an Action Round of its own. Decisions that belong to the opponent *inside* this round
    -- an event of ours that asks them something -- are answered by the same model and do not end
    it.
    """
    probe = state.clone()
    turn, round_no = int(probe.turn), int(probe.action_round)

    for _ in range(max_steps):
        _drain(probe)
        if ts.Engine.is_terminal(probe):
            return probe
        if int(probe.turn) != turn or int(probe.action_round) != round_no:
            return probe
        ctx = probe.ctx()
        if (ctx.decision_type == ts.DecisionType.SELECT_CARD
                and ctx.decision_player not in (mover, ts.Player.NONE)
                and probe.current_phase == ts.Phase.ACTION_ROUND):
            return probe
        who = ctx.decision_player if ctx.decision_player != ts.Player.NONE else mover
        probs = policy(model, probe, who, device)
        if not probs.any():
            return probe
        ts.Engine.step_flat(probe, int(np.argmax(probs)))
    return probe


@dataclass
class RoundComparison:
    """What the critic thinks of the human's round against the model's, from one side."""

    label: str
    human: List[float] = field(default_factory=list)
    model: List[float] = field(default_factory=list)
    # Rounds where the model chose to play the same card the human did.
    same_card: int = 0
    rounds: int = 0
    model_cards: Dict[str, int] = field(default_factory=dict)

    def gap(self) -> float:
        """v(human's board) - v(model's board). Positive means the critic prefers the human."""
        if not self.human:
            return float("nan")
        return float(np.mean(np.asarray(self.human) - np.asarray(self.model)))

    def stderr(self) -> float:
        if len(self.human) < 2:
            return float("nan")
        diff = np.asarray(self.human) - np.asarray(self.model)
        return float(np.std(diff, ddof=1) / np.sqrt(len(diff)))

    def human_preferred(self) -> int:
        return int(sum(1 for h, m in zip(self.human, self.model) if h > m))


def compare(model: Any, pairs: Sequence[RoundPair], device: Any,
            label: str = "") -> RoundComparison:
    """Score both boards with the same critic, from the side that played the round."""
    from ai.eval.battleground_value import value_of

    out = RoundComparison(label=label)
    for pair in pairs:
        after_model = play_one_round(model, pair.before, pair.mover, device)
        if after_model is None:
            continue
        out.rounds += 1

        # What the model chose to do with the round, for the record.
        probs = policy(model, pair.before, pair.mover, device)
        if probs.any():
            ma = ts.decode_flat_action(pair.before, int(np.argmax(probs)))
            if ma.decision_type == ts.DecisionType.SELECT_CARD and 1 <= int(ma.primary_id) <= 110:
                name = card_name(int(ma.primary_id))
                out.model_cards[name] = out.model_cards.get(name, 0) + 1
                if name.lower() in pair.card.lower():
                    out.same_card += 1

        out.human.append(value_of(model, pair.after_human, pair.mover, device))
        out.model.append(value_of(model, after_model, pair.mover, device))
    return out
