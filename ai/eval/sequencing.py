"""Disposing of a card you must not play.

`experiments.md` §25 names turn sequencing as the gap between this agent and a mediocre human,
and the position it names is about *exits*, not about severity. Turn 10, USSR holding UN
Intervention, Tear Down this Wall and Grain Sales to Soviets. Both US cards are DEFCON-suicide
there -- playing either for Operations fires its event, the event drops DEFCON to 1, and
`resolve_defcon_one_loss` makes the phasing player the loser. That is not the distinction. The
distinction is what each card can do *instead* of being played:

| card | Ops | at that space box (3 required) |
|:---|---:|:---|
| Tear Down this Wall | 3 | spaceable -- spent on the track, the event never fires |
| Grain Sales to Soviets | 2 | not spaceable -- too few Ops for the next box |

A card you must not play has three exits: space it, run it through UN Intervention, or hold it at
end of turn. Tear Down this Wall had all three; Grain Sales had two. UN Intervention is the
scarce exit and belongs on the card with fewer. The model spent it on the one that could have
spaced itself and was then left holding the one that could not.

And UN Intervention costs a card most of the accounting misses. Its event plays an opponent card
for Operations *in the same action round*, so it consumes two cards in one AR -- which is exactly
the card that would otherwise have been held back at end of turn. Using it does not only spend
itself; it removes the "hold it" exit for everything else in the hand.

This is a baseline, not a gate. `dcedb8a` counted 34.1% of self-play games ending at DEFCON 1
with 82.7% of those provoked -- the loser pushed into it -- and Grain Sales alone a quarter of
them, so the §25 position is the modal failure rather than an anecdote. These are the numbers a
later strategy arm has to move.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

import numpy as np
import ts_engine as ts

from ai.eval.blunders import defcon_suicide_cards, hand_of, in_action_round_at_defcon_2
from ai.stats import wilson_interval

UN_INTERVENTION = 32

#: Flat action indices. Cards are 0..109 for ids 1..110; the play modes follow.
CARD_ACTION_START = 0
PLAY_MODE_START = 110
SPACE_ACTION = PLAY_MODE_START + int(ts.PlayMode.SPACE)


@dataclass
class Rate:
    """One measure: how often the chance arose, and how often it was taken badly."""

    name: str
    opportunities: int = 0
    mistakes: int = 0
    #: Card ids involved, so a rate can be read back as a story rather than a number.
    examples: List[str] = field(default_factory=list)

    def rate(self) -> float:
        return self.mistakes / self.opportunities if self.opportunities else float("nan")

    def band(self) -> Tuple[float, float]:
        return wilson_interval(self.mistakes, self.opportunities)

    def note(self, mistake: bool, detail: str = "") -> None:
        self.opportunities += 1
        if mistake:
            self.mistakes += 1
            if detail and len(self.examples) < 8:
                self.examples.append(detail)


@dataclass
class DisposalCounts:
    """The three measures, in order of how little judgement each needs."""

    #: UN Intervention was spent on a card that was not the one it was needed for.
    spent_elsewhere: Rate = field(default_factory=lambda: Rate("un_spent_elsewhere"))
    #: Two suicide cards, one spaceable and one not, and it went to the spaceable one.
    spent_on_spaceable: Rate = field(default_factory=lambda: Rate("un_spent_on_spaceable"))
    #: The suicide card was played for Operations anyway, with UN Intervention still in hand.
    played_raw: Rate = field(default_factory=lambda: Rate("suicide_played_raw"))
    games: int = 0

    def all_rates(self) -> List[Rate]:
        return [self.spent_elsewhere, self.spent_on_spaceable, self.played_raw]

    def metrics(self) -> Dict[str, float]:
        out: Dict[str, float] = {}
        for r in self.all_rates():
            out[f"disposal/{r.name}"] = r.rate()
            out[f"disposal/{r.name}_opportunities"] = float(r.opportunities)
        return out


def _name(card_id: int) -> str:
    try:
        return str(ts.CardData.get_card_info(int(card_id))["name"])
    except Exception:
        return f"card {card_id}"


def legal_cards(state: ts.GameState) -> Set[int]:
    """Card ids the mask allows at this node."""
    mask = np.asarray(ts.get_flat_action_mask(state), dtype=np.uint8)
    return {int(i) + 1 for i in np.flatnonzero(mask) if i < 110}


def is_spaceable(state: ts.GameState, card_id: int) -> bool:
    """Can this card go on the space track from here?

    Asked of the engine rather than worked out from printed Ops. `can_attempt_space` compares
    *effective* Ops against the next box's minimum, so Brezhnev Doctrine or Containment can make
    a 2-Ops card reach a 3-Ops box -- and the attempts-per-turn limit and the track position
    matter too. Stepping to the card's play-mode node and reading the mask asks exactly the
    question the engine would answer if the player tried it.
    """
    if not (1 <= card_id <= 110):
        return False
    probe = state.clone()
    action = CARD_ACTION_START + card_id - 1
    mask = np.asarray(ts.get_flat_action_mask(probe), dtype=np.uint8)
    if action >= mask.shape[0] or not mask[action]:
        return False
    if not ts.Engine.step_flat(probe, action):
        return False
    if probe.ctx().decision_type != ts.DecisionType.SELECT_PLAY_MODE:
        return False
    modes = np.asarray(ts.get_flat_action_mask(probe), dtype=np.uint8)
    return bool(modes[SPACE_ACTION])


def is_un_intervention_companion_node(state: ts.GameState) -> bool:
    """The node where UN Intervention's event asks which opponent card to spend it on.

    `trigger_un_intervention` sets a SELECT_CARD decision with UN Intervention as the card being
    resolved, and `ActionMask` then offers exactly the opponent-associated cards in hand.
    """
    ctx = state.ctx()
    return (ctx.decision_type == ts.DecisionType.SELECT_CARD
            and int(ctx.resolving_card) == UN_INTERVENTION
            and int(ctx.pending_op_card) == UN_INTERVENTION)


def observe(state: ts.GameState, action: int, counts: DisposalCounts,
            carry: Optional[Dict[str, Any]] = None) -> None:
    """Score one decision, before it is taken, against the three measures.

    `carry` is this env's scratch space, and it exists for one reason. Spaceability has to be
    asked of a position where playing the card is the live question -- at the companion node a
    card action means "UN Intervention takes this one", so stepping it there answers a different
    question and reports everything as unspaceable. So the answer is taken one decision earlier,
    where the player chose to play UN Intervention at all, and carried forward.
    """
    ctx = state.ctx()
    player = ctx.decision_player
    if player == ts.Player.NONE:
        return
    if carry is None:
        carry = {}

    # --- the companion node: what is UN Intervention actually spent on? -----------------------
    if is_un_intervention_companion_node(state):
        if not in_action_round_at_defcon_2(state):
            return          # "must not play" is a DEFCON-2 statement; elsewhere there is no bar
        chosen = action - CARD_ACTION_START + 1
        offered = legal_cards(state)
        banned = defcon_suicide_cards(state, player) & offered
        if not banned:
            carry.pop("spaceable", None)
            return

        counts.spent_elsewhere.note(
            chosen not in banned,
            f"spent on {_name(chosen)} with {sorted(_name(c) for c in banned)} in hand")

        # The sharper question, and the §25 error exactly: among the cards that must not be
        # played, was the one with another way out the one it was spent on?
        known = carry.pop("spaceable", None)
        if known is not None:
            spaceable = {c for c in banned if known.get(c, False)}
            unspaceable = banned - spaceable
            if spaceable and unspaceable:
                counts.spent_on_spaceable.note(
                    chosen in spaceable,
                    f"spent on {_name(chosen)} (spaceable) with "
                    f"{sorted(_name(c) for c in unspaceable)} unspaceable")
        return

    # --- an ordinary card choice: if UN Intervention is being played, record what the cards it
    # could be spent on could have done instead, while that is still answerable. ---------------
    if ctx.decision_type == ts.DecisionType.SELECT_CARD:
        chosen = action - CARD_ACTION_START + 1
        if chosen == UN_INTERVENTION and in_action_round_at_defcon_2(state):
            banned = defcon_suicide_cards(state, player)
            carry["spaceable"] = {c: is_spaceable(state, c)
                                  for c in banned if c in set(hand_of(state, player))}
        return

    # --- the play-mode node: was a card that must not be played played anyway? ----------------
    if ctx.decision_type == ts.DecisionType.SELECT_PLAY_MODE:
        if not in_action_round_at_defcon_2(state):
            return
        card = int(ctx.pending_op_card)
        if card not in defcon_suicide_cards(state, player):
            return
        # Only counts while the tool was still available. Playing a suicide card with no exit
        # left is a position, not a mistake.
        if UN_INTERVENTION not in hand_of(state, player):
            return
        counts.played_raw.note(
            action != SPACE_ACTION,
            f"{_name(card)} played as {'SPACE' if action == SPACE_ACTION else 'OPS/EVENT'} "
            f"with UN Intervention in hand")


def measure(select_action: Any, num_games: int = 500, base_seed: int = 20260911,
            batch_size: int = 128, max_steps: int = 4000) -> DisposalCounts:
    """Play games and score every decision that touches card disposal.

    `select_action(obs, masks) -> flat action indices`, batched, as the setup probe takes it.
    """
    counts = DisposalCounts()
    done = 0
    while done < num_games:
        n = min(batch_size, num_games - done)
        runner = ts.VectorizedBatchRunner(n, base_seed + done * 7919)
        carries: List[Dict[str, Any]] = [{} for _ in range(n)]
        for _ in range(max_steps):
            terminals = runner.get_terminals()
            if all(terminals):
                break
            obs = np.asarray(runner.get_observations(), dtype=np.float32)
            masks = np.asarray(runner.get_action_masks())
            actions = select_action(obs, masks)
            for i in range(n):
                if terminals[i]:
                    continue
                observe(runner.get_state(i), int(actions[i]), counts, carries[i])
            runner.step_flat_all([int(a) for a in actions], auto_advance=False)
        done += n
    counts.games = num_games
    return counts


def format_report(counts: DisposalCounts, label: str = "") -> str:
    lines = [f"=== card disposal: {label or 'policy'} ({counts.games} games) ==="]
    for r in counts.all_rates():
        if not r.opportunities:
            lines.append(f"  {r.name:<24} no opportunities arose")
            continue
        lo, hi = r.band()
        lines.append(f"  {r.name:<24} {r.rate() * 100:5.1f}% [{lo * 100:4.1f}, {hi * 100:4.1f}]"
                     f"   {r.mistakes}/{r.opportunities}")
        for ex in r.examples[:2]:
            lines.append(f"      e.g. {ex}")
    return "\n".join(lines)
