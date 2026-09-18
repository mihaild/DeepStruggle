"""Disposing of a card you must not play.

Turn sequencing is the gap between a trained model and a mediocre human: the model can play a
tactic when the card is in front of it, and cannot choose which card to spend it on. Nothing
else in `ai/eval/` measures that, which is why this probe exists.

**The position it comes from** (turn 10, USSR to play). The USSR
held UN Intervention, Tear Down this Wall and Grain Sales to Soviets. Both US cards are
DEFCON-suicide there — that is not the distinction. The distinction is the *exit* each one has:

| card | ops | at a space box requiring 3 |
|:---|---:|:---|
| Tear Down this Wall (`cid 96`) | 3 | **spaceable** — the event never fires |
| Grain Sales to Soviets (`cid 67`) | 2 | **not spaceable** |

A card you must not play has exactly three exits: space it, run it through UN Intervention, or
hold it at end of turn. Tear Down this Wall has all three; Grain Sales has two. So UN
Intervention — the scarce exit — belongs on the card with fewer of them. The model spent it on
the card that could have spaced itself and was left holding the other.

Three measures, in order of how little judgement each needs. The first two are here; the third
is `blunders.py`'s `defcon_suicide_with_alternative`, reported beside them by `format_report`
so the chain reads in one place.

1. **`un_intervention_off_target`** — the hand holds a DEFCON-suicide card, UN Intervention is
   being resolved and a companion is being chosen, and the choice falls on some other card. No
   judgement about the best line: the problem was in hand, the tool was in hand, it went
   elsewhere.
2. **`un_intervention_on_spaceable`** — two or more suicide cards are available as companions,
   at least one spaceable and at least one not, and the choice falls on a spaceable one. A
   strict subset of (1), and the sharpest single number, because the alternative is not a
   matter of taste.

**Spaceability is read from the engine, never reimplemented.** `SpaceRace::can_attempt_space` is
not exposed to Python and should not be added for this: the candidate is stepped to its
`SELECT_PLAY_MODE` node and the legality of the space action is read off the mask, which is the
engine's own rule including every Ops modifier. Printed Ops is not the test — the same card is
spaceable or not depending on the box, the attempts already made and the modifiers in play.

That test needs a state where the player is still choosing a card, so the probe keeps the last
ordinary card-selection node per environment and asks the question there. By the time UN
Intervention is resolving, its own play has already consumed the action round.

The two `SELECT_CARD` nodes are told apart by `pending_op_card == UN_INTERVENTION`
(`action_mask.cpp:58`). Verified over 66,535 selection nodes: that field and `resolving_card`
agree in every case, with no stale-context disagreement of the kind that produced the Aldrich
Ames defect.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np

import ts_engine as ts

from ai.eval.blunders import Blunder, BlunderCounts, _name, defcon_suicide_cards
from ai.eval.positions import legal_play_modes, step_to_play_mode

UN_INTERVENTION = 32

RULES = ("un_intervention_off_target", "un_intervention_on_spaceable")


def new_counts() -> BlunderCounts:
    """A counter that enumerates this probe's rules rather than `blunders.RULES`."""
    return BlunderCounts(rules=RULES)


def is_companion_node(state: ts.GameState) -> bool:
    """Is the engine asking which card to run through UN Intervention?"""
    ctx = state.ctx()
    return (ctx.decision_type == ts.DecisionType.SELECT_CARD
            and int(ctx.pending_op_card) == UN_INTERVENTION)


def legal_companions(mask: np.ndarray) -> List[int]:
    """Card ids selectable at a companion node, from the flat flat mask.

    The flat action that selects a card is `card_id - 1` (`positions.card_action`) — not the
    card id itself, which is the indexing the engine's internal 112-wide sub-mask uses.
    """
    return [int(i) + 1 for i in np.flatnonzero(np.asarray(mask)[:110])]


def spaceable(pre_play: ts.GameState, card_id: int) -> Optional[bool]:
    """Can `card_id` be put on the space race from this position?

    `None` when the question cannot be asked — the card is not selectable from the state that
    was kept, so the answer would be about a different position. Counted and reported rather
    than guessed: a probe that silently treats "unknown" as "not spaceable" would score the
    model on a rule the engine never confirmed.
    """
    try:
        at_mode = step_to_play_mode(pre_play, card_id)
    except ValueError:
        return None
    return "space" in legal_play_modes(at_mode)


def classify(state: ts.GameState, pre_play: Optional[ts.GameState], mask: np.ndarray,
             chosen_action: int, counts: BlunderCounts) -> None:
    """Record the opportunity and, if taken, the mistake, for one companion node."""
    ctx = state.ctx()
    player = ctx.decision_player
    if player == ts.Player.NONE:
        player = state.phasing_player
    side = "US" if player == ts.Player.US else "USSR"

    companions = legal_companions(mask)
    if len(companions) < 2:
        return                      # forced: one legal companion is not a choice
    banned = defcon_suicide_cards(state, player)
    in_hand = [c for c in companions if c in banned]
    if not in_hand:
        return                      # nothing in hand it must not play; no problem to solve

    chosen = int(chosen_action) + 1
    turn, ar = int(state.turn), int(state.action_round)

    counts.note_opportunity("un_intervention_off_target")
    if chosen not in banned:
        counts.note_blunder(Blunder(
            rule="un_intervention_off_target", player=side, card_id=chosen,
            card_name=_name(chosen), turn=turn, action_round=ar,
            detail=(f"UN Intervention spent on {_name(chosen)} while holding "
                    + ", ".join(_name(c) for c in in_hand) + " — cards it must not play"),
        ))

    # The sharper rule needs at least two banned candidates that differ in their exits.
    if pre_play is None or len(in_hand) < 2:
        return
    exits = {c: spaceable(pre_play, c) for c in in_hand}
    if any(v is None for v in exits.values()):
        counts.note_opportunity("spaceability_unknown")
        return
    if not (any(exits.values()) and not all(exits.values())):
        return                      # all spaceable or none: the choice carries no information

    counts.note_opportunity("un_intervention_on_spaceable")
    if exits.get(chosen) is True:
        other = [c for c, sp in exits.items() if sp is False]
        counts.note_blunder(Blunder(
            rule="un_intervention_on_spaceable", player=side, card_id=chosen,
            card_name=_name(chosen), turn=turn, action_round=ar,
            detail=(f"UN Intervention spent on {_name(chosen)}, which could have spaced "
                    f"itself, leaving " + ", ".join(_name(c) for c in other)
                    + " with one exit fewer"),
        ))


def measure_sequencing(model: Any, num_games: int = 500, base_seed: int = 830_000,
                       temperature: float = 0.1, max_iters: int = 20_000) -> BlunderCounts:
    """Drive self-play and score every UN Intervention companion choice.

    Batched for the same reason `measure_blunders_batched` is: the per-decision inspection is
    cheap and the forward pass is not.
    """
    import torch

    from bindings.ts_env import TsVectorizedEnv, check_obs_width

    check_obs_width(model)
    device = next(model.parameters()).device
    was_training = model.training
    model.eval()

    env = TsVectorizedEnv(num_envs=num_games, base_seed=base_seed)
    obs, masks, _ = env.reset_all()

    counts = new_counts()
    # Per env, the last ordinary card-selection node, with the turn and player it belonged to.
    # Spaceability is asked there, because UN Intervention's own play has consumed the action
    # round by the time the companion is chosen.
    pre_play: List[Optional[Tuple[ts.GameState, int, Any]]] = [None] * num_games
    done = [False] * num_games

    try:
        for _ in range(max_iters):
            if all(done):
                break
            obs_t = torch.from_numpy(np.asarray(obs, dtype=np.float32)).to(device)
            mask_t = torch.from_numpy(np.asarray(masks)).to(device)
            with torch.no_grad():
                actions, *_ = model.sample_action(obs_t, mask_t, temperature=temperature)
            acts = actions.cpu().numpy().astype(np.int64)

            for i in range(num_games):
                if done[i]:
                    continue
                state = env.runner.get_state(i)
                if ts.Engine.is_terminal(state):
                    done[i] = True
                    continue
                ctx = state.ctx()
                if ctx.decision_type != ts.DecisionType.SELECT_CARD:
                    continue
                player = ctx.decision_player
                if player == ts.Player.NONE:
                    player = state.phasing_player
                if is_companion_node(state):
                    kept = pre_play[i]
                    # Only usable if it is the same player's own turn -- otherwise it describes
                    # a different position and the spaceability answer would be about that one.
                    usable = (kept[0] if kept is not None
                              and kept[1] == int(state.turn) and kept[2] == player else None)
                    classify(state, usable, np.asarray(masks[i]), int(acts[i]), counts)
                elif int(ctx.pending_op_card) == 0:
                    pre_play[i] = (state.clone(), int(state.turn), player)

            obs, masks, *_ = env.step(acts)
    finally:
        if was_training:
            model.train()
    return counts


def metrics(counts: BlunderCounts, prefix: str = "sequencing") -> Dict[str, float]:
    return counts.metrics(prefix=prefix)


def format_report(counts: BlunderCounts, blunders: Optional[BlunderCounts] = None,
                  label: str = "") -> str:
    """The chain in one place: the tool misplaced, then the card played raw anyway."""
    head = f"=== sequencing probe{(' ' + label) if label else ''} ==="
    lines = [head, counts.summary()]
    unknown = counts.opportunities.get("spaceability_unknown", 0)
    if unknown:
        lines.append(f"  (spaceability unreadable at {unknown} node(s); excluded from the "
                     f"second rule rather than assumed)")
    if blunders is not None:
        lines.append("  and the outcome those two are upstream of:")
        rule = "defcon_suicide_with_alternative"
        c = blunders.committed.get(rule, 0)
        n = blunders.opportunities.get(rule, 0)
        lines.append(f"  {rule:34} {c:5}/{n:<6}"
                     + (f" ({100.0 * c / n:5.1f}%)" if n else " (no chances)"))
    for b in counts.examples[:5]:
        lines.append(f"    {b}")
    return "\n".join(lines)
