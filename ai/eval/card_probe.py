"""Put a specific card in a specific hand at a real human decision, and read the policy back.

The battleground diagnostic in §12 perturbs the *board*. This perturbs the *hand*, which is where
the questions with a known right answer live. De-Stalinization and Decolonization are the clean
cases: both are USSR events that place Influence far from Europe, so for the USSR on turn 2 they
are ordinary plays, and for the US they are cards whose event helps the opponent -- to be held past
the turn if that can be afforded, spaced if not, and never played for Ops early where the event
fires for the other side.

Positions come from the human corpus rather than self-play, so the question is asked where a person
actually faced it. `convert_game`'s `on_decision` seam hands over the `GameState` as the human saw
it; from there the hand can be edited and the same node re-asked of either side.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

import ts_engine as ts

DE_STALINIZATION = 33
DECOLONIZATION = 30

# ts::PlayMode, from engine/include/ts/types.hpp:112.
MODE_NAMES = {0: "event", 1: "ops", 2: "space", 3: "pass"}


@dataclass
class Node:
    """One human decision, kept whole so the hand can be edited and the node re-asked."""

    state: ts.GameState
    mover: ts.Player
    turn: int
    phase: str
    card: str
    chosen: int


def capture_nodes(game: Dict, turn: int,
                  decision: ts.DecisionType = ts.DecisionType.SELECT_CARD,
                  action_round_only: bool = True) -> List[Node]:
    """Every decision of one kind on one turn of one human game.

    Action Rounds only by default. A headline is also a `SELECT_CARD`, but neither of the answers
    this module is asking about exists there: you cannot space a headline and you cannot decline to
    commit one, so a headline node cannot tell you whether a side would hold an opponent's card.
    """
    from tools.lib.ts_replayer_convert import convert_game

    out: List[Node] = []

    def hook(state: ts.GameState, mover: ts.Player, entry: Any, chosen: int) -> None:
        if int(entry.turn) != turn or state.ctx().decision_type != decision:
            return
        if action_round_only and state.current_phase != ts.Phase.ACTION_ROUND:
            return
        out.append(Node(state, mover, int(entry.turn), str(entry.phase),
                        str(entry.card or ""), chosen))

    convert_game(game, on_decision=hook)
    return out


def hand(state: ts.GameState, player: ts.Player) -> List[int]:
    want = ts.CardLocation.HAND_US if player == ts.Player.US else ts.CardLocation.HAND_USSR
    return [c for c in range(1, 111) if state.get_card_location(c) == want]


def card_name(cid: int) -> str:
    return str(ts.CardData.get_card_info(cid)["name"])


def give_card(state: ts.GameState, player: ts.Player, card: int,
              drop: Optional[int] = None) -> ts.GameState:
    """A copy of `state` with `card` in `player`'s hand, swapped for one they already hold.

    Swapped rather than added, so the hand size stays what the rules make it. Hand size is in the
    observation and it drives how many Action Rounds remain, so a hand one card too long is a
    different position, not the same position with an extra card.
    """
    probe = state.clone()
    if drop is None:
        # Never a scoring card: those must be played before the turn ends, so removing one
        # changes what the rest of the hand is under pressure to do.
        held = [c for c in hand(state, player)
                if c != card and not bool(ts.CardData.get_card_info(c)["is_scoring"])]
        # Drop their highest-Ops card, so the swap cannot be waved away as having handed them a
        # weaker hand: whatever they do with the new card, they gave up their best Ops to hold it.
        drop = max(held, key=lambda c: int(ts.CardData.get_card_info(c)["ops"])) if held else None
    if drop is not None:
        probe.set_card_location(drop, ts.CardLocation.DISCARD_PILE)
    probe.set_card_location(
        card, ts.CardLocation.HAND_US if player == ts.Player.US else ts.CardLocation.HAND_USSR)
    return probe


def policy(model: Any, state: ts.GameState, mover: ts.Player, device: Any) -> np.ndarray:
    import torch

    obs = np.asarray(ts.extract_observation(state, mover), dtype=np.float32)[None, :]
    mask = np.asarray(ts.get_flat_action_mask(state), dtype=np.uint8)[None, :]
    with torch.no_grad():
        logits = model(torch.from_numpy(obs).to(device), torch.from_numpy(mask).to(device))[0]
        probs = torch.softmax(logits, dim=-1).squeeze(0).cpu().numpy()
    return np.asarray(probs) * np.asarray(mask[0], dtype=np.float64)


def describe(state: ts.GameState, action: int) -> str:
    ma = ts.decode_flat_action(state, int(action))
    dt = ma.decision_type
    if dt == ts.DecisionType.SELECT_CARD:
        cid = int(ma.primary_id)
        if 1 <= cid <= 110:
            ops = int(ts.CardData.get_card_info(cid)["ops"])
            return f"play {card_name(cid)} (#{cid}, {ops} ops)"
        return f"select card #{cid}"
    if dt == ts.DecisionType.SELECT_PLAY_MODE:
        mode = int(ma.secondary_id) or int(ma.primary_id)
        return f"as {MODE_NAMES.get(mode, mode)}"
    return f"{dt}"


def top_actions(model: Any, state: ts.GameState, mover: ts.Player, device: Any,
                k: int = 5) -> List[Tuple[float, int, str]]:
    probs = policy(model, state, mover, device)
    order = np.argsort(-probs)[:k]
    return [(float(probs[a]), int(a), describe(state, int(a))) for a in order if probs[a] > 0]


def card_action(state: ts.GameState, card: int) -> Optional[int]:
    mask = np.asarray(ts.get_flat_action_mask(state))
    for a in np.flatnonzero(mask):
        ma = ts.decode_flat_action(state, int(a))
        if ma.decision_type == ts.DecisionType.SELECT_CARD and int(ma.primary_id) == card:
            return int(a)
    return None


def _drain(state: ts.GameState) -> None:
    while (not ts.Engine.is_terminal(state)
           and state.ctx().decision_player == ts.Player.NONE
           and state.ctx().decision_type == ts.DecisionType.ROLL_DIE):
        ts.Engine.step(state, ts.MicroAction(ts.DecisionType.ROLL_DIE, 0, 0, 0))


def mode_preference(model: Any, state: ts.GameState, mover: ts.Player, device: Any,
                    card: int) -> Optional[List[Tuple[float, int, str]]]:
    """Given that this card is played, how does the policy want to play it?

    Read by forcing the card selection and asking at the mode node, because the two are separate
    decisions: what the top-level distribution says about *whether* to play a card says nothing
    about whether the event would be fired for the opponent.
    """
    action = card_action(state, card)
    if action is None:
        return None
    probe = state.clone()
    ts.Engine.step_flat(probe, action)
    _drain(probe)
    if ts.Engine.is_terminal(probe):
        return None
    if probe.ctx().decision_type != ts.DecisionType.SELECT_PLAY_MODE:
        return None
    return top_actions(model, probe, mover, device, k=4)


@dataclass
class ProbeResult:
    label: str
    node: Node
    played_card: List[Tuple[float, int, str]] = field(default_factory=list)
    modes: Optional[List[Tuple[float, int, str]]] = None
    hand_ids: List[int] = field(default_factory=list)
    card_rank: Optional[int] = None
    card_prob: float = 0.0
    # How many cards were selectable at this node. Not the hand size: The China Card is playable
    # without sitting in either hand, so ranking out of len(hand) can report "8 of 7".
    playable: int = 0


def probe(model: Any, node: Node, card: int, device: Any, mover: Optional[ts.Player] = None,
          label: str = "") -> ProbeResult:
    """What the policy does at `node` with `card` in the mover's hand."""
    who = mover if mover is not None else node.mover
    state = node.state if card in hand(node.state, who) else give_card(node.state, who, card)
    out = ProbeResult(label=label or card_name(card), node=node)
    out.hand_ids = hand(state, who)
    out.played_card = top_actions(model, state, who, device, k=5)
    out.modes = mode_preference(model, state, who, device, card)

    probs = policy(model, state, who, device)
    action = card_action(state, card)
    if action is not None:
        out.card_prob = float(probs[action])
        # Rank among the cards it could play, not among all 212 actions.
        cards = [a for a in np.flatnonzero(np.asarray(ts.get_flat_action_mask(state)))
                 if ts.decode_flat_action(state, int(a)).decision_type
                 == ts.DecisionType.SELECT_CARD]
        out.playable = len(cards)
        out.card_rank = 1 + sum(1 for a in cards if probs[a] > probs[action])
    return out


def card_fate(model: Any, state: ts.GameState, who: ts.Player, card: int, device: Any,
              max_steps: int = 400) -> str:
    """Play the rest of the turn greedily and report what became of `card`.

    "Hold it" is not an action the engine offers, so it cannot be read off one node's
    distribution: holding a card means reaching the end of the turn without ever having selected
    it. That takes a rollout. Both sides move under the same policy, which is what the agent would
    actually face.

    Returns one of: `held`, `event`, `ops`, `space`, `pass`, `discarded`, or `terminal`.
    """
    import torch

    probe = state.clone()
    turn = int(probe.turn)
    watching = False

    for _ in range(max_steps):
        _drain(probe)
        if ts.Engine.is_terminal(probe):
            return "terminal"
        if int(probe.turn) != turn:
            break

        ctx = probe.ctx()
        mover = ctx.decision_player
        if mover == ts.Player.NONE:
            mover = probe.phasing_player
        if mover == ts.Player.NONE:
            break

        probs = policy(model, probe, mover, device)
        if not probs.any():
            break
        action = int(np.argmax(probs))

        # The step after our card is selected is the one that says how it was played.
        if watching and ctx.decision_type == ts.DecisionType.SELECT_PLAY_MODE:
            ma = ts.decode_flat_action(probe, action)
            mode = int(ma.secondary_id) or int(ma.primary_id)
            return MODE_NAMES.get(mode, str(mode))

        ma = ts.decode_flat_action(probe, action)
        if (ma.decision_type == ts.DecisionType.SELECT_CARD and int(ma.primary_id) == card
                and mover == who):
            watching = True

        ts.Engine.step_flat(probe, action)

    want = ts.CardLocation.HAND_US if who == ts.Player.US else ts.CardLocation.HAND_USSR
    if probe.get_card_location(card) == want:
        return "held"
    if watching:
        return "played, mode unread"
    return "discarded"
