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


@dataclass
class RolloutResult:
    """Realized outcomes from playing both boards out, against what the critic predicted."""

    label: str
    human_wins: int = 0
    human_games: int = 0
    model_wins: int = 0
    model_games: int = 0
    predicted_human: float = 0.0
    predicted_model: float = 0.0

    @property
    def human_rate(self) -> float:
        return 100.0 * self.human_wins / max(1, self.human_games)

    @property
    def model_rate(self) -> float:
        return 100.0 * self.model_wins / max(1, self.model_games)

    @property
    def realized_gap(self) -> float:
        """Win-rate points the human's board is actually worth, played out by this policy."""
        return self.human_rate - self.model_rate

    @property
    def predicted_gap(self) -> float:
        """The same quantity as the critic predicted it, in win-rate points."""
        return 50.0 * (self.predicted_human - self.predicted_model)


def play_out(model: Any, states: Sequence[ts.GameState], mover: ts.Player, device: Any,
             seeds: int = 8, max_iters: int = 4000, temperature: float = 0.1) -> Tuple[int, int]:
    """Finish each position `seeds` times under the model's own policy; count wins for `mover`.

    This is the test the value head cannot self-report. `v_win` is an *on-policy* estimate: it
    predicts the outcome under the policy that will actually play the rest of the game. So a
    critic that scores a human's De-Stalinization spread below the model's own line may be wrong
    about the card -- or may be right that the spread is worth nothing to a policy that will not
    defend or build on it. Only playing both out separates the two.
    """
    import torch

    wins = games = 0
    for seed in range(seeds):
        runner = ts.VectorizedBatchRunner(len(states), 310000 + seed * 7919)
        for i, st in enumerate(states):
            runner.set_state(i, st)

        for _ in range(max_iters):
            terminals = runner.get_terminals()
            if all(terminals):
                break
            obs = np.asarray(runner.get_observations(), dtype=np.float32)
            masks = np.asarray(runner.get_action_masks())
            with torch.no_grad():
                logits = model(torch.from_numpy(obs).to(device),
                               torch.from_numpy(masks).to(device))[0]
                if temperature > 0.0:
                    picks = torch.multinomial(
                        torch.softmax(logits / temperature, dim=-1), 1).squeeze(-1).cpu().numpy()
                else:
                    picks = logits.argmax(dim=-1).cpu().numpy()
            runner.step_flat_all([int(a) for a in picks], auto_advance=True)

        for u in runner.get_terminal_utilities():
            u = float(u)
            if u == 0.0:
                continue                      # a draw counts for neither side
            games += 1
            if (u > 0) == (mover == ts.Player.US):
                wins += 1
    return wins, games


def rollout_check(model: Any, pairs: Sequence[RoundPair], device: Any, seeds: int = 8,
                  label: str = "") -> RolloutResult:
    """Was the critic right? Compare its ranking with what actually happens."""
    from ai.eval.battleground_value import value_of

    out = RolloutResult(label=label)
    human_states, model_states = [], []
    for pair in pairs:
        after_model = play_one_round(model, pair.before, pair.mover, device)
        if after_model is None:
            continue
        human_states.append(pair.after_human)
        model_states.append(after_model)
        out.predicted_human += value_of(model, pair.after_human, pair.mover, device)
        out.predicted_model += value_of(model, after_model, pair.mover, device)

    if not human_states:
        return out
    out.predicted_human /= len(human_states)
    out.predicted_model /= len(model_states)
    mover = pairs[0].mover
    out.human_wins, out.human_games = play_out(model, human_states, mover, device, seeds=seeds)
    out.model_wins, out.model_games = play_out(model, model_states, mover, device, seeds=seeds)
    return out


def play_out_per_position(model: Any, states: Sequence[ts.GameState], mover: ts.Player,
                          device: Any, seeds: int = 12, max_iters: int = 4000,
                          temperature: float = 0.1) -> np.ndarray:
    """Win rate for each position separately, so the pair can be differenced position by position.

    Pooling every rollout into one rate and quoting a binomial error over it is wrong: the twelve
    rollouts of one position are not twelve independent games, and the unit of replication is the
    position. Differencing the two arms per position also removes the position's own difficulty,
    which is the dominant source of variance here.
    """
    import torch

    wins = np.zeros(len(states), dtype=np.float64)
    games = np.zeros(len(states), dtype=np.float64)
    for seed in range(seeds):
        runner = ts.VectorizedBatchRunner(len(states), 310000 + seed * 7919)
        for i, st in enumerate(states):
            runner.set_state(i, st)

        for _ in range(max_iters):
            terminals = runner.get_terminals()
            if all(terminals):
                break
            obs = np.asarray(runner.get_observations(), dtype=np.float32)
            masks = np.asarray(runner.get_action_masks())
            with torch.no_grad():
                logits = model(torch.from_numpy(obs).to(device),
                               torch.from_numpy(masks).to(device))[0]
                if temperature > 0.0:
                    picks = torch.multinomial(
                        torch.softmax(logits / temperature, dim=-1), 1).squeeze(-1).cpu().numpy()
                else:
                    picks = logits.argmax(dim=-1).cpu().numpy()
            runner.step_flat_all([int(a) for a in picks], auto_advance=True)

        for i, u in enumerate(runner.get_terminal_utilities()):
            u = float(u)
            if u == 0.0:
                continue
            games[i] += 1
            if (u > 0) == (mover == ts.Player.US):
                wins[i] += 1
    return np.divide(wins, np.maximum(games, 1.0)) * 100.0


def paired_rollout(model: Any, pairs: Sequence[RoundPair], device: Any,
                   seeds: int = 12) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per-position win rates for the human board, the model board, and the critic's prediction."""
    from ai.eval.battleground_value import value_of

    human_states, model_states, predicted = [], [], []
    for pair in pairs:
        after_model = play_one_round(model, pair.before, pair.mover, device)
        if after_model is None:
            continue
        human_states.append(pair.after_human)
        model_states.append(after_model)
        predicted.append(50.0 * (value_of(model, pair.after_human, pair.mover, device)
                                 - value_of(model, after_model, pair.mover, device)))
    if not human_states:
        return np.zeros(0), np.zeros(0), np.zeros(0)
    mover = pairs[0].mover
    return (play_out_per_position(model, human_states, mover, device, seeds=seeds),
            play_out_per_position(model, model_states, mover, device, seeds=seeds),
            np.asarray(predicted))
