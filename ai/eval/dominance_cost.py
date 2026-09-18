"""Does spacing the dominated card actually cost the game?

§9.3 measured that humans almost never space their own or a neutral card while holding an equal-Ops
opponent recurring event -- 3 errors in 712 decidable plays -- against our agents' 12.6-18.1%. That
is a large behavioural gap, and the obvious inference is that closing it is worth something.

§4.4 is the reason not to trust that inference. Forced-win take rate looked like a 20-30% failure
rate and was worth about *two points* of win rate, because declining a forced win is usually free:
the declining player still won 82-95% of the time. A behavioural gap is not a cost until the cost
is measured.

**Method.** The fork has to be at card *selection*, not at play mode: by the time the engine asks
how to play a card, the card is already chosen, and the dominance error is about which card went to
the track. So positions are taken at `SELECT_CARD`, and the agent's own greedy continuation decides
whether the position counts -- it must actually go on to space its own or a neutral card while an
equal-Ops opponent recurring event sits in the same hand. The branches are then

  * **dominated** -- what the agent wanted: select that card, space it;
  * **dominant** -- select the opponent's card instead, space that;

continued from there by the same policy over several die streams, comparing how often the player who
chose goes on to win. The difference is what the error costs, in the only currency that matters.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, List, Optional, Sequence, Tuple

import numpy as np

import ts_engine as ts
from bindings.settle import SettleMode, settle

from ai.eval.dominance import space_dominance_alternatives
from bindings.action_encoder import ActionEncoder

SPACE_MODE = 2


@dataclass
class Branch:
    """One position, forked two ways."""

    dominated: ts.GameState
    dominant: ts.GameState
    mover: ts.Player
    spaced_card: int
    alternative: int


@dataclass
class CostResult:
    positions: int = 0
    dominated_wins: int = 0
    dominated_games: int = 0
    dominant_wins: int = 0
    dominant_games: int = 0
    notes: List[str] = field(default_factory=list)

    @property
    def dominated_rate(self) -> float:
        return 100.0 * self.dominated_wins / max(1, self.dominated_games)

    @property
    def dominant_rate(self) -> float:
        return 100.0 * self.dominant_wins / max(1, self.dominant_games)

    @property
    def cost(self) -> float:
        """Win-rate points given up by taking the dominated branch."""
        return self.dominant_rate - self.dominated_rate


def _greedy(model: Any, state: ts.GameState, mover: ts.Player, device: Any) -> int:
    import torch

    obs = np.asarray(ts.extract_observation(state, mover), dtype=np.float32)[None, :]
    mask = np.asarray(ActionEncoder.get_legal_mask(state), dtype=np.uint8)[None, :]
    with torch.no_grad():
        logits = model(torch.from_numpy(obs).to(device),
                       torch.from_numpy(mask).to(device))[0]
    return int(logits.argmax(dim=-1).item())


def _drain(state: ts.GameState) -> None:
    settle(state, SettleMode.CHANCE)


def _space_action(state: ts.GameState) -> Optional[int]:
    mask = np.asarray(ActionEncoder.get_legal_mask(state))
    for a in np.flatnonzero(mask):
        ma = ts.decode_flat_action(state, int(a))
        if ma.decision_type != ts.DecisionType.SELECT_PLAY_MODE:
            continue
        if int(ma.secondary_id) == SPACE_MODE or int(ma.primary_id) == SPACE_MODE:
            return int(a)
    return None


def _card_action(state: ts.GameState, card: int) -> Optional[int]:
    mask = np.asarray(ActionEncoder.get_legal_mask(state))
    for a in np.flatnonzero(mask):
        ma = ts.decode_flat_action(state, int(a))
        if ma.decision_type == ts.DecisionType.SELECT_CARD and int(ma.primary_id) == card:
            return int(a)
    return None


def _fork(model: Any, state: ts.GameState, mover: ts.Player, device: Any) -> Optional[Branch]:
    """Build both branches from a SELECT_CARD position, or None if it does not qualify."""
    chosen = _greedy(model, state, mover, device)
    ma = ts.decode_flat_action(state, chosen)
    if ma.decision_type != ts.DecisionType.SELECT_CARD:
        return None
    card = int(ma.primary_id)
    if not (1 <= card <= 110):
        return None

    # Does it actually go on to space that card?
    dominated = state.clone()
    ts.Engine.step_flat(dominated, chosen)
    _drain(dominated)
    if ts.Engine.is_terminal(dominated):
        return None
    space = _space_action(dominated)
    if space is None or _greedy(model, dominated, mover, device) != space:
        return None

    alts = space_dominance_alternatives(state, mover, card)
    if not alts:
        return None
    alternative = int(alts[0])

    # The dominant branch: the opponent's card to the track instead.
    dominant = state.clone()
    alt_action = _card_action(dominant, alternative)
    if alt_action is None:
        return None
    ts.Engine.step_flat(dominant, alt_action)
    _drain(dominant)
    if ts.Engine.is_terminal(dominant):
        return None
    alt_space = _space_action(dominant)
    if alt_space is None:
        return None

    ts.Engine.step_flat(dominated, space)
    _drain(dominated)
    ts.Engine.step_flat(dominant, alt_space)
    _drain(dominant)
    return Branch(dominated, dominant, mover, card, alternative)


def find_branches(model: Any, device: Any, num_envs: int = 128, base_seed: int = 990000,
                  max_iters: int = 3000, limit: int = 120) -> List[Branch]:
    """Play self-play and fork every position where the agent spaces a dominated card."""
    import torch

    runner = ts.VectorizedBatchRunner(num_envs, base_seed)
    out: List[Branch] = []
    model.eval()

    for _ in range(max_iters):
        if len(out) >= limit:
            break
        terminals = runner.get_terminals()
        if all(terminals):
            break
        obs = np.asarray(runner.get_observations(), dtype=np.float32)
        masks = np.asarray(runner.get_action_masks())
        with torch.no_grad():
            choice = model(torch.from_numpy(obs).to(device),
                           torch.from_numpy(masks).to(device))[0].argmax(dim=-1).cpu().numpy()

        # Taken at SELECT_CARD, not SELECT_PLAY_MODE: by the time the engine asks how to play a
        # card the card is already chosen, and which card went to the track is the decision.
        for i in range(num_envs):
            if terminals[i] or len(out) >= limit:
                continue
            state = runner.get_state(i)
            ctx = state.ctx()
            if ctx.decision_type != ts.DecisionType.SELECT_CARD:
                continue
            mover = ctx.decision_player
            if mover == ts.Player.NONE:
                mover = state.phasing_player
            branch = _fork(model, state.clone(), mover, device)
            if branch is not None:
                out.append(branch)

        runner.step_flat_all([int(c) for c in choice], auto_advance=True)

    return out


def play_out(model: Any, states: Sequence[ts.GameState], movers: Sequence[ts.Player],
             device: Any, seeds: int = 8, max_iters: int = 4000,
             temperature: float = 0.1) -> Tuple[int, int]:
    """Continue each position `seeds` times and count wins for the player who chose.

    Several die streams per position because one is a coin flip: the branch differs by a single
    card, and a single rollout would report the dice rather than the decision.
    """
    import torch

    wins = games = 0
    for seed in range(seeds):
        n = len(states)
        runner = ts.VectorizedBatchRunner(n, 700000 + seed * 9173)
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
                    probs = torch.softmax(logits / temperature, dim=-1)
                    picks = torch.multinomial(probs, 1).squeeze(-1).cpu().numpy()
                else:
                    picks = logits.argmax(dim=-1).cpu().numpy()
            runner.step_flat_all([int(a) for a in picks], auto_advance=True)

        utils = runner.get_terminal_utilities()
        for i, mover in enumerate(movers):
            u = float(utils[i])
            if u == 0.0:
                continue                      # a draw counts for neither side
            games += 1
            if (u > 0) == (mover == ts.Player.US):
                wins += 1
    return wins, games


def measure(model: Any, device: Any, limit: int = 120, seeds: int = 8) -> CostResult:
    """The whole gate: find the positions, play both branches, compare."""
    branches = find_branches(model, device, limit=limit)
    out = CostResult(positions=len(branches))
    if not branches:
        out.notes.append("no dominated space plays found")
        return out

    movers = [b.mover for b in branches]
    out.dominated_wins, out.dominated_games = play_out(
        model, [b.dominated for b in branches], movers, device, seeds=seeds)
    out.dominant_wins, out.dominant_games = play_out(
        model, [b.dominant for b in branches], movers, device, seeds=seeds)
    return out
