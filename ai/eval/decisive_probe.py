"""Measures whether a policy takes forced wins and avoids avoidable forced losses.

Win rate barely registers these: a decisive choice arises at roughly 0.6% of decisions, so
a policy can throw away a quarter of them and still look fine. But each one is worth a
whole game, which makes these the most sensitive quality signals available -- and unlike
the behavioural suite's constructed positions, they are measured on states the policy
actually reaches.

Two exclusions matter for the numbers to mean anything, both learned by getting them wrong:

* **Chance nodes.** At ROLL_DIE no choice is being made, so counting the outcome against
  the policy is simply wrong.
* **Already-lost positions.** Where every legal action loses, taking one is not a blunder.
  A held scoring card that must eventually be played is the common case.
"""

from dataclasses import dataclass
from typing import Any, Callable, List, Optional

import numpy as np
import ts_engine as ts

from ai.eval.safety import classify_legal_actions

# (state, player) -> chosen flat action
ActionFn = Callable[[ts.GameState, ts.Player], int]


@dataclass
class DecisiveStats:
    episodes: int = 0
    decisions: int = 0
    win_available: int = 0
    win_taken: int = 0
    loss_avoidable: int = 0
    loss_taken: int = 0
    loss_forced: int = 0

    @property
    def win_take_rate(self) -> float:
        return self.win_taken / self.win_available if self.win_available else float("nan")

    @property
    def loss_avoid_rate(self) -> float:
        if not self.loss_avoidable:
            return float("nan")
        return 1.0 - (self.loss_taken / self.loss_avoidable)

    def as_metrics(self) -> dict:
        return {
            "decisive_win_take_rate": self.win_take_rate,
            "decisive_loss_avoid_rate": self.loss_avoid_rate,
            "decisive_win_available": float(self.win_available),
            "decisive_loss_avoidable": float(self.loss_avoidable),
        }


def measure_decisive(
    action_fn: ActionFn,
    num_games: int = 40,
    base_seed: int = 77000,
    max_steps: int = 1500,
) -> DecisiveStats:
    """Plays self-play games and records how the policy handles decisive positions."""
    st_out = DecisiveStats()
    for g in range(num_games):
        state = ts.GameState()
        ts.Engine.init_game(state, base_seed + g)
        steps = 0
        while not ts.Engine.is_terminal(state) and steps < max_steps:
            ctx = state.ctx()
            player = ctx.decision_player if ctx.decision_player != ts.Player.NONE else state.phasing_player
            is_chance = ctx.decision_type == ts.DecisionType.ROLL_DIE
            kinds = {} if is_chance else classify_legal_actions(state, player)
            action = int(action_fn(state, player))

            if kinds:
                st_out.decisions += 1
                wins = [a for a, k in kinds.items() if k == "win"]
                losses = [a for a, k in kinds.items() if k == "loss"]
                if wins:
                    st_out.win_available += 1
                    if kinds.get(action) == "win":
                        st_out.win_taken += 1
                if losses:
                    if len(losses) < len(kinds):
                        st_out.loss_avoidable += 1
                        if kinds.get(action) == "loss":
                            st_out.loss_taken += 1
                    else:
                        st_out.loss_forced += 1

            try:
                ts.Engine.step_flat(state, action)
            except Exception:
                break
            # Resolve the chance nodes an action lands on, exactly as the vectorized
            # runner does internally. Feeding them policy actions instead is not
            # equivalent: it leaves a different phasing_player and collapses games to
            # roughly half their true length, which silently truncated every decisive
            # measurement taken through this path.
            while (not ts.Engine.is_terminal(state)
                   and state.ctx().decision_player == ts.Player.NONE
                   and state.ctx().decision_type == ts.DecisionType.ROLL_DIE):
                ts.Engine.step(state, ts.MicroAction(ts.DecisionType.ROLL_DIE, 0, 0, 0))
            steps += 1
    return st_out

def measure_decisive_batched(
    model: Any,
    num_envs: int = 128,
    base_seed: int = 77_000,
    temperature: float = 0.1,
    max_iters: int = 20_000,
) -> DecisiveStats:
    """measure_decisive over parallel environments, with one forward pass per batch.

    Nearly all of the single-state cost is the network, not the probing: over a full game
    the split is 96.9% policy forward, 3.0% classify_legal_actions, 0.1% engine step. So
    handing the GPU one state at a time was the whole expense -- 890 decisions/sec against
    ~106,000 for 512 batched envs, while training itself runs at ~7,900 env-steps/sec.

    Exactly one episode is measured per environment -- the first -- and the loop runs until
    every env has finished it, so the sample is num_envs games and cannot be selected by
    length.

    An earlier version buffered decisions per env and folded them in on completion, which
    fixed attribution, but still stopped once num_episodes episodes had finished with
    num_episodes well below num_envs. That made the sample the fastest N of num_envs. Its
    own docstring named the reason this matters and kept the bug anyway: decisive positions
    cluster near the end of a game, and the games that run long are exactly the ones still
    in flight when the budget expires. The position profiler had the identical defect.
    """
    import numpy as np
    import torch

    from bindings.ts_env import TsVectorizedEnv

    device = next(model.parameters()).device
    was_training = model.training
    model.eval()

    env = TsVectorizedEnv(num_envs=num_envs, base_seed=base_seed)
    obs, masks, _ = env.reset_all()

    out = DecisiveStats()
    pending: List[List[tuple]] = [[] for _ in range(num_envs)]
    counted = [False] * num_envs
    episodes = 0

    def flush(i: int) -> None:
        nonlocal episodes
        for (n_kinds, n_wins, n_losses, chosen_kind) in pending[i]:
            out.decisions += 1
            if n_wins:
                out.win_available += 1
                if chosen_kind == "win":
                    out.win_taken += 1
            if n_losses:
                if n_losses < n_kinds:
                    out.loss_avoidable += 1
                    if chosen_kind == "loss":
                        out.loss_taken += 1
                else:
                    out.loss_forced += 1
        pending[i] = []
        episodes += 1
        out.episodes += 1

    try:
        for _ in range(max_iters):
            if all(counted):
                break

            obs_t = torch.from_numpy(np.asarray(obs, dtype=np.float32)).to(device)
            mask_t = torch.from_numpy(np.asarray(masks)).to(device)
            with torch.no_grad():
                actions_t, _, _, _, _ = model.sample_action(
                    obs_t, mask_t, temperature=temperature)
            actions = actions_t.cpu().numpy()

            for i in range(num_envs):
                if counted[i]:
                    continue
                state = env.runner.get_state(i)
                if ts.Engine.is_terminal(state):
                    continue
                ctx = state.ctx()
                if ctx.decision_type == ts.DecisionType.ROLL_DIE:
                    continue  # no choice is being made at a chance node
                player = (ctx.decision_player if ctx.decision_player != ts.Player.NONE
                          else state.phasing_player)
                kinds = classify_legal_actions(state, player)
                if not kinds:
                    continue
                n_wins = sum(1 for k in kinds.values() if k == "win")
                n_losses = sum(1 for k in kinds.values() if k == "loss")
                pending[i].append((len(kinds), n_wins, n_losses, kinds.get(int(actions[i]))))

            obs, masks, _, dones, _ = env.step(actions)
            for i, done in enumerate(dones):
                if done and not counted[i]:
                    flush(i)
                    counted[i] = True
    finally:
        if was_training:
            model.train()

    return out

