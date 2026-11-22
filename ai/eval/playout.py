"""The probes' one batched self-play loop, and the view every probe builds a model's inputs in.

Before this module some twenty probes each carried their own copy of the same loop -- a runner or a
`TsVectorizedEnv`, `get_observations`, a forward pass, `step_flat_all` -- and almost every copy fed
the network the *base* observation in the *E4* action view. A model's observation feature set is
part of what it reads: `HEADLINE_REVEAL` (the E8 default) changes the content of the base card rows
at no width, so a featured model fed the base layout loads, runs and silently misreads
(bindings/ts_env.py `model_obs_features`). Likewise an E4.1 (merged-influence) network reads the E4
mask wrongly. Everything here takes the view explicitly (`bindings.ts_env.AgentView`).

* `model_view(model, merged)` -- the view a bare network decides in (its features from its weights,
  its action view from the caller, who read it from the run directory: `ai.eval.loading`).
* `make_env` / `make_runner` -- a batch whose every env and side is in that view, set BEFORE the
  first reset so no cached observation or mask is ever built in another one.
* `Sampler` -- the policy's move over a batch, seeded: a probe that samples is reproducible from
  its seed, which `model.sample_action` (torch's global generator) was not.
* `play_out` -- finish given positions under the policy, several dice streams each, and count wins
  for a side, per position. A game still running at `max_iters` raises `GamesUnfinished`: counting
  it as a draw (what every copy did) hides a stuck policy inside a win rate.
"""

from __future__ import annotations

from typing import Any, List, Optional, Sequence

import numpy as np
import torch

import ts_engine as ts
from bindings.ts_env import AgentView, TsVectorizedEnv, check_obs_width, model_obs_features

_UINT64 = 1 << 64


class GamesUnfinished(RuntimeError):
    """Games were still running when a probe's iteration budget ran out."""


def model_view(model: Any, merged_influence: bool) -> AgentView:
    """The view a bare network decides in: its own observation feature set, and the action view the
    caller read from its run directory (`tools.lib.action_view.checkpoint_merged_influence`)."""
    check_obs_width(model)
    if bool(getattr(model, "macro_influence", False)):
        # Fix verification S-9.6 (2026-11-22): AgentView has no macro view; every env here masks in E4 or merged.
        raise ValueError("a macro-view network (--macro-influence, 304 slots) is not supported by the probes' "
                         "playout: its envs mask in the E4 or merged view")
    return AgentView(model_obs_features(model), bool(merged_influence))


def make_env(num_envs: int, base_seed: int, view: AgentView, **kwargs: Any) -> TsVectorizedEnv:
    """A `TsVectorizedEnv` with both sides of every env in `view`. Call `reset_all()` after: the
    view is set first, so the observations and masks the reset builds are already in it."""
    env = TsVectorizedEnv(num_envs=num_envs, base_seed=base_seed, **kwargs)
    env.set_obs_features(view.features, view.features)
    env.set_merged_influence(view.merged_influence, view.merged_influence)
    return env


def make_runner(num_envs: int, seed: int, view: AgentView) -> "ts.VectorizedBatchRunner":
    """A bare `VectorizedBatchRunner` with both sides of every env in `view`."""
    runner = ts.VectorizedBatchRunner(num_envs, seed)
    runner.set_obs_features([int(view.features)] * num_envs, [int(view.features)] * num_envs)
    runner.set_merged_influence([bool(view.merged_influence)] * num_envs,
                                [bool(view.merged_influence)] * num_envs)
    return runner


def load_states(runner: "ts.VectorizedBatchRunner", states: Sequence["ts.GameState"],
                reseed: Optional[Sequence[int]] = None) -> None:
    """Put `states` into the runner's envs (clones; with `reseed`, each with that rng_state) and
    rebuild the cached observations and masks, which `set_state` alone leaves describing the
    runner's own discarded deal."""
    for i, st in enumerate(states):
        s = st.clone()
        if reseed is not None:
            s.rng_state = int(reseed[i]) % _UINT64
        runner.set_state(i, s)
    runner.refresh_all()


def check_finished(terminals: Sequence[bool] | np.ndarray, max_iters: int, what: str) -> None:
    """Raise `GamesUnfinished` if any game is still running after a probe's whole budget."""
    running = int(np.size(terminals) - np.count_nonzero(np.asarray(terminals, dtype=bool)))
    if running:
        raise GamesUnfinished(
            f"{what}: {running} of {np.size(terminals)} games still running after {max_iters} "
            f"iterations -- a stuck policy or too small a budget; refusing to count them as draws")


class Sampler:
    """The policy's move over a batch of observations and masks, from its own seeded generator.

    `temperature <= 0` is the argmax. Rows are truncated to the model's width (a batch may be wider
    when another model's set is in it). A row with no legal action raises -- a softmax over an all
    -1e9 row samples uniformly, i.e. an illegal move the engine then refuses somewhere else --
    unless the caller marks it `terminal` (a bare runner keeps finished games, with empty masks,
    and ignores the action it is handed for them).
    """

    def __init__(self, model: Any, temperature: float, seed: int) -> None:
        self.model = model
        self.device = next(model.parameters()).device
        self.width = check_obs_width(model)
        self.temperature = float(temperature)
        self.generator = torch.Generator(device=self.device)
        self.generator.manual_seed(int(seed) % (1 << 63))

    def forward(self, obs: np.ndarray, masks: np.ndarray, terminal: Optional[Sequence[bool] | np.ndarray] = None
                ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """(masked logits, v_win, v_vp) of the batch."""
        o = np.ascontiguousarray(np.asarray(obs, dtype=np.float32)[:, :self.width])
        m = np.asarray(masks)
        empty = ~m.any(axis=1)
        if terminal is not None:
            empty &= ~np.asarray(terminal, dtype=bool)
        if empty.any():
            raise ValueError(f"{int(empty.sum())} live rows have no legal action")
        with torch.no_grad():
            logits, v_win, v_vp = self.model(torch.from_numpy(o).to(self.device),
                                             torch.from_numpy(np.ascontiguousarray(m)).to(self.device))[:3]
        return logits, v_win, v_vp

    def pick(self, logits: torch.Tensor) -> np.ndarray:
        """Actions from masked logits, at this sampler's temperature."""
        with torch.no_grad():
            if self.temperature <= 0.0:
                return logits.argmax(dim=-1).cpu().numpy().astype(np.int64)
            probs = torch.softmax(logits / max(self.temperature, 1e-4), dim=-1)
            return torch.multinomial(probs, 1, generator=self.generator).squeeze(-1).cpu().numpy().astype(np.int64)

    def __call__(self, obs: np.ndarray, masks: np.ndarray,
                 terminal: Optional[Sequence[bool] | np.ndarray] = None) -> np.ndarray:
        return self.pick(self.forward(obs, masks, terminal)[0])


def play_out(model: Any, states: Sequence["ts.GameState"], movers: Sequence["ts.Player"],
             view: AgentView, *, seeds: int, base_seed: int, temperature: float = 0.1,
             max_iters: int = 4000) -> tuple[np.ndarray, np.ndarray]:
    """Finish each position `seeds` times under the policy; per-position (wins, decided games) for
    the side in `movers`. A draw counts for neither side.

    Each repetition gives every position its own dice stream (`rng_state` from `base_seed`, the
    repetition and the position's index), so the repetitions are not the same game replayed -- a
    stored state carries its rng_state, and `set_state` keeps it -- while position i gets the same
    stream in two calls with the same `base_seed`: two branches of one fork are compared on common
    dice. Raises `GamesUnfinished` if a game outlives `max_iters`.
    """
    n = len(states)
    wins = np.zeros(n, dtype=np.int64)
    games = np.zeros(n, dtype=np.int64)
    if n == 0:
        return wins, games
    us = np.array([m == ts.Player.US for m in movers], dtype=bool)
    for rep in range(seeds):
        rng = np.random.default_rng([int(base_seed), rep])
        runner = make_runner(n, int(base_seed) + rep, view)
        load_states(runner, states, reseed=[int(x) for x in rng.integers(1, 1 << 63, size=n)])
        sampler = Sampler(model, temperature, int(rng.integers(1, 1 << 62)))
        for _ in range(max_iters):
            terminals = np.asarray(runner.get_terminals(), dtype=bool)
            if terminals.all():
                break
            acts = sampler(np.asarray(runner.get_observations()), np.asarray(runner.get_action_masks()),
                           terminals)
            runner.step_flat_all([int(a) for a in acts], auto_advance=True)
        check_finished(runner.get_terminals(), max_iters, "play_out")
        utils = np.asarray(runner.get_terminal_utilities(), dtype=np.float64)
        decided = utils != 0.0
        games += decided
        wins += decided & ((utils > 0) == us)
    return wins, games


def win_rates(wins: np.ndarray, games: np.ndarray) -> np.ndarray:
    """Per-position win rate in points (0 where nothing was decided)."""
    return np.divide(wins.astype(np.float64), np.maximum(games, 1).astype(np.float64)) * 100.0


def decision_players(states: Sequence["ts.GameState"]) -> List["ts.Player"]:
    """Who decides at each state: the decision player, else the phasing player."""
    out: List[ts.Player] = []
    for st in states:
        p = st.ctx().decision_player
        out.append(p if p != ts.Player.NONE else st.phasing_player)
    return out
