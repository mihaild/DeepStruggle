"""First-play urgency and the Gumbel root for play.

FPU must mean the same in both trees (the C++ tree is held to the Python one), leave the search
unchanged at 0, and keep the budget on fewer moves as it grows. The Gumbel root must return legal
moves, take the network's top moves deterministically at scale 0 with k = 1 (so one candidate is the
argmax), never evaluate more positions than PUCT at the same budget, and be what the gumbel: agent
spec plays -- in a tournament and in play_match alike.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, List

import numpy as np
import pytest
import torch

import ts_engine as ts
from ai.models.coldwar_net_v2 import ColdWarNetV2, create_coldwar_net_v2
from ai.search.batched_mcts import BatchedMCTS, BatchedMCTSAgent, BatchedMCTSConfig
from ai.search.gumbel_root import halving_phases, phase_share, world_shares
from bindings.action_encoder import ActionEncoder
from tools.lib.player_agent import load_agent, search_spec_config
from tools.play_match import _SearchBot, resolve_agent


def _model() -> ColdWarNetV2:
    torch.manual_seed(0)
    m = create_coldwar_net_v2("cpu")
    m.eval()
    return m


def _states(n: int = 24, advance: int = 60) -> List[ts.GameState]:
    runner = ts.VectorizedBatchRunner(n, 4242)
    for _ in range(advance):
        masks = np.asarray(runner.get_action_masks())
        runner.step_flat_all([int(np.flatnonzero(m)[0]) if m.any() else 211 for m in masks], True)
    return [runner.get_state(i) for i in range(n) if not ts.Engine.is_terminal(runner.get_state(i))]


class _CountingModel(torch.nn.Module):
    """The model, counting the positions it evaluates."""

    def __init__(self, inner: ColdWarNetV2) -> None:
        super().__init__()
        self.inner = inner
        self.rows = 0
        self.obs_feature_bits = getattr(inner, "obs_feature_bits", 0)

    def forward(self, obs: torch.Tensor, masks: torch.Tensor) -> Any:
        self.rows += int(obs.shape[0])
        return self.inner(obs, masks)


@pytest.mark.parametrize("fpu", [0.2, 1.0])
def test_fpu_means_the_same_in_both_trees(fpu: float) -> None:
    states = _states()
    roots = {}
    for backend in ("python", "cpp"):
        cfg = BatchedMCTSConfig(simulations=16, backend=backend, seed=3, fpu_reduction=fpu)
        roots[backend] = BatchedMCTS(_model(), device="cpu", config=cfg,
                                     featurise_capacity=256)._search(states)
    for a, b in zip(roots["python"], roots["cpp"]):
        assert a is not None and b is not None and a.n == b.n


def test_a_larger_fpu_spreads_the_budget_over_fewer_moves() -> None:
    states = _states()
    tried = []
    for fpu in (0.0, 1.0):
        cfg = BatchedMCTSConfig(simulations=32, seed=3, fpu_reduction=fpu)
        roots = BatchedMCTS(_model(), device="cpu", config=cfg, featurise_capacity=256)._search(states)
        tried.append(sum(sum(1 for x in r.n if x > 0) for r in roots if r is not None))
    assert tried[1] < tried[0]


def test_the_halving_schedule_never_overspends() -> None:
    for sims in (1, 2, 4, 8, 16, 33, 64, 256):
        for k in (2, 3, 4, 5, 8, 16):
            m = min(k, sims)
            if m < 2:
                continue
            phases, alive, left = halving_phases(m), m, sims
            for ph in range(phases):
                per = phase_share(sims, phases, alive, left)
                if per == 0:
                    break
                left -= per * alive
                alive = 1 if ph == phases - 1 else (alive + 1) // 2
            assert left >= 0, (sims, k)


@pytest.mark.parametrize("backend", ["cpp", "python"])
@pytest.mark.parametrize("sims,k", [(16, 4), (64, 4), (8, 16), (16, 16), (32, 8)])
def test_the_gumbel_root_evaluates_no_more_positions_than_puct(backend: str, sims: int,
                                                               k: int) -> None:
    """The budget is network evaluations: a Gumbel root at `sims` spends at most what PUCT at
    `sims` does (`sims` per position plus the root's own), and every move it returns is legal."""
    states = _states()
    spent = {}
    for name, gk in (("gumbel", k), ("puct", 0)):
        model = _CountingModel(_model())
        cfg = BatchedMCTSConfig(simulations=sims, determinize=True, gumbel_k=gk, gumbel_scale=0.0,
                                fpu_reduction=0.2, backend=backend, seed=5)
        picks = BatchedMCTS(model, device="cpu", config=cfg,
                            featurise_capacity=256).best_actions(states)
        for st, a in zip(states, picks):
            assert np.asarray(ActionEncoder.get_legal_mask(st))[a]
        spent[name] = model.rows
    assert spent["puct"] == len(states) * (sims + 1)
    assert 0 < spent["gumbel"] <= spent["puct"]


def test_the_gumbel_root_is_reproducible_from_its_seed() -> None:
    states = _states()
    cfg = BatchedMCTSConfig(simulations=16, determinize=True, gumbel_k=4, gumbel_scale=1.0, seed=7)
    a = BatchedMCTS(_model(), device="cpu", config=cfg, featurise_capacity=256)
    b = BatchedMCTS(_model(), device="cpu", config=cfg, featurise_capacity=256)
    assert a.best_actions(states) == b.best_actions(states)


def test_one_candidate_without_noise_is_the_networks_top_move() -> None:
    states = _states()
    cfg = BatchedMCTSConfig(simulations=8, gumbel_k=1, gumbel_scale=0.0)
    picks = BatchedMCTS(_model(), device="cpu", config=cfg, featurise_capacity=256).best_actions(states)
    agent = BatchedMCTSAgent(_model(), device="cpu", config=BatchedMCTSConfig(), featurise_capacity=256)
    assert picks == agent._policy_actions(states)


def test_the_gumbel_spec_plays_the_measured_configuration(tmp_path: Path) -> None:
    path = tmp_path / "m.pt"
    torch.save(_model().state_dict(), str(path))
    agent = load_agent(f"gumbel:{path}", device="cpu")
    assert isinstance(agent, BatchedMCTSAgent)
    cfg = agent.mcts.cfg
    assert (cfg.simulations, cfg.gumbel_k, cfg.gumbel_scale, cfg.fpu_reduction) == (256, 8, 0.0, 0.2)
    assert cfg.determinize and cfg.node_filter == "all" and not cfg.advance_root
    _, cfg, label = search_spec_config(f"gumbel:{path}:64:4")
    assert (cfg.simulations, cfg.gumbel_k, cfg.fpu_reduction, label) == (64, 4, 0.2, "gumbel64-k4")
    _, cfg, label = search_spec_config(f"gumbel:{path}:64:4:0")
    assert (cfg.fpu_reduction, label) == (0.0, "gumbel64-k4-fpu0")
    # Searching only card and play-mode decisions, the agent's own policy elsewhere.
    _, cfg, label = search_spec_config(f"gumbel:{path}:256:8:0.2:card")
    assert (cfg.node_filter, cfg.gumbel_k, label) == ("card_playmode", 8, "gumbel256-k8-card")


def test_a_card_only_gumbel_agent_plays_its_own_policy_off_card_decisions() -> None:
    """At a decision the filter skips, the move is the network's greedy pick, not a searched one."""
    agent = BatchedMCTSAgent(_model(), device="cpu", featurise_capacity=256,
                             config=BatchedMCTSConfig(simulations=16, temperature=0.0,
                                                      determinize=True, node_filter="card_playmode",
                                                      gumbel_k=4, gumbel_scale=0.0))
    states = _states(24)
    skipped = [st for st in states if not agent.mcts.should_search(st)]
    assert skipped, "the fixture has no non-card decision to check"
    assert agent.select_actions_batch(skipped) == agent._policy_actions(skipped)


def test_the_search_spec_takes_first_play_urgency_last(tmp_path: Path) -> None:
    _, cfg, label = search_spec_config("search:m.pt:128:determinize:all::cpp:0.2")
    assert (cfg.simulations, cfg.determinize, cfg.subsample, cfg.fpu_reduction) == (128, True, 1.0, 0.2)
    assert label == "search128-det-fpu0.2"
    _, cfg, label = search_spec_config("search:m.pt:64")
    assert (cfg.fpu_reduction, cfg.gumbel_k, label) == (0.0, 0, "search64")


def test_play_match_plays_the_same_searcher(tmp_path: Path) -> None:
    path = tmp_path / "m.pt"
    torch.save(_model().state_dict(), str(path))
    bot, name = resolve_agent(f"gumbel:{path}:16:4", "US", device="cpu")
    assert isinstance(bot, _SearchBot)
    cfg = bot.searcher.cfg
    assert (cfg.simulations, cfg.gumbel_k, cfg.fpu_reduction) == (16, 4, 0.2)
    assert name.startswith("gumbel16-k4")
    state = _states()[0]
    choice = bot.select_from_state(state)
    assert choice["decision_type"] == int(state.ctx().decision_type)


def _walk(node: Any) -> List[Any]:
    out = [node]
    for ch in node.children.values():
        out.extend(_walk(ch))
    return out


def test_a_best_response_search_plays_the_greedy_move_wherever_the_other_side_moves() -> None:
    """opponent="greedy": inside the tree every node where the side not searching moves has visits
    on one move only, its top prior; the searching side's nodes still branch."""
    states = _states()
    cfg = BatchedMCTSConfig(simulations=64, backend="python", opponent="greedy", seed=3)
    roots = BatchedMCTS(_model(), device="cpu", config=cfg, featurise_capacity=256)._search(states)
    branched = False
    for r in roots:
        if r is None or r.terminal:
            continue
        for nd in _walk(r):
            if nd.terminal or not nd.expanded or nd.total == 0:
                continue
            visited = [i for i, x in enumerate(nd.n) if x > 0]
            if nd.mover != r.searcher:
                assert visited == [max(range(len(nd.priors)), key=nd.priors.__getitem__)]
            elif len(visited) > 1:
                branched = True
    assert branched


def test_a_best_response_search_needs_the_python_tree() -> None:
    with pytest.raises(ValueError, match="Python tree only"):
        BatchedMCTS(_model(), device="cpu", featurise_capacity=256,
                    config=BatchedMCTSConfig(simulations=8, opponent="greedy"))._search(_states(4))


def test_a_phase_share_splits_evenly_over_at_most_that_many_worlds() -> None:
    assert world_shares(10, 1) == [10]
    assert world_shares(10, 4) == [3, 3, 2, 2]
    assert world_shares(3, 16) == [1, 1, 1]          # never a world with no evaluation
    assert world_shares(0, 4) == [0]
    for per in range(1, 40):
        for m in (1, 2, 5, 16, 64):
            assert sum(world_shares(per, m)) == per


@pytest.mark.parametrize("worlds", [4, 16])
def test_more_worlds_spend_the_same_evaluations(worlds: int) -> None:
    """Worlds split a candidate's share; they never add evaluations, and the picks are legal and
    reproducible from the seed. Equal up to the children that end the game, which cost no
    evaluation and depend on the dice each world draws."""
    states = _states()
    spent, picks = {}, {}
    for m in (1, worlds):
        model = _CountingModel(_model())
        cfg = BatchedMCTSConfig(simulations=64, determinize=True, gumbel_k=8, gumbel_scale=0.0,
                                fpu_reduction=0.2, gumbel_worlds=m, seed=11)
        mcts = BatchedMCTS(model, device="cpu", config=cfg, featurise_capacity=256)
        picks[m] = mcts.best_actions(states)
        spent[m] = model.rows
        for st, a in zip(states, picks[m]):
            assert np.asarray(ActionEncoder.get_legal_mask(st))[a]
    assert spent[worlds] <= len(states) * (64 + 1)
    assert abs(spent[worlds] - spent[1]) <= 0.01 * spent[1]
    cfg = BatchedMCTSConfig(simulations=64, determinize=True, gumbel_k=8, gumbel_scale=0.0,
                            fpu_reduction=0.2, gumbel_worlds=worlds, seed=11)
    again = BatchedMCTS(_model(), device="cpu", config=cfg, featurise_capacity=256).best_actions(states)
    assert again == picks[worlds]


def test_the_gumbel_spec_takes_worlds_last(tmp_path: Path) -> None:
    path = tmp_path / "m.pt"
    torch.save(_model().state_dict(), str(path))
    _, cfg, label = search_spec_config(f"gumbel:{path}:256:8:0.2:all:16")
    assert (cfg.gumbel_worlds, cfg.node_filter, label) == (16, "all", "gumbel256-k8-w16")
    _, cfg, label = search_spec_config(f"gumbel:{path}:256:8:0.2:card")
    assert (cfg.gumbel_worlds, label) == (1, "gumbel256-k8-card")
