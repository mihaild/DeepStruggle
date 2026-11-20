"""ai/search/rollout_root.py: legal and reproducible picks, the network's own move with one
candidate, rollouts that spend rows only with a horizon, the measured player as the defaults, a
model read in its own observation feature set, a stuck rollout refused, the spec, a leaf network
(P33a) that reads the leaves in place of the root's own critic, racing, and two searchers deciding side
by side in a tournament with the games they play in turn."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pytest
import torch

import ts_engine as ts
import ai.search.rollout_root as rollout_root
from ai.models.coldwar_net_v2 import create_coldwar_net_v2
from ai.models.ladder_net import create_ladder_net
from ai.search.rollout_root import (RolloutConfig, RolloutRoot, RolloutRootAgent, RolloutTeacher, parse_rule,
                                   rollout_spec_config, rollout_spec_leaf)
from bindings.action_encoder import ActionEncoder
from tools.lib.batch_tournament import BatchMatchRunner
from tools.lib.player_agent import load_agent


def _model() -> torch.nn.Module:
    torch.manual_seed(0)
    m = create_coldwar_net_v2("cpu")
    m.eval()
    return m


_LADDER: Dict[str, Any] = dict(
    input_mode="grouped", aggregation="flatten", entity_dim=16, entity_proj_dim=256,
    card_self_attention=False, cross_attention=False, per_entity_heads=64, head_context=True,
    head_static=True, head_entities="country", head_center=True, identity_dim=0, drop_static=True,
    hidden_dim=64, num_res_blocks=0, num_attn_heads=4, card_lookup=False, card_lookup_heads=0,
    card_lookup_dim=0, card_lookup_identity_dim=0, categorical_value=False)


def _states(n: int = 12, advance: int = 60) -> List[ts.GameState]:
    runner = ts.VectorizedBatchRunner(n, 4242)
    for _ in range(advance):
        masks = np.asarray(runner.get_action_masks())
        runner.step_flat_all([int(np.flatnonzero(m)[0]) if m.any() else 211 for m in masks], True)
    return [runner.get_state(i) for i in range(n) if not ts.Engine.is_terminal(runner.get_state(i))]


@pytest.mark.parametrize("rule", ["argmax", "z2", "kl0.05"])
def test_picks_are_legal_and_reproducible(rule: str) -> None:
    states = _states()
    cfg = RolloutConfig(k=3, worlds=4, horizon=1, rule=rule, seed=3)
    a = RolloutRoot(_model(), cfg).choose(states)
    b = RolloutRoot(_model(), cfg).choose(states)
    assert a == b
    for st, x in zip(states, a):
        assert np.asarray(ActionEncoder.get_legal_mask(st))[x]


def test_one_candidate_is_the_networks_move() -> None:
    states = _states()
    one = RolloutRoot(_model(), RolloutConfig(k=1, worlds=2, horizon=0, rule="argmax")).choose(states)
    raw = RolloutRoot(_model(), RolloutConfig(k=1, worlds=2, horizon=0, rule="argmax"))
    lg, masks, _ = raw._forward(states)
    assert one == [int(np.argmax(np.where(m.astype(bool), l, -np.inf))) for l, m in zip(lg, masks)]


def test_a_horizon_spends_rollout_rows() -> None:
    states = _states()
    r0 = RolloutRoot(_model(), RolloutConfig(k=2, worlds=2, horizon=0))
    r0.choose(states)
    r2 = RolloutRoot(_model(), RolloutConfig(k=2, worlds=2, horizon=2))
    r2.choose(states)
    assert r2.rows > r0.rows > 0


def test_the_defaults_are_the_measured_player() -> None:
    # research/log/E7_search_depth_and_value.md section 4: rollout:<ckpt>:4:16:4:z2. A bare spec
    # must give that player, not the one-move-ahead root the same note measured at -170 a game.
    d = RolloutConfig()
    assert (d.k, d.worlds, d.horizon, d.rule) == (4, 16, 4, "z2")
    _, cfg, label = rollout_spec_config("rollout:m.pt")
    assert (cfg.k, cfg.worlds, cfg.horizon, cfg.rule, label) == (4, 16, 4, "z2", "rollout-k4-w16-h4-z2")
    _, cfg, _ = rollout_spec_config("rollout:m.pt::8")
    assert (cfg.k, cfg.worlds, cfg.horizon, cfg.rule) == (4, 8, 4, "z2")


def test_a_model_is_read_in_its_own_feature_set() -> None:
    # A model that declares OPS_BUDGET refuses the base width; the root must ask it for its set.
    torch.manual_seed(0)
    net = create_ladder_net("cpu", **{**_LADDER, "obs_features": int(ts.OBS_FEATURE_OPS_BUDGET)})
    net.eval()
    states = _states(6)
    root = RolloutRoot(net, RolloutConfig(k=2, worlds=2, horizon=1, rule="argmax"))
    assert root.features == int(ts.OBS_FEATURE_OPS_BUDGET)
    picks = root.choose(states)
    for st, x in zip(states, picks):
        assert np.asarray(ActionEncoder.get_legal_mask(st))[x]


def test_a_stuck_rollout_is_an_error(monkeypatch: pytest.MonkeyPatch) -> None:
    # An unfinished rollout is never read where it stands.
    monkeypatch.setattr(rollout_root, "_MAX_ROLLOUT_STEPS", 1)
    with pytest.raises(RuntimeError, match="still running"):
        RolloutRoot(_model(), RolloutConfig(k=2, worlds=2, horizon=3)).choose(_states(4))


def test_the_spec(tmp_path: Path) -> None:
    path = tmp_path / "m.pt"
    torch.save(_model().state_dict(), str(path))
    _, cfg, label = rollout_spec_config(f"rollout:{path}:4:32:2:z2")
    assert (cfg.k, cfg.worlds, cfg.horizon, cfg.rule, label) == (4, 32, 2, "z2", "rollout-k4-w32-h2-z2")
    agent = load_agent(f"rollout:{path}:2:2:0:argmax", device="cpu")
    assert isinstance(agent, RolloutRootAgent)
    st = _states(4)
    assert len(agent.select_actions_batch(st)) == len(st)
    with pytest.raises(ValueError):
        parse_rule("best")


class _ZeroCritic(torch.nn.Module):
    """A network's policy with a critic that says 0 everywhere."""

    def __init__(self, inner: torch.nn.Module) -> None:
        super().__init__()
        self.inner = inner

    def forward(self, obs: torch.Tensor, masks: torch.Tensor) -> Any:
        lg, v, aux = self.inner(obs, masks)
        return lg, torch.zeros_like(v), aux


def test_the_roots_own_network_as_leaf_is_the_plain_root() -> None:
    states = _states()
    cfg = RolloutConfig(k=3, worlds=4, horizon=1, rule="z2", seed=5)
    model = _model()
    plain = RolloutRoot(model, cfg)
    leafed = RolloutRoot(model, cfg, leaf_model=model)
    assert plain.choose(states) == leafed.choose(states)
    assert plain.last_stats == leafed.last_stats


def test_the_leaf_network_reads_the_leaves() -> None:
    """A leaf critic that says 0 gives every candidate the same value at a horizon-1 leaf, so no lead
    ever forms and the root keeps the network's move -- while the root's own critic departs somewhere."""
    states = _states(24)
    cfg = RolloutConfig(k=3, worlds=4, horizon=1, rule="argmax", seed=5)
    model = _model()
    root = RolloutRoot(model, cfg, leaf_model=_ZeroCritic(model))
    picks = root.choose(states)
    checked = 0
    for st, pick in zip(root.last_stats, picks):
        if st["worlds"] == 0:
            continue
        leaf_values = [q for a, q in st["q"].items()]
        # a finished game is valued by its result, not by the critic: those rows may still differ
        if all(abs(q) < 1e-9 for q in leaf_values):
            assert pick == st["candidates"][0]
            checked += 1
    assert checked > 0


def test_a_leaf_checkpoint_in_the_spec(tmp_path: Path) -> None:
    a, b = tmp_path / "a.pt", tmp_path / "b.pt"
    torch.save(_model().state_dict(), str(a))
    torch.save(_model().state_dict(), str(b))
    spec = f"rollout:{a}:2:2:1:z2:{b}"
    assert rollout_spec_leaf(spec) == str(b) and rollout_spec_leaf(f"rollout:{a}:2:2:1:z2") is None
    agent = load_agent(spec, device="cpu")
    assert isinstance(agent, RolloutRootAgent) and agent.root.leaf_model is not None
    assert "leaf" in agent.name
    st = _states(4)
    assert len(agent.select_actions_batch(st)) == len(st)


def test_cuda_graphs_are_eager_on_the_cpu_and_parse_from_the_spec(tmp_path: Path) -> None:
    states = _states()
    cfg = RolloutConfig(k=3, worlds=4, horizon=1, rule="z2", seed=3)
    eager = RolloutRoot(_model(), cfg).choose(states)
    graphed = RolloutRoot(_model(), RolloutConfig(k=3, worlds=4, horizon=1, rule="z2", seed=3, cuda_graphs=True))
    assert not graphed._graphed                          # no graphs off the GPU
    assert graphed.choose(states) == eager
    path = tmp_path / "m.pt"
    torch.save(_model().state_dict(), str(path))
    _, c, _ = rollout_spec_config(f"rollout:{path}:4:16:4:z2:later:graphs")
    assert c.stop == "later" and c.cuda_graphs
    _, c, label = rollout_spec_config(f"rollout:{path}:4:16:4:z2:earlier")
    assert c.stop == "earlier" and label.endswith("-h4-untilturnend-z2")
    _, c, label = rollout_spec_config(f"rollout:{path}:4:16:4:z2:earlier:decider:graphs")
    assert c.leaf_side == "decider" and c.cuda_graphs and label.endswith("-h4-untilturnend-dleaf-z2")
    assert rollout_spec_leaf(f"rollout:{path}:4:16:4:z2:decider") is None
    _, c, label = rollout_spec_config(f"rollout:{path}:4:16:4:z2:turn:own")
    assert c.leaf_side == "own" and label.endswith("-turnend-ownleaf-z2")
    assert rollout_spec_leaf(f"rollout:{path}:4:16:4:z2:graphs:{path}") == str(path)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
def test_cuda_graphs_agree_with_the_eager_root() -> None:
    """Graphs pad the batch, which can move a value by ~1e-7; the picks on these positions agree."""
    states = _states(24)
    dev = torch.device("cuda")
    model = _model().to(dev)
    eager = RolloutRoot(model, RolloutConfig(k=3, worlds=4, horizon=2, rule="z2", seed=8), device=dev).choose(states)
    root = RolloutRoot(model, RolloutConfig(k=3, worlds=4, horizon=2, rule="z2", seed=8, cuda_graphs=True), device=dev)
    assert root._graphed
    assert root.choose(states) == eager


def test_racing_spends_fewer_rows_and_picks_only_candidates_left_in_the_race() -> None:
    states = _states(24)
    whole = RolloutRoot(_model(), RolloutConfig(k=4, worlds=8, horizon=1, rule="argmax", seed=5))
    whole.choose(states)
    raced = RolloutRoot(_model(), RolloutConfig(k=4, worlds=8, horizon=1, rule="argmax", seed=5, race=4))
    picks = raced.choose(states)
    assert raced.rows < whole.rows
    dropped = 0
    for st, pick, s in zip(states, picks, raced.last_stats):
        assert np.asarray(ActionEncoder.get_legal_mask(st))[pick]
        if s["worlds"] == 0:
            continue
        assert pick not in s["dropped"] and s["candidates"][0] not in s["dropped"]
        dropped += len(s["dropped"])
    assert dropped > 0
    assert RolloutRoot(_model(), RolloutConfig(k=4, worlds=8, horizon=1, rule="argmax", seed=5, race=4)).choose(states) == picks
    with pytest.raises(ValueError, match="race"):
        RolloutRoot(_model(), RolloutConfig(worlds=8, race=8))
    # A teacher's targets read every candidate over every world, which a race does not give.
    with pytest.raises(ValueError, match="not raced"):
        RolloutTeacher(_model(), RolloutConfig(worlds=8, race=4))


def test_racing_in_the_spec() -> None:
    _, cfg, label = rollout_spec_config("rollout:m.pt:4:32:4:z2:earlier:own:keyed:aligned:box4:race16")
    assert (cfg.worlds, cfg.race, cfg.keyed, cfg.box4) == (32, 16, True, True)
    assert label == "rollout-k4-w32-h4-untilturnend-ownleaf-keyed-aligned-box4-race16-z2"
    assert rollout_spec_leaf("rollout:m.pt:4:32:4:z2:earlier:race16") is None
    assert rollout_spec_config("rollout:m.pt:4:32")[1].race == 0
    with pytest.raises(ValueError, match="race"):
        rollout_spec_config("rollout:m.pt:4:16:4:z2:race16")


def test_two_searchers_side_by_side_play_the_games_they_play_in_turn(tmp_path: Path,
                                                                     monkeypatch: pytest.MonkeyPatch) -> None:
    """batch_tournament._choose_all runs two rollout searchers' decisions in two threads; each builds the
    batches it builds alone and draws only from its own generators, so every game is the same."""
    torch.manual_seed(1)
    other = create_coldwar_net_v2("cpu").eval()
    logs: Dict[bool, str] = {}
    picks: Dict[bool, List[List[int]]] = {}
    for together in (False, True):
        monkeypatch.setattr(RolloutRootAgent, "concurrent_decisions", together)
        a = RolloutRootAgent(_model(), "a", RolloutConfig(k=2, worlds=4, horizon=1, seed=3, race=2))
        b = RolloutRootAgent(other, "b", RolloutConfig(k=3, worlds=2, horizon=2, seed=4, stop="earlier"))
        picks[together] = []
        for agent in (a, b):            # every batch of picks, in each agent's own order
            mine: List[int] = []
            picks[together].append(mine)
            plain = agent.root.choose

            def recorded(states: Any, hands: Any = None, plain: Any = plain, mine: List[int] = mine) -> List[int]:
                out = plain(states, hands)
                mine.extend(out)
                return out
            monkeypatch.setattr(agent.root, "choose", recorded)
        log = tmp_path / f"{together}.jsonl"
        BatchMatchRunner.play_parallel_matchup(a, b, games_per_side=1, device="cpu", temperature=0.0,
                                               log_games_file=str(log))
        logs[together] = log.read_text()
    assert logs[False] and logs[True] == logs[False]
    assert all(picks[False]) and picks[True] == picks[False]
