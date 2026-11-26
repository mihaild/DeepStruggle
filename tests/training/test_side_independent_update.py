"""--side-independent-update (owner, 2026-11-26: "nothing should mix signals for different sides. Critics from different
sides should not be compared, gradient normalizations should be independent, etc. It should always be effectively two
completely independent models").

Held to: with it, the US network's gradient in an update is exactly (bit for bit) the one it gets with the USSR rows
taken out of the minibatch, or with the USSR rows' advantages and targets and the USSR network's weights changed --
on the plain update and on the side-split update compiled through dynamo; without it, it is not. Each network's
gradient is clipped on its own. Each seat's advantages are normalised from its own rows only. It is refused without
two separate networks, without per-player GAE and with the options that decide one scalar from both sides. A new A39
launch takes it by default; a resume keeps what its state recorded, off for every state written before it."""

from __future__ import annotations

import os
import sys
from typing import Any, Callable, Dict, Iterator, List, Optional, Sequence, Tuple, cast

import numpy as np
import pytest
import torch

from ai.models.ladder_net import LadderNet, create_ladder_net
from ai.training import train as train_cli
from ai.training.generic_trainer import resolve_side_independent_update, save_resume_state
from ai.training.nash_pg import NashPGTrainer, side_mean, side_parameters
from ai.training.rollout_buffer import RolloutBuffer
from ai.training.side_split import SplitForward
from bindings.ts_env import TsVectorizedEnv
from tests.training.test_launch_flags import _lf, _run
from tests.training.test_side_split import CFG


def _net(**over: Any) -> LadderNet:
    torch.manual_seed(0)
    return create_ladder_net("cpu", **{**CFG, "side_nets": True, **over})


def _trainer(side_independent: bool = True, net: Optional[torch.nn.Module] = None, **kw: Any) -> NashPGTrainer:
    args: Dict[str, Any] = dict(active_net=_net() if net is None else net, env=TsVectorizedEnv(num_envs=8, base_seed=7),
                                num_envs=8, buffer_size=48, batch_size=96, lr=3e-4, eta=0.1, ref_update_freq=500,
                                cuda_graphs=False, device=torch.device("cpu"), per_player_gae=True,
                                afterstate_value_coef=1.0, afterstate_frac=1.0, predeal_value_coef=1.0,
                                zero_avoid_coef=0.01, z_loss_coef=1e-3, logit_gap_coef=1e-3, adv_filter_quantile=0.1,
                                target_kl=5.0, head_l2_init_coef=1e-3, side_independent_update=side_independent)
    args.update(kw)
    return NashPGTrainer(**args)


def _is_ussr_param(name: str) -> bool:
    return name.startswith("ussr_net.")


Batch = Tuple[torch.Tensor, ...]


def _us_grads(side_independent: bool, change: Callable[[NashPGTrainer, Batch], Batch],
              compiled: bool = False) -> List[torch.Tensor]:
    """The US network's (clipped) gradient in one optimizer step of the update, on one minibatch of a self-play
    rollout that `change` may alter (the trainer and the minibatch given; it returns the minibatch to train on)."""
    torch.manual_seed(11)
    np.random.seed(11)
    t = _trainer(side_independent, num_epochs=1)
    t.collect_rollouts()
    batch = next(iter(t.buffer.get_batches(t.batch_size)))
    batch = change(t, batch)

    def one_batch(*_a: Any, **_k: Any) -> Iterator[Batch]:
        yield batch
    t.buffer.get_batches = one_batch                       # type: ignore[method-assign]
    if compiled:
        # The compiled update's side-split call, through dynamo on the CPU (aot_eager: no C++ toolchain needed).
        torch._dynamo.reset()
        t.compile_update = "default"
        t._compiled_methods[(id(t.active_net), "forward")] = (
            t.active_net, SplitForward(cast(LadderNet, t.active_net), "forward", {"backend": "aot_eager"}, bucket=8))
        t._compiled_methods[(id(t.reference_net), "forward")] = (
            t.reference_net, SplitForward(cast(LadderNet, t.reference_net), "forward", None, bucket=8))
    names = [n for n, _ in t.active_net.named_parameters()]
    grads: List[List[torch.Tensor]] = []
    step = t.optimizer.step

    def spy(*a: Any, **k: Any) -> Any:
        grads.append([torch.zeros_like(p) if p.grad is None else p.grad.detach().clone()
                      for p in t.active_net.parameters()])
        return step(*a, **k)
    t.optimizer.step = spy                                  # type: ignore[method-assign]
    torch.manual_seed(12)
    t.train_step()
    assert len(grads) == 1
    return [g for n, g in zip(names, grads[0]) if not _is_ussr_param(n)]


def _same(a: List[torch.Tensor], b: List[torch.Tensor]) -> bool:
    return all(torch.equal(x, y) for x, y in zip(a, b))


def _keep(batch: Batch, rows: torch.Tensor) -> Batch:
    return tuple(x[rows] for x in batch)


def _us_rows_only(t: NashPGTrainer, batch: Batch) -> Batch:
    us = batch[11] == 1
    assert 0 < int(us.sum()) < int(us.numel()), "the minibatch must hold both sides' rows"
    return _keep(batch, us)


def _ussr_changed(t: NashPGTrainer, batch: Batch) -> Batch:
    """The USSR rows' advantages, stored log-probs and value targets changed, and the USSR network's weights."""
    b = list(batch)
    ussr = (b[11] == -1).float()
    b[3] = b[3] - 0.7 * ussr                                # stored log-probs
    b[4] = b[4] * (1.0 - ussr) + (-40.0 * b[4] + 3.0) * ussr   # advantages
    b[5] = b[5] + 25.0 * ussr                               # win returns: a USSR value gradient far past the clip
    b[6] = b[6] - 9.0 * ussr                                # VP returns
    with torch.no_grad():
        g = torch.Generator().manual_seed(9)
        for n, p in t.active_net.named_parameters():
            if _is_ussr_param(n):
                p.add_(0.3 * torch.randn(p.shape, generator=g))
    return tuple(b)


def _unchanged(t: NashPGTrainer, batch: Batch) -> Batch:
    return batch


@pytest.mark.parametrize("change", [_us_rows_only, _ussr_changed])
def test_the_us_gradient_is_its_own_rows_alone(change: Callable[[NashPGTrainer, Batch], Batch]) -> None:
    """With the option, the US network's gradient is bitwise the same with the USSR rows removed from the minibatch or
    with them (and the USSR network) changed; without it, the pooled means and the joint clip move it."""
    assert _same(_us_grads(True, change), _us_grads(True, _unchanged))
    assert not _same(_us_grads(False, change), _us_grads(False, _unchanged))


def test_the_compiled_side_split_update_keeps_the_sides_apart() -> None:
    """The same through the compiled update's per-side call (side_split.SplitForward, padded buckets, dynamo)."""
    assert _same(_us_grads(True, _ussr_changed, compiled=True), _us_grads(True, _unchanged, compiled=True))
    assert _same(_us_grads(True, _us_rows_only, compiled=True), _us_grads(True, _unchanged, compiled=True))


def _norm(params: Sequence[torch.nn.Parameter]) -> float:
    return float(torch.sqrt(torch.stack([(p.grad ** 2).sum() for p in params if p.grad is not None]).sum()))


def test_each_network_is_clipped_on_its_own() -> None:
    for on in (True, False):
        t = _trainer(on)
        us, ussr = side_parameters(t.active_net)
        for p in us:
            p.grad = torch.full_like(p, 1e-4)
        for p in ussr:
            p.grad = torch.full_like(p, 1.0)
        us_norm = _norm(us)
        assert us_norm < t.max_grad_norm
        t._clip_grads()
        us_after, ussr_after = _norm(us), _norm(ussr)
        assert ussr_after == pytest.approx(t.max_grad_norm, rel=1e-4)
        if on:
            assert us_after == pytest.approx(us_norm, rel=1e-6), "the USSR norm must not scale the US gradient"
        else:
            assert us_after < 0.01 * us_norm


def test_side_mean_is_each_sides_own_mean() -> None:
    x = torch.tensor([1.0, 2.0, 3.0, 10.0, 20.0])
    w = torch.tensor([1.0, 1.0, 0.0, 1.0, 1.0])
    us = torch.tensor([1.0, 1.0, 1.0, 0.0, 0.0])
    assert float(side_mean(x, w, None)) == pytest.approx(33.0 / 4)
    assert float(side_mean(x, w, (us, 1.0 - us))) == pytest.approx(1.5 + 15.0)


def _buffer(seat_us_adv: List[float], seat_ussr_adv: List[float], learner_ussr: Optional[List[float]] = None
            ) -> RolloutBuffer:
    n = len(seat_us_adv) + len(seat_ussr_adv)
    b = RolloutBuffer(buffer_size=1, num_envs=n, obs_dim=3, device="cpu")
    b.advantages = torch.tensor([seat_us_adv + seat_ussr_adv])
    b.players = torch.tensor([[1] * len(seat_us_adv) + [-1] * len(seat_ussr_adv)], dtype=torch.int8)
    if learner_ussr is not None:
        b.learner = torch.tensor([[1.0] * len(seat_us_adv) + learner_ussr])
    b.per_seat_adv_norm = True
    b.side_independent = True
    return b


def test_each_seats_advantages_come_from_its_own_rows() -> None:
    a = _buffer([1.0, 2.0, 4.0], [5.0, -3.0, 0.5])
    a.normalise_advantages()
    b = _buffer([1.0, 2.0, 4.0], [500.0, 70.0, -9.0, 3.0])
    b.normalise_advantages()
    assert torch.equal(a.advantages[0, :3], b.advantages[0, :3])
    us = torch.tensor([1.0, 2.0, 4.0])
    assert torch.allclose(a.advantages[0, :3], (us - us.mean()) / (us.std() + 1e-8))
    # A seat with too few learner rows for the learner-only statistics takes its own rows', never the batch's.
    c = _buffer([1.0, 2.0, 4.0], [5.0, -3.0, 0.5], learner_ussr=[1.0, 0.0, 0.0])
    c.adv_norm_learner_only = True
    c.normalise_advantages()
    ussr = torch.tensor([5.0, -3.0, 0.5])
    assert torch.allclose(c.advantages[0, 3:], (ussr - ussr.mean()) / (ussr.std() + 1e-8))
    d = c.diagnostics()
    assert d["adv_learner_n_ussr"] == 1.0 and d["adv_learner_n_us"] == 3.0
    assert d["adv_learner_mean_us"] == pytest.approx(7.0 / 3, rel=1e-6)
    assert d["adv_learner_mean_norm_us"] == pytest.approx(0.0, abs=1e-6)


@pytest.mark.parametrize("kw, needle", [
    (dict(per_player_gae=False, afterstate_value_coef=0.0), "--per-player-gae"),
    (dict(wolf_seat_weight=True, target_kl=0.0), "--wolf-seat-weight"),
    (dict(setup_entropy_floor=1.0), "--setup-entropy-floor"),
    (dict(setup_mc_credit=True), "--setup-mc-credit"),
    (dict(aux_own_coef=0.1), "--aux-ownership"),
])
def test_it_refuses_what_decides_one_scalar_from_both_sides(kw: Dict[str, Any], needle: str) -> None:
    with pytest.raises(ValueError, match=needle):
        _trainer(**kw)


@pytest.mark.parametrize("over", [dict(side_nets=False), dict(side_nets=False, side_heads=True)])
def test_it_needs_two_networks_sharing_nothing(over: Dict[str, Any]) -> None:
    with pytest.raises(ValueError, match="--ladder-side-nets"):
        _trainer(net=_net(**over))


def test_the_command_line_refuses_it_without_side_nets(monkeypatch: pytest.MonkeyPatch,
                                                       capsys: pytest.CaptureFixture[str]) -> None:
    monkeypatch.setattr(sys, "argv", ["train.py", "--side-independent-update"])
    with pytest.raises(SystemExit):
        train_cli.main()
    assert "--ladder-side-nets" in capsys.readouterr().err


def test_a_new_a39_run_takes_it_and_a_resume_keeps_what_it_recorded(tmp_path: Any) -> None:
    assert resolve_side_independent_update(None, True, None) is True
    assert resolve_side_independent_update(None, False, None) is False
    assert resolve_side_independent_update(False, True, None) is False
    old = os.path.join(str(tmp_path), "old.pt")
    torch.save({"total_env_steps": 1}, old)                  # a state written before the option
    assert resolve_side_independent_update(None, True, old) is False
    assert resolve_side_independent_update(True, True, old) is True       # a branch that asks for it
    for on in (True, False):
        t = _trainer(on)
        path = os.path.join(str(tmp_path), f"resume_{on}.pt")
        save_resume_state(path, t.active_net, t, iteration=1, total_env_steps=10, elapsed_seconds=1.0)
        assert torch.load(path, weights_only=False)["side_independent_update"] is on
        assert resolve_side_independent_update(None, True, path) is on


def test_codes_and_relaunch_commands() -> None:
    """A run's code names the option only when it was on; a relaunch of an A39 run that ran without it says so."""
    lf = _lf()
    a39 = {"ladder_config": {"side_nets": True}}
    old_a39, new_a39, a8 = _run(a39), _run({**a39, "side_independent_update": True}), _run({})
    assert "side_independent_update" not in lf.non_default(old_a39)
    assert lf.non_default(new_a39)["side_independent_update"] is True
    assert "side_independent_update" not in lf.non_default(a8)
    assert lf.launch_settings(old_a39)["side_independent_update"] is False
    assert "--no-side-independent-update" in lf.command_flags(lf.launch_settings(old_a39))[0]
    assert "side_independent_update" not in lf.launch_settings(new_a39)
    assert "side_independent_update" not in lf.launch_settings(a8)
