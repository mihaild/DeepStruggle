"""P25 steps 3j-3l: the advantage-normaliser floor, the one-sided entropy ceiling and the per-seat
KL early stop. Each is off at 0, and off must leave the update exactly what it was."""

from __future__ import annotations

import types
from typing import Any, Dict

import pytest
import torch

from ai.models import create_coldwar_net
from ai.training import NashPGTrainer
from ai.training.nash_pg import BaseNashPGTrainer
from ai.training.rollout_buffer import RolloutBuffer


def _floor_buffer(c: float, n_per_batch: int = 1_000_000) -> Any:
    b = RolloutBuffer.__new__(RolloutBuffer)
    b.buffer_size, b.num_envs = n_per_batch, 1
    b.adv_norm_floor = c
    b.adv_norm_floor_memory_steps = 20_000_000.0
    b.adv_norm_floor_warmup_steps = 2_000_000.0
    b.adv_std_ema = 0.0
    b.adv_std_ema_steps = 0.0
    b.adv_norm_divisor = 0.0
    b.adv_norm_floor_bound = 0.0
    return b


def test_the_floor_waits_for_its_warmup() -> None:
    b = _floor_buffer(0.8)
    assert b._floored_divisor(0.35) == 0.35      # first batch: no EMA yet
    assert b._floored_divisor(0.01) == 0.01      # 1M seen < 2M warm-up: still no floor
    assert b.adv_norm_floor_bound == 0.0


def test_the_floor_binds_when_the_spread_falls_below_c_times_its_average() -> None:
    b = _floor_buffer(0.8)
    for _ in range(5):
        b._floored_divisor(0.35)
    assert b.adv_std_ema == pytest.approx(0.35)
    div = b._floored_divisor(0.20)                # a collapse-sized drop
    assert div == pytest.approx(0.8 * 0.35)
    assert b.adv_norm_floor_bound == 1.0
    assert b._floored_divisor(0.34) == 0.34      # a healthy batch is its own divisor


def test_the_floor_is_taken_before_the_batch_joins_the_average() -> None:
    """A collapsing batch must not lower its own floor."""
    b = _floor_buffer(0.5)
    for _ in range(4):
        b._floored_divisor(0.4)
    ema_before = b.adv_std_ema
    assert b._floored_divisor(0.05) == pytest.approx(0.5 * ema_before)
    assert b.adv_std_ema < ema_before


def test_the_average_has_a_bounded_memory() -> None:
    b = _floor_buffer(0.8)
    for _ in range(40):
        b._floored_divisor(0.4)
    for _ in range(60):                           # 60M steps at a lower level: 3 memories
        b._floored_divisor(0.2)
    assert b.adv_std_ema == pytest.approx(0.2, abs=0.01)


def _ceiling(ent_coef: float = 0.01, ceiling: float = 1.8, steps: float = 10e6) -> Any:
    return types.SimpleNamespace(
        ent_coef=ent_coef, entropy_ceiling=ceiling, entropy_ceiling_lr=0.01,
        entropy_ceiling_min_coef=-0.02, entropy_ceiling_grace_steps=5e6,
        total_env_steps=steps, ent_coef_seat={1: ent_coef, -1: ent_coef})


def _step(t: Any, m: Dict[str, float]) -> None:
    BaseNashPGTrainer._update_entropy_ceiling(t, m)


def test_below_the_ceiling_the_coefficient_is_the_fixed_bonus() -> None:
    t = _ceiling()
    for _ in range(100):
        _step(t, {"entropy_us": 1.2, "entropy_ussr": 1.6})
    assert t.ent_coef_seat == {1: 0.01, -1: 0.01}, "one-sided: it must never push entropy up"


def test_above_the_ceiling_only_that_seat_is_pushed_down_to_a_penalty() -> None:
    t = _ceiling()
    for _ in range(10):
        _step(t, {"entropy_us": 2.0, "entropy_ussr": 1.5})
    assert t.ent_coef_seat[1] == pytest.approx(0.01 - 10 * 0.01 * 0.2)
    assert t.ent_coef_seat[-1] == 0.01
    for _ in range(100):
        _step(t, {"entropy_us": 2.0, "entropy_ussr": 1.5})
    assert t.ent_coef_seat[1] == -0.02


def test_the_coefficient_recovers_once_back_under_the_ceiling() -> None:
    t = _ceiling()
    for _ in range(100):
        _step(t, {"entropy_us": 2.0})
    for _ in range(1000):
        _step(t, {"entropy_us": 1.5})
    assert t.ent_coef_seat[1] == 0.01


def test_nothing_moves_during_the_grace_period() -> None:
    t = _ceiling(steps=1e6)
    _step(t, {"entropy_us": 2.5, "entropy_ussr": 2.5})
    assert t.ent_coef_seat == {1: 0.01, -1: 0.01}


def _trainer(**kw: Any) -> NashPGTrainer:
    torch.manual_seed(0)
    dev = torch.device("cpu")
    model = create_coldwar_net(dev)
    return NashPGTrainer(active_net=model, num_envs=8, buffer_size=16, lr=3e-4, eta=0.1,
                         ref_update_freq=500, cuda_graphs=False, device=dev, **kw)


def test_a_kl_target_that_never_binds_is_bitwise_the_control() -> None:
    """The per-seat masking multiplies by exactly 1.0 while a seat is active, and the per-seat
    KL logging is detached, so an unbinding target must train the same parameters bit for bit."""
    torch.manual_seed(1)
    a = _trainer()
    ma = a.train_iteration()
    torch.manual_seed(1)
    b = _trainer(target_kl=1e9)
    mb = b.train_iteration()
    assert mb["kl_stop_frac_us"] == 0.0 and mb["kl_stop_frac_ussr"] == 0.0
    assert ma["approx_kl_us"] == mb["approx_kl_us"]
    for pa, pb in zip(a.active_net.parameters(), b.active_net.parameters()):
        assert torch.equal(pa, pb)


def test_a_tiny_kl_target_stops_both_seats() -> None:
    m = _trainer(target_kl=1e-12).train_iteration()
    assert m["kl_stop_frac_us"] > 0.5 and m["kl_stop_frac_ussr"] > 0.5


def test_every_lever_runs_and_reports() -> None:
    t = _trainer(adv_norm_floor=0.8, entropy_ceiling=0.5)
    t.buffer.adv_norm_floor_warmup_steps = 0.0
    t.entropy_ceiling_grace_steps = 0.0
    m = {}
    for _ in range(2):
        m = t.train_iteration()
    for k in ("adv_norm_divisor", "adv_norm_floor_bound", "adv_std_ema",
              "ent_coef_us", "ent_coef_ussr", "approx_kl_us", "approx_kl_ussr"):
        assert k in m, k
    assert t.ent_coef_seat[1] < 0.01 or t.ent_coef_seat[-1] < 0.01, (
        "a 0.5-nat ceiling is far below a fresh policy's entropy; a coefficient should have moved")


@pytest.mark.parametrize("kw", [dict(adv_norm_floor=0.8, wolf_seat_weight=True),
                                dict(entropy_ceiling=1.8, wolf_seat_weight=True),
                                dict(target_kl=0.02, wolf_seat_weight=True),
                                dict(adv_norm_floor=0.8, per_seat_adv_norm=True),
                                dict(target_kl=-1.0)])
def test_undefined_combinations_are_refused(kw: Dict[str, Any]) -> None:
    with pytest.raises(ValueError):
        _trainer(**kw)


def test_the_cli_offers_the_three_levers_off_by_default() -> None:
    from ai.training.train import build_parser
    a = build_parser().parse_args([])
    assert (a.adv_norm_floor, a.entropy_ceiling, a.target_kl) == (0.0, 0.0, 0.0)


def test_bonus_entropy_is_raw_by_default_and_a_fraction_of_the_maximum_when_normalized() -> None:
    import math
    from ai.training.nash_pg import bonus_entropy
    ent = torch.tensor([0.0, math.log(4.0), 1.0, 2.0])
    mask = torch.zeros(4, 10, dtype=torch.uint8)
    mask[0, :1] = 1      # forced
    mask[1, :4] = 1      # uniform over 4
    mask[2, :4] = 1
    mask[3, :10] = 1
    assert torch.equal(bonus_entropy(ent, mask, False), ent)
    got = bonus_entropy(ent, mask, True)
    want = torch.tensor([0.0, 1.0, 1.0 / math.log(4.0), 2.0 / math.log(10.0)])
    assert torch.allclose(got, want)


def test_entropy_normalize_off_is_bitwise_the_control_and_on_trains() -> None:
    torch.manual_seed(1)
    a = _trainer()
    a.train_iteration()
    torch.manual_seed(1)
    b = _trainer(entropy_normalize=False)
    b.train_iteration()
    for pa, pb in zip(a.active_net.parameters(), b.active_net.parameters()):
        assert torch.equal(pa, pb)
    torch.manual_seed(1)
    c = _trainer(entropy_normalize=True)
    c.train_iteration()
    assert any(not torch.equal(pa, pc) for pa, pc in zip(a.active_net.parameters(),
                                                       c.active_net.parameters()))


def test_the_cli_offers_entropy_normalize_off_by_default() -> None:
    from ai.training.train import build_parser
    assert build_parser().parse_args([]).entropy_normalize is False
