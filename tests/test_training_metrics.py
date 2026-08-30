"""Tests for the training diagnostic metrics: explained variance, the frozen entropy
probe, episode aggregation (game length / ending mix), and TensorBoard logging safety."""

import json
import os
from typing import Any, Dict, List, Tuple

import numpy as np
import pytest
import torch
import torch.nn as nn

from ai.training.generic_trainer import (
    EPISODE_DEPENDENT_KEYS,
    TB_TAGS,
    TensorBoardLogger,
    summarize_completed_episodes,
)
from ai.training.nash_pg import ENTROPY_PROBE_SIZE, FixedEntropyProbe
from ai.training.rollout_buffer import RolloutBuffer, explained_variance
from bindings.ts_env import ENDING_REASON_KEYS, TsVectorizedEnv


class TestExplainedVariance:
    """1 - Var(G - V) / Var(G) over the value head's predictions."""

    def test_perfect_prediction_is_one(self):
        g = torch.tensor([-1.0, -0.5, 0.0, 0.25, 1.0, 0.7, -0.3])
        assert explained_variance(g, g.clone()) == pytest.approx(1.0, abs=1e-6)

    def test_constant_mean_predictor_is_zero(self):
        g = torch.tensor([-1.0, -0.5, 0.0, 0.25, 1.0, 0.7, -0.3])
        v = torch.full_like(g, float(g.mean()))
        assert explained_variance(g, v) == pytest.approx(0.0, abs=1e-6)

    def test_constant_offset_predictor_is_still_zero(self):
        # A biased but variance-free predictor explains no variance (bias shifts the
        # residual mean, not its variance).
        g = torch.tensor([-1.0, -0.5, 0.0, 0.25, 1.0, 0.7, -0.3])
        v = torch.full_like(g, 5.0)
        assert explained_variance(g, v) == pytest.approx(0.0, abs=1e-6)

    def test_anticorrelated_prediction_is_negative(self):
        g = torch.tensor([-1.0, -0.5, 0.0, 0.25, 1.0, 0.7, -0.3])
        v = -g
        ev = explained_variance(g, v)
        assert ev < 0.0
        # V = -G gives residual 2G, so Var(res) = 4 Var(G) and EV = -3.
        assert ev == pytest.approx(-3.0, abs=1e-5)

    def test_partial_prediction_between_zero_and_one(self):
        torch.manual_seed(0)
        g = torch.randn(500)
        v = 0.5 * g  # explains a strict subset of the variance
        ev = explained_variance(g, v)
        assert 0.0 < ev < 1.0

    def test_degenerate_inputs_do_not_blow_up(self):
        # Constant returns -> ratio undefined -> defined as 0.0, never NaN/inf.
        const = torch.full((16,), 0.3)
        assert explained_variance(const, torch.zeros(16)) == 0.0
        assert explained_variance(torch.zeros(0), torch.zeros(0)) == 0.0
        assert explained_variance(torch.zeros(4), torch.zeros(8)) == 0.0

    def test_shape_agnostic_over_buffer_layout(self):
        g = torch.randn(7, 5)
        v = g.clone()
        assert explained_variance(g, v) == pytest.approx(1.0, abs=1e-6)

    def test_buffer_diagnostics_report_value_head_fit(self):
        buf = RolloutBuffer(buffer_size=4, num_envs=3, obs_dim=8, action_dim=6, device="cpu")
        buf.returns_win.copy_(torch.randn(4, 3))
        buf.values_win.copy_(buf.returns_win.clone())
        buf.advantages.copy_(torch.randn(4, 3))
        diag = buf.diagnostics()
        assert set(diag) == {"explained_variance", "adv_std", "adv_std_raw", "adv_frac_near_zero"}
        assert diag["explained_variance"] == pytest.approx(1.0, abs=1e-6)
        assert 0.0 <= diag["adv_frac_near_zero"] <= 1.0

    def test_buffer_diagnostics_flag_dead_advantages(self):
        buf = RolloutBuffer(buffer_size=4, num_envs=3, obs_dim=8, action_dim=6, device="cpu")
        buf.advantages.zero_()
        diag = buf.diagnostics()
        assert diag["adv_frac_near_zero"] == pytest.approx(1.0)
        assert diag["adv_std"] == pytest.approx(0.0, abs=1e-6)


class _CountingNet(nn.Module):
    """Minimal (obs, mask) -> (logits, v_win, v_vp) stand-in for ColdWarNet."""

    def __init__(self, obs_dim: int = 12, action_dim: int = 6, scale: float = 1.0):
        super().__init__()
        self.linear = nn.Linear(obs_dim, action_dim)
        self.scale = scale
        self.seen_obs: List[torch.Tensor] = []

    def forward(self, obs: torch.Tensor, mask: torch.Tensor | None = None):
        self.seen_obs.append(obs.detach().clone())
        logits = self.linear(obs) * self.scale
        if mask is not None:
            logits = logits.masked_fill(mask == 0, -1e9)
        v = torch.zeros((obs.shape[0], 1))
        return logits, v, v


def _probe_sample(n: int = 64, obs_dim: int = 12, action_dim: int = 6, seed: int = 0) -> Tuple[torch.Tensor, torch.Tensor]:
    gen = torch.Generator().manual_seed(seed)
    obs = torch.randn((n, obs_dim), generator=gen)
    masks = (torch.rand((n, action_dim), generator=gen) < 0.7).to(torch.uint8)
    masks[:, 0] = 1
    masks[:, 1] = 1  # guarantee >= 2 legal actions so every row is eligible
    return obs, masks


class TestFixedEntropyProbe:
    """The probe pool must be sampled once and never resampled."""

    def test_pool_is_frozen_across_calls(self):
        probe = FixedEntropyProbe(size=16)
        obs_a, masks_a = _probe_sample(seed=1)
        assert probe.fill_from(obs_a, masks_a) is True
        assert probe.is_filled

        first_obs = probe.obs
        first_masks = probe.masks
        assert first_obs is not None and first_masks is not None
        snapshot = first_obs.clone()

        # A second (and third) fill attempt with entirely different data is a no-op.
        obs_b, masks_b = _probe_sample(seed=99)
        assert probe.fill_from(obs_b, masks_b) is False
        assert probe.fill_from(obs_b * 3.0, masks_b) is False

        assert probe.obs is first_obs, "probe pool tensor identity changed"
        assert probe.masks is first_masks
        assert torch.equal(probe.obs, snapshot), "probe pool contents changed"

    def test_pool_is_a_copy_not_a_view_of_the_rollout_buffer(self):
        # The rollout buffer's storage is overwritten in place every iteration, so the
        # probe must own its own copy or it would silently track the live distribution.
        probe = FixedEntropyProbe(size=16)
        obs, masks = _probe_sample(seed=5)
        probe.fill_from(obs, masks)
        snapshot = probe.obs.clone() if probe.obs is not None else None
        obs.zero_()
        masks.zero_()
        assert snapshot is not None and probe.obs is not None
        assert torch.equal(probe.obs, snapshot)
        assert probe.masks is not None and int(probe.masks.sum()) > 0

    def test_entropy_is_measured_on_the_same_states_every_time(self):
        probe = FixedEntropyProbe(size=16, chunk_size=8)
        obs, masks = _probe_sample(seed=3)
        probe.fill_from(obs, masks)

        net = _CountingNet()
        e1 = probe.mean_entropy(net)
        first_pass = torch.cat(net.seen_obs)
        net.seen_obs.clear()
        e2 = probe.mean_entropy(net)
        second_pass = torch.cat(net.seen_obs)

        assert e1 is not None and e2 is not None
        assert e1 == pytest.approx(e2)
        assert torch.equal(first_pass, second_pass)

    def test_entropy_falls_when_logits_sharpen(self):
        probe = FixedEntropyProbe(size=32, chunk_size=8)
        obs, masks = _probe_sample(seed=7)
        probe.fill_from(obs, masks)

        flat = _CountingNet(scale=0.0)     # uniform over legal actions -> max entropy
        sharp = _CountingNet(scale=1000.0)  # near-deterministic -> ~0 entropy
        sharp.load_state_dict(flat.state_dict())

        e_flat = probe.mean_entropy(flat)
        e_sharp = probe.mean_entropy(sharp)
        assert e_flat is not None and e_sharp is not None
        assert e_flat > e_sharp
        assert e_sharp < 0.02 * e_flat
        # Masked-out actions must not contribute: entropy caps at log(#legal actions).
        max_legal = int(probe.masks.sum(dim=-1).max()) if probe.masks is not None else 0
        assert e_flat <= np.log(max_legal) + 1e-5

    def test_unfilled_probe_returns_none_and_never_raises(self):
        probe = FixedEntropyProbe(size=8)
        assert probe.is_filled is False
        assert probe.mean_entropy(_CountingNet()) is None
        # No state offers a real choice -> nothing to probe, still no exception.
        obs, masks = _probe_sample(seed=11)
        assert probe.fill_from(obs, torch.zeros_like(masks)) is False
        assert probe.is_filled is False

    def test_probe_preserves_net_training_mode(self):
        probe = FixedEntropyProbe(size=8)
        obs, masks = _probe_sample(seed=13)
        probe.fill_from(obs, masks)

        net = _CountingNet()
        net.train()
        probe.mean_entropy(net)
        assert net.training is True
        net.eval()
        probe.mean_entropy(net)
        assert net.training is False

    def test_default_probe_size_is_the_documented_constant(self):
        assert ENTROPY_PROBE_SIZE == 2000
        assert FixedEntropyProbe().size == 2000


class TestEpisodeSummaries:
    """Game length and ending-reason mix over the episodes finishing in one iteration."""

    def _episodes(self) -> List[Dict[str, Any]]:
        return [
            {"turn": 4, "ending_reason": "defcon1_self"},
            {"turn": 10, "ending_reason": "final_scoring"},
            {"turn": 8, "ending_reason": "20vp"},
            {"turn": 6, "ending_reason": "held_scoring"},
        ]

    def test_mean_and_median_turn(self):
        stats = summarize_completed_episodes(self._episodes())
        assert stats["episodes_completed"] == 4.0
        assert stats["mean_turn"] == pytest.approx(7.0)
        assert stats["median_turn"] == pytest.approx(7.0)

    def test_ending_fractions_sum_to_one(self):
        stats = summarize_completed_episodes(self._episodes())
        frac_keys = [f"ending_frac_{k}" for k in ENDING_REASON_KEYS]
        assert sum(stats[k] for k in frac_keys) == pytest.approx(1.0)
        assert stats["ending_frac_defcon1_self"] == pytest.approx(0.25)
        assert stats["ending_frac_final_scoring"] == pytest.approx(0.25)
        assert stats["ending_frac_defcon1_provoked"] == pytest.approx(0.0)

    def test_empty_iteration_is_all_zeros(self):
        stats = summarize_completed_episodes([])
        assert stats["episodes_completed"] == 0.0
        assert stats["mean_turn"] == 0.0
        assert stats["median_turn"] == 0.0
        for key in ENDING_REASON_KEYS:
            assert stats[f"ending_frac_{key}"] == 0.0

    def test_every_summary_key_has_a_tensorboard_tag(self):
        stats = summarize_completed_episodes(self._episodes())
        for key in stats:
            assert key in TB_TAGS, f"{key} would fall back to a misc/ tag"
        assert EPISODE_DEPENDENT_KEYS <= set(stats)


class TestEnvEpisodeReporting:
    """The vectorized env must report the terminal turn and ending reason per episode."""

    def test_completed_episodes_carry_turn_and_ending_reason(self):
        env = TsVectorizedEnv(num_envs=8, base_seed=4242)
        rng = np.random.RandomState(0)
        _, masks, _ = env.reset_all()
        seen: List[Dict[str, Any]] = []

        for _ in range(4000):
            actions = [int(rng.choice(np.flatnonzero(m))) for m in masks]
            _, masks, _, _, info = env.step(np.array(actions, dtype=np.int64))
            seen.extend(info["completed_episodes"])
            if len(seen) >= 8:
                break

        assert seen, "no episode completed under random play"
        for ep in seen:
            assert 1 <= ep["turn"] <= 10
            assert ep["ending_reason"] in ENDING_REASON_KEYS
        stats = summarize_completed_episodes(seen)
        assert 1.0 <= stats["mean_turn"] <= 10.0
        assert sum(stats[f"ending_frac_{k}"] for k in ENDING_REASON_KEYS) == pytest.approx(1.0)


class TestTensorBoardLogger:
    """Logging must never be able to take down a multi-hour run."""

    def test_disabled_logger_is_inert(self, tmp_path):
        tb = TensorBoardLogger(str(tmp_path / "tb"), enabled=False)
        assert tb.active is False
        tb.log_metrics({"loss": 1.0}, step=1)
        tb.flush()
        tb.close()
        assert not (tmp_path / "tb").exists()

    def test_event_files_are_written(self, tmp_path):
        pytest.importorskip("tensorboard")
        log_dir = str(tmp_path / "tb")
        tb = TensorBoardLogger(log_dir, enabled=True)
        assert tb.active
        tb.log_metrics({"iteration": 3, "loss": 0.5, "explained_variance": 0.2}, step=3)
        tb.flush()
        tb.close()
        assert any(f.startswith("events.out.tfevents") for f in os.listdir(log_dir))

    def test_a_broken_writer_disables_itself_instead_of_raising(self, tmp_path):
        class ExplodingWriter:
            def add_scalar(self, *args: Any, **kwargs: Any) -> None:
                raise RuntimeError("disk full")

        tb = TensorBoardLogger(str(tmp_path / "tb"), enabled=False)
        tb.writer = ExplodingWriter()
        tb.log_metrics({"loss": 1.0}, step=1)  # must not raise
        assert tb.active is False, "a failing writer must be disabled, not retried forever"

    def test_non_numeric_and_skipped_metrics_are_ignored(self, tmp_path):
        recorded: List[Tuple[str, float, int]] = []

        class RecordingWriter:
            def add_scalar(self, tag: str, value: float, step: int) -> None:
                recorded.append((tag, value, step))

        tb = TensorBoardLogger(str(tmp_path / "tb"), enabled=False)
        tb.writer = RecordingWriter()
        tb.log_metrics(
            {
                "iteration": 7,
                "loss": 1.5,
                "mean_turn": 0.0,
                "completed_episodes": [{"turn": 3}],
                "note": "text",
            },
            step=7,
            skip_keys=EPISODE_DEPENDENT_KEYS,
        )
        tags = [t for t, _, _ in recorded]
        assert tags == ["train/loss"], f"unexpected tags logged: {tags}"

    def test_jsonl_schema_is_still_a_flat_scalar_record(self, tmp_path):
        # Guards the JSONL contract the run loop writes: one flat JSON object per line.
        stats = summarize_completed_episodes([{"turn": 5, "ending_reason": "20vp"}])
        record: Dict[str, Any] = {"iteration": 1, "loss": 0.1, "explained_variance": 0.5}
        record.update(stats)
        line = json.dumps(record)
        parsed = json.loads(line)
        assert all(isinstance(v, (int, float)) for v in parsed.values())
        assert parsed["ending_frac_20vp"] == 1.0
