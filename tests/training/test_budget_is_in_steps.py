"""A run's schedule is stated in env steps, and there is no time flag to get wrong.

The snapshot interval used to be *derived* from two time flags:

    eval_every_steps = train_steps // (duration_seconds // snapshot_interval_seconds)

which is indirect enough that landing a snapshot on a chosen step count meant solving for it.
E3-22-28's first attempt inherited `duration_seconds=3600` and `snapshot_interval_seconds=600`
and so snapshotted every 26.7M steps, where the E3-20-28 baseline it was being compared against
snapshotted every ~5M. Snapshots feed the self-play opponent pool, so at 45M steps the arm had 2
pool opponents spanning 26.7M against the baseline's 9 spanning 40M -- with `--opponent-frac 0.3`
that is a second uncontrolled factor, and the run was discarded at 47M steps.

These tests pin that the time flags are gone and that a nonsensical budget fails loudly.
"""

from __future__ import annotations

import os
import subprocess
import sys

import pytest
import ts_engine

from ai.training.generic_trainer import train_pipeline
from ai.training.train import build_parser

REMOVED = ["--duration-seconds", "--seconds-to-train",
           "--snapshot-interval-seconds", "--snapshot-every",
           "--curriculum-switch-seconds"]


@pytest.mark.parametrize("flag", REMOVED)
def test_the_time_flags_are_gone(flag: str) -> None:
    """Removed, not deprecated. A silently-accepted time flag is how this went wrong."""
    parser = build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args([flag, "600"])


def test_the_step_flags_are_present_and_default_sanely() -> None:
    args = build_parser().parse_args([])
    assert args.train_steps > 0, "a run must carry a positive step budget by default"
    assert args.snapshot_every_steps > 0, (
        "the snapshot cadence must be positive: it also sets how fast the opponent pool grows")
    assert args.curriculum_switch_steps is None, "the curriculum switch defaults to a fraction"


@pytest.mark.parametrize("kwargs, needle", [
    (dict(train_steps=0), "train_steps must be positive"),
    (dict(train_steps=-1), "train_steps must be positive"),
    (dict(snapshot_every_steps=0), "snapshot_every_steps must be positive"),
    (dict(snapshot_every_steps=-5), "snapshot_every_steps must be positive"),
])
def test_a_nonsense_budget_raises_before_anything_is_built(kwargs, needle: str) -> None:
    """It must fail before a device is resolved or an output directory is made.

    A run that creates a directory and *then* rejects its own arguments leaves an empty
    checkpoint dir behind that looks like a crashed experiment.
    """
    base = dict(train_steps=1_000_000, snapshot_every_steps=100_000)
    base.update(kwargs)
    with pytest.raises(ValueError, match=needle):
        train_pipeline(**base)  # type: ignore[arg-type]


def test_the_cli_rejects_a_zero_budget_end_to_end() -> None:
    """Through the real entry point, so the check cannot be bypassed by the CLI layer."""
    # The CLI imports the engine before anything else (it checks the build is fresh), so it is
    # launched the documented way: the engine this suite imported on PYTHONPATH.
    path = os.pathsep.join([os.path.dirname(os.path.abspath(ts_engine.__file__)), os.getcwd()])
    proc = subprocess.run(
        [sys.executable, "tools/train.py", "--train-steps", "0"],
        capture_output=True, text=True, timeout=300, env={**os.environ, "PYTHONPATH": path})
    assert proc.returncode != 0, "a zero step budget must not start a run"
    combined = proc.stdout + proc.stderr
    assert "train_steps must be positive" in combined, combined[-2000:]


@pytest.mark.parametrize("kwargs, needle", [
    # Audit T-12: the shaping wrapper hides the targeted reward from its attach / seat / metrics hooks.
    (dict(vp_potential=0.05, target_region="europe"), "--vp-potential does not work"),
    (dict(vp_potential=0.05, target_endings=["defcon"]), "--vp-potential does not work"),
    # Audit T-9: the restart pool was dropped silently under a forced opening; the start pool harvests in the base
    # observation and the E4 view.
    (dict(forced_opening="x", restart_pool_frac=0.25), "--forced-opening and --restart-pool-frac"),
    (dict(start_pool_frac=0.25, obs_features=1 << 2), "--start-pool-frac harvests"),
    (dict(start_pool_frac=0.25, merged_influence=True), "--start-pool-frac harvests"),
    # Fix verification V-5: per-player GAE's end-of-rollout bootstrap from a side's own last afterstate moment.
    (dict(per_player_gae=True), "afterstate critic at every decision"),
    (dict(per_player_gae=True, afterstate_value_coef=1.0, afterstate_frac=0.25), "afterstate critic at every decision"),
    (dict(per_player_gae=True, afterstate_value_coef=1.0, afterstate_frac=1.0, gamma=0.99), "--gamma 1.0"),
    (dict(per_player_gae=True, afterstate_value_coef=1.0, afterstate_frac=1.0, reward_scheme="shaped"), "terminal-only"),
    (dict(per_player_gae=True, afterstate_value_coef=1.0, afterstate_frac=1.0, vp_potential=0.05), "terminal-only"),
    (dict(per_player_gae=True, afterstate_value_coef=1.0, afterstate_frac=1.0, target_region="europe"),
     "terminal-only"),
    # Audit T-5: the critic prior against mechanisms that read the unshifted critic or treat pool games otherwise.
    (dict(critic_prior_window=200), "needs --opponent-per-game"),
    (dict(critic_prior_window=200, opponent_per_game=True), "needs an opponent pool"),
    *[(dict(critic_prior_window=200, opponent_per_game=True, opponent_frac=0.3, **kw), needle) for kw, needle in (
        (dict(afterstate_value_coef=1.0), "--afterstate-value-coef"),
        (dict(predeal_value_coef=1.0), "--predeal-value-coef"),
        (dict(same_perspective_bootstrap=True), "--same-perspective-bootstrap"),
        (dict(value_selfplay_only=True), "--value-selfplay-only"),
        (dict(value_learner_only=True), "--value-learner-only"),
    )],
])
def test_combinations_that_would_corrupt_silently_raise_before_anything_is_built(kwargs, needle: str) -> None:
    base = dict(train_steps=1_000_000, snapshot_every_steps=100_000)
    base.update(kwargs)
    with pytest.raises(ValueError, match=needle):
        train_pipeline(**base)  # type: ignore[arg-type]
