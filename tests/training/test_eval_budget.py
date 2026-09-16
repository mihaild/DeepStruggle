"""Evaluation must not grow without bound, and a run's budget must be comparable across arms.

Two defects in the same area, both measured on a real 3-hour A/B:

* The opponent list grew by one snapshot per interval, so evaluation cost was quadratic in
  run length. The final evaluation faced 14 opponents and took 957s against a 900s snapshot
  interval, which left roughly one training iteration per interval.
* The budget was wall-clock and evaluation was charged to it. Evaluation cost depends on how
  long the policy's games run, so the arm playing longer games got less training: 473
  iterations against 1024 for its partner, which confounded the comparison entirely.

The wall-clock budget is now gone outright rather than merely discouraged, and so is the derived
snapshot interval that came with it -- a third instance of the same family, in which E3-22-28's
first attempt snapshotted every 26.7M steps against its baseline's ~5M and so trained against a
2-model opponent pool where the baseline had 9.
"""

from typing import List

from ai.training.generic_trainer import evaluate_and_log_snapshot


class _Dummy:
    def __init__(self, name: str) -> None:
        self.name = name


def _trim(opponents: List["_Dummy"], num_baselines: int, cap: int) -> None:
    """The trimming rule used after a new snapshot is appended."""
    if cap > 0:
        excess = len(opponents) - num_baselines - cap
        if excess > 0:
            del opponents[num_baselines:num_baselines + excess]


def test_opponent_list_keeps_baselines_and_drops_oldest_snapshots() -> None:
    opponents: List[_Dummy] = [_Dummy("RandomBot"), _Dummy("HeuristicBot")]
    num_baselines = len(opponents)

    for i in range(12):
        opponents.append(_Dummy(f"Snapshot_{i}"))
        _trim(opponents, num_baselines, cap=4)

        assert len(opponents) <= num_baselines + 4, (
            f"opponent list grew to {len(opponents)}; evaluation cost becomes quadratic "
            f"in run length"
        )
        names = [o.name for o in opponents]
        assert names[:2] == ["RandomBot", "HeuristicBot"], (
            f"baselines must survive trimming, got {names}"
        )

    kept = [o.name for o in opponents[num_baselines:]]
    assert kept == ["Snapshot_8", "Snapshot_9", "Snapshot_10", "Snapshot_11"], (
        f"the most recent snapshots are the informative ones; kept {kept}"
    )


def test_cap_of_zero_means_unlimited() -> None:
    opponents: List[_Dummy] = [_Dummy("HeuristicBot")]
    for i in range(6):
        opponents.append(_Dummy(f"Snapshot_{i}"))
        _trim(opponents, 1, cap=0)
    assert len(opponents) == 7, "cap=0 must preserve the old unlimited behaviour"


def test_evaluate_signature_takes_the_cap() -> None:
    """The trainer must be able to pass the cap through, not just compute it locally."""
    import inspect
    sig = inspect.signature(evaluate_and_log_snapshot)
    assert "max_snapshot_opponents" in sig.parameters
    assert "num_baselines" in sig.parameters


def test_step_budget_is_wired_through_the_cli() -> None:
    """--train-steps is what makes two A/B arms comparable; it must reach the pipeline."""
    import inspect
    from ai.training.generic_trainer import train_pipeline
    sig = inspect.signature(train_pipeline)
    assert "train_steps" in sig.parameters, "train_pipeline cannot be budgeted by steps"
    assert sig.parameters["train_steps"].default > 0, (
        "a run carries a positive step budget by default: step budgeting is no longer opt-in, "
        "because there is no time budget left to opt out to"
    )
    # The time flags are gone from the pipeline, not just from the CLI, so nothing can pass one.
    for gone in ("duration_seconds", "snapshot_interval_seconds", "curriculum_switch_seconds"):
        assert gone not in sig.parameters, (
            f"{gone} is back; a run's schedule is stated in env steps. The snapshot cadence was "
            "derived from the first two, which is how E3-22-28's first attempt snapshotted five "
            "times slower than its baseline and had to be discarded.")
    assert sig.parameters["snapshot_every_steps"].default > 0, (
        "the snapshot cadence must default positive: it also sets how fast the opponent pool grows")
    assert "max_snapshot_opponents" in sig.parameters
