"""The A/B comparison tool must find the confound it was built for, and not invent ones.

`tools/compare_runs.py` exists because E3-22-28's first attempt was compared against E3-20-28 by
hand and read as "no detectable difference", when the two runs also differed in the snapshot
cadence that feeds the self-play opponent pool. Two properties matter, and they pull opposite ways:

* a real difference must be found even when NEITHER run's metadata records the flag responsible,
  which is why the pool check reads the runs' own logs rather than their metadata;
* a key that one run simply predates must NOT be reported as a difference, or every comparison
  against an older baseline invents confounds and the warning stops meaning anything.
"""

from __future__ import annotations

from typing import Any, Dict, List

from tools.compare_runs import (
    config_diff,
    mean_ci,
    observed_confounds,
    pool_trace,
    window_mean,
)


def _rows(pairs: List[tuple[int, float]]) -> List[Dict[str, Any]]:
    return [{"total_steps": s, "opp_pool_size": n} for s, n in pairs]


def test_a_key_the_baseline_predates_is_unverifiable_not_a_difference() -> None:
    recorded, unverifiable, _ = config_diff({"seed": 1}, {"seed": 1, "per_player_gae": True})
    assert not recorded, f"invented a confound from an unrecorded key: {recorded}"
    assert any("per_player_gae" in line for line in unverifiable)


def test_a_genuine_recorded_difference_is_reported() -> None:
    recorded, _, _ = config_diff({"seed": 1, "eta": 0.1}, {"seed": 1, "eta": 0.0})
    assert any("eta" in line for line in recorded), recorded


def test_identical_configs_report_nothing() -> None:
    cfg = {"seed": 1, "eta": 0.1, "snapshot_every_steps": 5_000_000}
    recorded, unverifiable, _ = config_diff(cfg, dict(cfg))
    assert not recorded
    assert not unverifiable


def test_an_unrecorded_snapshot_cadence_is_called_out_even_when_both_omit_it() -> None:
    """Both runs omitting it is the exact case that hid the original confound."""
    _, unverifiable, _ = config_diff({"seed": 1}, {"seed": 1})
    assert any("snapshot_every_steps" in line for line in unverifiable), unverifiable


def test_divergent_pool_growth_is_detected_from_the_logs() -> None:
    """The real confound: the baseline's pool grew every 5M, the arm's every 26.7M."""
    base = _rows([(65_536, 1), (5_000_000, 2), (10_000_000, 3), (15_000_000, 4),
                  (20_000_000, 5), (25_000_000, 6), (30_000_000, 7), (40_000_000, 9)])
    arm = _rows([(65_536, 1), (26_700_000, 2)])
    found = observed_confounds(base, arm, 45_000_000)
    assert found, "the pool divergence that voided a whole run was not detected"
    assert "opponent pool" in found[0]


def test_matched_pool_growth_is_not_flagged() -> None:
    """A false positive here would make the warning worthless."""
    pairs: List[tuple[int, float]] = [
        (65_536, 1.0), (5_000_000, 2.0), (10_000_000, 3.0), (15_000_000, 4.0), (20_000_000, 5.0)]
    assert observed_confounds(_rows(pairs), _rows(list(pairs)), 20_000_000) == []


def test_a_one_model_pool_difference_is_tolerated() -> None:
    """Snapshots land on slightly different steps; an off-by-one must not read as a confound."""
    base = _rows([(65_536, 1), (5_000_000, 2), (10_000_000, 3), (15_000_000, 4), (20_000_000, 5)])
    arm = _rows([(65_536, 1), (5_100_000, 2), (10_100_000, 3), (15_100_000, 4), (20_100_000, 6)])
    assert observed_confounds(base, arm, 21_000_000) == []


def test_pool_trace_records_only_the_changes() -> None:
    rows = _rows([(1, 1), (2, 1), (3, 2), (4, 2), (5, 3)])
    assert pool_trace(rows) == [(1, 1.0), (3, 2.0), (5, 3.0)]


def test_window_mean_averages_only_inside_the_window() -> None:
    rows: List[Dict[str, Any]] = [
        {"total_steps": 1_000_000, "critic_auc": 0.5},
        {"total_steps": 4_000_000, "critic_auc": 0.7},
        {"total_steps": 6_000_000, "critic_auc": 0.9},   # outside
    ]
    assert window_mean(rows, "critic_auc", 0, 5_000_000) == pytest_approx(0.6)
    assert window_mean(rows, "critic_auc", 5_000_000, 10_000_000) == pytest_approx(0.9)
    assert window_mean(rows, "critic_auc", 10_000_000, 15_000_000) is None


def test_mean_ci_straddling_zero_is_distinguishable_from_one_that_does_not() -> None:
    m, half, _ = mean_ci([0.01, -0.01, 0.02, -0.02, 0.0])
    assert m - half <= 0.0 <= m + half, "a symmetric spread must straddle zero"
    m2, half2, _ = mean_ci([0.20, 0.21, 0.19, 0.22, 0.20])
    assert m2 - half2 > 0.0, "a tight positive spread must exclude zero"


def pytest_approx(x: float, tol: float = 1e-9) -> Any:
    import pytest
    return pytest.approx(x, abs=tol)
