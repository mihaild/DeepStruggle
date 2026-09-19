"""The watcher must alarm on metric conditions, not only on process conditions.

Until 2026-09-19 `tools/scripts/watch_run.py` emitted CRASH, STALL and NOSTART but merely
*printed* the training metrics alongside PROGRESS. Both collapses this repo has diagnosed were
therefore caught by a human reading numbers off a monitor line, which is what
`research/method/detecting_collapse.md` exists to replace.

These tests pin the three alarms and -- just as importantly -- the two triggers that were tried,
measured against the real runs, and **rejected for firing on healthy arms**:

* `opp_win_rate_mean > 0.9`: E3-37-31, one of the healthiest arms in the record, sits above it for
  947 of 1,229 iterations while its pool grows to 8. A strong policy beating its own older
  snapshots is improvement, not starvation.
* any one-sidedness threshold: swept over trailing windows of 100/200/300/400 iterations, the
  pinned fraction does not separate healthy from degenerate anywhere. The clean 320M arm sustains
  0.50-0.72 pinned against the one-sided arm's 0.54-0.87.

A test that only pinned the positives would let either be reintroduced.
"""

from __future__ import annotations

import importlib.util
import json
import os
from typing import Any, Dict, List

import pytest

_SPEC = importlib.util.spec_from_file_location(
    "watch_run", os.path.join(os.path.dirname(__file__), "..", "..",
                              "tools", "scripts", "watch_run.py"))
assert _SPEC is not None and _SPEC.loader is not None
watch_run = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(watch_run)


def _write(tmp_path, rows: List[Dict[str, Any]]) -> str:
    d = tmp_path / "run"
    d.mkdir(exist_ok=True)
    with open(d / "training_metrics.jsonl", "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    return str(d)


def _kinds(run_dir: str) -> List[str]:
    out = []
    for m in watch_run.health_alarms(run_dir, set()):
        if "no `opp_pool_size`" in m:
            out.append("NOPOOL")
        elif "never exceeded" in m:
            out.append("POOLSTUCK")
        elif "kl_div spiked" in m:
            out.append("KLSPIKE")
        else:
            out.append("OTHER:" + m[:40])
    return out


def _healthy(n: int = 200, pool: float = 8.0) -> List[Dict[str, Any]]:
    return [{"iteration": i, "total_steps": i * 1000, "opp_pool_size": pool,
             "opp_win_rate_mean": 0.5, "kl_div": 0.1,
             "episodes_completed": 150.0, "episodes_completed_won_us": 75.0}
            for i in range(n)]


def test_healthy_run_is_silent(tmp_path) -> None:
    assert _kinds(_write(tmp_path, _healthy())) == []


def test_missing_pool_key_alarms(tmp_path) -> None:
    rows = _healthy()
    for r in rows:
        del r["opp_pool_size"]
    assert "NOPOOL" in _kinds(_write(tmp_path, rows))


def test_pool_that_never_grows_alarms(tmp_path) -> None:
    assert "POOLSTUCK" in _kinds(_write(tmp_path, _healthy(pool=1.0)))


def test_kl_spike_is_relative_to_the_runs_own_median(tmp_path) -> None:
    rows = _healthy()
    for r in rows[-3:]:
        r["kl_div"] = 50.0
    assert "KLSPIKE" in _kinds(_write(tmp_path, rows))


def test_a_uniformly_high_kl_does_not_alarm(tmp_path) -> None:
    """No absolute cutoff: a run whose kl_div is simply large but steady is not a spike."""
    rows = _healthy()
    for r in rows:
        r["kl_div"] = 5.0
    assert "KLSPIKE" not in _kinds(_write(tmp_path, rows))


def test_a_strong_policy_beating_its_pool_does_not_alarm(tmp_path) -> None:
    """The rejected trigger. E3-37-31 would have fired on it for 947 of 1,229 iterations."""
    rows = _healthy()
    for r in rows:
        r["opp_win_rate_mean"] = 0.97
    assert _kinds(_write(tmp_path, rows)) == []


def test_one_sidedness_never_alarms(tmp_path) -> None:
    """The other rejected trigger: no threshold separates, so there is no threshold."""
    rows = _healthy()
    for r in rows:
        r["episodes_completed_won_us"] = 0.0
    assert _kinds(_write(tmp_path, rows)) == []


def test_alarms_latch_and_do_not_repeat(tmp_path) -> None:
    run_dir = _write(tmp_path, _healthy(pool=1.0))
    fired: set = set()
    first = watch_run.health_alarms(run_dir, fired)
    second = watch_run.health_alarms(run_dir, fired)
    assert first and not second


@pytest.mark.parametrize("done,won", [(0.0, 0.0), (None, None)])
def test_rows_without_episode_counts_are_excluded(done, won) -> None:
    """`won_us / max(1, 0)` once read "no episodes recorded" as "the US won none of them"."""
    row = {"episodes_completed": done, "episodes_completed_won_us": won}
    assert watch_run.us_episode_frac(row) is None
