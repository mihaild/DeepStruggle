"""A watcher must track ITS run, not "some training process somewhere".

`tools/scripts/watch_run.py` decided liveness with `pgrep -f tools/train.py`, which is true
whenever *any* run is on the box. On 2026-09-16 E3-22-28 was killed and relaunched, and the old
watcher -- pointed at the run that had been killed, whose directory had even been renamed -- kept
reporting `STALL: alive but stuck` instead of `CRASH`. Confusing a dead run with a stuck one is
the exact failure this script exists to prevent.

The run now records its own PID in `run.pid` and the watcher checks that. The pattern match
survives only as a fallback for runs started before the file existed.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time

import pytest

WATCHER = "tools/scripts/watch_run.py"


def _dead_pid() -> int:
    """A PID that certainly is not running: start a trivial process and reap it."""
    proc = subprocess.Popen([sys.executable, "-c", "pass"])
    proc.wait()
    return proc.pid


def _run_dir(tmp_path, steps: int, pid: int | None) -> str:
    d = tmp_path / "E9-99-01_20260101_000000"
    d.mkdir()
    with open(d / "training_metrics.jsonl", "w", encoding="utf-8") as f:
        f.write(json.dumps({"iteration": 1, "total_steps": steps,
                            "elapsed_seconds": 10}) + "\n")
    if pid is not None:
        (d / "run.pid").write_text(f"{pid}\n", encoding="utf-8")
    return str(d)


def _watch(run_dir: str, extra: list[str] | None = None, seconds: int = 8) -> str:
    """Run the watcher briefly and return what it said.

    It is NOT an error for the watcher to still be running at the deadline -- it exits only on a
    terminal state, so a healthy run means it never exits. Waiting for exit would hang. Take
    whatever it printed and kill it.
    """
    proc = subprocess.Popen(
        [sys.executable, WATCHER, run_dir, "--target-steps", "160000000",
         "--interval", "1", "--grace", "1", "--stall-after", "2", *(extra or [])],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    try:
        return proc.communicate(timeout=seconds)[0]
    except subprocess.TimeoutExpired:
        proc.kill()
        return proc.communicate()[0]


def test_a_dead_run_reports_crash_even_while_another_run_is_alive(tmp_path) -> None:
    """The regression. A live sibling must not make a dead run look merely stalled."""
    run_dir = _run_dir(tmp_path, steps=43_515_904, pid=_dead_pid())

    # A process whose command line matches the default --pattern, standing in for the relaunch.
    sibling = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(60)  # tools/train.py"])
    try:
        time.sleep(0.5)
        out = _watch(run_dir)
    finally:
        sibling.kill()
        sibling.wait()

    assert "CRASH" in out, out
    assert "STALL" not in out, f"a dead run was reported as merely stuck:\n{out}"


def test_a_live_run_is_not_reported_as_crashed(tmp_path) -> None:
    """The other direction: the PID check must not kill a healthy watch."""
    proc = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
    try:
        run_dir = _run_dir(tmp_path, steps=1_000_000, pid=proc.pid)
        out = _watch(run_dir)
        assert "CRASH" not in out, f"a live run was declared dead:\n{out}"
        assert "NOSTART" not in out, out
        assert "PROGRESS" in out or "STALL" in out, f"the watcher said nothing:\n{out}"
    finally:
        proc.kill()
        proc.wait()


def test_a_run_without_a_pid_file_still_falls_back(tmp_path) -> None:
    """Runs launched before run.pid existed must keep working, pattern match and all."""
    run_dir = _run_dir(tmp_path, steps=1_000_000, pid=None)
    assert not os.path.exists(os.path.join(run_dir, "run.pid"))
    out = _watch(run_dir, ["--pattern", "a-pattern-that-matches-nothing-xyzzy"])
    assert "CRASH" in out, f"with no pid file and no matching process this is a dead run:\n{out}"


def test_the_trainer_records_its_pid(tmp_path) -> None:
    """The producer side: the file the watcher depends on is actually written."""
    import inspect

    from ai.training import generic_trainer

    src = inspect.getsource(generic_trainer.train_pipeline)
    assert '"run.pid"' in src, "train_pipeline no longer records its PID for the watcher"
