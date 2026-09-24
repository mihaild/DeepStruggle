"""The engine is built with clang, and its batch runner shares torch's one OpenMP thread pool.

Three ways this goes wrong without anything failing:

* a build directory configured under GCC keeps producing a GCC engine -- about 20% slower on the
  step + observation path, playing the same game, so nothing else notices;
* the batch runner goes back to `#pragma omp` under clang, which links LLVM's libomp: a second
  pool of spinning workers beside torch's libgomp, measured 5-7% slower on a rollout loop even
  though the engine's own code is faster;
* it is built with clang's `-fopenmp=libgomp`, which does not parallelise at all -- the loops run
  serially and only the throughput says so.

Each check runs in a fresh interpreter, so thread counts start from a known place.
"""
from __future__ import annotations

import os
import subprocess
import sys
import textwrap

import ts_engine as ts

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _run(code: str, threads: int) -> str:
    # OpenBLAS pinned to one thread: its idle workers are not the engine's and only add noise.
    env = dict(os.environ, OMP_NUM_THREADS=str(threads), OPENBLAS_NUM_THREADS="1")
    # The interpreter's own path, so the child imports the very ts_engine this test did.
    env["PYTHONPATH"] = os.pathsep.join(p for p in sys.path if p)
    out = subprocess.run([sys.executable, "-c", textwrap.dedent(code)], env=env, cwd=REPO,
                         capture_output=True, text=True, timeout=120)
    assert out.returncode == 0, out.stderr
    return out.stdout.strip()


def test_the_engine_is_built_with_clang() -> None:
    assert ts.BUILD_COMPILER.startswith("clang "), (
        f"ts_engine was built by {ts.BUILD_COMPILER!r}; the engine is built with clang. "
        f"Rebuild with tools/scripts/check_engine_fresh.sh, which reconfigures a GCC build dir.")


def test_the_batch_runner_adds_no_thread_pool_beside_torchs() -> None:
    counts = _run("""
        import os
        import torch
        torch.set_num_threads(4)
        a = torch.randn(256, 256)
        for _ in range(5):
            a = torch.tanh(a @ a / 256)          # torch's OpenMP pool is up
        before = len(os.listdir("/proc/self/task"))
        import ts_engine as ts
        r = ts.VectorizedBatchRunner(64, 1)
        r.refresh_all()
        for _ in range(5):
            m = r.get_action_masks()
            r.step_flat_all([int(row.argmax()) for row in m])   # the engine's parallel loop
        after = len(os.listdir("/proc/self/task"))
        print(before, after)
    """, threads=4)
    before, after = map(int, counts.split())
    assert after == before, (
        f"stepping the engine grew the process from {before} to {after} threads: the batch "
        f"runner brought its own OpenMP runtime instead of sharing torch's libgomp "
        f"(see gomp_parallel_for in bindings/ts_bindings.cpp)")


def test_the_batch_runner_really_runs_in_parallel() -> None:
    busy = _run("""
        import os, time
        import ts_engine as ts

        def cpu_ticks():
            out = {}
            for tid in os.listdir("/proc/self/task"):
                with open(f"/proc/self/task/{tid}/stat") as f:
                    fields = f.read().rsplit(")", 1)[1].split()
                out[tid] = int(fields[11]) + int(fields[12])   # utime + stime
            return out

        r = ts.VectorizedBatchRunner(512, 3)
        r.refresh_all()
        start = cpu_ticks()
        t = time.time()
        while time.time() - t < 1.0:
            m = r.get_action_masks()
            r.step_flat_all([int(row.argmax()) for row in m])
            for i, done in enumerate(r.get_terminals()):
                if done:
                    r.reset_game(i, i + 1)
        end = cpu_ticks()
        work = [v - start.get(tid, 0) for tid, v in end.items()]
        # A share of the work, not "any": numpy's BLAS threads tick over a little on their own,
        # and counting them is how a serial build first passed this test.
        print(sum(1 for w in work if w >= 0.3 * max(work)))
    """, threads=4)
    assert int(busy) >= 2, (
        f"only {busy} thread(s) did a real share of the work while the batch runner stepped "
        f"512 games with OMP_NUM_THREADS=4: its loops are running serially")
