# Training throughput: where the time and the CPU go, measured cleanly

> **Revised 2026-09-23.** The first version of this log (kept below as "First reading") had two
> errors. It said the engine and batch runner are single-threaded; the runner parallelises over envs
> with OpenMP (`bindings/ts_bindings.cpp:1243`, `:1252`). The throughput figures in its first
> follow-up were also taken while another session was loading the machine: that baseline read
> 27–32k steps/s against a clean 39–40k, and the masked_fill fix read +45–65% against a clean +9%.
> Everything in "Clean measurements" was taken with no other training or tournament process
> running (checked before every run), old and new code alternating.

## Clean measurements

`/workspace/data/logs/perf/perf_run.sh`: M2d as on the P25 bench, 512 envs, 3M steps from
scratch, no snapshots. Steady steps/s is measured on the training clock from 1M steps on. The old
code was run from a detached worktree at `e67fbe5`. Run-to-run noise is about ±5% when clean.

| code | CPU threads | steps/s | wall | CPU-s |
|:---|:---|---:|---:|---:|
| old (`e67fbe5`) | default (16) | 39,322 / 40,124 | 92–95 s | 808–855 |
| + masked_fill | default | 43,691 / 42,741 | 85–91 s | 743–811 |
| + masked_fill | 1 / 2 / 4 | 26,569 / 35,747 / 40,960–43,691 | — | 131 / 163 / 228–244 |
| **+ masked_fill + targeted refresh** | **default** | **49,152 / 46,811** | **76–79 s** | 636–654 |
| + both | 4 | 43,691 / 42,741 | 85–86 s | 220–221 |
| + both | 8 | 44,684 / 47,953 | 78–83 s | 360–382 |

### Where the time went

* **`torch.tensor(-1e9, device=cuda)` in every forward pass.** The masked logits were filled with
  `torch.where(mask, logits, torch.tensor(-1e9, device=...))`. Building that scalar is a blocking
  host-to-device copy, so the CPU waited for the GPU at every forward pass. The fix is
  `logits.masked_fill(~mask, -1e9)`, with identical values (`tests/training/test_masked_logits.py`),
  in both `ColdWarNet` and `ColdWarNetV2` and therefore in `LadderNet`. Clean gain: **+9%**.
* **Observations built twice per step.** Per env and single-threaded, measured on the runner:
  * a game step takes **~0.8 µs** (the engine benchmark gives 1.5M steps/s);
  * a mask takes 0.67 µs;
  * an **observation takes 7.9 µs**.

  `step_flat_all` already rebuilds each env it steps, and `reset_game` each env it resets. Yet
  `TsVectorizedEnv.step` called `refresh_all()` whenever any game ended, which is nearly every step
  at 512 envs, rebuilding all 512 again. That was ~4.4 ms of single-threaded work per step. Now
  the refresh runs only when a start position was injected through `set_state`. Clean gain: a
  further **~+11%**, and **~+20% in total** over the old code.
* **The CPU threads.** Both PyTorch and the runner use OpenMP. With 16 threads, idle workers spin,
  but the runner's parallel loops over envs are real work on the critical path, which is why one
  thread is 40% slower. After both fixes, 8 threads give full speed at ~43% less CPU. 4 threads
  cost ~10% of speed for two-thirds less CPU.

### Found on the way: start positions were injected without a refresh

`reset_all` refreshed **before** injecting start positions, and `reset_env` injected without
refreshing at all. An injected env therefore kept the old observation and mask. On the old code,
`tests/training/test_env_refresh.py` fails on `reset_all` and hits an engine refusal when every env
starts injected ("the cache is stale on 13 entries"). No P25 or late-dynamics run uses start
positions (`--start-pool-frac`), so no result in this record is affected.

### The excess CPU is OpenMP spin-waiting, and passive waiting removes it

Both PyTorch and the batch runner use OpenMP. With the default wait policy, idle workers spin
between parallel regions. Same current code, clean, two runs each:

| `OMP_WAIT_POLICY` | steps/s | CPU-s | mean cores |
|:---|---:|---:|---:|
| default (spin) | 46,811 / 49,152 | 636–724 | ~8.5 |
| **passive** | **45,723 / 46,811** | **129–130** | **~1.6** |

**About 80% of a training process's CPU was idle workers spinning, at no throughput cost to
remove.** `tools/train.py` and `tools/tournament.py` now set `OMP_WAIT_POLICY=PASSIVE` before torch
or `ts_engine` load libgomp, and an explicit value in the environment still wins. With no variable
set, the training harness now gives 47,953 steps/s on 123 CPU-s. A 5-player, 2,000-game tournament
goes from 139.6 games/s on 235 CPU-s (spin) to **167.1 games/s on 22 CPU-s** (passive).

**Why it used to cost less CPU.** The code is not the cause. The 09-20 code (`d8aa324c`, which the
E4-08 seed sweep ran at ~51k steps/s), built from its own sources and run on the same harness today,
spins just as hard: 34,493 / 41,831 steps/s on 926 / 768 CPU-s. No commit ever set a thread count
or a wait policy. What did change is the kernel, from 7.1.8 to 7.2.6 at the first reboot on
09-23. The high CPU was first noticed after that reboot, and how long libgomp's spinning workers
keep a core busy depends on the scheduler. That is the likely explanation, unconfirmed, because
the old kernel cannot be booted here. Passive waiting makes the question moot either way.


### Main-thread throughput, 2026-09-23 (late)

Clean runs as above, two each. The harness's steps/s is quantised, because `elapsed_seconds` has
whole-second resolution: 65,536 means one 65,536-step iteration per second. Wall time for the
same 3M steps is given alongside.

| commit | change | steps/s | wall |
|:---|:---|---:|---:|
| `31810cc` | passive OpenMP waiting (baseline for this section) | 45,723–47,953 | 78–81 s |
| `4124562` | no CPU-GPU syncs in the PPO minibatch loop | 49,152 / 53,137 | 71–76 s |
| `7766e3d` | two rollout forwards instead of three | 56,174 / 56,174 | 67–69 s |
| `7766e3d` | pi_ref log-probs once per update, not per minibatch | 65,536 / 65,536 | 59–60 s |
| next commit | rollout forwards as CUDA-graph replays, learner and opponent overlapped | **70,217 / 70,217** | 57–59 s |

Against the clean start of this log (old code, 39–40k steps/s, 808–855 CPU-s), that is **~1.8×
the throughput on ~1/8 of the CPU**. Equivalence was checked with one seeded collect-and-update
step on CUDA against the previous commit, using `/workspace/data/logs/perf/train_step_equivalence.py`:

* **No syncs in the minibatch loop:** weights bitwise identical, with or without a pool. Metrics
  identical without a pool, and within 1e-8 relative with one.
* **pi_ref once per update:** weights and metrics bitwise identical, over 8 shuffled minibatches
  and 2 chunks.
* **Two forwards instead of three:** the learner's logits come from the full-batch pass, so the
  weights differ by at most 9e-7 after one update. It computes the same function.
* **CUDA graphs:** per network, bitwise equal to eager at the same batch, including after
  in-place optimiser steps. In a seeded rollout the buffers are identical without a pool. With a
  pool the actions, observations and values are identical and the log-probs agree within 1e-5
  (`tests/training/test_graphed_forward.py`).
* **A bug caught on the way:** two graphs captured on the shared default capture stream record
  the same cuBLAS workspace, and replayed concurrently they race on it. Probabilities were off by
  up to 0.89. Each graph now captures on its own stream, and a test covers concurrent replay.

### What is left

* **The update is compute-bound**, at ~0.52 s per iteration against ~0.53 s for the rollout
  (instrumented, before graphs): 64 minibatches of 4,096 at ~8 ms each. Going further needs a
  numerics change, such as TF32, bf16 or `torch.compile` (P11's gates), and that is the owner's
  decision.
* **Observation building** is the largest engine-side cost, at ~10× a game step. Making it
  cheaper is an engine change and needs the owner's approval.
* **Rollout host work:** `env.step` is ~1.0 ms per step and the observation copy to the GPU
  ~0.5 ms. Pinned host memory for the runner's observation buffer would make the copy
  asynchronous.
* **Thread count:** superseded by passive waiting, which takes the CPU down further (~1.6 cores)
  at full speed, without capping the threads the parallel loops can use.

## First reading (superseded; kept for the record)


**2026-09-23. A diagnosis only; nothing has been changed yet.** The owner noticed that `tools/train.py`
uses far more CPU than an engine running at more than 1M steps/s on one core can explain.

## What was measured

On a live run, E4-39-03 (λ 0.99 M2d, 512 envs, one of two runs on the machine):

* **Throughput.** Each process makes ~19–20k env steps/s. That is ~40 vector steps of all 512 envs
  per second, or ~25 ms per step. At 1M steps/s the engine accounts for about 0.5 ms of that.
* **CPU.** 24,540 CPU-seconds in 3,211 s of wall time, an average of 7.6 cores:
  * the main thread runs at 75–90% of one core;
  * **16 worker threads each run at ~35–43%, continuously,** with near-identical accumulated times
    (~24 CPU-minutes each after ~50 minutes). Together that is about 6.4 cores.
* **Whose threads.** The 16 workers are PyTorch's intra-op OpenMP pool (`at::get_num_threads() = 16`,
  `OMP_NUM_THREADS` unset). The engine and the batch runner are single-threaded: `engine/` and
  `bindings/` contain no OpenMP and no `std::thread`.
* **Inference is on the GPU.**
  * Each process holds ~2.9 GB of GPU memory.
  * Pool opponents and eval agents are moved to the training device (`generic_trainer.py:1757`,
    `:2124`, `:932`, `:1060`) and called on GPU tensors (`nash_pg.py:504`).
  * GPU utilisation is ~24%.
* **Not a regression in cost per step.** The machine's total step rate is about constant however
  many runs share it: E4-27-03 alone did 38.3k/s (the code before this session), two E4-39 runs do
  ~38.4k/s together, and five bench runs did ~34.9k/s. The share a single process gets moves with
  how many processes there are.

## Reading

* **The pool is mostly spinning.** The per-step CPU tensor work (converting observations, masks and
  rewards before the copy to the GPU) is small. Each operation still enters an OpenMP parallel
  region, and libgomp workers spin-wait between regions. That is ~6 cores per process of
  near-useless CPU.
* **The step rate is bound by the main thread,** not the engine and not the GPU. The ~25 ms per
  vector step goes to:
  * per-step Python loops over the 512 envs;
  * building the observations (512 × 3,824 floats) and copying them to the GPU, ~7.8 MB per step;
  * the pool opponent's forward pass;
  * Python overhead around the GPU calls.

## Proposed, not done

1. Cap PyTorch's CPU threads in the trainer (e.g. `torch.set_num_threads(2)`), then compare steps/s
   and CPU against an uncapped run.
2. Profile the main thread on a short run (cProfile, or py-spy if it is added to the environment).
   Then remove the per-env Python loops and the host-side copies. The engine could supply more than
   10× the current rate.
