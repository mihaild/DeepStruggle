# E7 training throughput: where the time goes, and what stands between it and 2× (2026-10-05)

**Conclusion.** At 512 envs the rollout (0.27 s) and the update (0.36 s) split an iteration; the
shared-context per-entity head and `torch.compile` each take ~15% and together +32% end to end
(update 0.364 to 0.213 s). The compile A/B (E7-23-44) came out ~-15 Elo and -2.5 points as USSR, so
`--compile-update` stays opt-in.

The owner asked, after the network shrank to E7's shallow recipe (1.33M parameters), what larger
batches would buy and what stops training from being twice as fast. Nothing here changed training
code; the scripts are in `data/logs/perf/` (`e7_bench.py`, `e7_pe_head.py`, `e7_pe_equiv.py`,
`e7_sgemm.py`) and the job's `bench_batch.sh` / `phase_prof.py`.

## Larger batches: the env count matters, the minibatch does not

`tools/train.py` with E7-20-44's flags, 5 minutes per config, steady steps/s from the training
clock (noise about ±5%):

| envs \ minibatch | 4,096 | 8,192 | 16,384 | 32,768 |
|---:|---:|---:|---:|---:|
| 512 | **96.3k** (recipe) | 98.4k | 92.4k | 98.4k |
| 1,024 | 104.7k | 100.6k | 99.2k | -- |
| 2,048 | **114.0k** | -- | 113.6k | 112.2k |

* **The minibatch does nothing** for speed. The update's cost is its total arithmetic (4 epochs over
  the rollout), which a minibatch size only reslices.
* **2,048 envs is ~+18%**, by amortising each rollout step's fixed costs over 4× the rows. It is
  not free for learning: 262,144 steps per iteration, so 256 gradient steps between rollouts and a
  reference refresh every iteration. Untested for strength.

## Where an iteration goes

py-spy, non-blocking, main thread only (`--idle`, 100 Hz), 5 minutes of the E7 recipe with the
opponent pool; the profiled run trained at 96.4k steps/s, the same as unprofiled, so the shares
are not distorted:

| phase | share of wall |
|:---|---:|
| update: backward | 29.3% |
| update: forward + loss | 17.5% |
| update: optimizer step + grad clip | 4.4% |
| rollout: C++ step + observation (`step_flat_all`) | 10.4% |
| rollout: rest of `env.step` (Python) | 5.8% |
| rollout: forward (CUDA-graph replays, learner + pool opponent) | 8.5% |
| rollout: observation copy to the GPU (7.8 MB per step, pageable) | 7.5% |
| rollout: other Python (sampling, per-seat stats, buffer, critic tracker, GAE) | ~16% |

Timed directly on the trainer (`e7_bench.py`, 512 envs, no pool, TF32): **rollout 0.266 s, update
0.361 s** per 65,536-step iteration. torch.profiler inside each phase:

* **The update is GPU-bound:** 0.357 s of kernel time in 0.364 s of wall, 25,443 kernels.
* **The rollout is CPU- and sync-bound:** the GPU is busy 0.103 s of 0.33 s. Every step is serial
  -- build observations (C++), copy them, forward, sync the actions back, Python bookkeeping, step
  -- and nothing overlaps.

## The update's largest single cost is the per-country head, and most of it is redundant

`pe_country` is a 90 → 64 → 1 MLP run on all 84 countries of every sample: 344,064 rows per
minibatch of 4,096. Its input is `cat[raw country slots (26), ctx (64)]`, where `ctx =
pe_trunk(h)` is **the same vector for all 84 countries**. Two consequences:

* **No tensor cores.** K = 90 is not 8-aligned, so cuBLAS runs it as plain fp32
  (`ampere_sgemm_64x64_tn`, 15% of a forward+backward by itself) despite TF32.
* **The [4096, 84, 90] concatenation and the 64 ctx columns are recomputed 84 times.**

`Linear(cat[x, ctx]) = x·Wxᵀ + (ctx·Wcᵀ + b)` computes the same function with the shared term once
per sample and broadcast (the same parameters, so checkpoints are unaffected). Measured:

| | fwd+bwd, minibatch 4,096 | rollout fwd, 512 |
|:---|---:|---:|
| as built | 4.74 ms | 0.34 ms |
| ctx term split out | **3.59 ms** | 0.32 ms |

Equivalence: on real mid-game positions the two agree to 1.0–1.4e-4 in logits under both TF32 and
exact fp32 (rounding order only). Uniform-random inputs give logits up to ~87, where TF32's 1e-3
relative rounding is ~0.08; that is TF32, not the rewrite.

## The levers, measured on the trainer (512 envs, minibatch 4,096, no pool)

| variant | rollout s | update s | steps/s |
|:---|---:|---:|---:|
| as trained (TF32) | 0.266 | 0.361 | 104k |
| **ctx split** (same function) | 0.267 | **0.289** | **118k (+13%)** |
| bf16 autocast in the update only | 0.265 | 0.263 | 124k (+19%) |
| ctx split + bf16 update | 0.265 | 0.239 | 130k (+25%) |

(Production with the opponent pool runs ~8% below this harness: 96k against 104k.)

## What stops 2×

After the update levers, **the rollout is the floor**: 0.265 s per iteration caps training at
~250k steps/s even with a free update, and at 2.07 ms per step it is already half the iteration.
2× (~0.31 s per iteration) needs both halves at ~0.16 s:

* **Update, 0.36 → ~0.16 s:** ctx split (exact function), then bf16 in the update (P26 measured
  it, never adopted: it changes numerics and needs P26's gates), then `torch.compile` to fuse the
  elementwise kernels that are the other ~40% of the update (opt-in since P26; its A/B was "not
  shown harmless").
* **Rollout, 0.27 → ~0.16 s**, which no numerics option moves:
  * more envs (2,048: +18%) amortise the per-step fixed costs;
  * two per-step host syncs (`actions_t.cpu()` and the critic tracker's `v_win_t.cpu()`) and the per-step
    statistics kernels (218 kernels per step in all, forward included) could be cut to one sync;
  * pipelining two halves of the envs, so the C++ step of one half overlaps the copy and forward
    of the other, hides the GPU part (~0.8 ms of 2.07 ms per step).
* **The structural option:** collect iteration k+1's rollout while iteration k's update runs. The
  GPU is idle ~70% of the rollout and the CPU idle during the update, so the iteration would cost
  max(rollout, update) instead of their sum -- ~250k steps/s with the update levers, ~2.4×. It
  makes the data one update stale, which is a recipe change and needs an A/B.

## Found on the way

`tools/scripts/launch_flags.py` printed `--no-ladder-card-self-attention` and
`--no-ladder-cross-attention` for E7-20-44, which `train.py` rejects, and omitted
`--ladder-head-context` / `--ladder-head-static`, which `--arch ladder` requires. Its output for
an E7 run does not launch as printed.

## The shared-context head under `torch.compile` (2026-10-05)

**The same function.** Against the head as built, E7-20-44@3,600M on 256 real mid-game positions:
log-probs over legal actions and every parameter's gradient differ by 1.5e-4 and 1.7e-5 (relative)
in fp32, and by **3e-13 and 4e-14 in float64** (`data/logs/perf/e7_pe_grad_equiv{,64}.py`). The
fp32 difference is summation order only.

**Timed on the trainer** (`e7_bench.py --compile`, the trainer's own `--compile-update`, which
compiles the update's forwards; the rollout keeps its CUDA graphs). 512 envs, minibatch 4,096, no
pool, TF32; two passes, alternating:

| `--compile-update` | head | update s | steps/s |
|:---|:---|---:|---:|
| off | as built | 0.364 / 0.364 | 102.3k / 102.6k |
| off | shared context | 0.290 / 0.290 | 118.4k / 117.7k |
| default | as built | 0.282 / 0.281 | 118.7k / 117.5k |
| default | shared context | **0.218 / 0.213** | **134.6k / 136.1k** |
| max-autotune | as built | 0.276 / 0.276 | 119.7k / 119.3k |
| max-autotune | shared context | 0.214 / 0.209 | 132.1k / 137.1k |

* **The two gains compose.** Under compile the shared-context head still takes ~0.065 s off the
  update, as it does eagerly (0.074 s): compile fuses the elementwise work but does not remove the
  84-fold duplicated matmul columns.
* **Together: update 0.364 → 0.213 s (−41%), +32% end to end.** Compile alone and the head alone
  are each worth ~+15%.
* **max-autotune buys nothing over default** here (within noise), at a much longer first compile.
* The rollout (0.27 s) is untouched by either and is now ~56% of the iteration.

## The compile A/B on E7: E7-23-44 (2026-10-05)

The owner asked why `--compile-update` is not always on (P26's E4-58 left it "not shown harmless")
and to test it on E7. **E7-23-44** is E7-02-44's recipe from scratch, seed 44, plus
`--compile-update default` (`launch_flags --diff`: only that flag), 1,200M at ~115k steps/s.
Training looked healthy throughout: self-play US share 46–51% per 80M window, entropy and critic AUC
within a few hundredths of E7-02-44's over 560–1,200M (E7-02-44's metrics file starts at 560M).

One field (`data/reports/e7_23_rr.{md,json}`), 1,000 games per side per pair, temperature 0; the
owner's rule against E7-02-44 (eager, same recipe and seed) on the E6-03-44 panel:

| comparison | US (arm / control, Δ ± SE) | USSR (arm / control, Δ ± SE) | head to head |
|:---|:---|:---|:---|
| snapshots 1,160 / 1,180 / 1,200M | 82.3 / 82.8, −0.5 ± 0.6 | 80.4 / 83.0, **−2.6 ± 0.6** | **47.8% ± 0.4** |
| SWA 1,120–1,200M | 85.9 / 86.3, −0.5 ± 0.9 | 83.8 / 86.3, **−2.5 ± 0.9** | **46.2% ± 1.1** |
| reference: E7-03-44 SWA (E7-02-44 branched at 870M, new randomness only) | 87.5 / 86.3, +1.1 ± 0.9 | 85.9 / 86.3, −0.4 ± 0.9 | 50.3% ± 1.1 |

Elo in the same field: SWAs E7-02-44 1613, E7-03-44 1609, **E7-23-44 1596**; snapshots E7-02-44
1561–1575, **E7-23-44 1550–1562**.

* **By the owner's rule compile is not accepted:** level as US, worse as USSR by ~2.5 points (4 SE
  of game noise on the snapshots), and below 50% head to head on both comparisons.
* **What the ± does not cover is the draw of the run.** Compile changes rounding from the first
  update, so E7-23-44 is a different trajectory from step 0 -- in effect another seed. The only
  same-recipe spread on record is E7-03-44, which shares E7-02-44's first 870M and differs by
  +1.1 / −0.4 and 50.3%; a from-scratch replica would spread more, and none exists for E7.
* **The sign now repeats.** E4-58 (3 seeds, E4 network) was −84 Elo against TF32 on the arm means;
  E7-23-44 is ~−15 Elo head to head and −2.5 points in one seat. Each is inside what a seed could
  do; together they lean the same way, with no known mechanism (E7 has no dropout; compile's
  gradients match eager to ~1e-3 relative).
* **Verdict: `--compile-update` stays opt-in.** For +15% throughput the evidence does not clear it.
  An eager from-scratch replica (E7 recipe, another seed) would say whether −2.5 / 47.8% is within
  the seed spread, i.e. whether compile is harmless after all.

## Two arms on one machine: alternation and MPS (2026-10-18)

A13+R48's recipe from its 12,800M state (opponent pool restored, frac 0.3), scratch directories, no
snapshots, 6 minutes per configuration, steady steps/s over the last 4 (`--phase-lock` is
`tools/train.py`'s: the rollout under one flock, the update under another, so two arms take turns).

| configuration | without MPS: per arm / total | with CUDA MPS: per arm / total |
|:---|---:|---:|
| one arm alone | 91.8k | 96.9k |
| two arms, no coordination | 52.3k / 104.5k | **63.0k / 125.9k** |
| two arms, `--phase-lock` | 46.4k / 92.8k | 54.0k / 108.1k |

* **Taking turns does not help.** Under the lock an arm's iteration is 1.32 s against 0.70 s alone:
  each phase runs about as long as a whole solo iteration while the other arm runs the other phase,
  so the rollout and the update are not independent resources. The flag stays (off, numerically
  inert -- `tests/training/test_phase_lock.py`) but is not recommended.
* **MPS is worth +20% an arm when two run** (52k -> 63k each, 1.30x one arm in total). Without it
  the driver time-slices the two processes' kernels, and the rollout's per-step syncs wait for the
  other arm's slice. It was run scoped (a daemon on a private pipe directory, stopped after).
* What remains is CPU-side: the rollout's Python and per-step syncs against the other arm's update
  dispatch. The levers left are the ones above (cutting the rollout's syncs, the overlapped rollout
  -- parked on `wip/overlap-rollout` with an intermittent CUDA race -- and compile / bf16 as recipe
  A/Bs).

## Addendum, 2026-10-28: no slowdown -- runs share one GPU

The owner asked why E8 runs trained at 33-58k steps/s against E7's ~100k. Measured alone on the idle GPU:

| measurement | Oct 5-7 | 2026-10-28 |
|:---|---:|---:|
| bare loop (`e7_bench.py`, E7-20-44's network, 512 envs, no pool), E7-era code (`e1445574`) | rollout 0.266 s, update 0.361 s | rollout 0.236 s, update 0.290 s (124.6k steps/s) |
| same, current code | -- | rollout 0.230 s, update 0.290 s (126.0k steps/s) |
| update GPU kernel time per iteration | 0.357 s | 0.286 s (both codes) |
| full `tools/train.py`, the A8 control's flags, alone | 99k (E7-33-44, Oct 7) | **106.8k** (py-spy run, 300 s) |

Nothing slowed: code, flags, libraries (torch 2.13 / CUDA 13, unchanged since 2026-08-24) and the GPU are as fast as
on Oct 5 or faster. The per-run rates are the GPU shared: two A8 trainers make about what one makes alone (R48 and
its head-norm branch 50k + 49k), the deep sphere trunk beside an A8 run 32k + 33k, and the reads and other sessions'
GPU jobs take their share. A first benchmark the night before read 68k for both codes because another session's
leaderboard evaluation was on the GPU throughout; the MPS daemon (pipe `/workspace/data/mps/pipe`) is not used by
the trainers, which connect to the default pipe and run as plain CUDA clients. Phase shares of the 300 s profile:
update forward+loss 23.0%, backward 15.9%, optimizer and clip 5.2%; rollout C++ step 11.6%, forward 11.3%, other
18.8%; pool / snapshot / eval 6.0%.

So two arms in parallel split the throughput rather than add to it: one arm alone reaches each read in about half
the wall-clock.
