# P26 quick screen: what each throughput candidate is worth (2026-09-24)

The plan is [`../plans/P26_training_throughput.md`](../plans/P26_training_throughput.md). This
screen measures how much each candidate could bring, before any of them goes through the
adoption gates. Nothing here changed training code; every number comes from scripts in
`data/logs/perf/`.

**Setup.** `data/logs/perf/p26_bench.py` builds the real `NashPGTrainer` on M2d with
E4-08-36's `ladder_config` and its 80M weights: 512 envs, buffer 128, batch 4096, 4 epochs,
blunder-aware rewards. It has no opponent pool. It times `collect_rollouts()` and `train_step()`
separately, taking the median of 8–12 iterations after 2 warm-up iterations. The GPU (RTX 4090)
ran nothing else. Run-to-run noise is about ±5%.

## Per-iteration time (65,536 steps)

| variant | rollout s | update s | steps/s | vs fp32 |
|:---|---:|---:|---:|---:|
| fp32, graphs (today) | 0.319–0.335 | 0.480–0.501 | 78–82k | — |
| fp32, no graphs | 0.358 | 0.494 | 77k | |
| **TF32** | 0.309–0.312 | 0.409–0.410 | 91k | **+13–16%** |
| bf16 autocast inside the network | 0.320 | 0.340 | 99k | +24% |
| `torch.compile`, graphs kept for the rollout | 0.330 | 0.436 | 86k | +7% |
| **compile + TF32, graphs kept** | 0.300–0.406 | 0.333–0.349 | 88–104k | **+10–30%** |
| compile + bf16, graphs kept | 0.320 / 0.467 | 0.289 / 0.293 | 86–108k | |
| compile + TF32, no graphs | 0.323 | 0.338 | 99k | |

* **The update is where the gains are.**
  * TF32 takes it from 0.50 s to 0.41 s.
  * compile + TF32 takes it to about 0.34 s.
  * bf16 takes it to about 0.34 s, and compile + bf16 to 0.29 s.
* **The rollout does not move with any numerics option.** It is CPU- and sync-bound.
* **compile makes the rollout erratic.** It measured 0.30–0.41 s across repeats, where every
  eager variant stayed within 0.31–0.34 s. That is unexplained. Before compile is worth
  anything end to end, this needs finding: recompilation, guard checks, or CPU contention from
  inductor's workers.

## How far the numerics move the policy

`data/logs/perf/p26_numerics.py` compares E4-08-36@80M against its own fp32 outputs on 5,120
observations from 60 rollout steps.

| | KL(fp32 ‖ x), mean / max | max \|Δ log p\| over legal actions, median / max | argmax agrees | \|ΔV\| mean / max |
|:---|---:|---:|---:|---:|
| TF32 | 1.2e-7 / 1.7e-5 | 2.2e-3 / 0.17 | 100.00% | 1.0e-4 / 8.6e-4 |
| bf16 autocast | 3.6e-2 / 2.56 | 1.03 / 7.47 | 93.11% | 2.3e-3 / 2.1e-2 |

**TF32 is harmless at inference. bf16 across the whole network is not.** bf16 changes the top
action on 7% of decisions and moves individual log-probabilities by up to 7 nats. It would need
a per-layer investigation (which layers must stay fp32) before it can be tried in a run, and it
is not a quick win.

## Where the rollout's time goes

This is from cProfile over three rollouts (fp32, graphs). The profiler inflates the total to
0.40 s per rollout.

| item | s per rollout | note |
|:---|---:|:---|
| host↔device copies and the waits they absorb | 0.105 | pageable copies are stream-ordered, so this includes the forward itself |
| `env.step` (engine, observation, mask) | 0.088 | a bare random-action loop takes 0.143 s for 128 steps |
| `GraphedForward.stale()`, every step | ~0.05 | walks every parameter and buffer of the net on every `get()` |
| `compute_gae` | 0.044 | on the GPU already, but about 40 small kernels × 128 steps |
| other Python in the loop | ~0.1 | |

**Candidates that leave the arithmetic unchanged:**
* **Check graph staleness once per rollout, not every step.** Parameters are updated in place, so
  a per-step check has nothing to catch. Measured by skipping the check: the rollout drops from
  0.335 to 0.314 s, about +5% on the iteration.
* **Capture `compute_gae` as a CUDA graph.** The kernels would be the same, so the result stays
  bit-identical. Estimated at 30–35 ms, about +4%. Not measured.
* **Page-lock the runner's observation buffer in place** (P26 item 5). `data/logs/perf/p26_h2d.py`
  shows the copy of 512 × 3824 fp32 (7.8 MB) going from 0.39 to 0.33 ms per step. That is 8 ms
  per rollout, about 1%, so it is not worth the complexity.

## Snapshot evaluation

A solo production run, E4-55-36, spent 83 s of wall per 5M steps. 69 s of that was training at
72.5k steps/s. The other ~14 s per snapshot is **17% of wall**. Paired, E4-08-36 spent 6,613 s of
wall for 5,262 s of training (20%).

`data/logs/perf/p26_snapshot_eval.py` breaks one snapshot's evaluation down:

| part | s |
|:---|---:|
| decisive probe (128 games) | 4.0 |
| blunder probes (32 games at τ 0.1 and at τ 1.0) | 1.6 |
| position diagnostics (128 games) | 0.9 |
| matches vs HeuristicBot and RandomBot (50 per side each) | 1.1 |
| matches vs recent snapshots (50 per side, ~1.3 s each, up to 4) | ~5.2 |
| **total** | ~12.8 + saving the snapshot and resume files |

Evaluating every 10M instead of every 5M would halve this, with snapshots still saved every 5M
for tournaments and the pool. Cutting the snapshot matches to 2 recent opponents would save
about 2.6 s. Neither changes training.

## What it adds up to

These are estimates for a solo run on the bench's arithmetic. The production run has a pool,
which adds an opponent forward to the rollout, and paired runs share the GPU, which makes the
update gains count for more.

| stack | iteration s | train steps/s | with snapshot overhead |
|:---|---:|---:|---:|
| today | 0.80 | ~82k | 17% of wall |
| + staleness check per rollout, GAE graph (bit-identical) | ~0.74 | ~89k | |
| + TF32 | ~0.65 | ~101k | |
| + compile (if its rollout can be made steady) | ~0.58 | ~113k | |
| + evaluate every 10M | | | ~9% of wall |

Taken together, wall-clock per training step drops by roughly 35–45%, and about 60% of that
needs no change to numerics or only TF32.

## Adopted: the bit-identical changes and a 10M snapshot interval (`bea6311`, 2026-09-24)

* **What changed:**
  * `GraphCache` checks staleness once per rollout.
  * `compute_gae`'s recursion is replayed as a CUDA graph.
  * `--snapshot-every-steps` defaults to 10M, and the pool grows on its own
    `--pool-every-steps` (default 5M).
  * Snapshot evaluation restores the RNG streams it draws from.
* **Bit identity:**
  * Old and new trees give identical results over three collect+update iterations with a pool
    opponent: every metric, the advantages and the weights.
  * The same run evaluated every 1M and every 2M gives identical results over 5M steps: all 92
    logged metrics on all 77 iterations, and the final weights.
* **End to end** (`perf_run.sh`, 3M steps, three alternating runs each):

| | median per-iteration steps/s | wall s for 3M |
|:---|---:|---:|
| before (`3dcb2f2`) | 64.9k / 63.9k / 64.3k | 61.9 / 61.8 / 62.3 |
| after (`bea6311`) | 66.9k / 70.1k / 72.3k | 60.1 / 59.1 / 56.0 |

That is **+8%** in steps/s. The 10M snapshot interval comes on top, removing about half of the
~17% evaluation overhead.

**The TF32 ablation runs next.** It uses the E4-08 recipe to 80M on seeds 40–42: E4-56 is the
fp32 control, E4-57 runs with `--tf32`, and the two arms of a seed run as a pair.

## The TF32 A/B (E4-56 fp32, E4-57 `--tf32`), 2026-09-24

**Setup:**
* the E4-08 recipe (M2d, λ 0.98, pool 0.3/12), from scratch to 80M, on `bea6311`;
* seeds 40–42, with the two arms of a seed run as a pair;
* `launch_flags.py --diff` against E4-08-36 shows only the seed, the snapshot interval, the
  budget and `--tf32`.

All six runs finished without a crash. The rating field is
`data/reports/p26_tf32_80M.{md,json}`: 33 players, 100 games per side per pair, temperature 0,
HeuristicBot at 1500. Tournaments run in fp32, so this field is also the checkpoint round-trip
check. A TF32-trained checkpoint loads and plays in an ordinary fp32 process.

**Self-play balance** (USSR share by 5M):

| run | 0–80M | peak |
|:---|:---|---:|
| E4-56-40 | .51 .68 .62 .58 .58 .68 .84 .91 .92 .80 .64 .50 .54 .54 .58 .67 | 0.92 |
| E4-57-40 | .54 .54 .68 .64 .71 .62 .54 .62 .65 .58 .67 .62 .53 .59 .49 .48 | 0.71 |
| E4-56-41 | .58 .69 .42 .58 .63 .68 .85 .86 .81 .57 .78 .84 .70 .74 .92 .88 | 0.92 |
| E4-57-41 | .56 .62 .84 .72 .75 .71 .58 .60 .35 .46 .42 .53 .49 .52 .49 .55 | 0.84 |
| E4-56-42 | .51 .66 .61 .51 .73 .63 .62 .79 .73 .69 .67 .65 .69 .74 .77 .80 | 0.80 |
| E4-57-42 | .52 .61 .83 .86 .85 .94 .92 .97 .84 .45 .47 .52 .54 .58 .49 .53 | 0.97 (one bucket, recovered by 50M) |

**Strength:**

| run | 40M | 50M | 60M | 70M | 80M | mean 50–80M |
|:---|---:|---:|---:|---:|---:|---:|
| E4-56-40 (fp32) | 1762 | 1847 | 1936 | 1988 | 2020 | 1948 |
| E4-57-40 (TF32) | 1746 | 1870 | 1971 | 1995 | 1958 | 1948 |
| E4-56-41 (fp32) | 1805 | 1875 | 1853 | 1733 | 1736 | 1799 |
| E4-57-41 (TF32) | 1760 | 1876 | 1966 | 1961 | 2023 | 1956 |
| E4-56-42 (fp32) | 1753 | 1812 | 1909 | 1875 | 1884 | 1870 |
| E4-57-42 (TF32) | 1695 | 1897 | 1980 | 2043 | 2019 | 1985 |

For reference, E4-08-36@80M rates 2034 and E4-08-37@80M rates 1952.

**Head to head, the TF32 arm against its own seed's fp32 arm**, pooled over 16 pairings of the
late snapshots (50–80M):

| seed | TF32 wins | as USSR / as US | Elo |
|:---|---:|:---|---:|
| 40 | 50.4% | 60 / 41 | +3 |
| 41 | 71.1% | 72 / 70 | +156 |
| 42 | 64.1% | 72 / 56 | +101 |
| mean | 61.9% | | +84 |

* **TF32 is at least level on every seed, so it passes the "does no harm" test.**
* **It should not be read as making training better.** TF32 cannot plausibly add strength of
  its own. Each arm is a different trajectory at the 1e-9 level (P25 part 0), and the seed spread
  is ~100 Elo. What the table shows is that the three fp32 draws happened to include two
  chronically USSR-leaning runs.
* **E4-56-41 is another stall without a pin, like E4-08-37.** It lost 139 Elo over 50–80M while
  leaning 0.78–0.92, and its US seat wins 4–13% against the references.
* **The one TF32 pin (E4-57-42, 35–40M) recovered by 50M**, and that run ends as the strongest of
  the six on its late mean.

**Throughput, paired, both arms with the same setting** (`data/logs/perf/pair.sh`, 3M steps):

| | median steps/s per arm | wall s for 3M |
|:---|---:|---:|
| fp32 + fp32 | 41.8k / 42.1k | 86.0 / 86.9 |
| TF32 + TF32 | 48.3k / 48.4k | 74.8 / 75.6 |

That is +15% per arm, the same as solo. A mixed pair shows almost none of it, since the fp32
partner's kernels hold the GPU (E4-57-40 against E4-56-40: 44.6k against 44.0k). The bit-identical
rollout changes (+8% solo) are CPU-side and do not show in a GPU-bound pair at all. TF32 is what
speeds up the two-at-a-time workload.

**TF32 passes all three adoption gates:**
1. **Inference drift:** KL 1e-7 against fp32.
2. **Matched A/B:** level or better on 3 of 3 seeds, per seat.
3. **Checkpoint round-trip:** TF32-trained checkpoints load and play in fp32.

## TF32 is the default (`72db935`), and where throughput stands (2026-09-24)

Every figure here is measured through `tools/train.py`, using `data/logs/perf/perf_run.sh`: M2d,
512 envs, 3M steps, the opponent pool, and the median per-iteration steps/s after 0.5M steps.
"Paired" means two runs at once, which is how the programme runs; each arm is measured.

| level | code | solo steps/s | paired steps/s per arm |
|:---|:---|---:|---:|
| L0: before P26 | `3dcb2f2`, fp32 | 64.4k (64.9 / 63.9 / 64.3) | 39.2k (39.1 / 39.3) |
| L1: bit-identical rollout changes | `bea6311`, `--no-tf32` | 69.7k (66.9 / 70.1 / 72.3), +8% | 41.3k (40.5 / 40.6 / 41.8 / 42.1), +5% |
| **L2: + TF32, now the default** | `72db935` | **86.4k** (86.3 / 85.3 / 87.7), **+34%** | **48.2k** (48.3 / 48.4 / 47.9 / 48.1), **+23%** |
| L3: + `--compile-update max-autotune` (opt-in) | this commit | 95.5k (97.3 / 93.7), +48% | 52.2k (52.2 / 52.2), +33% |
| L3': + `--compile-update default` (opt-in) | this commit | 91.5k (92.7 / 90.3), +42% | 51.9k (51.6 / 52.2), +32% |

**Evaluation overhead**, as a share of wall time in real paired runs:

| snapshot interval | runs | overhead |
|:---|:---|---:|
| 5M | E4-08-36 | 20% |
| 10M | E4-56/57, six runs | 7–10% |

**An 80M paired run, end to end:**
* before P26: 80M / 39.2k / 0.80, about 42.5 min;
* now (L2 at 10M): 80M / 48.2k / 0.915, about 30 min, which is 1.41× faster;
* with compile (L3): about 28 min, 1.52× faster.

The TF32 ablation pairs themselves took 1,901–1,923 s of wall. Those were mixed pairs, one fp32
and one TF32 run.

## `torch.compile`, examined

* **The "erratic rollout" was never compile.** Timed per iteration, the eager rollout is bimodal
  as well, at 0.26 or 0.43 s. In a slow rollout every piece of CPU work doubles (buffer adds, the
  Python loop, env stepping), while the GPU forward does not. The bench scripts omitted
  `OMP_WAIT_POLICY=PASSIVE`, which `tools/train.py` sets before torch loads. With it unset, idle
  OpenMP workers spin, and when one spins on the main thread's hyperthread sibling, the main
  thread halves. Production per-iteration throughput is unimodal. Once warm, compile has 3 graphs
  and no recompiles (`TORCH_LOGS=recompiles`).
* **Compile belongs on the update only.** `--compile-update` compiles the learner's
  per-minibatch forward and the π_ref log-probs, lazily and keyed by the network's identity. The
  rollout keeps its CUDA graphs of the eager network, and the bootstrap forward stays eager, so
  dynamo never runs inside a rollout. Parameters are shared, so checkpoints are the eager module's
  (no `_orig_mod.` prefix).
* **`reduce-overhead` (inductor's CUDA graphs) is out.** It raised a CUDA illegal memory access
  beside the trainer's own graphs. `max-autotune` means `max-autotune-no-cudagraphs`.
* **Update-phase bench** (TF32, passive OpenMP): eager 0.398 s, compile `default` 0.356 s,
  `max-autotune` 0.341 s.
* **Gate 1, numerics** (`data/logs/perf/p26_compile_numerics.py`, E4-08-36@80M, 4,096 real
  observations, TF32):
  * forward, eval mode: KL 6.5e-8 mean and 4.7e-6 max; max |Δ log p| median 1.7e-3; argmax agrees
    on 99.98%;
  * gradients: relative L2 difference 8.8e-4 (cos 0.9999996). That is smaller than the TF32 vs
    fp32 difference in the same measurement (2.0e-3), which the TF32 A/B found harmless.
  * **The comparison has to be made in eval mode.** In train mode two passes of even the *same*
    eager network differ by ~17%. That is the residual blocks' `nn.Dropout(0.05)`
    (`ai/models/coldwar_net_v2.py`), so any train-mode comparison measures dropout noise, not
    compilation.
* **Compile time:** the first iteration took 3–8 s here, with inductor's cache warm on this
  machine. A cold cache costs more once, most for `max-autotune`.

**Status:** compile is opt-in and passes gate 1 (numerics). Gate 2, a matched A/B like TF32's,
has not run. Gate 3, checkpoints, holds by construction.
