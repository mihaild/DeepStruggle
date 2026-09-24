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
