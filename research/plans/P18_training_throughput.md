# P18 — why one training arm cannot fill the GPU

**Status: planned, not started.** Runs after `E4-01-01`'s continuation (180M→240M) and
`E4-02-01`'s (240M→320M) finish, so nothing here competes with them for the card.

## The observation that motivates it

Measured 2026-09-19 on an RTX 4090 (24 GB) with 24 CPU cores, by running one arm and then two:

| | one arm | two arms |
|:---|---:|---:|
| throughput | ~13,500 st/s | 8,343 + 6,616 = **~15,000 st/s** |
| GPU utilisation | 55% | 98% |
| GPU memory | 9.3 GB | 16.7 GB |
| host CPU | ~4 cores | ~7.7 cores (load 21 of 24) |

**Doubling the work bought 11% more throughput while utilisation went 55% → 98%.** Both halves are
informative and they point in opposite directions, which is why this needs measuring rather than
guessing:

* 55% utilisation on one arm says the GPU is idle almost half the time — so *something else* is the
  constraint.
* But a second arm, which should have soaked up that idle time, produced almost nothing — so the
  GPU was closer to its real limit than 55% suggested.

The reconciliation is probably that **`utilization.gpu` is the fraction of time at least one kernel
is resident, not occupancy.** A stream of small kernels separated by host synchronisation reads as
~55% while leaving the SMs mostly empty; two such streams interleave to 98% without completing
much more work. If that is right, the single arm is **latency-bound, not compute-bound**, and the
fix is batching and overlap rather than a bigger GPU.

## Two hypotheses already measured, and H3 is dead

Both taken 2026-09-19 **while the two arms held the machine**, so both are contended lower
bounds -- the useful direction, since the gaps only widen on an idle box.

| | measured | as a share of training's 13,500 st/s |
|:---|---:|---:|
| engine, single core (`ts_benchmark`) | **1,439,159 st/s** | **0.94%** |
| forward pass, batch 512 | 309/sec, 3.23 ms each = 158,311 st/s | **8.5%** |
| training, actual | 13,500 st/s = 26.4 forwards/sec | — |

**H3 (the engine is the ceiling) is refuted.** The engine is ~107x faster than training needs, on
one contended core, and accounts for under 1% of the budget. The owner's estimate of >2M st/s
uncontended is consistent with 1.44M measured under load. Buying CPU would not help.

**H1 (the batch is too small) is refuted as stated.** The device manages 309 forward passes a
second at batch 512 while already serving two training arms; the loop asks for 26.4. The model and
batch are not what limits this.

So **~90% of wall time is neither the engine nor the rollout forward pass.** That residue is what
the investigation is actually about, and it is a real split rather than pure waste: PPO's backward
and optimiser passes over the rollout buffer are legitimate compute that neither measurement above
includes. The open question is how much of the 90% is gradient work and how much is overhead.

## Remaining hypotheses, each with a test that can refute it

Ordered cheapest first.

**H0 — the backward and optimiser passes are simply the bulk of the work**, and there is no
pathology. *Test:* `torch.profiler` over ~50 iterations, split by phase. If backward plus optimiser
account for most of the 90%, the answer is that training is training and the only lever is a
cheaper update. This is the null hypothesis and it is tested first so the others are not chased
for nothing.

**H2 — the loop is serial: step envs, then forward, then step envs.** Nothing overlaps, so the GPU
waits on the CPU and vice versa. *Test:* read the rollout loop in `nash_pg.py` for a sync point
between `env.step` and `model.forward`; confirm with a `torch.profiler` trace over ~50 iterations,
which shows the gaps directly. The fix, if confirmed, is double-buffering: step batch *n+1* on the
host while batch *n* is on the device.

**H4 — host-side Python overhead dominates.** Observation assembly, mask handling and the action
codec run per step in Python. *Test:* `py-spy record` against a live run for 60 s. A flame graph
that is mostly `ts_env` / `action_encoder` rather than `forward` settles it.

**H5 — transfer bound.** Observations cross PCIe every step. *Test:* in the profiler trace, compare
`Memcpy HtoD` time against kernel time; and try `pin_memory` plus a non-blocking copy.

## What would make this worth acting on

The arms are ~4 hours each at 13.5k st/s. If H2, H4 or H5 holds and the fix is a 2–3x throughput
improvement, every future sweep gets proportionally cheaper — and the immediate plan calls for
**more seeds and a parameter sweep**, which is exactly the workload that multiplies.

The "buy CPU instead" branch is already closed: the engine is under 1% of the budget, so more
cores would buy nothing for a single arm. Running **more arms concurrently** remains a real option
regardless of what the profiler says -- two arms already yield 15,000 st/s against one arm's
13,500 -- but it scales badly, and a 2-3x fix to one arm is worth more than a 1.1x from a second.

## Method note, so the measurement is not wasted

`steps_per_sec` is meaningless without the GPU *and the concurrency*, which
[`../archive/E3_ladder/findings/throughput.md`](../archive/E3_ladder/findings/throughput.md)
records after a cross-run comparison was confounded by exactly this: `E3-30-28` and `E3-31-28`
shared a card and depressed each other. Every number in this plan must carry both. The two-arm
figures above are labelled as such for that reason.

## Explicitly out of scope

Changing `--num-envs` for a *training* arm is not a free optimisation: it changes the batch
composition, and therefore the gradient, and therefore the run. Any throughput finding that
suggests a different `num_envs` has to be validated as a **training** change with its own arm
before the ladder adopts it.
