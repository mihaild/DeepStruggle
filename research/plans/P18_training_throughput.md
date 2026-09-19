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

## Hypotheses, each with a test that can refute it

Ordered cheapest first. Stop when one explains the 11%.

**H1 — the batch is too small to fill the device.** 512 envs × 3,824 floats is a small forward pass
for a 4090. *Test:* sweep `--num-envs` over 256 / 512 / 1024 / 2048 for ~2M steps each and plot
steps/sec and utilisation. Near-linear scaling refutes saturation and localises the problem to
underfeeding; a flat line means the device is genuinely busy.

**H2 — the loop is serial: step envs, then forward, then step envs.** Nothing overlaps, so the GPU
waits on the CPU and vice versa. *Test:* read the rollout loop in `nash_pg.py` for a sync point
between `env.step` and `model.forward`; confirm with a `torch.profiler` trace over ~50 iterations,
which shows the gaps directly. The fix, if confirmed, is double-buffering: step batch *n+1* on the
host while batch *n* is on the device.

**H3 — the engine is the ceiling.** The C++ env may simply not produce transitions fast enough.
*Test:* `./build/release/engine/ts_benchmark` gives the engine's standalone rate. If it is near
13.5k st/s per worker, the GPU is irrelevant and the work is in the engine or its binding.

**H4 — host-side Python overhead dominates.** Observation assembly, mask handling and the action
codec run per step in Python. *Test:* `py-spy record` against a live run for 60 s. A flame graph
that is mostly `ts_env` / `action_encoder` rather than `forward` settles it.

**H5 — transfer bound.** Observations cross PCIe every step. *Test:* in the profiler trace, compare
`Memcpy HtoD` time against kernel time; and try `pin_memory` plus a non-blocking copy.

## What would make this worth acting on

The arms are ~4 hours each at 13.5k st/s. If H1 or H2 holds and the fix is a 2–3x throughput
improvement, every future sweep gets proportionally cheaper — and the immediate plan calls for
**more seeds and a parameter sweep**, which is exactly the workload that multiplies.

If instead H3 holds, the conclusion is the opposite and useful in its own way: buy CPU, not GPU,
and run more arms concurrently rather than making one faster.

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
