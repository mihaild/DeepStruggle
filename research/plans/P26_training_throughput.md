# P26 — training throughput: what is left after the 09-23 pass

**Status:** quick screen done (2026-09-24): [`../log/P26_quick_screen.md`](../log/P26_quick_screen.md). The current focus, since P25 closed.
**Needs approval:** items 1–3 change training numerics. Item 6 is an engine change. The rest are
not.

## Where things stand

The 09-23 pass took a 512-env M2d training process from 39–40k to **70.2k steps/s**, and from
~8.5 to ~1.6 cores. Every step was checked to leave the training arithmetic unchanged
([`../log/training_throughput_cpu.md`](../log/training_throughput_cpu.md)). An iteration of 65,536
steps is now about half rollout and half PPO update. The update is compute-bound: 64 minibatches
of 4,096 at ~8 ms each.

**The engine is not the bottleneck, and needs no work now.**
* **Per core:** a game step is ~0.8 µs, 1.5M steps/s, about 21× current training. Including the
  observation (7.9 µs) and the mask (0.67 µs) a core supplies ~106k env-steps/s, about 1.5×
  current.
* **Across cores:** the batch runner spreads that over all cores with OpenMP.
* **When to revisit:** only if training gets several times faster, or the runner is limited to
  one core.

## Candidates, by expected value

| # | what | expected gain | changes numerics? | cost |
|:---|:---|:---|:---|:---|
| 1 | **TF32 matmuls** (`torch.backends.cuda.matmul.allow_tf32 = True`) | large on the compute-bound update; RTX 4090 tensor cores | yes: 10-bit mantissa in the matmul inputs | one line, plus P11-style gates |
| 2 | **bf16 autocast** in the update, and possibly the rollout | large: halves memory traffic, uses tensor cores | yes, more than TF32; the loss and softmax stay fp32 | small code change, plus gates |
| 3 | **`torch.compile`** of the network (P11) | measured 1.37× end to end on the old v2 network; M2d unmeasured | yes: fusion reorders arithmetic (P11: gradients agree to 1.6e-7 in eval mode) | P11's remaining gates: a matched A/B at equal steps, and a checkpoint round-trip (`_orig_mod.` prefix) |
| 4 | **Faster snapshot evaluation.** The probes run every 5M steps and cost ~15% of a real run's wall time. In a 3M-step profile, `ai/eval/safety.py` alone took 9.6 s, all Python recursion (`_follow_forced`, 1.7M calls) | ~10% of wall time | no | moderate: batch or cache the forced-line walk |
| 5 | **Pinned host memory for the runner's observation buffer** (`cudaHostRegister` on the runner's buffer), with an asynchronous copy | ~0.3 ms of a ~3 ms rollout step | no | small, but the buffer moves when `reset_all(base_seed)` builds a new runner, so the registration must follow it |
| 6 | **Cheaper observation building** in the engine | none at present (see above) | no, if the output is identical | engine change: owner's approval |
| 7 | **Finer `elapsed_seconds`** in `training_metrics.jsonl` | measurement only: it has whole-second resolution, which quantises steps/s (65,536 = one iteration per second) | no | trivial |

## How to adopt a numerics-changing item (1–3)

These are the rules P11 set for `torch.compile`, applied to all three. A changed numeric changes
every arm, so none of these can go into a run in the middle of a comparison.

1. **Eager equivalence where it should hold.** For `compile`, eval mode against eager to float
   rounding. For TF32 and bf16, the size of the logit and gradient differences, reported rather
   than assumed.
2. **A matched A/B at equal steps.** Two arms differing only in the flag, same seed and budget,
   compared per seat on the P25 criteria, not on one snapshot's Elo.
3. **Checkpoint round-trip.** A checkpoint written by a run with the flag loads into an eager run
   and scores identically (for `compile`, the `_orig_mod.` key prefix).

It becomes the default only after all three pass, and until then the flag stays opt-in. The
throughput log records the flag, so runs remain comparable.

## Measurement harness

`/workspace/data/logs/perf/perf_run.sh` runs M2d as on the P25 bench, 512 envs, 3M steps, no
snapshots:
* `REPO=` runs another checkout's Python code;
* `BUILD=` uses another checkout's engine build;
* it refuses to run while any other training or tournament is running;
* it records the load average before and after.

`/workspace/data/logs/perf/train_step_equivalence.py` runs one seeded collect-and-update step and
dumps the metrics and a weight sample, for old/new equivalence. Clean run-to-run noise is about
±5%.
