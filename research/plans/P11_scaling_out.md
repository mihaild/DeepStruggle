# P11 — The experiment programme, and running it in parallel

**Status:** proposed. Nothing rented, no account created, no money spent.
**Depends on:** P10's fix working at all (E3-19-22, in progress).

## Why this needs many runs now

The tournament established that **seed spread is ~95 Elo** at 80M
([`../log/P9_architecture.md`](../log/P9_architecture.md)), which is larger than every
architecture effect measured so far and which reversed the sign of the graph-depth comparison.
Any conclusion about architecture from here needs **several seeds per arm**, and at 3,200 games a
pairing the binomial error is already 0.88pp against a 20.4pp seed spread -- so the budget goes
into more seeds, not more games.

That turns the three questions into a programme rather than a handful of runs.

## The three phases

### Phase 1 — fix the critic collapse on 0 layers

The arm to fix is the E3-17 family (0 graph layers), resumed from its healthy 80M checkpoint.

| run | what | status |
|:---|:---|:---|
| E3-18-22 | control: 160M -> 240M unchanged | **finished; no result written up** |
| E3-19-22 | pooled opponents, frac 0.30, 8 snapshots | **finished**, rated in `arena80_160` ([`../findings/pooling.md`](../findings/pooling.md) §1) |
| — | frac 0.15 | **never launched** |
| — | frac 0.50 | **never launched** |
| — | single frozen 160M opponent, learner locked US (P10 exp 2) | **never launched** |

*Status column updated 2026-09-16 from `data/checkpoints/*/metadata.json` and the arena reports;
the prose below is as written. Phase 2 was in effect run as the 4 × 4 replication — see
[`../findings/pooling.md`](../findings/pooling.md) — and Phase 3, graph depth × 4 seeds, has not
been run, which is why graph depth is still open.*

Judged on `critic_auc`, `critic_brier_skill` and `adv_std_raw` holding up, **not** on the US
recovering -- the runaway is bidirectional, so the target is that neither side's signal dies.

### Phase 2 — does the fix improve stability?

"Stability" here is the seed spread itself. With the best Phase 1 configuration and the unchanged
control, **4 seeds each**, same step budget. The question is whether the fix narrows the ~95 Elo
spread, which is a claim about variance and therefore needs the seeds to measure at all.

8 runs.

### Phase 3 — how it affects the number of layers

Graph depth {0, 1, 2} x 4 seeds, with the fix on. Depth 0 is already covered by Phase 2, so this
adds 8 runs. This is the question the seed finding withdrew, re-asked with enough replication to
answer it.

### Totals

About **19 new runs** at 80M steps each. On the local RTX 4090 that is ~1.8h per run, so ~34
GPU-hours plus four tournaments (~2h). Serially that is a working week of wall clock; the runs are
independent, so it is a few hours if run in parallel.

## Running them in parallel

**The compute is cheap; the setup is the real cost.** At marketplace prices an RTX 4090 is
[$0.11-0.34/hr](https://vast.ai/pricing/gpu/RTX-4090) depending on spot versus on-demand and
verified versus community hosts, so the whole programme is roughly **$7-14 of GPU time**. That is
not the number to optimise.

### What makes this workload a good fit

* **Step-budgeted, not time-budgeted.** Arms are compared at equal `--train-steps`, so a slower or
  faster host does not change the result. This matters more than it sounds: it is what makes
  heterogeneous rented hardware usable at all.
* **Interruption-tolerant.** Snapshots every 600s with `--resume-every-snapshot`, so a reclaimed
  spot instance costs at most ten minutes. Spot pricing is therefore fully usable.
* **The C++ engine is CPU-side.** It links no CUDA, so `build/release` is portable to any host
  with a compatible glibc -- build once, ship the tree, and only PyTorch needs the GPU. No
  per-instance rebuild.
* **Two runs fit per card.** Each uses ~10GB of the 24GB, so three instances give six concurrent
  runs and the programme lands in an afternoon.

### Caveat that constrains how arms are assigned

Different GPU models can differ numerically (TF32 and kernel selection), so **arms that are being
compared should run on the same GPU model**, ideally the same instance type. Assign by arm, not
round-robin across whatever is cheapest at the moment. This is the same class of mistake as the
confounded A/B in [`../log/early_training_signal.md`](../log/early_training_signal.md) §3.1, where
unequal training between arms produced a directional and wrong conclusion.

### Providers

| | price (RTX 4090) | fit |
|:---|:---|:---|
| [Vast.ai](https://vast.ai/pricing/gpu/RTX-4090) | ~$0.11 spot, ~$0.34 community on-demand | cheapest; a peer-to-peer marketplace, so host quality varies. Best fit for fault-tolerant research, which this is |
| [RunPod](https://www.runpod.io/product/cloud-gpus) | ~$0.34/hr community cloud | vetted hosts, container orchestration, network volumes, millisecond billing. Costs a little more for less babysitting |
| [Salad / Theta](https://getdeploying.com/gpus/nvidia-rtx-4090) | ~$0.16-0.23/hr on-demand | cheaper on-demand, thinner tooling |

Both Vast and RunPod take a custom Docker image and offer network volumes that outlive the pod, so
the venv and the built engine are prepared once and mounted by every run rather than rebuilt.
RunPod images generally run on Vast with little modification, so the image is not a lock-in.

**Recommendation: Vast.ai spot for the bulk, RunPod if babysitting becomes the bottleneck.**

### What is needed to start

An account and an API key on whichever provider, and a decision on whether checkpoints come home
or stay remote. Metrics (`training_metrics.jsonl`, the TensorBoard tree) are small and should sync
continuously; snapshots are large and only the ones that feed a tournament need to come back.


## Queued: does `torch.compile` hold up under training?

**Scheduled to run immediately after the pooling replication settles, and not before** -- it
changes the numerics of every arm, so introducing it mid-experiment would confound the comparison
it is meant to accelerate.

A local A/B measured **1.37x** end-to-end on `train_iteration()` (21,716 vs 15,808 steps/s). On
the cheapest qualifying 4090 that is $1.24/arm falling to about $0.90, and it compounds across
every arm of every future sweep, so it is worth more than a one-off saving.

It is **not adopted yet**, for a specific reason: gradients from the compiled and eager paths
agree only to **cosine 0.966**, and that discrepancy is unexplained. A 3.4% angular difference in
the gradient is not obviously harmless in a self-play system that we already know oscillates, and
"it trains fine" is not a measurement. Speed was never the open question.

What the experiment has to establish, in order:

1. **Where the 0.966 comes from.** TF32 matmuls, a fused kernel reassociating a reduction, and a
   genuine bug all produce a number like that and are not equally acceptable. Compare eager,
   eager+TF32-off, and compiled on identical inputs; if TF32 explains it, the remaining gap is the
   thing to chase.
2. **A matched A/B at equal steps.** Two arms differing only in `--compile`, same seed, same
   budget, compared on the pooling experiment's own endpoint (mean |USSR - 0.5| over the final
   40M) rather than on final-checkpoint Elo, which an oscillating quantity makes noisy.
3. **Checkpoint round-trip.** A checkpoint written by a compiled run must load into an eager run
   and score identically. Compiled modules wrap parameter names, and a checkpoint that silently
   loads with a prefix mismatch would be the same class of failure as handing a model the wrong
   observation layout: it returns a number instead of raising.

Only if all three pass does it become the default; the flag stays opt-in until then.


### Resolved 2026-09-15: the gradient discrepancy was dropout, not a defect

Gate 1 is cleared. `torch.compile` computes the same gradients as eager, to float32 precision.

| comparison | cosine | relative L2 |
|:---|---:|---:|
| eager vs eager (train mode) | 1.000000 | 0 |
| compiled vs compiled (train mode) | 1.000000 | 0 |
| eager vs compiled (train mode) | 0.996814 | 8.5e-02 |
| **eager vs compiled (eval mode, dropout off)** | **1.000000** | **1.6e-07** |

The network carries four dropout layers at p=0.05. Each path is individually deterministic under a
fixed seed, but `torch.compile` consumes the RNG stream differently, so the two were computing
gradients for **different dropout masks** -- two valid samples of the same stochastic estimator,
not a disagreement about the answer. With dropout disabled they agree to 1.6e-07, which is float32
rounding.

Two wrong diagnoses were made on the way, and both are worth recording because each looked
convincing:

* **"CUDA graphs under `reduce-overhead`"** -- disproved by `max-autotune-no-cudagraphs`, which
  excludes CUDA graphs and shows the same discrepancy.
* **"A bug, because the error is concentrated"** -- the worst parameters by *relative* error were
  biases, which is the classic artefact of dividing by a near-zero gradient norm; but ranking by
  *absolute* difference showed the largest weight matrices off by 14-78%, which genuinely is not
  rounding. That reasoning was sound and the conclusion still wrong, because the missing control
  was never run: **compiled vs compiled**. Once run, it was bit-identical, which rules out
  nondeterminism in the compiled path and points at the only remaining stochastic element.

The lesson generalises past this gate: when two implementations of a stochastic function disagree,
compare each against *itself* before comparing them against each other.

**Remaining gates before adoption:** the matched A/B at equal steps, and the checkpoint round-trip
(a compiled module prefixes parameter names with `_orig_mod.`, which is exactly the kind of
silent-mismatch hazard gate 3 exists to catch -- it was hit in this very diagnostic).
