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
| E3-18-22 | control: 160M -> 240M unchanged | running |
| E3-19-22 | pooled opponents, frac 0.30, 8 snapshots | running |
| — | frac 0.15 | to run |
| — | frac 0.50 | to run |
| — | single frozen 160M opponent, learner locked US (P10 exp 2) | to run |

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
