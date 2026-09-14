# Running training on rented GPU hosts

Provider-agnostic: everything here needs an SSH target and nothing else, so it works on Vast.ai,
RunPod, or any box you can reach. See [`../../../research/plans/P11_scaling_out.md`](../../../research/plans/P11_scaling_out.md)
for why the experiment programme needs this.

## What to rent — measured, not guessed

**One run per GPU, unless MPS is enabled.** Two concurrent runs on one RTX 4090 produced
7,231 + 5,912 = **13,143 steps/s against 13,169 solo**.

That result is real but it does *not* prove the GPU is saturated, and it is worth being careful
here because the wrong reading leads to the wrong purchase. Without the CUDA Multi-Process
Service, two processes **time-slice** the device rather than running concurrently, so they would
split throughput even if each one individually left the GPU half idle. The measurement shows
time-slicing, not saturation.

What the other instruments say about saturation on this workload: **8.0GB of 24GB** used per run,
**325W of a 450W** power limit, and SM clocks at **2820 of 3135 MHz**. That is a GPU working hard
but not pinned -- consistent with a small network (a 512-wide trunk over 3,824 inputs) that does
not fill a 4090.

Two practical consequences:

* **To get two runs onto one card at full speed, enable MPS** (`nvidia-cuda-mps-control -d`).
  Without it, rent one GPU per run.
* **VRAM is not the binding constraint.** At 8GB per run, a 12GB or 16GB card fits one run with
  room to spare, so cheaper smaller cards are on the table and the choice should be made on
  **$ per 80M steps**, not on $ per hour and not on VRAM.

**CPU is not free, and on a marketplace it is not automatic.** The engine is a C++ simulator that
links no CUDA, so it runs on the host CPU, and a run measured **2.8 cores sustained while sharing
a GPU** — call it **~5.6 cores at full GPU**, across 62 threads. On Vast you are renting a slice
of a machine and the CPU allocation is a property of the *offer*, so it has to be filtered for:
the field is `cpu_cores_effective`. An instance with two effective cores will run at a fraction of
the speed no matter how good the GPU is.

A reasonable filter, giving headroom over the measured 5.6 cores:

```
vastai search offers 'gpu_name=RTX_4090 num_gpus=1 cpu_cores_effective>=8 \
    cpu_ram>=32 disk_space>=100 reliability>0.98' -o 'dph+'
```

**Arms being compared must run on the same GPU model.** TF32 behaviour and kernel selection
differ between cards, and an architecture comparison split across hardware is confounded in a
directional way rather than a noisy one — the same shape of mistake as the unequal-training A/B in
[`../../../research/log/early_training_signal.md`](../../../research/log/early_training_signal.md) §3.1.
`remote_launch.sh` records the GPU model into `hardware.json` in the run directory at launch, so
this is checkable afterwards instead of assumed.

## The three scripts

```bash
# 1. Prepare a host: ship source, build the engine, prove it imports and matches the sources.
tools/scripts/cloud/remote_bootstrap.sh root@1.2.3.4

# 2. Start a run, detached, with the hardware recorded.
tools/scripts/cloud/remote_launch.sh root@1.2.3.4 E3-20-01 -- \
    --arch v2 --graph-layers 0 --identity-dim 16 --per-entity-heads 64 --self-transform \
    --mode train --reward-scheme blunder_aware --num-envs 512 --train-steps 80000000

# 3. Bring it home, continuously.
tools/scripts/cloud/remote_sync.sh root@1.2.3.4 E3-20-01
```

Then watch it the same way as a local run — `tools/scripts/watch_run.py` reads the *synced* local
directory, so one mechanism covers local and remote:

```bash
tools/scripts/watch_run.py /workspace/data/checkpoints/E3-20-01 --target-steps 80000000
```

## What comes home

Measured on E3-17-22 (160M steps, 34 snapshots): the whole run directory is 2.0GB, of which
resume files are 1.6GB (48MB each) and snapshots 442MB (13MB each); metrics and TensorBoard
together are 25MB.

`remote_sync.sh` brings home metrics, TensorBoard, `hardware.json`, `train.log`, **every
snapshot**, the newest `resume_state.pt`, and a step-tagged resume roughly every
`RESUME_STRIDE` steps (80M by default; `--all-resume` for all, `RESUME_STRIDE=0` for none).

Resume files are not only for restarting an interrupted host -- they are what lets an
experiment branch from a partial result, which is how the P10/P11 arms were started from
E3-17-22's 80M. One per 80M steps is the useful branching granularity and costs ~48MB each.

Selection is by **spacing, not modulo**: real step counts are 5046272, 40042496, 80019456 --
none is an exact multiple of anything, so a `steps % stride == 0` test keeps zero files.
Verified against E3-17-22's 32 real resume files: stride 80M keeps 2, stride 40M keeps 4, and
a modulo test would have kept 0.

## The image

`Dockerfile` pins Python 3.14, torch 2.13+cu130 and Ubuntu 26.04 to match the local stack. The
nanobind extension links against the exact Python it was built for, so a mismatched base produces
an engine that will not import. Build and push once; the source is *not* baked in, so ordinary
experiment changes need only an rsync, not an image rebuild.

Without the image, `remote_bootstrap.sh` still works on any host that has Python 3.14, cmake and a
compiler — it just pays the ~2.5GB torch download per instance, which is about 9% of a 1.8h run.

## Which GPU model

The metric is **$ per 80M steps** = price per hour divided by steps per hour. The local RTX 4090
does 13,169 steps/s solo, so 80M steps is **1.69 hours**, and at $0.11-0.34/hr that is
**$0.19-0.57 per run**.

Anything cheaper per step wins, and VRAM does not decide it -- a run needs 8GB. The candidates
worth pricing, all offered by both [Vast.ai](https://vast.ai/pricing/gpu/RTX-4090) and
[RunPod](https://computeprices.com/providers/runpod), which between them list 29 models including
L4, RTX 3090, RTX 4070 Ti, RTX 5090 and RTX A4000:

| | VRAM | why it is a candidate |
|:---|---:|:---|
| RTX 3090 | 24GB | roughly half a 4090's compute, usually well under half the price |
| RTX 4070 Ti | 12GB | fits one run at 8GB; newer architecture than a 3090 |
| RTX A4000 / L4 | 16 / 24GB | low power, often the cheapest per hour on datacenter-tier hosts |
| RTX 5090 | 32GB | only worth it **with MPS**, where its extra compute could carry two runs |

**This cannot be settled from here.** Steps/s on a card we have never run on is not predictable
from spec sheets -- this workload is a small network plus a CPU-side simulator, which is exactly
the shape that fails to scale with headline FLOPS. The honest way to pick is to rent one hour of
each candidate, run `--train-steps` for a fixed small budget, and divide. That costs well under a
dollar and settles it; guessing from TFLOPS tables does not.

The same measurement answers the MPS question: run two instances with MPS enabled and see whether
combined throughput exceeds the 13,143 steps/s that time-slicing gives.

## Interruption

Spot instances get reclaimed. Runs snapshot every 600s and support `--resume`, so an interruption
costs at most ten minutes — which is what makes spot pricing usable here. To resume after one,
re-bootstrap the replacement host, push back the newest `resume_*.pt`, and relaunch with
`--resume`. Remember the budget is cumulative: the target is the **total** step count, not the
remaining increment.
