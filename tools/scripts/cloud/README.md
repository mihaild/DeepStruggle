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
links no CUDA, so it runs on the host CPU, and a solo run measures **2.7 cores sustained**, across
62 threads. On Vast you are renting a slice of a machine and the CPU allocation is a property of
the *offer*, so it has to be filtered for: the field is `cpu_cores_effective`. An instance with two
effective cores will run at a fraction of the speed no matter how good the GPU is.

An earlier draft of this file reported ~5.6 cores, by taking the 2.8 cores measured *while two
runs shared a GPU* and doubling it. That extrapolation assumed CPU demand scales with the GPU time
a run gets, and direct measurement of a solo run says it does not: the figure is 2.7, not 5.6. The
number mattered, because it is the one that decides which offers are even considered — the stale
version filtered out cheap low-core hosts that are in fact adequate.

A reasonable filter, giving headroom over the measured 2.7 cores:

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

## What the GPU-selection attempt actually established

Two rented hosts, $0.106 spent, **no throughput number obtained**. Both spent their whole life in
environment setup: an A4000 in Japan never left `loading`, and a 3090 stalled in apt/pip for 40
minutes. The setup cost -- a 3.7GB CUDA image plus a 2.5GB torch wheel before a two-minute
measurement -- dominates everything on a short-lived instance.

It was abandoned rather than pushed, because the corrected local measurements made the decision
much less valuable than it first looked:

* the workload is **96.4% GPU-bound** and needs only **2.7 CPU cores**, so the cheap low-core
  listings are fine and the 4090 at $0.268/hr is already near the cheapest per step;
* the only candidate with a real margin was the 3090, which had to be less than 2.38x slower --
  a coin flip worth perhaps 10-20%;
* `torch.compile` measured **1.37x on hardware already owned**, which is larger than the best
  plausible GPU saving and costs nothing to rent.

If this is revisited, bake the image first. A custom image with torch preinstalled turns 40
minutes of setup into two, and that is the whole problem.

**Teardown: trust the leak guard, not the trap.** `bench_gpu.sh` destroys its instance on `EXIT
INT TERM`, and that works when the script exits or errors. It did **not** work under
`pkill -TERM` while the script was blocked in `ssh` -- the instance kept billing and
`leak_guard.sh` is what caught it. Run the guard whenever an instance exists.

## Interruption

Spot instances get reclaimed. Runs snapshot every 600s and support `--resume`, so an interruption
costs at most ten minutes — which is what makes spot pricing usable here. To resume after one,
re-bootstrap the replacement host, push back the newest `resume_*.pt`, and relaunch with
`--resume`. Remember the budget is cumulative: the target is the **total** step count, not the
remaining increment.


## Measured: cost per 160M-step arm

`$/arm = (160e6 / steps_per_sec / 3600) * $/hr`. Throughput measured by `bench_gpu.sh` at the
`num_envs` each card did best at; **the same 512 everywhere**, since batch size changes the
gradient estimator and cannot be varied per arm.

| GPU | steps/s | note |
|:---|---:|:---|
| RTX 4090 | 12,200 | local reference |
| RTX 4080 | 9,348 | |
| RTX 3090 | 5,072 – 7,336 | **same card, different hosts** |
| RTX A4000 | 5,546 | saturated at 512; 1024 gains nothing |
| RTX 5090 | — | needs CUDA 12.8+; see below |

**Host CPU moves throughput by 45% on identical silicon.** A 3090 on a 13.7-core host ran 5,072
steps/s while another on a 16+ core host ran 7,336. The engine is a C++ simulator on the host CPU,
so `cpu_cores_effective` is a first-class filter, not a footnote — and `$/hr` alone ranks offers
wrongly.

**The cheapest offer is not the price of four.** Renting four concurrent hosts of one model means
paying the *fourth*-cheapest price. The A4000 looked like $0.71/arm on its cheapest host and is
$8.12/arm on its fourth. Always price the arm you will actually rent.

**`dph_total` is the bill; the search price is not.** One arm advertised around $0.16/hr and bills
$0.417 — storage for the 60 GB disk is not in the headline. Read `dph_total` back per instance.

## Marketplace reliability, measured

Provisioning four training hosts took **nine rentals**: roughly one clean provision in five. The
four failures on one slot had four distinct causes, which is why no single filter fixes it:

1. 35-minute image pull that never finished
2. host stuck in `loading` with no status message at all
3. image pulled, reached `running`, sshd accepted then immediately closed the connection
4. `failed to register layer: write /usr/local/lib/python3.12/dist-packages` — host disk fault

Consequences now built into the scripts:

* **`inet_down >= 1000` matters more than it looks.** The 3 GB image pull is the single commonest
  failure, and it is invisible in the price.
* **Fail fast.** A working host reaches ssh in about 7 minutes. Waiting 30+ minutes to learn a host
  is broken costs three times as much as giving up at 15 and re-renting.
* **`ADOPT=<instance-id>`** picks up an instance that timed out but is still downloading. It bills
  from creation either way, so abandoning one at the cap pays for a download and discards it.
* **`KEEP=<id,...>` on `leak_guard.sh`.** Age cannot distinguish a leak from a deliberate six-hour
  arm, and a guard that warns about the arms every five minutes is one whose warnings get ignored.

For work with a deadline, a managed tier is worth pricing against this: at measured throughput a
RunPod Secure 4090 is about $2.51/arm with a 99% SLA, against $2.53/arm actually paid on a
marketplace 3090 that also takes 60% longer per arm. The marketplace is cheapest for throwaway
benchmarks where a failure costs $0.05.

## Benchmark on the training image, always

`bench_gpu.sh` previously built its own venv and pip-installed `torch` from the **cu124** index.
Every 5090 measurement failed with `CUDA error: no kernel image is available for execution`:
Blackwell is sm_120 and needs CUDA 12.8+. Worse than the failure, it meant benchmarks ran on a
torch no training host uses, so the numbers predicted something we never deploy.

Both scripts now use `pytorch/pytorch:2.13.0-cuda13.0-cudnn9-runtime` and its bundled torch —
2.13.0+cu130, the same build as the local venv, whose arch list ends `sm_100, sm_120`. The bench
prints the arch list, so an unsupported card is obvious in the first line rather than five failed
rows later.
