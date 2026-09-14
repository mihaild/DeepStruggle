# Running training on rented GPU hosts

Provider-agnostic: everything here needs an SSH target and nothing else, so it works on Vast.ai,
RunPod, or any box you can reach. See [`../../../research/plans/P11_scaling_out.md`](../../../research/plans/P11_scaling_out.md)
for why the experiment programme needs this.

## What to rent — measured, not guessed

**One run per GPU.** Two concurrent runs on one RTX 4090 produced 7,231 + 5,912 = **13,143
steps/s against 13,169 solo**: the GPU is already saturated, so a second run splits the same
throughput rather than adding to it. Memory would allow two (~10GB each of 24GB); throughput does
not. Rent more GPUs, not bigger ones.

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

`remote_sync.sh` brings home metrics, TensorBoard, `hardware.json`, `train.log` and **every
snapshot** — roughly **500MB per run**, which is everything a tournament or an offline probe
needs. Resume files stay remote unless `--with-resume` is passed: they exist to restart an
interrupted run *on that host*, and only the newest is ever useful. Across the ~19-run programme
that is about 9.5GB home rather than 38GB.

## The image

`Dockerfile` pins Python 3.14, torch 2.13+cu130 and Ubuntu 26.04 to match the local stack. The
nanobind extension links against the exact Python it was built for, so a mismatched base produces
an engine that will not import. Build and push once; the source is *not* baked in, so ordinary
experiment changes need only an rsync, not an image rebuild.

Without the image, `remote_bootstrap.sh` still works on any host that has Python 3.14, cmake and a
compiler — it just pays the ~2.5GB torch download per instance, which is about 9% of a 1.8h run.

## Interruption

Spot instances get reclaimed. Runs snapshot every 600s and support `--resume`, so an interruption
costs at most ten minutes — which is what makes spot pricing usable here. To resume after one,
re-bootstrap the replacement host, push back the newest `resume_*.pt`, and relaunch with
`--resume`. Remember the budget is cumulative: the target is the **total** step count, not the
remaining increment.
