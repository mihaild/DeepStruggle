# Throughput — and the two confounds in every rate quoted before this

A steps-per-second figure from this project is close to meaningless without the GPU it ran on and
what else was running beside it. Both confounds were present in the numbers used to cost the search
programme, and they push in the same direction.

## The measurements

Median of the last 30 `st/s` lines in each run's `train.log`, paired with that run's own metadata.

| run | GPU | opponent pool | bootstrap | st/s |
|:---|:---|:---|:---|---:|
| E3-17-26 | **RTX 4090** | no | no | **14,173** |
| E3-21-28 | **RTX 4090** | yes | yes | **10,300** |
| E3-20-28 | RTX 3090 | yes | no | 7,200 |
| E3-20-27 | RTX 3090 | yes | no | 6,207 |
| E3-17-25 | RTX 3090 | no | no | 6,190 |

## Confound 1: two different cards

The project has run on both a 3090 and a 4090 and no rate recorded the difference. E3-17-25 and
E3-17-26 are the *same configuration on different seeds* and differ by 2.3x — entirely hardware.

## Confound 2: concurrency

E3-17-25, E3-20-27 and E3-20-28 started at 03:20, 03:21 and 03:18 on the same day, on the same
3090. Three runs sharing one card. None of those three rates measures an uncontended anything, and
the spread among them (6,190 / 6,207 / 7,200) is contention, not configuration.

## What the clean comparison says

4090 against 4090, which is the only pair with neither confound:

* **14,173** unpooled, no bootstrap
* **10,300** with the opponent pool and the same-perspective bootstrap — **27% slower**

Two extra forward passes per step explain it: the frozen opponent's logits under the pool, and
`V(s_{t+1}, p_t)` for the bootstrap. The two cannot be separated from what exists, because no 4090
run has the pool without the bootstrap. Isolating them needs one such run.

## What this invalidates

**Every wall-clock estimate for a search arm.** Those were built on "end-to-end training runs at
~8,300 steps/s", derived from the E3-20-28 continuation — a 3090 figure, taken while snapshots
were interleaved with evaluation, from a period when three runs shared the card. On a 4090 the
comparable rate is ~10,300 with the pool, so those durations are pessimistic by something like
1.7x.

**It does not touch the relative search results.** The coverage curve, 96 versus 384 simulations,
and privileged versus honest search were each measured against the other on one machine in one
sitting. A common factor on both sides cancels.

**Rule going forward:** a rate is quoted with its GPU and with what else was running, or it is not
quoted. `hardware.json` is written into every run directory and was the thing that resolved this;
nothing had read it before.
