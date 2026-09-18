# The X4b collapse was opponent-pool starvation, from a path bug in resume

**Measured 2026-09-18.** The dense-snapshot replay `E3-34-28_20260917_222015` was launched to
photograph the X4b collapse at 250k resolution instead of 5M. **It did not collapse.** That
non-result turned out to be the result.

## The replay peaks exactly where the original fell over

Same resume state (`E3-29-28_20260917_041110` at 20,054,016 steps), same seed, same flags. Per-seat
win rate against the frozen anchor `p28_280M`, 100 games a side, identical protocol, **at the
identical step count 25,034,752**:

| | overall | US | USSR |
|:---|---:|---:|---:|
| original continuation `E3-29-28_20260917_074357` | **24.0%** | 36.0% | 12.0% |
| dense replay `E3-34-28_20260917_222015` | **90.0%** | 98.0% | 82.0% |

The replay's full window, 20.25M → 27.8M, never leaves the 73–90% band. There is no cliff in it
at all.

## What actually differed: the pool

`opp_pool_size` and `opp_win_rate_mean`, straight out of each run's `training_metrics.jsonl`:

| run | pool size, 20→25M | pool span | mean win rate vs pool |
|:---|---:|---:|---:|
| `E3-29-28` continuation | **1** | 0.0M | **0.982 → 0.997** |
| `E3-31-28` (from scratch) | 5 | 20.1M | 0.574 |
| `E3-34-28` dense replay | 4 → 12 | 24.8M | 0.546 |

The collapsing run spent the entire window training against **one** opponent — a frozen copy of
its own 20M self — and beating it **99.7%** of the time. That is not self-play with a pool; it is
a policy grinding against a fixed, thoroughly-solved opponent, with almost no advantage signal
left anywhere in the batch.

Its other instruments at 25.03M read the same way: `mean_turn` **3.82** (games ending at turn 4)
and **60.8%** of games ending in DEFCON 1, against the replay's 7.81 and 25.3%.

## The cause: `dirname` of a directory

Before `266e891` the rebuild resolved the run directory as

```python
_run_dir = os.path.dirname(os.path.abspath(resume))
```

`--resume` accepts **either** a `resume_state.pt` path **or** the run directory. Given the
directory, `dirname` returns its *parent* — `/workspace/data/checkpoints` — which contains run
directories and no `snapshot_*steps.pt` files at all. The glob found nothing, `_seed_paths` stayed
empty, and control fell through to the branch meant for a *fresh* run: seed the pool with one
frozen copy of the current policy.

Verified directly: on the real source directory the old expression matches **0** snapshots and the
new one matches **4**.

The `resumed_from` field separates the two invocation styles cleanly, and they behave completely
differently:

| run | what was passed | pool |
|:---|:---|---:|
| `E3-20-28_20260915_064054` | `.../E3-20-28/resume_state.pt` — a **file** | 12 ✓ |
| `E3-26-28_20260917_074533` | `.../E3-26-28_20260917_025539` — a **directory** | starved early, max opp WR 0.999 |
| `E3-29-28_20260917_074357` | `.../E3-29-28_20260917_041110` — a **directory** | **1**, opp WR 0.998 |

Two documented, equally valid ways of naming the same checkpoint, producing two different training
regimes, with nothing in the log saying which one you got.

## Why this is attributable, and not just correlated

`E3-29-28`'s continuation and the replay differ in exactly two things, and both act on the pool:

1. **Base commit.** Between `ef6ec61` and `b8711a2` the only commits touching `ai/training/` are
   `3cf4b18` and `f099cc6` (metrics only), `da4ee60` and `2d24830` (new flags, both default-off
   and recorded off in the replay's metadata), and **`266e891`, the pool fix**. The search-target
   off-by-one fix is *not* in this window, so it is identical in both runs and cannot explain the
   difference.
2. **`--snapshot-every-steps`** 5M vs 250k — which is itself how a self-growing pool acquires
   members.

So the channel is the opponent pool either way. What this pair cannot separate is *restored on
resume* from *grows 20x faster*; both raise pool size, and both were in the replay.

## What it does not explain

**`E3-31-28` declined at 25M with a healthy pool** — 80.8% → 57.0% overall, and the fall is
USSR-specific (75.0% → 39.0%) while its US seat held. It ran from scratch, pool 5, mean win rate
0.574. Pool starvation cannot be the cause there.

Its instruments point somewhere else entirely: at 25.03M its `kl_div` is **62.8**, against 0.033
on the replay and 0.332 on the starved run. Something in the search-CE arm detaches from the
reference policy hard, and that is a separate pathology from the one diagnosed here.

So the honest summary is: **the severe collapse (to 24%) is pool starvation; a milder decline
(to 57%) survives its removal and is still unexplained.** The three points line up monotonically
with pool size — 1 → 24%, 5 → 57%, 12 → 90% — but those runs differ in other ways (resumed vs
from-scratch), so that ordering is suggestive, not a controlled dose-response.

## What this retires

Two mechanisms were tested against the collapse and exonerated: the CE coefficient (a tenfold
reduction only delayed it) and the reference-refresh interval (a 25x slower anchor halved the
damage but left the cliff in place). Both exonerations were correct and neither was the cause.
The entropy-rises-into-collapse and `grad_frac`-falls observations
([`P15_X4b_search_during_rl.md`](P15_X4b_search_during_rl.md)) were measured on the starved run
and describe a policy with no opponent, not a general property of search-CE training.

**Any conclusion drawn from a run resumed with a directory argument before 2026-09-17 19:15 UTC
should be re-read with the pool size checked first.** The audit above covers every self-pool run
on disk; `E3-29-28_20260917_074357` is the one materially affected, with
`E3-26-28_20260917_074533` starved early before recovering.

## Method note

`opp_pool_size` and `opp_win_rate_mean` were already logged on every run, for the whole time. The
collapse was investigated for two days through the loss, the entropy, the CE gradient share and
the KL, and the answer was sitting in two columns of the same file. The instrument that finally
caught it was a *replay that failed to reproduce* — worth more than any amount of staring at the
run that did.

---

## Important qualification (appended 2026-09-18): a real decline exists, later and milder

`E3-35-28`, the extension of the non-collapsing replay, has begun declining at **~30M with a
demonstrably healthy pool** — capacity 12, span 30M, `opp_win_rate_mean` 0.45–0.49.

Per seat against `p28_280M`:

| steps | overall | US | USSR |
|---:|---:|---:|---:|
| 28.77M | 77.5% | 84.0% | 71.0% |
| 29.03M | 75.5% | 84.0% | 67.0% |
| 29.29M | 70.0% | 85.0% | 55.0% |
| 29.56M | 72.5% | 81.0% | 64.0% |
| 29.75M | 72.5% | 86.0% | 59.0% |
| 30.02M | **58.0%** | 68.0% | **48.0%** |

And the training instruments show the documented collapse signature arriving with it:

| | 29.43M | 30.28M |
|:---|---:|---:|
| `kl_div` | 0.073 | **7.486** (spiking through 2.36, 3.62, 7.49) |
| `entropy` | 0.774 | **0.947**, rising |
| `search_ce_grad_frac` | 0.207 | **0.033**, falling |
| `explained_variance` | 0.804 | 0.665 |
| `mean_turn` | 6.83 | 6.44 |

**So the entry above overstates the case.** What the pool bug explains is the *severity and
timing* of the 20M–25M collapse: 24.0% at 25M, against 90.0% for the same configuration with a
working pool. What it does not explain is a genuine decline that arrives around 30M anyway, from a
much higher peak, with entropy rising and the CE gradient share falling exactly as recorded in
[`P15_X4b_search_during_rl.md`](P15_X4b_search_during_rl.md).

Read together with `E3-31-28` — healthy pool of 5, declining from 20M — the shape now looks like:
**a real failure mode of search-CE training that pool starvation dramatically accelerated and
deepened.** The starved run reached 24%; this one is at 58% and falling from 90%.

Notably the KL bimodality appears here too, at `ref_update_freq` **200k**, where
[`P15_kl_domination.md`](P15_kl_domination.md) observed it at 5M. A stale reference is therefore
*not* necessary for it — it is milder here (7.5 against 47) but the same pattern, which weakens
the pooled-opponent-draw hypothesis offered there and makes the reference interval look like a
severity knob rather than the cause.

**This is three to six evaluation points, each ±4–5 pp, on a run still in flight.** The direction
is consistent across the instruments, which is why it is recorded now, but the magnitude is not
settled and the arm should be read again at 35M and 40M before anything is concluded from it.
