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

**Confirmed, and it plateaus (2026-09-18).** Three consecutive points after the drop: 58.0%,
57.5%, 59.0% — mean **58.2%** against the previous five-point band's **73.6%**, a 15.4 pp fall at
200 games a point. Not noise.

The important difference from the starved run is what happens next: this one **stops falling**.
The starved arm went to 24.0% and kept going; this settles at ~58% and holds, with the US seat at
72–73% and the USSR seat at 42–48%. So a healthy pool does not prevent the decline, but it appears
to **bound** it — consistent with the pool's stated job of keeping a trailing side in winnable
games. Whether it holds at 58% or resumes falling is the thing to read at 35M and 40M.

---

## The healthy-pool decline is one-sided: USSR only (2026-09-18)

Twelve consecutive evaluation points on `E3-35-28`, per seat against `p28_280M`, 100 games a side:

| steps | US | USSR |
|---:|---:|---:|
| 28.77M | 84.0% | **71.0%** |
| 29.03M | 84.0% | 67.0% |
| 29.29M | 85.0% | 55.0% |
| 29.56M | 81.0% | 64.0% |
| 29.75M | 86.0% | 59.0% |
| 30.02M | 68.0% | 48.0% |
| 30.28M | 73.0% | 42.0% |
| 30.54M | 72.0% | 46.0% |
| 30.80M | 77.0% | 44.0% |
| 31.06M | 82.0% | 57.0% |
| 31.26M | 69.0% | 40.0% |
| 31.52M | 82.0% | **26.0%** |

**The US seat does not decline.** It scatters between 68% and 86% with no trend across the whole
window, beginning and ending at 82–84%. The USSR seat falls monotonically from 71% to 26% — a
45 pp collapse — and the overall number is that fall, diluted.

This is the same shape as `E3-31-28`, whose decline was also USSR-specific (75.0% → 39.0%) while
its US seat held. **Two independent search-CE arms, with healthy pools, degrade on the USSR seat
and only the USSR seat.**

That is a far more specific target than "the policy declines", and it is only visible per seat: the
pooled series reads as a vague 77.5% → 54.0% slide and gives no hint that half the policy is
untouched — the exact failure mode
[`../../../method/measurement_pitfalls.md`](../../../method/measurement_pitfalls.md) was written about.

**What it suggests to look at.** The search is *determinized*: it resamples the hidden state before
each search, and in this game the two seats do not hold symmetric hidden information. A
determinization that misleads more often for one seat would produce targets that are worse for that
seat while leaving the other's alone — which is the shape observed. `search_target_entropy` shows
the targets are equally *sharp* across the decline, but sharpness is not correctness, and nothing
measured so far separates target quality by seat. That is the next probe: search-target agreement
with a strong reference, split US against USSR.

---

## Two retractions: it does not plateau, and it is not USSR-only (2026-09-18)

Both of this document's later claims are wrong, and both failed the same way — a trend read off
endpoints of a noisy series, within a window that had not finished.

**It does not plateau at ~58%.** The arm held 54–60% for about 1.5M steps and then resumed falling.
At 33.03M it is **43.0%**, and the last three points read 48.0 / 43.0 / 46.5.

**It is not USSR-only.** Window means over 21 evaluation points, which do not have the
endpoint-sampling failure mode:

| window | US | USSR |
|:---|---:|---:|
| early — 28.31M–29.75M (n=7) | **80.7%** | **62.4%** |
| middle — 30.02M–31.52M (n=7) | 74.7% | 43.3% |
| late — 31.78M–33.29M (n=7) | **63.4%** | **32.9%** |
| **change** | **−17.3 pp** | **−29.6 pp** |

Both seats decline substantially. The USSR seat falls **1.71x** as much, and earlier, but the US
seat loses 17 points and is still falling at the last measurement.

The earlier claim compared the US column's first and last values inside a window that ended at
31.52M, where the last point happened to be **82** — one of its highest samples — next to an early
84. That reads as "flat" and is an artifact of which two samples the window's edges landed on.
Nothing about the analysis was subtle; it simply used two numbers where it should have used
twenty-one.

### What this costs the conclusions above

The strong form of the seat argument is gone. It said: targets degrade equally on both seats
(US +15.9%, USSR +16.0%) while only the USSR seat collapses, therefore target degradation cannot
be the mechanism. With both seats declining, that exclusion weakens to a **quantitative mismatch**
— equal target degradation against a 1.71x difference in win-rate loss — which is suggestive and
no longer decisive. Target quality returns to the candidate list, downgraded rather than restored.

What survives untouched is the separate observation that `E3-31-28` also declined USSR-first, and
the standing 14–16% USSR target-diffuseness handicap, which was measured directly and does not
depend on either retracted claim.

**The lesson is the same one as the target-entropy window**, one level up: this project's
evaluation points carry ±4–5 pp, so *any* claim about a trend needs a window mean and a stated n,
and "the first and last points of my window" is not a trend. Both errors tonight were made by an
instrument that was right and a reading that was lazy.

### The collapse is global and staggered, and the seat ratio is not a quantity (2026-09-18)

Six more evaluation points, 27 in total, re-read with the window means the section above
prescribes:

| window | US | USSR |
|:---|---:|---:|
| early — 28.31M–30.28M (n=9) | **78.4%** | **58.6%** |
| middle — 30.54M–32.51M (n=9) | 72.0% | 38.4% |
| late — 32.77M–34.80M (n=9) | **49.4%** | **33.6%** |
| **change** | **−29.0 pp** | **−25.0 pp** |

**The US seat has now lost more than the USSR seat.** The ratio quoted one section earlier, "USSR
falls 1.71x as much", computed on 21 points, is **0.86x** on 27.

The shape is sequential, not seat-specific. The USSR seat falls first and hard (58.6 → 38.4) and
then levels off (38.4 → 33.6); the US seat holds through that (78.4 → 72.0) and then collapses
(72.0 → 49.4). Both end heavily degraded.

So **there is no seat-specific failure to explain** — there is one failure that reaches the seats
in order. Three consequences:

1. The "USSR-only" framing is gone entirely, including the softened version. `E3-31-28`'s
   USSR-first decline now reads as the same staggered pattern caught early rather than as a
   different phenomenon.
2. **Target degradation is no longer a poor fit.** It degrades equally on both seats, and both
   seats collapse. The mismatch that made it look wrong was an artifact of measuring during the
   stagger. It is back to being a live candidate, on the same footing it had before any of this.
3. **A ratio between two quantities that move on different schedules is not a summary of
   anything.** Switching from endpoints to window means fixed the first error and did not fix
   this one: the statistic was better and still described a moment rather than the process. Where
   timing differs per series, report the series.

---

## The healthy pool does not bound the collapse — it delays it (2026-09-18)

Retracting this document's other surviving claim. Window means, 15 points from 33.03M:

| window | US | USSR |
|:---|---:|---:|
| 33.03M–34.01M (n=5) | 57.6% | 33.0% |
| 34.28M–35.26M (n=5) | 31.8% | 30.6% |
| 35.52M–36.50M (n=5) | **14.4%** | **7.6%** |

**`E3-35-28` has collapsed completely.** From **90.0%** against `p28_280M` at 25.03M to roughly
**11%** at 36.5M, both seats, still falling at the last measurement. It is not a decline to a lower
plateau; it is the same destination the starved run reached, arrived at later.

For direct comparison the starved continuation read **24.0%** at 25M — *higher* than this arm's
current 11%, though that run was never continued past 28M, so the comparison is of trajectories
rather than endpoints.

### What survives of this document

**Intact.** The pool bug is real, the mechanism is verified, the attribution is tight: at the
identical step 25,034,752 the starved run scored 24.0% and this one 90.0%, and the only
behavioural code change between them was the pool fix. A resumed run trained against one opponent
it beat 99.7% of the time, and that is worth fixing regardless of what follows.

**Retracted.** Every claim about what the healthy pool *prevents*:

* it does not prevent the collapse — the arm reached ~11%;
* it does not bound the collapse — "plateaus at ~58%" was a pause, not a floor;
* it does not make the failure seat-specific — both seats collapse, in sequence.

**The corrected conclusion is simpler and harder on the treatment.** Search CE at coefficient 0.5
with a 200k reference anchor **collapses completely, with or without a healthy opponent pool.**
The pool changes *when*: starved, it fell apart by 25M; healthy, it peaked at 90% around 25M and
was gone by 36M. An 11M-step reprieve and the same end.

That also reframes the pool bug's significance for this programme. It explains why the collapse
looked so abrupt and so early, and it does **not** rescue the X4b treatment, which was the hope
behind extending the arm at all. `E3-35-28` was launched to ask whether search-CE sustains once
the pool is healthy. It does not.
