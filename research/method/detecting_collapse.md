# Detecting a seat collapse from internal metrics

Win rate against a frozen opponent is ground truth but arrives only every 5M steps, and **side
balance is not a collapse metric at all** — it cannot distinguish one side collapsing from both
sides improving unevenly. So: what do the training loop's own metrics say?

**The answer, tested on three collapses across two engines: there is no single detector, because
there is no single collapse.** Each failure mode has its own clean signature and is invisible to
the other's instrument — pool starvation shows in `opp_pool_size` and not in `kl_div`; KL
domination shows in `kl_div` and not in the pool, whose pool was healthier than its control's.
What does *not* work in any of them is the family people reach for first: critic quality, entropy,
clip fraction and side balance, all of which either fail to separate or reverse direction between
datasets.

Watch the two cause-specific instruments. Treat the dynamics metrics as a description of what kind
of wrong, never as a trigger.

## The datasets

Two labelled pairs, each a controlled comparison rather than two unrelated runs.

| pair | collapsed | healthy control | differs only in |
|:---|:---|:---|:---|
| **E4** (post-P17 engine) | `E4-01-01`, no pool at all | `E4-02-01` | the opponent pool |
| **E3/X4b** (pre-P17 engine) | `E3-29-28_074357`, pool starved to 1 by a `dirname` bug | `E3-34-28_222015`, dense replay from the same resume state, same seed, same flags | the pool, by accident |

E3's pair is the stronger evidence: it is the same run replayed
(`archive/E3_ladder/log/P15_X4b_collapse_is_pool_starvation.md`).

## What replicated

| metric | E4 collapsed / healthy | E3 collapsed / healthy |
|:---|:---|:---|
| `opp_pool_size` | **absent from the metrics entirely** / present | **1.2** / 10.6 |
| `opp_win_rate_mean` | not logged (no pool) | **0.960** / 0.504 |

Three for three, counting the two E3-17 runs that also ran `opponent_frac 0.0` and also have no
`opp_pool_size` key. A pool of one, or of none, beaten ~96–99% of the time, is the whole
fingerprint. It is visible at iteration 1 and needs no threshold.

## A pooled run can collapse too, and it has a different signature

The pool check above is necessary, not sufficient. **`E3-31-28` declined with a perfectly healthy
pool** — size 3.0 against its control's 1.8, `opp_win_rate_mean` 0.64 against 0.90, so its pool was
*more* diverse than the healthy run's. The pool instrumentation sees nothing.

Its signature is `kl_div`, and it is not subtle. Against `E3-30-28`, launched three minutes earlier
with identical flags:

```
kl_div   DECLINED  0.02 0.17 0.20 0.07 0.10 0.02 0.07 0.09 0.08 0.18  47.46  33.32  106.65
         HEALTHY   0.02 0.14 0.13 0.11 0.13 0.05 0.13 0.15 0.11 0.14   0.05   0.13    0.14
```

Normal at ~0.1 for most of the run, then a **step change of two to three orders of magnitude**.
`archive/E3_ladder/log/P15_kl_domination.md` measured the KL term at 300x the policy gradient on
half that run's iterations; this is the same event seen from the metrics file. Nothing else needs a
threshold — a `kl_div` above ~1.0 is already a hundred times its own normal.

So there are **at least two collapse modes with two unrelated signatures**:

| mode | seen in | caught by | not caught by |
|:---|:---|:---|:---|
| opponent-pool starvation | `E4-01-01`, `E3-29-28`, `E3-17-25/26` | `opp_pool_size`, `opp_win_rate_mean` | `kl_div` (flat: 0.055 vs 0.056 in E4) |
| KL domination | `E3-31-28` | `kl_div` step change | pool metrics (its pool was fine) |

Both runs that hit the second mode used **search** (`search_ce_coef 0.5`), which the E4 runs do not
(`search_ce_coef 0.0`) — so whether it reproduces on the post-P17 engine is **untested**. That is
the open experiment, not a settled result.

One confound to carry: `E3-31-28` ran at 580 steps/s against `E3-30-28`'s 10,274 because the two
shared a GPU. A KL explosion is a per-iteration event and not obviously a throughput artefact, but
the pair is not clean on that axis.

## What did not replicate — including the thing this page previously recommended

An earlier version of this page, written from the E4 pair alone, reported entropy as the strongest
detector (standardised difference 5.6) and recommended watching it along with `clip_frac` and the
PPO ratio statistics. **Tested against E3 that is wrong.**

| metric | E4 | E3 matched pair | verdict |
|:---|:---|:---|:---|
| `entropy` | collapsed **1.40** vs healthy 0.99, d = 5.6 | collapsed 0.6545 vs healthy 0.6409 | gap of 0.014 against trajectories swinging 0.45–0.91: **noise** |
| `clip_frac` | collapsed **lower** (0.17 vs 0.24) | collapsed **higher** (0.251 vs 0.103) | **direction reverses** |
| `adv_std_raw` | collapsed **lower** | collapsed **higher** (0.235 vs 0.172) | **direction reverses** |
| `kl_div` | 0.055 vs 0.056 — nothing | **32.5** vs 0.282 — 115x | E3-specific |
| `logratio_max`, `ratio_negadv_max` | fired at 3M, no false positive | not logged in E3 | untested |

So each collapse had **its own dynamics fingerprint**: E4's policy failed to sharpen and barely
moved; E3-29-28's moved violently, with a KL term 115x its control's
(`archive/E3_ladder/log/P15_kl_domination.md` measured it at 300x the policy gradient). Two
different mechanisms, opposite signatures, same underlying cause.

That is the lesson. A dynamics metric fitted to one collapse describes *that* collapse.

## What is useless in both

| metric | E4 | E3 |
|:---|:---|:---|
| `critic_auc` | 0.75 / 0.74, 0.80 / 0.79, 0.82 / 0.83 | 0.807 / 0.832 |
| `critic_brier_skill` | 0.185 / 0.169 → 0.304 / 0.322 | 0.272 / 0.310 |
| `explained_variance` | 0.775 / 0.809 → 0.889 / 0.882 | — |
| `value_loss` | 0.055 / 0.051 → 0.047 / 0.050 | — |

**Critic quality never separates**, and by some measures the collapsing run scores better. This is
not a paradox: a degenerate policy produces an *easy* prediction problem. Critic quality measures
the critic, not the policy.

`critic_base_rate` is worse than neutral — `max(p, 1−p)`, always ≥ 0.5, carrying **no direction**,
so it cannot name a side. E3 recorded that the answer it appeared to give was *backwards*
(`archive/E3_ladder/log/P15_control_per_seat.md`). A threshold on it fired three times during
E4-02-01 and was wrong every time.

Also not health signals, despite topping the raw E4 ranking: `steps_per_sec_avg` and
`episodes_completed` are configuration artefacts of running without a pool.

## The protocol

1. **Iteration 1** — is `opp_pool_size` in the metrics, and is `opp_win_rate_mean` below ~0.9?
   Absent, or a pool of 1 beaten 96% of the time, is the failure itself. Catches mode 1 only.
2. **Continuously, and it needs no tuning** — `kl_div` above ~1.0. Normal is ~0.02-0.2, so the
   collapse reading of 47-107 is a hundred times its own baseline. Catches mode 2.
3. **Every few snapshots** — the pool growing to capacity, `opp_pool_span_m` rolling forward,
   `opp_win_rate_mean` staying off 1.0.
4. **Watch the remaining dynamics metrics as a *description*, not a trigger.** `entropy`, `clip_frac`,
   `adv_std_raw`, `kl_div`, `logratio_max` are worth printing because when something does go wrong
   they say *what kind* of wrong — but no threshold on them survived contact with a second
   dataset.
5. **Never** — critic quality, `critic_base_rate`, side balance, `steps_per_sec`.
6. **To confirm before acting** — per-seat win rate against a frozen opponent.

## What this still does not establish

**Two modes found, and there may be more.** The pool check is 3/3 on starvation and blind to KL
domination; `kl_div` is the reverse. Each was found by having a labelled control to compare
against. A third cause would likely need a third instrument, and nothing here predicts which.

**The E3 pair is short.** It covers 20–26M steps, because that is where X4b fell over. The E4 pair
runs to 184M. A metric could in principle separate only over long horizons and be invisible in
E3's window.

**`logratio_max` and `ratio_negadv_max` remain untested.** They were the earliest and cleanest
detectors on E4 — firing at 3M with no false positive — and E3 predates them. They are the first
thing to check against the next collapse, and should not be trusted until then.

## `ref_update_freq` is not one of the signals

E3-31-28 carried the slow reference anchor (`ref_update_freq = 5000000`) and collapsed by KL
domination anyway, while E4-02-01 ran the fast default for a clean 320M. Neither setting predicts
the outcome -- see
[`../findings/training/ref_update_freq_open_ablation.md`](../findings/training/ref_update_freq_open_ablation.md).
It is held constant at 200,000 by decision and is a candidate axis for a deliberate sweep.

## The two signatures are mutually blind (47 E3 arms + 5 E4 legs, 2026-09-19)

Watching one metric catches at most one of the two modes:

| arm | pool | max `kl_div` | iters with one side >98% of episodes | mean US share |
|:---|:---|---:|---:|---:|
| E3-19-23 | **none** | **0.06** | **32%** | 0.114 |
| E3-31-28 | healthy | **193.7** | **0%** | 0.462 |

E3-19-23 spent a third of its iterations with one side winning essentially every game while
`kl_div` stayed at a textbook-healthy 0.06. E3-31-28 reached `kl_div` 193.7 with side balance
never once flagging. **`kl_div` alone cannot detect the one-sided mode and side balance alone
cannot detect KL domination**, so `tools/scripts/watch_run.py` watches both and a run is healthy
only when both are.

Not a small-sample artefact: restricting to iterations with >=100 completed episodes moves
E3-19-23 from 31.6% to 32.2% and leaves every other arm unchanged (median episodes per iteration
is 93-224 throughout).

### One-sidedness is a flag, not a verdict

It has a high false-positive rate. **Both** E4 continuations sit at ~13% -- the pooled
E4-02-01 (13.2%) as well as the unpooled E4-01-01 (13.4%) -- and E4-02-01 is the clean 320M arm
that the round robin ranks first. One side genuinely improving faster produces the same number as
one side degenerating, which is why **per-side win rate against a frozen reference remains the
arbiter**; this metric only says where to look. Its value is that it is internal to self-play and
needs no external opponent, unlike win rate against HeuristicBot.

### Two candidates that did NOT replicate

* **`adv_std_raw` < 0.01** (with `explained_variance` -> 1.0, the critic trivially perfect because
  every game ends the same way) marks a contiguous 23-iteration event at ~218-221M in E4-01-01's
  unpooled continuation, US win fraction 0.00 throughout, and never fires in the pooled arm. But
  it has **0 hits across all 47 E3 runs**. E4-specific artefact, not a detector.
* **The entropy gap**, refuted earlier the same way (0.014 on E3, i.e. noise).

Both looked good in-sample and died out-of-sample, which is the standing argument for checking a
proposed metric against the other lineage before acting on it.

### A measurement trap that produced a false finding here

The first version of this section claimed E3-24-28 was 100% one-sided with healthy KL. **That was
an artefact.** E3-24-28 logs `episodes_completed = 0.0` on every row, and the sweep computed
`won_us / max(1, episodes_completed)` -- so "no episodes recorded" was silently read as "the US
won none of them". Any arm whose episode counters are absent will manufacture a perfect
one-sidedness score this way. **Require `episodes_completed > 0` and report the excluded rows**,
rather than defaulting a denominator. E3-24-28 has no usable episode data and cannot be analysed
for side balance at all.
