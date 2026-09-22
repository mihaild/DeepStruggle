# E4 — the collapse census measured entry, not collapse

Seven arms scored `COLLAPSED` were each continued by a further 80M steps, 2026-09-22.
**Five recovered.** The four seeds behind [`side_collapse.md`](../findings/training/side_collapse.md)'s
headline 12.5% rate are all among them.

This supersedes the rate, the strength cost, and the "terminal" framing in that document. The
*entry* phenomenon it describes survives intact and is unaffected.

## Why the test was run

Every arm the study had called COLLAPSED had **less runway after onset than the slowest observed
recovery**:

```
recovered arms needed    7.7M – 75.7M steps from onset to last pinned row
terminal arms had only  18.8M – 53.5M steps of runway after onset
```

Not one had been watched long enough to exclude recovery. So every terminal verdict was
potentially censored, and the census's rate was a statement about where the budget happened to
fall relative to an episode.

## Result

| arm | rung | onset | post-onset runway | outcome |
|:---|:---|---:|---:|:---|
| `E4-08-01@160M` (seed 1) | M2d | 33.3M | 126.7M | **recovered** 0.293 |
| `E4-08-12@160M` (seed 12) | M2d | 30.4M | 129.6M | **recovered** 0.434 |
| `E4-08-22@160M` (seed 22) | M2d | 44.6M | 115.4M | **recovered** 0.386 |
| `E4-08-26@160M` (seed 26) | M2d | 61.2M | 98.8M | **recovered** 0.517 |
| `E4-17-03@240M` (M2.5b s3) | M2.5b | 110.0M | 130.0M | **recovered** 0.565 |
| `E4-08-14@240M` (seed 14) | M2d | 106.5M | 133.5M | COLLAPSED 0.013 |
| `E4-17-04@240M` (M2.5b s4) | M2.5b | 132.0M | 108.0M | COLLAPSED 0.009 |

Each keeps its source `--seed`, so torch's stream is restored and sampling continues. The engine
RNG cannot be carried across a resume, so the deals restart — i.i.d. either way, and a far weaker
perturbation than a seed change, but it means a recovery reads as *"recovered with a fresh deal
stream"* rather than *"recovered in an uninterrupted run"*. Seed 1 bounds that worry: its exit
began **before** the resume (below).

## A collapse that recovers costs nothing in strength

Four M2d arms that collapsed and recovered, against four M2d seeds that never collapsed, all at
160M, same rung, same protocol — `data/reports/P21_arrest_cost.json`:

| group | arms | mean Elo | sd |
|:---|:---|---:|---:|
| recovered from collapse | seeds 1, 12, 22, 26 | **2224.3** | 12.0 |
| never collapsed | seeds 3, 4, 5, 6 | **2225.7** | 40.0 |
| never recovered, at **240M** | seed 14 | **1830.3** | — |

**−1.4 Elo, 0.05 pooled sd.** Four arms lost 30–60M steps to a dead advantage signal and arrived
indistinguishable from four that never lost anything. The arrest is a **pause, not damage**.

The arm that never recovered is **−394 Elo** — with 80M *more* compute than either group. So the
two outcomes are not two points on a scale; they are different events.

## The detector scored seed 1 wrong, and the bug is structural

Seed 1's last pinned row is at **79,429,632**; its budget ended at **80,019,456**. It left the
pinned state **0.59M steps — about 9 logged rows — before anyone looked.**

The terminal criterion is the mean of the last 40 rows. Roughly 31 of those were pinned and 9 were
the recovery, giving 0.016: comfortably "collapsed", for an arm that had already climbed out and
went on to 0.44. **A fixed-width tail window cannot distinguish "still collapsed" from "just
exited"**, and this is the third bug in the same detector, after the one-sided check and the
pinned-fraction-versus-terminal-state confusion.

That also disposes of the "recovered with a fresh deal stream" caveat for this arm: the climb
began inside the source run.

## Nothing measured at onset predicts which outcome follows

Five candidates tested against the full record, all falsified:

| candidate | falsifier |
|:---|:---|
| depth of `adv_std_raw` | M2.5b s6 reached **0.0020** — deeper than either terminal arm — and recovered |
| duration pinned | `E4-08-22@160M` held **323 consecutive** pinned rows, the longest ever recorded, and recovered |
| exact-zero run length | the longest run at exactly 0.0 in the study, **20 rows**, belongs to an arm that recovered to 0.473 |
| exact-zero count | `E4-08-22@160M` spent **217 rows** at exactly 0.0 and recovered; seed 26 was called terminal with a longest zero-run of **2** |
| onset timing | M2.5b s3 entered at **110.0M** and recovered; seed 14 entered at **106.5M** and did not |

**The one quantity that tracks the split is `adv_std_raw` recovery.** Every recovered arm restored
it to 0.10–0.20; both terminal arms never left 0.001–0.013. Seed 14 spent 80M steps making
repeated partial escapes — reaching `us_frac` 0.055, a third of rows un-pinned — and sliding back
each time, with the advantage signal never returning. That is the learning signal dying rather
than the win rate being lopsided, and it is a mechanism rather than a symptom.

## Branching one collapse under new seeds

M2.5b seed 6, `E4-17-06`, is the one collapse with a known window before any censoring question
arises: `adv_std_raw` sits below 0.0313 (the census's collapsed band) from **45.9M to 73.1M**, then
the arm recovers on its own and finishes 160M clean. Four branches were resumed from its states
under **new seeds**, two before the window and two inside it:

| branch | from | seed | budget | `adv_std_raw` < 0.0313 | last 40 rows: `adv_std_raw` / `us_frac` |
|:---|:---|---:|---:|:---|:---|
| parent `E4-17-06` | — | 6 | 160M | 45.9M – 73.1M, 74 rows | 0.221 / 0.335 |
| `E4-17-06-5M.11` | healthy 5M state | 11 | 60M | **never** | 0.224 / 0.152 |
| `E4-17-06-5M.12` | healthy 5M state | 12 | 60M | **never** | 0.187 / 0.084 |
| `E4-17-06-50M.11` | inside the pin, 50M | 11 | 110M | 51.9M – 96.1M, 105 rows | 0.233 / 0.475 |
| `E4-17-06-50M.12` | inside the pin, 50M | 12 | 110M | 52.0M – 76.9M, 63 rows | 0.220 / 0.335 |

* **Entry is not fixed by the 5M state.** Neither branch taken before the window entered it,
  though both ran past the parent's 45.9M onset. At M2.5b's entry rate (3 of 4 seeds) two misses
  in a row happen about 1 time in 16 by chance, so this is suggestive, not established.
* **Once inside, a new seed does not stop escape.** Both in-pin branches left the band; one
  within 4M of the parent's exit, the other 23M later. Escape timing belongs to the seed stream;
  escape itself, here, did not.

Measured from `training_metrics.jsonl` only. No branch has been rated in a tournament.

## What "exactly 0.0" is worth, and what "pinned" measures

17 of 18 arms examined reached `us_frac` exactly 0.0 at some point. It means the US won **none of
the episodes that finished in that logging interval** — typically a few dozen games, not an
infinite sample. A policy at a true 2% win rate produces a genuine 0.0 row about 45% of the time,
so runs of 10–20 consecutive zeros are entirely consistent with a low-but-nonzero rate. The pinned
family of metrics is a noisy read on a continuous quantity, not a state indicator.

`us_episode_frac` is also measured on **self-play**, including pool opponents, not against a
frozen reference — so it cannot by itself separate "one side collapsed" from "the pool is
one-sided". Tournament per-side rates have corroborated it every time (the terminal M2.5b arm wins
**0.0% as US** against a fixed opponent), but the training-time metric alone does not prove it.

## The direction is not random, and it is expected

Across 124,609 logged rows in 41 arms:

```
USSR-dominant rows (US wins ~none)   9,570   7.68%   in 41 arms
US-dominant rows   (USSR wins ~none)     4   0.00%   in  1 arm
```

A 2,400:1 asymmetry — not a symmetric dynamical instability, which would pin both seats at
comparable rates across 41 independent arms. **This is expected, not a defect.** Twilight Struggle
is USSR-favoured at low skill: the USSR dominates the early war and surviving it as US is the
skill-heavy job. The 44,136-game championship corpus is 50.1/49.9, which says the game is balanced
*between strong players* — entirely consistent with a weak agent losing as US.

An earlier draft of this analysis compared these arms against that 50/50 figure and concluded the
pipeline was biased. That was the wrong control and the conclusion is withdrawn.

## What this changes operationally

**Entry is not a reason to retrain.** Five of seven recovered on their own, and retraining them
would have discarded four arms that ended up indistinguishable from clean ones. The detector
should mark an arm as *in an episode* and keep going.

**The alarm should watch `adv_std_raw` failing to recover**, not `us_episode_frac` pinning. The
current alarm fired on all four false cases and would have been silent on none of them; the
proposed one fires on the case that actually costs 394 Elo.

## What survives from `side_collapse.md`

* entry is real, reproducible and bitwise deterministic;
* `adv_std_raw` and `explained_variance` separate *entry* cleanly, and the critic looking excellent
  exactly when the policy is one-sided remains the sharpest instance of "critic quality measures
  the critic, not the policy";
* M2.5b enters on **3 of 3** seeds, which makes it a collapse generator and is unaffected by any
  of this;
* the state is sticky but not absorbing — now demonstrated from a *shared prefix* rather than
  across independent seeds.
