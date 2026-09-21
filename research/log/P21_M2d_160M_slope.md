# P21 — M2d from 80M to 160M: the slope, and what a collapse costs

One tournament, 22 entrants, 100 games per side at `temperature 0.0`, run 2026-09-21.
`/workspace/data/reports/P21_M2d_160M_slope.{json,md}`.
Companion to [`../findings/training/side_collapse.md`](../findings/training/side_collapse.md) and
[`P21_M2d_country_head_collapse.md`](P21_M2d_country_head_collapse.md).

Seven M2d seeds appear **at both budgets**, so the 80M→160M difference is within-seed and the
~100 Elo seed spread cancels out of it entirely. The seeds were chosen to span every leg pattern
the extension sweep produced, including the one arm that collapsed after 80M.

## The slope

| seed | 80M | 160M | slope | leg pattern |
|---:|---:|---:|---:|:---|
| 3 | 2104.4 | 2227.2 | **+122.8** | clean / clean |
| 5 | 2023.6 | 2145.3 | **+121.7** | clean / clean |
| 10 | 1992.2 | 2179.0 | **+186.8** | clean / clean |
| 21 | 2014.6 | 2220.3 | **+205.8** | entered L1, clean L2 |
| 9 | 2053.7 | 2212.1 | **+158.4** | clean L1, entered L2, escaped |
| 13 | 2053.3 | 2216.4 | **+163.1** | entered both legs |
| 14 | 2012.0 | 1762.7 | **−249.2** | clean L1, **COLLAPSED** L2 |

**Mean over the six non-collapsed seeds: +159.8 Elo, sd 33.7, sem 13.8.** Every one is positive
and the spread is small against the effect, so M2d is still improving steeply at 80M — the rung
has not plateaued and an 80M measurement understates it.

That mean is **survivor-conditioned**. Folding in the collapse risk observed in the sweep so far
(1 terminal collapse in 12 completed extensions) gives an unconditional expectation of about
**+126 Elo** for extending an arm from 80M to 160M.

## It beats the anchor, and the honest version of that claim

All six non-collapsed 160M arms rate above the P21 anchor `E4-03-01@80M` (2089.7 in this field):
2145.3, 2179.0, 2212.1, 2216.4, 2220.3, 2227.2.

**At matched steps this is not a fair comparison** — 160M against 80M. On the ladder's primary
axis, 80M, M2d's seven seeds average 2036.3 against the anchor's 2089.7, so the anchor is still
**54 Elo ahead**, and only the best M2d seed (2104.4) edges it.

**At matched wallclock the comparison reverses, decisively.** Steps are not the resource; seconds
are, and M2d runs at ~50,000 steps/s against the anchor architecture's 11,732 — **4.3×**:

| arm | Elo | steps | steps/s | GPU-h | vs anchor |
|:---|---:|---:|---:|---:|:---|
| `E4-11-03` | 2227.2 | 160M | 45,004 | 0.99 | **+137.5 Elo for 52% of the cost** |
| `E4-11-21` | 2220.3 | 160M | 51,026 | 0.87 | **+130.6 Elo for 46%** |
| `E4-11-13` | 2216.4 | 160M | 50,643 | 0.88 | **+126.8 Elo for 46%** |
| `E4-11-09` | 2212.1 | 160M | 50,445 | 0.88 | **+122.4 Elo for 47%** |
| `E4-11-10` | 2179.0 | 160M | 50,809 | 0.87 | **+89.3 Elo for 46%** |
| `E4-11-05` | 2145.3 | 160M | 48,550 | 0.92 | **+55.6 Elo for 48%** |
| `E4-03-01` (anchor) | 2089.7 | 80M | 11,732 | 1.89 | — |

Doubling M2d's steps still costs **under half** the anchor's wallclock, because the architecture is
four times cheaper per step. Every one of the six is strictly better on both axes at once.

**Answered 2026-09-21, and not the way this section anticipated.** The anchor was carried to
160M as `E4-12-01` and **lost ~330 Elo**
([`../findings/training/entropy_inflation.md`](../findings/training/entropy_inflation.md)). The
matched-steps gap at 160M is therefore +465.8 to M2d, and that number is misleading on its own:
M2d did not gain 465, the anchor fell. The defensible comparison is against the anchor's best
measured state — **M2d@160M 2203.1 vs anchor@80M 2093.3, +110 Elo**, at twice the steps and 46%
of the wallclock. The anchor's instability is a separate finding and is reported as one.

The paragraph below is kept as written because it states the standard the result had to meet.

**What was missing before the anchor could re-point: an anchor arm at 160M.** The comparison above shows
M2d wins on wallclock, but it does not show M2d wins at matched steps, and the anchor may also
improve with a second 80M. That arm costs ~3.8 GPU-h and is the obvious next measurement. Until it
exists, the defensible claim is *"M2d dominates the anchor per GPU-hour"*, not *"M2d is stronger
than the anchor"*.

## Entering the pinned state and escaping costs nothing detectable

`side_collapse.md` left open whether a survived pinned phase leaves a mark. It does not — if
anything the opposite. The three arms that entered and escaped rank **2nd, 3rd and 4th** at 160M,
inside 15 Elo of the never-pinned leader:

* seed 21 entered in leg 1 (124 pinned rows, `adv_std_raw` to 0.0226) and has the **largest slope
  in the set, +205.8**
* seed 13 entered in **both** legs and rates 2216.4
* seed 9 entered in leg 2 (88 rows, run of 63) and rates 2212.1

Three arms, so this is not a strong claim about the population — but the prior worry, that a
pinned episode leaves lasting damage, has no support at all in these numbers.

## What a collapse costs

Seed 14 rates **1762.7**, −249.2 against its own 80M self and about **−409 against its
counterfactual** (the +159.8 it would have been expected to gain). It lands with the other
collapsed and low rungs — `E4-08-01` seed 1 at 1741.6, M1 at 1758.1, M2e at 1746.3 — confirming
from a second direction that *a collapsed M2d arm ends up worse than not having the mechanism at
all*.

Its per-side split is the signature: **47.6% as USSR against 5.8% as US**, a +41.8 pp gap where
every healthy arm in the field sits between +2.5 and +16.9.

## Side balance did not improve with budget

The 160M arms are stronger but no better balanced: gaps +2.5 to +12.9, against +3.1 to +16.8 at
80M, all USSR-favouring. The anchor remains the only arm near even at **−2.7**, and it is still
the target profile. Extra training buys strength, not symmetry.

## M2d ≈ M2, once the collapsed seed is out of the way

M2d's seven 80M seeds average 2036.3 against M2-with-both-heads at 2050.2 — level within seed
spread, at **1.4× M2's throughput** (50,800 vs 35,744 steps/s). The earlier reading that "M2d sits
at M1's level" came from seed 1, which had collapsed; with clean seeds the country head alone
recovers essentially all of M2.

## Method notes

* Bradley-Terry ratings are **field-relative**. The anchor rates 2089.7 here and 2149.9 in the
  tournament quoted in the plan, unchanged — only deltas *within* this one JSON are meaningful,
  and none of these numbers may be compared against another tournament's.
* `temperature 0.0` (argmax), matching every earlier P21 tournament. It matters here more than
  usual: a collapsed arm's policy entropy differs sharply from a healthy one's, and at
  `temperature 0.1` the sharper policy wins partly on temperature rather than on strength.
* `steps/s` is each run's own median during training, not a tournament measurement, so arms that
  trained alongside other jobs read low — `E4-08-03`'s 42,268 against its siblings' ~50,800 is
  contention, not architecture. The anchor's 11,732 was measured the same way and the 4.3× gap is
  far larger than that noise.
* The 80M rows for seeds 3/5/10/21/9/13/14 are the *source* runs of the corresponding
  continuations, so each pair shares a byte-identical first leg.
