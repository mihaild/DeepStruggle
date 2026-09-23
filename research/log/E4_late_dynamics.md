# E4-29..33 — training-dynamics changes applied only after 160M

**Running; updated as snapshots are rated.** The from-scratch arms
([`E4_dynamics_ref_lambda.md`](E4_dynamics_ref_lambda.md)) rejected both a slow π_ref and λ 0.99.
The question here is whether a parameter matters only early in training. Each arm resumes
`E4-08-03`'s 160M state on seed 3 and runs to 240M with **one** change. `launch_flags.py --diff`
against E4-08-03's own 160→240M leg (`E4-08-03_20260922_202204`) shows that one flag and nothing
else, and every arm restores the same 10-member opponent pool. That leg is the step-matched control.

| arm | change | directory |
|:---|:---|:---|
| E4-29-03 | `--gae-lambda 0.99` | `E4-29-03_20260923_070934` |
| E4-30-03 | `--gae-lambda 0.97` | queued |
| E4-31-03 | `--ref-update-freq 5000000` | `E4-31-03_20260923_070934` |
| E4-32-03 | `--ref-update-freq 100000` | queued |
| E4-33-03 | `--eta 0.05` (KL coefficient, default 0.1) | queued |
| E4-08-03-160M.11 | no change, seed 11: a second same-config baseline | `E4-08-03-160M.11_20260923_072049` |

**Read against the 160M start, not the control.** The control leg dips at 200M below its own start
(it wins 45% as USSR / 30% as US against 160M), so margins against the control are inflated there.
The start does not move.

## 200M

`data/reports/late_at200M.{md,json}`. Win rates are the row's **as USSR / as US**.

| # | model | Elo | vs E4-08-03@200M | vs E4-08-03@160M | vs anchor E4-03-01@80M |
|---:|:---|---:|---:|---:|---:|
| 1 | **E4-31-03@200M** (π_ref 5M) | **2291.1** | 71 / 67 | **74 / 61** | 75 / 82 |
| 2 | E4-08-03@160M (start) | 2187.6 | 70 / 55 | — | 70 / 69 |
| 3 | E4-08-03@200M (control) | 2106.3 | — | 45 / 30 | 46 / 52 |
| 4 | E4-03-01@80M | 2092.8 | 48 / 54 | 31 / 30 | — |
| 5 | **E4-29-03@200M** (λ 0.99) | **1682.3** | 11 / 1 | 11 / 1 | 7 / 1 |
| 6 | HeuristicBot | 1500.0 | 9 / 2 | 6 / 0 | 1 / 1 |

* **A slow π_ref helps when applied late — but by less than this table suggests.** From scratch
  it lost on both seeds (−168, −68 at 80M). From 160M it beats the start 74% / 61%. The large
  margin over the control is mostly the control's own dip; see the seed-11 baseline below,
  against which the lead is ~+43.
* **λ 0.99 is destructive late too.** Self-play US wins fell from 32% to under 2% by 210M, entropy
  rose 1.34 → 1.89, and the learner loses to its own pool (38%). Against frozen opponents both
  seats have gone: 11% as USSR, 1% as US against its own start.

### The control's 200M dip is seed 3's, not M2d's

`E4-08-03-160M.11`, the same 160M state continued with no change under seed 11, rated at 200M
(`data/reports/late_branch_at200M.{md,json}`):

| # | model | Elo | vs E4-08-03-160M.11@200M | vs E4-08-03@160M | vs anchor E4-03-01@80M |
|---:|:---|---:|---:|---:|---:|
| 1 | E4-31-03@200M (π_ref 5M) | 2225.2 | **59 / 60** | 74 / 61 | 75 / 82 |
| 2 | **E4-08-03-160M.11@200M** (seed 11, no change) | **2182.1** | — | **77 / 53** | 68 / 75 |
| 3 | E4-08-03@160M (start) | 2109.2 | 47 / 23 | — | 70 / 69 |
| 4 | E4-08-03@200M (seed-3 control) | 2036.0 | 31 / 33 | 45 / 30 | 46 / 52 |
| 5 | E4-03-01@80M | 2015.9 | 25 / 32 | 31 / 30 | — |
| 6 | HeuristicBot | 1500.0 | 1 / 1 | 6 / 0 | 1 / 1 |

**The seed-11 continuation does not dip.** It beats the 160M start 77% / 53% and the seed-3
control's 200M snapshot 69% / 67%. The dip is seed 3's trajectory.

That shrinks the slow-anchor result: against the healthy seed-11 baseline E4-31-03 wins
**59% / 60%, about +43 Elo**, not the +185 it showed against the dipping control. It is still ahead
on both seats. **It also bears on search distillation**: `E4-28-03`'s +168 to +196 at 190–200M
were against the same dipping control, and its +54 to +70 at 165–180M, before the dip began, is the
fair size ([`E4_search_distillation.md`](E4_search_distillation.md)). Every late arm will be rated
against both baselines.

## 240M, both baselines in one field

`data/reports/late_round1_240M.{md,json}`. Both baselines at 200M and 240M, the shared 160M start,
and search distillation at 200M. Win rates are the row's **as USSR / as US**.

| # | model | Elo | vs seed-11@240M | vs seed-3@240M | vs 160M start |
|---:|:---|---:|---:|---:|---:|
| 1 | **E4-31-03@240M** (π_ref 5M) | **2375.7** | **62 / 64** | **67 / 67** | 76 / 65 |
| 2 | E4-28-03@200M (search distillation) | 2359.0 | 57 / 53 | 76 / 59 | 77 / 58 |
| 3 | E4-31-03@200M | 2312.9 | 58 / 46 | 63 / 54 | 74 / 61 |
| 4 | E4-08-03-160M.11@240M (seed-11 baseline) | 2292.6 | — | 56 / 55 | 62 / 53 |
| 5 | E4-08-03-160M.11@200M | 2274.6 | 42 / 43 | 54 / 53 | 77 / 53 |
| 6 | E4-08-03@240M (seed-3 control) | 2252.7 | 45 / 44 | — | 61 / 53 |
| 7 | E4-29-03@240M (λ 0.99) | 2220.1 | 37 / 45 | 34 / 50 | 65 / 43 |
| 8 | E4-08-03@160M (start) | 2211.1 | 47 / 38 | 47 / 39 | — |
| 9 | E4-08-03@200M | 2154.3 | 32 / 33 | 45 / 32 | 45 / 32 |
| 10 | E4-03-01@80M | 2110.8 | 19 / 38 | 19 / 38 | 31 / 30 |
| 11 | E4-29-03@200M | 1674.2 | 4 / 0 | 4 / 3 | 11 / 1 |
| 12 | HeuristicBot | 1500.0 | 1 / 1 | 0 / 2 | 6 / 0 |

Matched-step Elo against the non-dipping seed-11 baseline:

| | 200M | 240M |
|:---|---:|---:|
| E4-31-03, π_ref 5M from 160M | +38.4 | **+83.0** (+123.0 against seed 3) |
| E4-28-03, search distillation | **+84.4** | — (stopped at 200M) |
| E4-29-03, λ 0.99 from 160M | −600.4 | −72.5 (−32.5 against seed 3) |

* **A slow π_ref applied after 160M is the strongest E4 checkpoint rated so far.** At 240M it
  beats *both* same-config baselines on both seats, 62% / 64% and 67% / 67%, and its margin grows
  from 200M to 240M while the baselines flatten. It costs nothing per step. From scratch the same
  flag lost on both seeds, so **the parameter's effect depends on when it is applied**. One seed.
* **Search distillation's fair size is +84 at 200M**, against the baseline that does not dip.
  Head-to-head at 200M it beats the slow-π_ref arm 62% as USSR / 55% as US; the slow anchor's 240M snapshot
  plays search's 200M one at 52% / 40%. Their relative strength at equal steps past 200M is
  unmeasured.
* **λ 0.99 from 160M collapses and recovers inside the leg**: −600 at 200M, −72 at 240M. Its
  damage is not confined to early training, and neither is its recovery.

### Reading

The simplest account is that **a slow anchor suits a policy that is already good and hurts one
that is still learning fast**. Early on a stale π_ref holds the policy back toward where it was
5M steps ago; late, when the policy moves less, it damps the oscillation measured in
[`P21_M2d_setup_west_germany.md`](P21_M2d_setup_west_germany.md) without costing learning speed.
That is a hypothesis. It predicts that a schedule — 200k early, 5M late — beats both, and that the
switch point matters.
