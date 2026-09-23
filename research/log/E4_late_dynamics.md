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
