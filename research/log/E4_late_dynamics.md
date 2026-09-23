# E4-29..33 — training-dynamics changes applied only after 160M

**Done, 2026-09-23.** The final rating of every arm is the full-field round at the end, [240M, the full field](#240m-the-full-field-every-late-arm-both-seeds-e41). The from-scratch arms
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

## λ 0.97 and π_ref 100k, rated against both baselines

`data/reports/late_round2_240M.{md,json}`. Win rates are the row's **as USSR / as US**.

| # | model | Elo | vs seed-11@240M | vs seed-3@240M | vs 160M start |
|---:|:---|---:|---:|---:|---:|
| 1 | E4-31-03@240M (π_ref 5M) | 2295.0 | 62 / 64 | 67 / 67 | 76 / 65 |
| 2 | **E4-30-03@240M** (λ 0.97) | **2244.3** | 48 / 63 | 58 / 67 | 72 / 62 |
| 3 | E4-30-03@200M | 2233.9 | 51 / 51 | 51 / 60 | 69 / 67 |
| 4 | E4-08-03-160M.11@240M | 2205.8 | — | 56 / 55 | 62 / 53 |
| 5 | E4-08-03-160M.11@200M | 2173.1 | 42 / 43 | 54 / 53 | 77 / 53 |
| 6 | E4-08-03@240M | 2167.1 | 45 / 44 | — | 61 / 53 |
| 7 | E4-08-03@160M | 2128.9 | 47 / 38 | 47 / 39 | — |
| 8 | **E4-32-03@240M** (π_ref 100k) | **2120.9** | 30 / 35 | 31 / 53 | 40 / 47 |
| 9 | E4-08-03@200M | 2055.2 | 32 / 33 | 45 / 32 | 45 / 30 |
| 10 | E4-03-01@80M | 2009.9 | 19 / 38 | 19 / 38 | 31 / 30 |
| 11 | **E4-32-03@200M** | **1714.1** | 9 / 3 | 6 / 5 | 12 / 1 |
| 12 | HeuristicBot | 1500.0 | 1 / 1 | 0 / 2 | 6 / 0 |

Against the seed-11 baseline at matched steps:

| arm, from 160M | 200M | 240M |
|:---|---:|---:|
| π_ref 5M (E4-31-03) | +38.4 (first field) | **+89.2** |
| λ 0.97 (E4-30-03) | +60.8 | +38.5 |
| π_ref 100k (E4-32-03) | **−459.0** | −84.9 |
| λ 0.99 (E4-29-03) | −600.4 (first field) | −72.5 (first field) |

* **Both knobs point the same way, late.** A slower reference helps and a faster one collapses:
  E4-32-03's self-play US share rose to ~95% by 170M, and at 200M it wins 1–12% against frozen
  opponents. A shorter credit horizon (λ 0.97) is mildly positive, a longer one (0.99) collapses.
  The settings that damp the update help past 160M; the ones that loosen it break it. Both
  collapses recover by 240M, as every E4 collapse followed this far has.
* **λ 0.97 is within seed noise.** It is split by seat against the healthy baseline (48% as USSR,
  63% as US), and its lead shrinks from 200M to 240M. One seed; not a result to adopt on.
* **The slow π_ref is the only late change with a clean lead.** It beats both baselines on both
  seats at 240M. The seed-5 replicate (`E4-31-05`) and the earlier switch point (`E4-34-03`, from
  80M) are running.

## The switch at 80M: E4-34-03, interim at 160M

`E4-08-03@80M` resumed with `--ref-update-freq 5000000` (`E4-34-03_20260923_095214`), rated against
the plain 80→160M leg (`data/reports/e4_34_at160M.{md,json}`). That leg has no dip in this range,
so it is a fair baseline.

| # | model | Elo | vs E4-08-03@160M | vs E4-08-03@80M | vs E4-03-01@80M |
|---:|:---|---:|---:|---:|---:|
| 1 | E4-31-03@200M | 2276.5 | 76 / 62 | 79 / 79 | 75 / 82 |
| 2 | **E4-34-03@160M** | **2221.2** | 60 / 36 | 72 / 82 | 64 / 71 |
| 3 | E4-08-03@160M | 2169.9 | — | 68 / 66 | 70 / 69 |
| 4 | E4-34-03@120M | 2128.7 | 50 / 36 | 71 / 67 | 71 / 65 |
| 5 | E4-08-03@120M | 2086.2 | 51 / 32 | 72 / 52 | 57 / 61 |
| 6 | E4-03-01@80M | 2037.9 | 31 / 30 | 42 / 60 | — |
| 7 | E4-08-03@80M | 2020.0 | 34 / 32 | — | 40 / 58 |

**+42.5 at 120M and +51.3 at 160M** against the plain leg at matched steps, about the size of the
160M switch at the same distance from its switch point (+38 at 40M in). At 160M the head-to-head is
split by seat: 60% as USSR, 36% as US. The 80M switch is not clearly better or worse than the
160M one. It continues to 240M, and whether the gain keeps growing, as the 160M switch's did,
decides between them.

## Seed 5: the plain continuation collapses on its own

`E4-08-05_20260923_095214`, the plain seed-5 160→240M leg, collapsed without any change applied.
Self-play US share was 1% at 209M, and the learner won 42% against its own pool. It was recovering by
225M (US share 12%, `adv_std_raw` 0.20). It is therefore a poor sole baseline for `E4-31-05`, and a
second one, `E4-08-05-160M.11` (same state, seed 11), was launched; both are used. The late
collapses are not confined to the loosened arms: an unmodified run does it too.

## 240M, the full field: every late arm, both seeds, E4.1

`data/reports/everything_2026-09-23.{md,json}`: 26 players, 100 games per side per pair, 65,000
games, temperature 0, HeuristicBot anchored at 1500. Head-to-head cells are the row's win rate
**as USSR / as US**. This field is larger than the earlier rounds, so absolute Elo shifts between
reports; compare within this table.

| # | model | Elo | vs E4-08-03@160M | vs E4-08-03-160M.11@240M | vs E4-08-05@160M |
|---:|:---|---:|---:|---:|---:|
| 1 | **E4-31-03@240M** (slow π_ref from 160M) | **2236.7** | 76 / 65 | 62 / 64 | 81 / 74 |
| 2 | **E4-34-03@240M** (slow π_ref from 80M) | **2233.3** | 69 / 75 | 65 / 60 | 76 / 74 |
| 3 | E4-28-03@200M (search distillation) | 2225.4 | 77 / 58 | 57 / 53 | 82 / 68 |
| 4 | E4-30-03@240M (λ 0.97) | 2194.3 | 73 / 62 | 48 / 65 | 74 / 66 |
| 5 | E4-34-03@200M | 2186.7 | 68 / 48 | 68 / 46 | 85 / 69 |
| 6 | E4-08-03-160M.11@240M (seed-3 baseline, seed 11) | 2155.0 | 62 / 53 | — | 73 / 70 |
| 7 | E4-33-03@240M (η 0.05) | 2134.2 | 68 / 41 | 57 / 43 | 84 / 50 |
| 8 | E4-34-03@160M | 2130.5 | 60 / 36 | 57 / 43 | 80 / 55 |
| 9 | E4-08-03@240M (seed-3 control) | 2128.6 | 61 / 53 | 45 / 43 | 72 / 63 |
| 10 | E4-31-05@200M (slow π_ref from 160M, seed 5) | 2112.3 | 64 / 47 | 51 / 35 | 86 / 56 |
| 11 | E4-08-03@160M | 2092.4 | — | 47 / 38 | 73 / 51 |
| 12 | E4-29-03@240M (λ 0.99) | 2081.8 | 65 / 43 | 37 / 46 | 71 / 63 |
| 13 | E4-32-03@240M (π_ref 100k) | 2071.4 | 42 / 49 | 32 / 35 | 60 / 47 |
| 14 | E4-08-03@200M | 2024.2 | 45 / 32 | 31 / 32 | 62 / 28 |
| 15 | E4-08-05@160M (seed-5 start) | 2021.6 | 49 / 27 | 30 / 27 | — |
| 16 | E4-03-01@80M (anchor) | 1986.0 | 31 / 30 | 19 / 37 | 37 / 39 |
| 17 | E4-08-03@80M | 1956.1 | 34 / 32 | 20 / 27 | 45 / 37 |
| 18 | E4-31-05@240M *(collapsed at 235M)* | 1947.9 | 66 / **8** | 51 / **9** | 81 / **6** |
| 19 | E4-08-05@80M | 1908.1 | 31 / 23 | 15 / 22 | 41 / 26 |
| 20 | E4-08-05@200M *(collapsed from 195M)* | 1842.2 | 53 / **1** | 25 / **6** | 48 / **8** |
| 21 | E4-08-05@240M *(collapsed)* | 1790.9 | 20 / **1** | 19 / **2** | 43 / **1** |
| 22 | E4-08-05-160M.11@240M *(collapsed from 215M)* | 1778.3 | 34 / **0** | 26 / **0** | 40 / **1** |
| 23 | E4.1-01-03@80M *(collapsed from 25M)* | 1696.9 | 5 / 1 | 1 / 1 | 14 / 2 |
| 24 | E4-33-03@200M *(inside its episode)* | 1681.9 | 12 / 1 | 2 / 0 | 12 / 0 |
| 25 | E4-08-05-160M.11@200M *(USSR-weak swing)* | 1680.7 | **0** / 29 | **0** / 23 | **0** / 18 |
| 26 | HeuristicBot | 1500.0 | 6 / 0 | 1 / 1 | 2 / 4 |

### Strength of each change, at 240M unless stated

Against the seed-3 control at the same step (E4-08-03@240M, 2128.6) and against the second seed-3
baseline (E4-08-03-160M.11@240M, 2155.0):

| change | vs control | vs seed-11 baseline | note |
|:---|---:|---:|:---|
| slow π_ref from 160M (E4-31-03) | **+108** | **+82** | best in the field |
| slow π_ref from 80M (E4-34-03) | **+105** | **+78** | ties the 160M switch; +163 over the control at 200M, where the control dips |
| search distillation, 160→200M (E4-28-03@200M) | +201 at 200M | — | against the dipped 200M control; +133 over the 160M start |
| λ 0.97 (E4-30-03) | +66 | +39 | |
| η 0.05 (E4-33-03) | +6 | −21 | 1682 at 200M, inside its collapse episode |
| λ 0.99 (E4-29-03) | −47 | −73 | collapsed at 185–205M |
| π_ref 100k (E4-32-03) | −57 | −84 | collapsed at 180–195M |
| E4.1 view, from scratch, at 80M (E4.1-01-03) | −259 against E4-08-03@80M | — | collapsed from 25M; see [`P23_E4_1_ab.md`](P23_E4_1_ab.md) |

Seed 5, against its 160M start (E4-08-05@160M, 2021.6), because every seed-5 continuation
collapsed before 240M:

| run | 200M | 240M |
|:---|---:|---:|
| plain continuation (E4-08-05) | −179 | −231 |
| seed-11 branch (E4-08-05-160M.11) | −341 (the USSR seat was losing) | −243 |
| slow π_ref from 160M (E4-31-05) | **+91** | −74 (collapsed at 235M) |

### Reading

* **Slow π_ref switched on late is the one change that helps on both seeds.**
  * On seed 3, it is worth about +80 over the better baseline, and switching at 80M is as good as
    switching at 160M.
  * On seed 5, it was the only continuation still stronger than its start at 200M (+91), while
    both baselines had lost 180–340.
  * It did not prevent the collapse, though; it only delayed it
    ([`E4_collapse_census_per_seat.md`](E4_collapse_census_per_seat.md)).
* **A collapse is the largest effect in the table.** The collapsed snapshots win 0–9% as the US
  against every healthy reference. One seat going wrong takes 180–340 Elo off a run, several times
  what any training change here adds. That is the case for P25 before anything else.
* **Loosening the update late loses** (λ 0.99, π_ref 100k), through the collapse each one caused.
  **λ 0.97 is a modest gain**, +39 to +66, and has not been replicated.
* **Search distillation at 200M is the third-strongest player**, over only 40M of training. Its
  control at the same step is the dipped E4-08-03@200M, so the fair size is its margin over the
  160M start, **+133**. The earlier estimate of +54..+84 came from a smaller field.
