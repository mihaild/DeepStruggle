# Collapse census of the late-dynamics and E4.1 runs, and the first per-seat reading

**2026-09-23.** Every run from the late-dynamics round
([`E4_late_dynamics.md`](E4_late_dynamics.md)), from the λ / π_ref arms before it
([`E4_dynamics_ref_lambda.md`](E4_dynamics_ref_lambda.md)) and from the P23 A/B
([`../plans/P23_merged_influence_E4_1.md`](../plans/P23_merged_influence_E4_1.md)), scanned for
collapse episodes. This is the evidence behind
[`../plans/P25_collapse_robustness.md`](../plans/P25_collapse_robustness.md).

**Criterion.** An episode is a run of consecutive 5M buckets in which one seat wins at most 10%
of self-play games. A bucket is the mean of the per-iteration `ussr_win_rate`. The *drift* start
is the last bucket before the episode where the losing seat still won at least 30%. This is a
stricter, simpler test than `watch_run.py`'s collapse detector, which the earlier census used
([`../findings/training/side_collapse.md`](../findings/training/side_collapse.md)), so the counts
are not directly comparable with it.

## The census

| run | leg | change | episode (US ≤ 10%) | drift from | out by | min `adv_std_raw` |
|:---|:---|:---|:---|---:|:---|---:|
| E4-27-03 | 0 → 80M | λ 0.99 | 30–65M | 15M | 70M | 0.151 |
| E4-27-05 | 0 → 160M | λ 0.99 | 55–95M | 15M | 100M | 0.092 |
| E4-29-03 | 160 → 240M | λ 0.99 | 185–205M | 180M | 210M | 0.149 |
| E4-32-03 | 160 → 240M | π_ref 100k (fast) | 180–195M | 175M | 200M | 0.135 |
| E4-33-03 | 160 → 240M | η 0.05 | 185–190M, 205M | 180M | 195M, 210M | 0.158 |
| E4-08-05 | 160 → 240M | none (control) | 195–240M | 180M | **not out** | 0.086 |
| E4-08-05-160M.11 | 160 → 240M | none, seed 11 | 215–240M | 205M | **not out** (run ended) | 0.089 |
| E4-31-05 | 160 → 240M | π_ref 5M (slow) | 235–240M | 190M | **not out** (run ended) | 0.084 |
| E4.1-01-03 | 0 → 80M | E4.1 view | 25–80M | 5M | **not out** | 0.089 |
| E4.1-01-05 | 0 → 30M (stopped) | E4.1 view | 30M | 15M | **not out** | 0.185 |

**No episode:**

| run | span | change |
|:---|:---|:---|
| E4-08-03 | 0–240M | none (control) |
| E4-08-03-160M.11 | 160–240M | none, seed 11 |
| E4-08-05 | 0–160M | none (control) |
| E4-26-03 | 0–80M | π_ref 5M (slow) |
| E4-26-05 | 0–160M | π_ref 5M (slow) |
| E4-31-03 | 160–240M | π_ref 5M (slow) |
| E4-34-03 | 80–240M | π_ref 5M (slow) |
| E4-30-03 | 160–240M | λ 0.97 |
| E4-28-03 | 160–200M | search distillation |
| E4-28-05 | 80–105M | search distillation |

## What it shows

1. **The US is the losing seat in all 11 episodes, across 10 runs.** Nothing here collapsed the
   other way. The seed-5 E4.1 run did swing to 21% USSR at 10M, but it recovered within 5M and
   then collapsed toward the US.
2. **Loosening the update late brings on a collapse.** From E4-08-03's 160M state on seed 3:
   * λ 0.99, π_ref 100k and η 0.05 each had an episode within 15–25M;
   * the control leg, the seed-11 branch and λ 0.97 had none.
3. **Slow π_ref delays a collapse, but does not prevent one: 1 of 5 arms had an episode.** All
   three seed-5 runs from the 160M state collapsed:
   * the control at 195M;
   * the seed-11 branch at 215M;
   * the π_ref-5M arm (E4-31-05) at 235M, with its drift starting at 190M and ending as deep as
     the control's (`adv_std_raw` 0.084).

   *Corrected on 2026-09-23:* the first version of this log counted E4-31-05 as clean "through
   220M" and read 0 of 5 as prevention. Its last 20M collapsed. The other four slow-π_ref arms had
   no episode: E4-26-03/05 from scratch, E4-31-03, and E4-34-03 to 240M. On seed 3 none of the
   late runs with default settings collapsed either, so they do not discriminate.
4. **E4.1 collapsed early on both seeds, and E4 did not on the same seeds.** That is 2 of 2
   against 0 of 2 over the first 80M, so it is suggestive only.
   [`P23_E4_1_ab.md`](P23_E4_1_ab.md) has the strength side.
5. **The deepest collapses are the ones still unresolved when their runs ended.** These are
   E4-08-05 (from 195M), E4.1-01-03 (from 25M) and E4-31-05 (from 235M, 5M before its run
   ended). They had the lowest `adv_std_raw` in the table: 0.086, 0.089 and 0.084. That fits the
   earlier finding that `adv_std_raw` returning is what marks a recovery
   ([`E4_collapse_is_recoverable.md`](E4_collapse_is_recoverable.md)).

## The first collapse logged per seat

Only `E4.1-01-05` was launched after the per-seat metrics went in (`9c49329`), so it is the only
collapse seen with them. Figures are 1M buckets:

| step | USSR self-play share | `adv_std_us` / `_ussr` | `adv_mean_us` / `_ussr` | `entropy_us` / `_ussr` | explained var. |
|---:|---:|---:|---:|---:|---:|
| 15M | 0.39 | 0.288 / 0.283 | −0.010 / +0.004 | 1.95 / 2.03 | 0.85 |
| 18M | 0.52 | 0.300 / 0.298 | −0.007 / +0.011 | 2.07 / 1.87 | 0.84 |
| 21M | 0.84 | 0.220 / 0.229 | +0.008 / −0.006 | 2.43 / 1.49 | 0.93 |
| 24M | 0.92 | 0.197 / 0.202 | +0.010 / −0.007 | 2.67 / 1.86 | 0.95 |
| 28M | 0.82 | 0.223 / 0.234 | +0.012 / 0.000 | 2.54 / 1.88 | 0.91 |

* **The two seats' advantage spreads stay equal throughout.** The spread shrinks for both at once,
  because the critic gets better as the outcome becomes predictable: explained variance rises from
  0.85 to 0.95. This looks structural. Under zero-sum GAE a TD error at one seat's decision is
  mirrored at the other's, so in the same games neither seat can have a much smaller spread.
* **The per-seat means stay near 0**, at ±0.01 against a std of ~0.2, so the losing seat is not
  fed a biased signal.
* **Only entropy separates the seats:** the losing seat's rises from 1.9 to 2.7, and the winning
  seat's falls.

The shared normalisation already scales both seats' shrinking spread back up to unit size, so the
"shared divisor starves the losing seat" step of P25's hypothesis does not happen. The reading
that fits is **a signal made of noise**. Once the losing seat loses nearly every game, its
advantages measure the critic's leftover error rather than the quality of its moves.
Normalisation scales that up to full size, and the only consistent pressure left on the seat's
policy is the entropy bonus.

That predicts `--per-seat-adv-norm` alone does nothing, and that the levers that work are ones that
give the losing seat games it can win (`--seat-balance`). Slow π_ref (point 3) delayed the
collapse on seed 5 without preventing it.
The P25 bench is built to test those predictions.

## Data

* Census: the `training_metrics.jsonl` of each run directory in
  [`../runs.md`](../runs.md), bucketed at 5M.
* E4.1-01-05: `/workspace/data/checkpoints/E4.1-01-05_20260923_133846`, stopped at 30M. Its
  resume point is `resume_30015488steps.pt`.
