# P25 stress bench: three anti-collapse levers on the λ 0.99 recipe

**2026-09-23. No lever passes on both seeds with a real gain.** Seat balancing is the only one that
helped at all: on seed 5 it both held the collapse off and was much stronger, +151, but on seed 3
it only kept the collapse just under the threshold, at no gain in strength. Slow π_ref from scratch
kept seed 3's self-play balanced without making it stronger, and did nothing on seed 5.
Per-seat advantage normalisation collapsed earlier than its control, as predicted.

The plan is [`../plans/P25_collapse_robustness.md`](../plans/P25_collapse_robustness.md), and the
mechanism behind the levers is in
[`E4_collapse_census_per_seat.md`](E4_collapse_census_per_seat.md).

**Setup.** Every run uses E4-27's recipe: M2d from scratch with `--gae-lambda 0.99`, which
collapsed on both seeds. Each adds exactly one lever and runs to 60M. `launch_flags.py --diff`
against the E4-27 run on the same seed shows the lever and `--train-steps` and nothing else.

## Collapse: USSR share of self-play by 5M bucket

For the seat-balanced runs the share is the pool's **pure** self-play estimate
(`1 − opp_seat_sp_us`). Their logged `ussr_win_rate` includes the pool games the lever steers
onto the weak seat, so it is not a self-play measure (P25, "Which share to read"). For the other
runs it is the logged share, as for their controls. That dilutes the controls symmetrically and
understates how extreme they are, so the test applied to the seat-balanced runs is the stricter
one.

| run | lever | 0–60M, by 5M | peak | buckets ≥ 0.9 |
|:---|:---|:---|---:|:---|
| E4-27-03 | none (control) | .53 .60 .50 .56 .89 .88 .92 .95 .95 .93 .97 .97 | 0.97 | 30–55M |
| E4-35-03 | per-seat advantage norm | .60 .48 .65 .97 .99 .98 .98 .97 .97 .96 .93 .95 | 0.99 | 15–55M |
| E4-36-03 | seat balancing | .50 .49 .58 .57 .67 .85 .88 .78 .84 .88 .89 .86 | 0.89 | none |
| E4-37-03 | slow π_ref (5M) | .52 .45 .47 .60 .49 .39 .51 .76 .68 .51 .54 .56 | 0.76 | none |
| E4-27-05 | none (control) | .52 .36 .41 .50 .70 .75 .84 .82 .90 .89 .87 .92 | 0.92 | 55M |
| E4-36-05 | seat balancing | .51 .54 .52 .30 .55 .78 .79 .74 .68 .62 .76 .75 | 0.79 | none |
| E4-37-05 | slow π_ref (5M) | .50 .59 .69 .62 .76 .72 .78 .81 .88 .85 .86 .92 | 0.92 | 55M |

* **E4-36-03 passes by the letter of the rule only.** It sat at 0.85–0.89 in pure self-play from
  25M to 60M. That is under the 0.9 line, but it is not a healthy run: the seat-balance pressure
  was at its maximum from 24M on, and the learner's win rate against its own pool fell from 0.84
  to 0.48.
* **E4-36-05 swung between seats,** US-weak, then USSR-weak over 14–20M (US 63–70%), then
  US-weak again. The controller overshoots, but the swings stayed shallow.
* **E4-37-05 matched its control exactly,** peaking at 0.92 at 55M.

## Strength at 60M

`data/reports/p25_bench_60M.{md,json}`: 11 players, 100 games per side per pair, temperature 0.
E4-08-0s@60M is the default recipe (λ 0.98) at the same step.

| # | model | Elo | vs E4-27-03@60M | vs E4-27-05@60M | vs E4-08-03@60M | vs E4-08-05@60M |
|---:|:---|---:|---:|---:|---:|---:|
| 1 | E4-03-01@80M (anchor) | 2190.2 | 94 / 92 | 97 / 87 | 65 / 82 | 48 / 79 |
| 2 | E4-08-05@60M (λ 0.98) | 2039.6 | 92 / 84 | 93 / 79 | 44 / 55 | — |
| 3 | E4-08-03@60M (λ 0.98) | 2039.1 | 97 / 75 | 97 / 78 | — | 45 / 56 |
| 4 | **E4-36-05@60M** (seat balancing) | **1907.0** | 91 / 46 | **86 / 56** | 46 / 20 | 33 / 29 |
| 5 | E4-35-03@60M (per-seat norm) | 1764.1 | 98 / 11 | 89 / 7 | 39 / 2 | 43 / 7 |
| 6 | E4-37-05@60M (slow π_ref) | 1764.1 | 93 / 16 | 88 / 11 | 27 / 7 | 25 / 11 |
| 7 | E4-27-05@60M (control) | 1756.4 | 99 / 4 | — | 22 / 3 | 21 / 7 |
| 8 | E4-36-03@60M (seat balancing) | 1751.1 | 95 / 6 | 89 / 8 | 21 / 9 | 16 / 10 |
| 9 | E4-37-03@60M (slow π_ref) | 1749.3 | 94 / 20 | 85 / 16 | 7 / 14 | 9 / 16 |
| 10 | E4-27-03@60M (control) | 1740.6 | — | 96 / 1 | 25 / 3 | 16 / 8 |
| 11 | HeuristicBot | 1500.0 | 19 / 9 | 29 / 3 | 12 / 4 | 17 / 11 |

Against its own control, the cell being the run's win rate **as USSR / as US**:

| lever | seed 3 | seed 5 |
|:---|:---|:---|
| seat balancing | +10; 95 / 6, level on both seats | **+151; 86 / 56, better on both seats** |
| slow π_ref | +9; 94 / 20 | +8; 88 / 11 |
| per-seat norm | +24; 98 / 11 | — |

## Reading

* **Balanced self-play is not the same as a healthy run.** E4-37-03 had the most balanced
  self-play on the bench, 0.56 at 60M, and is no stronger than its collapsed control (+9). Its
  seats are balanced because both are weak: its USSR wins 7% as USSR against the default recipe,
  where the control's USSR wins 25%. This is the mirror image of
  "one-sidedness is not degeneration". Neither direction can be read from the self-play share,
  which is why the rule has a strength half.
* **Every λ 0.99 run has a strong USSR and a weak US,** and every lever leaves it that way except
  seat balancing on seed 5. That run's US wins 56% against the control's USSR, where the other
  runs' US wins 1–20%. Seed 5 is the only case where a lever changed what the run learned.
* **Nothing reaches the default recipe.** The best bench run, E4-36-05, is 133 below λ 0.98 at the
  same step. The bench is a stress test of the levers, not a candidate recipe.
* **Slow π_ref from scratch is again not a gain,** as E4-26 found (−168 / −68 at 80M). Its benefit
  is late ([`E4_late_dynamics.md`](E4_late_dynamics.md)).

## Verdict for P25

| lever | collapse half | strength half | verdict |
|:---|:---|:---|:---|
| `--seat-balance` | passes on both, seed 3 borderline (0.89) | seed 5 +151; seed 3 level | **partial**: it helps on one seed of two |
| slow π_ref from scratch | seed 3 passes, seed 5 fails | level on both | fails |
| `--per-seat-adv-norm` | fails, earlier than its control | level | fails, as predicted |

Seat balancing changes which games are played but still trains the winning seat at full speed. In
E4-36-03 the USSR sharpened while the US kept learning: USSR entropy fell from 1.65 to ~0.9, and
US entropy held at ~1.75. The next lever should act on the winning seat directly. The owner's
proposal of per-seat gradient weights driven by the self-play win rate, a WoLF-style
variable-rate rule, does that.
