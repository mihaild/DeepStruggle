# P25 stress bench: anti-collapse levers on the λ 0.99 recipe

> **Step 3b (WoLF seat weights, E4-38) is at the end.** It is the only lever with no collapse on either seed, but as built it fails on strength on seed 5 (−105), because it held the policy's entropy up.

**2026-09-23. No lever passes on both seeds with a real gain.** Seat balancing is the only one that
helped at all: on seed 5 it both held the collapse off and was much stronger, +151, but on seed 3
it only kept the collapse just under the threshold, at no gain in strength. Slow π_ref from scratch
kept seed 3's self-play balanced without making it stronger, and did nothing on seed 5.
Per-seat advantage normalisation collapsed earlier than its control, as predicted.

The plan is [`../plans/P25_collapse_robustness.md`](../archive/E4_ladder/plans/P25_collapse_robustness.md), and the
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

## Step 3b: WoLF seat weights (E4-38), 2026-09-23

`--wolf-seat-weight` at power 1, on the same recipe: each seat's PPO surrogate is scaled by
w_us = 2x and w_ussr = 2(1−x), with x the USSR's smoothed pure-self-play share. **As built, only
the surrogate was weighted**, not the entropy bonus, the KL to π_ref or the value loss.
`launch_flags.py --diff` against E4-27 on each seed shows `--wolf-seat-weight` and `--train-steps`
only.

### Collapse half: passes on both seeds, with the widest margin on the bench

| run | 0–60M, by 5M (logged USSR share) | peak | buckets ≥ 0.9 |
|:---|:---|---:|:---|
| E4-38-03 | .49 .46 .40 .68 .67 .57 .57 .58 .65 .61 .60 .47 | 0.68 | none |
| E4-38-05 | .52 .66 .71 .52 .47 .35 .64 .67 .69 .71 .62 .46 | 0.71 | none |

The seats' entropies stayed level throughout, and `adv_std_raw` held at 0.32–0.35, where the
controls fell to 0.15–0.25. Seed 5 swung between the seats, from 0.35 to 0.71 USSR, as the
weights flipped between 1.43/0.57 and 0.66/1.34.

### Strength half: +62..+131 on seed 3, −45..−105 on seed 5

Two rounds, both at 100 games per side per pair, temperature 0:
* `data/reports/p25_bench_60M_wolf.{md,json}`: the whole bench plus the references at 60M, 13
  players.
* `data/reports/p25_wolf_50_60M.{md,json}`: both WoLF runs and both controls at 50, 55 and 60M,
  so that one snapshot cannot decide it.

The second round:

| seed | step | WoLF Elo | control Elo | Δ | WoLF vs control (as USSR / as US) | WoLF vs λ 0.98 @60M (as USSR / as US) |
|---:|:---|---:|---:|---:|---:|---:|
| 3 | 50M | 1801 | 1739 | +62 | 81 / 43 | 19 / 14 |
| 3 | 55M | 1789 | 1725 | +64 | 88 / 42 | 16 / 22 |
| 3 | 60M | 1840 | 1709 | **+131** | 89 / 48 | 12 / 29 |
| 5 | 50M | 1626 | 1672 | −45 | 72 / 14 | 5 / 11 |
| 5 | 55M | 1633 | 1715 | −82 | 52 / 22 | 4 / 9 |
| 5 | 60M | 1638 | 1743 | **−105** | 70 / 3 | 2 / 7 |

In the 13-player field, E4-38-03@60M rates +82 over E4-27-03@60M and E4-38-05@60M rates −115
under E4-27-05@60M. Seat balancing (E4-36-05) is the strongest bench run in both fields.

### Why: the brake became an entropy push

Mean policy entropy by 10M bucket, 0–60M, and the fixed-probe entropy over 50–60M:

| run | entropy by 10M bucket | fixed probe |
|:---|:---|---:|
| E4-38-03 (WoLF) | 1.70 1.71 1.74 1.71 1.74 1.81 | 1.33 |
| E4-38-05 (WoLF) | 1.77 1.70 1.68 1.76 1.80 1.75 | **1.55** |
| E4-27-03 (control) | 1.74 1.70 1.69 1.64 1.53 1.55 | 1.01 |
| E4-27-05 (control) | 1.80 1.77 1.78 1.75 1.66 1.60 | 1.24 |
| E4-36-03 (seat balancing) | 1.65 1.59 1.48 1.32 1.23 1.08 | 0.84 |
| E4-36-05 (seat balancing) | 1.74 1.72 1.70 1.48 1.39 1.15 | 0.55 |
| E4-08-03 (λ 0.98) | 1.56 1.41 1.37 1.51 1.50 1.39 | 1.12 |
| E4-08-05 (λ 0.98) | 1.65 1.43 1.45 1.39 1.36 1.40 | 0.80 |

**The WoLF runs never sharpen.** Their entropy is flat at ~1.75 for all 60M, where every other run's
falls.

The cause is the design choice to weight the surrogate alone. On the down-weighted seat, the
unweighted entropy bonus gains on the policy gradient, so that seat is not merely slowed but pushed
toward uniform. When the weights oscillate, as on seed 5, both seats spend time in that role.
Their self-play stays balanced because both are held near random, and E4-38-05 is weak on both
seats: 2–5% as USSR and 7–11% as US against the λ 0.98 recipe. It is the E4-37-03 pattern again.
Balanced self-play is not health.

### Verdict

**Fails as built:** the collapse half passes on both seeds, and the strength half fails on seed 5.
WoLF proper is a per-agent **learning rate**. The fix that matches it is to scale each seat's whole
policy objective by its weight: surrogate, entropy bonus and KL to π_ref together. Then the
winning seat learns more slowly without its balance tipping toward entropy. Implementing that as
an option, and testing it on the same bench, needs the owner's go-ahead.

## Step 3c: WoLF as a per-seat learning rate (E4-39), 2026-09-23

E4-38's flags plus `--wolf-scope policy`: each seat's whole policy objective (surrogate, entropy
bonus and KL to π_ref) is scaled by its weight. `launch_flags.py --diff` against E4-38 on each seed
shows `--wolf-scope` only. Both runs ran 0 → 60M and exited cleanly.

### Collapse half: passes on both seeds

| run | 0–60M, by 5M (logged USSR share) | peak | buckets ≥ 0.9 | entropy by 10M |
|:---|:---|---:|:---|:---|
| E4-39-03 | .47 .42 .46 .63 .57 .63 .46 .56 .62 .74 .79 .58 | 0.79 | none | 1.62 1.70 1.84 1.73 1.70 1.72 |
| E4-39-05 | .44 .41 .57 .63 .41 .49 .67 .61 .53 .52 .62 .55 | 0.67 | none | 1.73 1.76 1.72 1.73 1.80 1.77 |

Seed 5 swung less than under the surrogate-only scope, 0.41–0.67 against E4-38-05's 0.35–0.71.
**Entropy still does not fall.** It stays at ~1.7–1.8 on both seeds, as it did in E4-38. So
scaling the entropy bonus with the surrogate did not bring back the sharpening that the controls
and the λ 0.98 recipe show. The explanation given for E4-38, that the unweighted entropy bonus
pushed the braked seat toward uniform, is therefore at best incomplete.

### Strength half: no better than E4-38; seed 5 still weak on both seats

`data/reports/p25_wolf_policy_50_60M.{md,json}`: E4-39, E4-38 and E4-27 on both seeds at 50, 55
and 60M, plus E4-08-0s@60M. 21 players, 100 games per side per pair, temperature 0.

| seed | step | E4-39 − E4-27 | E4-39 − E4-38 | E4-39 vs E4-27 (as USSR / as US) | E4-39 vs λ 0.98 @60M (as USSR / as US) |
|---:|:---|---:|---:|---:|---:|
| 3 | 50M | +27 | −47 | 83 / 37 | 18 / 15 |
| 3 | 55M | +91 | +28 | 88 / 24 | 18 / 20 |
| 3 | 60M | +49 | −73 | 93 / 24 | 9 / 8 |
| 5 | 50M | −79 | −24 | 68 / 15 | 5 / 4 |
| 5 | 55M | −75 | +28 | 70 / 7 | 5 / 8 |
| 5 | 60M | **−114** | +8 | 53 / 8 | 7 / 7 |

*This round was run twice. The second run, after the machine restart, overwrote the report file
the first had written. The Elo differences above are from the file on disk, the second run. The
first run's numbers differed by at most 5 Elo per row (seed 3: +27/+94/+54, seed 5:
−78/−74/−110), and the win-rate cells are identical.*

**Verdict: fails, like E4-38.** Scaling the whole policy objective changed neither result:
* seed 3 is modestly ahead of its collapsed control, +27 to +91;
* seed 5 is weak on both seats, −75 to −114, winning 4–8% against the λ 0.98 recipe.

The entropy explanation for E4-38 is therefore **withdrawn**. It was not the unweighted entropy
bonus. Both WoLF variants hold the self-play split near even, and both keep entropy at ~1.75. On
seed 5 both are weak. What the two variants share is that they slow whichever seat is ahead. One
reading, untested, is that in a game whose equilibrium favours one side, slowing the winner also
slows the policy's approach to it, which would make "balanced self-play" the wrong target.

## Step 3d, part 1: WoLF at power 0.5 (E4-40), 2026-09-24

E4-39's flags (λ 0.99 from scratch, `--wolf-seat-weight --wolf-scope policy`) with `--wolf-power
0.5`: the seat ratio is (x/(1−x))^0.5, so a 90/10 split weights 3:1 instead of 9:1.
`launch_flags.py --diff` against E4-39 on each seed shows `--wolf-power` only.

### Collapse half: passes on both seeds, but swings wider

| run | 0–60M, by 5M (logged USSR share) | peak | lowest after 15M | buckets ≥ 0.9 |
|:---|:---|---:|---:|:---|
| E4-40-03 | .52 .55 .42 .29 .37 .53 .74 .80 .78 .60 .59 .51 | 0.80 | 0.29 | none |
| E4-40-05 | .49 .52 .51 .59 .35 .41 .67 .54 .62 .82 .75 .51 | 0.82 | 0.35 | none |

The softer brake lets the lead run further before pulling it back: E4-40 spans 0.29–0.82, where
E4-39 spanned 0.41–0.79.

### Strength half: the seeds swap

`data/reports/p25_e440_50_60M.{md,json}`: E4-40, E4-39 and E4-27 at 50, 55 and 60M on both seeds,
plus E4-08-0s@60M. 20 players, 100 games per side per pair, temperature 0.

| seed | step | E4-40 − E4-27 | E4-40 − E4-39 | E4-40 vs E4-27 (as USSR / as US) | E4-40 vs λ 0.98 @60M (as USSR / as US) |
|---:|:---|---:|---:|---:|---:|
| 3 | 50M | −169 | −194 | 41 / 8 | 8 / 6 |
| 3 | 55M | −123 | −206 | 77 / 9 | 3 / 3 |
| 3 | 60M | −148 | −192 | 65 / 4 | 4 / 4 |
| 5 | 50M | +160 | +243 | 84 / 52 | 13 / 15 |
| 5 | 55M | +109 | +195 | 86 / 43 | 16 / 25 |
| 5 | 60M | +36 | +149 | 75 / 26 | 6 / 26 |

At power 1 (E4-38, E4-39), seed 3 gained (+27 to +131) and seed 5 lost (−45 to −114). At power 0.5
it is the other way round. **Which seed comes out ahead under WoLF looks like chance, not a property
of the seed.** That is the seed lottery P25 exists to remove, not a cure for it.

## An in-run strength signal: fixed-probe entropy

The strong run of each WoLF pair is the one whose **fixed-probe entropy** fell:
* E4-39-03 1.01 against E4-39-05 1.79;
* E4-40-05 1.07 against E4-40-03 1.75.

Fixed-probe entropy is the policy's entropy on 2,000 positions frozen at the start of the run.
Across all 11 bench runs, strength against the control at 60M (from each run's own rating round)
against fixed-probe entropy over 50–60M, measured relative to the same seed's control:

| run | Δ Elo vs control | probe entropy | probe − control | on-policy entropy |
|:---|---:|---:|---:|---:|
| E4-36-05 | +151 | 0.55 | −0.69 | 1.15 |
| E4-38-03 | +131 | 1.33 | +0.32 | 1.81 |
| E4-39-03 | +49 | 1.01 | +0.00 | 1.72 |
| E4-40-05 | +36 | 1.07 | −0.16 | 1.69 |
| E4-35-03 | +24 | 0.99 | −0.02 | 1.38 |
| E4-36-03 | +10 | 0.84 | −0.17 | 1.08 |
| E4-37-03 | +9 | 1.39 | +0.38 | 1.79 |
| E4-37-05 | +8 | 1.31 | +0.08 | 1.68 |
| E4-38-05 | −105 | 1.55 | +0.31 | 1.75 |
| E4-39-05 | −114 | 1.79 | +0.55 | 1.77 |
| E4-40-03 | −148 | 1.75 | +0.74 | 1.84 |

**Correlation −0.74** between Δ Elo and probe entropy above the control, against −0.43 for on-policy
entropy. The one clear exception is E4-38-03 (+131 at +0.32). The on-policy figure is weaker
because it mixes in which positions the run happens to reach. The fixed probe holds the positions
constant.

This matters for P25 beyond WoLF. **A run that has stopped sharpening on the fixed probe is a weak
run**, and that can be read during training with no tournament. It is a candidate trigger for step
5's auto-rewind, next to the self-play split, which catches collapses but not this failure.

## Step 3d, part 2: WoLF with a dead zone (E4-41), 2026-09-24

E4-39's flags with `--wolf-dead-zone 0.15`. The weights stay 1 while the USSR share is within
0.35–0.65. Outside that band, the share is shifted toward 0.5 by 0.15 before the plain rule, so a
0.90 split weights like 0.75 did (3:1). `launch_flags.py --diff` against E4-39 on each seed shows
`--wolf-dead-zone` only.

### Collapse half: seed 3 passes, seed 5 fails at the end

| run | 0–60M, by 5M (logged USSR share) | peak | buckets ≥ 0.9 | entropy by 10M | probe 50–60M |
|:---|:---|---:|:---|:---|---:|
| E4-41-03 | .52 .61 .74 .75 .61 .71 .84 .81 .67 .57 .75 .66 | 0.84 | none | 1.72 1.68 1.62 1.62 1.58 1.57 | 0.79 |
| E4-41-05 | .51 .57 .74 .76 .68 .57 .55 .58 .67 .73 .90 .96 | 0.96 | 50M, 55M | 1.74 1.56 1.50 1.52 1.57 1.55 | 0.89 |

**This is the first WoLF variant whose policy sharpens.** Fixed-probe entropy is 0.79 and 0.89,
below both controls (1.01, 1.24) and far below E4-38/39/40 (1.01–1.79).

### Strength half: ahead of the control on both seeds

`data/reports/p25_e441_50_60M.{md,json}`: E4-41, E4-39 and E4-27 at 50, 55 and 60M, E4-40 at 60M,
and E4-08-0s@60M. 100 games per side per pair, temperature 0.

| seed | step | E4-41 − E4-27 | E4-41 − E4-39 | E4-41 vs E4-27 (as USSR / as US) | E4-41 vs λ 0.98 @60M (as USSR / as US) |
|---:|:---|---:|---:|---:|---:|
| 3 | 50M | +145 | +127 | 93 / 53 | 31 / 29 |
| 3 | 55M | +81 | +19 | 92 / 37 | 31 / 13 |
| 3 | 60M | +147 | +124 | 92 / 32 | 26 / 19 |
| 5 | 50M | +168 | +258 | 94 / 48 | 29 / 22 |
| 5 | 55M | +161 | +246 | 92 / 39 | 45 / 15 |
| 5 | 60M | +46 | +171 | 90 / 14 | 39 / 11 |

**The best strength on the bench, and the closest any run has come to the λ 0.98 recipe.** The
drop on seed 5 at 60M (+46, and US 14% against the control) is the collapse in its last 10M.

### Reading

The dead zone leaves the dynamics alone near an even split, which is where plain WoLF kept
trading the lead back and forth and never sharpened. That part works. Outside the zone the brake
is too soft and comes too late. With the shift, a 0.90 split gets 3:1 instead of plain WoLF's 9:1,
and on seed 5 the drift from 0.67 to 0.96 took 15M steps against that. The next variant keeps the
dead zone but applies the full rule once outside it, a jump instead of a shift. A narrower zone is
the other option.

## Step 3e: WoLF on the normal recipe (E4-42), 2026-09-24

E4-08's flags (M2d, λ 0.98, from scratch) plus `--wolf-seat-weight --wolf-scope policy` at power 1,
seeds 3 and 5, to 80M. `launch_flags.py --diff` against E4-08-0s shows the WoLF flags only.

**The legs.** Both runs stalled together at ~76–77.6M, a two-process hang that did not reproduce
(see [`training_throughput_cpu.md`](training_throughput_cpu.md), "Open issue"). Each was then
continued 75M → 80M from its own 75M resume state, in `E4-42-0s_20260924_02*`. On resume the pool
dropped its initial-policy seed member, which has no snapshot on disk; the other 11 were restored.

**Self-play.** The split swung more than the control's did. E4-42-03 went 0.72 → 0.81 → 0.33 → 0.29
→ 0.66 over 10–60M, where E4-08-03 moved 0.83 → 0.89 → 0.44 on its own. It settled by 80M:
0.60 and 0.61 USSR, with entropy 1.20 and 1.26 on level seats.

**Strength:** `data/reports/p25_e442_60_80M.{md,json}`, E4-42 against E4-08 at 60, 70 and 80M on
both seeds, plus the anchor. 100 games per side per pair.

| seed | step | Δ Elo vs E4-08 | E4-42 vs E4-08 at the same step (as USSR / as US) |
|---:|:---|---:|---:|
| 3 | 60M | +34 | 64 / 47 |
| 3 | 70M | −36 | 47 / 44 |
| 3 | 80M | −5 | 40 / 48 |
| 5 | 60M | −105 | 26 / 43 |
| 5 | 70M | −10 | 60 / 40 |
| 5 | 80M | +33 | 63 / 53 |

**WoLF is strength-neutral on the normal recipe by 80M:** −5 and +33, and level per seat. There was
no early collapse to prevent on these seeds, so this measures only its cost, and at 80M there is
none. The −105 on seed 5 at 60M closed by 70M.

## Step 3f: WoLF dead zone with a jump (E4-43), 2026-09-24

E4-41's flags with `--wolf-dead-zone-mode jump`: inside 0.35–0.65 the weights stay 1, and outside
it the plain rule applies to the unshifted share (9:1 at 0.90). `launch_flags.py --diff` against
E4-41 on each seed shows `--wolf-dead-zone-mode` only. E4-43-03 and E4-41-03 matched to every printed
digit at 10M: the same seed, and the same arithmetic until the share first leaves the zone.

### Collapse half: passes on both seeds, tightly

| run | 0–60M, by 5M | peak | buckets ≥ 0.9 | entropy by 10M | probe 50–60M |
|:---|:---|---:|:---|:---|---:|
| E4-43-03 | .52 .61 .55 .53 .69 .54 .55 .62 .71 .66 .61 .41 | 0.71 | none | 1.72 1.67 1.69 1.77 1.83 1.82 | 1.38 |
| E4-43-05 | .51 .58 .42 .58 .73 .72 .60 .58 .60 .60 .67 .68 | 0.73 | none | 1.75 1.63 1.76 1.77 1.80 1.79 | 1.64 |

### Strength half: fails, as the fixed-probe entropy predicted

`data/reports/p25_e443_50_60M.{md,json}`: E4-43, E4-41 and E4-27 at 50, 55 and 60M, plus
E4-08-0s@60M. 100 games per side per pair, temperature 0.

| seed | step | E4-43 − E4-27 | E4-43 − E4-41 | E4-43 vs E4-27 (as USSR / as US) | E4-43 vs λ 0.98 @60M (as USSR / as US) |
|---:|:---|---:|---:|---:|---:|
| 3 | 50M | −140 | −274 | 58 / 4 | 6 / 4 |
| 3 | 55M | −107 | −189 | 75 / 4 | 3 / 6 |
| 3 | 60M | −19 | −171 | 84 / 11 | 5 / 13 |
| 5 | 50M | −50 | −217 | 72 / 15 | 8 / 8 |
| 5 | 55M | −107 | −257 | 46 / 11 | 7 / 5 |
| 5 | 60M | −95 | −136 | 64 / 5 | 7 / 6 |

**The brake/sharpen trade-off is the finding of 3d–3f.** The same rule, shifted or jumped at the
edge of one zone, gives:
* **Shift (E4-41):** a gentle brake. The policy sharpens (probe 0.79/0.89) and the runs are strong,
  +46 to +168, but seed 5 collapses at 50–60M.
* **Jump (E4-43):** the full brake outside the zone. It holds the split (peaks 0.71/0.73), but the
  policy stays near random (probe 1.38/1.64) and the runs are weaker than the collapsed control.

Whenever the WoLF weights are strongly engaged, the leading seat stops learning, and on this
recipe the leading seat is the one that was learning. The next two arms test whether the backstop
can come from somewhere else: seat balancing (E4-46), or a slightly earlier but still gentle brake
(E4-47).

## Step 3g, part 1: WoLF dead zone (shift) + seat balancing (E4-46), 2026-09-24

E4-41's flags plus `--seat-balance`. `launch_flags.py --diff` against E4-41 shows `--seat-balance`
only. Collapse is judged on the pool's pure self-play share, as for E4-36.

### Collapse half: passes on both seeds, and the policy sharpens most of any run

| run | 0–60M, by 5M (pure self-play USSR share) | peak | buckets ≥ 0.9 | entropy by 10M | probe 50–60M |
|:---|:---|---:|:---|:---|---:|
| E4-46-03 | .52 .56 .64 .67 .71 .81 .75 .77 .78 .73 .77 .78 | 0.81 | none | 1.75 1.57 1.36 1.21 1.15 1.10 | 0.67 |
| E4-46-05 | .51 .56 .75 .80 .79 .74 .69 .68 .73 .76 .79 .86 | 0.86 | none | 1.77 1.46 1.20 1.28 1.28 1.16 | 1.04 |

### Strength half: level with the control, and well below E4-41

`data/reports/p25_e4-46_50_60M.{md,json}`: E4-46, E4-41 and E4-27 at 50, 55 and 60M, E4-43 at
60M, and E4-08-0s@60M. 100 games per side per pair.

| seed | step | E4-46 − E4-27 | E4-46 − E4-41 | E4-46 vs E4-27 (as USSR / as US) | E4-46 vs λ 0.98 @60M (as USSR / as US) |
|---:|:---|---:|---:|---:|---:|
| 3 | 50M | −24 | −148 | 82 / 19 | 11 / 8 |
| 3 | 55M | −47 | −133 | 83 / 14 | 9 / 7 |
| 3 | 60M | −43 | −196 | 82 / 16 | 14 / 5 |
| 5 | 50M | −25 | −177 | 76 / 11 | 9 / 11 |
| 5 | 55M | +4 | −161 | 87 / 24 | 13 / 7 |
| 5 | 60M | −20 | −61 | 86 / 9 | 14 / 8 |

**Fails on strength.** E4-46 prevents the collapse, but it gives back all of E4-41's gain.

### Why, and a correction to the fixed-probe signal

The sharpening is **one-sided**. At 60M the USSR seat's entropy is 0.85–0.95 and the US seat's
1.62–1.73. The head-to-head shows the same split: 82–87% as USSR, 9–24% as US. At full pressure
seat balancing puts the learner on the weak seat, the US, in ~90% of pool games, against past
snapshots. The US then learns to beat old selves, while the USSR, playing mostly self-play and
braked only gently by the shifted dead zone, runs ahead. The pure self-play split stays under 0.9
because the old-snapshot games keep the US from vanishing, not because the US is good.

**Fixed-probe entropy is not a strength oracle when the seats diverge.** E4-46's probe entropy
(0.67 and 1.04) is the lowest on the bench, yet it is no stronger than its control. The probe
averages positions from both seats, and a sharp USSR pulls it down. The −0.74 correlation (above)
held across runs whose seats were roughly symmetric. Used as a run-health signal it needs to be
split by seat.

## Step 3g, part 2: a narrower dead zone (E4-47), 2026-09-24

E4-41's flags with `--wolf-dead-zone 0.10` (shift). The brake starts at 0.60 instead of 0.65, and
at a 0.90 split it weighs 4:1 instead of 3:1. `launch_flags.py --diff` against E4-41 shows the
dead zone only.

### Collapse half: passes on both seeds, seed 3 near the line at the end

| run | 0–60M, by 5M (self-play USSR share) | peak | buckets ≥ 0.9 | entropy by 10M | last 10M: US / USSR entropy | probe 50–60M |
|:---|:---|---:|:---|:---|:---|---:|
| E4-47-03 | .52 .53 .48 .56 .69 .59 .61 .77 .64 .72 .79 .86 | 0.86 | none | 1.74 1.69 1.69 1.72 1.66 1.71 | 1.89 / 1.60 | 1.08 |
| E4-47-05 | .51 .66 .67 .67 .71 .65 .46 .60 .49 .53 .51 .39 | 0.71 | none | 1.75 1.68 1.67 1.73 1.70 1.73 | 1.68 / 1.76 | 1.46 |
| E4-41-03 (dz 0.15) | .52 .61 .74 .75 .61 .71 .84 .81 .67 .57 .75 .66 | 0.84 | none | 1.72 1.68 1.62 1.62 1.58 1.57 | 1.73 / 1.50 | 0.79 |
| E4-41-05 (dz 0.15) | .51 .57 .74 .76 .68 .57 .55 .58 .67 .73 .90 .96 | 0.96 | 50–60M | 1.74 1.56 1.50 1.52 1.57 1.55 | 1.82 / 1.39 | 0.89 |

Seed 3 ends at 0.86 and rising, with the US seat's entropy climbing (1.89) and the advantage
spread falling (0.247 at 60M). That is how E4-41-05's collapse began at 45–50M. This is a pass at
60M, but not by much.

### Strength half: between E4-43 and E4-41, and below E4-41 on both seeds

`data/reports/p25_e4-47_50_60M.{md,json}`: the E4-46 field with E4-47 in place of E4-46.

| seed | step | E4-47 − E4-27 | E4-47 − E4-41 | E4-47 vs E4-27 @60M (as USSR / as US) |
|---:|:---|---:|---:|---:|
| 3 | 50M | +71 | −52 | 96 / 31 |
| 3 | 55M | +50 | −20 | 89 / 25 |
| 3 | 60M | +91 | −54 | 93 / 37 |
| 5 | 50M | +23 | −155 | 65 / 17 |
| 5 | 55M | +21 | −146 | 71 / 30 |
| 5 | 60M | −57 | −121 | 67 / 21 |

At 60M, E4-47 rates +137 (seed 3) and +44 (seed 5) above E4-43 (jump).

**Fails on strength against E4-41, passes narrowly against the control.** The dead zone behaves as a
single dial. A wider zone (0.15) sharpens more and is stronger, but can collapse. A narrower one
(0.10) holds the split better and is weaker. The jump (E4-43) is the far end. Seed 5 never
sharpened (entropy 1.73 at 60M, probe 1.46), and it is the weaker seed. The fixed-probe entropy
ranks the two E4-47 seeds correctly, and both sit above E4-41's.

No setting of this dial gets both halves on both seeds. What the bench is asking for is a brake
that acts on the *losing* seat's signal without flattening the *winning* seat's. Every WoLF variant
tried here scales the two seats' objectives against each other. Seat balancing does act on the
losing seat alone, but E4-46 showed it buys that with one-sided sharpening.

## Step 3i: new seeds for the control and E4-41 (seeds 6 and 7), 2026-09-24

E4-27's flags (the control) and E4-41's flags (WoLF dead zone 0.15, shift) on seeds 6 and 7, to
60M. `launch_flags.py --diff` against the seed-3/5 runs shows the seed only. Seed 7's two runs are
identical through 10M (the same 5M buckets, advantage spread and entropy), as they must be: inside
the dead zone both weights are 1.

### Collapse half: E4-41 collapses on 3 of 4 seeds, the control on 2 of 4

| run | 0–60M, by 5M (self-play USSR share) | peak | buckets ≥ 0.9 | entropy by 10M | last 10M: US / USSR entropy | probe 50–60M |
|:---|:---|---:|:---|:---|:---|---:|
| E4-27-06 | .53 .56 .61 .77 .67 .80 .74 .69 .68 .80 .79 .51 | 0.80 | none | 1.75 1.62 1.63 1.67 1.66 1.68 | 1.79 / 1.61 | 1.23 |
| E4-41-06 | .53 .62 .59 .56 .74 .79 .84 .92 .84 .87 .88 .84 | 0.92 | 35–40M | 1.75 1.63 1.70 1.76 1.63 1.64 | 1.84 / 1.57 | 1.18 |
| E4-27-07 | .54 .51 .55 .70 .62 .58 .53 .68 .41 .47 .53 .63 | 0.70 | none | 1.70 1.59 1.66 1.71 1.61 1.74 | 1.78 / 1.70 | 1.18 |
| E4-41-07 | .54 .51 .55 .61 .61 .74 .78 .82 .75 .85 .77 .93 | 0.93 | 55–60M | 1.70 1.61 1.66 1.76 1.77 1.81 | 1.97 / 1.69 | 1.27 |

Across the four seeds:

| | seed 3 | seed 5 | seed 6 | seed 7 |
|:---|:---|:---|:---|:---|
| control (E4-27) | collapses | collapses | passes (0.80) | passes (0.70) |
| E4-41 | passes (0.84) | collapses (0.96) | collapses (0.92) | collapses (0.93) |

### Strength half: E4-41 behind its control on both new seeds

`data/reports/p25_seed{6,7}_50_60M.{md,json}`. Each field holds the seed's two runs at 50, 55 and
60M, plus E4-41-0{3,5}, E4-27-0{3,5} and E4-08-0{3,5} at 60M.

| seed | step | E4-41 − E4-27 | E4-41 vs E4-27 @60M (as USSR / as US) |
|---:|:---|---:|---:|
| 6 | 50M | −71 | 52 / 7 |
| 6 | 55M | −18 | 47 / 15 |
| 6 | 60M | +54 | 74 / 46 |
| 7 | 50M | −62 | 69 / 25 |
| 7 | 55M | −84 | 76 / 13 |
| 7 | 60M | −14 | 72 / 14 |

**E4-41 fails both halves on the new seeds.** Its seed-3/5 gain (+46..+168) was a property of those
two seeds. The bench's "collapse on 2 of 2 seeds" was also partly a property of seeds 3 and 5: on
seeds 6 and 7 the plain λ 0.99 control stays under 0.80.

In both new seeds the E4-41 run drifts further than its control once the brake engages (above
0.65). With w_us at 1.3–1.6, the US seat's entropy climbs (1.84, 1.97), and the split then widens
rather than closes. This matches the E4-38 failure ("the brake became an entropy push"): the
losing seat's whole objective is up-weighted, entropy bonus included. That mechanism was not
isolated here. If it is the cause, the dead zone only delays it.

**Where WoLF stands.** Seven variants (E4-38..47) have not produced one that passes both halves on
more than the seeds it was tuned on. Seed-level variance on this bench is as large as any lever's
effect, so a two-seed bench cannot rank levers. Any further lever needs at least four seeds.

## Step 3h: the late collapse was the pool (E4-48), 2026-09-24

Recorded separately in [`P25_pool_resume_bug.md`](P25_pool_resume_bug.md). A resume drained the opponent pool to its
recent end (fixed in `3803d5d`). Rerun on a fixed pool, all three seed-5 continuations from 160M
avoid the pin their drained twins hit. The two rated ones finish 170–390 Elo above their twins. The bench is from scratch
and never resumes, so nothing above is affected.

## Step 3j–3l, part 0: the controls, and why the bench stopped pinning (2026-09-24)

The 3j–3l bench (E4-49..53, λ 0.99 from scratch, seeds 10–13, 80M) opened with its four controls.
None pinned. Two more controls followed on seeds 3 and 5 (E4-49-03/05). These are the seeds whose
first λ 0.99 controls, E4-27-03/05, pinned for 6 and 4 five-million-step buckets. The question was
whether the code had changed the collapse rate.

| run | 0–80M, by 5M (self-play USSR share) | peak | buckets ≥ 0.95 |
|:---|:---|---:|---:|
| E4-49-03 | .52 .61 .77 .75 .63 .66 .82 .87 .88 .86 .94 .92 .85 .88 .85 .76 | 0.94 | 0 |
| E4-49-05 | .51 .61 .61 .66 .85 .69 .74 .64 .72 .63 .66 .34 .51 .70 .52 .52 | 0.85 | 0 |
| E4-49-10 | .64 .54 .66 .69 .72 .79 .55 .65 .50 .59 .54 .62 .75 .63 .70 .50 | 0.79 | 0 |
| E4-49-11 | .55 .40 .51 .40 .64 .55 .76 .61 .71 .77 .67 .65 .76 .88 .88 .88 | 0.88 | 0 |
| E4-49-12 | .61 .66 .68 .73 .74 .74 .72 .77 .78 .70 .46 .48 .65 .50 .55 .59 | 0.78 | 0 |
| E4-49-13 | .59 .52 .42 .51 .72 .67 .45 .58 .67 .47 .64 .65 .74 .63 .71 .70 | 0.74 | 0 |

Seed 3 went through a long episode (40–75M at 0.86–0.94) and recovered without pinning. The
original seed-3 control sat at 0.95–0.97 for 30M steps.

**No code change caused this.** The check (`data/logs/p25b/check_commit.sh`) runs E4-27-05's
flags for 10 iterations at a commit and compares every logged value with E4-27-05's own log:

| commit | against E4-27-05 |
|:---|:---|
| 318f01f (E4-27-05's own) | bit-identical, so training is deterministic on this machine |
| d6c89ad, bfbb789 (E4.1 engine), 9c49329 (per-seat signals), bfb7ace (dropped sync/refresh) | bit-identical |
| 4124562 (masked means as sum/count) | rounding: 3.7e-9 at iteration 1, 8e-5 by iteration 3 |
| 7766e3d (π_ref log-probs once per update) | rounding: 8.7e-10 at iteration 1 |
| a5a8e87, 7e260ab, 2a9b5a4 (graphs, serial replay, levers off) | bit-identical to 7766e3d |

Two commits change the arithmetic, both at 1e-9. The rollouts (USSR win rates) are identical
through iteration 3, so no commit changed the game or the update. Training is chaotic: rounding
of that size makes a different run by the third iteration.

**The pin rate is a property of the recipe, and it is low.** Across all plain λ 0.99 controls:
* pinned: 2 of 10 (E4-27-03, E4-27-05);
* not pinned: E4-27-06/07 (to 60M); E4-49-03, 05 and 10–13 (to 80M).

The old seeds pinned because of those two exact trajectories. Neither the seed number nor the
code decides it. At about 20% per run, four seeds per arm expect about 0.8 pins in the control,
so this bench cannot show a lever *preventing* a pin. It can measure a lever's strength cost or
gain, and the depth and length of the episodes (such as E4-49-03's). Testing prevention needs
about 10–15 seeds per arm, or a bench that pins more often.

**The rounding demonstration.** E4-27-05's flags (seed 5, λ 0.99) at 4124562, a commit that
differs from E4-27-05's own only by the 1e-9 rounding above. Its output is in
`data/bisect/s5_4124562_80M`, kept out of the checkpoint tree.

| run | 0–80M, by 5M (self-play USSR share) | buckets ≥ 0.95 |
|:---|:---|---:|
| E4-27-05 (318f01f) | .52 .36 .41 .50 .70 .75 .84 .82 .90 .89 .87 .92 .97 .98 .98 .98 | 4 |
| rounding demo (4124562) | .42 .56 .62 .71 .75 .68 .55 .56 .47 .51 .53 .54 .60 .80 .84 .90 | 0 |

A last-digit change to one mean moved the collapse from 40M to about 65M. Within 80M it also
turned a pin into an episode that is still at 0.90 at the end. Both runs head for the same
place, so the tendency to collapse belongs to the recipe (λ 0.99). Whether a particular run pins
inside a fixed budget, and when, is a draw. "Seed 5 collapses" was never a property of seed 5.

## Closing P25 (2026-09-24)

The owner closed P25 on 2026-09-24. The owner's criterion was that a collapse matters only if
training dies or is delayed for a long time. The evidence gathered against that criterion:

* **On the production recipe (E4-08: λ 0.98, pool 0.3/12) a collapse is a delay.**
  * About 1 in 6 seeds from scratch entered a pin within 80M.
  * The long episodes each cost about 50–70M steps.
  * By 160M the collapsed seeds had caught up: −1.4 Elo against the seeds that never collapsed
    ([`E4_collapse_is_recoverable.md`](E4_collapse_is_recoverable.md)).
* **Every permanent stall came after a resume on the drained pool.** The fix is `3803d5d`. On
  the fixed pool, 3 of 3 continuations did not pin
  ([`P25_pool_resume_bug.md`](P25_pool_resume_bug.md)).
* **The clean replicates on the fixed code did not pin.** E4-08-36 and E4-08-37 ran straight to
  240M with no pin ([`E4_clean_replicates.md`](E4_clean_replicates.md)).
  * E4-08-37 did lean USSR for its whole run, and its two ~40M flat stretches line up with
    episodes that peaked at 0.91–0.92.
  * So a long delay without a pin is possible. It has been seen once.
* **The λ 0.99 bench pins in about 1 run in 5, and that is a draw, not a property of the seed or
  the code** (part 0 above). A 4-seed arm expects under one pin in its control, so the 3j–3l bench
  could not have shown a lever preventing one.

**The 3j–3l bench was stopped part-way, unrated.**

| run | lever | reached | 0–80M, by 5M (self-play USSR share) | buckets ≥ 0.95 |
|:---|:---|---:|:---|---:|
| E4-50-10 | floor 0.8 | 80M | .61 .43 .35 .28 .65 .61 .56 .48 .53 .67 .87 .71 .78 .77 .77 .86 | 0 |
| E4-50-11 | floor 0.8 | 80M | .49 .59 .72 .60 .69 .58 .58 .56 .67 .67 .90 .75 .92 .97 .97 .97 | 3 |
| E4-50-12 | floor 0.8 | 80M | .57 .53 .79 .87 .72 .84 .68 .70 .79 .83 .77 .70 .70 .73 .63 .62 | 0 |
| E4-50-13 | floor 0.8 | 45M | .48 .51 .67 .75 .75 .75 .77 .66 .72 | 0 |
| E4-51-10 | floor 0.5 | 40M | .61 .43 .34 .28 .35 .42 .58 .58 | 0 |

The 0.8 floor pinned once (E4-50-11, 65–80M), where none of the six λ 0.99 controls on the
current code did. At the bench's pin rate that is not evidence against the floor, and it is
certainly not evidence for it. The entropy ceiling (E4-52) and the per-seat KL stop (E4-53) never
ran. All three levers stay in the code, off by default (`--adv-norm-floor`, `--entropy-ceiling`,
`--target-kl`).

**What P25 leaves behind:**
* the pool resume fix (`3803d5d`);
* the per-seat signals (`9c49329`);
* `launch_flags.py`, which now diffs every recorded flag;
* the finding that training is chaotic at 1e-9;
* the CUDA-graph fix (`9bfceb1`), found while chasing the bench's crashes.

**What moves on:**
* Step 5's codified rewind goes to [`reserve`](../plans/reserve.md) as an optional safety net.
* The slow π_ref switch (step 4) goes to the strength queue. It won late on seed 3, but only on
  the drained pool.
* **The collapse measure should be progress, not a pin.** Use per-seat Elo against the run's own
  earlier snapshots, since E4-08-37 stalled without a pin.
