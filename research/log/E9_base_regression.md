# The E9 base's regression after ~950M (2026-11-22..23)

**Conclusion (2026-11-23): no defect -- the base's dip is the recipe's ordinary dynamics, pool or not.** Runs of
this recipe family make real, temporary losses of strength -- 4-8 points against a fixed opponent, lasting ~300M
steps -- with and without an opponent pool (section 6: R119, pooled, and B6, pure self-play, the same recipe and seed,
each dip in 5 of 23 points against their own past, both around 1.3-1.5B). The E9 base's 950 -> 1,200M stretch is such a
dip; comparing it with R169 -- which happened not to dip there -- made it look like a regression. The E9 engine, the
2026-11-22 fixes, the resume, torch.compile, the GPU and the toolchain are each ruled out (sections 2-4). Section 5's
first reading (pooled runs never dip) was wrong: it measured every pooled run at 1,200M only. Why a run this far from
its plateau loses ground at all is open.

All strength numbers are greedy (temperature 0) head-to-head scores of 80M SWAs (uniform weight averages of the
window's 10M snapshots, `tools/scripts/average_weights.py`), 2,000 deals a side (+-0.8) unless marked 1,000
(+-1.1), on the current E9 engine build. "Own 950" = the run's own 870-950M SWA (for continuations resumed at 890M:
the original R169's, their shared past).

## 1. The symptom

| E9 base vs R169 (pre-E9, same recipe and seed) at matched windows | score |
|:---|---:|
| 320-400M | 48.9% |
| 720-800M | 50.1% |
| 870-950M | 50.7% |
| 1,120-1,200M | **41.7%** |

The base's 1,120-1,200M SWA scores 43.7% against its own 870-950M SWA. Every logged training statistic (entropy,
explained variance, value losses, KL to pi_ref, clip fraction, seat balance, endings) is level with R169's throughout.

## 2. What it is not

Each row is a continuation trained to 1,210M; the score is its 1,120-1,200M SWA vs its own 950.

| continuation | from | code / engine | seed | vs own 950 |
|:---|:---|:---|:---|---:|
| original R169, uninterrupted | -- | 2c333b4, E8.1 | 45 | **51.5%** |
| base, resumed at 950M | base | merged fixes, E9 | 45 | 43.7% |
| rerun old code | base@920M | fdbd2c3, E9 | 45 | (vs R169 1,200: 42.5%) |
| rerun new code | base@920M | merged, E9 | 45 | (vs R169 1,200: 45.2%) |
| Europe Control 40 VP | base@550M | ca915af, E9 (40 VP) | 45 | 47.6% |
| Europe Control 100 VP | base@550M | fdbd2c3, E9 | 46 | 40.4% |
| X | R169@550M | current, E9 | 45 | 45.5% |
| Y | base@550M | 0cd1d6a, E8.1 (pre-audit) | 45 | 50.5% |
| P | R169@890M | 6b3a859, E8.1 | 45 | 42.8% |
| Q | R169@890M | 0d84934, E9 | 45 | 44.3% |
| R | R169@890M | 0cd1d6a, E8.1 | 45 | 43.4% |
| S | R169@890M | 2c333b4 (R169's own), E8.1 | 45 | 46.1% |
| E: eager update | R169@890M | 2c333b4 | 45 | 47.0% |
| F: compiled, fresh private cache | R169@890M | 2c333b4 | 45 | 44.6% |
| seed 46 | R169@890M | 2c333b4 | 46 | 42.5% |

Healthy reference stretches, uninterrupted runs: B6 1,120-1,200 vs 870-950 **53.4%**; R169 1,520-1,600 vs
1,270-1,350 **55.5%**; R169 1,920-2,000 vs 1,670-1,750 **54.9%** (1,000 deals a side).

Ruled out by these rows: the E9 engine fixes, the 2026-11-22 audit's training changes (R169's own code degrades
too), the Europe Control encoding (40 VP degrades; and the slot reads +/-5 in only 0.57% of decisions), the seed
(46 degrades), torch.compile and its shared cache (eager degrades; a fresh cache degrades).

Also checked directly:
* **The resume.** A state audit (CPU and GPU, R169's recipe at small sizes): model, pi_ref, optimizer state and
  membership, hooks, modes, every trainer / buffer / runner attribute and RNG are restored bit for bit; one update
  on the same buffer is bitwise equal; an uninterrupted run given the same env restart a resume does stays bitwise
  identical through two pi_ref refreshes. Only the 512 in-flight games restart (from the run's step-0 deals), a
  transient of ~6 iterations. (`tmp/resume_audit/` in the job directory.)
* **The GPU.** fp32 GPU vs CPU: logits within 1.5e-5, gradients within 1.7e-6 relative; 300 repeated forwards
  bit-identical; compiled vs eager forward within TF32 tolerance.
* **The optimizer settings and reward.** R169's and the base's saved optimizer groups are identical; the retired
  reward flags were off by default; the removed blunder window was already off (`--no-blunder-window`).
* **The KL to pi_ref.** Identical sawtooth (refresh every 200,000 steps, 4 iterations) and mean (0.00342 vs
  0.00343 over the same 4,730 iterations) in the original and a continuation.
* **Not non-transitivity.** Against a fixed third opponent (B6's soup, greedy, 1,000 deals a side): R169 870-950
  25.3%, R169 1,120-1,200 25.6%, S 1,120-1,200 **21.6%**, base 870-950 22.9%, base 1,120-1,200 **19.0%** -- the
  continuations are weaker, not just differently matched.
* **Matched-window noise is large.** S vs the original at matched 40M windows: 50.1% (910-950), 53.4% (990-1,030),
  39.5% (1,070-1,110), 48.1% (1,160-1,200), 1,000 deals a side -- single-window head-to-head reads swing far beyond
  their sampling error.

## 3. A fresh run degrades too

R169 from scratch on its own code (2c333b4, seed 45), one process, no resume, trained 2026-11-23: level with the
original at 360-440M (51.4%, 1,000 deals a side), then 1,120-1,200M vs its own 870-950M **43.6%** and vs the
original's 1,120-1,200M 42.2%. So the resume is not the cause either.

**Every run of this recipe trained since the evening of 2026-11-22 degrades over 950 -> 1,200M; the only runs
that improve over that stretch (the original R169, 2026-11-21; B6, 2026-11-15) were trained before.** Code,
engine, seed, compile, starting state and resuming are each ruled out above. What changed in between is the
machine: the GPU's device-to-host latency fault overnight 2026-11-21/22 and the host reboot on 2026-11-22 15:16 UTC
(an RTX 4090, no ECC, SM clock 2,865 MHz under load against a 2,520 MHz reference boost).

## 4. Not the machine

* **GPU stress, 20 minutes hot** (~80 C, ~400 W): 982 rounds of 16 GB memory pattern write / triple re-read and
  repeated 8192^2 matmuls (fp32, TF32, bf16) compared bit for bit -- 0 mismatches.
* **Settings:** every recorded launch setting of the fresh run equals the original R169's.
* **Toolchain:** the same compiler (Ubuntu clang 21.1.8) in a build from 2026-11-19 and one from today.
* **Before the reboot too:** three pure-self-play runs trained before it, 1,120-1,200M vs own 870-950M (1,000 deals a
  side): R166 (both extra critics, compiled -- nearly R169) 47.7%, R165 47.4%, R167 44.3%.

## 5. Pool or no pool: a first, too narrow reading (superseded by section 6)

| run, 1,120-1,200M vs own 870-950M (1,000 deals a side) | opponent pool | score |
|:---|:---|---:|
| E8-A8-R93-S44 | 0.3 | 59.1% |
| E8-A8-R95-S44 | 0.3 | 56.8% |
| E8-A8-R102-S44 | 0.3 | 55.2% |
| E8-A8-R103-S44 | 0.3 | 53.9% |
| E8.1-A8-R119-S45 | 0.3 | 56.1% |
| E8.1-A8-R132-S45 (B6) | none | 53.4% |
| E8.1-A8-R169-S45 | none | 51.5% |
| E8.1-A8-R166-S45 | none | 47.7% |
| E8.1-A8-R165-S45 | none | 47.4% |
| E8.1-A8-R167-S45 | none | 44.3% |
| every continuation in section 2, and the fresh run | none | 40.4-50.5% |

Caveats: beating one's own past is what a pool trains for, so this metric flatters pooled runs -- but the
fixed-opponent check (section 2: S and the base lose ~4 points against B6's soup over the stretch, R169 holds) says
the pure-self-play loss is real; and the pooled runs are older recipes on other seeds. P47 found the pooled
recipe's bias at 11-14B steps, not here; the two findings are about different stages.

## 6. The dip census: R119 (pool 0.3) and B6 (no pool), 600M -> 12B

The 80M SWA ending at t against (a) the run's own 80M SWA ending at t - 250M and (b) a fixed independent opponent (the
E7 R34 soup, greedy); 400 deals a side (+-1.7). Same recipe and seed 45; B6 is R119 with `--opponent-frac 0`.

| t (M) | R119 vs own past | R119 vs fixed | B6 vs own past | B6 vs fixed |
|---:|---:|---:|---:|---:|
| 600 | 66.8 | 30.2 | 65.1 | 32.3 |
| 700 | 59.2 | 29.9 | 61.0 | 40.9 |
| 800 | 58.4 | 27.9 | 60.5 | 39.6 |
| 900 | 52.9 | 28.9 | 53.0 | 39.2 |
| 1000 | 54.9 | 26.9 | 56.4 | 42.6 |
| 1100 | 55.1 | 26.9 | 51.1 | 42.6 |
| 1200 | 55.9 | 33.6 | 53.4 | 45.6 |
| 1300 | 56.5 | 32.9 | 45.9 | 42.2 |
| 1400 | 46.7 | 29.1 | 42.6 | 37.7 |
| 1500 | 45.6 | 34.7 | 45.7 | 43.1 |
| 1600 | 52.3 | 33.1 | 60.8 | 42.0 |
| 1700 | 53.1 | 36.4 | 59.1 | 47.0 |
| 1800 | 55.1 | 37.8 | 55.4 | 49.1 |
| 1900 | 54.4 | 36.4 | 52.9 | 48.4 |
| 2000 | 50.4 | 38.1 | 54.1 | 48.3 |
| 2500 | 52.2 | 39.8 | 50.9 | 50.8 |
| 3000 | 53.4 | 45.4 | 47.6 | 56.2 |
| 4000 | 52.9 | 46.5 | 50.0 | 59.1 |
| 5000 | 49.7 | 53.2 | 51.6 | 60.9 |
| 6000 | 48.0 | 53.1 | 53.3 | 60.4 |
| 8000 | 51.0 | 58.1 | 51.1 | 63.6 |
| 10000 | 50.2 | 58.8 | 50.0 | 64.2 |
| 12000 | 48.8 | 61.6 | 46.7 | 65.1 |

* Both runs dip: below 50% against their own past in 5 of 23 points each; both at 1,300-1,500M. Against the fixed
  opponent the dips are real losses: B6 45.6 -> 37.7 (1,200 -> 1,400M), recovered by ~1,700M; R119 32.9 -> 29.1.
* Pure self-play is the stronger run against the fixed opponent at every point after 600M (P47's finding).
* No scheduled change explains the common timing: the learning rate is constant and the KL, clip and entropy settings
  fixed; the logged statistics drift smoothly through the dip (the Event share of plays rises a little in both around
  1.25-1.35B). The shared seed and recipe are the likelier reason the two runs dip together.

## 7. What the dipping network gets worse at

Five dips (the fresh rerun, the base and S over 950 -> 1,200M; R119 and B6 at 1,400 vs 1,150M) against five healthy
stretches (R169, R119 and B6 at 1,200 vs 950M; B6 and R119 at 1,700 vs 1,450M), later SWA A vs earlier SWA B of the
same run, 2,000 deals a side greedy (T = 1 where noted). Scripts, tables and the recorded games: the job's
`tmp/dip_why/` (`summary.md` first).

**Where the games are lost (net points of A's score):**

| pair | A as US, Europe decided* | A as USSR, final scoring | A's score |
|:---|---:|---:|---:|
| fresh | -3.8 | -5.4 | 43.4 |
| base | -3.7 | -5.5 | 42.7 |
| S | -1.9 | -4.8 | 45.5 |
| R119 1,400 | -3.0 | -3.5 | 45.4 |
| B6 1,400 | -2.9 | -5.8 | 41.6 |
| healthy (5 pairs) | -0.5 to -1.7 | -0.3 to -1.9 | 52.4 to 55.4 |

\* Europe Control, or the loser holding a Europe Scoring card it cannot play without losing. The deficit sits in
turns 7-11; turns 1-6 are about even; DEFCON, Wargames and early endings carry none.

**What changed in the policy (same positions, A minus B).** Every dip moves both seats from placing influence to
realigning and couping: the influence play mode loses 10-16 points of probability for the US and 8-12 for the USSR
(healthy: -7 to +2); per game, US influence plays -4.8 to -7.2 and realignments +2.8 to +5.7, USSR influence plays
-3.9 to -5.6 with coups +1.4 to +1.9, realignments +0.4 to +2.3, and more Space and Events. Influence leaves Europe
for Central America: in self-play, US placements in Europe fall from 14-15 to 10-11 a game and the USSR's from
12.5-13.4 to 8-10; US influence and realignments go to Cuba and Mexico. At final scoring against B, A as US gives up
1.4-4.2 VP in Europe (and some in Asia and the Middle East) for 1.5-2.4 in Central America; A as USSR loses Europe
the same way. Policy entropy is unchanged. Healthy stretches show none of it (R169: about half the shift, 52.4%).

**Mostly not forgetting.** A 2x2 seat matrix on the same deals (US from {B, A} x USSR from {B, A}): switching the US
from B to A loses 5-9 points against B's USSR and 5-9.5 against A's own USSR (greedy; one exception, R119 1,400 vs
B's USSR, -0.4); at T = 1, 0.9-5.5 points against A's own USSR. A is worse in absolute terms, as the fixed-opponent
drops say. One part is forgetting: B's USSR wins by Europe Control against A's US in 5.5-7.1% of games, a line in
only 0.9-1.6% of A's own self-play -- A's USSR stopped going for Europe as A's US stopped defending it.

**The critic** is unbiased with equal Brier on its own self-play, but on head-to-head positions A's is the more
over-optimistic about A (+0.10 to +0.13 against B's +0.03 to +0.11): it is calibrated to a self-play in which the new
style goes unpunished. At T = 1 the dips are smaller (S and R119 1,400 about 50% against ~45% greedy).

**Interpretation (not measured).** Both seats drift together into a style consistent with itself -- Central
American realignments, coups and quick Events, giving up the long-term presence in Europe (and Asia) that pays only
at final scoring. Since both sides neglect Europe alike, self-play sends no signal against it until the drift goes far
enough to be punished; the runs recover by ~1,700M. Candidate remedies: opponents that punish the drift (a pool
weighted toward snapshots ~250M back -- R119's uniform 0.3 pool still dipped); a longer pi_ref horizon (it refreshes
every 200,000 steps); and an early alarm on the influence share of play modes and Europe placements per seat, which
move before strength drops.
