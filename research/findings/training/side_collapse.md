# Side collapse: a reproducible, escapable learning-arrest

**2026-09-20.** Collapse was the dominant failure of the late E3 ladder and was only ever observed
after the fact, in arms too expensive to re-run. The P21 architecture ladder produced a
configuration that collapses at **80M steps in about 30 minutes on a 3.2M-parameter network**, so
for the first time it can be measured rather than described. This is the census from 30 distinct
seeds.

**The configuration:** P21 arm M2d — grouped positional input, per-entity head on **countries
only** (`pe_card` removed), cold start, pooled (`frac 0.3`, capacity 12), 80M steps. Arms differ
only in `--seed`.

## What it is

One seat stops winning. `us_episode_frac` — the share of self-play episodes the US wins — falls to
zero and stays there. Every observed instance has been USSR-dominant; **no US-side collapse has
ever occurred** in any arm of the ladder (`us_pinned` is 0.0% everywhere), though the detector
checks both tails.

It is **not a degradation**. The arrested arm does not get worse; it stops getting better. Rating
a collapsing seed's snapshots against a clean seed's at matched steps:

| steps | seed 1 (collapses) | seed 3 (clean) |
|---:|---:|---:|
| 20M | **1752.4** | 1727.2 |
| 40M | **1820.6** | 1811.4 |
| 60M | 1802.4 | **2004.9** |
| 80M | 1792.7 | **2167.8** |

Per 20M interval the collapsing seed gains +68.2, −18.2, −9.7 — flat after 40M — while the clean
seed gains +84.2, +193.5, +162.9. Before the onset the two are indistinguishable and the
collapsing one is fractionally *ahead*.

## Rate

| outcome | seeds | share |
|:---|---:|---:|
| **terminal collapse** | 4 — seeds 1, 12, 22, 26 | **13.3%** |
| **entered and escaped** | 1 — seed 21 | 3.3% |
| clean | 25 | 83.3% |

**4 of 30, Wilson 95% CI [5.3%, 29.7%].** Two seed-1 re-runs are excluded: they are replicates,
not seeds, and counting them would have reported 6 of 32.

## Entry and persistence are different things

**5 of 30 arms (17%) entered the pinned state** — 15+ consecutive logged rows with
`us_episode_frac < 0.02`. **Of those, 1 of 5 escaped.**

| seed | onset | longest pinned run | terminal | outcome |
|---:|---:|---:|---:|:---|
| 21 | **35M** | 45 rows | 0.343 | **escaped** |
| 12 | 39M | 143 rows | 0.010 | collapsed |
| 1 | 44M | 87 rows | 0.016 | collapsed |
| 22 | 49M | 91 rows | 0.016 | collapsed |
| 26 | **72M** | 70 rows | 0.010 | collapsed |

**Onset does not predict persistence.** The seed that escaped entered *earliest*, and onset spans
35M to 72M. Whatever decides the outcome happens after entry, not at it.

**The state is sticky, not absorbing.** Seed 21 held `us_episode_frac` at 0.0000 for 45
consecutive rows with `adv_std_raw` down to 0.0226 — squarely inside the collapsed range — and
then climbed back to finish at 0.343. An earlier version of this record called the collapse an
absorbing state and a clean bifurcation; **both are wrong**, and seed 21 is the counterexample.

## Strength cost

Rated in one 26-entrant field, 200 games per pair, τ=0.0:

| group | n | mean | sd | range |
|:---|---:|---:|---:|:---|
| clean | 21 | **1993.3** | 77.5 | 1829.7 – 2128.7 |
| recovered (seed 21) | 1 | 1969.8 | — | — |
| **collapsed** | 4 | **≈1697** | 12.4 | **1677.6 – 1716.1** |

References in the same field: anchor 2055.3, M2 with both heads 2016.4, M1 1718.2.

**Collapse is an attractor in strength too.** The four collapsed arms span 38 Elo where the clean
arms span 299 — they converge to essentially the same place. And all four rate **below M1**, the
rung beneath: a collapsed arm does not merely forfeit the mechanism's gain, it ends worse than not
having it.

The recovered seed lands inside the clean band, near its lower edge.

## It is bitwise reproducible

Three independent 80M runs of seed 1 — launched hours apart, with tournaments competing for the
same GPU — produced **byte-identical final checkpoints** (`sha256[:16] = fbfc08d937071163`) and
identical values on every logged metric across all 1,221 iterations. Independently, all three rate
**1700.8** in a 65,000-game tournament.

This holds despite `cudnn.benchmark`, `cudnn.deterministic` and `use_deterministic_algorithms` all
being **False** and `CUBLAS_WORKSPACE_CONFIG` unset. The explanation is mundane: those flags do not
*create* determinism, they *force* it where a non-deterministic kernel would otherwise be chosen.
This model has none — dense `Linear`, GELU and LayerNorm, no convolutions, no scatter-add, no
atomics, `allow_tf32=False`.

**It is therefore a property of this configuration and must be re-verified, not assumed.** P21's
M3 and M4 rungs add attention, and some scaled-dot-product backends use `atomicAdd`.

## Detection

Two metrics separate **entry into the pinned state** with no overlap across 30 seeds:

| | clean arms | arms that entered |
|:---|:---|:---|
| minimum `adv_std_raw` | 0.0874 – 0.1983 | **0.0036 – 0.0313** |
| maximum `explained_variance` | 0.9736 – 0.9913 | **0.9990 – 1.0000** |

Advantage variance collapses because every game ends the same way, and the critic's
`explained_variance` climbs to ~1.0 *because* the policy is degenerate — a one-sided policy is a
trivially easy prediction problem. This is
[`../../method/detecting_collapse.md`](../../method/detecting_collapse.md)'s "critic quality
measures the critic, not the policy" with a live instance: the critic looks **excellent** exactly
when the run is failing.

**Neither metric predicts escape.** Seed 21's minimum `adv_std_raw` was 0.0226, inside the
collapsed range. They detect entry, not outcome.

`adv_std_raw < 0.01` was previously **rejected** as a detector because it had 0 hits across 47 E3
runs. That rejection stands for E3 — but within this configuration the direction separates
perfectly. Consistent with the scope rule: the mechanism transfers, the threshold does not.

## Claims that were made and withdrawn

Recorded because the reasoning should be traceable, and because three of the four share one cause.

| claim | why it was wrong |
|:---|:---|
| "the country head alone buys nothing" | read off a **collapsed** arm's Elo, which measures the collapse |
| "the two heads are superadditive" | same collapsed arm |
| "M2d reaches the anchor" | a **four-seed** mean, and the four landed within 9 points of the best of all 495 possible four-seed draws |
| "bifurcation into an absorbing state; bimodal with nothing between" | seed 21 entered fully and escaped, and sits at 10.2% pinned — inside the gap claimed to be empty |
| "a partial per-entity correction distorts the card-vs-country trade-off" | impossible: over 25,600 decision points the legal mask offers **both** blocks 0.00% of the time |

The detector itself was wrong twice, in the same direction — once one-sided (it would have scored a
US-side collapse as clean), once using pinned *fraction* instead of *terminal state* (it scored the
recovery as a collapse and stopped a sweep early). Both errors would have silently corrupted the
statistic the sweep existed to produce.

## What is open

* **Why escape happens.** One instance. The decisive window is the rows just after entry, and
  resume states every 5M now bracket it for every arm.
* **Whether it is specific to `pe_card`'s removal.** M2 with both heads has never collapsed, but
  it has far fewer seeds. The comparison needs a matched sweep.
* **Whether the pinned phase costs strength even when survived.** Seed 21 finished 23 Elo below
  the clean mean — within noise, and n=1.
* **Which source of randomness carries it.** Attempted and inconclusive; see
  [`../../log/E4_collapse_attribution.md`](../../log/E4_collapse_attribution.md). Eight arms split
  `--seed` into initialisation, sampling, deals/dice and the opponent draw, one at a time in both
  directions. All eight came back clean, which **conclusively refutes sufficiency** for every
  single stream -- initialisation in particular, in both directions, so "some seeds just start in a
  bad place" is wrong. It establishes nothing about necessity: under a 12.5% base rate an
  all-clean result has probability 0.34, so "every stream is necessary" is simply what that base
  rate looks like from the inside. Answering it properly needs conditional rates over ~15-50 arms
  per cell.
* **Whether intervention can prevent it.** The branch points exist: take the 35M or 45M resume
  state and change exactly one thing — reseed only the sampling, widen the pool, alter
  `ref_update_freq`, restore `pe_card`. Every branch shares a byte-identical prefix, so a
  difference is attributable to the intervention alone. None has been run.
