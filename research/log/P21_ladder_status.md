# P21 architecture ladder — status at 2026-09-21, and what remains

Everything below comes from **one tournament**,
`/workspace/data/reports/P21_M2d_160M_slope.json` (22 entrants, 100 games per side,
`temperature 0.0`). Bradley-Terry ratings are field-relative, so these numbers may be compared
with each other and with nothing else.

Sources: [`P21_M0_flat_mlp.md`](P21_M0_flat_mlp.md), [`P21_M1_grouped.md`](P21_M1_grouped.md),
[`P21_M2_lookup.md`](P21_M2_lookup.md),
[`P21_M2d_country_head_collapse.md`](P21_M2d_country_head_collapse.md),
[`P21_M2d_160M_slope.md`](P21_M2d_160M_slope.md),
[`E4_collapse_attribution.md`](E4_collapse_attribution.md),
[`../findings/training/side_collapse.md`](../findings/training/side_collapse.md).

## Where the ladder stands

| rung | what it adds | Elo @80M | steps/s | Elo/GPU-h |
|:---|:---|---:|---:|---:|
| **M0** flat MLP | nothing — the floor | 1651.7 | 63,187 | 431 |
| E4-04-01 | the *default* architecture | 1689.7 | 14,057 | 120 |
| **M2e** card head only | `pe_card`, no `pe_country` | 1746.3 | 46,313 | 513 |
| **M1** grouped projections | board/card/global each get their own dense projection | 1758.1 | 59,561 | 692 |
| **M2d** country head only | `pe_country`, no `pe_card` | **2040.3** (6 seeds) | ~50,800 | **1266** |
| **M2** both heads | `pe_country` + `pe_card` | 2050.2 | 35,744 | 885 |
| **anchor** E4-03-01 | the late-E3 bundle | 2089.7 | 11,732 | 311 |
| — HeuristicBot | | 1500.0 | — | — |

**The decomposition is now clean.** Against the 438 Elo from M0 to the anchor:

* grouped projections (M1) are worth **+106**
* the **country** per-entity head is worth **+282** on top of M1 — the single largest mechanism
* the **card** per-entity head alone is worth **−12**, i.e. nothing

M2d reaches **99.5% of full M2** at **1.4× its throughput**, and closes **89%** of the
MLP→anchor gap with no attention, no identity vectors, no pooling and no graph convolution.

## Settled

**1. Structure is worth far less than the late-E3 bundle implied.** The E3-era claim was ~110 Elo
for "architecture vs MLP". On this engine the *default* architecture is worth **+38** over a plain
MLP, while the late-E3 bundle is worth **+438** — and most of that +438 is one mechanism, the
country lookup.

**2. The card head buys nothing, and why is NOT established.** `forward_pass_trace.md` predicted
`pe_card` would be near-inert and that `pe_country` would carry the result. The prediction held —
but the reason it gave (that `pe_card` cannot distinguish 95 of 110 cards without an identity
vector) does not survive inspection, because only **4 of a card's 14 slots are static**. Ops, era,
one-time and is-scoring are constant; the other ten — location, playability, and the side slot
that flips with perspective — move constantly. A card is not a static object in this observation.

An earlier version of this section explained the −12 by saying `pe_card` addresses card *types*
and that type-level information is most of what card timing needs. That is an argument for why the
head *would* work, carried over from when seed 1's collapse made `pe_card` look load-bearing, and
it is incoherent as an explanation for a null result. It is withdrawn. The measurement stands; the
mechanism is open.

**3. The rung is still improving steeply at 80M: +159.8 Elo from 80M to 160M**, within-seed over
six seeds, sd 33.7. An 80M measurement understates M2d.

**4. Per GPU-hour, M2d at 160M dominates the anchor at 80M** — +56 to +138 Elo for 46–52% of the
wallclock, because the architecture is 4.3× cheaper per step. At matched *steps and 80M* the
anchor is still 54 Elo ahead. At matched steps and 160M the anchor is far behind, but only
because it fell (see Open, first item) — that margin is not M2d's to claim.

**5. The side collapse is real and not attributable to any single seed stream — but it is
recoverable, and mostly free.** Entry reaches ~12.5% of seeds by 80M. It is **not** an outcome:
5 of 7 arms scored COLLAPSED recover when continued 80M further, and a collapse that recovers
costs **−1.4 Elo** against arms that never collapsed. One that does not recover costs **−394**.
The "costs ~250 Elo and lands below M1" reading is withdrawn — it was measured on checkpoints
taken mid-episode
([`E4_collapse_is_recoverable.md`](E4_collapse_is_recoverable.md)). Splitting `--seed` into initialisation / sampling / deals / opponent-draw
and moving one at a time, in both directions, left all eight arms clean: **no single stream is
sufficient**, initialisation refuted in both directions. Entering the pinned state and escaping
costs nothing detectable.

**6. There is a second failure mode, and nothing here detects it.** The anchor's 330 Elo fall came
with `adv_std_raw` between 0.217 and 0.262 and `us_episode_frac` at 0.301 — every collapse
indicator healthy and correctly so. The policy became *indecisive*, not one-sided, and Elo tracked
policy entropy inversely across the whole leg. The side-collapsed M2d arm shows the same sustained
entropy inflation, so entropy may be the more general indicator.
[`../findings/training/entropy_inflation.md`](../findings/training/entropy_inflation.md).

## Open

* ~~**The anchor at 160M.**~~ **Measured, and it changed the question.** `E4-12-01` lost ~330 Elo
  over its second 80M — see [`../findings/training/entropy_inflation.md`](../findings/training/entropy_inflation.md).
  So the matched-steps comparison at 160M is +465.8 to M2d, but **the anchor fell rather than M2d
  pulling ahead**, and that margin must not be quoted as M2d's. The fair statement is M2d at 160M
  (2203.1) against the anchor's best measured state (2093.3 at 80M): **+110 Elo, at twice the
  steps and 46% of the wallclock.** Whether the anchor's fall is a property of the architecture or
  of one seed is unresolved — it is a single arm.
* **Whether collapse is M2d-specific.** M2 with both heads has never collapsed but has far fewer
  seeds. The remaining rungs answer this for free if collapse rate is recorded per rung.
* **Side balance.** Every ladder arm is USSR-favouring (+2.5 to +16.9 pp); the anchor is the only
  arm near even at **−2.7**. More training does not fix it — the 160M arms are no better balanced
  than their 80M selves. Nothing on the ladder currently targets this.
* **Why escape happens.** Resume states every 5M bracket the window for every arm; no branch
  experiment has been run.
* **Whether the anchor's fall replicates.** One arm, run with `seed = None`. A second anchor to
  160M is ~3.9 GPU-h and now outranks the remaining extension arms in value.
* **Whether `ref_update_freq` drives entropy inflation.** `kl_div` falls while entropy rises,
  which is the shape of a ratchet through `π_ref` refreshes. That turns the open
  `ref_update_freq` ablation into a test with a specific prediction.

## The plan from here

### Rungs remaining, in order

| rung | change | question it answers |
|:---|:---|:---|
| ~~M2a~~ ✅ | dynamic + constants, **no ctx** | **Done: −84 to −146 Elo.** The head does need the trunk; `pe_trunk` stays |
| ~~M2b~~ ✅ | dynamic + ctx, **no constants** | **Done: −40 to −73 Elo.** They are pulling weight; the trunk does not already carry them |
| ~~M2c~~ ✅ | dynamic only | **Done: −87 to −189 Elo** — but still +144 to +252 over M1, so the dynamic slots alone are 54–74% of the head |
| ~~M2.5~~ ✅ | full + **identity** | **Done: −47 to −123 Elo.** Rejected |
| ~~M2.5b~~ ✅ | identity, no constants | **Done: −60 to −178 Elo, and it COLLAPSES** on two seeds where M2d runs clean. Identity does not subsume the constants |
| **M3** | card↔card self-attention | |
| **M4** | card→country cross-attention | |
| **M5** | flatten → pooling | deliberately last: the most dubious mechanism, tested as a *removal* |

**The whole M2 family is now measured** —
[`P21_M2abc_head_inputs.md`](P21_M2abc_head_inputs.md) and
[`P21_identity_rungs.md`](P21_identity_rungs.md). **Nothing beats M2d.** The three removals lose
(M2a/M2b/M2c) and so do the three additions (M2.5, M2.5c, M2.5b) — on both seeds, at 80M, which is
the adoption axis. M2d's head input stands exactly as it is: dynamic slots, per-type constants and
trunk context, no identity vector, country only. What remains of the ladder is **M3, M4 and M5**.

### Protocol, unchanged

Two arms per rung: one to 80M, one to 160M. 80M is the primary axis (matched steps); the 160M arm
gives the slope within-seed. A rung advances only if it beats the rung below at 80M on both seeds.

### Protocol, amended by what the M2d sweep taught

1. **Run the collapse detector on every arm; a collapsed arm is re-run on a new seed and is never
   read as a strength measurement.** This is not optional hygiene — seed 1's collapse produced the
   published conclusion "the country head buys nothing", which was wrong by ~300 Elo and took a
   30-seed sweep to overturn.
2. **Record collapse rate per rung** as a standing column, so "is this M2d-specific?" is answered
   as a byproduct rather than with dedicated compute.
3. **The live alarm must trigger on a trend in `adv_std_raw`, not a threshold.** The census's
   "clean ≥0.0874, entered ≤0.0313, no overlap" separator broke on the first arms outside the
   original 30: seed 11 entered at 0.0827, inside the band reserved for clean arms. Outcome is
   bimodal; entry is a gradient.
4. **Never compare Elo across tournaments.** The anchor rates 2089.7 here and 2149.9 in the
   tournament quoted in the plan, unchanged.

### Expected cost, and the collapse tax

Eight rungs × 2 arms, at M2d-class throughput (~50,000 steps/s), is ~1.9 GPU-h per rung and
**~15 GPU-h total**.

**The collapse tax is far smaller than this section originally claimed.** It said ~2.2 of 16 arms
would collapse and need re-running, a ~14% overhead. That rested on treating a `COLLAPSED` verdict
as a failed arm, and it is withdrawn: entry reaches ~12.5% of seeds, but **an arm that enters and
recovers is worth as much as one that never entered** (−1.4 Elo, 0.05 pooled sd). Only
non-recovery costs anything, at 2 of 7 observed. Budget roughly **one re-run in twenty arms**, and
trigger it on `adv_std_raw` failing to return rather than on `us_episode_frac` pinning, which
flagged four arms that were fine
([`E4_collapse_is_recoverable.md`](E4_collapse_is_recoverable.md)).

That overhead is the argument for finishing the ladder before investigating the collapse:
routing around it costs ~1.5 GPU-h, while a conditional-rate attribution design costs 30–40 and is
not guaranteed to resolve anything. The ladder also generates collapse data across eight
architectures for free, and architecture variation is the one lever that has not been tried —
seed variation is exhausted.

### Housekeeping

The M2d sweep produced ~70 directories and 29 GB, all of one rung. **Seed 13 is kept as the rung's
representative** — rank 3 of 6 at both 80M and 160M, rank spread 0, so it biases later comparisons
in neither direction; a cherry-picked best arm would bias every later rung toward rejection.
Everything else moves to `data/checkpoints/archive/P21_M2d/`. Nothing is deleted: the resume
states of the collapsed arms and of the escapee are the branch points the intervention
experiments depend on.
