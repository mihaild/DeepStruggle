# How much an early read says, and how to read strength faster (2026-11-06)

**Conclusion.** The owner's suspicion is half right. A read against A13+R48's SWA is *not* noise:
its read-to-read scatter is 1.3–1.6 points (SD; the binomial floor is 1.1), and across the twelve
from-scratch E8 arms the 400M read already ranks the 2,000M reads with Spearman 0.93. What the
early reads cannot do is order **arms that are close** -- within ~4 points (~30 Elo) of each other --
because at 400–1,200M a run gains 50–150 Elo per 400M, so a 2-point lead is 70–200M of *pace*, the
size of a seed effect, and pace is what flips: of the nine matched pairs that were within 4 points
at 800M, three reversed sign later and one ended level; of the five pairs 4 or more points apart at
800M, none reversed. Every documented reversal in the record (tokens C1, deep vs shallow, seed 43
vs 44, A30-R90 vs the control, J1′ vs J1, J5 vs J1) is a close pair read before ~1,600M. The one
large early lead that evaporated, J5's +9 over J1 at 200M against the reference, was contradicted
from the start by the head-to-head (J1 58.4% at 400M): the reference read and the pairwise
comparison disagreed by ~90 Elo at 400M, a non-transitive triangle that had closed by 800M.

So the fix is not more games against the reference. It is (1) make the **matched-step
head-to-head against the designated control** the decision instrument and the reference read the
trend instrument; (2) add the **own-past match** (SWA(t) against SWA(t − 400M)) so pace is measured
rather than inferred; (3) use **decision rules gated on margin, not on step**: a verdict before
1,600M only when the pairwise gap is at least 6 points (≈ 40 Elo) at two consecutive reads, or a
catastrophe (a read under 60% of the control's, a failed stability check); (4) stop calling
plateaus from the reference reads -- the R48 line gained ~34 Elo against its own past while its
reads sat in a 48–52% band -- and call them from SWA(t) against SWA(t − 1,600M). A read costs 5–15 s
of GPU (2,000 games; 181 s for the 4×4 J1/J5 field), so the protocol's extra matches are free next to
the runs they judge. §6 gives a 1.5-GPU-hour validation on checkpoints already on disk.

Scripts: `/tmp/rr/*.py` (ad hoc, not committed); every number below is reproducible from the files
named in §7. Related: [`variance_and_noise.md`](variance_and_noise.md) (tournament noise ~1.5
points, within-run oscillation ~30 Elo), [`../method/measurement_pitfalls.md`](../method/measurement_pitfalls.md),
[`../method/running_experiments.md`](../method/running_experiments.md),
[`../findings/training/training_length_and_evaluation.md`](../findings/training/training_length_and_evaluation.md).

---

## 1. What a read is, and what it costs

A read is the SWA of the run's last 80M (one snapshot per 10M), greedy, 1,000 deals a side against
`E7-A8-R1-S44@6400M+R34@7200M+R48@9170M+A13@12730..12800M` ("the reference", E7's strongest greedy
network, A8-shaped). `tools/tournament.py` plays every deal from both seats under the same shuffle
(`BatchMatchRunner.play_parallel_matchup`: "deal luck then cancels between the halves"), and the
default `--base-seed 10000` gives every read the same 1,000 deals, so reads at different steps and
of different arms are already **common-random-number** designs as far as the deal goes; the
agents' play diverges from the first differing decision, so the pairing buys the opening deal and
little after ([`seed_variance_and_pooling.md`](seed_variance_and_pooling.md), method notes).

Cost, from `total_time_seconds` in the report JSONs (GPU, beside a training run): **5–11 s** for an
A8-shaped arm, **13–14 s** for the transformer J5 (`recipe_E8-A35-R102-S44_*_vs_a13.json`), 18–19 s
for a J5 self-play match, 181 s for the 4×4 J1/J5 field (six pairings, 12,000 games). On CPU the
old measurement is 167 games/s on 22 CPU-s for a five-player field (`tools/README.md`, *Throughput
and CPU*), so a 2,000-game read is about a minute of cores without touching the GPU. **The read is
not expensive; its information is what is scarce.** Everything below spends games, not GPU-hours.

The scale: at the reference's strength a read of p sits at Elo 400·log10(p/(1−p)); the SE of 1,000
deals a side is 0.67 points at p = 0.10 (13 Elo), 0.89 at 0.20 (9.7 Elo), 1.1 at 0.50 (7.8 Elo). The
compression at low win rates costs a factor ~1.7 in Elo precision at 10% and ~1.25 at 20% -- real
but minor next to the effects below.

## 2. How predictive early reads have been

### 2.1 The dataset

All from-scratch E8 arms with a reference read at 400M and at least one at 2,000M or later, from
`data/reports/recipe_<run>_<step>M_vs_a13.json` (SWA reads; `raw_J*_*` are raw snapshots). Values
are the matchup's `win_rate_a` in percent (draws count as losses, so they sit ~0.3 under the
numbers quoted in [`../runs.md`](../runs.md), which use wins over decided games).

| arm | 400 | 800 | 1,200 | 1,600 | 2,000 | 2,400 | 4,000 | last |
|:---|---:|---:|---:|---:|---:|---:|---:|---:|
| E8-A8-R1-S44 (control, R1 4k) | 9.7 | 16.2 | 18.1 | 24.1 | 24.9 | 28.7 | 33.3 | 39.9 @12,000 |
| E8-A8-R93-S44 (clip 0.1) | 10.1 | 15.2 | 24.6 | 24.6 | 27.3 | 31.1 | 35.6 | 35.6 @4,000 |
| E8-A8-R95-S44 (16k, clip, muP) | 14.5 | 18.5 | 24.4 | 30.5 | 35.4 | 37.3 | 43.5 | 47.9 @7,600 |
| E8-A8-R102-S44 (J1) | 15.7 | 30.0 | 33.1 | 34.8 | 38.0 | 40.6 | 47.2 | 62.7 @17,600 |
| E8-A8-R103-S44 (J1′) | 16.0 | 31.2 | 35.3 | 38.6 | 39.3 | 40.8 | 48.5 | 57.2 @10,800 |
| E8-A30-R90-S44 (deep, clip) | 10.9 | 14.1 | 20.3 | 22.4 | 23.0 | 28.3 | 31.7 | 32.8 @4,800 |
| E8-A30-R95-S44 (deep, muP) | 12.0 | 20.7 | 23.9 | 26.4 | 28.3 | . | . | 28.3 @2,000 |
| E8-A26-R1-S44 (wide) | 10.0 | 15.4 | 17.2 | 20.6 | 20.7 | 23.9 | . | 23.9 @2,400 |
| E8-A27-R86-S44 (sphere) | 7.1 | 12.1 | 14.2 | 17.8 | 18.0 | . | . | 18.0 @2,000 |
| E8-A30-R86-S44 (sphere 0.35) | 8.1 | 9.9 | . | 17.3 | 17.2 | . | . | 17.2 @2,000 |
| E8-A25-R47-S44 (deep, 16k) | 9.7 | 15.2 | 17.8 | 23.6 | . | 26.1 | . | 26.1 @2,400 |
| E8-A26-R84-S44 (wide, lr 2e-4) | 7.8 | 13.4 | 15.7 | 19.1 | . | . | . | 19.1 @1,600 |
| E8-A35-R102-S44 (J5, transformer) | 23.1 | 32.6 | . | . | . | . | . | 32.6 @800 |

**Confounds, stated once.** (a) *Censoring:* arms that read badly early were stopped (A22, A25-R1,
A27-R82, A30-R89, A32 ... at 400M; A26-R85 at 800M), so the sample contains no early loser that was
allowed to recover, and the correlations below are for survivors. (b) *Family structure:* the
ordering is dominated by one split -- the trust-region recipes (R95/R102/R103: 16k minibatch, clip
0.1, muP) against the 4k-minibatch ones -- which any read separates; the interesting question is
the ordering *within* a family. (c) One seed each (S44). (d) Every read is against one reference
network from another epoch (E7); the E7-era cases in §2.3 were read against other panels.

### 2.2 Rank correlation, early read against later read

| early → late | n | Spearman | Kendall (concordant / discordant pairs) |
|:---|---:|---:|:---|
| 400 → 2,000M | 10 | +0.93 | +0.82 (41 / 4) |
| 800 → 2,000M | 10 | +0.93 | +0.82 (41 / 4) |
| 1,200 → 2,000M | 9 | +0.93 | +0.83 (33 / 3) |
| 1,600 → 2,000M | 10 | +1.00 | +1.00 (45 / 0) |
| 400 → 4,000M | 6 | +0.83 | +0.73 (13 / 2) |
| 800 → 4,000M | 6 | +0.94 | +0.87 (14 / 1) |
| 400 → last read ≥ 4,000M | 6 | +0.71 | +0.47 |
| 800 → last read ≥ 4,000M | 6 | +0.94 | +0.87 |
| 1,200 → last read ≥ 4,000M | 6 | +0.66 | +0.47 |
| 2,000 → last read ≥ 4,000M | 6 | +0.89 | +0.73 |

High, and misleading in the way (b) predicts: the discordant pairs are all *close* pairs. The
**last-read** column is the honest one and it is not monotone in the early step (1,200M ranks the
finals worse than 800M) because with six arms a single close-pair flip moves Kendall by 0.13.

### 2.3 Matched-step pairs: when the sign held and when it flipped

Difference (first minus second) of the reference reads at matched steps, in points:

| pair | 400 | 800 | 1,200 | 1,600 | 2,000 | later | verdict |
|:---|---:|---:|---:|---:|---:|:---|:---|
| J1 − J1′ | −0.3 | −1.2 | −2.2 | −3.8 | −1.3 | −0.2 … −5.4 to 8,000M, then +0.3 … +2.0 from 8,400M; J1′@9,200M scores 41.9% against J1@13,600M | **flipped** late: J1′ led 20 of 21 reads to 8,000M |
| A30-R95 − A8-R95 (deep vs shallow, same recipe) | −2.4 | +2.2 | −0.6 | −4.1 | −7.0 | — | **flipped**, then a 7-point deficit |
| A30-R90 − control | +1.2 | −2.1 | +2.2 | −1.8 | −1.9 | −0.4, −0.4, −1.6, +0.1 to 4,800M | **sign changed five times**; ends level |
| A8-R93 − control (clip 0.1) | +0.3 | −1.0 | +6.5 | +0.5 | +2.4 | +2.3, +3.2, +2.3 | small lead emerges only from 2,000M; the 1,200M +6.5 was a high read |
| A26-R1 − control (width) | +0.3 | −0.7 | −0.8 | −3.6 | −4.2 | −4.8 | level for three reads, then behind |
| A25-R47 − control | +0.0 | −0.9 | −0.3 | −0.5 | . | −2.6 @2,400 | level for four reads; head-to-head at 1,600M already 46.5% |
| J1 − A8-R95 (the term) | +1.2 | +11.5 | +8.6 | +4.3 | +2.6 | +3 … +8 to 7,600M, then J1 62 vs R95 48 plateau | **held** from 800M; invisible at 400M |
| A8-R95 − control (16k, clip, muP) | +4.8 | +2.3 | +6.4 | +6.4 | +10.5 | +8 … +12 | held |
| A30-R95 − A30-R90 (muP on the deep trunk) | +1.1 | +6.6 | +3.6 | +4.0 | +5.3 | — | held |
| A27-R86 − control | −2.6 | −4.1 | −3.8 | −6.3 | −6.9 | — | held |
| A30-R86 − control | −1.7 | −6.3 | . | −6.9 | −7.7 | — | held |
| A26-R84 − control | −1.9 | −2.8 | −2.4 | −5.0 | . | — | held |
| J5 − J1 (SWA reads) | +7.4 (+9.1 at 200M) | +2.6 | . | . | . | raw reads +5.1 → −0.8; head to head J1 58.4% at 400M, 49.6% at 800M | **the reference lead was never a head-to-head lead** |

Tally for pairs with an 800M read and a verdict: gap ≥ 4 points at 800M -- 5 pairs, 5 held; gap
< 4 points -- 9 pairs, 3 reversed, 1 ended level, 5 held (two of the five, A26-R1 and A25-R47,
"held" at 0.0–0.9 points, which is no information). At 400M the equivalent tally is worse: 3 of 10
close pairs reversed and the one large gap (J5) was not a strength gap.

The E7 record has the same shape (reads then were fields with a 3-snapshot panel, 400–1,000 games a
side):

* **Tokens C1** ([`P30_C1_tokens.md`](P30_C1_tokens.md)): +56% head to head against its paired
  control at 200M, ahead of every shallow run; level on 400M SWAs (51.2%); stopped by the rule.
* **Deep vs shallow trunk** ([`E7_06_44_deep_to_saturation.md`](E7_06_44_deep_to_saturation.md)):
  +7 to +30 Elo for shallow at 320–640M, i.e. level; +92 to +137 from 720M; 64.8% head to head at
  1,200M. The *absence* of a difference early was the misread.
* **Seed 43 vs 44** (E7-08-43, [`../runs.md`](../runs.md)): 55 Elo apart at 1,200M, level at 2,000M
  (50.9%) -- "the spread was pace".
* **A8 heads vs A4** ([`P31_branch_arms_B1_B2.md`](P31_branch_arms_B1_B2.md)): not accepted at
  1,200M (46.9%, US −3.3), level at 2,000M, ahead at 4,000M (53.9%) and at plateau. The eventual
  base architecture read *worse* at 1,200M.
* **The plateau calls**: E7-02-44 "saturated" at ~900M, E7-20-44 at 2,800M by the pre-registered
  rule, then rose to ~3,700M ([`E7_line_to_2000M.md`](E7_line_to_2000M.md)); J1 stopped at 6,810M by
  the one-firing rule on a 49.5% read whose successor was its best (53.6%)
  ([`../plans/P41_jaxcalibur_ingredients.md`](../plans/P41_jaxcalibur_ingredients.md)); the control's
  36–40% band from 5,600M hid a +18 Elo gain against its own past over 3,600M, and R48's 48–52% band
  a +34 Elo gain over 1,600M ([`../runs.md`](../runs.md)). Three of five plateau calls from reference
  reads were premature.

### 2.4 Why: the ruler moves faster than the noise

Elo gained per 400M window, from the reference reads: the control +101 (400→800M), +24, +63, +7,
+34; J1 +145, +25, +14, +24; A8-R95 +51, +62, +53, +38; A30-R90 +51, +76, +21, +6. **At 400–1,200M a
run gains 50–150 Elo per 400M**, i.e. 1 point of read (≈ 13 Elo at p ≈ 0.15) is 35–100M of
training. A 2-point lead at 400M is 70–200M of pace -- the size of the seed effect (two seeds 55 Elo
apart at 1,200M, level at 2,000M) -- and the read's own scatter (§3.1, ~1.4 points) is ±50–100M of
pace. Past ~4,000M the ruler moves 0–25 Elo per 400M and the scatter (±15–20 Elo) is the same size
as the signal, which is why plateau calls from these reads fail in the other direction.

Two further facts the matched-step record shows: differences **appear late** as often as they
disappear (A26-R1 level for 1,200M then −5; deep vs shallow level for 640M then −120 Elo; J1's term
+1.2 at 400M then +11.5), and **the eventual best run of the project was indistinguishable from its
control at 400M** (J1 15.7 against R95's 14.5).

### 2.5 Training-time proxies do not rank arms

Means over the 320–400M window of each run's `training_metrics.jsonl`, against the 2,000M read
(n = 10; `/tmp/rr/proxies.json`):

| metric at 400M | Spearman with the 2,000M read | note |
|:---|---:|:---|
| the 400M read itself | +0.93 | |
| per-update `approx_kl` / `kl_div` | −0.98 / −0.96 | indexes the recipe family (16k, clip 0.1 → small steps); the 60M step-size check already uses it ([`../method/network_stability.md`](../method/network_stability.md)) |
| `opp_win_rate_mean` (against the pool) | −0.90 | higher pool win rate, weaker later -- the pool is the run's own past |
| `entropy` | +0.62 | |
| `explained_variance`, `critic_auc`, `critic_brier_skill` | −0.50 / −0.41 / −0.41 | the deep trunks had the better critics and the worse play |
| `eval_win_rate_HeuristicBot` | (not computable: 99%+ for every arm) | saturated, as [`../method/measurement_pitfalls.md`](../method/measurement_pitfalls.md) says |
| decisive-win take rate, blunder rates, endings, mean ply | \|ρ\| ≤ 0.4 | |

Nothing here is a strength proxy. The per-update KL is a *recipe* proxy -- it says which family a
run is in, which a 60M check already does -- and the critic metrics point the wrong way. This
confirms the standing rule ("no live training metric ranks arms") with numbers.

## 3. The noise sources, ranked

### 3.1 Read-to-read scatter on plateaus: 1.3–1.6 points, 25–38 Elo peak to peak

On stretches the record calls flat, the SD of consecutive reads (and the first-difference noise,
which removes any slow trend):

| stretch | n | mean | SD | diff-noise | US SD | USSR SD | range (Elo) |
|:---|---:|---:|---:|---:|---:|---:|---:|
| J1 12,400–17,600M | 14 | 60.9 | 1.46 | 1.33 | 1.94 | 2.21 | 59.0–63.7 (35) |
| J1 8,800–11,600M | 8 | 57.7 | 1.34 | 1.15 | 1.32 | 2.04 | 55.5–59.5 (28) |
| J1′ 8,400–10,800M | 7 | 56.5 | 1.37 | 1.45 | 1.88 | 1.39 | 54.7–58.9 (30) |
| R94 13,600–16,000M | 5 | 51.1 | 1.46 | 1.45 | 1.96 | 1.15 | 48.8–52.9 (29) |
| R48 10,800–13,200M | 7 | 48.7 | 1.59 | 1.43 | 1.67 | 1.79 | 46.5–52.0 (38) |
| control 5,600–10,000M | 6 | 37.5 | 1.37 | 1.58 | 1.86 | 1.28 | 35.9–39.3 (25) |
| A8-R95 4,400–7,600M | 9 | 45.8 | 1.38 | 0.90 | 0.98 | 2.33 | 44.2–47.9 (26) |

The binomial SE of 2,000 games is 1.1 points; per seat (1,000 games) 1.55. So **a read carries
about 1.4 points of scatter, of which 1.1 is sampling and the rest is the policy moving between
80M windows**, and a per-seat number carries ~1.9 against a 1.55 floor. Independent confirmation:
the leaderboard's 67 greedy E7 matches fit a Bradley-Terry model with residual SD 1.23 points (RMS
z = 1.13), and two identical 6,000-game tournaments differed by 1.5–1.6 points
([`variance_and_noise.md`](variance_and_noise.md)). The E7 40M raw-snapshot trace (400 games a side)
oscillates 5.5 Elo around a moving average -- the SWA tames the snapshot wobble that was 30 Elo on
the E3 runs; what remains is sampling plus a little drift.

**Consequence:** two reads differ by 2 points at 1σ-ish; 3 points is the threshold at which a
*single-step* difference between two arms is more likely real than not, and a 4-point gap is
~2σ. The owner's stop rule (0.5 points over three reads) and the +1.5 plateau rule are both below
the scatter, which is why they fire on plateaus and miss slow gains.

### 3.2 Non-transitivity against a single reference: small among siblings, ~90 Elo for J5 at 400M

Among the E7 lineage (one architecture family), the Bradley-Terry residuals above say the greedy
pairwise results are transitive to within 1.2 points -- a single fixed reference ranks siblings
correctly. The J1/J5 case is different. A Bradley-Terry fit over the 4×4 head-to-heads plus the raw
reference reads (`j1_j5_400_800_4x4.json`, `raw_J*_vs_a13.json`) leaves residuals of **+3.4 points
for J5@400M against the reference, −3.5 for J1@400M against the reference, +4.4 for J1@400M against
J5@400M** (each 3–4 SE) and under 1 point for every pairing involving an 800M network. In Elo: the
reference said J5@400M was ~35 ahead of J1@400M, the head-to-head said ~60 behind -- a ~95 Elo
triangle that had closed by 800M. The mechanism is visible in the seats: J5's 400M US, the one that
dropped West Germany, scores 13.0% as US against the reference and loses 71% to J1's USSR, while
beating its own 200M and 320M selves by 57–60% (`j5_430M_vs_*M.json`). An early policy is tuned to
its own opponent pool; how much of that transfers to one foreign network is architecture- and
stage-dependent. This source is *large* exactly when the read is being asked to compare different
architectures early, and small otherwise.

### 3.3 Seat-strategy cycling: the biggest single-read swing

Per-seat reads move 2–5 points between consecutive reads with no overall change (R94 at 14,400 →
14,800M: US 47.2 → 43.2, USSR 58.6 → 58.1; J1 at 16,000 → 16,400M: US 59.4 → 54.8, USSR 63.6 →
66.1). J5's US seat fell 15.5 → 12.1 raw between 200M and 320M while its USSR rose 16.3 → 22.1; J1's
US setup swapped West Germany in and out over 1,600–7,200M. The overall read averages the two seats,
which halves the damage, but a per-seat acceptance rule read at one step sees a strategy swing as a
strength change. The per-seat bar (the control's own self-play rate in that seat) is right and
should stay; what is missing is reading seats over **two consecutive windows**.

### 3.4 SWA against raw: 3–9 points, and arm-dependent

| | 200M | 400M | 800M |
|:---|---:|---:|---:|
| J1 SWA − raw | +4.5 (+96 Elo) | +3.1 (+45) | +5.1 (+44) |
| J5 SWA − raw | +4.7 (+56) | +5.5 (+59) | +8.5 (+73) |

The SWA's gain over its raw snapshot is not a constant: it is 3 points larger for J5 than for J1 at
800M, so an SWA comparison and a raw comparison of the same two arms differ by ~30 Elo. Both are
legitimate players; the SWA is the better estimate of *the recipe's* strength (it removes the
optimiser wobble that a learning-rate step-down would), the raw snapshot of *the policy that is
training*. The decision instrument should use one consistently (SWA, as now), and the head-to-head
in §4 should be SWA against SWA.

### 3.5 Greedy determinism and deal variance

Greedy play means a read is a sum over 1,000 deals of nearly deterministic games; the variance is
deal variance plus the few sampled chance events. Greedy and T = 0.1 are the same player; T = 1
costs 100–370 Elo and moves arms differently (R34: +7.6 per seat greedy, 36.6% at T = 1;
[`../method/measurement_pitfalls.md`](../method/measurement_pitfalls.md)). The greedy read is the
right ruler for a greedy player; a T = 1 read is a separate question to ask only when a treatment
changes entropy.

### 3.6 Which dominate

1. **Pace confounded with ceiling** (§2.4) -- not a noise source but the reason early ordering of
   close arms is uninformative; dominates every reversal before 1,600M.
2. **Non-transitivity against the single reference** (§3.2) -- dominates cross-architecture reads
   before ~800M (the J5 case), negligible within a lineage.
3. **Seat cycling** (§3.3) -- dominates per-seat single-read verdicts at every stage.
4. **Sampling + window scatter** (§3.1), 1.4 points -- the floor; sets the thresholds.
5. SWA/raw choice (§3.4), compression at low p (§1): second order.

## 4. Methods for faster or more reliable strength estimates

What each would cost here (games; a 2,000-game A8 match is ~8 s of GPU), what it fixes, and what
the evidence is.

| method | cost here | fixes | evidence / caveat |
|:---|:---|:---|:---|
| **Matched-step head-to-head against the designated control**, SWA vs SWA, same deals | one extra 2,000–4,000-game match per read (8–30 s) | the reference's non-transitivity (§3.2) and its compression; reads the question actually asked ("is this arm better than its control?") | J1 vs J5: the head-to-head was right at 400M when the reference was not; A25-R47's 46.5% at 1,600M preceded the reference gap by 800M; this project's own E4–E7 practice before the single-reference reads |
| **Own-past match** SWA(t) vs SWA(t − 400M) and vs SWA(t − 1,600M) | one or two 2,000-game matches per read | separates pace from ceiling; detects slow gains that a flat reference band hides | control +18 Elo and R48 +34 Elo against their pasts while the reference reads were flat; J1/J5 each gained ~65% over 400→800M -- a direct pace number |
| **Paired deals / common random numbers** | none -- already in place (both seats per deal, same `--base-seed`) | deal variance between the two seats of one deal, and between arms | duplicate bridge, the poker "duplicate" estimator (Zinkevich et al. 2006); the tabletop measurement found only 0.1–0.4 point CI gains because agents diverge, which matches our note that a fixed seed buys the opening deal and little after |
| **Value-baseline variance reduction (AIVAT-style control variates)** | a critic pass per decision, offline | the variance from chance events (dice, deals) using the critic as the baseline; 85% SD reduction in poker (Burch et al., AAAI 2018, ~44× fewer games) | would need a trusted value function; ours is the thing under test and is miscalibrated early (critic AUC 0.72–0.83 at 400M), so the gain is uncertain; worth a probe only if game counts ever bind, which they do not |
| **Sequential tests (SPRT, as in fishtest)** | stops a match early: a 20-Elo question at ±1 point resolves in ~2,000–6,000 games instead of a fixed 20,000; pentanomial scoring of deal pairs tightens the interval further | over-spending on clear results and under-spending on close ones; the 0.5-point stop rule becomes a stated hypothesis with error rates | Stockfish's fishtest (GSPRT, normalized Elo, pentanomial model since paired openings: the trinomial model overstates the interval and runs ~20% longer). Here the binding constraint is the arm's pace, not games per match, so SPRT helps at the plateau (close head-to-heads) and not at 400M |
| **Fixed diverse panel / Elo on the leaderboard** instead of one reference | 3–5 matches per read | style-matching one opponent; averages the non-transitive residuals | the leaderboard already exists per epoch; among siblings the residuals are 1.2 points so the panel adds little; across architectures (J5) it would have halved the 400M triangle. Nash averaging / α-rank (Balduzzi et al. 2018; Lanctot et al. 2023's VasE) address cycles in a population -- overkill for 2–4 arms, and α-rank is sensitive to payoff noise |
| **League-style evaluation** (AlphaStar's Nash of the league, OpenAI Five's TrueSkill against past versions, Lc0/LZ self-play Elo) | a population to maintain | relative progress within one lineage | all three communities report inflation against fixed external opponents (Lc0 "cumulative self-play Elo" reaches thousands; Leela Zero's successive-version Elo inflates against ELF); the fix every one of them adopted is matches against a *fixed set* of external networks -- which is what the reference read is. Keep it as the trend |
| **Exploiters / best responses** | a 200M training run per exploiter (R25/R26: ~40% of a slot) | finds a seat's exploitability that win rates miss | R25 found +11.6 of USSR exploitability at the E7 plateau, R26 nothing; the league never published from a plateau main; a *strength* read it is not |
| **Temperature-1 reads** | one extra match | detects a greedy-only gain (R34) | run once per arm at 2,000M+, not per read |
| **Successive halving / Hyperband** | none; a schedule | spends compute on survivors | assumes the early ranking is informative, which §2.3 says holds only for gaps ≥ 4–6 points; the 2024 study (arXiv 2404.04111) found "minimal, oftentimes no utility" from early-stopping schedules even in supervised nets; with 1–2 arms at a time and one GPU, halving is already what the owner does by hand |
| **Learning-curve extrapolation** (freeze-thaw, LC-PFN, ifBO; Wistuba's learning-to-rank curves) | a model fit | predicts the final from the partial curve | built for supervised validation curves with a parametric prior; our curves have flat-then-jump stretches (J1 51.8 flat for 2,400M, then 55.8) and seat-driven detours that no such prior contains; not recommended beyond a visual trend |
| **Interquartile mean / stratified bootstrap over seeds** (Agarwal et al. 2021) | more seeds | seed spread (~15 Elo) | right but expensive here; one seed per arm is the standing constraint, so the protocol must be robust to it instead |
| **AlphaGo Zero-style gating** (promote at 55% of 400 games) | a match per snapshot | a stated acceptance threshold | AlphaZero dropped the gate as unnecessary; the number itself (55% ≈ 35 Elo at n = 400, SE 2.5 points) is a reminder that a 2–3 point margin at 2,000 games is below what any of these systems acted on |

## 5. The read protocol (recommended)

**Instruments.** Every 400M (200M stays as a sanity read only), on the SWA of the last 80M, greedy:

1. **The reference read**, 1,000 deals a side as now -- the common ruler across arms and epochs.
   Never the basis of a verdict on its own.
2. **The matched-step head-to-head against the designated control's SWA at the same step**,
   **2,000 deals a side** (4,000 games, 15–30 s), with the seats read against the control's own
   self-play rate at that step (one 2,000-deal self-play match per control step, cached and
   reused by every arm that shares the control). SE 0.8 points overall, 1.1 per seat.
3. **The own-past match**, SWA(t) against SWA(t − 400M), 1,000 deals a side (and against
   SWA(t − 1,600M) from 2,000M on). Reported as the arm's pace next to the control's pace at the
   same step.
4. Per seat, every number is quoted as the mean of the **last two reads** (two 80M windows 400M
   apart), never one.

Reads 2 and 3 for the transformer cost ~40 s per read; for an A8 arm ~20 s. A run read every 400M at
~5k–100k steps/s spends under 0.1% of its wall clock on this.

**Rules.**

* **Catastrophe (any step from 400M):** stop when the reference read is below 60% of the control's
  at the same step (≈ −90 Elo; every arm that read ≤ 5.5% against a ~10% control was stopped or
  superseded, and none was later shown to recover) or a stability check fails ([`../method/network_stability.md`](../method/network_stability.md)).
  This is the only verdict a 400M read supports.
* **Before 1,600M, no acceptance.** A stop for strength needs the head-to-head at or below **44%
  at two consecutive reads with the gap not shrinking** (≈ −40 Elo twice; taking the reference gap
  as a stand-in for the head-to-heads that were not played, A30-R86 and A27-R86 would have stopped
  at 1,600M and 2,000M, A26-R1 at 2,000M, and nothing that later recovered would have).
  An arm that trails by less keeps running: in the record such gaps reversed in 3 of 9 cases.
* **From 1,600M, acceptance over the control** needs the head-to-head at or above **54% at two
  consecutive reads** (≈ +28 Elo twice), with neither seat significantly below its bar. A lead
  between 50 and 54% is "not distinguished yet": decide by **pace** -- keep the arm if its own-past
  gain per 400M is at least the control's over the last two reads (its ceiling may be higher;
  J1's lead over R95 grew from +2.6 at 2,000M to +14 at the plateau), stop it if its pace is below
  the control's for two reads (the lead is residual pace from earlier and will close, as A30-R95's
  did).
* **Cross-architecture arms (a J5) get no verdict against the reference before 800M** -- the
  triangle in §3.2 -- and the head-to-head against the matched control is the only read that
  counts at 400M; a reference lead that the head-to-head does not confirm is reported as
  "reference-only".
* **Plateau** is called from SWA(t) against SWA(t − 1,600M): below 52.5% (≈ +17 Elo over 1,600M,
  2σ at 2,000 deals a side) at two consecutive reads. The reference reads stay as the record of
  where the plateau sits on the common scale. The 0.5-point two-firings rule is retired: it is
  below the read's scatter (§3.1) and fired falsely at 6,810M.
* **Close pairs at the plateau** (J1 vs J1′, R94 vs R48): decide with one **SPRT-style
  head-to-head** between the two best SWAs, bounds [0, +20 Elo] (≈ 50–53%), 4,000 deals a side cap,
  stopping early when the log-likelihood ratio crosses ±2.94; ~5–10 minutes of GPU at most. Below
  +20 Elo the two are the same recipe for every purpose this project has.
* **Temperature 1** once per accepted arm at its plateau, against the control at T = 1, when the arm
  changed the entropy term; not otherwise.

**What stays unchanged.** Greedy, SWA of 80M, one seed, per-seat bars from the control's
self-play, the leaderboard for ratings, `--base-seed 10000`.

## 6. The validation experiment (existing checkpoints, no training)

Everything needed is on disk: every from-scratch E8 arm has `swa_320-400M.pt` … `swa_1920-2000M.pt`
(e.g. `data/checkpoints/E8-A8-R95-S44_20261029_155800/`, `E8-A30-R90-S44_20261027_045701/`), and
six of them have SWAs to 4,000M or beyond.

1. **Replay the record under the protocol.** For the ten arms of §2.1 with 2,000M SWAs, play every
   pair at 400, 800, 1,200, 1,600 and 2,000M, SWA vs SWA, 2,000 deals a side: 45 pairs × 5 steps ×
   4,000 games = 900k games, **~55 min of GPU** at the measured ~270 games/s (or ~90 min of CPU
   cores at ~170 games/s, leaving the GPU to J5). Truth = the 4,000M-or-later head-to-head for the
   six arms that have one, the 2,000M head-to-head otherwise. Score: at which step does the §5 rule
   give the right verdict, how often does it give a wrong one, and how many GPU-days of training
   it would have saved or cost against the stops actually made.
2. **Own-past pace.** The same ten arms, SWA(t) vs SWA(t − 400M) at each step: 40 matches × 2,000
   games, **~5 min**. Check the claim that the arms whose leads reversed had the lower pace at the
   time (A30-R95 against A8-R95 at 1,200–1,600M is the test case).
3. **Noise calibration.** Repeat 30 of the matches with `--base-seed 20000`: **~7 min**. Gives the
   replicate scatter of a paired-deal head-to-head directly (the record's 1.5–1.6 points comes from
   6,000-game tournaments in 2026-09), and, by playing the four raw snapshots 1,970–2,000M of two
   arms against the control beside their SWA, the split between sampling and window drift.
4. **The triangle.** The 4×4 J1/J5 design at 1,200M and 1,600M when J5 gets there (12,000 games,
   3 min each): whether the reference/head-to-head disagreement stays closed past 800M, which
   decides whether the cross-architecture exception in §5 can be dropped at 800M or must stay to
   1,600M.

Total: about 1.5 GPU-hours, or an afternoon of CPU; no new training, no engine change, and the
outputs are ordinary `tools/tournament.py` JSONs under `data/reports/`. If (1) shows the 54% / 44%
thresholds mis-call more than one of the ten arms, the thresholds move to 55% / 43% before the
rule is used on J5.

## 7. Data

* Reference reads: `data/reports/recipe_<run>_<step>M_vs_a13.json` (220 files, `matchup` block;
  `win_rate_a`, `win_rate_a_as_us`, `win_rate_a_as_ussr`, `total_time_seconds`); raw reads
  `raw_J1_*`, `raw_J5_*`.
* Head-to-heads: `j1_j5_400_800_4x4.json` (`per_side`), `recipe_A26_vs_A8-2_*.json`,
  `recipe_A25-R47_vs_A8-2_1600M.json`, `j1_vs_r95_*.json`, `r95_selfplay_*.json`, `j1_vs_r94_*.json`,
  `control_10000M_vs_control_6400M.json`, `r48_10400M_vs_control_10000M.json`,
  `j1_17200M_vs_j1_16800M.json`, `j1p_9200M_vs_j1_*.json`, `j5_430M_vs_*M.json`, `j5_*M_selfplay.json`.
* Training-time proxies: each run's `training_metrics.jsonl`, means over `total_steps` in
  [320M, 400M], [720M, 800M], [1,920M, 2,000M].
* E7: `data/reports/e7_trace_400_2000M.json`, `leaderboard/matches/E7.jsonl` (67 greedy matches, 14
  players, Bradley-Terry residual SD 1.23 points).
* Literature consulted (web, 2026-11-06): Fishtest mathematics (GSPRT, normalized Elo, pentanomial
  model); Burch et al. 2018 (AIVAT); Zinkevich et al. 2006 (duplicate-style low-variance estimators);
  Balduzzi et al. 2018 (Nash averaging) and 2019 (open-ended learning); Lanctot et al. 2023 (VasE);
  Agarwal et al. 2021 (IQM, stratified bootstrap); Adriaensen et al. 2023 (LC-PFN), Rakotoarison
  et al. 2024 (ifBO), Wistuba & Pedapati 2020 (learning to rank learning curves), arXiv 2404.04111
  (early stopping offers little utility); Lc0 / Leela Zero self-play-Elo inflation notes; AlphaStar
  league evaluation by the league's Nash; OpenAI Five's TrueSkill. The project's own bibliography:
  [`../method/references.md`](../method/references.md).
