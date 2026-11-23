# The E9 base's regression after ~950M (2026-11-22..23)

**Conclusion (2026-11-23): no defect -- the recipe.** The E9 base's weak 950 -> 1,200M stretch is ordinary for
pure self-play (`--opponent-frac 0`, the flagship since B6): runs of that recipe family, trained before and after
every change investigated here, mostly stall or lose ground against their own past over 950 -> 1,200M (40-53%),
while every pooled run measured (`--opponent-frac 0.3`) gains (54-59%, section 5). Comparing the base against R169
-- one of the two pure-self-play runs that happened to improve there -- made a normal stretch look like a regression.
The E9 engine, the 2026-11-22 fixes, the resume, torch.compile, the GPU and the toolchain are each ruled out (sections
2-4). The base continues; whether pure self-play needs a pool (or a longer pi_ref horizon) to improve reliably at
this stage is the owner's question.

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

## 5. Pool or no pool

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
