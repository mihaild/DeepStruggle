# P21 rung M0 — the flat MLP floor

**2026-09-19.** The first rung of [P21](../plans/P21_architecture_ladder.md). Two arms on
different seeds, one to 80M and one to 160M, parameter-matched at 3.177M against the anchor
architecture's 3.149M (+0.9%), cold-started, pooled, `--drop-static`.

All ratings below come from **one tournament** (`data/reports/P21_M0_complete.md`, 100 games per
side, τ=0.0), because Bradley-Terry ratings are field-relative and only within-tournament deltas
mean anything.

| rank | model | Elo |
|---:|:---|---:|
| 1 | `E4-03-01@80M` — the anchor | **2179.5** |
| 2 | **`M0@160M` (seed 2)** | **1813.3** |
| 3 | `E4-04-01@80M` — the E4 defaults | 1706.9 |
| 4 | `M0@80M` (seed 2) | 1666.6 |
| 5 | `M0@80M` (seed 1) | 1650.0 |
| 6 | `HeuristicBot` | 1500.0 |

## The three readings the protocol asks for

**Variance at 80M: 16.6 Elo.** Two independent seeds, 1650.0 and 1666.6, playing each other to
49.5% (99W-99L-2D). That is far below the ~95 Elo this plan budgeted from E3, so the ladder
resolves adjacent rungs much more finely than feared.

**Slope 80M → 160M: +146.7 Elo**, measured *within* seed 2 so seed variance cancels out of it.
Head to head, the 160M checkpoint beats its own 80M self 67.0%.

**Against the anchor: −366.2 Elo at 160M**, 9.0% as USSR and 9.0% as US. Symmetric, because it
is being beaten from both seats.

## The result that matters: the default architecture loses to an MLP on compute

`M0@160M` beats `E4-04-01@80M` **64.5%** — and it is not close on cost:

| | steps/s | wall clock |
|:---|---:|---:|
| `M0` | **60,814** | `@160M` = **43.8 min** |
| `E4-04-01` defaults | 14,057 | `@80M` = 94.9 min |
| `E4-03-01@80M` anchor | 11,732 | `@80M` = 113.6 min |

M0 runs **4.3x** the defaults and **5.2x** the anchor. So M0 wins that matchup using **46% of the
GPU time**, and at the defaults' own 80M wall clock M0 could have run **346M steps** instead of
160M.

**The E4 default architecture is worse than a plain MLP per unit of compute.** Not marginally:
beaten 64.5% by an arm that cost less than half as much.

The anchor is untouched by the same argument. At its 80M wall clock M0 could run 415M steps, and
extrapolating the +147 Elo per doubling from 1813 puts that near 1960-2000 — still 180+ Elo short,
and slope extrapolation over three doublings is not something to trust anyway. **The bundle earns
its 5.2x cost; the defaults do not earn their 4.3x.**

## Side asymmetry is a property of this rung

M0 is a competent USSR and a poor US, on both seeds and at both budgets:

| M0 vs | as USSR | as US | gap |
|:---|---:|---:|---:|
| `E4-04-01` defaults (@160M) | **80.0%** | **49.0%** | **+31 pp** |
| `E4-04-01` defaults (@80M, seed 1) | 53.0% | 26.0% | +27 pp |
| `HeuristicBot` (@160M) | 87.0% | 81.0% | +6 pp |
| anchor (@160M) | 9.0% | 9.0% | 0 pp |

It showed in training first: `us_episode_frac` ran 0.11–0.23 across both seeds, against the
anchor's balanced −1.0 pp. Two seeds agreeing makes it the rung's property, not noise. The gap
vanishes against the anchor only because M0 is beaten from both seats there, and narrows against
`HeuristicBot` — it is widest against the opponent M0 is closest to in strength.

**This is why the standing report format is per side.** Pooled, M0@160M against the defaults is
"64.5%", which shows neither the 80% nor the 49%.

## Health

Both arms clean: pool to capacity 12 and held, `kl_div` peaking at 0.043, no alarm fired across
1,237 and 2,474 iterations.

## What it establishes for the ladder

* The floor is **1650–1666 at 80M**, and every rung above must beat it.
* **Structure is worth 455 Elo** measured against the bundle, and **42 Elo** measured against the
  defaults. Naming which arm is meant is mandatory: `E3-09` vs `E3-01` was the MLP against the
  *defaults*, so 42 is the number comparable to E3's 110, and the engine change collapsed it.
* Seed variance is small enough that the remaining rungs can be read at ~20 Elo resolution.
* The compute-parity gate is the decisive rule for this ladder, not a footnote — M0 is 4-5x
  faster than anything it is compared against.
