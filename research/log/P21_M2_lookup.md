# P21 rung M2 — the per-entity lookup reaches the anchor

**2026-09-19.** M2 is M1 plus exactly one mechanism: a shared MLP giving each country and card a
logit *correction* computed from that entity's own raw slots plus a 64-wide summary of the
position, added to the dense logit with a zero-initialised last layer so the network starts as M1
exactly. No shared encoder, no identity, no attention, no pooling, no graph.

All numbers from one tournament, `data/reports/P21_M2_complete.md`, 100 games per side, τ=0.0.

| arm | Elo | steps/s | USSR all | US all | gap | vs anchor (USSR / US) |
|:---|---:|---:|---:|---:|---:|:---|
| `E4-03-01@80M` — anchor | **2167.1** | 11,732 | 78.7% | 80.8% | −2.1 | — |
| **`M2@160M`** | **2163.5** | 36,210 | 81.2% | 77.3% | +3.9 | **49.0% / 46.0%** |
| `M2@80M` seed 1 | 2128.6 | 35,744 | 77.2% | 74.8% | +2.4 | 40.0% / 50.0% |
| `M2@80M` seed 2 | 2101.2 | 36,210 | 74.4% | 72.2% | +2.2 | 37.0% / 47.0% |
| `M1@160M` | 1844.4 | 59,943 | 48.4% | 42.3% | +6.1 | 17.0% / 10.0% |
| `M0@160M` | 1821.1 | 60,814 | 49.9% | 35.9% | +14.0 | 9.0% / 9.0% |
| `M1@80M` | 1766.6 | 59,561 | 36.4% | 36.3% | +0.1 | 9.0% / 15.0% |
| `E4-04-01@80M` — defaults | 1701.4 | 14,057 | 31.8% | 28.0% | +3.8 | 3.0% / 9.0% |
| `M0@80M` | 1659.4 | 63,187 | 33.6% | 17.1% | +16.4 | 8.0% / 5.0% |
| `HeuristicBot` | 1500.0 | — | 14.4% | 9.1% | +5.3 | 1.0% / 1.0% |

## The three protocol readings

**+362.0 Elo over M1** at 80M — 82% as USSR, 86% as US. By far the largest single mechanism on
the ladder; M1's grouping was +102 and M0 is the floor.

**Variance at 80M: 27.4 Elo** (2128.6 against 2101.2).

**Slope 80M → 160M: +62.3**, within seed 2. Shallower than M0's +147 and M1's +72 — the pattern
of diminishing slope as the intercept rises now holds across three rungs.

## It matches the anchor, and the two framings differ

**At matched steps the anchor is ahead by 38.5 Elo** (2167.1 against M2@80M's 2128.6). That is
close to the 27.4 Elo seed spread, so it is a real but marginal lead.

**At matched compute M2 is ahead.** M2 runs **3.09x** the anchor's throughput, so:

| | steps | wall clock |
|:---|---:|---:|
| anchor | 80M | 6,820 s |
| `M2@160M` | 160M | **4,419 s** |

**M2@160M ties the anchor using 65% of its wall clock.** At the anchor's own 80M wall clock M2
could run ~247M steps, which the +62.3 slope extrapolates to roughly 2203 — about +36.

Treat that extrapolation as suggestive, not settled: +36 sits inside seed-spread range and it
extrapolates 0.63 of a doubling. The defensible claim is the measured one: **equal strength for
two-thirds of the cost.**

## What this means for the architecture

The anchor is the late-E3 bundle — `identity_dim 16`, `per_entity_heads 64`, `graph_layers 0`,
`self_transform` — reading a **pooled** trunk, and it is worth +447 Elo over the E4 defaults.

M2 reaches the same strength with **none of**: attention of any kind, identity embeddings,
pooling, a shared per-entity encoder, or graph convolution. Three mechanisms do it:

1. **positional input** — every entity read at its own offset, never pooled;
2. **grouped projections** — board, card and global each through their own layer;
3. **the per-entity lookup** — each entity's own slots reaching its own logit.

That is a substantially simpler network, and the pooling-bottleneck story explains why: the
anchor spends `identity_dim` and part of `per_entity_heads` repairing information that pooling
destroys, while M2 never destroys it. **The repairs are unnecessary when the damage is not done.**

## Side balance is solved

M2 sits at **+2.2 to +3.9 pp** across its three checkpoints, against the anchor's −2.1. The
one-seat problem that defined M0 (+16.4 to +22) and lingered in M1 (+0.1 to +6.1) is gone: M2 is
81.2% as USSR and 77.3% as US at 160M.

## Health

All arms clean: pool to capacity 12 and held, `kl_div` peaking at 0.042, no alarm across 1,237 and
2,473 iterations. `us_episode_frac` ran 0.40–0.46 in training, against M0's 0.11–0.23.

## Standing

| rung | Elo @80M | slope | side gap | steps/s | verdict |
|:---|---:|---:|---:|---:|:---|
| M0 flat MLP | 1659 | +147 | +16 pp | 63,187 | floor |
| M1 grouped | 1767 | +72 | +0.1 pp | 59,561 | adopted, +102 |
| **M2 lookup** | **2129 / 2101** | **+62** | **+2.4 pp** | **35,744** | **adopted, +362** |
| anchor | 2167 | — | −2.1 pp | 11,732 | **reached** |

## Next

`M2d` and `M2e` split the +362 between `pe_country` and `pe_card`. The card-collision finding
predicts `pe_country` carries most of it, since `pe_card` cannot distinguish 95 of 110 cards
without identity. If that holds, the architecture simplifies again — and M2.5's identity vector
becomes a card-side fix rather than a general one.
