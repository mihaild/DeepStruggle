# P21 — the country head is the whole mechanism, and removing the card head sometimes collapses

**2026-09-19.** M2 added per-entity heads for both countries and cards and was worth +315 Elo over
M1. M2d keeps only `pe_country`, M2e only `pe_card`; everything else is identical.

## The decomposition

Ratings from `data/reports/P21_M2d_uncollapsed.md`, one tournament, 100 games per side, tau=0.0.

| arm | Elo | steps/s | USSR all | US all | gap | note |
|:---|---:|---:|---:|---:|---:|:---|
| `E4-03-01@80M` — anchor | 2125.6 | 11,732 | 78.9% | 83.4% | −4.6 | |
| **`M2d@80M` seed 3** | **2111.5** | 42,268 | 79.9% | 79.6% | **+0.3** | country head only |
| `M2@80M` — both heads | 2062.7 | 35,744 | 77.1% | 72.7% | +4.4 | |
| `M1@80M` | 1770.9 | 59,561 | 42.1% | 39.7% | +2.4 | |
| `M2d@80M` seed 1 | 1756.1 | 42,059 | 58.9% | 19.4% | +39.4 | **collapsed** |
| `M2e@80M` — card head only | 1750.2 | 46,313 | 48.0% | 28.9% | +19.1 | |

| configuration | verdict |
|:---|:---|
| `pe_country` alone | **the whole mechanism** — 2111.5, level with M2's 2062.7 |
| `pe_card` alone | **nothing** — 1750.2 against M1's 1770.9, and it did *not* collapse |
| both together | no better than country alone |

The +48.8 by which M2d exceeds M2 is inside combined seed spread (~27 each), so the claim is
**equivalent**, not better. What is established is that removing `pe_card` entirely costs nothing.

## The prediction was right, and was wrongly withdrawn

[`../findings/training/forward_pass_trace.md`](../findings/training/forward_pass_trace.md)
predicted `pe_country` would carry the gain and `pe_card` would be near-inert, because `pe_card`
cannot distinguish 95 of 110 cards without identity while a country's dynamic slots — influence,
control, both deficits, can-place, can-coup, realignment modifier — separate countries in any real
position. **That is what the uncollapsed evidence shows.**

Two intermediate readings were wrong, and are recorded so the reasoning is traceable:

1. From M2d seed 1 alone: *"the country head buys nothing; the prediction is refuted; type-level
   card information must be what the mechanism needs."* That explained a **collapsed** arm's
   rating, which measures the collapse rather than the mechanism.
2. From M2d seed 1 plus M2e: *"neither head alone works, so the effect is superadditive."* Same
   defect — one of the two arms was degenerate.

The lesson is narrow and worth keeping: **a collapsed arm's Elo measures nothing but the
collapse.** Both errors came from using one as evidence.

A third hypothesis died separately: that a partial per-entity correction distorts the choice
*between* playing a card and placing influence. Measured over 25,600 decision points, the legal
mask offers the card block alone 25.1% of the time, the country block alone 52.9%, neither 22.0%,
and **both 0.00%** — never once. The blocks are never in competition, so the distortion is
impossible, not merely unsupported.

## Dropping the card head is a strict simplification — except for stability

| | M2 (both) | M2d (country only) |
|:---|---:|---:|
| Elo | 2062.7 | **2111.5** |
| steps/s | 35,744 | **42,268** (+18%) |
| side gap | +4.4 pp | **+0.3 pp**, the best in the ladder |
| collapse rate | 0 of 2 seeds | **1 of 3 seeds so far** |

M2d seed 3 is more side-balanced than the anchor itself. The only cost is instability.

## The collapse

Seed 1 abandoned the US seat: `us_episode_frac` fell 0.68 to 0.0000, finishing at 0.035, with
**42%** of its rows below 0.02 against seed 3's **1%**. The two diverge cleanly after ~40M:

| steps | seed 1 `us_frac` | seed 1 `adv_std` | seed 3 `us_frac` | seed 3 `adv_std` |
|---:|---:|---:|---:|---:|
| 30M | 0.057 | 0.173 | 0.217 | 0.255 |
| 40M | 0.172 | 0.202 | 0.121 | 0.211 |
| 45M | **0.008** | **0.068** | 0.138 | 0.258 |
| 75M | **0.000** | **0.036** | — | — |

Advantage variance collapses as `explained_variance` reaches 0.999 — the critic looking *excellent*
precisely as the policy degenerates, because a one-sided policy is a trivially easy prediction
problem. The rejected `adv_std_raw < 0.01` threshold still never fired (minimum 0.036), but the
direction now co-occurs with one-sidedness in a second independent arm.

**Rate so far: 1 of 3 seeds.** Seeds 4-6 continue; seed 4 was clean at 56%.

## Why this configuration is worth keeping

E3's collapses appeared at 200M+ steps in arms too expensive to re-run, so they were studied from
logs rather than by experiment. **This one reproduces at 80M in about 30 minutes on a
3.2M-parameter network**, at a rate high enough to catch within a handful of attempts. Every arm
sets `--seed` explicitly and records it in `metadata.json`, so `E4-08-01` is exactly reproducible
and `tools/scripts/launch_flags.py` reconstructs its command.

That makes country-head-only the repository's first *controllable* instance of the failure mode —
more useful for studying the mechanism than any of the E3 observations, and independent of whether
the configuration is ever adopted.

## The collapse is a learning-arrest, and it is a bifurcation

Rating the collapsing seed's 5M snapshots against a clean seed's, at matched steps, in one
tournament (`data/reports/P21_M2d_trajectory.md`):

| steps | seed 1 (collapses) | seed 3 (clean) | Δ |
|---:|---:|---:|---:|
| 20M | **1752.4** | 1727.2 | seed 1 *ahead* |
| 40M | **1820.6** | 1811.4 | seed 1 *ahead* |
| 60M | 1802.4 | **2004.9** | −202.5 |
| 80M | 1792.7 | **2167.8** | −375.1 |

Per 20M interval:

```
seed 1:   +68.2    −18.2     −9.7      flat after 40M
seed 3:   +84.2   +193.5   +162.9      accelerates after 40M
```

**Seed 1 does not get worse. It stops improving.** From 40M onward it sits at 1821 → 1802 → 1793,
flat within noise, while seed 3 climbs 356 Elo over the same span. Before 40M the two are
indistinguishable and seed 1 is fractionally *ahead* at both 20M and 40M.

That timing coincides with the side-balance divergence, which also separates after ~40M. Strength
and balance break together, so neither is a lagging symptom of the other.

### This settles bifurcation versus gradient

Two trajectories identical for 40M, after which one enters an absorbing state and the other keeps
climbing. There is no gradual separation. Together with the bimodal pinned-row distribution —
41.5% for the collapsed seed against ≤0.5% for every other arm in the ladder, with nothing in
between — this is evidence **against** the hypothesis that the same mechanism quietly degrades
results without collapsing them. Clean arms appear genuinely clean rather than partially damaged.

Two caveats keep it from being settled: the ladder has only seven arms, and sub-collapse damage
could express itself as something other than side imbalance — a shallower slope, or a capability
gap this field of opponents cannot resolve.

### What it points at

The question is not "what makes training worse" but **"what happens around 40M that stops learning
entirely"**. A frozen rather than degrading policy suggests a self-play equilibrium it cannot
leave: if USSR wins essentially every game, the US seat generates no useful gradient, and the
opponent pool fills with snapshots that confirm the imbalance rather than punishing it.

That is a testable story. `E4-08-07` re-runs seed 1 from the same starting conditions — identical
initial weights, action-sampling stream, and deals and dice, though not bitwise identical, since
nothing sets `torch.use_deterministic_algorithms` or `cudnn.deterministic` and GPU reductions are
non-associative. If the 40M event recurs, the collapse is determined by those starting conditions
and can be summoned at will for study. If it does not, the configuration sits near a bifurcation
in float-noise space, and the rate is a property of the configuration rather than of seeds.
