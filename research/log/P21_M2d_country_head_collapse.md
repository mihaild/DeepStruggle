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
sets `--seed` explicitly and records it in `metadata.json`, so `E4-08-01@80M` is exactly reproducible
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

That is a testable story. `E4-08-01-2` re-runs seed 1 from the same starting conditions — identical
initial weights, action-sampling stream, and deals and dice, though not bitwise identical, since
nothing sets `torch.use_deterministic_algorithms` or `cudnn.deterministic` and GPU reductions are
non-associative. If the 40M event recurs, the collapse is determined by those starting conditions
and can be summoned at will for study. If it does not, the configuration sits near a bifurcation
in float-noise space, and the rate is a property of the configuration rather than of seeds.

## Training is bitwise reproducible, and so is the collapse

`E4-08-01-2` re-ran seed 1 from scratch. The expectation — mine and the owner's — was that it
could not be bitwise identical, since the seed sets `torch.manual_seed`, `np.random.seed` and
`env_base_seed` but **nothing** sets `torch.use_deterministic_algorithms` or
`cudnn.deterministic`, leaving GPU reductions non-associative.

**That expectation was wrong.** Comparing every logged key — `entropy`, `kl_div`, `clip_frac`,
`adv_std_raw`, `loss`, `explained_variance` — across all 672 iterations the two runs share:

```
IDENTICAL on every key across all 672 common iterations   (~44M steps)
```

**and across the complete 80M run:**

```
1221 common iterations — IDENTICAL on every key
E4-08-01@80M  snapshot_final.pt  sha256[:16] = fbfc08d937071163
E4-08-01-2  snapshot_final.pt  sha256[:16] = fbfc08d937071163
```

The two final checkpoints are **byte-identical**, not merely metric-identical. Two independent
80M runs, launched hours apart with tournaments competing for the same GPU, produced the same
3.2M-parameter weight file bit for bit. Plausibly because `cudnn.benchmark` defaults to off, so
kernel selection is fixed, and this architecture is dense matmuls and GELUs with no
atomic-scatter operations.

### The audit, because the claim is surprising

Byte-identity without any determinism flag set deserves more than one hash of one file. Checked:

| check | result |
|:---|:---|
| three **distinct** directories, distinct inodes, mtimes spanning ~4 hours | yes |
| **five** checkpoints compared — 5M, 20M, 40M, 60M, final | all identical, distinct inodes so not hardlinks |
| a file that **should** differ — `metadata.json` | **differs across all three**, so the comparison can detect differences |
| `cudnn.benchmark` / `cudnn.deterministic` / `use_deterministic_algorithms` | **all False** |
| `CUBLAS_WORKSPACE_CONFIG` | unset |
| `matmul.allow_tf32` | **False** |

The `metadata.json` row is the control that matters: the same method reports a difference where
one exists.

**The explanation is mundane.** Those flags do not *create* determinism — they *force* it where a
non-deterministic kernel would otherwise be selected. This model contains none: dense `Linear`,
GELU and LayerNorm, no convolutions (the history branch is disabled), no scatter-add, no atomics.
With `allow_tf32=False` and `benchmark=False` the GEMMs are fixed-algorithm full-FP32. With no
non-deterministic operation in the graph the flags are moot.

**So this is a property of the current configuration and must be re-verified, not assumed.** The
ladder's own M3 and M4 rungs add attention, and some scaled-dot-product backends use `atomicAdd`;
that is the first place determinism could quietly stop holding.

An intermediate check appeared to show divergence at 24M. That was a **bug in the comparison
script**, which matched rows by nearest step within a 2M tolerance and so compared the replica's
22M row against the original's 24M row while the replica had not yet reached 24M. The bitwise
comparison by iteration index is the correct instrument.

### What this makes possible

A reproducible collapse is useful. A *bitwise* reproducible one is much more so, because it
supports genuine counterfactuals rather than correlational readings of logs:

* take the **40M snapshot**, immediately before the arrest;
* branch with exactly one change — reseed only the action sampling, enlarge the opponent pool,
  perturb the weights, freeze one seat's updates, alter `ref_update_freq`;
* every branch shares an identical prefix, so any difference in outcome is attributable to the
  intervention and nothing else.

E3's collapses were observed at 200M+ steps in arms too expensive to re-run, so they could only
ever be studied from logs after the fact. This one costs ~30 minutes to reach, arrives on rails,
and can be branched at any point. It is the first instance in this repository that can be
*experimented on* rather than merely described.


## Corrections to the statistical claims above

Three claims in earlier revisions of this file overstated what five seeds can support.

### "M2d has ~2.5x M2's seed variance" — withdrawn

That compared M2d's four-seed spread (97.5) against M2's **two**-seed spread (27.4). A two-seed
spread is not an estimate of anything. Every two-seed spread drawable from M2d's own four clean
seeds:

| pair | spread |
|:---|---:|
| s5 + s6 | **9.5** |
| s3 + s4 | 24.6 |
| s3 + s6 | 63.4 |
| s3 + s5 | 72.9 |
| s4 + s6 | 88.0 |
| s4 + s5 | **97.5** |

M2's 27.4 is a single draw from a distribution spanning 9.5 to 97.5. Had the two M2d seeds been
s5 and s6, the same reasoning would have concluded M2d has *less* variance than M2. **No variance
comparison between these two configurations is currently possible.**

### "The clean arms spread continuously" — mischaracterised

The four clean ratings are **2056.0, 2065.5 | 2128.9, 2153.5**: two tight pairs, 9.5 and 24.6
apart, separated by 63–98. Reporting this as "sd 47.7" hides that structure. With four points the
apparent bimodality is not establishable either — the honest statement is that the values do not
look like draws from one unimodal distribution, and four seeds cannot tell.

### "A 20% collapse rate" — asserts precision that is absent

One collapse in five seeds. Wilson 95% interval: **[3.6%, 62.4%]**. The rate is essentially
unconstrained; all that is established is that the collapse is neither certain nor vanishingly
rare in this configuration.

## Note on run naming

The scheme is `<engine>-<attempt>-<seed>`, so the trailing number is the **seed**, and a re-run of
the same seed appends a replicate index. The two same-seed re-runs on this page were therefore
misnamed:

| as launched | correct name | what it is |
|:---|:---|:---|
| `E4-08-01-2` | **`E4-08-01-2`** | seed 1, replicate 2 |
| `E4-08-01-3` | **`E4-08-01-3`** | seed 1, replicate 3, dense resume states |

`E4-08-03@80M` … `E4-08-06@80M` are correct: those are seeds 3–6. The directories keep their launched
names because `metadata.json` records `run_name`, and renaming would leave the two inconsistent;
the mapping is recorded here instead and the convention is used from now on.

## Twelve seeds overturn the headline: the country head alone is WORSE

The conclusions above were drawn from four clean seeds. Twelve now exist, rated in one field
(`data/reports/P21_M2d_all_seeds.md`, 18 entrants, 3,400 games each, τ=0.0):

```
clean M2d seeds: n=12
  mean 1998.0   sd 79.0   range 269.4 (1849.7 .. 2119.1)   SEM 22.8

references, same field:
  anchor         2073.2    M2d mean  -75.2
  M2 both heads  2029.6    M2d mean  -31.7
  M1             1736.7    M2d mean +261.3
```

**`pe_country` alone is worse than both heads together, not equivalent to it.** The gap to the
anchor is ~3.3 standard errors.

### The four-seed result was a near-best draw

Seeds 3–6 average **2059.4** in this field; the eight later seeds average **1967.3** — a 92-point
difference. Across all 495 possible four-seed subsets the means span 1916.2 to 2068.0, and
**none** exceeds the anchor's 2073.2. The original four landed at 2059.4, within 9 points of the
most favourable draw available.

So "M2d reaches the anchor" was produced by the luckiest end of the sampling distribution, and
reported as if it were the configuration's value. This is the same error as the withdrawn
variance claim, one level up: **a four-seed mean is not an estimate when the sd is 79.**

### The distribution is not one smooth spread

```
1849.7, 1860.4     low pair
        ↕ 114.9
1975.3 … 2037.4    eight seeds, every internal gap ≤ 21.8
        ↕  50.2
2087.6, 2119.1     high pair
```

The middle eight span 62 points in total, separated from two outlying pairs by 115 and 50. The
owner's suspicion of multimodality, raised when only four points existed, has more support at
twelve — though twelve points still cannot establish it.

### Collapse costs more than the mechanism is worth

The two collapsed seeds rate **1719.9** and **1683.4**, both *below* M1's 1736.7. A collapsed arm
does not merely forfeit the per-entity head's gain; it ends up worse than the rung beneath it.

### Consequence for the ladder

**M2 keeps both heads.** Dropping `pe_card` is worse on average, more variable, and carries
collapse risk. The earlier reading — "a strict simplification: equal strength, 18% faster, better
balanced" — is withdrawn in full.

It remains true that `pe_card` alone buys nothing (M2e ≈ M1, uncollapsed). Both heads are needed
together, which is the interaction hypothesis that an earlier revision proposed from a collapsed
arm, withdrew as unsupported, and which the twelve-seed evidence now supports on proper grounds.
