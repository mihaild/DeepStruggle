# Country identity without graph convolution: a per-country bias that disambiguates 54 countries

**2026-09-19.** Asked of the configuration `identity_dim=16, per_entity_heads=64,
graph_layers=0, self_transform` — the one every late E3 arm used and the one `E4-03-01` is queued
to train. Measured on `E3-30-28`, which ran exactly that configuration and kept `snapshot_0s.pt`,
so trained embeddings compare against their own initialisation rather than an assumed scale.

**Lineage note.** Magnitudes below are E3's. They are reported because the mechanism question —
does this pathway carry weight at all — is about architecture code shared by both lineages, and
the configuration is identical. The E4 answer comes from `E4-03-01`.

## There are two country paths, and identity does a different job in each

With `graph_layers=0` the board tensor `(B, 84, 26)` goes through a **shared** per-node `Linear`
(`board_fc`), and the result is used twice:

**1. The pooled path — position is destroyed.** `board_mean` and `board_max` are taken over the
84 nodes and fed to `board_proj` and thence the trunk. **This pooling happens regardless of
`graph_layers`**; it is not a graph-convolution feature. Mean and max are symmetric, so the
trunk's view of the board is a *bag* of country tokens. Two countries whose slots happen to match
are interchangeable here, and the trunk cannot tell which of them contributed what.

**2. The per-entity path — position is preserved.** `pe_country` computes country *i*'s logit from
country *i*'s token, indexed. Nothing is pooled. Here identity needs to supply no addressing at
all, because the index already does that: it enters as a **constant per-country vector added to
the pre-activation of a `Linear`**, i.e. exactly a learned per-country bias — one that shifts a
GELU rather than merely adding to the output, but a static bias nonetheless.

So "it is just a static bias" is the right reading for `pe_country`, and the wrong one for the
pooled trunk path, where symmetric pooling makes identity the only way country-specific
information survives at all.

## The static slots do not identify a country

That bias is not cosmetic, because the observation is far from identifying which country a slot
is. The country-constant slots in `observation.cpp` are stability (3), battleground (4),
superpower adjacency (8, 9), the 6-way region one-hot (10–15) and three sub-region flags
(16–18). Everything else in the 26 is state-dependent.

| | count |
|:---|---:|
| countries | 84 |
| **distinguishable by their static slots alone** | **30** |
| **indistinguishable from at least one other country** | **54** |
| distinct static signatures | 46 |

The collisions are not exotic. **Egypt, Iran and Libya are identical** on every static slot; so are
Bulgaria, Czechoslovakia, Hungary and Yugoslavia; so are six African countries at stability 2, and
six more at stability 1, and five Central American, and five South American.

Without identity, nothing distinguishes Iran from Libya except their *current* influence and
control. The network cannot hold a different standing prior for them.

**And adjacency is absent entirely.** All 84 countries have neighbours in `map.json`, and none of
that reaches the per-country slots — only `superpower_adjacent` does. With `graph_layers=0` the
network has no access to the map's edges by any route, so identity is the only channel through
which anything topological could be learned at all.

## It is heavily used

Ablation on the trained network, 2,560 real observations from engine self-play. `pe_country` is
read through a forward hook, so every weight feeding the measurement is trained — the E3/E4 action
repack touches only the policy head's final layer, which is dropped and never enters it.

| ablation | Δ `pe_country` (mean abs) | as a fraction of the across-country spread | Δ value (mean abs) |
|:---|---:|---:|---:|
| `country_identity` → 0 | 4.43 | **1.10x** | 0.232 |
| `country_identity` → random, same row norms | 4.65 | **1.15x** | 0.325 |
| `card_identity` → 0 *(scale reference)* | 1.85 | 0.46x | 0.118 |

The across-country spread of `pe_country` is 4.03 and the value head's own standard deviation is
0.404. Removing country identity perturbs the per-country logits by **more than the entire spread
between countries** — the ranking is rearranged, not nudged — and moves the value estimate by
**57% of its own standard deviation**. The same ablation on the card pathway is less than half as
large.

Norm-matched random rows are **as damaging as deletion**, so the network depends on the particular
values it learned, not merely on having a tie-breaker present.

The embeddings do train: mean movement from initialisation is 32% of the initial row norm, and
rows stay near-orthogonal (mean pairwise cosine +0.011), so they neither froze nor collapsed.

## But they encode no map

A linear probe of the 16 dimensions against country properties. Sixteen dimensions over 84 points
fit a lot by chance, so the baseline is a norm-matched random embedding, never zero:

| embedding | battleground R² | stability R² | region R² |
|:---|---:|---:|---:|
| trained | 0.263 | 0.355 | 0.193 |
| at initialisation | 0.150 | 0.236 | 0.172 |
| random, norm-matched | **0.306** | 0.284 | **0.216** |

**The trained embedding predicts battleground status and region no better than random rows do.**
Same-region pairs are slightly more aligned than different-region pairs (+0.0097 cosine, against
−0.0059 at initialisation), but the effect is tiny.

Consistent with the rest: the observation already carries region, stability and battleground per
country, so there is nothing for identity to gain by re-encoding them. What it can add is a
standing per-country adjustment for the 54 countries the static slots fail to separate — and that
is a lookup table, not a map. It is also why random rows hurt as much as zeros: any consistent
labelling would serve, but the trained weights were organised around these particular values.

## What this does and does not establish

**Does:** in the no-graph-convolution configuration the country identity pathway carries real
weight. It is not vestigial, and a run in that configuration is not equivalent to one without it.

**Does not:** an ablation drives the network out of distribution, so a large output change proves
*dependence*, not *usefulness*. A network trained without identity might reach the same strength
by other means.

**`E4-03-01` vs `E4-04-01` will not settle it.** Those arms differ in **four** flags at once —
`identity_dim`, `per_entity_heads`, `graph_layers`, `self_transform` — so no strength difference
can be attributed to identity. Isolating it needs one further arm: `graph_layers=0` with
`identity_dim=0` against the E4-03 configuration, and it is worth one 80M arm only if the
architecture proves worth carrying at all ([P19](../../plans/P19_architecture_ab.md)).

## Identity has been trained together with a graph — and graph depth is still open

An earlier version of this page claimed no arm had ever paired `identity_dim > 0` with
`graph_layers >= 1`. **That was wrong.** The trained combinations are:

| `identity_dim` | `graph_layers` | `per_entity_heads` | arms |
|---:|---:|---:|---:|
| 16 | **2** | 64 | 2 — `E3-15-22` |
| 16 | **1** | 64 | 1 — `E3-16-21` |
| 16 | 0 | 64 | 39 — `E3-17` onward |
| 16 | *(pre-flag)* | 64 / none | 7 — `E3-12` … `E3-15-21` |
| 0 | 2 | 0 | 6 — `E3-33-30`, all E4 arms |

So graph depth *was* ablated with identity held on, and `graph_layers=0` was the outcome. But the
outcome does not stand: the archived finding records **graph depth as open, not settled at 0** —
"the comparison that removed the map graph reverses sign with the seed", and `E3-15` rests on one
seed — while `--graph-layers 0` became the backbone of every arm from `E3-17` onward regardless
([`../../archive/E3_ladder/findings/architecture.md`](../../archive/E3_ladder/findings/architecture.md),
[`../../archive/E3_ladder/findings/seed_variance.md`](../../archive/E3_ladder/findings/seed_variance.md)).

The E4 default inverted the pairing — `graph_layers=2` with no identity — without that being an
experiment. So the cell that has *never* been trained is not "identity with a graph" but
**`identity_dim=0` with `graph_layers=0`**: nothing distinguishing the 54 colliding countries and
no edges either.
