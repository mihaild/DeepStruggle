# Country identity without graph convolution: a per-country bias that disambiguates 54 countries

**2026-09-19.** Asked of the configuration `identity_dim=16, per_entity_heads=64,
graph_layers=0, self_transform` — the one every late E3 arm used and the one `E4-03-01` is queued
to train. Measured on `E3-30-28`, which ran exactly that configuration and kept `snapshot_0s.pt`,
so trained embeddings compare against their own initialisation rather than an assumed scale.

**Lineage note.** Magnitudes below are E3's. They are reported because the mechanism question —
does this pathway carry weight at all — is about architecture code shared by both lineages, and
the configuration is identical. The E4 answer comes from `E4-03-01`.

## Identity enters at three points, doing a different job in each

With `graph_layers=0` the board tensor `(B, 84, 26)` is concatenated with the identity rows and
passed through a **shared** per-node `Linear` (`board_fc`). The resulting `h_board` is then used
three ways.

**1. The pooled path — position is destroyed.** `board_mean` and `board_max` over the 84 nodes
feed `board_proj` and the trunk. **This pooling happens regardless of `graph_layers`**; it is not
a graph-convolution feature. Mean and max are symmetric, so the trunk sees a *bag* of country
tokens, and identity is the only thing that lets country-specific information survive it.

**2. `pe_country` — a per-country bias, and the shared bias cannot absorb it.** Country *i*'s
logit is computed from country *i*'s token by an MLP. The index already does the addressing, so
identity enters as a constant added to the pre-activation of a `Linear`: a learned per-country
bias, exactly as it appears.

The reason that is not redundant is **weight sharing**. `pe_country[0]` is one `Linear` applied to
all 84 countries, so its bias term is a *single* vector for every country. A neuron cannot "just
learn the constant", because it has only one constant to learn and 84 countries to spend it on.
Identity is precisely the mechanism that turns that one shared bias into 84 different ones,
`bias_eff(i) = b + W_id · id_i` — a rank-≤16 lookup table over countries.

Measured on `E3-30-28`:

| quantity | value |
|:---|---:|
| shared bias, ‖b‖ | 1.69 |
| per-country offset ‖W_id · id_i‖, mean over countries | **5.15** |
| across-country std of that offset, per unit | 0.620 |
| ... relative to the shared bias's per-unit scale | **2.94x** |
| ... relative to the **data-dependent** term's across-country std | **0.80x** |
| pairwise distance between two countries' offsets, mean | 7.21 |
| Egypt–Iran / Egypt–Libya / Iran–Libya | 6.32 / 7.38 / 7.85 |

The per-country offset is three times the shared bias in magnitude, and it varies across countries
**four fifths as much as the entire state-dependent input does**. Egypt, Iran and Libya — identical
on every static slot — receive offsets 6–8 apart. This is the disambiguation, and no reweighting of
the shared bias could produce it.

**3. The card→country cross-attention — identity is in the keys.** `cross_attn(h_cards, h_board,
h_board)` has each of the 110 cards attend over the 84 country tokens, and `h_board` carries
identity. Two countries with identical state features have keys that differ *only* by identity, so
it is the only thing that lets a card single one of them out.

It is not decorative. Baseline attention is sharply peaked — mean max weight 0.244 against 0.0119
for uniform, a factor of 20 — and ablating identity moves the attention mass wholesale:

| ablation | total-variation shift of the attention map (mean / median / p95) |
|:---|---:|
| `country_identity` → 0 | 0.558 / 0.636 / 0.772 |
| `country_identity` → random, same norms | 0.685 / 0.788 / 0.902 |

**More than half the attention mass lands on different countries.**

### A naming trap

`per_entity_heads` is **not** attention despite the name — it is the width of a per-country
residual MLP. The genuinely attentional read-out is `attn_readout`, and it is **0** in these arms,
so `ro_country_kv` is never even constructed. The attention identity actually participates in is
`cross_attn`, which is always present and is governed by `num_attn_heads` (default 4), a flag no
arm has varied.

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
architecture proves worth carrying at all ([P19](../../archive/E4_ladder/plans/P19_architecture_ab.md)).

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
