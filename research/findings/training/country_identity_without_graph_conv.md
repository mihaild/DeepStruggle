# Country identity without graph convolution: load-bearing, but not a map

**2026-09-19.** Asked of the configuration `identity_dim=16, per_entity_heads=64,
graph_layers=0, self_transform` — the one every late E3 arm used and the one `E4-03-01` is queued
to train. Measured on `E3-30-28`, which ran exactly that configuration and kept `snapshot_0s.pt`,
so trained embeddings can be compared against their own initialisation rather than an assumed
scale.

**Lineage note.** The magnitudes below are E3's. They are reported because the *mechanism* question
— does this pathway carry weight at all — is about architecture code shared by both lineages, and
the configuration is identical. The E4 answer comes from `E4-03-01`; see the caveat at the end.

## Why the question is sharp

With `graph_layers=0` the board path is a **shared** per-node `Linear` followed by mean/max
pooling, and `pe_country` — which emits one logit per country — is likewise a **shared** MLP
applied per country. The whole country pathway is therefore permutation-equivariant: two countries
that differ only in *which country they are* receive identical treatment.

Exactly three things can break that symmetry:

1. the raw per-country observation slots (influence, control, battleground flag, region, stability);
2. adjacency, via `norm_adj` — but that is used only when `graph_layers >= 1`;
3. the country identity embedding.

So with no graph convolution, identity is one of only two sources of country-specific structure.

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
0.404. So removing country identity perturbs the per-country logits by **more than the entire
spread between countries** — the ranking is rearranged, not nudged — and moves the value estimate
by **57% of its own standard deviation**. The same ablation on the card pathway is less than half
as large.

Replacing the embedding with random rows of identical norm is **as damaging as deleting it**. The
network is not merely using the presence of a tie-breaking signal; it depends on the particular
values it learned.

The embeddings do train: mean movement from initialisation is 32% of the initial row norm, and
rows stay near-orthogonal (mean pairwise cosine +0.011), so they neither froze nor collapsed.

## But it is not encoding the map

A linear probe of the 16 dimensions against country properties, with the honest baselines — 16
dimensions over 84 points fit a lot by chance, so the comparison is against a norm-matched random
embedding, never against zero:

| embedding | battleground R² | stability R² | region R² |
|:---|---:|---:|---:|
| trained | 0.263 | 0.355 | 0.193 |
| at initialisation | 0.150 | 0.236 | 0.172 |
| random, norm-matched | **0.306** | 0.284 | **0.216** |

**The trained embedding predicts battleground status and region no better than random rows do.**
Same-region pairs are slightly more aligned than different-region pairs (+0.0097 cosine, against
−0.0059 at initialisation), but the effect is tiny.

That is coherent rather than contradictory. The observation *already* carries battleground flags,
region membership and stability per country, so identity has no reason to re-encode them. What it
supplies is **addressability** — a stable, arbitrary handle the rest of the network can hang
country-specific adjustments on. That also explains why norm-matched random rows are as damaging
as zeros: any consistent labelling would have served as an address, but the specific learned
values are what the trained weights were organised around.

## What this does and does not establish

**Does:** in the no-graph-convolution configuration, the country identity pathway carries real
weight. It is not vestigial, and a run in that configuration is not equivalent to one without it.

**Does not:** an ablation drives the network out of distribution, so a large output change proves
*dependence*, not *usefulness*. A network trained without identity might reach the same strength by
other means — it would simply lose the addressing mechanism and have to distinguish countries from
raw features alone.

**`E4-03-01` vs `E4-04-01` will not settle it.** Those two arms differ in **four** flags at once —
`identity_dim`, `per_entity_heads`, `graph_layers` and `self_transform` — so a strength difference
cannot be attributed to identity. Isolating it needs one further arm: `graph_layers=0` with
`identity_dim=0`, against the E4-03 configuration. Worth one 80M arm only if the architecture
proves worth carrying at all, which is what [P19](../../plans/P19_architecture_ab.md) asks first.

**Untested: whether graph convolution makes identity redundant.** No arm has trained
`identity_dim > 0` together with `graph_layers >= 1`; every E3 arm paired identity with
`graph_layers=0`, and the E4 default pairs `graph_layers=2` with no identity at all. The two have
only ever been varied together, so the natural hypothesis — that adjacency supplies the country
structure identity otherwise has to provide — has never been put to a test.
