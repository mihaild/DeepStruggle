# P21 — build the architecture up from an MLP, one mechanism at a time

**Status: proposed, not launched.** Requested by the owner, 2026-09-19: start from the simplest
architecture, add features one by one, measure each against the previous rung and against a fixed
baseline, and leave the dubious pooling until last.

## Why this and not more one-off arms

Every architectural choice in `ColdWarNetV2` was added on top of the ones already there, and
several were measured only against whatever the baseline happened to be at the time. The result
is a network nobody can currently justify component by component: `attn_readout` is dead code in
every arm, `num_attn_heads` has never been varied, `per_entity_heads` exists to repair a pooling
loss, and `country_identity` exists partly to repair the same loss
([`../findings/training/forward_pass_trace.md`](../findings/training/forward_pass_trace.md)).

Building up instead of down gives every mechanism a matched control by construction, and answers
a question no existing arm does: **which of these things is actually paying for itself?**

**Governing principle: the simpler architecture wins ties.** A rung is adopted only if it beats
the rung below it by more than seed spread. Otherwise the simpler variant carries forward. This is
the opposite of the current default, where complexity accumulated because nothing removed it.

## The fixed anchor

Every rung is rated against **`E4-04-01@80M`** — cold, pooled, default architecture, completed
2026-09-19 clean ([`../method/detecting_collapse.md`](../method/detecting_collapse.md)) — as well
as against the rung below it. The anchor gives absolute placement; the rung-to-rung comparison
gives the increment.

Re-point the anchor to `E4-03-01@80M` if [P19](P19_architecture_ab.md) finds the late-E3 bundle
stronger.

## The input

3,824 floats: board `84 x 26` = 2,184, cards `110 x 14` = 1,540, global 100. History is off
(`use_history=False`) and is not part of this.

## The ladder

Each rung is the one below it plus exactly one mechanism. **Rungs M0–M4 keep position: entity
tokens are flattened, never pooled.** Pooling enters only at M7.

| rung | what it adds | what it isolates |
|:---|:---|:---|
| **M0** | flat MLP: `obs(3824) → Linear → GELU → residual blocks → heads` | the floor. Everything else must beat this |
| **M1** | grouped input projections — separate `Linear` for board / card / global, concatenated | does splitting the input by semantic block help at all? |
| **M2** | shared per-entity encoder, then **flatten** — one `Linear(26→d)` applied to all 84 countries, `Linear(14→d)` to all 110 cards, then flatten | does weight sharing across entities help, **with position held constant**? |
| **M3** | graph convolution over `norm_adj`, still flattened | does the map's adjacency help? **This is the clean graph test** — E3 only ever varied graph depth *with* pooling, and withdrew the answer anyway |
| **M3b** | self-transform on the graph layer | E3 measured +53/+83 Elo for this, but with pooling |
| **M4** | card→country cross-attention | does attention between the two entity sets help? |
| **M7** | **replace flatten with mean+max pooling** | **what pooling costs**, with everything else held at the best configuration found |

M0–M4 are a strict ladder. M7 is deliberately last, and is a *removal* — it takes away positional
information that every rung below it had.

## Two mechanisms that are NOT ladder rungs

`per_entity_heads` and `country_identity` **exist to repair the pooling loss**. Testing them
before pooling asks a different question than testing them after, so putting them on the ladder
would answer neither. They get a 2x2 against the pooling axis instead:

| | flattened (positional) | pooled |
|:---|:---|:---|
| no per-entity heads, no identity | M4 | M7 |
| + per-entity heads | M5f | M5p |
| + country identity | M6f | M6p |

The interesting prediction, and the reason this is worth running: **with position preserved, both
should be redundant.** `pe_country` routes a country's own token to its own logit because the
pooled trunk cannot — but a flattened trunk already can. `country_identity` supplies a per-country
bias and addressing that position supplies for free
([`country_identity_without_graph_conv.md`](../findings/training/country_identity_without_graph_conv.md)).
If M5f ≈ M4 and M6f ≈ M4 while M5p > M7 and M6p > M7, then both are confirmed as pooling repairs
and the architecture simplifies substantially.

## Parameter matching is mandatory

[`P6`](P6_attention_backbone.md) records that capacity is not v2's bottleneck, so an uncontrolled
parameter increase would make "bigger won" look like "the mechanism won". **Every arm is matched
to 3.1M ± 5%** by adjusting the trunk width or the per-entity width `d`, and the realised count
goes in the run's `metadata.json` description.

Indicative costs at `d = 16`:

| component | params |
|:---|---:|
| flat `Linear(3824, 512)` (M0) | 1.96M |
| grouped: board 2184→256, card 1540→256, global 100→128 (M1) | 0.97M |
| shared encoder then flatten: 84x16→256 plus 110x16→256 (M2) | 0.79M |
| current pooled board path, for reference | 0.04M |

Pooling is *cheap*; that is the trade being measured. A positional path costs real parameters and
the question is whether it buys more than they would buy elsewhere — hence the matching.

## Measurement

Per [P19](P19_architecture_ab.md)'s protocol, which is already pre-registered:

* **Head-to-head, `tools/tournament.py`, 200 games per pair — 100 per seat**, reported per side.
* Sanity gates before any headline is read: the arm beats `heuristic` and `heuristic_mcts`
  decisively, and is silent under the health alarms (`NOPOOL` / `POOLSTUCK` / `KLSPIKE`).
* **Two seeds per rung.** Seed spread here is ~95 Elo, larger than most architecture effects in
  this record, and two architecture conclusions have already been withdrawn for exactly that
  reason. One seed screens; it does not decide.

**Adoption rule, fixed in advance.** A rung advances only if it beats the rung below on **both**
seeds and the pooled margin exceeds the two arms' own seed spread. A rung that ties is **not**
adopted — the simpler variant carries forward, and the tie is recorded as "no detected effect at
80M, 2 seeds", never as "no effect".

## Budget

`E4-04-01` ran 80M in ~100 minutes, so an arm is ~1.7 GPU-hours.

| stage | arms | GPU-hours |
|:---|---:|---:|
| screen M0–M4, M7 at 40M, one seed | 6 | ~5 |
| confirm survivors at 80M, two seeds | ~12 | ~20 |
| the 2x2 (M5f/M5p/M6f/M6p) at 80M, two seeds | 8 | ~14 |
| **total** | **~26** | **~39** |

Affordable, and the screening stage can drop rungs before the expensive confirmation.

## What has to be built first

There is no model that can express M0–M2; `ColdWarNetV2` hard-codes pooling and the token path.
This needs a configurable backbone with **explicit, non-defaulted** options for the encoder
(`dense | shared | graph`) and the aggregation (`flatten | pool`).

**No silent defaults.** A defaulted `layout` parameter was the mechanism behind five instances of
one bug in this repo, because handing a model the wrong variant returns a number instead of
raising. Every option is required at construction, the realised configuration is written into the
checkpoint, and loading asserts it — the same discipline `check_obs_width` applies to width.

## Ordering and risk

Behind [P19](P19_architecture_ab.md), which is running. Note the dependency: if P19 finds the
late-E3 bundle stronger, the anchor moves and M3/M3b become more interesting, since that bundle is
`graph_layers=0` plus identity plus per-entity heads — three of the things this ladder is built to
separate.

The main risk is that everything ties. At 80M on two seeds, against ~95 Elo of spread, the ladder
may simply lack the resolution to separate adjacent rungs. If the first two rungs tie, raise the
budget or the seed count before continuing rather than reading ties as answers — a ladder of
inconclusive steps is worse than no ladder, because it looks like a result.
