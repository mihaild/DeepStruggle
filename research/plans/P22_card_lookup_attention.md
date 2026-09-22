# P22 — identity-keyed card lookup

**Status: proposed, not launched.** Requested by the owner 2026-09-22, after P21 closed the whole
M2 family without beating M2d.
**Gate:** the width probe (`E4-23-*`) reports first — see *Ordering*.
**Needs approval:** no engine or observation change. One contained refactor of
`ai/models/ladder_net.py` (`input_mode` → `board_mode` / `card_mode`).

## The capability this is for

Two decisions the owner named, both of which the ladder has never targeted:

> *"I have European Scoring, Europe is negative, I am USSR and hold Five Years Plan → hold both
> until AR7, play Five Years Plan then."*
>
> *"I am US, I hold Lone Gunman, DEFCON is 3, it is hard to dispose of → play it now."*

These are conjunctions over **which specific cards are held** × **board and global state** ×
**timing**. Every ingredient is already in the observation, so this is purely architectural:

| fact | where it already is |
|:---|:---|
| "Europe is negative" | `global_features[64 + EUROPE]` — per-region VP differential |
| "DEFCON is 3" | `global_features[1]` |
| "it is AR7" | `global_features[7]`, with turn at `[6]` |
| "I am USSR" | `global_features[61]` (`I_AM_US`) |
| "I hold European Scoring" | that card's row, location one-hot slot `MY_HAND` |

## What a card row is

```
slots 0-7    location one-hot   DECK_OR_HIDDEN, MY_HAND, KNOWN_OPPONENT_HAND, DISCARD,
                                REMOVED, ONGOING, PEEKED, NOT_IN_GAME
slot  8      ops / 4                                                      static
slot  9      rel_side           +1 mine, -1 opponent, 0 neutral           flips with perspective
slots 10-12  era, one_time, is_scoring                                    static
slot  13     ACTIVE_CARD        1.0 this decision's card, 0.6 suspended, 0.3 next
```

The row already splits the way a lookup wants: **slots 8–13 are key material, slots 0–7 are the
value**.

## Why the dense path is not enough — stated as the honest null

M2d's card input is 110 × 10 dynamic slots → `Linear(1100 → 256)`. Each card sits at a fixed
offset, so *"is European Scoring in my hand"* is **already a linear readout**. The model is not
blind to it, and this rung must be judged against that, not against "it cannot see the card".

What attention adds is:

* **Selectivity.** The query comes from the trunk, so a lookup can be conditional — *"Europe is
  negative, therefore check Europe Scoring"* — where the dense projection computes every lookup
  unconditionally, always, at fixed cost.
* **Parameter efficiency.** Addressing all 110 cards individually through the dense path needs
  ~110 of its 256 output dims allocated that way, and nothing pressures SGD to allocate them so.
  A keyed lookup makes it structural instead of hoped-for.

That is an **inductive-bias argument, not an expressivity one**. It is the standard reason
attention beats a wide MLP, and it is a harder null than the ladder's earlier rungs faced.

## Two designs that were considered and dropped

**Location-bucketed pooling** — since location is a one-hot, `bucket[L] = Σ_i onehot_i[L] ·
enc(props_i)` is an exact masked sum, permutation-invariant and nearly free. Dropped on the
owner's objection, which is correct: the bucket contains only *total ops, count by side, era mix,
how many scoring cards* — about five numbers per location — and it discards precisely the
distinction the use case turns on.

**Property-only keys** — addressing by (ops, era, one-time, is-scoring, side) instead of identity,
which would avoid re-introducing a mechanism P21 measured negative. Dropped, again correctly:
**Europe Scoring and Asia Scoring are identical on every property**, so a property query retrieves
an average over the cards it needed to tell apart. Not a weaker version of the mechanism — a
broken one.

Both are recorded because they are the obvious cheap alternatives and the reasons they fail are
the reasons the expensive version is the right one.

## The mechanism

```
identity  E ∈ R^{110 × d_id}                        learned, one row per card
keys      K = [ E ‖ props(8..13) ]                  → Linear → (110, d_k) per head
values    V = [ location(0..7) ‖ props(8..13) ]     → Linear → (110, d_v) per head
query     q = Linear(trunk_h, n_heads × d_k)        state-dependent, therefore conditional
out       concat_heads( softmax(q·Kᵀ/√d_k) · V ) → Linear → into the TRUNK
```

**Into the trunk, not into a per-card head.** `pe_card` measured **−7** alone (M2e) and was the
worst variant when combined with identity (M2.5c, −90/−182 at 80M). Two independent measurements
say per-card corrections do not pay. Putting the retrieved answers in the trunk also makes them
available to country decisions and to the value head, and the Five Years Plan example is
ultimately a *timing* judgement that should inform more than one logit.

**Sizing**, for `d_id = 16`, `n_heads = 4`, `d_k = d_v = 32`:

| tensor | shape | params |
|:---|:---|---:|
| identity `E` | 110 × 16 | 1,760 |
| key proj | 22 → 4·32 | 2,816 |
| value proj | 14 → 4·32 | 1,792 |
| query proj | 480 → 4·32 | 61,440 |
| output proj | 128 → 128 | 16,384 |
| | **total** | **≈84k, +2.6% of M2d** |

Small enough that an Elo difference is not a capacity effect — the property that made M2a/M2b/M2c
clean, and which the width probe deliberately gives up.

## The refactor

`input_mode` currently governs both blocks, so tokenising cards would tokenise the board too —
and the board's positional path is the ladder's single biggest win (**M1, +109**). Splitting
`input_mode` into `board_mode` and `card_mode` lets the board stay `grouped` while cards gain
tokens, so the rung changes one thing.

This is also what blocked M3 on 2026-09-22: `grouped + card_self_attention` is refused because
grouped forms no tokens, and `entity + drop_static` is refused because a shared encoder needs the
static slots to tell its tokens apart. The split dissolves both.

## Rungs

| rung | change | tests |
|:---|:---|:---|
| **P22-a** | card lookup attention, `d_id = 16`, output to trunk | the mechanism |
| **P22-b** | a-with identity **removed** from the keys (properties only) | confirms identity is load-bearing here, and re-tests it in the role `forward_pass_trace.md` argued for rather than the one M2.5 tested |
| **P22-c** | drop the dense card projection, keeping only the lookup | is the positional card path still earning its 256 dims? |

P22-b is not the same experiment as the dropped "property-only keys" design: run *after* a works,
it is the ablation that attributes the gain, rather than a standalone proposal expected to work.

## Ordering

The width probe `E4-23-03/05` (`entity_proj_dim` 256 → 512, +26.7% params) reports first.

* **Null** → capacity is not the binding constraint, and P22 proceeds as the thing width cannot do.
* **Positive** → localize before building. `entity_proj_dim` widens board and card projections
  together, so the gain could be either or neither specifically; the `card_mode` split in this
  plan is what would allow the card projection to be widened alone.

## Protocol

Unchanged: two arms per rung, seed 3 → 160M and seed 5 → 80M, both clean for M2d at both budgets
and already rated there, so every comparison is within-seed. Adoption requires beating M2d at 80M
on **both** seeds. Every arm gets the collapse detector, and the per-rung collapse count is
recorded.
