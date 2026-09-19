# ColdWarNetV2, exactly: where a country's state goes

`CLAUDE.md` describes the architecture in one line ("GNN GraphConv + card/country cross-attention
+ global ResNet with masked action heads"). This is the actual trace, written from
`ai/models/coldwar_net_v2.py::_encode` and `_policy_logits`, because the answer to "does X reach
the decision" has been guessed wrong more than once. **Update discipline: rewrite in place when
the forward pass changes.**

## Every country is transformed by the same weights

The board slice is `(B, 84, 26)` — 84 countries, 26 floats each. With `identity_dim > 0` the
country identity rows are concatenated, giving `(B, 84, 26 + identity_dim)`.

That tensor goes through **one shared transform**, applied identically to all 84 rows:

* `graph_layers == 0` → `board_fc`, a single `Linear(board_in, 64)` + GELU. **No adjacency is
  used at all.**
* `graph_layers >= 1` → `gconv1` (and `gconv2`) over `norm_adj`, still one shared weight matrix
  per layer, plus the self-transform weight when `--self-transform` is set.

Either way the result is `h_board`, `(B, 84, 64)`: one token per country, produced by weights
shared across countries.

The card path is the same shape of thing: `(B, 110, card_features)` ⊕ card identity → `card_fc` →
`h_cards` `(B, 110, 64)`.

## Where `h_board` goes — four places

```
h_board (B, 84, 64)
 │
 ├─1─ mean + max over the 84 countries → board_proj → e_board (B, 256)
 │       symmetric: position is destroyed here
 │
 ├─2─ cross_attn(Q = h_cards, K = h_board, V = h_board)      [nn.MultiheadAttention, 4 heads]
 │       → attn_out (B, 110, 64), attn_weights (B, 110, 84)
 │       → h_cards_cross = LayerNorm(h_cards + attn_out)
 │       → mean + max over the 110 CARDS → cross_card_proj → e_cross (B, 256)
 │
 ├─3─ attn_readout: trunk queries the tokens              [DISABLED — 0 in every arm run]
 │
 └─4─ pe_country([h_board_i ‖ board_nodes_i ‖ pe_trunk(h)]) → one scalar per country
         the ONLY unpooled route from country i's state to country i's logit
```

## The card path is the same shape — and has no card↔card attention

```
card_raw (B,110,14) ++ card_identity → card_fc  [ONE shared Linear → 64]  → h_cards (B,110,64)
 │
 ├─1─ mean + max over the 110 cards → card_proj → e_card (B,256)
 │       symmetric: which card is which is destroyed here, exactly as for countries
 │
 ├─2─ Q of cross_attn(Q=h_cards, K=h_board, V=h_board)
 │       → h_cards_cross = LayerNorm(h_cards + attn_out)
 │       → mean + max over the 110 CARDS → cross_card_proj → e_cross (B,256)
 │
 └─3─ pe_card([h_cards_i ++ card_nodes_i ++ pe_trunk(h)]) → one scalar per card
```

**There is no attention between cards.** `cross_attn` is cards attending over *countries*; no
self-attention over the 110 card tokens exists anywhere in the model. Each card's token is
computed independently by the shared `card_fc`, and cards meet each other only through the
symmetric pooling in routes 1 and 2, and afterwards in the trunk.

Note that route 2 pools **over cards** as well. So the cross-attention output — the one place a
card's relationship to the board is represented — is itself collapsed to a mean and a max before
it reaches the trunk. Only `pe_card` sees it per card.

That leaves hand composition unrepresentable except through pooled statistics: "this scoring card
is dangerous *because* I hold no exit for it", or "I hold two high-Ops cards and one opponent
event", are card-to-card relations with no pathway. Whether that matters is untested — no arm has
ever had card self-attention to compare against.

The trunk:

```
fused = [e_board(256) ‖ e_card(256) ‖ e_cross(256) ‖ e_global(128) (‖ e_hist(128))]
h     = res_blocks(fusion_in(fused))                                      (B, 512)
```

And the action logits:

```
logits = policy_head(h)                        dense, all 220 actions, off the 512 trunk
       + [ pe_card | 0 | pe_country | 0 ]      per-entity correction, ADDED not substituted
```

The zero blocks are the resolution node, roll-die, branch, confirm, DEFCON and region actions —
they name no entity, so they stay dense. The correction's final layer is **zero-initialised**, so
the network begins exactly as the dense baseline and learns the refinement on top.

## Why route 4 exists: pooling is the bottleneck

A country's state reaches its own logit two ways, and the first one loses nearly everything.
Measured by representation probe
([`../../archive/E3_ladder/findings/architecture.md`](../../archive/E3_ladder/findings/architecture.md)):

| read from | fraction of the country-influence gap closed |
|:---|---:|
| the pre-pooling token (`gconv1` / `board_fc` output) | **~89–94%** |
| the 512-float trunk | **~14%** |

No read-out that ends in one fixed-size summary can do better; mean and max over 84 countries are
symmetric functions, so *which* country held what does not survive them. `pe_country` routes
around the pool rather than fixing it.

Two consequences that have already cost compute:

* **Without `per_entity_heads` the pooled trunk is the only path**, so every country logit is
  computed from a vector holding ~14% of the per-country detail.
* **Making the correction a replacement rather than an addition cost 219 Elo.** Routing a logit
  through `pe_trunk` alone made a 64-float projection the only path from the trunk to that logit,
  where the dense head reads all 512 — each logit gained its own entity's detail and lost seven
  eighths of its view of the situation.

## What this means for country identity

The three live routes use identity for three different purposes, which is why "is it just a static
bias?" has no single answer
([`country_identity_without_graph_conv.md`](country_identity_without_graph_conv.md)):

| route | what identity does there |
|:---|:---|
| 1, pooled | the **only** way country-specific information survives a symmetric pool |
| 2, attention keys | genuine **addressing** — lets a card select *that* country, not "a country that looks like this" |
| 4, `pe_country` | a **per-country bias**; the shared `Linear` has one bias vector for all 84, and identity is what makes it 84 |

## Naming traps in the flags

* **`per_entity_heads` is not attention.** It is the width of the per-country/per-card residual
  MLP. The attentional read-out is `attn_readout`, which is 0 everywhere.
* **`num_attn_heads` (default 4) governs the real attention**, `cross_attn`, which is always
  present regardless of `graph_layers` — and no arm has ever varied it.
* **Pooling is not a graph-convolution feature.** `board_mean`/`board_max` happen at
  `graph_layers = 0` exactly as they do at 2. Turning the graph off does not turn pooling off.

## What the per-entity observation slots actually are

"26" and "14" appear throughout this record without a definition anywhere. They are the
hand-designed per-entity feature counts of observation layout v2.3, fixed in
`engine/src/observation.cpp`.

### 26 per country

| slot | meaning |
|---:|:---|
| 0, 1 | my / opponent influence ÷ 10 |
| 2 | net realignment modifier ÷ 5 — iterates `c_info.neighbors`, ±1 per controlled neighbour, plus influence majority and superpower adjacency |
| 3 | stability ÷ 5 **(constant)** |
| 4 | battleground **(constant)** |
| 5, 6 | controlled by me / by opponent |
| 7 | couping here would take DEFCON to 1 and lose outright |
| 8, 9 | superpower-adjacent to me / to opponent |
| 10–15 | region one-hot, 6 regions **(constant)** |
| 16–18 | in Western Europe / Eastern Europe / Southeast Asia **(constant)** |
| 19, 20 | I / opponent can place influence here — iterates the neighbours |
| 21, 22 | I / opponent can coup here — opponent influence, DEFCON 8.1.5, NATO, The Reformer; **no** adjacency test |
| 23 | node count ÷ 5 — times this country was acted on in the current op |
| 24, 25 | influence deficit to my / opponent control ÷ 5, capped at 2.0 |

**Eleven of the 26 are constant per country** (3, 4, 10–18) = 924 of the 2,184 board floats. That
is what makes `--drop-static` correct for a positional reader and wrong for a shared encoder
([`country_identity_without_graph_conv.md`](country_identity_without_graph_conv.md)).

### 14 per card

| slot | meaning |
|---:|:---|
| 0–7 | location one-hot: deck-or-hidden, my hand, known opponent hand, discard, removed, ongoing, peeked, not in game |
| 8 | Ops ÷ 4 **(constant)** |
| 9 | side relative to me: +1 mine, −1 opponent, 0 neutral |
| 10 | era ÷ 2 **(constant)** |
| 11 | one-time — removed after its event **(constant)** |
| 12 | is a scoring card **(constant)** |
| 13 | active-card marker: 1.0 this decision, 0.6 suspended below it, 0.3 committed next |

Four of the 14 are constant = 440 of the 1,540 card floats.

**The consequence for the ladder.** These are *already engineered* features — control, coup hazard,
placement and realignment legality and both influence deficits are precomputed. A shared
`Linear(26 → d)` in front of them is a lossy compression of a designed feature set, not a learned
encoder discovering structure, which is why P21's M2 was rewritten as a per-entity *lookup* on the
raw slots rather than a shared projection of them.

## `pe_card` cannot tell most cards apart, and that is why identity travels with it

Raised by the owner, 2026-09-19: *"does it mean that Voice of America and Colonial Rear Guard
always get the same correction?"* Measured from `rules/cards.json`:

```
110 cards, 46 distinct static signatures (ops, side, era, one_time, is_scoring)
  uniquely identified by their static slots :  15
  indistinguishable from >=1 other card     :  95
  largest group                             :   6
```

`The Voice of America`, `Colonial Rear Guards` and `Grain Sales to Soviets` are one such
group — all US-sided, 2 Ops, mid-war, recurrent. `pe_card`'s input is the card's own 14 slots
plus a *global* context vector, and of those 14 only five are static properties, eight are a
location one-hot and one is the active marker. **So whenever those three sit in the same location
and none is active, `pe_card` receives an identical input and a shared MLP must emit an identical
correction.** Six-card groups exist too.

**Countries are much less affected**, and the asymmetry is the point. 54 of 84 countries collide
on their *static* slots, but a country carries rich *dynamic* slots — influence, control, both
influence deficits, can-place, can-coup, the realignment modifier — that differ between countries
in any real position. A card's dynamic slots are its location and whether it is active. So
`pe_country` can address a specific country; `pe_card` can mostly only address a *type*.

**This is not a defect, it is a factorisation** — and worth stating precisely, because the
overclaim is tempting. The dense policy head still gives every card its own output row, so card
identity reaches the decision positionally through `base`; the per-entity term then adds a
*type-level* correction on top. Identity-aware base plus type-aware correction is a coherent
design. Whether it earns its parameters is an empirical question, and is what P21's M2 measures.

**It does explain why `identity_dim` and `per_entity_heads` appear together in the bundle that is
worth 447 Elo.** Identity supplies per-entity distinguishability in the head's *input*; the head
supplies per-entity addressing in the *output*. On the card side neither is much use without the
other, which predicts an interaction between them rather than two independent effects — see
[`country_identity_without_graph_conv.md`](country_identity_without_graph_conv.md) for the same
complementarity seen from the identity side.

## Card actions and country actions are never legal at the same decision

Measured over 25,600 decision points from engine self-play:

| legal actions at a decision | share |
|:---|---:|
| card block only `[0, 110)` | 25.1% |
| country block only `[116, 200)` | 52.9% |
| neither block | 22.0% |
| **both blocks** | **0.00%** |

The game alternates between choosing a card and choosing a target, and the mask reflects that
exactly — it never offers both. Worth recording because it kills a plausible-sounding class of
explanation: nothing that depends on the network trading a card action off against a country
action can be right, since the two never compete. A P21 hypothesis about partial per-entity
corrections distorting that trade-off died here.
