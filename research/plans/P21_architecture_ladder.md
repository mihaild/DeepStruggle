# P21 — build the architecture up from an MLP, one mechanism at a time

**Status: proposed, not launched.** Requested by the owner, 2026-09-19: start from the simplest
architecture, add features one by one, measure each against the previous attempt and against the
newly trained cold-start pooled arm, and leave the dubious pooling until last.

## Why build up instead of down

Every choice in `ColdWarNetV2` was added on top of the ones already there, several measured only
against whatever baseline existed at the time. The result is a network nobody can currently
justify component by component: `attn_readout` is dead code in every arm, `num_attn_heads` has
never been varied, and both `per_entity_heads` and `country_identity` exist substantially to
repair a loss that pooling causes
([`../findings/training/forward_pass_trace.md`](../findings/training/forward_pass_trace.md)).

Building up gives every mechanism a matched control by construction and answers a question no
existing arm does: **which of these is paying for itself?**

**Governing rule: the simpler architecture wins ties.** A rung is adopted only if it beats the rung
below by more than seed spread, on both seeds. Otherwise the simpler variant carries forward. This
inverts the status quo, where complexity accumulated because nothing removed it.

## Scoping decision: no graph convolution

Dropped from the ladder by the owner, 2026-09-19. The record supports it, with one correction to
how it is usually stated: graph depth is **open, not settled negative** — *"the comparison that
removed the map graph reverses sign with the seed"*, and `E3-15` rests on one seed — but
`--graph-layers 0` has been the backbone of every arm from `E3-17` onward and **no measured
benefit has ever been demonstrated**, including on the arms that paired a graph with identity
(`E3-15-22` at 2 layers, `E3-16-21` at 1).

**Adjacency still reaches the network without a graph layer** — a correction to an earlier
version of this plan, which claimed it reached it by no route at all. It is not encoded as
topology, but it is encoded in exactly the form the *rules* use it, by four derived per-country
slots (`engine/src/observation.cpp`):

| slot | what it is | how adjacency enters |
|:---|:---|:---|
| 2 | `net_realign` | `compute_net_realign_mod` iterates `c_info.neighbors` and **counts controlled neighbours**, ±1 each, plus influence majority and superpower adjacency |
| 19, 20 | `can_my_place` / `can_opp_place` | `Operations::can_place_influence` iterates the neighbours and is true if **any holds friendly influence** |

`Scoring` also uses `can_place_influence` for the accessible-battleground counts in the global
block, so it reaches there too.

**Correction (owner, 2026-09-19): the coup slots 21/22 do *not* carry adjacency**, and an earlier
version of this file said they did. `Operations::can_coup` is `can_coup_or_realign` plus The
Reformer, and `can_coup_or_realign` tests only that the opponent has influence, the DEFCON
regional restrictions of rule 8.1.5, the US/Japan Pact and NATO protection. Couping needs no
access. The misreading came from a `can_place_influence` call on line 110 of `ops.cpp`, which is
inside `place_influence`, not the coup path. **The engine is correct here**; the claim about it
was not.

This makes dropping the graph **better motivated than a bare null**: a graph layer would be
re-deriving from raw edges what the observation already computes exactly, in the rule-relevant
aggregate. That is a plausible mechanism for why graph depth never showed a benefit, rather than
leaving it an unexplained result.

**What the derived slots do not carry**, and what a graph layer would therefore still be for:

* **Multi-hop structure.** Every slot above is one hop. "If I take Poland, East Germany becomes
  placeable next turn" is two hops and is not represented.
* **Which** neighbour. The slots are counts and booleans; the network sees *how many* controlled
  neighbours a country has, never which ones.

So the honest scope is: 1-hop, rule-relevant adjacency is present; topology and planning depth are
not. If the ladder stalls, multi-hop reachability is the thing to add — and it may be cheaper to
add it as observation features than as a graph layer, which is the owner's call under the standing
rule on observation changes.

## Static per-entity features only do work when the reader is shared or position-blind

Raised by the owner, 2026-09-19. A slot that never varies contributes a **fixed vector** to the
first layer's pre-activation. Where each entity has its own weights — a flat MLP, or any flattened
positional encoder — that vector is exactly a bias, so the weights reading it add **no
expressivity whatsoever**.

Measured over 25,600 observations from deep random rollouts, the slots that are constant for
*every* entity are precisely the ones `observation.cpp` derives from static `MapData` / card
properties:

| block | constant slots | floats |
|:---|:---|---:|
| board | 3, 4, 10–18 — stability, battleground, the 6-way region one-hot, WE / EE / SEA | 84 x 11 = **924** |
| card | 8, 10, 11, 12 — ops, era, one_time, is_scoring | 110 x 4 = **440** |
| | | **1,364 of 3,824 = 36%** |

(A shallower sample reports 54–68%, but that is contaminated: short rollouts never reach states
where `REMOVED`, `ONGOING` or `NOT_IN_GAME` fire. Only the figures above are structural.)

**Consequence for the ladder, and it is a confound, not a curiosity.** At `H=384` those 1,364
inputs consume `1364 x 384 = 524k` parameters that are functionally a bias — **17% of a 3.1M
budget**. Matching M0 to 3.1M *total* therefore hands it ~2.6M *useful*, and quietly biases the
comparison against the simplest rung.

**So: strip the 1,364 structurally-constant columns in M0 and M1.** This is
**expressivity-neutral** — the removed contribution is exactly a constant vector the bias can
already express — and it frees the 524k for trunk width, making the parameter match honest.

**Do not strip them anywhere else.** From M2 onward the entity encoder is *shared*, and that is
exactly when a static feature carries information: the shared function needs "is this a
battleground, and how stable" to compute a useful token, because it cannot know which country it
is looking at. Under pooling (M4) they are load-bearing for the same reason, plus they are all
that survives to distinguish country *types* in a symmetric summary.

**This sharpens the M1→M2 prediction.** Part of what weight sharing buys is the ability to *use*
36% of the observation that a positional dense layer provably cannot. If M2 beats M1, that is a
candidate mechanism; if M2 does not beat M1 even with 36% more usable input, the entity framing is
in real trouble.

It is the same complementarity as `country_identity`, from the other side: static features,
weight sharing and pooling are coupled. A static per-entity feature is informative exactly when
its reader is position-blind — and `country_identity` is informative exactly when it is not.

## Two comparisons, with different jobs

Fixed by the owner, 2026-09-19: **every modification is measured against the previous one, and
against whatever comes out best from the current four E4 runs.** These answer different questions
and must not be conflated.

**1. Against the rung below — the increment. This is what drives adoption.** Matched by
construction: same budget, same protocol, same parameter target, one mechanism different. It is
the only comparison in this plan that supports a causal claim about a mechanism.

**2. Against the best of E4 — absolute placement. This is a yardstick, not a control.**
**Settled 2026-09-19: both anchors are `E4-03-01@80M`.** It is the best of the four arms at 80M
*and* the best at any budget — it beats `E4-02-01@320M` 88.5% and `E4-01-01@240M` 81.5% despite
four times fewer steps and no warm start
([`../log/E4_architecture_ab_result.md`](../log/E4_architecture_ab_result.md)). The
budget-matched and best-overall anchors the owner asked for therefore coincide, which removes the
warm-start confound that was expected to attach to the second one.

**The anchor is a hard target.** `E4-03-01@80M` rates **2149.9** against `HeuristicBot` at 1500
and beats it 99.0%. Early rungs will lose to it heavily and that is expected — the anchor exists
to give one absolute scale across the ladder, so "M3 beat M2" can be placed against "and both are
still far behind the best thing we have", which the rung-to-rung comparison alone can never show.

Note it is also a *cold 80M* arm, so unlike the warm-started alternatives it is matched to the
ladder's own budget and start. That makes it a fair target as well as a demanding one.

If a rung ever beats it, that is a major result and the anchor re-points.

## The input

3,824 floats, fixed: board `84 x 26` = 2,184 at offset 0, cards `110 x 14` = 1,540 at offset 2,184,
global 100 at offset 3,724. History is off (`use_history=False`). Action space is 220.

---

# The rungs

## M0 — flat MLP

```
obs (B, 3824)
  → Linear(3824, H) → LayerNorm → GELU
  → R residual blocks, each: Linear(H,H) → LN → GELU → Linear(H,H) → +skip → GELU
  → policy: Linear(H,256) → LN → GELU → Linear(256,220)
  → value:  the existing scalar heads, unchanged
```

No structure whatsoever: every one of the 3,824 slots is an independent feature, and the first
layer may mix any slot with any other. `H` and `R` are set to hit the parameter target; `H=384,
R=4` lands near 2.8M before heads.

**Isolates:** the floor. Everything above must beat this.

**Why it matters more than it looks.** If M0 matches the current architecture, the entire
structured backbone is decoration, and that is the single most valuable result this ladder can
produce. It is not far-fetched: the observation is already engineered — control, stability,
battleground and coup-hazard are precomputed per country — so much of what a structured encoder
would have to discover is handed to it.

**Confound to watch:** M0 has by far the largest first layer (1.47M at H=384), so at matched total
parameters it gets the *narrowest* trunk. If M0 loses, check it is not losing on trunk width
alone — the parameter-matched control for M0 is M0 at a different H/R split.

**This endpoint has been measured before, on E3.** *"Is the structured backbone worth it over an
MLP?"* is recorded as **settled** — `E3-09` vs `E3-01`, **structure worth ~110 Elo, with the MLP
carrying 2.6x the parameters** ([`../questions.md`](../questions.md),
[`../archive/E3_ladder/findings/architecture.md`](../archive/E3_ladder/findings/architecture.md)).
So M0 is expected to lose, and losing tells us little.

**That is an argument for the middle rungs, not against the ladder.** E3 compared *no structure*
against *all of it* and learned only that the bundle is worth 110 Elo — not which part. M1, M2, M3
and the pooling removal were never run, and the 2x2 never existed.

### M0's result, 2026-09-19 — and the comparison it must be read against

| comparison | E3 | E4 |
|:---|---:|---:|
| MLP vs the **default** architecture | ~110 Elo | **42 Elo** |
| MLP vs the **late-E3 bundle** | never measured | **455 Elo** |

**`E3-09` vs `E3-01` was the MLP against the defaults**, so the like-for-like number on this
engine is **42 Elo**, not 455. Stated the other way round and more usefully: the default
architecture is worth **42 Elo over a plain MLP**, while the late-E3 bundle is worth **455**. The
engine change did not make structure matter more — it collapsed what the *default* structure was
worth, from 110 to 42.

Getting this wrong is easy and was got wrong once already: "structure is worth 455 here against
110 on E3" compares the MLP to a *different architecture* than E3 compared it to. **Always name
which structured arm the MLP is being measured against.**

The consequence for this ladder is the owner's reading: the default architecture is close to
worthless, and effectively all of the value sits in the four flags of the bundle. The ladder's
job is to find which of them carry it.

## M1 — grouped input projections

```
board_raw  (B,2184) → Linear(2184,256) → LN → GELU   → e_board  (256)
card_raw   (B,1540) → Linear(1540,256) → LN → GELU   → e_card   (256)
global_raw (B,100)  → Linear(100,128)  → LN → GELU   → e_global (128)
concat (640) → Linear(640,H) → LN → GELU → R residual blocks → heads
```

Parameters in the input stage: 559k + 394k + 13k = **966k**, against M0's 1.47M — so M1 can afford
a wider trunk at the same total.

**Isolates:** whether forbidding board↔card mixing in the first layer helps. M0's first layer can
form any linear combination of any slots; M1 forces each semantic block through its own bottleneck
first. This is purely an inductive-bias/sparsity question — no weight sharing, no tokens, position
fully preserved in both.

**Prediction:** small in either direction. The honest reason to run it is that it is the shape V1
and V2 both use, and it has never been tested against the ungrouped alternative.

## M2 — per-entity lookup on raw features

**Revised 2026-09-19 by the owner**, and the revision matters. M2 was first written as a shared
`Linear(26 -> d)` applied to all 84 countries, then flatten. That is **not "M1 plus weight
sharing"** — it is a *lossy restriction of M1*:

```
M1 board path:   Linear(2184, 256)                          full, 559k params
M2 (as written): Linear(26, 16) shared, GELU, Linear(1344, 256)
```

Ignore the GELU and the composite is a constrained factorisation of M1's first layer: every
country's 26 slots forced through the same rank-16 map before anything downstream sees them.
**26 -> 16 is lossy**, so that M2 could not represent functions M1 can. On a ladder that adds one
mechanism per rung, it was a *subtraction* dressed as an addition, and its only genuinely new
element was the per-country nonlinearity.

The owner's objection is the right one: the observation is **already hand-crafted features at
fixed positions** — control, stability, battleground, coup hazard, placement and realignment
legality are all precomputed per country. Compressing them per country buys nothing the flatten
does not already give. What is actually worth isolating is the **downstream lookup**: country
*i*'s own slots reaching country *i*'s logit.

That needs no learned token space at all. `pe_country` already reads the raw slots alongside the
token, so with `input_mode="grouped"` the raw features *are* the tokens.

```
board (B,84,26) ── flatten ─→ Linear(2184→256) ─┐
card  (B,110,14) ─ flatten ─→ Linear(1540→256) ─┼→ trunk (as M1)
global ─────────────────────→ Linear(100→128) ──┘
                                                 │
pe_country([board_nodes_i ‖ pe_trunk(h)]) → scalar added to country i's logit
pe_card(   [card_nodes_i  ‖ pe_trunk(h)]) → scalar added to card i's logit
```

**Isolates:** whether a per-entity lookup helps *at all*, with position already preserved and no
compression in front of it. Against M1 this is exactly one mechanism.

**Prediction:** genuinely uncertain, and that is what makes it worth running. The pooling-bottleneck
story says `pe_country` exists because the pooled trunk holds ~14% of a country's influence — but
M1's trunk is *positional* and holds far more, so the head may be redundant here. If it still
helps, the lookup is doing something beyond routing around pooling.

Configuration: `--ladder-input-mode grouped --per-entity-heads 64 --drop-static`, hidden 480 /
proj 256 = **3.190M** (+1.3% of the anchor), the same trunk width as M0 and M1 so only the
mechanism differs.

**The shared encoder is not deleted, only demoted.** It stops being a rung and becomes what it
actually is: one way of forming a token space, to be chosen only if attention turns out to need
one at a width other than 26. Attention on raw features is possible — `nn.MultiheadAttention`
takes `kdim`/`vdim`, so cards (14) can attend over countries (26) directly — and is the cheaper
thing to try first.

## M2.5 — a trainable identity vector, for the head only

Added by the owner, 2026-09-19, and it is the natural consequence of what M2 exposes.

`pe_card` receives a card's own 14 slots plus a *global* context. Only five of those are static
properties, so **95 of 110 cards are indistinguishable from at least one other** — `The Voice of
America`, `Colonial Rear Guards` and `Grain Sales to Soviets` are all US-sided 2-Ops mid-war
recurrent cards, and whenever they share a location the shared MLP must give them the identical
correction. M2.5 gives each entity its own learned vector so the head can tell them apart.

```
M2    pe input = [ raw(26) ‖ ctx(64) ]              =  90
M2.5  pe input = [ raw(26) ‖ identity(16) ‖ ctx(64) ] = 106
```

**Identity goes to the head and nowhere else**, which is a cleaner placement than the current
architecture uses. Identity is needed exactly where a function is *shared* across entities; with a
positional trunk that is the per-entity head alone, because the trunk already reads every entity
at its own offset. `ColdWarNetV2` instead concatenates identity into `board_nodes` *before* the
shared encoder, so it feeds both the pooled path and the head — one of which does not need it.
A test asserts the trunk's input width is unchanged, so a leak fails rather than passing quietly.

Two properties make this an unusually clean rung:

* **It costs 0.16% of parameters** — 3,195,233 against M2's 3,190,081, being the two embedding
  tables plus 16 extra inputs on each head. An Elo difference here **cannot** be a capacity
  effect, so the parameter-matching caveat does not apply at all.
* **It makes a split prediction.** The *card* head should gain substantially, since 95 of 110
  cards are currently indistinguishable to it. The *country* head should gain much less: 54 of 84
  countries collide on static slots, but a country's dynamic slots — influence, control, both
  deficits, can-place, can-coup, realignment modifier — differ between countries in any real
  position, so `pe_country` can already address. If both gain equally, that reasoning is wrong
  and worth knowing.

This is also the interaction predicted by
[`../findings/training/forward_pass_trace.md`](../findings/training/forward_pass_trace.md):
`identity_dim` and `per_entity_heads` travel together in the bundle worth 447 Elo because identity
gives the head distinguishable input and the head gives identity an addressable output. M2 and
M2.5 measure the two halves in order.

## M3 — card↔card self-attention

```
h_cards from M2                                              (B,110,d)
attn_out, _ = MultiheadAttention(embed_dim=d, num_heads=4)(Q=h_cards, K=h_cards, V=h_cards)
h_cards = LayerNorm(h_cards + attn_out)
         → flatten (B,110d) → Linear(110d,256) → LN → GELU → e_card
```

**Isolates:** whether a card can be valued *relative to the rest of the hand*.

**Why it is here at all.** The current architecture has **no attention between cards** — none, of
any kind. `cross_attn` is cards attending over *countries*; there is no self-attention over the
110 card tokens anywhere in the model
([`../findings/training/forward_pass_trace.md`](../findings/training/forward_pass_trace.md)).
Each card's token is computed independently by the shared `card_fc`, and cards meet each other
only through symmetric pooling and afterwards in the trunk. So relations like *"this scoring card
is dangerous **because** I hold no exit for it"*, or *"two high-Ops cards and one opponent event"*,
have no pathway at all. Hand composition is a first-class Twilight Struggle concept and the
network currently cannot represent it except as an average.

**Prediction:** this is the rung most likely to produce a real gain, precisely because nothing in
the current architecture covers it. The behavioural record supports that: the standing blunder
categories are `spaced_own_or_neutral` and `defcon_suicide_with_alternative` — both *disposal*
errors, which are exactly about what else is in hand.

**Confound:** the 110 tokens include every card in the deck, discard and removed pile, not only
the hand. Self-attention over all 110 may be dominated by cards that are not in hand. The location
one-hot (slots 0–7) distinguishes them, so it is learnable, but if M3 disappoints, masking the
attention to hand cards only is the obvious retry and should be tried before concluding.

## M4 — card→country cross-attention

```
h_board, h_cards from M2
attn_out, _ = MultiheadAttention(embed_dim=d, num_heads=4)(Q=h_cards, K=h_board, V=h_board)
h_cards_cross = LayerNorm(h_cards + attn_out)                       (B,110,d)
              → flatten (B,110d) → Linear(110d,256) → LN → GELU → e_cross
concat [e_board, e_card, e_cross, e_global] (896) → trunk → heads
```

**Isolates:** whether letting each card attend over the 84 countries adds anything beyond both
being present in the trunk. This is the mechanism the architecture is named after and it has never
been ablated.

**Prediction:** genuinely unknown. The case for it is that "how good is this card *here*" is a
card×country interaction the trunk would otherwise have to learn from the concatenation. The case
against is that with position preserved the trunk already sees both sides in full, and attention
is a lossy summary of an interaction the trunk could form directly.

**Confound:** at `d=16` with 4 heads the head dimension is 4, which is very small. If M3 loses,
re-run at `d=32` before concluding attention does not help — a null from an undersized attention
is a null about the size, not the mechanism.

## M5 — replace flatten with pooling  *(the last change, deliberately)*

```
h_board (B,84,d)  → [mean over 84 ; max over 84] (2d) → Linear(2d,256) → e_board
h_cards (B,110,d) → [mean ; max]                 (2d) → Linear(2d,256) → e_card
h_cards_cross     → [mean ; max]                 (2d) → Linear(2d,256) → e_cross
```

Everything else stays at the best configuration found. **This is a removal, not an addition:** it
takes away positional information every rung below it had, and it is what the current architecture
does.

**Isolates:** exactly what pooling costs.

**Why it is last:** so that its cost is measured against the best positional architecture rather
than against whatever happened to exist when pooling was chosen. The existing evidence says the
cost is large — the pre-pooling token holds ~89% of a country's influence and the 512-float trunk
~14% — but that was a representation probe, not a strength measurement, and the two have diverged
before in this record.

**Note the parameter asymmetry:** pooling makes the projection `Linear(2d,256)` — about 8k at
`d=16` against 344k flattened. Pooling is *much* cheaper. At matched total parameters, M4 gets a
substantially wider trunk than M3, which **flatters pooling**. Report both matched and unmatched;
if pooling wins only when it is handed the freed capacity, that is a different claim.

---

# The 2x2: two mechanisms that are not rungs

`per_entity_heads` and `country_identity` exist to repair the pooling loss. Testing them before
pooling asks a different question than after, so they get a 2x2 against the pooling axis rather
than a place on the ladder.

| | flattened (M4) | pooled (M5) |
|:---|:---|:---|
| neither | M4 | M5 |
| + per-entity heads | **M4-pe** | **M5-pe** |
| + country identity | **M4-id** | **M5-id** |

**`per_entity_heads`** adds `pe_country([h_board_i ‖ board_nodes_i ‖ pe_trunk(h)]) → scalar`,
added to country *i*'s logit, last layer zero-initialised. It exists because "a country's exact
influence is almost entirely recoverable from its own token and almost entirely absent from the
pooled trunk".

**`country_identity`** adds a learned 16-dim row per country, concatenated to that country's raw
slots before the shared encoder. Measured on `E3-30-28`, it is load-bearing in the pooled regime:
zeroing it moves per-country logits by 1.10x the entire across-country spread, and it disambiguates
the **54 of 84 countries that are indistinguishable from at least one other** by their static slots
([`../findings/training/country_identity_without_graph_conv.md`](../findings/training/country_identity_without_graph_conv.md)).

**The prediction that makes this worth running:** with position preserved, both should be
**redundant**. `pe_country` routes a country's token to its own logit because the pooled trunk
cannot — a flattened trunk already can. `country_identity` supplies per-country bias and addressing
that position supplies for free.

| outcome | reading |
|:---|:---|
| M4-pe ≈ M4 **and** M5-pe > M5 | per-entity heads confirmed as a pooling repair |
| M4-id ≈ M4 **and** M5-id > M5 | identity confirmed as a pooling repair |
| M4-pe > M4 | the heads do something pooling-independent — they also see `board_nodes_i` raw, so this is possible |
| M5 ≥ M4 with both repairs on | pooling plus its repairs is as good as position; the current architecture is vindicated |

That last row is a real possible outcome and the ladder must be able to report it.

---

# Protocol

Per [P19](P19_architecture_ab.md), already pre-registered:

* **Head-to-head, `tools/tournament.py`, 200 games per pair — 100 per seat**, reported per side.
* **Sanity gates before any headline is read:** the arm beats `heuristic` and `heuristic_mcts`
  decisively, and is silent under the health alarms (`NOPOOL` / `POOLSTUCK` / `KLSPIKE`).
### Two arms per modification: one to 80M, one to 160M

Fixed by the owner, 2026-09-19. **Every modification gets two arms on different seeds — one run to
80M, one run to 160M.** Snapshots stay at every 5M, so the 160M arm yields its whole curve, not
two points.

That single choice buys three things at once:

| read | from | why it is clean |
|:---|:---|:---|
| **variance at 80M** | arm A @ 80M vs arm B @ 80M | two independent seeds at the same step count |
| **slope, 80M → 160M** | arm B @ 80M vs arm B @ 160M | **within-seed**, so seed variance cancels out of the slope entirely |
| **level at 160M** | arm B @ 160M | one seed only — suggestive, not decisive (see below) |

**Intercept versus slope is the point.** A mechanism that lifts the curve and then runs parallel
has bought a constant; one that changes the slope is still paying off at 160M and would pay more
at 320M. Those are different findings with different consequences, and no comparison at a single
budget can tell them apart. This record has already been bitten by the single-budget version of
this: part of what was credited to interventions turned out to be what longer training does
anyway — the 240M control matched identity@160M on almost every behavioural line while being
92–109 Elo weaker.

**Honest limits, recorded now rather than argued later:**

* Two seeds is one degree of freedom. It bounds the variance loosely; it does not estimate it
  well. The protocol is built for effects that are large relative to spread.
* **The 160M comparison between rungs is single-seed on each side**, so a difference there is
  confounded with seed. The *within-seed slope* is the trustworthy quantity at that budget, not
  the level.
* Seed spread here is ~95 Elo and two architecture conclusions have already been withdrawn for
  exactly that reason.

**The owner's stated expectation is that early ablations will be far larger than variance.** If
they are not — if the first two rungs land inside spread — that is the signal to stop and add
measures ad hoc rather than to keep climbing. A ladder of inconclusive steps is worse than no
ladder, because it looks like a result.

* **Adoption:** a rung advances only if it beats the rung below **at 80M on both seeds** and the
  margin exceeds the two arms' own spread. A tie carries the simpler variant forward and is
  recorded as *"no detected effect at 80M, 2 seeds"* — never as "no effect". Report the slope
  alongside, whichever way adoption goes: a rung that ties at 80M but has a clearly steeper
  within-seed slope is a candidate to re-test at a larger budget, not a dead end.

## Matched by steps or by compute? Both, with steps primary

Raised by the owner: an MLP will likely be faster than the current architecture at the same
parameter count, so equal parameters is not equal compute.

**It is a real effect, and now measured.** `E4-03-01` and `E4-04-01` differ only in the network,
so their throughput gap is exactly the architecture's compute cost:

| arm | architecture | steps/s (median, post-warmup) | 80M costs |
|:---|:---|---:|---:|
| `E4-04-01` | defaults, `graph_layers=2` | **14,057** | 1.58 h |
| `E4-03-01` | late-E3: identity + per-entity heads, `graph_layers=0` | **11,756** | 1.89 h |

**20% slower per step**, with under 2% spread across ~1,500 iterations. Note it has *no* graph
convolution, so the cost is the per-entity heads and identity. An isolated forward-pass benchmark
put the same pair 18% apart, so **architecture cost passes through to training throughput almost
1:1** — the network is a dominant share of a step here, not the engine.

**The policy:**

1. **Primary axis: matched steps** (80M). The ladder asks a representation question — does this
   mechanism extract more from the same experience? That is sample efficiency, and it is the axis
   that is reproducible across machines. Compute matching is hardware-dependent and this record
   has already been confounded that way once: `E3-31-28` ran at 580 steps/s against `E3-30-28`'s
   10,274 purely because they shared a GPU.
2. **Always record `steps_per_sec_avg` and wall-clock** for every arm, so each mechanism's compute
   price is visible rather than inferred.
3. **Compute parity is an adoption *gate*, not the measurement.** If a rung wins at matched steps
   but is materially slower, it has to beat what the cheaper rung would have done with the same
   wall clock. Concretely: **if the winner is >10% slower, re-run the loser at matched wall-clock
   — `N_loser = N_winner x (sps_loser / sps_winner)` — and adopt only if the winner still wins.**
   Expressing parity in *steps* rather than seconds keeps it reproducible.

One extra arm, and only when the speed gap is material and the slower rung won. At ~1.7 GPU-hours
per arm that is cheap insurance against adopting a mechanism that is really just spending more
compute.

**Expect M0 to be the fastest by some margin.** Its 3,824→H first layer is a single large matmul,
which is far more GPU-efficient per parameter than 84 small per-entity applications plus
multi-head attention. So the compute gate will bite hardest exactly where the ladder is most
likely to want to adopt something — which is the reason to fix the rule now rather than after
seeing the numbers.

**Parameter matching is mandatory**, 3.1M ± 5%, by adjusting `H` and `R`. [P6](P6_attention_backbone.md)
records that capacity is not v2's bottleneck, so an uncontrolled increase would read "bigger won"
as "the mechanism won". The realised count goes in each run's `--description`.

## How every rung is reported

Fixed by the owner, 2026-09-19. **One table per new arm**, so twelve modifications produce twelve
comparable reports instead of twelve summaries that each emphasise something different.

**Rows:** every ladder arm so far, plus `E4-03-01@80M`, `E4-04-01@80M` and `HeuristicBot`.

**Columns:** Elo | steps/s | USSR and US win rate against the **previous rung** | USSR and US win
rate against the **anchor**.

```
| arm                        |     Elo |  steps/s | USSR v prev | US v prev | USSR v anch | US v anch |
|----------------------------|---------|----------|-------------|-----------|-------------|-----------|
| E4-03-01@final             |  2179.5 |   11,732 |           — |         — |           — |         — |
| E4-05-02@final  (M0 @160M) |  1813.3 |   60,814 |           — |         — |        9.0% |      9.0% |
| E4-04-01@final             |  1706.9 |   14,057 |           — |         — |        3.0% |      9.0% |
| E4-05-02@80M    (M0 @80M)  |  1666.6 |   60,814 |           — |         — |        3.0% |      7.0% |
| E4-05-01@final  (M0 @80M)  |  1650.0 |   63,187 |           — |         — |        8.0% |      5.0% |
| HeuristicBot               |  1500.0 |        — |           — |         — |        1.0% |      1.0% |
```

`tools/scripts/ladder_report.py <tournament.json> --previous <label> [--anchor <label>]`.

Three properties of the format, each there because of something already measured:

* **Per side, never pooled.** M0 beats the E4 defaults **80% as USSR and 49% as US**; the pooled
  64.5% shows neither. Side asymmetry that large is a property of a rung worth seeing before it is
  carried upward.
* **steps/s in every row.** The rungs differ by **4-5x** in throughput — M0 runs 60,814 against
  the anchor's 11,732 — so an Elo quoted without its compute price is half a result, and the
  compute-parity gate cannot be applied without it.
* **One tournament per table.** Bradley-Terry ratings are field-relative: the anchor rated 2107.2,
  2158.3 and 2179.5 in three tournaments, unchanged, purely because the entrants differed. Only
  deltas inside a single tournament mean anything, so the tool reads exactly one JSON and a rung
  is always rated in a field containing both the anchor and the rung below it.

**The anchor is `E4-03-01@80M`.** (The request named `E4-01-01@80M`; that arm is the warm-started
unpooled one and is not the anchor either anchor rule selects, so this is read as a slip. The
reference is a `--anchor` flag, so it is one word to change if it was not.)

## Budget

`E4-04-01` ran 80M in ~100 minutes, so an arm is ~1.7 GPU-hours.

One modification = one 80M arm + one 160M arm = **240M steps**, which at the measured
11,756–14,057 steps/s is **4.7–5.7 GPU-hours**.

| stage | modifications | steps | GPU-h |
|:---|---:|---:|---:|
| the ladder, M0–M5 | 6 | 1.44B | ~28 |
| the 2x2 extras (M4-pe, M4-id, M5-pe, M5-id) | 4 | 0.96B | ~19 |
| `d ∈ {8, 32}` screen at M2, 80M single arms | 2 | 0.16B | ~3 |
| compute-parity checks | — | 0 | tournaments only |
| **total** | **12** | **~2.6B** | **~50** |

Rates are measured, not assumed: 14,057 steps/s for the cheap architecture and 11,756 for the
expensive one. Budget the ladder at the slow rate; the early rungs should beat it comfortably.

## The backbone — built, 2026-09-19

`ai/models/ladder_net.py`. `LadderNet` subclasses `ColdWarNetV2`, so the heads, the value contract
and the 220-action space are untouched and only `_encode` differs. `create_like` rebuilds a copy
from the model's own `ladder_config()` rather than from an argument list, which is how two runs
previously died at their first snapshot.

**Every axis is required, at construction and at the CLI.** `--arch ladder` refuses to run unless
each is named, and naming a ladder flag under another `--arch` is refused rather than ignored.
`tests/training/test_ladder_net.py` pins the rungs *and* the refusals — 37 tests.

Four combinations are refused outright, each encoding something the plan establishes:

| refused | because |
|:---|:---|
| per-entity heads with `flat`/`grouped` | there are no tokens for them to read |
| attention with `flat`/`grouped` | there is nothing to attend over |
| `identity_dim > 0` with `flat`/`grouped` | position already identifies the entity; identity there is pure redundancy |
| `drop_static` with `entity` | the static slots are how a *shared*, position-blind encoder knows what it is looking at |

The rungs, as commands:

```bash
COMMON="--ladder-entity-proj-dim 256 --ladder-hidden-dim 384 --ladder-res-blocks 4"
# M0  tools/train.py --arch ladder --ladder-input-mode flat    --ladder-aggregation flatten \
#                    --ladder-entity-dim 16 --drop-static $COMMON
# M1  ... --ladder-input-mode grouped --ladder-aggregation flatten --drop-static
# M2  ... --ladder-input-mode entity  --ladder-aggregation flatten
# M3  ... --ladder-input-mode entity  --ladder-aggregation flatten --ladder-card-self-attention
# M4  ... M3 plus --ladder-cross-attention
# M5  ... M4 but --ladder-aggregation pool
# 2x2 ... add --per-entity-heads 64 or --identity-dim 16 to M4 / M5
```

**Measured parameter counts** at `d=16`, `H=384`, `p=256`, 4 blocks — note the asymmetry the plan
warned about:

| rung | params |
|:---|---:|
| M0 flat | 2.24M |
| M1 grouped | 2.36M |
| M2 entity/flatten | 2.55M |
| M3 + card self-attention | 2.55M |
| M4 + cross-attention | 3.10M |
| **M5 pooled** | **1.88M** |

M5 is **1.2M cheaper than M4** on identical settings, because pooling replaces `Linear(84d, 256)`
with `Linear(2d, 256)`. Matching total parameters would hand pooling a much wider trunk — which is
exactly why the plan requires reporting matched *and* unmatched.

## Risk

At 80M on two seeds against ~95 Elo of spread, the ladder may lack the resolution to separate
adjacent rungs. **If the first two rungs tie, raise the budget or the seed count before
continuing** rather than reading ties as answers: a ladder of inconclusive steps is worse than no
ladder, because it looks like a result.
