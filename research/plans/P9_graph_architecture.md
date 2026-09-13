# P9 — What comes after the self-transform

Written against measurements, not as a survey. The point is which architectural moves address a
failure this project has *observed*, and which are attractive in general but aim at nothing we
can show is broken.

## What is already known, and therefore not worth re-deriving

| finding | where |
|:---|:---|
| A country's exact influence is recoverable from its own raw observation slots at 97% | §21.12 |
| Plain GCN destroyed a third of that before pooling; loss tracked node degree | §21.12 |
| A self-transform recovers it: 60% → 94%, worth **+53 and +83 Elo** | §21.13 |
| Pooling destroys the rest: token 89%, trunk 6-14% | §21.12-13 |
| A single-query attention read-out fixes none of that and costs the gain | §21.13 |
| The fix makes the board *available*, not contested: one seed used it, one did not | §21.14 |

So the graph layer is no longer the bottleneck, and the pooled trunk is being addressed by
per-entity heads (E3-14, running). **Neither of those is what the options below should be judged
against.** The question is what is left.

## Three things we cannot currently do, in order of evidence

### 1. The graph has one relation, and the game has several

`build_normalized_adjacency_matrix` uses `neighbors` and nothing else. But adjacency is not the
relation that decides most of the game:

* **Scoring is by region.** Presence, domination and control are counted over a region's members,
  and `Scoring::evaluate_region` folds in battlegrounds and superpower adjacency. Two countries in
  the same region interact strongly whether or not they are adjacent.
* **Placement legality** is adjacency *to a controlled country*, which is adjacency conditioned on
  state, not static adjacency.
* **Superpower adjacency** is already a per-country observation flag rather than a graph edge.

A country in Europe currently reaches another European country only by walking the adjacency
chain, at one hop per layer, through two layers. Regional scoring is a global function of the
region and the network has to rebuild it from two-hop neighbourhoods.

**This is the option with the most direct evidence behind it**, because §1.4.2 already records
that the per-region scoring scalars are not architecturally connected to their countries.

*Concretely*: a second relation (same-region edges) with its own weight matrix, i.e. a
relational GCN, `h_i = W_self x_i + W_adj Σ_adj + W_region Σ_region`. It is the same shape as the
change that just earned +53/+83, one more matrix.

### 2. Countries cannot see the global state

The global block — DEFCON, VP, the tracks, the decision context — is a separate branch,
concatenated only at the fusion trunk. A country token is computed without it. So "couping here
takes DEFCON to 1 and loses the game" is not something a country token can represent; the
observation supplies it as a **hand-engineered per-country flag** (board slot 7,
`is_coup_nuclear_hazard`), which is the tell. Each such flag is a computation the architecture
could not do, patched at the input.

*Concretely*: a global token concatenated to every node's features before the graph layers, or a
virtual node connected to all countries — standard, cheap, and it lets the network derive that
class of fact instead of being handed instances of it.

### 3. Edge weights are fixed, and the useful ones are state-dependent

`D^-1/2 (A+I) D^-1/2` weights a neighbour by degree alone. Whether Poland matters to East Germany
right now depends on who controls it and what is in hand. Attention over neighbours (GAT) learns
those weights — with a caveat worth knowing: original GAT's attention is
[*static*](https://arxiv.org/pdf/2105.14491), meaning the ranking of a node's neighbours is the
same for every query node, which is precisely the property we would be buying attention to avoid.
[GATv2](https://nn.labml.ai/graphs/gatv2/index.html) fixes this by reordering the nonlinearity,
is strictly more expressive, and the gap between them is reported to be
[largest exactly on heterogeneous graphs](https://kumo.ai/pyg/concepts/graph-attention-network/)
— which is what option 1 would make this graph into. **If GAT, then GATv2**, and it pairs
naturally with the multi-relational change rather than competing with it.

## Putting the superpowers on the map

Proposed: make US and USSR **nodes** in the graph rather than two bits per country, and drop the
bits. Two halves, and they deserve opposite answers.

**What superpower adjacency actually drives** — three distinct jobs, which is more than the two
bits suggest:

| site | job |
|:---|:---|
| `ops.cpp:87` | placement legality: influence may go into a country adjacent to your superpower |
| `ops.cpp:366-367` | realignment modifier, +1 per adjacent superpower |
| `scoring.cpp:48-55` | **regional scoring**: countries you control that are adjacent to the *opponent's* superpower are counted separately and change the region's VP |

The third is the interesting one, and it is exactly the disconnection §1.4.2 recorded: a regional
score depends on a *conjunction* — controlled **and** adjacent to the opponent's superpower —
aggregated per region. Today the network must rebuild that from two per-country bits and a pooled
trunk. With superpower nodes **and** the region relation above it becomes a two-hop path in the
graph, which is the form a message-passing network can actually compute.

**Add the nodes: yes, and it costs nothing to try.** The adjacency lives in `MapData`, and the
matrix is built model-side in `build_normalized_adjacency_matrix`. An 86-node graph with nine
extra edges — US to Canada, Japan, Mexico, Cuba; USSR to Finland, Poland, Romania, Afghanistan,
North Korea — is a **pure architecture change**: no observation change, no checkpoint
invalidated, comparable against E3-12 immediately.

**Drop the bits: no.** We have already run that experiment. E3-11 dropped 1,364 observation slots,
35.7% of the input, and measured **−40 and −0 Elo at identical throughput** (§21.9). These are 168
floats, 4.4% of the input, and dropping them changes the observation — which invalidates every
checkpoint including the +83 `E3-12-22` and the running E3-14, and resets the E3 Elo ladder. That
is paying the ladder for a saving already measured at zero.

Worse: unlike E3-11's slots, **these are not constant**. Slot 8 is `superpower_adjacent ==
my_player`, so it flips with the viewer — Canada reads 1 to the US and 0 to the USSR. They carry
real information, so removing them is strictly worse than the null E3-11 returned. Keeping them
alongside the nodes is redundant, and §21.9 is the evidence that redundant input slots cost
nothing.

**One framing to keep straight.** Superpower nodes are the right model of *adjacency* and the
wrong model of *global context*. They touch nine countries in one hop; Brazil never sees them. If
the goal is "a country can know DEFCON is 2", that is option 2's virtual node connected to all 84,
not these. Conflating the two would give a global-context mechanism that works for nine countries
— and a partial version of a feature is worse than none.

## On graph transformers specifically

Attractive, and I would not start here.

The case for them is real: self-attention lets a node keep its own identity because it attends to
everything with learned weights, which is
[the standard framing of why they resist over-smoothing](https://kumo.ai/research/introduction-to-graph-transformers/),
and full attention removes the hop limit that makes regional scoring hard to assemble. On 84
nodes the quadratic cost is irrelevant.

Three reasons to hold off:

1. **The failure they fix is the one we already fixed.** Over-smoothing destroying node identity
   is exactly what §21.12 measured and the self-transform repaired at 60% → 94%, for one weight
   matrix. Buying a transformer to solve it now would be re-paying for delivered goods.
2. **They are not immune to it.** Over-smoothing
   [is observed in transformers too](https://arxiv.org/pdf/2312.04234), and depth stops helping
   past a point, so "transformer" is not itself the guarantee.
3. **Losing the inductive bias is a real cost.** The
   [over-globalizing problem](https://arxiv.org/pdf/2405.01102) is that full attention spends
   capacity on distant nodes when the useful information is local. This map is small, sparse and
   genuinely local for most mechanics — coups, realignments and placement are all neighbourhood
   operations — so the prior we would be discarding is one the game actually has.

The honest version of the transformer case is **hybrid**: keep message passing for the local
mechanics and add a global attention channel for the regional and long-range part, which is what
GraphGPS-style architectures do and what
[the MPNN/transformer connection work](https://arxiv.org/pdf/2301.11956) argues can be equivalent
given a virtual node. Option 2 above is the cheap first instalment of exactly that.

## Proposed order

1. **E3-14 per-entity heads** — running; settles whether the trunk bottleneck was costing play.
2. **Region relation + superpower nodes** (option 1 and above) — strongest evidence, smallest
   change, same shape as the change that just worked, and the two combine: the regional-scoring
   conjunction that `scoring.cpp:48-55` computes is a two-hop path only once *both* exist. Both
   are architecture-only, so neither costs a checkpoint.
3. **Global token into the nodes** (option 2) — cheap, and it would let us *remove* a hand-built
   observation flag rather than add one, which is the kind of change that pays twice.
4. **GATv2 edge weights** (option 3) — only after 2, since it is most valuable once the graph is
   heterogeneous.
5. **Hybrid global attention channel** — only if 2 and 4 leave regional scoring still weak.

Each is falsifiable with instruments that already exist. The trunk ladder
(`ai/eval/state_readout.py`) measures what survives each stage, and `position_diagnostics`
measures whether the board gets contested. A region relation should show up as regional scoring
being linearly recoverable from a country token, which is a probe we do not yet have and should
build before the arm rather than after.

**A caution carried from §21.14.** Two seeds of one architecture disagreed about whether to
contest two entire regions. Any of these arms needs two seeds before its behavioural numbers mean
anything, and the Elo needs the pooled four-snapshot protocol (§20.7).

## Sources

- [An Introduction to Graph Transformers](https://kumo.ai/research/introduction-to-graph-transformers/)
- [How Attentive are Graph Attention Networks? (GATv2, ICLR 2022)](https://arxiv.org/pdf/2105.14491)
- [Graph Attention Networks v2](https://nn.labml.ai/graphs/gatv2/index.html)
- [Graph Attention Network: which neighbours matter](https://kumo.ai/pyg/concepts/graph-attention-network/)
- [Graph Convolutions Enrich the Self-Attention in Transformers](https://arxiv.org/pdf/2312.04234)
- [Less is More: the Over-Globalizing Problem in Graph Transformers](https://arxiv.org/pdf/2405.01102)
- [On the Connection Between MPNN and Graph Transformer](https://arxiv.org/pdf/2301.11956)
