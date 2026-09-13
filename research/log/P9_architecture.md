# P9 — the architecture programme, as it happened

The journal of the work that took the network from "it cannot tell 86% of its cards apart" to the
current recipe: how each defect was diagnosed, what was built in response, what the probe said
before and after, and what the Elo said. **Update discipline: append-only.** The two failed arms
(the attention read-out and the replacing form of the per-entity heads) and the retracted
readings (the anchor that inverted orderings, the influence probe that scored below its own
baseline twice, the trunk figure that moved the right way while play halved) are the most
valuable entries in the file and are never removed. The *conclusions* — what the architecture is
now and what each change is worth — are kept current and short in
[`../findings/architecture.md`](../findings/architecture.md); this file is why they are believed.
The plan this programme was run from is
[`../plans/P9_graph_architecture.md`](../plans/P9_graph_architecture.md), and the arm names are
decoded in [`../method/run_nomenclature.md`](../method/run_nomenclature.md).

These entries were written under the section numbers of the original `research/experiments.md`, by
way of the retired `research/metrics.md`. The numbers are recorded here so older references still
resolve; new entries get a descriptive heading and no number.

| written as | now |
|:---|:---|
| §1.4.2 | The per-region scoring scalars are not architecturally connected to their countries |
| §21 | Where the programme started: what an 80M arm spends its budget learning |
| §21.1 | The critic does not see a provoked DEFCON-1 coming, at any node |
| §21.2 | Windowing a provoked DEFCON-1 moves exactly the class it targets |
| §21.3 | Rate arms against H2 @480M, not against HeuristicBot |
| §21.4 | The model cannot tell 86% of cards apart |
| §21.5 | The structured backbone is worth ~110 Elo, and the anchor said the opposite |
| §21.6 | Identity embeddings are worth ~115 Elo |
| §21.7 | What the trunk actually encodes, read off linearly |
| §21.8 | Identity embeddings hold ~110 Elo across a doubling |
| §21.9 | Dropping the 1,364 constant observation slots buys nothing |
| §21.10 | One Elo scale for E3, and identity is worth a doubling of compute |
| §21.11 | What the 160M arms actually do, and where identity's Elo does *not* show up |
| §21.12 | Exact influence and control, read off the trunk per country |
| §21.13 | Fixing the graph layer works; attention-pooling into the trunk does not |
| §21.14 | Does a network that can see the board contest more of it? |
| §21.15 | Per-entity policy heads, built wrong: −219 Elo, and why |
| §21.16 | The same idea built two ways spans 413 Elo |
| §21.17 | The architecture progression on one scale |
| §21.18 | Does the network price a country for the operation it is performing? |
| §21.19 | Does it know which region a scoring card scores? |

---

## Before the programme: the per-region scoring scalars are not architecturally connected to their countries

`global_features[64..69]` carry the live net VP differential per region, from
`Scoring::evaluate_region`, so presence, domination, control, battlegrounds and superpower
adjacency are all folded in. Measured on Europe: empty 0.000, domination **+0.500**, control
**+1.000**. Control is forced to ±20 because its raw `net_delta` is *8* against domination's
*10* -- Europe's `control_vp` is 0, so without the special case the observation would rank a won
position below a merely dominated one.

Each country also carries a 6-way region one-hot in `board_features[10..15]`, plus Western
Europe / Eastern Europe / South-East Asia flags at 16..18. So the country-to-region *mapping* is
an explicit input; nothing has to learn it from data.

**What is missing is the binding between the two.** In `ColdWarNetV2.extract_features` the board
branch runs two GraphConv layers over the 84 countries and then **mean- and max-pools across all
of them** into 128 floats before the global block is ever seen. The region scalars arrive
through a separate `global_proj`, and the two meet only in the fused trunk. The policy head is
then `Linear(hidden -> 256) -> Linear(256 -> 212)` off that fused vector: there is no
per-country output path at all.

So there is no architectural route from "the Europe scalar rose" to "because of France". The
association has to be discovered statistically, through a trunk that has already discarded which
country is which. That is a plausible reason the critic values regional position poorly, and it
is not fixed by adding features -- the information is already there and correctly scaled.

Compounding it: arm H2 reaches Europe control in **0.0%** of games, so the top of that ramp is
essentially unvisited and whatever the critic predicts at 1.0 is extrapolation from the 0.5
neighbourhood. The feature makes control *learnable*; visitation is what would make it learned.

---

## Where the programme started: what an 80M arm actually spends its budget learning

Measured on `p1_scalar_nofilter`, binned by snapshot (eval rows carry `iteration`, not
`total_steps`; 80,019,456 steps at iteration 1221 is 65,536 steps/iteration):

| | 5M | 30M | 60M | 80M | 90% of the run's movement by |
|:---|---:|---:|---:|---:|---:|
| anchor win rate | 32% | 85% | 84% | 84% | **30M** |
| empty battlegrounds, turn 8 | 11.91 | 9.49 | 7.62 | 7.60 | **60M** |
| uncontrolled battlegrounds, turn 8 | 20.88 | 17.95 | 16.47 | 16.33 | **60M** |
| DEFCON-suicide blunder rate | 9.8% | 6.2% | 7.3% | **10.0%** | **flat** |

Two things follow.

**Most of the anchor number is bought in the first 30M**, about 38% of the budget, and the
remaining 50M moves it by a few points inside the noise band. The strategic diagnostics are the
opposite: they improve steadily and are still moving at 60M. So the anchor metric saturates long
before the behaviour does — another reason not to rate an arm by it
([`variance_and_noise.md`](variance_and_noise.md)).

**The DEFCON-suicide rate does not improve over a whole run.** It starts at 9.8%, wanders between
4.6% and 10.7%, and ends at 10.0%. This is the mistake the project goal names first, and 80M
steps of self-play do not touch it. That is a finding about the *reward*, not about the budget: a
warm start cannot save re-learning something nothing learns in the first place.

> **The taxonomy changed underneath this metric.** `blunder_defcon_suicide_with_alternative`
> was redefined (battleground-conditioned influence, Tear Down This Wall added, Nuclear Subs made
> one-sided, Ortega and Star Wars added, We Will Bury You / How I Learned / Junta removed). The
> table above is the **old** definition. Runs from the second P1 leg onward use the new one, so
> the series is **not continuous across that boundary** — a step change there is the definition
> moving, not the policy. Re-measure before comparing across it.

## The critic does not see a provoked DEFCON-1 coming, at any node

Traced through the `h2_480M_provoked_*` replays by replaying their recorded flat actions through
a fresh engine on the replay's own seed and reading `v_win` from the **losing** side's
perspective. (Perspective checked: the two sides sum to +0.05 at a sampled node, so these are
the loser's own values, not a sign error.)

Every value is read on the state *before* that node's action is applied, so the cost of a single
choice is the difference between the node where it is made and the node after it. Quoting the
value at a node against the value at the *previous* node measures the opponent's intervening
moves as well, and is not attributable to the choice.

In 7107, 7115 and 7118 the card was played **Ops first, event later**, so `PlayMode: OPS` on an
opponent card is the point of no return. In 7119 Grain Sales was **unspaceable**, so the card
selection is. Both were tested:

| game | v at the card node | delta from selecting the fatal card | delta from choosing OPS |
|:---|---:|---:|---:|
| 7107 Star Wars | −0.770 | **+0.004** | **+0.006** |
| 7115 Lone Gunman | −0.052 | **−0.003** | **−0.004** |
| 7118 Grain Sales | −0.691 | **−0.008** | **+0.007** |
| 7119 Grain Sales (unspaceable) | −0.711 | **−0.007** | **−0.014** |

**It never reacts.** Selecting the fatal card is worth at most 0.008 to the critic and choosing
to spend it for Operations at most 0.014; two of the eight deltas are positive. At the last
decision before DEFCON 1 it still reports −0.40 to −0.75 rather than anything near −1, and in
7115 the losing side sits at −0.05 -- essentially even -- four micro-actions from death.

(7122 is excluded: Missile Envy is a different case. It is neutral, usually safe to play, and
the information that would make it unsafe -- the opponent's highest-Ops card -- is hidden, so
there is no comparable "point of no return" to test.)

Two consequences.

**This is not only a credit-assignment problem.** The critic cannot represent the conjunction, so
sharpening the policy's credit will not by itself teach it -- but that is the argument *for*
windowing the provoked case rather than against it. Inside a blunder window the advantage is
`-1 - v_t`, which never consults the critic, so the window is precisely the mechanism that works
when the critic is blind. Outside one, the -1 has to flow back through a value function that
prices the position at -0.7 and rising, and is absorbed rather than attributed.

**And it raises the value of the auxiliary DEFCON-risk head**, a direct supervised signal for
what `v_win` demonstrably does not encode. Its label is the same `defcon_blunder` flag that
excludes provoked endings, so the one-line label fix is a prerequisite for either.

## Windowing a provoked DEFCON-1 moves exactly the class it targets

`--window-provoked-defcon` credits a provoked DEFCON-1 to the player who played the card, using
the existing turn-scoped blunder window, instead of letting the -1 propagate back as an ordinary
loss. One seed, 80M steps, against `p1_scalar_nofilter` at 80M -- same recipe, filtering off in
both, the window the only difference.

| ending | control | window | delta |
|:---|---:|---:|---:|
| DEFCON-1 total | 49.4% | **27.0%** | −22.4 |
| — provoked | 36.2% | **13.7%** | **−22.5** |
| — self-inflicted | 13.1% | 13.3% | **+0.1** |
| final scoring | 6.9% | **16.7%** | +9.8 |
| 20 VP | 39.6% | 53.9% | +14.3 |

**The self-inflicted share does not move.** That is the result: the intervention targets provoked
endings alone, and provoked endings alone changed, by 62% of their own value, while the
neighbouring class in the same metric family stayed put to a tenth of a point. It is as close to
a placebo control as a training change gets here, and it rules out the reading that the arm just
made every DEFCON-1 rarer by playing more timidly.

Games also got longer -- mean ply 93.4 to 103.1, final scoring 6.9% to 16.7% -- moving toward the
human distribution, where about 30% of games go the distance against this control's 7%.

**The behaviour replicates on a second seed; the strength cost does not, yet.**

| arm | provoked | self | final scoring | USSR | anchor |
|:---|---:|---:|---:|---:|---:|
| control | 36.2% | 13.1% | 6.9% | 47.0% | 80.6% |
| window ...921 | 13.7% | 13.3% | 16.7% | 63.7% | 71.4% |
| window ...922 | **10.7%** | 12.1% | 16.7% | 58.3% | **81.8%** |

Both seeds cut provoked endings by about two thirds and leave the self-inflicted share alone, so
the behavioural result is solid. But the two seeds sit **10.4 anchor points apart**, straddling
the control -- the same pattern the filtering arm produced, and the reason the anchor cannot
settle anything here.

**Both seeds lose, and by more than the first one suggested.** Pooled over sixteen snapshot
pairings each:

| | vs control | Elo |
|:---|---:|---:|
| window ...921 | 1,331/3,200 = 41.6% [39.9, 43.3] | **−59** |
| window ...922 | 1,250/3,200 = 39.1% [37.4, 40.8] | **−77** |
| pooled | 2,581/6,400 = **40.3%** [39.1, 41.5] | **−68** |

So the strength cost is real and two-seed confirmed at about **−68 Elo**.

**And the anchor was wrong about both seeds, in opposite directions.** Seed ...921 read 71.4%
against the control's 80.6% and is 59 Elo weaker; seed ...922 read **81.8%**, slightly *above*
the control, and is **77 Elo weaker** -- the largest single divergence this project has recorded
between the anchor and a pooled head-to-head. An earlier draft of this section argued the anchor
corroborated the 59 Elo to within 8; that was one seed, and it was luck. Take nothing from the
anchor win rate that a head-to-head has not confirmed.

The USSR share rose in both (63.7% and 58.3% against 47.0%), which is a move away from the human
49.9% and is measured over the whole run rather than 500 eval games.

**The likely cause is that the window is far wider than the mistake.** The blunder window is
*turn-scoped* -- `rollout_buffer` pins the blunderer for the whole turn the blunder happened in.
Measured above (*the critic does not see a provoked DEFCON-1 coming*), the fatal card play sits
**3-9 micro-actions** from the loss. A turn is up to seven action rounds. So the window sets the
advantage to `-1 - v_t` across a long stretch of play that was mostly fine, and the policy learns
to avoid far more than the one card choice that was wrong. The rising USSR share is consistent
with that: the arm is not making one behaviour rarer, it is distorting a whole turn's worth of
play.

That reading is testable and the fix is narrow: scope the window to the action round, or to the
segment from the card play to the terminal, instead of the turn. The turn-scoped form was built
for *unprovoked* suicides, where the mistake is the final move and the window's width costs
nothing because the game ends immediately after. For the provoked case the game also ends
immediately -- but the credited stretch reaches backwards over everything else the player did
that turn.

**The over-reach is 5.6x, counted.** On the five generated replays, the turn the game ended in
holds **124** of the losing player's decisions, of which only **22** are at or after the fatal
choice. Every one of the 124 gets `-1 - v_t`. So 82% of the credited decisions had nothing to do
with the loss, and the worst case (7122) pins 30 decisions for a mistake two decisions deep.

**And the critic still does not see it.** Both checkpoints were run over the *same* stored
positions from those replays, so the comparison is paired and any difference is the critic rather
than the games it happened to play. The delta from choosing to spend the card for Operations:

| | control critic | windowing critic |
|:---|---:|---:|
| 7107 Star Wars | −0.023 | **−0.001** |
| 7115 Lone Gunman | −0.026 | **+0.050** |
| 7118 Grain Sales | +0.024 | **−0.002** |
| 7119 Grain Sales | −0.036 | **+0.008** |

No more reactive than the control, and in two of four less. That is not a surprise on reflection:
a window's advantage is `-1 - v_t` *by construction*, so it teaches the policy while routing
around the value function entirely. The behaviour moved and the understanding did not.

Which is the best available explanation for the Elo: the policy learned a blunt avoidance over a
whole turn's play rather than the one conjunction that is actually fatal, because nothing in this
arm taught it the conjunction. It also makes the auxiliary DEFCON-risk head the natural next step
rather than a narrower window alone -- the head is the only piece on the table that would make
the network *represent* the danger instead of avoiding a region of the game.

**The opening got better, so that is not where the Elo went.** The setup probe on the same two
checkpoints, 1,500 games each:

| target | control | window |
|:---|---:|---:|
| USSR Poland ≥ 3 | 84.8% | **97.9%** |
| US Italy ≥ 2 | 54.6% | **82.3%** |
| US Iran ≥ 2 | 1.2% | **46.0%** |
| US West Germany ≥ 4 | 0.0% | 0.0% |

Three of the four targets improve substantially and none regresses, which localises the loss to
mid and late play -- exactly where a DEFCON-2 turn lives, and so consistent with the over-reach.
(Both arms place West Germany ≥ 4 in 0.0% of games. That is a property of this 80M lineage, not
of the window: H2 at 240M manages 73.2%. Do not read it across lineages or budgets.)

**Verdict: do not adopt as it stands.** The behavioural target is reachable -- 13.7% against the
human 11.7% is the closest this project has come -- but not at this price, and the next attempt
should narrow the window before anything else.

> **Naming, and one thing the old names hid.** Runs are named `<engine><attempt>-<steps>` --
> see [`../method/run_nomenclature.md`](../method/run_nomenclature.md). H2 is **E2**: it predates the Aldrich Ames
> and Star Wars mandatory-choice fixes, so it was trained on a different game from every P1 arm,
> which are **E3**. Every "against H2" figure below is therefore a *cross-engine* comparison,
> with the reference playing a game it never trained on. The handicap is small -- two rare cards
> -- but it runs against the reference, so it flatters the E3 arms slightly. In the new naming
> the control at 80M is `E3-03-80M` and the reference is `E2-02-480M`.

## Rate arms against H2 @480M, not against HeuristicBot

Every P1 and windowing arm was re-rated in **one pool** with H2 @480M, four late snapshots each,
50 games a side. Elo is not comparable across tournaments, so a common scale requires a common
pool.

| arm | vs H2 @480M | Elo gap | vs the 80M control | Elo |
|:---|---:|---:|---:|---:|
| control 80M | 26.6% | −176 | — | — |
| control 160M | 39.8% | −72 | 63.7% | +98 |
| categorical 80M | 21.4% | −226 | **46.0%** [43.6, 48.4] | **−28** |
| filter 921 / 922 80M | 26.0% / 31.1% | −182 / −138 | 52.9% / 53.6% | +20 / +25 |
| filter 921 / 922 160M | 41.9% / 41.3% | −57 / −61 | 65.4% / 67.2% | +111 / +125 |
| window 921 / 922 80M | 19.7% / 21.3% | −244 / −227 | 41.8% / 36.1% | −58 / −99 |
| H2 @480M | — | — | 73.1% | +174 |

**The weak anchor inverts orderings that the strong one gets right.** Against HeuristicBot,
windowing seed ...922 scored **81.8%** against the control's 80.6% — above it. Against H2 @480M
it scores 21.3% against 26.6% — correctly below. The head-to-head against the control always knew
(−99 Elo); the point is that the *anchor rate*, which is what a training run logs live and what a
monitor shows, is not merely noisy but can rank a materially weaker arm first. Rate against
H2 @480M.

**And it changed a verdict.** The categorical head was recorded as a null on 80.0% against
HeuristicBot versus the control's 80.6%. Head-to-head in this pool it is **−28 Elo** with the
interval excluding 50%: mildly harmful, not neutral. One seed, so the magnitude is soft, but the
sign is no longer in doubt.

That arm had never been rated at all, and could not have been: `NeuralAgent.from_checkpoint`
built a scalar `ColdWarNetV2` unconditionally, so a categorical checkpoint failed to load with
`value_dist_head` unexpected and `val_vp_head` missing. Every number previously reported for it
came from the training loop's own evaluation, which builds the model itself and so never hit the
path the tournament and every probe use. The head is now detected by weight name.

## The model cannot tell 86% of cards apart

Every card is described to the network by 8 slots saying where it is, **5 properties -- Ops,
side relative to the viewer, era, one-time, is-scoring** (`engine/src/observation.cpp:213-219`)
-- and a flag saying this decision is about it. Nothing about what the card *does*: no target,
no effect class, no "this hands the opponent Operations". Identity exists only as position in
the 110x14 block, and v2's card branch applies one shared MLP per token then mean+max pools, so
position is discarded.

Put every card in one hand, so only its own properties can separate it:

**110 cards collapse to 46 distinct signatures. 95 of them -- 86% -- share a vector with another
card.** Groups run to six. The trunk difference between a hand holding Marshall Plan and the same
hand holding US/Japan Pact is **5e-6**.

Every card this project's failures turn on is ambiguous:

| card | indistinguishable from |
|:---|:---|
| Grain Sales (hands the US your Operations) | Colonial Rear Guards, **The Voice of America** |
| Tear Down this Wall (a free US coup in Europe) | Iron Lady, North Sea Oil, **Chernobyl**, An Evil Empire, AWACS |
| Star Wars (takes a card out of the discard) | Reagan Bombs Libya, Solidarity |
| Junta | **Missile Envy**, Latin American Death Squads, One Small Step |
| Cuban Missile Crisis (a coup here loses the game) | **SALT Negotiations** |
| Olympic Games (the boycott ends the game) | Indo-Pakistani War |
| **Nasser** | **Blockade, Romanian Abdication** |
| UN Intervention | *unique* |

So the Nasser question -- can it know the card targets Egypt rather than some other 2-stability
Middle Eastern battleground -- does not get as far as Egypt. It cannot tell Nasser from Blockade.

**This is the common cause behind most of this programme.** The DEFCON blunder rate that never
moves in 80M steps, the UN Intervention misrouting, the Olympic Games rule, coups under Cuban
Missile Crisis: each needs the model to know *which card* it holds, and it does not. Meanwhile the
things it does competently -- spacing a card whose Ops clear the box, the setup probe's targets --
are decidable from **properties alone**. Property-level competence with identity-level blindness
fits every measurement in this section.

It also retires the reading of the `metrics.md` §21.1 linear probe. AUC 0.953 on "the card being
committed is one I must not play" cannot have been reading card identity, because there is none;
it was reading the board half of the conjunction -- DEFCON 2 plus an exposed battleground plus the
card's side and Ops -- which gates most of the danger set and ranks well without ever separating
Grain Sales from The Voice of America.

`--identity-dim` adds a learned embedding indexed by position, 5,152 parameters at width 16, and
is model-side: the observation is untouched. It makes the distinction *learnable*. It does not
make it known -- there is still no card-to-effect or card-to-target encoding, so the association
between a card and what it does must come from games in which it was played.

## The structured backbone is worth ~110 Elo, and the anchor said the opposite

An MLP control -- the graph convolution, per-card encoder and cross-attention replaced by two
dense layers over the flat observation, same heads, same recipe, 2.6x the parameters, 4.2x the
throughput -- scores **34.5%** and **35.2%** against the v2 control over 3,200 games each:
**−112 and −106 Elo**, two seeds agreeing.

So structure is worth about 110 Elo, and it is not capacity: the control has *more* parameters
and loses. Note the MLP keeps card identity for free, by position, and still loses -- identity is
not what makes v2 good, and the two findings are complementary rather than competing.

**The anchor read the MLP at 88.2% against the control's 80.6%** -- better, by a wide margin.
That is the third inversion in this session:

| arm | anchor | pooled head-to-head |
|:---|---:|---:|
| windowing seed ...922 | 81.8% (above control's 80.6%) | **−77 Elo** |
| MLP backbone | 88.2% (above control's 80.6%) | **−110 Elo** |
| filtering | ordering flips between 80M and 160M | +24 at both |

The bias is one-directional: the anchor **overrates arms that are weaker**. HeuristicBot is a
fixed script, and a differently-trained policy can exploit its habits without being stronger.
This is systematic, not noise, and it means no live training metric can rank arms. Rate against
H2 @480M, pooled over snapshots.

## Identity embeddings are worth ~115 Elo

`--identity-dim 16` adds a learned embedding indexed by position to each card and country token.
5,152 parameters, model-side, observation untouched. Two seeds, 80M steps, pooled over sixteen
snapshot pairings:

| | vs E3-01 control 80M | vs E2-01 H2 80M |
|:---|---:|---:|
| **E3-13** (seed 20260921) | 65.3% [63.6, 66.9] → **+110** | 59.7% → +68 |
| **E3-14** (seed 20260922) | 66.6% [65.0, 68.2] → **+120** | 61.2% → +79 |
| E3-01 control | — | 42.0% → −56 |

Five times the effect of advantage filtering, the largest single change measured in this project,
and the two seeds agree to 10 Elo. It also turns the control's deficit against the E2 baseline at
matched budget into a substantial lead.

The size is what the diagnosis predicted rather than a surprise (*the model cannot tell 86% of
cards apart*, above): 86% of cards share a
feature vector, so most of what a player knows about its own hand was unavailable, and the MLP
control (*the structured backbone is worth ~110 Elo*, above) had already priced the backbone at
~110 Elo. Behaviour moves with it -- DEFCON-1
endings 49.4% → 32.9-37.3%, provoked 36.2% → 20.9-22.7%, final scoring 6.9% → 13.3-16.7%,
explained variance +0.841 → +0.862/+0.889.

**One thing goes the wrong way in both seeds.** The USSR win rate rises 47.0% → 51.7% and 57.6%,
against a human 49.9%; the control was the most balanced arm measured. Windowing pushed the same
direction. Not disqualifying at this size of gain, but it should be tracked at longer budgets
rather than assumed to wash out.

**Adopt, and re-baseline.** Every E3 comparison to date used a control without identity
embeddings, so the recipe line moves and the control has to be re-run with them before the next
factor is screened.

## What the trunk actually encodes, read off linearly

`ai/eval/state_readout.py` fits a **linear** probe from the frozen 512-float trunk to facts about
the position. Linear on purpose: if a fact is not linearly available, no head can condition on it
either, and a deeper probe would only show it is recoverable in principle.

| arm | board R² | hand AUC | **which twin** (chance 0.36) | tracks R² |
|:---|---:|---:|---:|---:|
| E3-01-21-80M control | 0.234 | 0.861 | **0.512** | 0.781 |
| E3-09-21-80M mlp | **0.396** | 0.884 | **0.717** | 0.630 |
| E3-10-21-80M identity | 0.230 | 0.864 | **0.628** | 0.739 |
| E2-02-21-480M | 0.259 | 0.883 | **0.546** | 0.856 |

**The aggregate hand AUC answers nothing.** Every arm scores ~0.86, including ones that provably
cannot identify a card, because most cards *are* separable by properties: "a 3-Ops US early-war
card" narrows 110 to about six and lifts AUC far above chance without identity.

**The within-collision-group test is the real one.** Restricted to positions where exactly one
member of a same-feature group is in hand, the question is which -- and properties cannot help.
Identity embeddings take it from 0.512 to 0.628 and the positional MLP to 0.717, against a chance
rate of 0.36. That is the mechanism behind identity embeddings' +110 Elo (above), measured
directly rather than inferred. (Caveat: the non-identity arms sit above chance because the group's
other members are visibly in the discard or deck in a real position, which is a cue that is not
identity. It is shared by every arm, so the ordering holds.)

**And the unexpected result: the trunk barely encodes per-country influence.** Board R² is
0.23-0.40 everywhere. The board branch pools mean+max over 84 country tokens, so which country
holds what is largely gone by the time any head sees it -- the same destruction as for cards, and
the MLP scores highest (0.396) for the same reason it wins the twin test, by reading positionally.
This bears directly on the empty-battleground failure, and it is untouched by identity embeddings,
which address the card side only.

Tracks read well everywhere (DEFCON 0.85-0.92, victory points 0.93-0.97), so nothing is wrong
with the trunk in general -- it is specifically per-entity information that pooling removes.

## Identity embeddings hold ~110 Elo across a doubling

Matched budget, same engine, two seeds, pooled over sixteen snapshot pairings against
`E3-01-21-160M`. The first comparison in this work that needs no caveat about engine or budget.

| arm | vs control at 160M | at 80M |
|:---|---:|---:|
| E3-10-21 identity | 63.1% [61.4, 64.7] → **+93** | +110 |
| E3-10-22 identity | 65.8% [64.1, 67.4] → **+114** | +120 |
| E3-07-21 filter | 54.6% [52.9, 56.3] → +32 | +20 |
| identity vs filter, head to head | **58.8% [57.0, 60.4] → +61** | — |

Mean +115 at 80M and +104 at 160M, inside the between-seed spread. Under the log-linear law
([`../method/references.md`](../method/references.md) §7) that is an **intercept shift**: identity
reaches a given strength sooner and does not change the rate, so it is worth about 110 Elo at any
budget rather than compounding. Filtering has the same shape at a fifth of the size (+20, +24, +32
across three measurements).

**Adopt, and re-baseline.** Every E3 comparison to date used a control that could not identify
its own cards, so `--identity-dim 16` becomes part of the recipe line and the control has to be
re-run with it before the next factor is screened. Filtering's +32 was measured on top of a
blind policy and is not established on top of identity.

**A caution that applies to this section's behavioural numbers generally.** The control's own
DEFCON-1 share falls 49.4% → 32.4% between 80M and 240M with no intervention at all, and its USSR
share rises 47.0% → 65.4%. At 160M the identity arms' behavioural advantage is much smaller than
it looked at 80M -- DEFCON-1 33.9% against 38.7%, final scoring level, USSR imbalance now *worse*
than the control's. So part of what the windowing and identity entries above credited to
interventions is what longer training does anyway. The Elo comparisons were all at matched steps
and stand; the behavioural ones need a matched-budget control before they mean what they appear
to.

## Dropping the 1,364 constant observation slots buys nothing, and could not have

1,364 of the 3,824 observation floats never change value in any position -- per-country and
per-card properties (stability, region membership, Ops value, era) that the structured encoders of
the v2 backbone need, because a graph convolution and a per-card MLP see a *token* and have no
other way to know which country or card it is. An MLP reading a flat vector does not: it recovers
identity from the offset. So the question was whether those slots are dead weight for E3-09.

E3-11 is E3-09 with them masked out: 2,460 inputs, 6.6M parameters against 8.0M, two seeds at 80M.

| | vs E3-09-21-080M | vs E3-01-21-080M control |
|:---|---:|---:|
| E3-11-21-080M | 44.3% [42.6, 46.0] → **−40** | 35.1% [33.4, 36.7] → −107 |
| E3-11-22-080M | 50.0% [48.3, 51.7] → **−0** | 39.0% [37.3, 40.7] → −78 |

**Not adopted**, and the mean of −20 is inside a seed spread of 40. Throughput was unchanged
(64.4k against 65.5k steps/s), which is the practical answer on its own: the saving was supposed
to be speed and there was none.

The stronger statement is that a difference here *cannot* be information. A constant input
contributes `w·c` to every unit, which the bias already spans, so the two networks have the same
function class and the masked one loses nothing it could have used. Any real residual would be
optimisation-side: `nn.Linear` initialises `U(±1/√fan_in)`, and fan_in moving 3,824 → 2,460
rescales the init of *every* first-layer weight, the varying ones included. That is a reason to
expect small noise, not a reason to expect a loss.

**The mask was verified rather than assumed**, because the first attempt at it was wrong. Across
16,800 positions, **0 of the 1,364 claimed-constant dimensions takes a second value** -- counted
by distinct values, not by standard deviation, which is what had previously mislabelled a slot
that varies over a tiny range as constant.

## One Elo scale for E3, and identity is worth a doubling of compute

Everything below is on **one scale, anchored at `E3-01-21-080M` = 0**, from 228,000 games in three
pools of four late snapshots per arm-budget. Both of the first two pools contain the anchor, so
nothing here is stitched across pools through a third model.

| arm | vs anchor | 95% CI | **Elo** | BT | snapshot spread |
|:---|---:|:---:|---:|---:|---:|
| `E3-01-21-080M` control | — | — | **0** | 0 | 30 |
| `E3-01-21-160M` control | 64.9% | [63.7, 66.0] | **+106** | +99 | 17 |
| `E3-01-21-240M` control | 67.5% | [66.4, 68.7] | **+127** | +135 | 50 |
| `E3-10-21-080M` identity | 65.4% | [64.2, 66.6] | **+111** | +118 | 67 |
| `E3-10-22-080M` identity | 67.2% | [66.1, 68.4] | **+125** | +118 | 45 |
| `E3-10-21-160M` identity | 75.2% | [74.1, 76.3] | **+193** | +194 | 54 |
| `E3-10-22-160M` identity | 75.7% | [74.6, 76.7] | **+197** | +203 | 33 |
| `E3-11-21-080M` mlp-static | 34.3% | [33.2, 35.5] | **−113** | −115 | 48 |
| `E3-11-22-080M` mlp-static | 38.2% | [37.1, 39.4] | **−83** | −76 | 67 |

`BT` is the Bradley-Terry fit over the whole pool, recentred on the anchor's four snapshots. It
agrees with the pooled win rate within 8 Elo everywhere, so the pool is transitive and neither
column is doing hidden work.

**Identity at 80M is level with the control at 160M.** Measured directly rather than inferred
through the anchor -- `E3-10-21-080M` scores **52.4% [51.2, 53.6]** and `E3-10-22-080M` **49.9%
[48.7, 51.1]** against `E3-01-21-160M`. So 5,152 embedding parameters buy what doubling the budget
buys, which is the sharpest form of the intercept-shift claim in *identity embeddings hold ~110
Elo across a doubling* and the reason identity is in the recipe rather than on the list of things
to try.

It does not buy the *next* doubling as well: the same pool puts `E3-01-21-240M` at **+38** over
the 160M control, above identity@80M.

**The control's own returns are collapsing.** +106 for the first doubling, then +38 for the 1.5x
from 160M to 240M -- 36 per doubling against 106. Two consequences: an arm measured only at 80M
is measured on the steep part of the curve, and the 240M control is a much harder reference than
its step count suggests.

**Two cross-checks, both passed.** `E3-01-21-240M` vs `E3-01-21-160M` reads **+38 in two
independent pools**. And identity-vs-control at matched budget reads **+92 / +109** at 160M here
against **+93 / +114** measured separately above, and **+111 / +125** at 80M against
**+110 / +120** -- four figures reproducing across a different pool composition.

**Elo from a pooled win rate is not additive, and the matched pair is the number to quote.** The
160M control is +106 on the anchor's scale and the 240M control +127, but played against each
other directly the gap is +38, not +21. Nothing is wrong: converting each pooled win rate
separately compresses differences between two arms that are both far from the anchor. For a
specific comparison, re-anchor and measure that pair.

**The snapshot spread column is why *a single snapshot cannot measure an effect smaller than a
run's oscillation* ([`variance_and_noise.md`](variance_and_noise.md)) exists.** Four snapshots of
one arm-budget span 17 to 80 Elo. Every effect in the table except identity's is smaller than the
largest of those spreads, so a single-snapshot version of this table would have been noise dressed
as a result.

## What the 160M arms actually do, and where identity's Elo does *not* show up

Three probes at temperature 0.1 over **all four** late snapshots of each arm-budget -- the
denominator that matters is the run's own oscillation, and the numbers below show why. Spreads
are max − min over the four snapshots.

| share of all games | ctrl 160M | ctrl 240M | id·21 160M | id·22 160M |
|:---|---:|---:|---:|---:|
| **DEFCON-1 total** | 35.3% ±5.7 | 28.5% ±0.7 | 28.2% ±8.5 | 22.6% ±6.5 |
| own goal | 3.9 ±3.3 | 2.8 ±2.5 | 3.3 ±2.0 | 3.5 ±3.0 |
| bad bet | 0.6 ±0.5 | 0.6 ±1.3 | 0.6 ±0.5 | 0.3 ±1.0 |
| forced trap | 10.6 ±5.0 | 7.4 ±3.3 | 6.8 ±7.0 | 6.6 ±4.2 |
| **unforced trap** | 15.0 ±2.8 | 12.4 ±4.5 | 12.7 ±3.0 | 8.2 ±3.0 |
| unclassified | 1.8 ±0.5 | 2.2 ±2.0 | 1.1 ±1.7 | 1.8 ±1.7 |
| in headline | 3.5 ±0.5 | 3.1 ±2.2 | 3.8 ±3.3 | 2.2 ±1.7 |

| | ctrl 160M | ctrl 240M | id·21 160M | id·22 160M |
|:---|---:|---:|---:|---:|
| empty battlegrounds, turn 5 (of 29) | 11.45 ±0.67 | 11.16 ±0.71 | 10.85 ±0.78 | 10.65 ±0.41 |
| empty battlegrounds, turn 8 | 7.26 ±0.63 | 6.94 ±0.70 | 6.67 ±1.21 | 6.46 ±0.90 |
| uncontrolled battlegrounds, turn 8 | 15.58 ±0.98 | 15.61 ±1.37 | 15.35 ±1.46 | 14.84 ±1.02 |
| mean final turn | 7.13 ±0.77 | 6.94 ±0.64 | 7.30 ±0.72 | 7.22 ±0.46 |
| **blunder rate, pooled** | 2.0% ±0.4 | 1.9% ±1.1 | **3.0% ±1.0** | **2.5% ±1.2** |
| · spaced own or neutral | 9.7% ±3.7 | 7.9% ±4.6 | **15.9% ±8.3** | **15.7% ±13.1** |
| · DEFCON suicide with an alternative | 1.8% ±0.8 | 1.9% ±1.2 | 2.3% ±1.1 | 1.4% ±0.7 |
| · Olympic Games at DEFCON 2 | 0.3% ±0.5 | 0.3% ±0.5 | 0.2% ±0.3 | 0.2% ±0.2 |

**Most of this is not resolvable, and saying so is the finding.** Both identity seeds end fewer
games at DEFCON 1 than the control at the same budget, but by 7.1 and 12.7 points against
snapshot spreads of 5.7 to 8.5, and the two identity seeds differ from each other by 5.6. The
direction is consistent; the magnitude is not established. `E3-01` has only one seed at 160M, so
there is no control seed pair to compare that spread against -- a gap worth closing before any of
these behavioural numbers carries an argument.

**The 240M control matches identity@160M on almost every behavioural line** -- 28.5% against 28.2%
DEFCON-1, 6.94 against 6.67 empty battlegrounds. The identity-across-a-doubling entry warned that
part of what was credited to interventions is what longer training does anyway; at matched
*behaviour* rather than matched steps, that is exactly what this shows. Identity's 92-109 Elo over
the 160M control is real (§21.10) and is not visible in these aggregates.

**Identity blunders more, not less.** Pooled rate 3.0% and 2.5% against the control's 2.0%,
driven almost entirely by `spaced_own_or_neutral` -- 15.9% and 15.7% against 9.7%, the same
direction in both seeds and roughly a doubling. The other three rules are level. So the arm that
gained ~100 Elo also spends its own and neutral cards on the space race considerably more often.
Either the rule is mis-specified for a policy that can now tell its cards apart -- spacing a
*specific* low-value own card may be correct where the rule reads only "own or neutral" -- or
identity bought its Elo somewhere else and paid here. This is worth resolving before the rule is
used to judge another arm, and it is the one place in this table where the spread does not
swallow the effect.

#### Which battlegrounds, not how many

The per-country table is where identity is legible, because "8 of 29 empty" is a claim about
*which* eight. Late-game empty rate, mean of four snapshots:

| battleground | ctrl 160M | ctrl 240M | id·21 160M | id·22 160M |
|:---|---:|---:|---:|---:|
| **India** | **98.3%** | 87.7% | **54.4%** | **55.2%** |
| Algeria | 88.3% | 91.4% | 85.5% | 83.9% |
| Saudi Arabia | 82.1% | 69.7% | 93.9% | 78.6% |
| Libya | 79.2% | 42.7% | 59.7% | 56.7% |
| Argentina | 68.7% | 70.8% | 41.0% | 67.9% |
| Nigeria | 51.5% | 30.0% | 35.9% | 35.6% |
| Brazil | 43.6% | 49.5% | 30.9% | 15.7% |
| Cuba | 31.0% | 27.5% | 53.0% | 42.9% |
| **Pakistan** | **23.6%** | 14.1% | **2.6%** | **7.6%** |
| **West Germany** | **15.2%** | 29.3% | **36.8%** | **35.3%** |
| France | 4.5% | 15.4% | 13.2% | 15.6% |
| Italy | 4.5% | 2.5% | 12.1% | 14.2% |

**India stops being invisible.** 98.3% empty in the control -- effectively never touched, the
finding `position_diagnostics` has reported since it was written -- against 54.4% and 55.2% in
both identity seeds. Pakistan moves with it, 23.6% to 2.6% and 7.6%. Two seeds agreeing on a
40-point move is not snapshot noise, and the 240M control only reaches 87.7%, so this is not
simply what more training does.

**It is a reallocation, not an improvement.** West Germany goes the other way, 15.2% to ~36%, and
France and Italy roughly triple. Identity did not learn to contest more of the board; it learned
to contest a *different* part of it. Whether trading Western Europe for South Asia is right is a
question about the game, not about the probe -- but a 5-point battleground region where three
countries are now emptier deserves an answer before this is called a win.

**Four battlegrounds are still untouched in every arm**: Algeria (84-91%), Saudi Arabia (70-94%),
Libya (43-79%) and, in three of four arms, Argentina. Identity did not help there. The trunk
read-out above found per-country influence is the thing the representation destroys (R² 0.23-0.40,
unchanged by identity), and this is the behavioural face of the same gap.

## Exact influence and control, read off the trunk per country

The trunk read-out above reported per-country influence at R^2 0.23-0.40 and called it the thing
the representation destroys. R^2 is the wrong scale for the question actually being asked -- *does
the trunk know the position* -- so this measures the two numbers that answer it: the share of
held-out positions where a linear read-out names the **exact** influence, and the share where it
calls **control** correctly, each against the best-constant baseline.

**Three methodology notes. All of them changed the answer more than any arm difference did, and
the first two were caught only after being written up as findings.**

*The estimator, twice.* Rounding a least-squares fit scores **below** the constant baseline -- 61%
against 69% on battlegrounds -- because influence is 0 in most countries most of the time with an
occasional 3 or 4, so the MSE-optimal fit sits between the two and rounds to neither. Replacing it
with least-squares onto one-hot *class* columns looked like the fix and was not: that estimator
**masks intermediate classes** once there are three or more (ESL 4.2) and shrinks rare ones out of
the argmax. Handed a *noiseless* `influence / 10` column -- board slot 0, verbatim -- it recovered
79% against a 72% baseline, closing **24% of the gap with the answer in front of it**. An earlier
version of this section reported that ceiling as "the trunk closes about a fifth".

The probe is now **multinomial logistic regression**, still linear, and it is gated: any estimator
used for this question must first clear `tests/training/test_state_readout_probe.py`, which feeds
it the noiseless column and requires near-perfect recovery *and* requires pure noise to score
exactly the constant baseline. The current one gets 100%, 99.3% buried in 25 noise columns, and
baseline on noise.

*The aggregate.* A mean of per-country ratios is not a usable summary. India has 4% headroom, so
`(acc - base) / 0.04` turns two points of probe noise into -431%, and the average over countries
is then decided by whichever near-constant country wobbled. Every headline below is
`sum(acc - base) / sum(headroom)`, which weights each country by what it had to give.

*The split.* The probe held out whole environments rather than permuting positions, since
successive samples from one env are the same game six steps apart. It was worth doing and it
barely mattered: battleground exact-match 62.0% held-out against 63.2% permuted, about a point.
Kept because it is free, reported because the concern was real and the effect was not.

#### What the trunk actually recovers

Raw accuracy flatters a country that is empty in 97% of positions, so the headline is the share of
the **recoverable gap** closed: 1.0 is a perfect read-out and 0.0 is a representation adding
nothing a constant already gave. Negative means the probe fits noise.

| | ctrl 80M | ctrl 160M | ctrl 240M | id 80M | id 160M |
|:---|---:|---:|---:|---:|---:|
| **influence, gap closed (battlegrounds)** | 4.7% | 7.7% | **10.3%** | 3.1% | 10.0% |
| influence, gap closed (all 84) | 4.0% | 2.0% | 4.8% | 1.7% | 3.3% |
| **control, gap closed (battlegrounds)** | 24.0% | 40.8% | **43.4%** | 37.2% | 43.0% |
| control, gap closed (all 84) | 29.7% | 35.4% | 35.3% | 34.0% | 36.3% |
| exact influence, BG (raw accuracy) | 70.3% | 69.5% | 71.3% | 67.9% | 69.9% |
| · best-constant baseline | 68.8% | 66.9% | 68.0% | 66.8% | 66.5% |
| control correct, BG (raw accuracy) | 88.5% | 90.5% | 91.1% | 89.4% | 90.0% |
| · best-constant baseline | 85.5% | 83.8% | 84.7% | 82.3% | 82.4% |

**Exact influence is almost absent from the trunk.** Four to ten percent of the recoverable gap on
battlegrounds. The raw accuracy column is what makes this concrete: 71.3% against a 68.0% constant
-- three points for 512 floats of representation.

**Control is a different story: a quarter to well over 40%, and it improves sharply with compute**
where influence barely moves. 24.0% → 43.4% across the control's own run against 4.7% → 10.3%. The
trunk is learning *who holds what* and not *by how much* -- the right priority for scoring, and
the wrong one for knowing whether a coup or a placement flips a country.

**Identity helps control, not influence, and most at low budget.** 37.2% against 24.0% at 80M, a
13-point gain; by 160M the control has caught up (43.0 against 40.8). Influence is unmoved at both.
Identity embeddings name the *country*; they do not carry its *number*.

#### Where it is lost: pooling, measured

The numbers above say the trunk does not have it; they do not say where it went. Tapping one
rollout at four points localises it. `raw` is country i's own 26 observation floats -- slot 0 is
literally `my_influence / 10`, so it is also the check that the probe works at all.

Battlegrounds, share of the influence gap closed, whole games held out, **penalty selected per
stage on a validation split**:

| stage | ctrl 240M | id 160M |
|:---|---:|---:|
| `raw` -- country i's 26 observation floats | **96.8%** | 98.5% |
| `gconv1` -- after one graph convolution | 62.9% | 64.5% |
| `gconv2` -- after the second, still pre-pooling | **65.8%** | 65.1% |
| `trunk` -- the 512 floats every head reads | **−0.5%** | 6.9% |

*Tuning the penalty per stage is not a detail.* A single `l2`, fitted to a synthetic one-feature
problem, held the `raw` rung to 85% -- with the answer in slot 0. The same constant is far too
weak for 512 features and far too strong for one, and a sweep moved the trunk rung between 8.9%
and 28.0% depending only on that choice. Each rung now gets its own penalty, chosen on games held
out of training and never on the test games. `raw` lands at 96.8%, and at exactly 100% for most
individual countries, which is what makes the rest of the column readable.

**The trunk holds essentially nothing about exact influence.** Not "about a fifth", which was the
broken probe, and not 12%, which was the under-tuned one: **−0.5% and 6.9%**, at or below what a
constant gives.

**Pooling is where it goes: 66% → 0%.** The pre-pooling token holds two thirds of the recoverable
gap and the 512 floats hold none of it. Mean- and max-pooling over 84 countries is the step that
destroys the board.

**The graph convolution costs a third before that**, 96.8% → 62.9%, and the second layer adds
nothing back, so an attention read-out over `gconv2` tokens caps near 66% rather than 97%.

*Two controls, because "the trunk holds nothing" is the kind of claim a broken probe also makes.*
On the same checkpoint and the same pipeline the trunk gives **tracks R^2 0.847** and **hand AUC
0.874** -- so the representation and the probe are both working, and it is specifically
per-country influence that is absent. And the graph loss tracks node **degree**: correlation
+0.38 with the raw-to-`gconv1` loss, and +0.43 between `gconv1` recovery and the self-loop weight
`1/(deg+1)`.

Per country, the same ladder (control 240M, headroom in brackets):

| battleground | raw | gconv1 | gconv2 | trunk |
|:---|---:|---:|---:|---:|
| Japan (49%) | 99% | 92% | 97% | **79%** |
| South Africa (52%) | 98% | 86% | 93% | **71%** |
| North Korea (67%) | 76% | 65% | 64% | 42% |
| Poland (47%) | 86% | 50% | 45% | 35% |
| Iraq (55%) | 90% | 65% | 67% | 32% |
| South Korea (66%) | 85% | 58% | 58% | 14% |
| Iran (48%) | 80% | 41% | 45% | 8% |
| Israel (38%) | 100% | 33% | 57% | 4% |
| Italy (52%) | 83% | 40% | 45% | **−13%** |
| Egypt (34%) | 82% | 64% | 63% | **−17%** |
| Pakistan (38%) | 66% | 45% | 45% | **−22%** |
| France (33%) | 81% | 63% | 46% | **−41%** |

Only Japan and South Africa survive pooling intact. France, Pakistan, Egypt and Italy are
recoverable from their own token at 45-63% and are **worse than a constant** in the trunk.

#### The graph convolution is the wrong operator for this quantity

`GraphConvLayer` is `A_norm @ (W x) + b`, with `A_norm = D^-1/2 (A + I) D^-1/2` -- textbook GCN.
**One weight matrix is applied to a country and to its neighbours alike**, and the only thing
keeping a country's own value is the self-loop, whose weight is `1/(deg+1)`. So a country's own
influence is attenuated in proportion to how many neighbours it has, and mixed with theirs
through a transform that cannot tell the two apart. That is a low-pass filter, and exact
per-country influence is the high-frequency part of the signal.

The prediction that follows is that survival through `gconv1` should track degree, and it does:

| country | neighbours | self-loop weight | `raw` | `gconv1` |
|:---|---:|---:|---:|---:|
| Australia | 1 | 0.50 | 100% | **100%** |
| Canada | 1 | 0.50 | 99% | **99%** |
| Cuba | 2 | 0.33 | 95% | 95% |
| Panama | 2 | 0.33 | 94% | 96% |
| South Africa | 2 | 0.33 | 100% | 98% |
| Egypt | 3 | 0.25 | 99% | 81% |
| East Germany | 4 | 0.20 | 100% | 66% |
| Israel | 4 | 0.20 | 98% | 60% |
| France | 5 | 0.17 | 98% | **60%** |
| West Germany | 5 | 0.17 | 96% | **50%** |
| Italy | 5 | 0.17 | 100% | **34%** |

Degree-1 countries pass through untouched; the five-neighbour countries of Western Europe lose
half to two thirds. The correlation is +0.38 over the 33 countries with real headroom and is not
the whole story -- Austria (degree 4) loses 86% and Vietnam (degree 2) loses 70%, so traffic
matters too -- but the mechanism is visible and it is the one the operator implies.

**This is fixable without abandoning the map.** The defect is not that adjacency is modelled, it
is that self and neighbour share a transform. `h_i = W_self x_i + W_neigh * mean_j(x_j)`
(GraphSAGE-style), or simply a residual `h = GCN(x) + W x`, gives the network the option of
keeping a country's own value at full strength and costs one more weight matrix per layer.
Adjacency is genuinely part of this game -- placement legality, realignment, superpower adjacency
-- so the relation is worth keeping; what is wrong is being forced to average across it.

#### It tracks stability

Correlating gap-closed against country properties over the battlegrounds with at least 15%
headroom, **stability is the one that lines up** -- r = **+0.63**, monotone at every step:

| stability | mean gap closed | battlegrounds |
|---:|---:|:---|
| 1 | **−30.7%** | Angola, Nigeria, Zaire |
| 2 | **−11.9%** | Brazil, Egypt, Iran, Italy, Mexico, Pakistan, Panama, Thailand, Venezuela |
| 3 | **+17.0%** | Cuba, East Germany, France, Iraq, North Korea, Poland, South Africa, South Korea |
| 4 | **+23.3%** | Israel, Japan, West Germany |

By region: Asia +18.5%, Europe +1.1%, Middle East +0.6%, Africa −6.8%, Central America −7.9%,
South America −21.3%.

The obvious reading -- low-stability countries change hands more, so they are harder to track --
is contradicted by the other correlation in the same fit: gap-closed rises with *headroom*
(r = +0.72), so the countries whose influence varies most are read **better**, not worse, as a
share of what is there. Volatility alone does not explain it.

**A lead, not a conclusion.** n is 23 after the headroom filter, and stability is confounded with
region and with how often a country is contested at all -- and now also with **degree**, which
has a mechanism behind it where stability does not. Western Europe is both high-stability and
high-degree, so the two hypotheses are not separated by this data.

(An earlier version of this note said adjacency could not be tested because the country info
exposed no adjacency field. It does: the key is `neighbors`, not `adjacent`.)

#### What this licenses

An attention block reading the pre-pooling tokens has something real to find -- five times what
the trunk carries -- so the architecture proposal is aimed at the measured defect rather than a
guessed one. Two things the ladder adds to it:

1. **The tokens are already damaged.** Attention over `gconv2` caps near 61%. A skip from the raw
   26 per-country floats into the attention keys and values, or a residual around the graph
   layers, is what recovers the other quarter -- and it is cheap.
2. **It predicts what should change.** If the block works, recoverability at the decision point
   should move from ~12% toward 61%, and toward 85% with the skip. That makes the arm
   falsifiable by something other than whether Elo happened to go up.


## Fixing the graph layer works; attention-pooling into the trunk does not

Two architecture changes, both aimed at the pooling defect above, both with their predicted
effect written down in [`../method/run_nomenclature.md`](../method/run_nomenclature.md) before
the runs. One prediction held exactly and the other failed, which is
the more useful of the two outcomes.

* **E3-12** (`--self-transform`) gives each graph layer a second weight matrix applied to the node
  itself, so a country need not be averaged with its neighbours.
* **E3-13** (`--self-transform --attn-readout 64`) adds an attention read-out after the residual
  trunk: the state vector queries the 84 country and 110 card tokens, each concatenated with its
  raw observation slots, and the result is folded back into the trunk.

Share of the recoverable exact-influence gap, battlegrounds, 80M, seed 21, games held out:

| stage | E3-10 control | E3-12 self-transform | E3-13 + read-out |
|:---|---:|---:|---:|
| `raw` | 98.0% | 98.1% | 98.7% |
| `gconv1` | 60.1% | **94.3%** | **93.8%** |
| `gconv2` | 63.5% | **89.2%** | **88.5%** |
| `trunk` | 2.8% | **14.3%** | **14.2%** |

**The self-transform does what it was built to do.** The graph layer went from destroying 38
points of per-country influence to destroying 4: `gconv1` recovery 60.1% → 94.3%, against a raw
ceiling of 98%. The mechanism diagnosed above -- one weight matrix for a country and its neighbours,
self-loop weight `1/(deg+1)` -- was the right diagnosis, and a second weight matrix is the whole
fix. Per country the effect is uniform rather than concentrated: South Korea 97%, Iraq 100%,
Italy 94%, Poland 95%, all of which sat between 40% and 67% in the control.

**The attention read-out bought nothing.** E3-13's trunk is 14.2% and E3-12's, with no read-out
at all, is 14.3%. The entire trunk improvement over the control comes from the tokens being less
damaged before pooling; the attention block contributes zero.

That is a design error worth naming precisely, because it was mine. **A single-query read-out is
still a pooling operation.** One query vector over 84 countries returns one weighted average --
attention pooling instead of mean/max pooling. It is a better average, which is presumably part
of why 2.8% became 14%, but it cannot be more than an average, and the ladder now shows the
information is *there*: the token holds 89% and the trunk holds 14%.

**So the bottleneck is the single 512-float vector itself, not how it is filled.** No read-out
that ends in one fixed-size summary can carry 84 countries' worth of exact influence to a head.
That is the case for **per-entity output heads** -- `logit_i = f(token_i, trunk)` for each of the
110 card and 84 country actions -- which was deferred as the riskier change and is now the
indicated one. The ladder gives it a sharp prior: the tokens it would read carry 89%.

#### And it shows up in Elo

Both seeds of both arms, pooled over four late snapshots each, 200 games a side, 24 models,
anchored on the identity control:

| arm | vs anchor | 95% CI | **Elo** | snapshot spread |
|:---|---:|:---:|---:|---:|
| `E3-10-21-080M` control | — | — | **0** | 61 |
| `E3-10-22-080M` control, other seed | 47.4% | [46.2, 48.6] | −18 | 52 |
| **`E3-12-21-080M` self-transform** | 57.5% | [56.3, 58.7] | **+53** | 37 |
| **`E3-12-22-080M` self-transform** | 61.8% | [60.6, 63.0] | **+83** | 94 |
| `E3-13-21-080M` + attention read-out | 48.0% | [46.8, 49.2] | −14 | 60 |
| `E3-13-22-080M` + attention read-out | 46.8% | [45.5, 48.0] | −23 | 70 |

**The representation fix is worth +53 and +83 Elo**, both seeds positive against a control seed
pair 18 apart. Against `E3-12-21` directly, the two controls sit at −56 and −61.

**The read-out costs the entire gain**, on both seeds: −14 and −23 against the control, and −67
and −91 measured directly against `E3-12-21`. It adds no information (above) and it ends the
trunk in a non-residual `Linear → LayerNorm → GELU` after the residual blocks, which breaks the
identity path the stack was built around. Both reasons point the same way, and the ladder rules
out the charitable one.

The seed-22 ladder agrees with seed 21 throughout: `gconv1` 93.4% and 92.1% against the control's
60.1%, and trunks of 6.1% and 8.0% -- low everywhere, and not systematically higher with the
read-out than without it.

**This answers the question the arm was posed to answer.** The prediction registered before the
runs allowed that Elo might not move at all, which would have said per-country influence is not
what limits play. It moved, on both seeds. Exact per-country influence is worth most of a budget
doubling -- the E3 Elo scale above puts a doubling at +106 and this is +53 and +83 -- for one
extra weight matrix per graph layer and **no measurable throughput cost**: median 15,125 steps/s
against the control's 15,097 over the runs themselves. An earlier figure of ~17% came from a
2M-step smoke run and was startup-dominated.

**Adopt `--self-transform`; do not adopt `--attn-readout` in this form.**

## Does a network that can see the board contest more of it?

The self-transform gave that arm a representation that holds per-country influence where the
control's lost it. The behavioural question that follows is whether it *uses* it: eight of 29
battlegrounds sit empty from turn 8 in the control, always the same ones (*what the 160M arms
actually do*, above). Position diagnostics, four late snapshots each, temperature 0.1:

| | E3-10 control | E3-12-21 | E3-12-22 | E3-13-21 |
|:---|---:|---:|---:|---:|
| empty battlegrounds, turn 5 | 11.88 ±1.25 | 11.72 ±1.16 | **8.90 ±1.06** | 11.59 ±0.88 |
| empty battlegrounds, turn 8 | 7.05 ±1.07 | 7.61 ±1.40 | **4.71 ±1.26** | 8.38 ±0.59 |
| uncontrolled, turn 8 | 15.39 ±0.98 | 15.32 ±0.88 | 13.91 ±1.55 | 15.61 ±0.74 |
| mean final turn | 7.38 ±0.62 | 7.52 ±0.29 | 7.61 ±0.68 | 6.91 ±0.62 |

**The answer is "it can, not it does".** `E3-12-22` contests three more battlegrounds at turn 5 and
two and a half more at turn 8 -- far outside the ±1.2 snapshot spread. `E3-12-21`, the same
architecture with a different seed, is indistinguishable from the control. The architecture makes
the board *available*; which policy is found is still down to the seed.

It is suggestive that the seed which used it is the stronger one -- `E3-12-22` is the +83 Elo arm
and `E3-12-21` the +53 -- but that is one pair, and it is exactly the shape of correlation that
needs more seeds before it is a claim.

| battleground | E3-10 control | E3-12-21 | E3-12-22 | E3-13-21 |
|:---|---:|---:|---:|---:|
| **India** | 97.8% | **37.0%** | **61.5%** | 97.6% |
| Saudi Arabia | 92.9% | 77.3% | 94.2% | 95.2% |
| Algeria | 71.7% | 98.1% | **24.5%** | 96.8% |
| Libya | 68.0% | 97.2% | **14.2%** | 56.0% |
| Brazil | 52.1% | 89.2% | 27.6% | 99.6% |
| Argentina | 41.2% | 16.2% | 27.6% | 94.3% |
| Mexico | 19.0% | 96.6% | 13.5% | 64.5% |
| France | 6.7% | 20.6% | 4.3% | 22.8% |

**India is the one country both self-transform seeds agree on**, 97.8% → 37.0% and 61.5%, and the
attention arm leaves it at 97.6%. India was the flagship never-touched battleground in *what the
160M arms actually do* and the one identity embeddings had already halved; the representation fix
takes it further, and it is the one behavioural change that tracks the architecture rather than
the seed.

**Everything else is seed-divergent, and wildly.** Seed 21 *abandons* Africa and Central America
-- Algeria 71.7% → 98.1%, Libya 68.0% → 97.2%, Mexico 19.0% → 96.6% -- while seed 22 takes them
up: Algeria to 24.5%, Libya to 14.2%, Mexico to 13.5%. Two runs of one configuration, differing
only in seed, found opposite policies about two whole regions.

**A design limit worth stating.** Only `E3-10-21` was probed as the control, so the control's own
seed variance on these numbers is unmeasured -- and given how far the two E3-12 seeds are apart,
that variance is exactly what would be needed to attribute any of this to the architecture. The
India result survives because both arms agree and the third disagrees; nothing else here does.

`E3-13-21`, the arm whose read-out cost its Elo, is also the arm that contests least: 8.38 empty
at turn 8 against the control's 7.05, and South America effectively abandoned (Brazil 99.6%,
Argentina 94.3%). Its representation and its play agree with each other.

## Per-entity policy heads, built wrong: −219 Elo, and why

The read-out entry above argued that no read-out ending in one fixed-size summary can carry 84
countries to a head, and that the remaining move was to stop routing the board through the trunk:
compute a country's logit from that country's own token. E3-14 does that. Actions 0-109 (cards)
and 119-202 (countries) get a per-entity head; the 18 that name no entity stay dense.

| | vs `E3-12-21` | 95% CI | **Elo** |
|:---|---:|:---:|---:|
| `E3-12-21` self-transform | (anchor) | — | **0** |
| `E3-12-22` self-transform | 54.9% | [53.7, 56.1] | +34 |
| `E3-10-21` control (no self-transform) | 43.0% | [41.8, 44.2] | −49 |
| **`E3-14-21` per-entity heads** | 22.1% | [21.1, 23.1] | **−219** |

Worse than the control it was built on top of, by a margin no snapshot spread explains.

**The cause is in the implementation, not the idea.** The head was written as

```python
self.pe_trunk = nn.Linear(hidden_dim, d)      # 512 -> 64
ctx = self.pe_trunk(h)
country_in = cat([country_token, country_raw_slots, ctx])
```

so **the only path from the trunk to any action logit became 64 floats**, where the dense head
read all 512. Each logit gained its own country's detail and lost seven eighths of its view of
the situation. That trade is what −219 measures, and it is why the arm lands below a control that
at least kept the full trunk.

**The representation went the other way**, which is the tell. Battlegrounds, share of the
recoverable exact-influence gap:

| stage | `E3-10` control | `E3-12-21` | `E3-12-22` | `E3-14-21` |
|:---|---:|---:|---:|---:|
| `gconv1` | 60.1% | 94.3% | 93.4% | **95.4%** |
| `gconv2` | 63.5% | 89.2% | 84.3% | 88.6% |
| `trunk` | 2.8% | 14.3% | 6.1% | **26.9%** |

The trunk holds *more* per-country influence than in either E3-12 seed, plausibly because it is
now under pressure to make that information survive a 64-dim projection. So the arm improved the
representation and halved the play. **A representation probe moving the right way is not evidence
the change helped** -- which is worth having measured, since the two preceding entries leaned on
those probes to choose what to build.

**The prediction was wrong in both halves.** It said the trunk should not move and Elo should
rise. The trunk moved and Elo collapsed. Registering predictions did its job anyway: it made the
result legible instead of something to rationalise.

#### What to build instead

A **residual** form, which is strictly better than what was built:

```python
logit_i = dense_logit_i + per_entity_correction_i
```

At initialisation the correction is near zero, so the network starts *exactly* as the dense
baseline and learns a per-entity refinement on top of it. It keeps the full 512-dim context, it
cannot be worse than dense at initialisation, and it makes the per-entity path an addition rather
than a replacement. E3-14 instead replaced the baseline outright and cold-started from a
bottleneck.

The open question the read-out entry raised is therefore still open. Nothing here shows that
per-country information cannot help the policy; it shows that paying 448 floats of global context
for it is a bad trade.

## The same idea built two ways spans 413 Elo

E3-15 is E3-14's per-entity heads as a **residual** on the dense logit, with the correction's
output layers zero-initialised so the network starts as the dense baseline exactly.

| arm | vs `E3-12-21` | 95% CI | **Elo** | snapshot spread |
|:---|---:|:---:|---:|---:|
| `E3-12-21` self-transform | (anchor) | — | **0** | 57 |
| `E3-12-22` self-transform | 55.2% | [54.0, 56.4] | +36 | 83 |
| `E3-14-21` per-entity, replacing | 20.8% | [19.8, 21.8] | **−232** | 93 |
| **`E3-15-21` per-entity, residual** | 73.9% | [72.8, 75.0] | **+181** | 53 |

+181 against a snapshot spread of 53, and against a seed gap of 36 in the arm it is anchored on.

**The gap between the two builds is 413 Elo, and the only difference is whether the dense term
survives.** Both compute a per-entity correction from the same tokens with the same 64-float
context. E3-14 returned the correction alone, making that 64-float projection the sole path from
the trunk to any logit; E3-15 adds it to the dense logit, which still reads all 512. The idea was
never the problem.

**The representation is unchanged, so the gain is in the heads.** Battlegrounds: `gconv1` 95.0,
`gconv2` 90.8, `trunk` 14.8 -- E3-12-21's 94.3 / 89.2 / 14.3 to within noise. E3-15 reads the same
trunk as its baseline and plays 181 Elo better, which is the direct confirmation of the read-out
entry's claim: the information was present and stranded, and the fix was a path to it rather than
more of it.

**Both registered predictions held** -- it could not start worse than the dense baseline, and the
floor was E3-12. Recorded before the run, as with E3-14, whose predictions both failed.

A caution kept from the replacing form above: E3-14 moved the trunk ladder *up* while halving
play, so the ladder is reported here and not used to argue the arm is good. The Elo is the
argument.

**One seed.** The effect is 3.4x the snapshot spread and the two preceding architecture changes
were both legible from one seed, so this is reported rather than held -- but E3-15 now becomes the
recipe everything downstream is measured against, and a baseline resting on one seed is the kind
of thing *a single snapshot cannot measure an effect smaller than a run's oscillation*
([`variance_and_noise.md`](variance_and_noise.md)) exists to warn about.

## The architecture progression on one scale

Every arm of the progression, all seeds that exist, four late snapshots each, 28 models in one
pool anchored on the E3 control:

| arm | vs anchor | 95% CI | **Elo** | snapshot spread |
|:---|---:|:---:|---:|---:|
| `E3-01-21-080M` control | — | — | **0** | 33 |
| `E3-01-21-240M` control, 3x the **steps** | 66.9% | [65.7, 68.0] | **+122** | 55 |
| `E3-10-21-080M` identity | 64.9% | [63.8, 66.1] | +107 | 59 |
| `E3-10-22-080M` identity | 66.0% | [64.9, 67.2] | +115 | 63 |
| `E3-12-21-080M` + self-transform | 72.0% | [70.9, 73.1] | +164 | 51 |
| `E3-12-22-080M` + self-transform | 73.8% | [72.7, 74.8] | +179 | 92 |
| **`E3-15-21-080M` + per-entity residual** | 85.2% | [84.3, 86.0] | **+304** | 67 |

**Identity embeddings alone, at 80M, are worth about what tripling the step budget is worth** --
+107 and +115 against the 240M control's +122. The full stack at 80M clears it by roughly 180 Elo,
for 5,152 embedding parameters, one extra weight matrix per graph layer, and a zero-initialised
correction head.

**In wall clock the comparison is 2.4x, not 3x**, and the difference matters when the claim is
about compute rather than steps. Measured medians over the runs: control 15,097 steps/s, the
self-transform 15,125 (free), the residual heads 12,159. So 240M of control is about 4.4 hours
against 1.8 for 80M of E3-15. The architecture is still ahead on equal wall clock, by a smaller
margin than the step counts suggest.

**Quote the anchored column for the picture and the paired measurement for an increment.** Elo
converted from pooled win rates is not additive: this pool puts E3-15-21 164 above E3-12-21 by
subtraction, while the two measured head to head give **+181** (above). Both are right about
what they measure; the subtraction compresses differences between two arms that are each far from
the anchor (*one Elo scale for E3*, above).

**Each step came from a measurement, not a guess.** Identity followed from 86% of cards sharing a
feature vector; the self-transform from a graph layer that attenuated a country's own influence in
proportion to its degree; the residual heads from information that was present in the tokens at
89-91% and stranded before the heads at 6-14%. The two failures in the sequence -- the attention
read-out and the replacing form of the per-entity heads -- were diagnosed by the same instruments
that chose the successes.

**E3-15 rests on one seed**, and it is now the recipe. That is the standing caveat on this table.

## Does the network price a country for the operation it is performing?

Elo says a policy is better; it never says what it understands. Three probes built against the
engine's own answer key, so none of them needs a human judgement.

#### The same card, the same board, Event or Operations

The sharpest of the three, because the two branches differ in exactly one decision. Placement
costs **2 Ops per point in a country the opponent controls** and 1 elsewhere
(`Operations::get_influence_cost`), while event placement never consults cost at all
(`Operations::place_influence`). So free-placement events should want the countries ordinary Ops
should avoid, and a removal event should want them most.

All four cards are held by their own side, so playing for Ops does not fire the event and the
pairing is exact. Probability mass on opponent-controlled countries:

| card | E3-15 Event | E3-15 Ops | **gap** | E3-01 Event | E3-01 Ops | **gap** |
|:---|---:|---:|---:|---:|---:|---:|
| The Voice of America (removal) | 80.1% | 13.4% | **+66.7** | 76.9% | 26.8% | +50.1 |
| Ussuri River Skirmish | 40.3% | 16.0% | **+24.3** | 30.8% | 31.6% | **−0.8** |
| Colonial Rear Guards | 28.9% | 17.3% | **+11.6** | 17.3% | 26.8% | **−9.5** |
| Decolonization | 39.5% | 26.0% | **+13.5** | 11.7% | 20.5% | **−8.8** |

**E3-15 has the gap positive on all four. The control has it backwards on all three
free-placement events** -- it puts *less* weight on opponent-held countries when placement is free
than when it costs double. Voice of America is the easy case for both, because its legal mask
already restricts to countries holding Soviet influence (uniform baseline 60.1%); the
discriminating cases are the placement events, where the uniform baseline is 6-18%.

E3-15's Ops branch sits at 13-26% and its Event branch at 29-80% **on the same boards**. That is
one network conditioning its country choice on the operation, which is what the per-entity heads
were for, and it is invisible to Elo.

#### Where it aims, by operation

Opponent influence in the chosen country against the mean over that node's own legal set, so the
mask cannot manufacture the result:

| operation | legal | chosen | uniform | lift | battleground | unif |
|:---|---:|---:|---:|---:|---:|---:|
| event: Ussuri River Skirmish | 14.5 | 3.97 | 1.17 | **+2.80** | 0.81 | 0.38 |
| event: Brush War | 51.8 | 2.79 | 0.62 | **+2.17** | 0.94 | 0.29 |
| event: The Voice of America | 16.4 | 4.00 | 2.67 | **+1.33** | 0.91 | 0.64 |
| ops: influence | 38.0 | 2.00 | 0.49 | +1.51 | 0.79 | 0.39 |
| **ops: coup** | 9.0 | 2.09 | 2.11 | **−0.02** | **0.59** | **0.68** |
| ops: realign | 6.7 | 2.32 | 2.21 | +0.10 | 0.89 | 0.78 |
| event: Comecon | 7.5 | 0.10 | 0.07 | +0.03 | 0.15 | 0.21 |

**Couping is undiscriminating**: no opponent-influence lift and a *below-uniform* battleground
rate. Couping battlegrounds is what earns military operations and what moves regional scoring, so
this is a specific, named weakness rather than a general one.

#### Picking the best placement

Scored against the engine's true marginal regional VP for every legal country: chosen **1.038**
against 0.272 for a uniform legal pick and 3.162 for the best available -- **lift +0.767**, the
single best target taken **31.7%** of the time, average **93rd percentile** of the legal set. It
reliably finds a good country and often not the best one.

#### What did not work, and why it is recorded

The first version of this probe asked whether regional VP is recoverable from the trunk. It is
not a test: `global_features[64..69]` already carry the live per-region differential, so it asks
whether six floats can be copied. The counterfactual replacement was better and still weak --
board slot 24 is `my_deficit`, "how many Ops to reach control", which is most of the threshold
answer, so a linear probe on the raw input already scores 0.886 AUC and the encoder adds 0.010.
The lesson generalises: **a probe is only a test of understanding if its answer key is absent
from the observation**, and this observation is rich in precomputed per-country facts.

## Does it know which region a scoring card scores?

The one association the observation cannot supply. The card block carries properties and
location, never identity, and the six scoring cards collapse into **two feature groups** -- early
war (Asia, Europe, Middle East) and mid war (Central America, Africa, South America). Era
separates the groups; nothing separates within one. So this is a direct test of the identity
embedding.

Both arms are evaluated on **one shared set of boards**. An earlier run let each arm sample its
own self-play positions and produced the opposite conclusion on one of the two measures; that is
the same confound the operation-pricing probe above records for the marginal-value probe,
reintroduced and caught.

#### Location: moving the card, holding the board fixed

Placement mass into the card's own region, minus the mean shift into the other regions, so a
generic "a scoring card is in hand" reflex cancels:

| card | E3-15 | E3-01 control |
|:---|---:|---:|
| Asia Scoring | **+12.3%** | **−11.6%** |
| Europe Scoring | **+12.2%** | +7.0% |
| Middle East Scoring | **+30.3%** | +16.5% |
| Central America Scoring | **+7.8%** | −0.2% |
| Africa Scoring | **+2.7%** | +8.7% |
| South America Scoring | **+4.6%** | +2.6% |

**E3-15 responds in the right direction in all six regions; the control has Asia backwards and
Central America flat.** Moving Europe Scoring into hand raises the model's appetite for Europe
specifically -- the card-to-region association, visible in behaviour.

#### Ordering: which scoring card to play

| | E3-15 | E3-01 control |
|:---|---:|---:|
| took the best scoring card (US / USSR) | 27.4% / 34.4% | **59.7% / 55.7%** |
| rank correlation with true VP | +0.39 / +0.47 | **+0.50 / +0.56** |
| value captured | 74.2% / 71.8% | **89.3% / 88.7%** |

Chance is 16.7% and group-level guessing 33%, so both beat both -- but **the control is markedly
better**, on identical inputs. E3-15 knows *where* a scoring card points and is worse at deciding
*which* to play. That is a negative worth carrying: the arm that wins by 304 Elo loses this
comparison clearly.

#### A blind spot both share

Putting a scoring card in the **opponent's** hand moves neither model: the same difference in
differences runs −1.3% to +1.1% across all six cards and both arms. An opponent holding Europe
Scoring is a reason to defend Europe, and nothing in either policy reacts to it. Hand knowledge
of the opponent is partial, but the card's location *is* in the observation, so this is not an
information limit.

**Caveats.** The shared boards come from E3-15's self-play, so the population is one arm's
distribution even though both models see identical inputs. E3-15 is one seed.
