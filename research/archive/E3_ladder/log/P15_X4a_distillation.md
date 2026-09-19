# P15-X4a — one offline round of expert iteration

**Measured 2026-09-17.** Source checkpoint `E3-20-28 @200M` (the peak, not a final). The honest
searcher answered 75,592 card/play-mode decisions across 400 self-play games at 96 simulations;
the **raw policy played every move**, so the targets sit on the policy's own distribution. Dataset
`/workspace/data/datasets/x4a_search_targets.jsonl.gz`, searcher commit `fad0130b0a`, clean tree.

## Question one: is the search edge expressible as a policy?

**Yes, +47.7 Elo** — and from a much smaller re-weighting than expected.

| model | Elo | vs source, head to head |
|:---|---:|---:|
| **x4a_distilled** | **1666.4** | **57.1%** (60.8% as USSR, 53.4% as US) |
| source `@200M` | 1618.7 | — |
| anchor `@280M` | 1585.7 | |
| anchor `@080M` | 1500.0 | |

1000 games a side, `/workspace/data/tournaments/P15_X4a_distilled/`. The plan's bar was ≥ 40 Elo.

It also improves against the anchors specifically: `54.0% / 63.4%` against `@280M` where the
source manages `48.8% / 57.8%`.

## The surprise: there was almost nothing to distil

The agreement diagnostic, run **before** distilling because the headroom is worth knowing in
advance rather than as a post-hoc excuse:

| | before | after 2 epochs |
|:---|---:|---:|
| top-1 agreement with the searcher | **93.1%** | 94.0% |
| searcher's choice in the policy's top 3 | 99.6% | 99.8% |
| median rank the policy gives it | 0 | 0 |
| KL(search ‖ policy) | **0.0330** | 0.0228 |
| CE(search ‖ policy) | 0.7104 | 0.7002 |

The policy already ranked the searcher's choice **first** at a median decision, agreed with it
93.1% of the time, and sat 0.033 nats away. Distillation moved agreement by 0.9 pp and KL by 31%.

**And that bought 47.7 Elo.** A re-weighting far too small to be teaching the policy new moves is
worth about fifty rating points, which says the remaining disagreement is concentrated on
decisions that matter rather than spread thinly.

## Why the headroom was so small, and where the rest of the edge probably is

Mean legal actions at a searched decision: **4.1**. Card/play-mode decisions barely branch —
`SELECT_PLAY_MODE` averages 1.96 legal actions and `SELECT_CARD` 5.49. There is very little for a
searcher to discover where there is almost nothing to choose between.

`POINT_NODE` placements average **17.5** legal actions and reach 82. That is where the branching
is, and X4a did not look there: the plan's step 2 specifies card/play-mode nodes, following P3's
argument that those are "the decisions that matter".

Against that, [`P15_X0_search_on_200M.md`](P15_X0_search_on_200M.md) measured a searcher over
**all** nodes at +129.2 Elo on the same weights. X4a extracted 47.7 of it from card/play-mode
alone.

**The scoping was wrong** — see
[`../findings/which_decisions_to_search.md`](../findings/which_decisions_to_search.md).
Redoing this offline round over **all** decisions, games-matched at 400 games each, gives 1572.5
against the card/play-mode round's 1560.3 and the source's 1529.6, rated at temperature 0
(`/workspace/data/tournaments/P15_X4a_allnodes/`). Head to head the two filters are a **dead
heat** — 49.4% to all-nodes over 1,000 games, losing marginally while sitting 12 Elo higher in the
table. The scoping was still wrong, but **fixing it bought nothing measurable in one-shot offline
distillation**, despite `POINT_NODE` carrying 71.8% of the CE signal.

**Placements are where the signal is** —
[`P15_X4a_where_the_search_signal_is.md`](P15_X4a_where_the_search_signal_is.md). Searching all
85,113 decisions of 200 games shows top-1 agreement is uninformative (90.5–97.8% for *every*
decision type) while KL varies 7×, and `POINT_NODE` carries **71.8% of the total CE signal**
against card/play-mode's 20.9%. X4b is run with `--search-node-filter all` on the strength of it.

## Question two: does it survive RL?

**No — the edge erodes to nothing. But the first version of this section got the reason wrong,
and the control is what corrected it.**

Two arms, same seed, same recipe, same 20M steps of ordinary NashPG, differing only in whether
the starting point had been distilled:

* `E3-25-28` resumed from the **distilled** checkpoint
* `E3-26-28` resumed from the **undistilled** source

All five checkpoints rated in one field, 500 games a side,
`/workspace/data/tournaments/P15_X4a_washout_control/`:

| model | Elo | overall |
|:---|---:|---:|
| `distilled` (pre-RL) | **1548.8** | 56.8% |
| `source_200M` (pre-RL) | 1526.7 | 52.9% |
| `anchor_280M` | 1500.0 | 48.1% |
| `after_rl_distilled` | 1491.3 | 46.5% |
| `after_rl_control` | 1482.6 | 45.0% |

### What the control changed

Rated alone against its own starting point, the distilled arm looked catastrophic — 75 Elo below
the checkpoint it resumed from, last in its field. **That reading was wrong**, and it was wrong in
the specific way this project keeps having to guard against: it attributed to the treatment
something that the control shows happens anyway.

The control lost **44.1 Elo** over the same 20M steps, from a starting point that had been
distilled into nothing at all. The distilled arm lost 57.5. So the large drop is the **price of
training 20M steps past this lineage's peak** — which [`P15_X0_frozen_anchors.md`](P15_X0_frozen_anchors.md)
had already measured independently (2193.7 at 200M against 2168.0 at 240M) — and not the loss of
the distilled prior.

### What actually happened to the distilled edge

| | distilled vs source |
|:---|---:|
| before RL | **+22.1 Elo** |
| after 20M steps each | **+8.7 Elo** |

Head to head, the two post-RL arms split **51.2% / 48.8%** over 1,000 games. At SE ≈ 1.6 pp that
is a dead heat: after 20M steps, a checkpoint that was distilled and one that was not are
**statistically indistinguishable**.

So the edge does not survive. It decays from a real +22 to a residual inside noise, while both
arms separately pay a much larger step-cost that has nothing to do with distillation.

**Pre-registered branch, from the plan:** *"moves but washes out → the signal must be present
during RL; build X4b."* The edge moved (+47.7 Elo in step 3's field) and did not survive. **X4b is
the indicated next experiment**, and now for a defensible reason rather than an artefact.

### RL in this recipe degrades one side — but which one is not established here

> **Caveat added 2026-09-17.** The figures below are **side balance averaged across the tournament
> field**, which is not a property of a model: the same checkpoint reads −13.9 pp here and +2.8 pp
> in a different field ([`P15_control_per_seat.md`](P15_control_per_seat.md)). Rated per seat
> against *frozen* anchors, the no-search control's collapse is on the **US** seat, not the USSR
> one. The within-arm *change* below may still be real, being a before/after on one field, but the
> **side it names is not established** and this section should not be cited for a side-specific
> claim until the arms are re-rated per seat against fixed opponents.

The same tournament, side balance before and after the identical 20M steps:

| model | as USSR | as US | USSR − US | change |
|:---|---:|---:|---:|---:|
| `distilled` | 55.2% | 58.4% | −3.3 pp | |
| `after_rl_distilled` | 36.2% | 56.8% | −20.6 pp | **−17.3 pp** |
| `source_200M` | 52.2% | 53.6% | −1.4 pp | |
| `after_rl_control` | 38.5% | 51.5% | −13.0 pp | **−11.6 pp** |
| `anchor_280M` | 36.6% | 59.6% | −23.0 pp | |

This is stronger evidence than the cross-model version noted earlier, because it is a **within-arm
before/after on the same weights and the same seed**: 20M steps of this recipe moved side balance
by −17.3 pp and −11.6 pp in two independent arms, both toward US. `anchor_280M`, which is simply
this lineage trained further, sits at −23.0 pp.

Whatever is wrong past the peak is not a uniform loss of strength — **it is the USSR side being
given away**, which is the question [`P15_X1_frozen_exploiter.md`](P15_X1_frozen_exploiter.md) was
circling. Two arms is not many, but they agree in sign and rough size and the mechanism is now
worth naming as a target rather than an observation.

## Caveats

* One source checkpoint, one distillation, one seed.
* Two epochs at 1e-4. No sweep; the point was whether it moves at all.
* The value head took no loss, but the trunk is shared, so its inputs changed. The rating is
  therefore of the whole network, which is the honest thing to rate.
* +47.7 Elo is measured in a four-model field and does not travel to another tournament.
