# P19 — is the late-E3 architecture actually stronger on the E4 engine?

**Status: arms running.** `E4-04-01` (control) is in flight; `E4-03-01` is queued behind it on the
GPU with a monitor armed. This file is the **pre-registered** analysis: the measurement and the
decision rule are fixed here *before* the numbers exist, so that the metric cannot be chosen after
seeing them. Registration of the arms themselves is in [`../runs.md`](../runs.md).

## The question

Every late E3 arm carried `identity_dim 16`, `per_entity_heads 64`, `graph_layers 0`,
`self_transform` — the architecture E3 settled on, and the one E3's collapse appeared on. Both E4
arms so far ran the bare defaults, because the launch command was copied from a template rather
than derived from a healthy run
([`../findings/training/e4_architecture_discontinuity.md`](../findings/training/e4_architecture_discontinuity.md)).

So two things are unknown at once, and only one is worth spending 320M on:

1. **Is that architecture stronger on this engine?** E3 reported it was, on E3's engine.
2. **Does the collapse belong to the architecture rather than to the pool?** Only worth asking if 1
   holds — there is no point hunting a collapse in a network that is simply worse.

P19 answers 1. It does not answer 2.

## The arms

Both cold-started, both pooled (`--opponent-frac 0.3 --opponent-self-pool --opponent-pool-size 12`),
both 80M, same engine, same seed policy, same `ref_update_freq` (200,000 — held constant by
decision, [`../findings/training/ref_update_freq_open_ablation.md`](../findings/training/ref_update_freq_open_ablation.md)).
**Only the network differs:**

| arm | `identity_dim` | `per_entity_heads` | `graph_layers` | `self_transform` |
|:---|---:|---:|---:|:---|
| `E4-03-01` | 16 | 64 | 0 | yes |
| `E4-04-01` | 0 | 0 | 2 | no |

`E4-02-01` cannot serve as the control: it is warm-started, which biases precisely the first tens
of millions of steps this comparison covers.

## The measurement

**Head-to-head, `tools/tournament.py`, 200 games per model pair — 100 with the first model as US
and 100 with it as USSR**, reported per side. Not 30: a pairwise win rate from 30 games has a
standard error near 9 pp, which is wider than the effect being looked for.

Entrants: `E4-03-01@80M`, `E4-04-01@80M`, plus `heuristic` and `random` as fixed anchors. Labels
come from `tools/lib/checkpoint_id.py`, which refuses a name clash.

**Extended to settle P21's anchor.** Add the best snapshots of `E4-01-01` and `E4-02-01`, so the
same tournament determines which of the four E4 runs rates highest. That arm becomes the absolute
yardstick for every rung of [P21](P21_architecture_ladder.md) — see *Two comparisons, with
different jobs* there for why it is a yardstick and not a control.

Two sanity conditions that must hold before the headline number is read at all:

* **Both arms must beat `heuristic` and `heuristic_mcts` decisively.** If a network at 80M is not
  near-dominant over the scripted baselines, something is wrong with the arm or the harness and the
  A/B is meaningless.
* **Both arms must be silent under the health alarms** (`NOPOOL`, `POOLSTUCK`, `KLSPIKE` —
  [`../method/detecting_collapse.md`](../method/detecting_collapse.md)). A collapsed arm is not a
  measurement of its architecture.

## The decision rule, fixed in advance

Let *d* = `E4-03-01`'s win rate against `E4-04-01`, pooled across both seats, with its standard
error from 200 games (~3.5 pp).

| outcome | reading | action |
|:---|:---|:---|
| *d* > 55%, both seats above 50% | the architecture is stronger here too | extend `E4-03` to 320M and hunt the collapse in it |
| 45% ≤ *d* ≤ 55% | no detectable difference at 80M | **do not** spend 320M on it; the E3 collapse is not attributable to the architecture on this evidence, and the pool remains the leading explanation |
| *d* < 45%, both seats below 50% | the architecture is *worse* on this engine | drop it; record that an E3 result did not transfer, which is itself a finding about the lineage boundary |
| seats disagree in sign | one-sided result, not a strength result | report per-seat, extend neither, and treat side asymmetry as the thing to investigate |

**One seed per arm.** Seed spread in this repo has been measured at ~95 Elo
([`../archive/E3_ladder/findings/seed_variance.md`](../archive/E3_ladder/findings/seed_variance.md)), which is larger
than the middle band above. So a result in the 45–55% band is **genuinely inconclusive rather than
evidence of no effect**, and a result outside it is suggestive rather than settled. Stating that
here, in advance, is the point: the honest ceiling on what one seed can show is a property of the
design, not something to be negotiated after the number arrives.

## What will not count as evidence

* Training-loop metrics compared between the two arms as a proxy for strength. Entropy,
  `clip_frac` and `adv_std_raw` describe *how* a run trained, not how well it plays, and no
  threshold on them has survived a second dataset.
* Any comparison against an E3 number. Different lineage; magnitudes do not transfer
  ([`../method/detecting_collapse.md`](../method/detecting_collapse.md), scope rule).
* Win rate against `HeuristicBot` as the headline. It is a sanity gate above, not the measurement.
* The arms' own in-training `--eval-opponents` results, which use a different harness from the
  tournament and are there to catch breakage, not to rank.

## Afterwards

Whatever the outcome, both arms' `us_episode_frac`, `kl_div` and pool trajectories go into the E4
reference envelope in [`../method/detecting_collapse.md`](../method/detecting_collapse.md). They
will be the **first cold-start entries** in it — the existing E4 arms are both warm-started, so
there is currently no legitimate reference for a cold-start trajectory at all, and the gap is why
an entropy check on `E4-04-01` was briefly (and wrongly) made against E3's band.
