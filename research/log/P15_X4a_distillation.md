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
alone. **The obvious reading is that most of the remaining edge lives in placements**, and it is
untested — the one-line experiment is to regenerate targets with `--node-filter all` and re-run
the agreement diagnostic per decision type. It is queued rather than claimed.

## Question two: does it survive RL?

`E3-25-28_20260917_020529`, 20M steps of ordinary NashPG resumed from the distilled checkpoint —
ten times the ~2M-step washout this project measured for a BC warmup. All four models rated
together, 500 games a side, `/workspace/data/tournaments/P15_X4a_washout/`:

| model | Elo | overall | vs distilled | vs source |
|:---|---:|---:|---:|---:|
| **distilled** (pre-RL) | **1555.3** | 57.9% | — | 56.1% |
| source `@200M` | 1517.5 | 50.7% | 43.8% | — |
| anchor `@280M` | 1500.0 | 47.4% | 41.9% | 46.5% |
| **after 20M RL** | **1480.2** | 43.6% | **40.2%** | **44.5%** |

**The gain did not survive, and the checkpoint did not merely return to its starting point — it
finished last.** 75.1 Elo below the distilled checkpoint it was resumed from, and 37.3 Elo below
the undistilled source that distillation had improved on.

(The distilled margin over the source reads +37.8 here against +47.7 in the step-3 field. Both
fields are four models and neither scale travels; the sign and rough size agree, which is all a
cross-field comparison supports.)

### This is confounded on its own, and the control is what settles it

The tempting reading — "RL erased the distilled prior" — does not follow from this table alone.
[`P15_X0_frozen_anchors.md`](P15_X0_frozen_anchors.md) measured that **this lineage declines past
its 200M peak**: 2193.7 Elo at 200M against 2168.0 at 240M. So 20M more steps may cost Elo here
whether or not anything was distilled in first, and the −37.3 against the source could be the
price of the steps rather than the loss of the prior.

`E3-26-28_20260917_025539` is the control: same seed, same recipe, same 20M steps, resumed from
the **undistilled** source. Pre-registered reading, recorded in its `metadata.json` before it was
launched:

* control also loses ≈ 75 Elo → the loss belongs to the 20M steps, not to the distillation, and
  the washout conclusion above is wrong.
* control holds roughly flat → the loss belongs to the distilled prior, X4a step 4 is a genuine
  washout, and per the plan's pre-registered branch *"moves but washes out → the signal must be
  present during RL; build X4b"*.

**Running; the verdict is not claimed until it lands.**

### A side observation worth keeping

Side balance across the field, from the same tournament:

| model | as USSR | as US | USSR − US |
|:---|---:|---:|---:|
| distilled | 57.5% | 58.4% | **−0.9 pp** |
| source `@200M` | 49.0% | 52.3% | −3.3 pp |
| anchor `@280M` | 38.5% | 56.3% | −17.7 pp |
| after 20M RL | 35.3% | 51.9% | −16.6 pp |

The two checkpoints that received *more RL* — `@280M` and the post-RL arm — carry a ~17 pp US
bias; the two that did not sit within a few points of balanced. That is four models in one field,
not an established effect, and it is exactly the shape [`P15_X1_frozen_exploiter.md`](P15_X1_frozen_exploiter.md)
was circling. Noted as a hypothesis to test properly, not a result.

## Caveats

* One source checkpoint, one distillation, one seed.
* Two epochs at 1e-4. No sweep; the point was whether it moves at all.
* The value head took no loss, but the trunk is shared, so its inputs changed. The rating is
  therefore of the whole network, which is the honest thing to rate.
* +47.7 Elo is measured in a four-model field and does not travel to another tournament.
