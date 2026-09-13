# Architecture — what the network is, and what each part is worth

What is true now about the network's architecture: the recipe line, what each change bought, what
the representation demonstrably holds, and what is still broken. **Update discipline: current
conclusions, rewritten when they change.** Nothing here is history — when a number is superseded
it is replaced, and the run that produced it, the failures on the way, and the retracted readings
stay in [`../log/P9_architecture.md`](../log/P9_architecture.md), which is append-only. Keep this
file short: a claim that needs a paragraph of setup to state is a log entry, not a finding. The
arm names are decoded in [`../method/run_nomenclature.md`](../method/run_nomenclature.md).

## The recipe line

ColdWarNetV2's structured backbone — GraphConv over the 84 countries, per-card encoder,
card/country cross-attention, global ResNet trunk, masked action heads — plus three changes, all
model-side, **the observation untouched by every one of them**:

| in the recipe | flag | what it is |
|:---|:---|:---|
| identity embeddings | `--identity-dim 16` | a learned embedding indexed by position on each card and country token; 5,152 parameters |
| self-transform | `--self-transform` | a second weight matrix per graph layer applied to the node itself, so a country is not averaged with its neighbours |
| per-entity residual heads | (E3-15) | `logit_i = dense_logit_i + correction_i(token_i, trunk)`, correction zero-initialised so the network starts as the dense baseline exactly |

**Not adopted**, each for a measured reason:

| rejected | cost | why |
|:---|---:|:---|
| `--attn-readout 64` (attention read-out into the trunk) | −67 to −91 Elo | a single-query read-out is still pooling; it adds no information and breaks the residual identity path |
| per-entity heads *replacing* the dense logit | −219 Elo | leaves 64 floats as the only path from the trunk to any logit |
| masking the 1,364 constant observation slots | −20 Elo (inside seed spread) | a constant input is spanned by the bias; it cannot carry information, and throughput did not move |
| an MLP backbone in place of the structured one | −110 Elo | structure, not capacity: the MLP has 2.6x the parameters |
| categorical value head | −28 Elo | mildly harmful, not the null the anchor reported |
| `--window-provoked-defcon` | −68 Elo | the turn-scoped window is 5.6x wider than the mistake it credits |

## What each change is worth

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

Identity embeddings alone, at 80M steps, are worth about what tripling the step budget is worth.
In wall clock the comparison is 2.4x rather than 3x (the residual heads run at 12,159 steps/s
against the control's 15,097; the self-transform is free at 15,125). Elo from a pooled win rate
is not additive — for an increment, use the paired measurement: self-transform +53 and +83 over
identity, residual heads **+181** over the self-transform.

**Each change came from a measurement, not a guess.** Identity followed from 86% of the 110 cards
sharing a feature vector — 95 of them are indistinguishable to the network, including every card
this project's failures turn on. The self-transform followed from a graph layer that attenuated a
country's own influence in proportion to its degree (self-loop weight `1/(deg+1)`). The residual
heads followed from information sitting in the tokens at 89–91% and reaching the heads at 6–14%.

## What the representation holds

Share of the recoverable exact-influence gap on battlegrounds, tapped at four points of one
rollout, whole games held out, penalty selected per stage:

| stage | E3-10 control | E3-12 self-transform | E3-13 + read-out |
|:---|---:|---:|---:|
| `raw` | 98.0% | 98.1% | 98.7% |
| `gconv1` | 60.1% | **94.3%** | **93.8%** |
| `gconv2` | 63.5% | **89.2%** | **88.5%** |
| `trunk` | 2.8% | **14.3%** | **14.2%** |

The self-transform fixed the graph layer. **Pooling is still the bottleneck**: the pre-pooling
token holds ~89% and the 512-float trunk holds ~14%, and no read-out that ends in one fixed-size
summary can do better. The per-entity residual heads route around it rather than fixing it.

Two asymmetries hold across every arm measured: the trunk learns **who holds what and not by how
much** (control 24–43% of the gap closed, exact influence 3–10%), and tracks and hand identity
read well everywhere (DEFCON R² 0.85–0.92, VP 0.93–0.97, hand AUC ~0.87), so what is missing is
specifically per-entity board detail.

## What the current network does and does not understand

Probes built against the engine's own answer key, so none needs a human judgement
([the log](../log/P9_architecture.md) has the method and the caveats):

* **It prices a country for the operation it is performing.** On the same board with the same
  card, E3-15 puts 29–80% of its placement mass on opponent-controlled countries when placement
  is free and 13–26% when it costs double, on all four cards tested. The control has this
  *backwards* on all three free-placement events.
* **It knows where a scoring card points**, in all six regions; the control has Asia backwards.
* **Couping is undiscriminating** — no opponent-influence lift over the legal set, and a
  *below-uniform* battleground rate. Couping battlegrounds earns military operations and moves
  regional scoring, so this is a specific named weakness.
* **It picks a good placement and often not the best one**: 31.7% best-target rate, average 93rd
  percentile of the legal set.
* **It is worse than the control at choosing *which* scoring card to play** (27–34% against
  56–60%). The arm that wins by 304 Elo loses this comparison clearly.
* **Neither model reacts to a scoring card in the opponent's hand** — and the card's location
  *is* in the observation, so this is not an information limit.
* **Four battlegrounds are still untouched in every arm**: Algeria, Saudi Arabia, Libya and
  usually Argentina. Identity halved India's empty rate and the self-transform took it further;
  the rest is reallocation, not improvement.
* **The critic does not see a provoked DEFCON-1 coming at any node**, and no architecture change
  so far has moved that.

## Standing caveats

* **E3-15 rests on one seed**, and it is now the recipe everything downstream is measured
  against.
* **The behavioural numbers need a matched-budget control.** Part of what was credited to
  interventions is what longer training does anyway — the 240M control matches identity@160M on
  almost every behavioural line while being 92–109 Elo weaker.
* **Rate against a strong pooled reference, not `HeuristicBot`**, which systematically overrates
  weaker arms; and read a single snapshot's number against the run's own oscillation
  ([`../log/variance_and_noise.md`](../log/variance_and_noise.md)).
* **A representation probe moving the right way is not evidence a change helped.** The arm that
  most improved the trunk ladder is the one that halved play.
