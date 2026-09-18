# Rollout temperature: sampling AT the policy beats sampling sharper than it

**Measured 2026-09-18**, arm `E3-33-30_20260917_201441` against its matched control
`E3-30-28_20260917_104226`. Distinct from
[`P15_temperature_selfplay.md`](P15_temperature_selfplay.md), which measured *evaluation*
temperature on frozen weights. This one varies the temperature the trainer **collects rollouts
at**, which changes what the policy learns from rather than how a finished policy is scored.

## What the arm varies, and nothing else

The default rollout bands are `0.15 0.50 0.10 0.35` across the four environment groups. Every one
is **below 1.0**, so the schedule *sharpens* every environment relative to the policy's own
distribution, despite the source describing it as exploration. The arm replaces them with
`0.8 1.2 0.7 1.1` — straddling 1.0, i.e. sampling at roughly the policy as trained, which is the
textbook PPO choice and had never been measured here.

The pair is genuinely matched. A field-by-field diff of the two `metadata.json` files leaves
exactly one behavioural difference, `rollout_temps`; same seed 20260930, same architecture
(`identity_dim 16`, `graph_layers 0`, `per_entity_heads 64`, `self_transform`), same
`ref_update_freq` 5M, same pool (frac 0.3, capacity 12), same reward scheme, no search in either.
`train_steps` differs (240M vs 80M) but **nothing anneals on it**: there is no LR schedule in
`generic_trainer.py` or `nash_pg.py` at all, and the one budget-scaled quantity,
`curriculum_switch_at`, sits behind `is_curriculum`, which is false for `blunder_aware`. So the
budget sets only where each run stops.

## Result: per seat against the frozen anchor `p28_280M`

100 games a side, 200 a point, identical protocol on both arms — the per-seat split rather than
the pooled number, since a pooled rate hid a collapsing seat before
([`method/measurement_pitfalls.md`](../method/measurement_pitfalls.md)).

| steps | control US | control USSR | control overall | **temp US** | **temp USSR** | **temp overall** |
|---:|---:|---:|---:|---:|---:|---:|
| 10.0M | 0.0% | 1.5% | 0.8% | 1.0% | 0.5% | 0.8% |
| 20.1M | 0.5% | 1.0% | 0.8% | **12.0%** | 3.5% | **7.8%** |
| 30.0M | 4.0% | 3.5% | 3.8% | **14.5%** | 7.5% | **11.0%** |
| 40.0M | 2.0% | 7.0% | 4.5% | **21.0%** | 10.0% | **15.5%** |

Two-proportion z on the overall rate: 20.1M **z = 3.45** (p ≈ 0.0006), 30.0M **z = 2.75**
(p ≈ 0.006), 40.0M **z = 3.66** (p ≈ 0.0003). The two arms are indistinguishable at 10M and the
treatment is ahead at every point after.

## The pre-registration was wrong, and in the interesting direction

The arm's registered prediction, recorded in its own `metadata.json` before launch, was:

> higher bands trade rollout quality for coverage, so the arm is expected to look **WORSE**
> early. Judge at 40M and 80M against the matched control [...] an arm that is behind at 40M and
> level or ahead at 80M is the shape that would justify changing the default.

It is not behind at 40M. It is **ahead by a factor of three** on the pooled rate and ahead on both
seats. Whatever the sharpened bands were buying, it was not early progress, and the "trades
quality for coverage" story has no support at any measured point.

Worth stating plainly because the registration also defined what would justify changing the
default, and the observed shape is strictly stronger than the one nominated.

## The seat asymmetry runs the other way

The control is the arm whose US seat never recovers: across the full 240M its US rate sits in a
16–23% band from 120M onward and **ends at 13.5%**, while its USSR seat climbs to 56.0%. It is a
USSR specialist that never learned the other seat.

The temperature arm at 40M is **US-leaning** — 21.0% US against 10.0% USSR. It reaches the
control's *plateau* US strength (first crossed around 120–130M) at **40M**, roughly a threefold
step saving on the seat the control never fixed.

That is the result that would matter if it holds, and it is also the one most likely to be noise
of a different kind: one seed, one anchor, and an asymmetry measured at a single point on a
trajectory that is still climbing steeply. The control's own US seat wandered by 9 pp between
adjacent 10M evals (13.5% at 100M, 16.0% at 110M, 19.5% at 120M), so a single 21.0% is not a
stable property yet.

## The mechanism: sharpened rollouts train on nuclear war

The obvious hypothesis — higher bands buy a better-calibrated critic — is **wrong**, and the
metrics say so directly. At matched steps the treatment's `critic_auc` is *lower*
(0.709 vs 0.731 at 30M, 0.708 vs 0.742 at 40M) and its `critic_brier_skill` lower with it.

What actually separates the arms is how the games **end**. Self-play ending fractions at matched
steps:

| | control 30M | temp 30M | control 40M | temp 40M |
|:---|---:|---:|---:|---:|
| ends in **DEFCON 1** | **59.7%** | 28.9% | **48.9%** | 32.6% |
|  — of which own decision | 23.5% | 10.8% | 20.0% | 12.6% |
| reaches **final scoring** | 6.7% | **25.3%** | 7.8% | **23.2%** |
| mean turn reached | 6.38 | **7.71** | 6.93 | **8.08** |
| mean ply | 89.3 | **108.2** | 97.9 | **115.2** |

**Half to sixty percent of the control's training games end in nuclear war**, most before turn 7,
and only one game in fifteen reaches final scoring. The treatment halves the DEFCON-1 rate,
halves the pure-blunder share of it (`defcon1_self`), and more than triples the fraction of games
that play to the end.

The direction is counterintuitive until you write it out: sharpening concentrates mass on the
argmax, so wherever the policy's argmax is a DEFCON-lowering move, sharpening takes it *every
time*, while sampling near T=1 takes an alternative some of the time. Sharpening does not remove
blunders — it makes a single systematic one universal. That is the opposite of what a schedule
described as exploration is supposed to do.

It also supplies a candidate explanation for the control's permanently weak US seat, the arm's
most conspicuous pathology: if most games end by turn 6 in mutual destruction, the late game is
barely in the training distribution at all, and the late game is where the US seat's points come
from. The treatment reaches turn 8 on average and its US self-play win rate at 40M is 40.0%
against the control's 31.1%.

**The lower `critic_auc` is not evidence of a worse critic.** AUC is a property of the prediction
problem as much as the predictor, and the two arms are not predicting the same problem: longer,
more balanced games that run to scoring are harder to call than games decided by a turn-5
nuclear exchange. This is the `critic_base_rate` trap in another costume — a discrimination
metric compared across two different outcome distributions
([`method/measurement_pitfalls.md`](../method/measurement_pitfalls.md)). `explained_variance`,
which does not depend on class balance the same way, runs the other way: 0.904 vs 0.761 at 30M
and 0.867 vs 0.788 at 40M, favouring the treatment.

## What this does not settle

* **One seed per arm.** Seed variance on this lineage is documented and not small
  ([`seed_variance_and_pooling.md`](seed_variance_and_pooling.md)); a three-fold gap at 40M is
  well outside it, a seat asymmetry at one point is not obviously so.
* **Early lead is not a final result.** Both arms climb from cold toward an anchor with 280M
  behind it. The registered second judgment point is 80M and the arm is resuming toward it.
* **The mechanism above is a correlation, not a demonstrated cause.** Fewer DEFCON-1 endings and
  a stronger policy move together here; nothing in this arm shows the ending mix is what *drives*
  the strength rather than another symptom of it. The clean test is an arm that keeps the default
  sharpened bands but penalises or masks self-inflicted DEFCON-1 endings, so the ending mix moves
  without the temperature moving.
* **It says nothing about the collapse.** Neither arm runs search, so this is not evidence about
  the X4b failure ([`P15_X4b_search_during_rl.md`](P15_X4b_search_during_rl.md)).

## Operational note

The arm was paused at 40.04M to give the GPU to the dense collapse replay, and resumed toward
80M. The pause landed on the first registered judgment point by coincidence, which is why this
entry exists now rather than at 80M.
