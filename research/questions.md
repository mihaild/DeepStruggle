# The question index — what we asked, which arms bear on it, what the answer is

Indexed by **question**, not by arm. [`runs.md`](runs.md) answers "what was `E3-20`?";
this file answers "what did we learn about pooled versus non-pooled?" — which is the lookup that
was impossible before 2026-09-16, because the arms that bear on one question are scattered across
several programmes and the answer was never assembled anywhere.

**Update discipline: rewritten in place.** A verdict changes when the evidence does; the evidence
itself stays in `log/`.

**Every section is marked (A) or (B)**, because the two kinds of question decay differently:

| | | |
|:---|:---|:---|
| **(A)** | engine and instruments | pinned to a commit. When one of these answers changes, *absolute* numbers taken before it — Elo, win rates, distributional tables — are gone |
| **(B)** | models and training | a difference measured between two arms inside one engine. Expected to outlive the engine it was measured on |

That expectation is an assumption, stated with its evidence in
[`method/what_survives_an_engine_change.md`](method/what_survives_an_engine_change.md). The
findings are filed the same way: [`findings/engine/`](findings/engine/README.md) and
[`findings/training/`](findings/training/README.md).

**Verdict vocabulary**, used strictly:

| | |
|:---|:---|
| **settled** | replicated on at least two seeds, and the effect exceeds the spread between same-condition arms |
| **suggestive** | the sign is consistent but one of those two conditions is unmet |
| **open** | asked and not answered, or answered and withdrawn |
| **confounded** | the arms differ in more than the factor named |
| **not run** | queued or proposed only |

---

## Engine and instruments — (A)

Answers here are pinned to a commit, and when one changes it takes absolute numbers with it.

| question | evidence | verdict | where |
|:---|:---|:---|:---|
| **Does a CE term toward a search policy collapse RL training?** | X4b arms `E3-29-28` / `E3-31-28` / `E3-34-28` / `E3-35-28` | **yes, completely, with or without a healthy pool.** The famous early collapse was mostly pool starvation — `--resume <dir>` gave the run a pool of one it beat 99.7%, and the same config with a working pool scores 90.0% where the starved one scored 24.0% at the identical step. But the healthy-pool arm then collapses too, from 90.0% at 25M to ~11% at 36.5M, both seats. The pool changes *when*, not *whether* | [archive/E3_ladder/log/P15_X4b_collapse_is_pool_starvation.md](archive/E3_ladder/log/P15_X4b_collapse_is_pool_starvation.md) |
| **Should rollouts be sampled at the policy's own temperature?** | a completed 2x2: treatment and matched control at two seeds (`E3-33-30`/`E3-30-28`, `E3-36-31`/`E3-37-31`), 80M each | **it accelerates; persistence is unresolved.** At the registered 40M point the bands beat their own seed's control at **both** seeds (+11.0 pp z=3.7, +7.0 pp z=3.1) and stay significant through 60M. At the registered 80M point only seed A holds (+28.3 pp z=6.1); seed B's treatment plateaus at ~23% while its control climbs to 20.5%, leaving +3.0 pp (n.s.). **Do not change the default on 1-1** — run seeds three and four. Mechanism unknown: the ending-mix explanation failed its own control | [log/P15_rollout_temperature.md](log/P15_rollout_temperature.md) |
| **Should the value target be bootstrapped from the deciding player's own view?** | E3-21-28 against E3-20-28, matched seed | **settled — no, not by swapping the term.** The premise is right (V(s,US) + V(s,USSR) has mean absolute 0.144 where it should be 0) but the negation is what makes GAE telescope in an alternating-move game; replacing it raises return RMSE 43% and drops correlation with the realised outcome from 0.86 to 0.62. Fix the asymmetry at its source instead | [findings/training/value_bootstrap_perspective.md](findings/training/value_bootstrap_perspective.md) |
| **What does a training step actually cost?** | E3-17-25/26, E3-20-27/28, E3-21-28 | **settled** — 14,173 st/s unpooled on a 4090 against 10,300 with the pool and the value bootstrap; the 6-7k figures are all 3090 and three of them shared one card. Every search-arm duration quoted before this is ~1.7x pessimistic; the relative search results are unaffected | [findings/training/throughput.md](findings/training/throughput.md) |
| **Does an engine correctness change invalidate the arms that straddle it?** | P14 (mask/step collapse + the Missile Envy rule, `5938e52`), measured against `a09e15a` | **settled for this change — no.** 1,068 games / 385,812 steps across four policies, comparing outcome, length, action AND the legal mask at every step: 0 divergences. The engine letter stays E3 and E3-20-28 remains a matched baseline. Not a general licence — the answer came from measuring, not from the change looking small | [findings/engine/engine_change_decision_stream.md](findings/engine/engine_change_decision_stream.md) |
| Do checkpoint rankings survive a batch of rules fixes? | the 2026-09-05 fixes, 4 models, 6,000 games | **yes, for rankings** — the ordering held and the anchor rate read 90.2% against 88.9%. Everything measured through the self-play distribution had to be re-measured | [log/engine_reanchor_and_human_control.md](log/engine_reanchor_and_human_control.md) §7 |
| Does a *relative* training result survive a revision boundary? | — | **asserted, not measured.** No configuration has ever been trained on both sides of a letter boundary | [method/what_survives_an_engine_change.md](method/what_survives_an_engine_change.md) |
| What does an E2 checkpoint lose when rated on the E3 engine? | the two mandatory-choice cards | **unmeasured** — described as "small but not zero" and as flattering the E3 arms, with no number behind it, while `E2-02-21-480M` is the standing anchor | [findings/engine/engine_revisions.md](findings/engine/engine_revisions.md) |
| Is the engine biased toward the USSR? | the 300-game human corpus as a strong-player control | **no** — the corpus reproduces the US late-war recovery, which clears the engine of the hypothesis §4.7 left open | [log/engine_reanchor_and_human_control.md](log/engine_reanchor_and_human_control.md) §8 |
| Is `--auto-advance` outcome-neutral? | 256 games, deterministic policy | **yes per decision stream**, and it redefines a step: 4.2% fewer steps for the same game, so a budget with it on covers more game | [method/running_experiments.md](method/running_experiments.md) |
| Which instruments have reported confident nonsense? | ten-plus entries | **the full list, with what each voided** | [log/measurement_bugs.md](log/measurement_bugs.md), checklist in [method/measurement_pitfalls.md](method/measurement_pitfalls.md) |

## Training regime — (B)

| question | arms | verdict | where |
|:---|:---|:---|:---|
| **Does an opponent pool of past selves fix the side imbalance?** | E3-17-22/24/25/26 vs E3-20-22/27/28/29; earlier E3-19-22/23 | **settled on balance** — four pooled arms against four unpooled ones separate completely at 160M, 5.4 pp mean against 33.2 pp | [findings/training/pooling.md](findings/training/pooling.md) §3 |
| **Does an opponent pool make the agent stronger?** | the same eight | **open** — pooled is +98.7 Elo on arm means, but the pooled arms span 221 Elo among themselves, which fails the rule fixed in advance | [findings/training/pooling.md](findings/training/pooling.md) §3 |
| Does the imbalance settle with more steps? | E3-20-28, E3-20-29, E3-17-26 to 320M | **no** — both pooled arms moved further from balance, in the same direction, so not regression to the mean either | [findings/training/pooling.md](findings/training/pooling.md) §4 |
| What pool fraction and capacity? | all pooled arms ran frac 0.30, size 12 | **not run** — frac 0.15 and 0.50 were queued and never launched | [plans/P10](plans/P10_opponent_sampling.md) |
| Does the advantage signal recover on its own? | E3-18-22 (continue 160M→240M unchanged) | **open** — the arm ran and **no result was written down** | [findings/training/pooling.md](findings/training/pooling.md), *what the record does not say* |
| Does starting episodes from saved mid-game positions help? | `sp_pool_*`, `sp2_pool_*` | **settled negative** — −115 Elo, and the local gain is real but the allocation is wrong | [log/early_training_signal.md](log/early_training_signal.md) §3 |
| Is the NashPG KL penalty to `π_ref` worth keeping? | arm I (`eta` 0) vs arm H2 | **settled** — turning it off costs 169 Elo | [log/corrected_engine_arms_H_I.md](log/corrected_engine_arms_H_I.md) |
| Does filtering low-advantage samples help? | E3-07 (`--adv-filter-quantile 0.5`) | **settled** — +25 Elo at 80M, +24 at 160M, two seeds | [plans/P1](plans/P1_categorical_value_advantage_filtering.md) |
| Does a length-scaled decisiveness reward help? | `dec_turns20`, `dec_turns40` | **settled** — K=40 adopted; still ~200 Elo clear of everything trained since | [log/agent_deficiencies_and_decisiveness.md](log/agent_deficiencies_and_decisiveness.md) |
| Does confining a blunder penalty to its own turn help? | `abw_window_*`, `bw_*` | **retained, evidence stale** — measured on instruments since found broken | [log/early_training_signal.md](log/early_training_signal.md) §2.1 |
| Does prioritising decisive transitions help? | `dec_prio_on/off` | **open** — arms exist, conclusion never captured | [log/early_training_signal.md](log/early_training_signal.md) §2.2 |
| Does bootstrapping the value target from the mover's own view help? | E3-21-28 vs E3-20-28 | **not run** — registered, no checkpoint on disk | [runs.md](runs.md), *E3-21* |

## Architecture — (B)

| question | arms | verdict | where |
|:---|:---|:---|:---|
| Is the structured backbone worth it over an MLP? | E3-09 vs E3-01 | **settled** — structure is worth ~110 Elo at 2.6× fewer parameters | [findings/training/architecture.md](findings/training/architecture.md) |
| Do learned identity embeddings help? | E3-10, two seeds, 80M and 160M | **settled** — +107/+115 Elo, about what tripling the step budget buys | [findings/training/architecture.md](findings/training/architecture.md) |
| Does the graph layer need a self-transform? | E3-12 vs E3-10 | **settled** — +53/+83 Elo; `gconv1` recovery 60% → 94% | [findings/training/architecture.md](findings/training/architecture.md) |
| Can an attention read-out fix the pooled trunk? | E3-13 | **settled negative** — −14/−23 Elo; a single-query read-out is still pooling | [findings/training/architecture.md](findings/training/architecture.md) |
| Should per-entity heads replace the dense logit? | E3-14 | **settled negative** — −219 Elo; 64 floats became the only path from the trunk | [findings/training/architecture.md](findings/training/architecture.md) |
| …or be added as a residual? | E3-15 | **suggestive** — +181 Elo over E3-12, but **one seed**, and it is now the baseline | [findings/training/architecture.md](findings/training/architecture.md) |
| **How many map-graph layers?** | E3-15/E3-16/E3-17 | **open — the earlier answer is withdrawn.** The sign reverses with the seed | [findings/training/seed_variance.md](findings/training/seed_variance.md) |
| Can the 1,364 constant observation slots be dropped? | E3-11 vs E3-09 | **settled negative** — −40/−0 Elo, no throughput gain | [log/P9_architecture.md](log/P9_architecture.md) |
| Does a categorical value head help? | E3-06 (plus four VOID attempts) | **settled negative** — −28 Elo, mildly harmful | [findings/training/architecture.md](findings/training/architecture.md) |

## Observation — (B), on a third axis

| question | arms | verdict | where |
|:---|:---|:---|:---|
| Is the dead turn-history slice worth anything? | arm E (v2.1) vs arm D | **settled** — neutral; removed | [log/observation_layout.md](log/observation_layout.md) |
| Is the decision context worth showing the network? | arm F (v2.2) vs arm E | **settled** — +91.7 Elo, confirmed on a second seed | [log/observation_layout.md](log/observation_layout.md) |
| Does showing a staged card help? | arms G/G2 (`staged_cards`) | **open** — the flag is not demonstrated; the bit is reserved, no code varies on it | [log/observation_layout.md](log/observation_layout.md) |

## What the agent understands — (B)

| question | evidence | verdict | where |
|:---|:---|:---|:---|
| **Does the critic see a provoked DEFCON-1 coming?** | `h2_480M_provoked_*` replays, four games | **no — at any node**, flat from 5M to 480M | [findings/training/defcon_blunders.md](findings/training/defcon_blunders.md) |
| Can the provoked case be credited by windowing? | E3-08, two seeds | **behaviour yes, strength no** — −68 Elo; window 5.6× too wide | [findings/training/defcon_blunders.md](findings/training/defcon_blunders.md) |
| Does the network price a country for the operation it is performing? | probes on E3-15 vs the control | **yes for E3-15**, backwards for the control | [findings/training/architecture.md](findings/training/architecture.md) |
| Does it know which region a scoring card scores? | probes, six regions | **yes for E3-15**; the control has Asia backwards | [findings/training/architecture.md](findings/training/architecture.md) |
| Does it coup discriminatingly? | probes | **no** — below-uniform battleground rate | [findings/training/architecture.md](findings/training/architecture.md) |
| Does it react to a scoring card in the opponent's hand? | probes | **no**, and the card's location *is* in the observation | [findings/training/architecture.md](findings/training/architecture.md) |

## Data — (B), on instruments that are (A)

| question | arms | verdict | where |
|:---|:---|:---|:---|
| Is BC warmup on human games better than on self-play? | arms A/B | **not supported** — A beats B 57.4%; confounded with initialisation quality | [log/P7_human_bc_warmup.md](log/P7_human_bc_warmup.md) |
| Is continuous human injection worth it? | `inj_*`, `hum_*`, `g_*` | **settled negative** — the ablation puts the cost at 157–236 Elo | [log/P7_human_injection_and_its_cost.md](log/P7_human_injection_and_its_cost.md) |

## Measurement — (A)

| question | verdict | where |
|:---|:---|:---|
| **How many seeds does an arm need?** | two or three, for effects above ~40 Elo; below that, do not run it | [findings/training/seed_variance.md](findings/training/seed_variance.md) |
| Can `HeuristicBot` rate arms? | **no** — it systematically overrates weaker arms and has reversed a sign | [method/running_experiments.md](method/running_experiments.md) |
| Can self-play win rate measure side competence? | **no** — it measures balance; it reported −29 pp for an arm that lost 11 and +0.5 for one that gained 25 | [log/seed_variance_and_pooling.md](log/seed_variance_and_pooling.md) |
| Is a tournament reproducible from its configuration? | **no** — ~1.5 points of win rate between identical runs | [log/variance_and_noise.md](log/variance_and_noise.md) |
| Does search buy strength, and at what cost? **(B on an (A) instrument)** | saturates at 96 sims; privilege is worth nothing. **Both of those reverse the earlier reading** — privilege had been 100% vs 58.3% and 384 sims had beaten 96 — after re-measuring at `c7e3731`, past `8533a68`/`b6874af`/`9f78026`, on 120 games a cell instead of 12–40 | [log/search_cost_and_coverage.md](log/search_cost_and_coverage.md) §9 |

## Where the (A)/(B) labels are doing violence

Named rather than hidden, because a boundary with no exceptions listed is one nobody checked.

* **Observation** is neither. It is a representation decision, owner-held, and it invalidates a
  checkpoint outright rather than making it incomparable — silently, if the width is unchanged.
* **Search** sits in *Measurement* because its published numbers were invalidated twice by code
  fixes, once in the engine. The question it asks is (B).
* **Data** splits: whether human warmup helps is (B); whether the converter reads the corpus
  correctly is (A), and lives in [log/P7_replayer_conversion.md](log/P7_replayer_conversion.md).
* **Seed count** sits under *Measurement* but the number (~95 Elo between seeds) is a property of
  the training process, not of the apparatus.
* **The DEFCON questions** are (B) resting on an (A) taxonomy: the provoked/self-inflicted split
  is read off the engine's own `DEFCON_SUICIDE_PROVOKED` flag.
