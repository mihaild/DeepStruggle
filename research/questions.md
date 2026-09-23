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
| **Does a CE term toward a search policy collapse RL training?** | X4b arms `E3-29-28` / `E3-31-28` / `E3-34-28` / `E3-35-28` | **yes, completely, with or without a healthy pool.** The famous early collapse was mostly pool starvation — `--resume <dir>` gave the run a pool of one it beat 99.7%, and the same config with a working pool scores 90.0% where the starved one scored 24.0% at the identical step. But the healthy-pool arm then collapses too, from 90.0% at 25M to ~11% at 36.5M, both seats. The pool changes *when*, not *whether*. **E4:** `E4-28-03`, same configuration, shows no decline against its start through 35M into its leg (77% / 63% at 195M); not yet past E3's 36.5M ([log/E4_search_distillation.md](log/E4_search_distillation.md)) | [archive/E3_ladder/log/P15_X4b_collapse_is_pool_starvation.md](archive/E3_ladder/log/P15_X4b_collapse_is_pool_starvation.md) |
| **Should rollouts be sampled at the policy's own temperature?** | a completed 2x2: treatment and matched control at two seeds (`E3-33-30`/`E3-30-28`, `E3-36-31`/`E3-37-31`), 80M each | **it accelerates; persistence is unresolved.** At the registered 40M point the bands beat their own seed's control at **both** seeds (+11.0 pp z=3.7, +7.0 pp z=3.1) and stay significant through 60M. At the registered 80M point only seed A holds (+28.3 pp z=6.1); seed B's treatment plateaus at ~23% while its control climbs to 20.5%, leaving +3.0 pp (n.s.). **Do not change the default on 1-1** — run seeds three and four. Mechanism unknown: the ending-mix explanation failed its own control | [log/P15_rollout_temperature.md](archive/E3_ladder/log/P15_rollout_temperature.md) |
| **Should the value target be bootstrapped from the deciding player's own view?** | E3-21-28 against E3-20-28, matched seed | **settled — no, not by swapping the term.** The premise is right (V(s,US) + V(s,USSR) has mean absolute 0.144 where it should be 0) but the negation is what makes GAE telescope in an alternating-move game; replacing it raises return RMSE 43% and drops correlation with the realised outcome from 0.86 to 0.62. Fix the asymmetry at its source instead | [findings/training/value_bootstrap_perspective.md](archive/E3_ladder/findings/value_bootstrap_perspective.md) |
| **What does a training step actually cost?** | E3-17-25/26, E3-20-27/28, E3-21-28 | **settled** — 14,173 st/s unpooled on a 4090 against 10,300 with the pool and the value bootstrap; the 6-7k figures are all 3090 and three of them shared one card. Every search-arm duration quoted before this is ~1.7x pessimistic; the relative search results are unaffected | [findings/training/throughput.md](archive/E3_ladder/findings/throughput.md) |
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
| **Does an opponent pool of past selves fix the side imbalance?** | E3-17-22/24/25/26 vs E3-20-22/27/28/29; earlier E3-19-22/23 | **settled on balance** — four pooled arms against four unpooled ones separate completely at 160M, 5.4 pp mean against 33.2 pp | [findings/training/pooling.md](archive/E3_ladder/findings/pooling.md) §3 |
| **Does an opponent pool make the agent stronger?** | the same eight | **open** — pooled is +98.7 Elo on arm means, but the pooled arms span 221 Elo among themselves, which fails the rule fixed in advance | [findings/training/pooling.md](archive/E3_ladder/findings/pooling.md) §3 |
| Does the imbalance settle with more steps? | E3-20-28, E3-20-29, E3-17-26 to 320M | **no** — both pooled arms moved further from balance, in the same direction, so not regression to the mean either | [findings/training/pooling.md](archive/E3_ladder/findings/pooling.md) §4 |
| What pool fraction and capacity? | all pooled arms ran frac 0.30, size 12 | **not run** — frac 0.15 and 0.50 were queued and never launched | [plans/P10](archive/E3_ladder/plans/P10_opponent_sampling.md) |
| Does the advantage signal recover on its own? | E3-18-22 (continue 160M→240M unchanged) | **open** — the arm ran and **no result was written down** | [findings/training/pooling.md](archive/E3_ladder/findings/pooling.md), *what the record does not say* |
| Does starting episodes from saved mid-game positions help? | `sp_pool_*`, `sp2_pool_*` | **settled negative** — −115 Elo, and the local gain is real but the allocation is wrong | [log/early_training_signal.md](log/early_training_signal.md) §3 |
| Is the NashPG KL penalty to `π_ref` worth keeping? | arm I (`eta` 0) vs arm H2 | **settled** — turning it off costs 169 Elo | [log/corrected_engine_arms_H_I.md](log/corrected_engine_arms_H_I.md) |
| Does filtering low-advantage samples help? | E3-07 (`--adv-filter-quantile 0.5`) | **settled** — +25 Elo at 80M, +24 at 160M, two seeds | [plans/P1](archive/E3_ladder/plans/P1_categorical_value_advantage_filtering.md) |
| Does a length-scaled decisiveness reward help? | `dec_turns20`, `dec_turns40` | **settled** — K=40 adopted; still ~200 Elo clear of everything trained since | [log/agent_deficiencies_and_decisiveness.md](log/agent_deficiencies_and_decisiveness.md) |
| Does confining a blunder penalty to its own turn help? | `abw_window_*`, `bw_*` | **retained, evidence stale** — measured on instruments since found broken | [log/early_training_signal.md](log/early_training_signal.md) §2.1 |
| Does prioritising decisive transitions help? | `dec_prio_on/off` | **open** — arms exist, conclusion never captured | [log/early_training_signal.md](log/early_training_signal.md) §2.2 |
| Does bootstrapping the value target from the mover's own view help? | E3-21-28 vs E3-20-28 | **not run** — registered, no checkpoint on disk | [runs.md](runs.md), *E3-21* |

| **Does training past 80M keep paying?** | `E4-08-*@160M`, six seeds rated at both budgets; `E4-08-03@240M` | **settled for M2d to 160M** — **+159.8 Elo** from 80M to 160M, sd 33.7, measured *within-seed* so the ~100 Elo seed spread cancels out. **Still positive to 240M, but tapering**: +31.9 over 160M on one seed, winning both seats, after a −74 dip at 200M | [log/P21_M2d_160M_slope.md](log/P21_M2d_160M_slope.md), [log/P21_M2d_240M.md](log/P21_M2d_240M.md) |
| **How often does a side collapse happen?** | E4-08 census, 32 seeds; its 160M continuations; 7 censoring arms | **settled, and the earlier answer is withdrawn.** ~12.5% of seeds are *inside an entry episode* at 80M, but that is not an outcome: **5 of 7 recover** when continued, and a recovered collapse costs **−1.4 Elo** (0.05 pooled sd) against arms that never collapsed. Only non-recovery costs anything, **−394 Elo**, seen in 2 of 7 | [log/E4_collapse_is_recoverable.md](log/E4_collapse_is_recoverable.md) |
| **What predicts whether a collapse is permanent?** | 7 continuation arms | **open, and five candidates falsified** — depth, duration, exact-zero run length, exact-zero count and onset timing all fail. Only `adv_std_raw` returning to 0.10–0.20 tracks it | [log/E4_collapse_is_recoverable.md](log/E4_collapse_is_recoverable.md) |
| **What causes the side collapse?** | E4-10, eight arms, both directions | **open** — **no single seed stream is sufficient**, initialisation refuted in both directions. The necessity half cannot discriminate: at a 12.5% base rate an all-clean result has p = 0.34 | [log/E4_collapse_attribution.md](log/E4_collapse_attribution.md) |
| **Is the anchor stable past 80M?** | E4-03-01@160M | **open — one arm, no replicate.** It lost **356 Elo** over its second 80M with *every* collapse indicator healthy: the policy became indecisive rather than one-sided, and Elo mirrored policy entropy inversely. A second arm is ~3.9 GPU-h | [findings/training/entropy_inflation.md](findings/training/entropy_inflation.md) |

## Architecture — (B)

| question | arms | verdict | where |
|:---|:---|:---|:---|
| Is the structured backbone worth it over an MLP? | E3-09 vs E3-01; **re-measured on E4** by E4-05 vs E4-04/E4-03 | **settled, and the number changed.** E3 read ~110 Elo. On the E4 engine the *default* architecture is worth **+38** over a flat MLP and the late-E3 bundle **~+440** — and most of that is one mechanism, not the bundle | [log/P21_ladder_status.md](log/P21_ladder_status.md) |
| **Which part of the bundle carries it?** | E4-05…E4-09, plus a 32-seed census | **settled** — the **country** per-entity head, **+282** over grouped projections. The **card** head is **−7**: nothing. M2d reaches 99.5% of full M2 at 1.4× the throughput | [log/P21_ladder_status.md](log/P21_ladder_status.md) |
| **Does the board need a positional (non-pooled) path?** | E4-06 vs E4-05 | **settled** — **+109 Elo** for grouped positional projections over the flat MLP, with no pooling, no identity vector and no graph. This answered [P20](archive/E4_ladder/plans/P20_positional_board_encoder.md) without ever running it | [log/P21_M1_grouped.md](log/P21_M1_grouped.md) |
| **Can any part of the country head's input be removed?** | E4-13/E4-14/E4-15, two seeds each | **settled negative** — all three removals lose on both seeds. Context **−84…−146**, the per-type constants **−40…−73**, both **−87…−189**. But the 15 dynamic slots alone still carry 54–74% of the head | [log/P21_M2abc_head_inputs.md](log/P21_M2abc_head_inputs.md) |
| Do learned identity embeddings help? | E3-10, two seeds, 80M and 160M | **settled** — +107/+115 Elo, about what tripling the step budget buys | [findings/training/architecture.md](archive/E3_ladder/findings/architecture.md) |
| Does the graph layer need a self-transform? | E3-12 vs E3-10 | **settled** — +53/+83 Elo; `gconv1` recovery 60% → 94% | [findings/training/architecture.md](archive/E3_ladder/findings/architecture.md) |
| Can an attention read-out fix the pooled trunk? | E3-13 | **settled negative** — −14/−23 Elo; a single-query read-out is still pooling | [findings/training/architecture.md](archive/E3_ladder/findings/architecture.md) |
| Should per-entity heads replace the dense logit? | E3-14 | **settled negative** — −219 Elo; 64 floats became the only path from the trunk | [findings/training/architecture.md](archive/E3_ladder/findings/architecture.md) |
| …or be added as a residual? | E3-15 | **suggestive** — +181 Elo over E3-12, but **one seed**, and it is now the baseline | [findings/training/architecture.md](archive/E3_ladder/findings/architecture.md) |
| **How many map-graph layers?** | E3-15/E3-16/E3-17 | **open — the earlier answer is withdrawn.** The sign reverses with the seed | [findings/training/seed_variance.md](archive/E3_ladder/findings/seed_variance.md) |
| Can the 1,364 constant observation slots be dropped? | E3-11 vs E3-09 | **settled negative** — −40/−0 Elo, no throughput gain | [log/P9_architecture.md](log/P9_architecture.md) |
| Does a categorical value head help? | E3-06 (plus four VOID attempts) | **settled negative** — −28 Elo, mildly harmful | [findings/training/architecture.md](archive/E3_ladder/findings/architecture.md) |

## Observation — (B), on a third axis

| question | arms | verdict | where |
|:---|:---|:---|:---|
| Is the dead turn-history slice worth anything? | arm E (v2.1) vs arm D | **settled** — neutral; removed | [log/observation_layout.md](log/observation_layout.md) |
| Is the decision context worth showing the network? | arm F (v2.2) vs arm E | **settled** — +91.7 Elo, confirmed on a second seed | [log/observation_layout.md](log/observation_layout.md) |
| Does showing a staged card help? | arms G/G2 (`staged_cards`) | **open** — the flag is not demonstrated; the bit is reserved, no code varies on it | [log/observation_layout.md](log/observation_layout.md) |

## What the agent understands — (B)

| question | evidence | verdict | where |
|:---|:---|:---|:---|
| **Does the critic see a provoked DEFCON-1 coming?** | `h2_480M_provoked_*` replays, four games | **no — at any node**, flat from 5M to 480M | [findings/training/defcon_blunders.md](archive/E3_ladder/findings/defcon_blunders.md) |
| Can the provoked case be credited by windowing? | E3-08, two seeds | **behaviour yes, strength no** — −68 Elo; window 5.6× too wide | [findings/training/defcon_blunders.md](archive/E3_ladder/findings/defcon_blunders.md) |
| Does the network price a country for the operation it is performing? | probes on E3-15 vs the control | **yes for E3-15**, backwards for the control | [findings/training/architecture.md](archive/E3_ladder/findings/architecture.md) |
| Does it know which region a scoring card scores? | probes, six regions | **yes for E3-15**; the control has Asia backwards | [findings/training/architecture.md](archive/E3_ladder/findings/architecture.md) |
| Does it coup discriminatingly? | probes | **no** — below-uniform battleground rate | [findings/training/architecture.md](archive/E3_ladder/findings/architecture.md) |
| Does it react to a scoring card in the opponent's hand? | probes | **no**, and the card's location *is* in the observation | [findings/training/architecture.md](archive/E3_ladder/findings/architecture.md) |

## Data — (B), on instruments that are (A)

| question | arms | verdict | where |
|:---|:---|:---|:---|
| Is BC warmup on human games better than on self-play? | arms A/B | **not supported** — A beats B 57.4%; confounded with initialisation quality | [log/P7_human_bc_warmup.md](log/P7_human_bc_warmup.md) |
| Is continuous human injection worth it? | `inj_*`, `hum_*`, `g_*` | **settled negative** — the ablation puts the cost at 157–236 Elo | [log/P7_human_injection_and_its_cost.md](log/P7_human_injection_and_its_cost.md) |

## Measurement — (A)

| question | verdict | where |
|:---|:---|:---|
| **How many seeds does an arm need?** | two or three, for effects above ~40 Elo; below that, do not run it | [findings/training/seed_variance.md](archive/E3_ladder/findings/seed_variance.md) |
| Can `HeuristicBot` rate arms? | **no** — it systematically overrates weaker arms and has reversed a sign | [method/running_experiments.md](method/running_experiments.md) |
| Can self-play win rate measure side competence? | **no** — it measures balance; it reported −29 pp for an arm that lost 11 and +0.5 for one that gained 25 | [log/seed_variance_and_pooling.md](log/seed_variance_and_pooling.md) |
| Is a tournament reproducible from its configuration? | **no** — ~1.5 points of win rate between identical runs | [log/variance_and_noise.md](log/variance_and_noise.md) |
| Does search buy strength, and at what cost? **(B on an (A) instrument)** | saturates at 96 sims; privilege is worth nothing. **Both of those reverse the earlier reading** — privilege had been 100% vs 58.3% and 384 sims had beaten 96 — after re-measuring at `c7e3731`, past `8533a68`/`b6874af`/`9f78026`, on 120 games a cell instead of 12–40. **On E4:** 96-sim honest search beats `E4-08-03@240M` 64% (+99.5), and still beats the search-distilled net 59% | [log/search_cost_and_coverage.md](log/search_cost_and_coverage.md) §9–10 |
| **Does online search distillation help on E4?** | **settled for the first ~35M, open beyond it.** **Durability is the open half**: E3's identical configuration collapsed completely 30–36M into its leg with a healthy pool (row above), and E4-28-03 stopped at ~36M, no decline yet at 35M. E3's X4b configuration beats the step-matched control at every snapshot on both seats: +54 to +194 Elo on seed 3 (160→195M, 7 snapshots), +157 to +232 on seed 5 (80→100M, 4 snapshots). 20M steps with search beat 80M without. The gain is a level reached within 5M, not a slope. Cost: ~40x slower per step | [log/E4_search_distillation.md](log/E4_search_distillation.md) |

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
