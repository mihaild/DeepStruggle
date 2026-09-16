# The question index — what we asked, which arms bear on it, what the answer is

Indexed by **question**, not by arm. [`runs.md`](runs.md) answers "what was `E3-20`?";
this file answers "what did we learn about pooled versus non-pooled?" — which is the lookup that
was impossible before 2026-09-16, because the arms that bear on one question are scattered across
several programmes and the answer was never assembled anywhere.

**Update discipline: rewritten in place.** A verdict changes when the evidence does; the evidence
itself stays in `log/`.

**Verdict vocabulary**, used strictly:

| | |
|:---|:---|
| **settled** | replicated on at least two seeds, and the effect exceeds the spread between same-condition arms |
| **suggestive** | the sign is consistent but one of those two conditions is unmet |
| **open** | asked and not answered, or answered and withdrawn |
| **confounded** | the arms differ in more than the factor named |
| **not run** | queued or proposed only |

---

## Training regime

| question | arms | verdict | where |
|:---|:---|:---|:---|
| **Does an opponent pool of past selves fix the side imbalance?** | E3-17-22/24/25/26 vs E3-20-22/27/28/29; earlier E3-19-22/23 | **settled on balance** — four pooled arms against four unpooled ones separate completely at 160M, 5.4 pp mean against 33.2 pp | [findings/pooling.md](findings/pooling.md) §3 |
| **Does an opponent pool make the agent stronger?** | the same eight | **open** — pooled is +98.7 Elo on arm means, but the pooled arms span 221 Elo among themselves, which fails the rule fixed in advance | [findings/pooling.md](findings/pooling.md) §3 |
| Does the imbalance settle with more steps? | E3-20-28, E3-20-29, E3-17-26 to 320M | **no** — both pooled arms moved further from balance, in the same direction, so not regression to the mean either | [findings/pooling.md](findings/pooling.md) §4 |
| What pool fraction and capacity? | all pooled arms ran frac 0.30, size 12 | **not run** — frac 0.15 and 0.50 were queued and never launched | [plans/P10](plans/P10_opponent_sampling.md) |
| Does the advantage signal recover on its own? | E3-18-22 (continue 160M→240M unchanged) | **open** — the arm ran and **no result was written down** | [findings/pooling.md](findings/pooling.md), *what the record does not say* |
| Does starting episodes from saved mid-game positions help? | `sp_pool_*`, `sp2_pool_*` | **settled negative** — −115 Elo, and the local gain is real but the allocation is wrong | [log/early_training_signal.md](log/early_training_signal.md) §3 |
| Is the NashPG KL penalty to `π_ref` worth keeping? | arm I (`eta` 0) vs arm H2 | **settled** — turning it off costs 169 Elo | [log/corrected_engine_arms_H_I.md](log/corrected_engine_arms_H_I.md) |
| Does filtering low-advantage samples help? | E3-07 (`--adv-filter-quantile 0.5`) | **settled** — +25 Elo at 80M, +24 at 160M, two seeds | [plans/P1](plans/P1_categorical_value_advantage_filtering.md) |
| Does a length-scaled decisiveness reward help? | `dec_turns20`, `dec_turns40` | **settled** — K=40 adopted; still ~200 Elo clear of everything trained since | [log/agent_deficiencies_and_decisiveness.md](log/agent_deficiencies_and_decisiveness.md) |
| Does confining a blunder penalty to its own turn help? | `abw_window_*`, `bw_*` | **retained, evidence stale** — measured on instruments since found broken | [log/early_training_signal.md](log/early_training_signal.md) §2.1 |
| Does prioritising decisive transitions help? | `dec_prio_on/off` | **open** — arms exist, conclusion never captured | [log/early_training_signal.md](log/early_training_signal.md) §2.2 |
| Does bootstrapping the value target from the mover's own view help? | E3-21-28 vs E3-20-28 | **not run** — registered, no checkpoint on disk | [runs.md](runs.md), *E3-21* |

## Architecture

| question | arms | verdict | where |
|:---|:---|:---|:---|
| Is the structured backbone worth it over an MLP? | E3-09 vs E3-01 | **settled** — structure is worth ~110 Elo at 2.6× fewer parameters | [findings/architecture.md](findings/architecture.md) |
| Do learned identity embeddings help? | E3-10, two seeds, 80M and 160M | **settled** — +107/+115 Elo, about what tripling the step budget buys | [findings/architecture.md](findings/architecture.md) |
| Does the graph layer need a self-transform? | E3-12 vs E3-10 | **settled** — +53/+83 Elo; `gconv1` recovery 60% → 94% | [findings/architecture.md](findings/architecture.md) |
| Can an attention read-out fix the pooled trunk? | E3-13 | **settled negative** — −14/−23 Elo; a single-query read-out is still pooling | [findings/architecture.md](findings/architecture.md) |
| Should per-entity heads replace the dense logit? | E3-14 | **settled negative** — −219 Elo; 64 floats became the only path from the trunk | [findings/architecture.md](findings/architecture.md) |
| …or be added as a residual? | E3-15 | **suggestive** — +181 Elo over E3-12, but **one seed**, and it is now the baseline | [findings/architecture.md](findings/architecture.md) |
| **How many map-graph layers?** | E3-15/E3-16/E3-17 | **open — the earlier answer is withdrawn.** The sign reverses with the seed | [findings/seed_variance.md](findings/seed_variance.md) |
| Can the 1,364 constant observation slots be dropped? | E3-11 vs E3-09 | **settled negative** — −40/−0 Elo, no throughput gain | [log/P9_architecture.md](log/P9_architecture.md) |
| Does a categorical value head help? | E3-06 (plus four VOID attempts) | **settled negative** — −28 Elo, mildly harmful | [findings/architecture.md](findings/architecture.md) |

## Observation

| question | arms | verdict | where |
|:---|:---|:---|:---|
| Is the dead turn-history slice worth anything? | arm E (v2.1) vs arm D | **settled** — neutral; removed | [log/observation_layout.md](log/observation_layout.md) |
| Is the decision context worth showing the network? | arm F (v2.2) vs arm E | **settled** — +91.7 Elo, confirmed on a second seed | [log/observation_layout.md](log/observation_layout.md) |
| Does showing a staged card help? | arms G/G2 (`staged_cards`) | **open** — the flag is not demonstrated; the bit is reserved, no code varies on it | [log/observation_layout.md](log/observation_layout.md) |

## What the agent understands

| question | evidence | verdict | where |
|:---|:---|:---|:---|
| **Does the critic see a provoked DEFCON-1 coming?** | `h2_480M_provoked_*` replays, four games | **no — at any node**, flat from 5M to 480M | [findings/defcon_blunders.md](findings/defcon_blunders.md) |
| Can the provoked case be credited by windowing? | E3-08, two seeds | **behaviour yes, strength no** — −68 Elo; window 5.6× too wide | [findings/defcon_blunders.md](findings/defcon_blunders.md) |
| Does the network price a country for the operation it is performing? | probes on E3-15 vs the control | **yes for E3-15**, backwards for the control | [findings/architecture.md](findings/architecture.md) |
| Does it know which region a scoring card scores? | probes, six regions | **yes for E3-15**; the control has Asia backwards | [findings/architecture.md](findings/architecture.md) |
| Does it coup discriminatingly? | probes | **no** — below-uniform battleground rate | [findings/architecture.md](findings/architecture.md) |
| Does it react to a scoring card in the opponent's hand? | probes | **no**, and the card's location *is* in the observation | [findings/architecture.md](findings/architecture.md) |

## Data

| question | arms | verdict | where |
|:---|:---|:---|:---|
| Is BC warmup on human games better than on self-play? | arms A/B | **not supported** — A beats B 57.4%; confounded with initialisation quality | [log/P7_human_bc_warmup.md](log/P7_human_bc_warmup.md) |
| Is continuous human injection worth it? | `inj_*`, `hum_*`, `g_*` | **settled negative** — the ablation puts the cost at 157–236 Elo | [log/P7_human_injection_and_its_cost.md](log/P7_human_injection_and_its_cost.md) |

## Measurement

| question | verdict | where |
|:---|:---|:---|
| **How many seeds does an arm need?** | two or three, for effects above ~40 Elo; below that, do not run it | [findings/seed_variance.md](findings/seed_variance.md) |
| Can `HeuristicBot` rate arms? | **no** — it systematically overrates weaker arms and has reversed a sign | [method/running_experiments.md](method/running_experiments.md) |
| Can self-play win rate measure side competence? | **no** — it measures balance; it reported −29 pp for an arm that lost 11 and +0.5 for one that gained 25 | [log/seed_variance_and_pooling.md](log/seed_variance_and_pooling.md) |
| Is a tournament reproducible from its configuration? | **no** — ~1.5 points of win rate between identical runs | [log/variance_and_noise.md](log/variance_and_noise.md) |
| Does search buy strength, and at what cost? | saturates at 96 sims; privilege is worth nothing | [log/search_cost_and_coverage.md](log/search_cost_and_coverage.md) |
