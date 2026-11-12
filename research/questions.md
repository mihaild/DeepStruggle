# The question index — what we asked, which arms bear on it, what the answer is

Indexed by **question**, not by arm: [`runs.md`](runs.md) answers "what was `E7-A8-R1-S44@6400M+R34`?";
this file answers "what did we learn about search targets?". **Rewritten in place**: a verdict
changes when the evidence does; the evidence stays in `log/` (and, for E7 arms without a log, in the
arm's `runs.md` row).

Sections are marked **(A)** engine and instruments — pinned to a commit, and a changed answer takes
absolute numbers with it — or **(B)** models and training — a difference inside one engine, expected
to outlive it ([`method/what_survives_an_engine_change.md`](method/what_survives_an_engine_change.md)).

**Verdicts:** **settled** = replicated on ≥ 2 seeds and above the same-condition spread;
**suggestive** = consistent sign, one condition unmet; **open** = unanswered or withdrawn;
**confounded** = the arms differ in more than the factor; **not run**.

---

## Engine and instruments — (A)

| question | verdict | where |
|:---|:---|:---|
| Does an engine correctness change invalidate the arms that straddle it? | **settled for P14 — no**: 0 divergences in 385,812 steps, masks included. Measure, don't assume | [findings/engine/engine_change_decision_stream.md](findings/engine/engine_change_decision_stream.md) |
| Does a *relative* training result survive a revision boundary? | **suggestive yes on small boundaries**: E5-01 (+2 Elo across E4 → E5) and E7-01-44 (level across E6 → E7); never tested on a boundary that moves play | [method/what_survives_an_engine_change.md](method/what_survives_an_engine_change.md) |
| Do checkpoint rankings survive a batch of rules fixes? | **yes, for rankings** (90.2% against 88.9%); everything read off self-play had to be re-measured | [log/engine_reanchor_and_human_control.md](log/engine_reanchor_and_human_control.md) §7 |
| What does an E2 checkpoint lose on the E3 engine? | **unmeasured**, though `E2-02-21-480M` anchored every P1 arm | [findings/engine/engine_revisions.md](findings/engine/engine_revisions.md) |
| Is the engine biased toward the USSR? | **no** — the human corpus reproduces the US late-war recovery; ITS humans are 49.90% USSR | [log/engine_reanchor_and_human_control.md](log/engine_reanchor_and_human_control.md) §8, [method/human_play.md](method/human_play.md) |
| Is settling forced decisions (`auto_advance`) outcome-neutral? | **yes per decision stream**; it redefines a step (4.2% fewer per game) and is now the default | [method/running_experiments.md](method/running_experiments.md) |
| Which instruments have reported confident nonsense? | the full list, with what each voided | [log/measurement_bugs.md](log/measurement_bugs.md), [method/measurement_pitfalls.md](method/measurement_pitfalls.md) |

## Training — E5 to E7 — (B)

Most are late branches from the A8 plateau (`E7-A8-R1-S44@6400M`, 400M, against the plain leg), one
seed unless stated. Per-seat numbers are against the control's own self-play rate.

| question | arms | verdict | where |
|:---|:---|:---|:---|
| **How long must a recipe train?** | E7-01/02/17/19/20-44; A8; R48; A13+R48 | **much longer than it looks**: "saturated" at ~900M, then plateaus at ~3,700M (A4), 6,800M (A8), 9,200M (R48), 11,200M (A13+R48) | [findings/training/training_length_and_evaluation.md](findings/training/training_length_and_evaluation.md) |
| **Do SWAs and soups help?** | every E6–E7 line | **settled, as evaluation devices**: SWA +50–110 over snapshots early, 53.8% over the raw plateau snapshot; soups of branch SWAs +20–45 on the leaderboard | [findings/training/training_length_and_evaluation.md](findings/training/training_length_and_evaluation.md) |
| **Does the logit-gap penalty help?** | R34, R34-S45, R34-S46; E7-A8-R34-S44 | **settled as a late branch**: +7.6/+2.1, +5.6/+2.0, +5.6/+4.7 per seat; its soup 1650 against the heads soup's 1625. **From scratch it costs ~150–240 Elo** (code drift ruled out by a bit-identical re-run). Greedy only: at T = 1 it loses (36.6%) | [runs.md](runs.md) |
| **Do training targets from search help?** | E7-18-44, @6790M+R18, R29, R31, R32, R35, R36, R39, R45, R40–R42 | **a frozen, stronger teacher helps; the search's own margin barely distils.** PUCT/Gumbel targets from the learner itself flatten the policy (R29/R31 entropy loop); a frozen soup teacher fixes it (R32 +7.1/+8.4); the frozen R34 soup under the rollout root gave the largest gain (R39 +15.2/+8.3) — but the soup's raw policy as target is level with it (R45 48.8% against R39), and self-distilled rollout targets absorb nothing (R41 and R34 both 37% against the teacher). Search values into the critic (R35): level | [runs.md](runs.md), [log/E7_search_depth_and_value.md](log/E7_search_depth_and_value.md) |
| **What does search add at play time?** | Gumbel k8@256, rollout root | **+40 to +140 Elo**: Gumbel +41 on the heads soup; the rollout root +141 on the R34 soup, +88–103 on sharper networks; on R32 Gumbel *hurts* (its critic was not trained toward search) | [log/E7_gumbel_headroom.md](log/E7_gumbel_headroom.md), [log/E7_search_depth_and_value.md](log/E7_search_depth_and_value.md) |
| Can a logit cap fix saturated moves? | A9–A12 (± R28) | **settled negative**: the tanh cap cannot learn a saturated move (gradient ~1e-3), and every gradient past it (straight-through, leaks) drifts the shared trunk | [runs.md](runs.md) |
| Can the policy learn a trap whose bad move is saturated? | R28, R30, R33 | **only with a real loss term**: R28 alone 86–91% suicides (P(coup) = 1 to float precision); the gap penalty ×10 with the trap scenario (R33) 89 → 59%, by defending Cuba earlier | [runs.md](runs.md), [log/E7_cuban_missile_crisis_probe.md](log/E7_cuban_missile_crisis_probe.md) |
| Does per-decision turn credit (B4′) train in better moves? | R37, R38 | **no at this weight**: +1.7/+1.9 then level at 4× the signal; the clear-decision rate never falls (14%). The same pricing used at play time is +65 Elo | [runs.md](runs.md) |
| Is the plateau optimiser noise? | R46 (LR step-down), R47 (4× minibatch) | **partly**: both +15–25 Elo once (raw vs raw 56.9% / 56.0%), no higher ceiling (R47 to 8,000M) | [runs.md](runs.md) |
| Does a wider input projection help? | A13; A13+R48 | **suggestive yes**: the only branch still rising at 8,000M; on R48 the strongest greedy network (1679) but level with R48 itself | [runs.md](runs.md), [findings/training/trunk_depth_and_state.md](findings/training/trunk_depth_and_state.md) |
| Does a league help? | E5-17/19/21, E6-04/13, E7-21-44, @6400M+R19; `E8-A8-R102-S44@15210M+R120` (P44 0a) | **suggestive yes on J1's plateau (2026-11-12), one branch**: 800M with two seat-locked exploiters in ~15% of games reads 64.2% against the continuation's 60.9% on the old reference (4,000 games each) and 54.7% ± 0.6 head to head -- with the exploiters in their untrained seat half the time and a new opponent every 128 steps, as in every earlier league. Earlier: no measurable effect at a plateau (E7-21-44 vs its control 50.5%; exploiters rarely reached the publish bar); E5-21's acceptance is confounded with setup credit. Not yet attributed to the exploiters (P45 L1) | [log/P44_checks_2026-11-12.md](log/P44_checks_2026-11-12.md), [plans/P45_league.md](plans/P45_league.md), [log/E7_league_from_2800M.md](log/E7_league_from_2800M.md), [log/E5_21_setup_credit_plus_league.md](log/E5_21_setup_credit_plus_league.md) |
| Is the plateau the policy heads losing plasticity? | J1, J1', P44's control and its head-norm arm, R122/R123 (L2 toward the initial weights), R87 (head reset) | **no, by four readings**: the random-label probe decays mostly by 6,000M while J1 rises to ~13,000M; the head-norm arm is ahead of its control (55.3% at 3,200M) with a lower probe; J1's policy moves as much per update at 17,000M as at 4,000M; a head reset returns to the band. Holding the heads at fresh norm (L2) halves the strength at 400M. One seed each | [log/P44_checks_2026-11-12.md](log/P44_checks_2026-11-12.md), [plans/P44_plus_200_playdek.md](plans/P44_plus_200_playdek.md) |
| Are the plateau's seats exploitable? | R3, R25, R26, R27 | **the USSR seat is** (a US-locked exploiter +11.6 against it, a third of which is better US play in general); the US seat not shown. Distilling the exploiter's US play gains raw, not on the SWA | [runs.md](runs.md) |
| Does game-result setup credit teach openings? | E5-16, E5-21, E7-10/11/12-44 | **not evidence**: the mechanism adopts any scripted opening, including a −12.5 pp one (E7-11-44); setup work stopped | [log/E7_shallow_setup_lock.md](log/E7_shallow_setup_lock.md) |
| Does forcing events or the owner's openings teach play? | E7-29-44, R43 | **no**: forced events credited as the policy's own made it worse in both seats (43.4%); forced openings as environment: no gain (49.4%) | [log/P31_branch_arms_B1_B2.md](log/P31_branch_arms_B1_B2.md), [runs.md](runs.md) |
| Do card-conditioned heads help? | E7-30/31/32-44 (branch head), E7-33-44 → A8 | **behaviour yes, strength level then ahead**: Wargames' branch fixed; A8 level at 1,200–2,000M, 53.9% ahead of A4 at 4,000M, plateau 6,800M | [log/P31_branch_arms_B1_B2.md](log/P31_branch_arms_B1_B2.md) |
| Should rollouts be sampled at the policy's own temperature? | E5-06…E5-11, two seeds | **settled — flat 1.0, adopted 2026-09-27**: +55/+75 against a matched resume, +131/+171 from scratch at 240M | [log/E5_06_rollout_temperature.md](log/E5_06_rollout_temperature.md) |
| Smaller E7 levers | E7-07 (ops budget block), E7-09 (tokens), E7-13 (card-event aux), E7-14 (play-mode temp 2), E7-15 (win-only), E7-16 (VP potential), E7-23 (`--compile-update`), E7-28 (play-mode floor) | all **level or inside seed spread**; E7-13 a small positive, E7-14 and E7-23 slightly negative (compile stays opt-in) | [runs.md](runs.md), the P30 logs |

## Training — E3 and E4 — (B)

| question | arms | verdict | where |
|:---|:---|:---|:---|
| **Is an opponent pool required?** | E3 4 × 4; E4-01 | **yes** — pooled and unpooled separate completely on side balance (5.4 against 33.2 pp); without a pool one seat runs away and the critic loses skill (twice). Strength gain **open** (+98.7 on means, 221 Elo spread) | [archive/E3_ladder/findings/pooling.md](archive/E3_ladder/findings/pooling.md), [log/E4_pool_starvation_recurrence.md](log/E4_pool_starvation_recurrence.md) |
| Does a CE term toward a search policy collapse RL training? | E3 X4b arms; E4-28 | **E3: yes**, with or without a healthy pool (the early collapse was pool starvation; the healthy-pool arm still fell 90 → 11%). **E4: no**, +54 to +262 Elo over the step-matched control on two seeds, at ~40× the cost per step | [archive/E3_ladder/log/P15_X4b_collapse_is_pool_starvation.md](archive/E3_ladder/log/P15_X4b_collapse_is_pool_starvation.md), [log/E4_search_distillation.md](log/E4_search_distillation.md) |
| **How often does a side collapse happen, and does it cost?** | E4-08 census, 32 seeds; 7 continuations | **settled**: ~17% of seeds enter by 80M, 5 of 7 recover (−1.4 Elo), non-recovery −394. Earlier 12.5% "terminal" rate withdrawn | [findings/training/side_collapse.md](findings/training/side_collapse.md) |
| What causes / predicts a permanent collapse? | E4-10 (8 arms); 7 continuations | **open**: no single seed stream is sufficient; five predictors falsified | [log/E4_collapse_attribution.md](log/E4_collapse_attribution.md), [log/E4_collapse_is_recoverable.md](log/E4_collapse_is_recoverable.md) |
| Is the anchor stable past 80M? | E4-03-01@160M | **open, n = 1**: −356 Elo with every collapse indicator healthy (entropy inflation) | [findings/training/entropy_inflation.md](findings/training/entropy_inflation.md) |
| Does training past 80M keep paying (M2d)? | `E4-08-*@160M`, `E4-08-03@240M` | **settled to 160M**: +159.8 Elo within-seed (sd 33.7); tapering to 240M | [log/P21_M2d_160M_slope.md](log/P21_M2d_160M_slope.md), [log/P21_M2d_240M.md](log/P21_M2d_240M.md) |
| Slow π_ref, λ 0.99, smaller η? | E4-26/27, E4-29…34 | **rejected from scratch** (slow π_ref −168/−68, λ 0.99 −242/−286 by a US collapse); slow π_ref after 160M helped on a drained pool | [log/E4_dynamics_ref_lambda.md](log/E4_dynamics_ref_lambda.md), [log/E4_late_dynamics.md](log/E4_late_dynamics.md) |
| WoLF, seat balancing, per-seat normalisation, normaliser floors, entropy ceilings? | P25 bench, E4-35…E4-53 | see the P25 log | [log/P25_stress_bench.md](log/P25_stress_bench.md) |
| Should the value target be bootstrapped from the mover's own view? | E3-21-28, E3-22-28 | **settled no**: the negation is what makes GAE telescope; replacing it lost (per-player trajectories −520 Elo at 80M) | [archive/E3_ladder/findings/value_bootstrap_perspective.md](archive/E3_ladder/findings/value_bootstrap_perspective.md) |
| Is the NashPG KL penalty worth keeping? | arm I vs H2 | **settled** — off costs 169 Elo | [log/corrected_engine_arms_H_I.md](log/corrected_engine_arms_H_I.md) |
| Does filtering low-advantage samples help? | E3-07 | **settled** — +25 / +24 Elo, two seeds | [archive/E3_ladder/plans/P1_categorical_value_advantage_filtering.md](archive/E3_ladder/plans/P1_categorical_value_advantage_filtering.md) |
| Starting episodes from saved mid-game positions? | `sp_pool_*` | **settled negative** — −115 Elo | [log/early_training_signal.md](log/early_training_signal.md) §3 |
| A length-scaled decisiveness reward? | `dec_turns20/40` | **settled** (E1) — K=40 adopted | [log/agent_deficiencies_and_decisiveness.md](log/agent_deficiencies_and_decisiveness.md) |
| Confining a blunder penalty to its own turn; prioritising decisive transitions? | `abw_window_*`, `dec_prio_*` | **retained on stale evidence**; **open** (never written up) | [log/early_training_signal.md](log/early_training_signal.md) §2 |
| What pool fraction and capacity? | — | **not run** (all arms 0.3 / 12) | [archive/E3_ladder/plans/P10_opponent_sampling.md](archive/E3_ladder/plans/P10_opponent_sampling.md) |

## Architecture — (B)

| question | arms | verdict | where |
|:---|:---|:---|:---|
| **Which part of the structured backbone carries it?** | E4-05…E4-09, 32-seed census | **settled** — the **country** per-entity head, +282 over grouped projections; the card head −7. Grouped positional projections +109 over a flat MLP | [log/P21_ladder_status.md](log/P21_ladder_status.md), [log/P21_M1_grouped.md](log/P21_M1_grouped.md) |
| Can any part of the country head's input be removed? | E4-13/14/15, two seeds | **settled negative** — context −84…−146, per-type constants −40…−73 | [log/P21_M2abc_head_inputs.md](log/P21_M2abc_head_inputs.md) |
| **Do M2d's residual blocks add strength?** | E6-12 vs E6-03, two seeds; E7-06-44 | **settled negative** — the 0-block trunk wins (56.6% / 59.7%), and by more at saturation | [findings/training/trunk_depth_and_state.md](findings/training/trunk_depth_and_state.md) |
| Is the trunk the bottleneck for card × board interactions? | probes; E7-09, E7-13 | **representable (0.80, attention 0.90), not learned by RL (0.66)**; tokens and an aux target did not convert it into strength | [findings/training/trunk_depth_and_state.md](findings/training/trunk_depth_and_state.md) |
| E3 progression: identity, self-transform, attention read-out, per-entity heads, categorical value, dropping constants | E3-06…E3-15 | identity +107/+115; self-transform +53/+83; read-out −14/−23; heads replacing the logit −219, added as residual +181 (one seed); categorical value −28; dropping constant slots −40/−0 | [archive/E3_ladder/findings/architecture.md](archive/E3_ladder/findings/architecture.md) |
| How many map-graph layers? | E3-15/16/17 | **open** — the sign reverses with the seed | [archive/E3_ladder/findings/seed_variance.md](archive/E3_ladder/findings/seed_variance.md) |

## Observation, understanding, data — (B)

| question | verdict | where |
|:---|:---|:---|
| Dead turn-history slice (v2.1); decision context (v2.2); staged cards | neutral, removed; **+91.7 Elo** on two seeds; open (the bit is reserved, no code varies on it) | [log/observation_layout.md](log/observation_layout.md) |
| Does the critic see a provoked DEFCON-1 coming? Can windowing credit it? | **no, at any node**; behaviour yes, strength −68 Elo | [archive/E3_ladder/findings/defcon_blunders.md](archive/E3_ladder/findings/defcon_blunders.md) |
| Region pricing, scoring-region knowledge, discriminating coups, reacting to the opponent's scoring card (E3-15 probes) | yes; yes; no; no | [archive/E3_ladder/findings/architecture.md](archive/E3_ladder/findings/architecture.md) |
| BC warmup on human games vs self-play | **not supported** (57.4% for self-play; confounded with initialisation) | [log/P7_human_bc_warmup.md](log/P7_human_bc_warmup.md) |
| Continuous human injection | **settled negative** — −157 to −236 Elo | [log/P7_human_injection_and_its_cost.md](log/P7_human_injection_and_its_cost.md) |

## Measurement — (A)

| question | verdict | where |
|:---|:---|:---|
| How many seeds does an arm need? | two or three for effects above ~40 Elo | [archive/E3_ladder/findings/seed_variance.md](archive/E3_ladder/findings/seed_variance.md) |
| Can `HeuristicBot` or self-play win rate rate arms? | **no** — the bot overrates weaker arms and saturates; self-play measures balance | [method/measurement_pitfalls.md](method/measurement_pitfalls.md) |
| Is a tournament reproducible from its configuration? | **no** — ~1.5 points between identical runs | [log/variance_and_noise.md](log/variance_and_noise.md) |
| Does early search (PUCT, honest vs privileged) buy strength? | saturates at 96 sims, privilege worth nothing (both reversed after fixes at `611d83e`); on E4, 96-sim search beats `E4-08-03@240M` 64% | [log/search_cost_and_coverage.md](log/search_cost_and_coverage.md) §9–10 |

**Where the labels strain:** the observation is neither (an owner-held representation that
invalidates checkpoints outright); search sits under measurement because its numbers were twice
invalidated by code fixes; seed spread is a training property; the DEFCON questions rest on the
engine's `DEFCON_SUICIDE_PROVOKED` taxonomy.
