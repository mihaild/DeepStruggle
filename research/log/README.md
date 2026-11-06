# Experiment log — index

`research/log/` is the record of what was tried and what came out: one file per programme (one
question over several arms), named `PN_…` where a plan proposed it. Every measurement lands here
first and other documents link to it ([`../method/bookkeeping.md`](../method/bookkeeping.md)).
Results are not rewritten to match later belief: a correction is added next to the claim it
corrects, and withdrawn claims stay. Obsolete procedure (retired flags, layouts, stale status) is
trimmed; the earlier text is in git history. What is believed now is in
[`../findings/`](../findings/README.md); how to measure is in [`../method/`](../method/README.md).

An entry states the question, the setup, the result with its sample size and error, the verdict
and its caveats. Each file opens with a short **Conclusion**.

## Pre-E3: the first agents and the observation (the original `experiments.md` §1–§27)

Section numbers from the retired `experiments.md` (and `metrics.md`) are kept so old references
still resolve.

| file | § | what it records |
|:---|:---|:---|
| [`early_training_signal.md`](early_training_signal.md) | §1–§3 | blunder window kept; mid-game start sampling harmful (−115 Elo) |
| [`agent_deficiencies_and_decisiveness.md`](agent_deficiencies_and_decisiveness.md) | §4–§6 | forced-win floor, K=40 length-scaled reward adopted, the USSR imbalance, the dominance suite |
| [`engine_reanchor_and_human_control.md`](engine_reanchor_and_human_control.md) | §7–§8 | E1 re-anchor; E2: the human corpus clears the engine of a USSR bias |
| [`P7_human_bc_warmup.md`](P7_human_bc_warmup.md) | §9 | BC warmup on the human corpus, its washout, what the corpus carries |
| [`P7_human_injection_and_its_cost.md`](P7_human_injection_and_its_cost.md) | §10–§11, §20–§22 | human-data injection, and the ablation showing it cost 157–236 Elo |
| [`P7_replayer_conversion.md`](P7_replayer_conversion.md) | §8.6–§8.7, §9.6–§9.10 | converting the human logs: score reconciliation, hand reconstruction |
| [`critic_positional_value.md`](critic_positional_value.md) | §12–§14 | the critic prices control, not the road to it; the turn-2 event cards |
| [`critic_vs_policy_160M.md`](critic_vs_policy_160M.md) | §15–§18 | policy or critic, by card; the 160M run ties the previous best |
| [`observation_layout.md`](observation_layout.md) | §19, §23–§24 | dead history slice; v2.1 neutral; v2.2 +91.7 Elo (all retired since) |
| [`corrected_engine_arms_H_I.md`](corrected_engine_arms_H_I.md) | §25–§27 | arms H/H2 on v2.3; the KL penalty is worth 169 Elo; 240M → 480M |
| [`P9_architecture.md`](../archive/E3_ladder/log/P9_architecture.md) | — | archived (E3): identity embeddings → self-transform → per-entity heads → no map graph |

## Instruments and noise

| file | what it records |
|:---|:---|
| [`measurement_bugs.md`](measurement_bugs.md) | every instrument that reported confident nonsense, and what stops each recurring |
| [`variance_and_noise.md`](variance_and_noise.md) | tournament noise (~1.5 points), within-run oscillation, seed spread (~15 Elo) |
| [`seed_variance_and_pooling.md`](seed_variance_and_pooling.md) | the seed spread that withdrew the first pooling result; the US decay 80M → 160M |
| [`held_scoring_losses.md`](held_scoring_losses.md) | the zero-avoiding term doubles held-scoring losses in training games (J1 4-6% vs R95 2-3%) through sampling noise at the forced card choice; risk by scoring cards held and by the VP the card would cost |
| [`j5_attention.md`](j5_attention.md) | what J5's transformer attends to at 260M and 830M: regional layer 1, West Germany the layer-2 hub, by 830M the scoring cards (Europe Scoring above all) the deep layers' backbone |
| [`read_reliability_and_faster_reads.md`](read_reliability_and_faster_reads.md) | how much an early read predicts (ranks far-apart arms, not close ones: 3 of 9 pairs within 4 points at 800M reversed), J1/J5's non-transitive triangle, and a read protocol: matched-step head-to-head, own-past pace, margin-gated rules |

## E3

The P15 programme's logs are archived in [`../archive/E3_ladder/log/`](../archive/E3_ladder/README.md):
frozen anchors and the 24-model round robin (X0), a frozen exploiter (X1), a slow anchor (X2),
offline and online search distillation (X4a, X4b: **+165.6 Elo**), the X4b collapse traced to
opponent-pool starvation (24.0% → 90.0%), rollout temperature, KL domination, two-seat capacity,
per-seat control ratings, setup placement and temperature self-play.

| file | what it records |
|:---|:---|
| [`europe_control_and_held_scoring.md`](europe_control_and_held_scoring.md) | E3-15/17: an uncontested West Germany, and the critic collapsing to the base rate |
| [`search_cost_and_coverage.md`](search_cost_and_coverage.md) | what search costs per decision and how much coverage buys; re-measured on E4 and E5 |
| [`training_throughput_cpu.md`](training_throughput_cpu.md) | 39k → 70k steps/s on 1/8 of the CPU; two CUDA-graph hangs fixed |

## E4: the architecture ladder and training dynamics

| file | what it records |
|:---|:---|
| [`E4_architecture_ab_result.md`](E4_architecture_ab_result.md) | P19: the late-E3 architecture bundle is worth ~450 Elo at matched budget |
| [`P17_adapter_validation.md`](P17_adapter_validation.md) | P17: the same checkpoints score the same on both engines |
| [`P17_corpus_restoration.md`](P17_corpus_restoration.md) | P17: restoring the human corpus after the resolution merge |
| [`P17_ctx_stack_census.md`](P17_ctx_stack_census.md) | what `ctx_stack` holds, derived and censused |
| [`P21_M0_flat_mlp.md`](P21_M0_flat_mlp.md) | rung M0: a flat MLP beats the E4 default at 46% of its GPU time |
| [`P21_M1_grouped.md`](P21_M1_grouped.md) | rung M1: grouped input projections, +102 Elo, adopted |
| [`P21_M2_lookup.md`](P21_M2_lookup.md) | rung M2: the per-entity lookup, +362 Elo, reaches the anchor |
| [`P21_M2d_country_head_collapse.md`](P21_M2d_country_head_collapse.md) | the country head is the whole mechanism; the card head buys nothing |
| [`P21_M2abc_head_inputs.md`](P21_M2abc_head_inputs.md) | M2a/b/c: every input group of the country head pulls weight |
| [`P21_M2d_160M_slope.md`](P21_M2d_160M_slope.md) | M2d 80M → 160M: +159.8 Elo within seed; what a collapse costs |
| [`P21_M2d_240M.md`](P21_M2d_240M.md) | M2d to 240M: +31.9 Elo, still gaining slowly |
| [`P21_M2d_setup_west_germany.md`](P21_M2d_setup_west_germany.md) | the US setup flips West Germany on and off, following the critic |
| [`P21_identity_rungs.md`](P21_identity_rungs.md) | M2.5 / M2.5b / M2.5c: the identity vector is rejected |
| [`P22_width_probe_and_card_lookup.md`](P22_width_probe_and_card_lookup.md) | wider entity projections and the card lookup: neither beats M2d |
| [`P21_full_field.md`](P21_full_field.md) | every rung in one 41-model field |
| [`P21_ladder_status.md`](P21_ladder_status.md) | the ladder's standing state at 2026-09-21 |
| [`P23_E4_1_ab.md`](P23_E4_1_ab.md) | the E4.1 merged-influence view A/B from scratch |
| [`E4_collapse_attribution.md`](E4_collapse_attribution.md) | which randomness stream carries the M2d collapse: no single one |
| [`E4_collapse_is_recoverable.md`](E4_collapse_is_recoverable.md) | the census measured entry, not collapse: five of seven recovered |
| [`E4_collapse_census_per_seat.md`](E4_collapse_census_per_seat.md) | collapse census of the late-dynamics and E4.1 runs, read per seat |
| [`E4_round_robin_240M.md`](E4_round_robin_240M.md) | does the collapse show against a fixed reference? |
| [`E4_dynamics_ref_lambda.md`](E4_dynamics_ref_lambda.md) | a slower π_ref (−168) and λ 0.99 (−242) from scratch: both rejected |
| [`E4_late_dynamics.md`](E4_late_dynamics.md) | dynamics changes applied only after 160M |
| [`E4_clean_replicates.md`](E4_clean_replicates.md) | clean M2d replicates to 240M |
| [`E4_long_runs.md`](E4_long_runs.md) | M2d to 800M: TF32 drift of the logit level; centred per-entity heads adopted |
| [`E4_pool_starvation_recurrence.md`](E4_pool_starvation_recurrence.md) | E4-01-01 collapsed without an opponent pool: a launch error |
| [`E4_search_distillation.md`](E4_search_distillation.md) | online search distillation beats plain M2d on both seeds tried |
| [`E4_decision_stream_census.md`](E4_decision_stream_census.md) | how cards are played; merging the op mode would save 11% of decisions |
| [`E4_goal_probes.md`](E4_goal_probes.md) | the goal's acceptance probes on E4's best checkpoints |
| [`P25_pool_resume_bug.md`](P25_pool_resume_bug.md) | the opponent pool drained on every resume (fixed in `4eb58df`) |
| [`P25_stress_bench.md`](P25_stress_bench.md) | anti-collapse levers on the λ 0.99 recipe |
| [`P26_quick_screen.md`](P26_quick_screen.md) | what each throughput candidate is worth |
| [`P4_setup_arms.md`](P4_setup_arms.md) | setup credit (arm A) and a human setup anchor (arm B) |

## E5: setup credit and the league

| file | what it records |
|:---|:---|
| [`E5_01_engine_bump.md`](E5_01_engine_bump.md) | does the E4 → E5 rules change move training? |
| [`E5_06_rollout_temperature.md`](E5_06_rollout_temperature.md) | rollout temperature at the policy, from the plateau |
| [`E5_11_setup_lock_and_critic_views.md`](E5_11_setup_lock_and_critic_views.md) | a locked US opening that costs 5 points; the critic read from the wrong side |
| [`E5_11_selfplay_review.md`](E5_11_selfplay_review.md) | three self-play mistakes checked by playouts |
| [`E5_12_setup_credit_at_flat_temperature.md`](E5_12_setup_credit_at_flat_temperature.md) | per-side setup credit: no gain, and it locks the setup |
| [`E5_13_setup_entropy_floor.md`](E5_13_setup_entropy_floor.md) | a setup entropy floor keeps the opening moving, to worse places |
| [`E5_15_human_opening_resume.md`](E5_15_human_opening_resume.md) | the human opening on a trained model helps the US seat |
| [`E5_16_setup_mc_credit.md`](E5_16_setup_mc_credit.md) | setup credited by the game result repairs the US opening |
| [`E5_17_league_from_scratch.md`](E5_17_league_from_scratch.md) | the P24 league from scratch: level with E5-11 |
| [`E5_21_setup_credit_plus_league.md`](E5_21_setup_credit_plus_league.md) | game-result setup credit plus the league: the best E5 arm |
| [`P24_stage1_exploiter_collapse.md`](P24_stage1_exploiter_collapse.md) | the main exploiter collapses, and why |
| [`P27_ops_block_stage1.md`](P27_ops_block_stage1.md) | how the models spend an influence ops play |

## E6: strength recipes (SWA, soups, the shallow trunk)

| file | what it records |
|:---|:---|
| [`E6_01_finetune_check.md`](E6_01_finetune_check.md) | does a model that knows the corrected deck beat one that does not? |
| [`E6_04_league_seed44.md`](E6_04_league_seed44.md) | E5-21's recipe on E6, seed 44: it replicates |
| [`E6_04_43_seed43_replicate.md`](E6_04_43_seed43_replicate.md) | the adopted E6-04 recipe replicates on seed 43 |
| [`P28_step0_panel_and_E6_04_verdict.md`](P28_step0_panel_and_E6_04_verdict.md) | the E6 panel; E6-04-44 adopted |
| [`P28_step1_swa_and_search.md`](P28_step1_swa_and_search.md) | SWA beats every snapshot; search headroom shrinks |
| [`P28_step2a_lr_schedule.md`](P28_step2a_lr_schedule.md) | a stepped learning rate does not cut the spread |
| [`P28_step2b_ema_weights.md`](P28_step2b_ema_weights.md) | EMA weights beat raw snapshots; the window must be wide |
| [`E6_comprehensive_2026-09-29.md`](E6_comprehensive_2026-09-29.md) | every E6 arm on its snapshots and its 80M SWA |
| [`model_soups_2026-09-30.md`](model_soups_2026-09-30.md) | soups from a shared start: +45 over the best SWA |
| [`P29_bet1_wider_m2d.md`](P29_bet1_wider_m2d.md) | a 3.5× larger trunk is not better |
| [`P29_bet2_ownership_probes.md`](P29_bet2_ownership_probes.md) | the ownership target has not reached the trunk |
| [`P29_bet3_merged_view.md`](P29_bet3_merged_view.md) | the merged-influence view plays ~235 Elo weaker |
| [`P30_shallow_trunk.md`](P30_shallow_trunk.md) | M2d's residual blocks add nothing: the shallow net is stronger |
| [`P30_stage_A_trunk_probes.md`](P30_stage_A_trunk_probes.md) | the trunk drops much of the present; the critic is not short of it |
| [`P30_card_country_interaction.md`](P30_card_country_interaction.md) | card ↔ country interaction is learned, at about the right size |
| [`P30_card_board_targets.md`](P30_card_board_targets.md) | card × board targets: representable, not learned by RL |

## E7: the shallow line, its plateau, and search

**The line and its soups**

| file | what it records |
|:---|:---|
| [`E7_comprehensive_2026-09-30.md`](E7_comprehensive_2026-09-30.md) | 20 models on E7: the deep soup still leads |
| [`E7_02_44_saturation.md`](E7_02_44_saturation.md) | the shallow recipe to 1,200M: +110 Elo over 560M |
| [`E7_06_44_deep_to_saturation.md`](E7_06_44_deep_to_saturation.md) | the deep trunk saturates at ~560M, ~120 below the shallow one |
| [`E7_shallow_soup.md`](E7_shallow_soup.md) | the shallow soup at 1,200M: +61 over the deep soup |
| [`E7_line_to_2000M.md`](E7_line_to_2000M.md) | the seed-44 line to 4,800M: gains to ~3,700M, then a plateau |
| [`E7_league_from_2800M.md`](E7_league_from_2800M.md) | the P24 league from 2,800M adds nothing over its control |
| [`swa4720-4800_vs_soup1200.md`](swa4720-4800_vs_soup1200.md) | the 4,720–4,800M SWA beats the 1,200M soup 55.5% |
| [`E7_soup_4800M.md`](E7_soup_4800M.md) | a soup of three 4,800M branches: the strongest model at the time |
| [`E7_training_throughput_profile.md`](E7_training_throughput_profile.md) | where an iteration goes; the compile A/B (stays opt-in) |
| [`E7_shallow_setup_lock.md`](E7_shallow_setup_lock.md) | the USSR opening locks at Poland 6; unlocking attempts stopped |

**P30–P32 arms**

| file | what it records |
|:---|:---|
| [`P30_C4_ops_budget.md`](P30_C4_ops_budget.md) | C4: the ops budget in the observation |
| [`P30_C1_tokens.md`](P30_C1_tokens.md) | C1: card and country tokens with attention |
| [`P30_branch_arms_c2_play_mode_temp.md`](P30_branch_arms_c2_play_mode_temp.md) | C2's card-event head; exploration at play-mode decisions |
| [`P30_c2_card_probe_and_branch_soups.md`](P30_c2_card_probe_and_branch_soups.md) | C2's head reaches the trunk; soups of the branch arms |
| [`P30_win_only_and_vp_potential.md`](P30_win_only_and_vp_potential.md) | win-only reward and a small VP potential, from scratch |
| [`P30_oracle_critic_auc.md`](P30_oracle_critic_auc.md) | an oracle critic barely beats the honest one |
| [`P30_search_headroom_e7.md`](P30_search_headroom_e7.md) | what search adds on top of the best soup |
| [`P30_search_training_e7_18.md`](P30_search_training_e7_18.md) | search targets as a branch: strength moves from USSR to US |
| [`P31_branch_arms_B1_B2.md`](P31_branch_arms_B1_B2.md) | the play-mode floor and scenario seeding, from 4,390M |
| [`E7_league_exploiters_by_seat.md`](E7_league_exploiters_by_seat.md) | exploiting the heads plateau: one seat per exploiter |
| [`E7_cuban_missile_crisis_probe.md`](E7_cuban_missile_crisis_probe.md) | the heads soup does not understand Cuban Missile Crisis |
| [`E7_saturated_policy_and_caps.md`](E7_saturated_policy_and_caps.md) | a saturated policy cannot learn; the logit-gap loss (R33/R34) can |
| [`P32_cpu_checks.md`](P32_cpu_checks.md) | the forced-exit trap, Wargames at a lead of 6, Chernobyl off Europe |

**Search, leaks and the rollout root**

| file | what it records |
|:---|:---|
| [`search_performance_profile.md`](search_performance_profile.md) | what takes the time in search; the tree moved to C++ |
| [`E7_gumbel_headroom.md`](E7_gumbel_headroom.md) | a Gumbel root: +60 Elo at 64 evaluations, +10 more at 256 |
| [`E7_search_depth_and_value.md`](E7_search_depth_and_value.md) | search gains from depth; the critic cannot choose; the rollout root (+177) |
| [`E7_gchoice_distill.md`](E7_gchoice_distill.md) | one offline round of Gumbel-choice distillation: slightly weaker |
| [`E7_distill_ladder.md`](E7_distill_ladder.md) | the rollout root's corrections are local; an action-value head ranks them (r 0.80) and still cannot play them |
| [`R32_rollout_cpu_experiments.md`](R32_rollout_cpu_experiments.md) | R32 absorbed Gumbel's gain, not the rollout root's |
| [`E7_blunder_census.md`](E7_blunder_census.md) | engine-proven blunders are rare; the costly ones are late placements |
| [`E7_blunder_bank.md`](E7_blunder_bank.md) | 35 positions where the network's move costs 10+ points |
| [`E7_leak_by_kind_R34.md`](E7_leak_by_kind_R34.md) | where the R34 soup loses, by decision kind: placement 46% |
| [`E7_leak_by_kind_R39.md`](E7_leak_by_kind_R39.md) | R39: the leak halves; placement is 61% of it |
| [`E7_rollout_root_tuning_R39.md`](E7_rollout_root_tuning_R39.md) | the rollout root recovers the placement leak; 32 worlds and z2 (`4:32:4:z2`) |
| [`E7_rollout_race.md`](E7_rollout_race.md) | racing the worlds (16 for every candidate, then the challengers not trailing): 32 worlds' gain at 83% of the playouts; race 16 → 64 plays as 4:64 (+12 Elo over 4:32 on A13+R48) at ~1.5× 4:32's time |
| [`E7_searched_leak_R39.md`](E7_searched_leak_R39.md) | where R39 under the rollout root still loses |
| [`E7_searched_leak_A13_R48.md`](E7_searched_leak_A13_R48.md) | the same census on A13+R48 at 12,000M |
| [`E7_determinization_audit.md`](E7_determinization_audit.md) | determinization leaks nothing; a few deductions unused |
| [`P33_diagnostics.md`](P33_diagnostics.md) | the critic is the channel search keeps, blind where the searched player errs |

**Behaviour probes**

| file | what it records |
|:---|:---|
| [`expert_review_E7.md`](expert_review_E7.md) | a strong player's review: board judgment strong, card play leaks |
| [`critic_handover_calibration.md`](critic_handover_calibration.md) | the soup's critic: calibrated overall, blind to DEFCON-1 endings |
| [`decisive_win_misses_e7.md`](decisive_win_misses_e7.md) | half of forced wins missed, costing ~10 Elo |
| [`event_census_soup.md`](event_census_soup.md) | per-card event rates of the shallow soup |
| [`event_census_E7-02-44_E7-08-43.md`](event_census_E7-02-44_E7-08-43.md) | a fifth of the event rates move between seeds |
| [`event_census_E7-21-44.md`](event_census_E7-21-44.md) | longer training and the league each move a fifth |
| [`kal007_south_korea_control.md`](kal007_south_korea_control.md) | South Korea's control moves the KAL-007 headline weakly |
| [`star_wars_event_vs_ops_E7-02-44.md`](star_wars_event_vs_ops_E7-02-44.md) | Star Wars ahead in space: mostly for Ops |
| [`marshall_plan_E7_line.md`](marshall_plan_E7_line.md) | the US stopped headlining Marshall Plan at ~3,300M |
| [`E7_five_year_plan_timing.md`](E7_five_year_plan_timing.md) | early Five Year Plan plays are not a leak |
| [`per_checkpoint/`](per_checkpoint/README.md) | generated behaviour reports, one per rated checkpoint (`tools/scripts/checkpoint_report.py`) |

The subdirectories `E7_blunder_bank/`, `E7_leak_by_kind_*/`, `E7_rollout_root_tuning_R39/` and
`E7_searched_leak_*/` hold the generated tables their notes link to.
