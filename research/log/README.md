# Experiment log — index

`research/log/` is the **append-only history** of this project: what was tried, in order, and why
it stopped. It was split out of `research/experiments.md` (2,789 lines) on 2026-09-13, partitioned
by programme — one investigated question spanning several arms — with the original section numbers
kept so that every existing cross-reference, inside and outside these files, still names the right
entry. Nothing was rewritten: dead ends, abandoned arms, withdrawn claims and the self-corrections
that followed them stay exactly as written, and **no file here is edited to match current belief**.
For what is currently believed read `../findings/` — split since 2026-09-16 into
[`engine/`](../findings/engine/README.md), what the simulator does, and [`training/`](../findings/training/README.md),
what a training choice is worth; for how a measure is defined and what it can bear read
`../method/`; for what runs next read `../plans/`. The preamble of the
original file, including its maintenance rules, is preserved verbatim below the index.

## Where each section went

| file | sections | what it records |
|:---|:---|:---|
| [`early_training_signal.md`](early_training_signal.md) | §1–§3 | blunder window, decisive-transition prioritization, mid-game start sampling — the first training-signal knobs, pre-E1 engine |
| [`agent_deficiencies_and_decisiveness.md`](agent_deficiencies_and_decisiveness.md) | §4–§6 | the forced-win floor, the K=40 decisiveness reward, the side imbalance as first found, the dominance suite, open questions |
| [`engine_reanchor_and_human_control.md`](engine_reanchor_and_human_control.md) | §7–§8 | E1 re-anchor after the engine fixes; E2, the human corpus as the strong-player control that clears the engine of a USSR bias |
| [`P7_human_bc_warmup.md`](P7_human_bc_warmup.md) | §9 | E3 — BC warmup on the corpus, the 2M-step washout, how long to train BC, whether the corpus carries the dominance signal |
| [`P7_human_injection_and_its_cost.md`](P7_human_injection_and_its_cost.md) | §10–§11, §20–§22 | E5 injection dose, the cost of the dominance error, run-to-run variance, and the ablation showing injection cost 157–236 Elo |
| [`P7_replayer_conversion.md`](P7_replayer_conversion.md) | §8.6–§8.7.5, §9.6–§9.10, and others | how the human corpus is *read*: score reconciliation, hand reconstruction, corpus composition (moved here earlier) |
| [`critic_positional_value.md`](critic_positional_value.md) | §12–§14 | the 160M run probed: the critic prices control and access but not the road to it; the turn-2 event cards against a human baseline |
| [`critic_vs_policy_160M.md`](critic_vs_policy_160M.md) | §15–§18 | policy or critic, asked on real held positions; the counterfactual rounds played out; what the finished 160M run was worth |
| [`observation_layout.md`](observation_layout.md) | §19, §23–§24 | the dead history slice, opponent-card knowledge, and layouts v2.1 (neutral) and v2.2 (+91.7 Elo) |
| [`corrected_engine_arms_H_I.md`](corrected_engine_arms_H_I.md) | §25–§27 | arms H/H2 on v2.3 and the corrected engine, the NashPG KL-penalty ablation, and 240M → 480M |
| [`search_cost_and_coverage.md`](search_cost_and_coverage.md) | — | what search costs per decision, what fraction of decisions it is worth spending on, and why the +27pp figure does not transfer to a cheaper arm |
| [`europe_control_and_held_scoring.md`](europe_control_and_held_scoring.md) | — | E3-15 / E3-17: why the USSR takes Europe (West Germany is never contested), what the per-side held-scoring series actually says, and the forced human opening |
| [`seed_variance_and_pooling.md`](seed_variance_and_pooling.md) | — | the seed spread that withdrew the first pooling result, the US decay from 80M to 160M, and the analysis plan fixed before the 4 × 4 landed |
| [`../archive/E3_ladder/log/P15_X0_frozen_anchors.md`](../archive/E3_ladder/log/P15_X0_frozen_anchors.md) | — | P15-X0: three runs at 40M intervals to 320M against two frozen peer anchors — the oscillation with an amplitude at last, and why a field-averaged side gap hides five sixths of it |
| [`../archive/E3_ladder/log/P15_X0_round_robin.md`](../archive/E3_ladder/log/P15_X0_round_robin.md) | — | the full 24-model field behind P15-X0: Elo, head-to-head and per-side matrices |
| [`../archive/E3_ladder/log/P15_X4a_distillation.md`](../archive/E3_ladder/log/P15_X4a_distillation.md) | — | one offline round of expert iteration: +47.7 Elo from distilling a 96-sim searcher at card/play-mode nodes, out of a policy that already agreed with it 93.1% of the time and the two-arm washout test showing the edge decays to a dead heat after 20M steps while RL gives away the USSR side |
| [`../archive/E3_ladder/log/P15_X4a_where_the_search_signal_is.md`](../archive/E3_ladder/log/P15_X4a_where_the_search_signal_is.md) | — | a census of all 85,113 decisions in 200 games: top-1 agreement is uninformative (90.5–97.8% for every decision type) while KL varies 7×, and POINT_NODE placements carry 71.8% of the CE signal against card/play-mode's 20.9% |
| [`../archive/E3_ladder/log/P15_X4b_search_during_rl.md`](../archive/E3_ladder/log/P15_X4b_search_during_rl.md) | **+165.6 Elo** | the continuous form: an honest searcher supplies CE targets during RL and never acts, matched one-factor against the step-, seed- and cadence-identical control E3-26-28. Clears the pre-registered 25 Elo bar 6.6x, plays USSR at 61.9% against the control's 34.7%, and sheds 88 Elo of its peak in the final 5M — plus the off-by-one that voided two arms first |
| [`../archive/E3_ladder/log/P15_search_play_mode_bias.md`](../archive/E3_ladder/log/P15_search_play_mode_bias.md) | — | does the searcher dodge operations because it cannot place influence? No: over 18,097 play-mode decisions it shifts toward ops (+2.16 pp) and away from both EVENT and SPACE, taking each escape route from a placement decision less often than the raw policy |
| [`../archive/E3_ladder/log/P15_X4b_search_headroom.md`](../archive/E3_ladder/log/P15_X4b_search_headroom.md) | — | how much search still adds to a policy trained with search targets: flat across 5M-20M at ~58% win rate and KL ~0.036, which is what expert iteration looks like when the teacher improves with the student |
| [`../archive/E3_ladder/log/P15_X2_slow_anchor.md`](../archive/E3_ladder/log/P15_X2_slow_anchor.md) | — | does a 25x slower reference anchor prevent the collapse on its own? Interim: from scratch it has not crossed base_rate 0.80 by 72M where the 200k anchor crossed at 44.6M, and holds far more entropy early — one seed each and the seeds differ |
| [`../archive/E3_ladder/log/P15_control_per_seat.md`](../archive/E3_ladder/log/P15_control_per_seat.md) | — | which side actually degraded? Rated per seat against frozen anchors, the no-search control's **US** win rate falls 47.0% to 19.3% while USSR stays flat — the opposite of what critic_base_rate and a field-averaged side split appeared to say |
| [`../archive/E3_ladder/log/P15_setup_placement.md`](../archive/E3_ladder/log/P15_setup_placement.md) | — | opening influence placement is frozen in both from-scratch lineages — every point into one country, unchanged across 230M steps — and unfroze within 20M of a resume after 155M of nothing |
| [`../archive/E3_ladder/log/P15_two_seat_capacity.md`](../archive/E3_ladder/log/P15_two_seat_capacity.md) | — | can one network hold both seats? Yes: a student matches each teacher on its own seat with no interference, and a single checkpoint already plays 83.3% USSR / 79.7% US unaided — so the seat collapses are training dynamics, not capacity |
| [`../archive/E3_ladder/log/P15_arms_2026_09_17.md`](../archive/E3_ladder/log/P15_arms_2026_09_17.md) | — | one index for the day's dozen measurements: every arm from p28_200M and from scratch, what each showed against its matched ablation and against frozen snapshots, and what the two running arms decide |
| [`../archive/E3_ladder/log/P15_temperature_selfplay.md`](../archive/E3_ladder/log/P15_temperature_selfplay.md) | — | one checkpoint against itself at five sampling temperatures: greedy and T=0.1 are the same player (50.0%, 6.9 Elo), T=0.5 costs 134 Elo and T=1.0 costs 374 — which retires the temperature objection to comparing older tournament numbers with newer ones |
| [`../archive/E3_ladder/log/P15_X1_frozen_exploiter.md`](../archive/E3_ladder/log/P15_X1_frozen_exploiter.md) | — | training a USSR counter on purpose against a frozen @280M: +2.1 pp over the warm start (n.s.), and the discovery that the 'unanswered' US strategy was a 200-game artifact — at 1000 games it is level with a model that already existed |
| [`../archive/E3_ladder/log/P15_X0_selfplay_200M.md`](../archive/E3_ladder/log/P15_X0_selfplay_200M.md) | — | the strong anchor's own side split: E3-20-28 @200M against itself over 500 games, 51.0% US against 48.6% USSR, no detectable bias |
| [`../archive/E3_ladder/log/P15_X0_search_on_200M.md`](../archive/E3_ladder/log/P15_X0_search_on_200M.md) | — | honest 64-sim MCTS on E3-20-28 @200M against the frozen anchors and against its own base policy: +129.2 Elo, 64.8% over the raw policy, and the first search number taken after the searcher stopped proposing illegal moves |
| [`../archive/E3_ladder/log/P15_X4b_collapse_is_pool_starvation.md`](../archive/E3_ladder/log/P15_X4b_collapse_is_pool_starvation.md) | **24.0% -> 90.0%** | the dense replay launched to photograph the X4b collapse did not collapse, and that was the result: the original trained against an opponent pool of ONE, beating it 99.7%, because `--resume <dir>` rebuilt the pool from the parent directory. Severity and timing explained; a real, milder decline arrives at ~30M anyway and plateaus at 58% instead of falling to 24% |
| [`../archive/E3_ladder/log/P15_rollout_temperature.md`](../archive/E3_ladder/log/P15_rollout_temperature.md) | **46.0% vs 17.8%** | sampling rollouts AT the policy (0.8 1.2 0.7 1.1) against the default sharpened bands: ahead at every matched point after 10M, general across three lineages, and the mechanism is the ending mix -- 59.7% of control games end in DEFCON 1 against 28.9%, because sharpening makes one systematic blunder universal. Its opening is degenerate and it wins anyway |
| [`../archive/E3_ladder/log/P15_kl_domination.md`](../archive/E3_ladder/log/P15_kl_domination.md) | — | the KL regulariser reaching 130-900x the PPO surrogate on half of E3-31-28's iterations, invisible because `policy_loss` logs the surrogate alone; plus the search_ce spike diagnosis and the measurement that kills the CE feedback loop -- search targets stay sharp while the policy flattens away from them |

**Conclusions extracted on 2026-09-16.** Three topics were being looked up on their own and were
buried inside long journals. Their current verdicts now live in `../findings/` —
[`../archive/E3_ladder/findings/pooling.md`](../archive/E3_ladder/findings/pooling.md), [`../archive/E3_ladder/findings/seed_variance.md`](../archive/E3_ladder/findings/seed_variance.md) and
[`../archive/E3_ladder/findings/defcon_blunders.md`](../archive/E3_ladder/findings/defcon_blunders.md) — and each source section here carries a
one-line pointer under its heading. Nothing was removed: the setups, the predictions and the
retractions stay where they were written, which is the only property this directory has.

Several entries were moved to the old `../metrics.md` before this split and left stubs behind;
the stubs travelled with their section. `metrics.md` has since been split in turn, into
[`measurement_bugs.md`](measurement_bugs.md), [`variance_and_noise.md`](variance_and_noise.md),
[`P9_architecture.md`](P9_architecture.md), [`../method/running_experiments.md`](../method/running_experiments.md),
[`../method/human_play.md`](../method/human_play.md),
[`../method/measurement_pitfalls.md`](../method/measurement_pitfalls.md) and
[`../archive/E3_ladder/findings/architecture.md`](../archive/E3_ladder/findings/architecture.md); the stubs point at wherever their
entry landed. The verbatim preamble below still names `metrics.md`, and is left as written.

---

## Experiment Log

Running record of experiments actually run against this codebase, what they measured, and
what the result was. Companion to [`../archive/ideas_and_plans.md`](../archive/ideas_and_plans.md), which holds
the design intent; this file holds what happened when it was tried.

Only measurements taken in this repository belong here. An entry states what was compared,
how it was measured, and the number that came out, so that a later reader can tell a settled
question from an open one without rerunning anything.

**Scope.** This file is about how well the agents play, and what the human corpus says about that
by comparison. How that corpus is *read* -- reconciling the log's score against the engine's,
reconstructing the hands the log never states in full, and what the 300 files actually contain --
is [`P7_replayer_conversion.md`](P7_replayer_conversion.md). Entries that moved
there kept their original section numbers, and each left a stub here with the part that bears on
play.

---

### How to maintain this file

**Add an entry whenever an experiment finishes**, including one that failed or was
inconclusive — a negative result that is not written down gets re-run, and an inconclusive
one that is written down as positive is worse than useless.

Each entry should record:

1. **Question** — what was being decided, in one line.
2. **Setup** — arch, budget (steps or seconds), envs, checkpoints, what differed between arms.
3. **Result** — the numbers, with sample size and an error bar or significance where a
   comparison is being made.
4. **Verdict** — settled / suggestive / confounded, and the recommendation that follows.
5. **Caveats** — what the experiment cannot tell you. Single-seed runs, unmeasured
   variance, and known confounds go here.

**Revise an existing entry when a later finding invalidates it.** Do not silently delete a
superseded result: mark it and say what invalidated it. Two entries below
(§2.1, §3.1) are wrong in their original form and are kept precisely because the reason they
were wrong is the most useful thing in this file.

**Distrust a measurement before trusting a result.** Seven separate diagnostics in this repo
reported confident numbers that were wrong (now [`measurement_bugs.md`](measurement_bugs.md)). Every one
of them looked plausible. When an experiment produces a surprising result, the cheapest first
hypothesis is that the instrument is broken — check that before building on the finding.

**What belongs here, and what does not.** This file holds results: what a training, design or data
choice was worth, measured. Three companions hold the machinery, because it is read at different
moments and a result should not have to be read alongside the doubt about it.

| | |
|:---|:---|
| [`../method/measurement_pitfalls.md`](../method/measurement_pitfalls.md) | the checklist: what to verify before trusting a number |
| [`measurement_bugs.md`](measurement_bugs.md) | every instrument that reported confident nonsense, in full |
| [`variance_and_noise.md`](variance_and_noise.md) | how much of a rating is just where the run stopped |
| [`P7_replayer_conversion.md`](P7_replayer_conversion.md) | how the human corpus is read: score reconciliation, hand reconstruction, what the 300 files contain |
| [`../../engine/AGENTS.md`](../../engine/AGENTS.md) | engine design and rules defects, including why `CardLocation` encodes hand knowledge (§7) and the free-coup validation bug (§8) |
| [`E4_architecture_ab_result.md`](E4_architecture_ab_result.md) | — | P19's answer: the late-E3 architecture bundle is worth **+447 Elo** at matched budget and cold start, and `E4-03-01@80M` beats warm-started arms with 4x the steps. Both P21 anchors point at it |
| [`P21_M0_flat_mlp.md`](P21_M0_flat_mlp.md) | — | the ladder's floor: a flat MLP rates 1650-1666 at 80M, gains **+147 Elo** from 80M to 160M within seed, and at 160M beats the **E4 default architecture 64.5% using 46% of its GPU time** — the defaults are worse than an MLP per unit compute. Seed variance **16.6 Elo**, not the ~95 assumed |
| [`P21_M1_grouped.md`](P21_M1_grouped.md) | — | grouping the input by block is worth **+102 Elo** over a flat MLP and is **adopted** — but the gain is one seat (US +26 pp, USSR +0.8 pp) and it is an **intercept, not a slope**: the margin halves to +44 by 160M because M0's slope is twice M1's |
| [`P21_M2_lookup.md`](P21_M2_lookup.md) | — | the per-entity lookup is worth **+362 Elo** and **reaches the anchor**: M2@160M ties E4-03-01@80M using **65% of its wall clock**, with no attention, identity, pooling, shared encoder or graph. The bundle's repairs are unnecessary when pooling never destroys the information |
| [`P21_M2d_country_head_collapse.md`](P21_M2d_country_head_collapse.md) | — | the **country head alone is the whole mechanism** (2111.5, level with both-heads M2 at 2062.7, 18% faster and the best side balance in the ladder) while the **card head alone buys nothing**; removing the card head collapses 1 seed in 3, giving the first collapse that reproduces in 30 minutes |
| [`E4_collapse_attribution.md`](E4_collapse_attribution.md) | — | eight arms splitting `--seed` into initialisation / sampling / deals / opponent-draw, one at a time in both directions: **no single stream is sufficient** (conclusive, initialisation refuted both ways), but the necessity half is uninformative by construction -- all-clean has p=0.34 under the 12.5% base rate. Records the sizing a real answer would need |
| [`E4_collapse_is_recoverable.md`](E4_collapse_is_recoverable.md) | — | **the census measured entry, not collapse.** Seven arms scored COLLAPSED were continued 80M further and **five recovered**, including all four behind the 12.5% headline. A recovered collapse costs **−1.4 Elo** (0.05 pooled sd); a non-recovered one **−394**. Five predictors of permanence falsified; only `adv_std_raw` returning tracks it. Also records a third detector bug — a fixed 40-row tail window cannot tell "still collapsed" from "just exited" |
| [`P21_M2d_160M_slope.md`](P21_M2d_160M_slope.md) | — | seven M2d seeds rated at **both** 80M and 160M, so the slope is within-seed: **+159.8 Elo** (sd 33.7, six clean seeds), and all six clear the anchor while costing **under half its GPU-hours** at 4.3x the throughput. A collapse costs **-249** against the arm's own 80M self; entering the pinned state and escaping costs nothing detectable |
| [`P21_ladder_status.md`](P21_ladder_status.md) | — | **the ladder's standing state and the plan from here**: M1 is worth +106, the country head +282, the card head -12, and M2d reaches 99.5% of full M2 at 1.4x its throughput while closing 89% of the MLP-anchor gap with no attention, identity, pooling or graph. Lists the eight remaining rungs, the ~15 GPU-h they cost, and the four protocol amendments the M2d sweep forced |
| [`P21_M2abc_head_inputs.md`](P21_M2abc_head_inputs.md) | — | decomposes the country head's +282: **all three input groups are pulling weight** and no simplification survives. Context costs **-84 to -146** to remove (refuting the launch-time prediction), the hand-designed constants **-40 to -73**, and the minimal 15-input head still carries 54-74% of the mechanism. Six arms, zero collapses |
| [`P21_identity_rungs.md`](P21_identity_rungs.md) | — | **the identity vector is rejected**: M2.5, M2.5c and M2.5b all lose at 80M on both seeds, and M2.5c -- identity plus the card head, the configuration the plan's prediction was about -- is the *worst*. M2.5b collapses on two seeds where M2d runs clean, the ladder's first architecture-linked collapse, and neither ingredient collapses alone. Every arm that falls at 160M carries identity |
| [`P22_width_probe_and_card_lookup.md`](P22_width_probe_and_card_lookup.md) | — | **both P22 arms rejected.** Doubling `entity_proj_dim` loses **129–291 Elo**, so capacity is not what the card path lacks. The identity-keyed card lookup (P22-a) loses **−154 / −72** at 80M, but seed 3 closes to **−53** by 160M, gaining +234 against M2d's +133 over the same leg. That is one seed |
| [`P21_full_field.md`](P21_full_field.md) | — | **every rung rated in one field** (41 models): M2d is on top at every budget, every per-rung rejection holds, and the failures — collapsed M2.5b s3, the anchor's 160M leg, M2.5 s3@160M — are the bottom outliers. The table to quote when comparing across rungs |
| [`E4_search_distillation.md`](E4_search_distillation.md) | — | **online search distillation beats plain M2d on E4, on both seeds tried — durability unproven.** E3's X4b configuration beats its step-matched control at every snapshot on both seats (+54 to +196 on seed 3 over 8 snapshots, +157 to +262 on seed 5 over 5), and 20M steps with search beat 80M without. The gain is a level reached within 5M; search still beats the distilled net 59%, so headroom remains. E3's identical configuration collapsed 30–36M into its leg with a healthy pool; E4-28-03 went through that window to 40M without declining, one seed |
| [`E4_dynamics_ref_lambda.md`](E4_dynamics_ref_lambda.md) | — | phase 1 of two one-flag arms on M2d, seed 3, 80M: `--ref-update-freq 5M` is **−168 Elo** and `--gae-lambda 0.99` **−242**, both losing on both seats to seed-matched M2d. Neither fixes the oscillation usefully: the slow anchor raises entropy (against the ratchet hypothesis), and λ 0.99 leaves the US seat stuck at 2–4% for 45M steps with a live advantage signal |
| [`E4_decision_stream_census.md`](E4_decision_stream_census.md) | — | 386 decisions per game; **64% of card resolutions are ops** (influence 46%, coup 15%, realign 3%). Folding the op mode into the first placement saves **11.1%** of decisions (**15.5%** with coup/realign target heads). For credit assignment that equals raising λ from 0.98 to 0.983, a one-flag change |
| [`P21_M2d_setup_west_germany.md`](P21_M2d_setup_west_germany.md) | — | M2d s3 stops placing US setup influence in West Germany at **185M** (after wobbles at 165–180M), returns to it for one snapshot at 230M, then drops it again. Every snapshot is near-certain either way, so the opening is swapped wholesale rather than drifting. **The critic's preference between the two setups, read at every snapshot on 8 fixed deals, flips by up to 0.6 every 5–25M steps, and the policy follows it** (sign agrees in 12 of 17 snapshots, including the 185M switch and the 230M return). The setup follows an oscillating advantage signal |
| [`P21_M2d_240M.md`](P21_M2d_240M.md) | — | M2d seed 3 continued to 240M under its own name: **+31.9 Elo** over 160M, winning on both seats (61% / 53%) against +139.7 for the previous 80M, with the side gap closing to −0.1 pp. There is a **−74 dip at 200M** that fully recovers. n = 1 |

Sections that moved kept their original numbers and left a stub here carrying the part that bears
on a result. A new entry that turns out to be about an instrument rather than an agent belongs in
`metrics.md` from the start.

---
