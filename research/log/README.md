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
| [`P15_X0_frozen_anchors.md`](P15_X0_frozen_anchors.md) | — | P15-X0: three runs at 40M intervals to 320M against two frozen peer anchors — the oscillation with an amplitude at last, and why a field-averaged side gap hides five sixths of it |
| [`P15_X0_round_robin.md`](P15_X0_round_robin.md) | — | the full 24-model field behind P15-X0: Elo, head-to-head and per-side matrices |
| [`P15_X4a_distillation.md`](P15_X4a_distillation.md) | — | one offline round of expert iteration: +47.7 Elo from distilling a 96-sim searcher at card/play-mode nodes, out of a policy that already agreed with it 93.1% of the time and the two-arm washout test showing the edge decays to a dead heat after 20M steps while RL gives away the USSR side |
| [`P15_X4a_where_the_search_signal_is.md`](P15_X4a_where_the_search_signal_is.md) | — | a census of all 85,113 decisions in 200 games: top-1 agreement is uninformative (90.5–97.8% for every decision type) while KL varies 7×, and POINT_NODE placements carry 71.8% of the CE signal against card/play-mode's 20.9% |
| [`P15_X4b_search_during_rl.md`](P15_X4b_search_during_rl.md) | **+165.6 Elo** | the continuous form: an honest searcher supplies CE targets during RL and never acts, matched one-factor against the step-, seed- and cadence-identical control E3-26-28. Clears the pre-registered 25 Elo bar 6.6x, plays USSR at 61.9% against the control's 34.7%, and sheds 88 Elo of its peak in the final 5M — plus the off-by-one that voided two arms first |
| [`P15_search_play_mode_bias.md`](P15_search_play_mode_bias.md) | — | does the searcher dodge operations because it cannot place influence? No: over 18,097 play-mode decisions it shifts toward ops (+2.16 pp) and away from both EVENT and SPACE, taking each escape route from a placement decision less often than the raw policy |
| [`P15_X4b_search_headroom.md`](P15_X4b_search_headroom.md) | — | how much search still adds to a policy trained with search targets: flat across 5M-20M at ~58% win rate and KL ~0.036, which is what expert iteration looks like when the teacher improves with the student |
| [`P15_X2_slow_anchor.md`](P15_X2_slow_anchor.md) | — | does a 25x slower reference anchor prevent the collapse on its own? Interim: from scratch it has not crossed base_rate 0.80 by 72M where the 200k anchor crossed at 44.6M, and holds far more entropy early — one seed each and the seeds differ |
| [`P15_control_per_seat.md`](P15_control_per_seat.md) | — | which side actually degraded? Rated per seat against frozen anchors, the no-search control's **US** win rate falls 47.0% to 19.3% while USSR stays flat — the opposite of what critic_base_rate and a field-averaged side split appeared to say |
| [`P15_temperature_selfplay.md`](P15_temperature_selfplay.md) | — | one checkpoint against itself at five sampling temperatures: greedy and T=0.1 are the same player (50.0%, 6.9 Elo), T=0.5 costs 134 Elo and T=1.0 costs 374 — which retires the temperature objection to comparing older tournament numbers with newer ones |
| [`P15_X1_frozen_exploiter.md`](P15_X1_frozen_exploiter.md) | — | training a USSR counter on purpose against a frozen @280M: +2.1 pp over the warm start (n.s.), and the discovery that the 'unanswered' US strategy was a 200-game artifact — at 1000 games it is level with a model that already existed |
| [`P15_X0_selfplay_200M.md`](P15_X0_selfplay_200M.md) | — | the strong anchor's own side split: E3-20-28 @200M against itself over 500 games, 51.0% US against 48.6% USSR, no detectable bias |
| [`P15_X0_search_on_200M.md`](P15_X0_search_on_200M.md) | — | honest 64-sim MCTS on E3-20-28 @200M against the frozen anchors and against its own base policy: +129.2 Elo, 64.8% over the raw policy, and the first search number taken after the searcher stopped proposing illegal moves |

**Conclusions extracted on 2026-09-16.** Three topics were being looked up on their own and were
buried inside long journals. Their current verdicts now live in `../findings/` —
[`pooling.md`](../findings/training/pooling.md), [`seed_variance.md`](../findings/training/seed_variance.md) and
[`defcon_blunders.md`](../findings/training/defcon_blunders.md) — and each source section here carries a
one-line pointer under its heading. Nothing was removed: the setups, the predictions and the
retractions stay where they were written, which is the only property this directory has.

Several entries were moved to the old `../metrics.md` before this split and left stubs behind;
the stubs travelled with their section. `metrics.md` has since been split in turn, into
[`measurement_bugs.md`](measurement_bugs.md), [`variance_and_noise.md`](variance_and_noise.md),
[`P9_architecture.md`](P9_architecture.md), [`../method/running_experiments.md`](../method/running_experiments.md),
[`../method/human_play.md`](../method/human_play.md),
[`../method/measurement_pitfalls.md`](../method/measurement_pitfalls.md) and
[`../findings/training/architecture.md`](../findings/training/architecture.md); the stubs point at wherever their
entry landed. The verbatim preamble below still names `metrics.md`, and is left as written.

---

## Experiment Log

Running record of experiments actually run against this codebase, what they measured, and
what the result was. Companion to [`ideas_and_plans.md`](../archive/ideas_and_plans.md), which holds
the design intent; this file holds what happened when it was tried.

Only measurements taken in this repository belong here. An entry states what was compared,
how it was measured, and the number that came out, so that a later reader can tell a settled
question from an open one without rerunning anything.

**Scope.** This file is about how well the agents play, and what the human corpus says about that
by comparison. How that corpus is *read* -- reconciling the log's score against the engine's,
reconstructing the hands the log never states in full, and what the 300 files actually contain --
is [`experiments_replayer_conversion.md`](P7_replayer_conversion.md). Entries that moved
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
| [`experiments_replayer_conversion.md`](P7_replayer_conversion.md) | how the human corpus is read: score reconciliation, hand reconstruction, what the 300 files contain |
| [`../engine/AGENTS.md`](../../engine/AGENTS.md) | engine design and rules defects, including why `CardLocation` encodes hand knowledge (§7) and the free-coup validation bug (§8) |

Sections that moved kept their original numbers and left a stub here carrying the part that bears
on a result. A new entry that turns out to be about an instrument rather than an agent belongs in
`metrics.md` from the start.

---
