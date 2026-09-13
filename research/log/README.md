# Experiment log — index

`research/log/` is the **append-only history** of this project: what was tried, in order, and why
it stopped. It was split out of `research/experiments.md` (2,789 lines) on 2026-09-13, partitioned
by programme — one investigated question spanning several arms — with the original section numbers
kept so that every existing cross-reference, inside and outside these files, still names the right
entry. Nothing was rewritten: dead ends, abandoned arms, withdrawn claims and the self-corrections
that followed them stay exactly as written, and **no file here is edited to match current belief**.
For what is currently believed read `../findings/`; for how a measure is defined and what it can
bear read `../method/`; for what runs next read `../plans/`. The preamble of the
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
| [`europe_control_and_held_scoring.md`](europe_control_and_held_scoring.md) | — | E3-15 / E3-17: why the USSR takes Europe (West Germany is never contested), what the per-side held-scoring series actually says, and the forced human opening |

Several entries were moved to the old `../metrics.md` before this split and left stubs behind;
the stubs travelled with their section. `metrics.md` has since been split in turn, into
[`measurement_bugs.md`](measurement_bugs.md), [`variance_and_noise.md`](variance_and_noise.md),
[`P9_architecture.md`](P9_architecture.md), [`../method/running_experiments.md`](../method/running_experiments.md),
[`../method/human_play.md`](../method/human_play.md),
[`../method/measurement_pitfalls.md`](../method/measurement_pitfalls.md) and
[`../findings/architecture.md`](../findings/architecture.md); the stubs point at wherever their
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
