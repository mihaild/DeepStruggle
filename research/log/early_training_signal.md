# Early training-signal experiments — blunder window, decisive transitions, mid-game starts

Records §1–§3 of the original `experiments.md`: the first training-signal knobs tried in this
project, all on `--arch v2` at ~78M steps, 512 envs, on the **pre-E1 engine** (before the six
`fix(engine)` commits re-anchored in §7) and the legacy 4,293-float observation. The question the
programme asks is whether changing *where* the RL signal is applied helps — confining a blunder
penalty to its own turn, prioritizing decisive transitions, and resuming a share of self-play from
saved mid-game positions. Section numbers are the ones these entries were first written under and
are kept so existing cross-references resolve. **This file is append-only history**: the confounded
first A/B in §3.1, its withdrawn conclusion, and the caveats added later stay exactly as written,
and nothing here is edited to match current belief.

---

## 1. Measurement bugs — moved to [`measurement_bugs.md`](measurement_bugs.md)

**Read it before trusting any number logged here.** Seven defects in the evaluation and
diagnostic code, all fixed, each of which produced a plausible wrong number rather than an
error: survivorship bias in three separate batched diagnostics, a policy choosing its own
dice in evaluation, evaluation consuming most of a training run, a model silently misreading
an observation of the wrong width, and a snapshot sort that returned the right answer for one
arm and the wrong one for another.

Entries below that predate a fix are marked where they are affected.

## 2. Training-signal experiments

### 2.1 Blunder window — RETAINED, but the evidence needs re-checking

**Question.** Does confining an unprovoked blunder's penalty to the turn it happened in help?

**Setup.** arch v2, 3h per arm, 512 envs, run in parallel (`abw_window_on` / `abw_window_off`).

**Result as recorded at the time.** Over the last four snapshots, self-inflicted DEFCON-1
losses as US totalled **6 with the window on against 46 with it off** (2/1/1/2 versus
5/10/17/14, rising without it). Win rate was a wash.

**Verdict.** Kept enabled. The blunder rate difference is large and in the intended direction.

**Caveat — this predates the fixes in §1.** The ending-mix counts came from in-run evaluation,
which at the time ran on the buggy path (§1.2) and a length-biased sample (§1.1). The two arms
were well matched on training (348 vs 362 iterations), so the comparison is not confounded the
way §3.1 was, but the absolute counts should not be quoted without re-measuring.

### 2.2 Decisive-transition prioritization — RESULT NOT RECORDED

`dec_prio_on` / `dec_prio_off` exist in `data/checkpoints/` (arch v2, 7200s, 512 envs, 635 vs
598 iterations, so well matched). The conclusion was not captured in a form this log can cite.
Re-derive from those checkpoints before relying on `--priority-alpha` either way.

---

## 3. Mid-game start sampling (`--start-pool-frac`)

**Idea.** Self-play from turn 1 reaches turn 10 in a small minority of games, so late-game
states are barely sampled. Resume a share of environments from saved turn-boundary positions,
mixed 50/15/15/10/10 across turns 1/4/6/8/10 (`DEFAULT_TURN_MIX`). Positions are stored
pre-deal so each resume draws fresh cards, and only *salvageable* ones are kept
(|VP| <= 10 and |region net| <= 20).

### 3.1 First A/B — CONFOUNDED, conclusion was wrong

**Setup.** arch v2, 3h wall clock per arm, 512 envs, parallel, one flag apart.

**Result as first reported.** The pool arm cut empty battlegrounds at turn 8 from 10.35 to
8.64 while the control sat flat at 10–11, and beat HeuristicBot 88.3% against the control's
68.4%. Head-to-head was a wash (51.1%).

**Why it was wrong.** The arms did not receive equal training: **67.1M steps versus 31.0M**,
because the control stalled — 9 iterations between 6,302s and 10,032s — under the evaluation
cost described in §1.3. Evaluation cost scales with how long a policy's games run, so the arm
playing longer games is systematically given less training. That is directional, not noise.

**Verdict.** Withdrawn. Superseded by §3.2.

### 3.2 Second A/B, step-budgeted — SETTLED, negative

**Setup.** arch v2, `--train-steps 78000000` (both arms reached exactly 1,190 iterations),
512 envs, parallel, `--eval-max-snapshot-opponents 4`, ~3.04h training each, zero stalls,
spans within 0.2%. All §1 fixes in place.

**Result** (2,000 games, SE ~1.1%):

| comparison | control | pool arm |
|:---|---:|---:|
| head-to-head | **67.3%** | 32.7% |
| vs HeuristicBot | **80.2%** | 70.5% |
| Elo | **1760.2** | 1645.1 |
| empty battlegrounds at turn 8 (last 6 snapshots) | **7.90** | 8.66 |

**Verdict.** Mid-game start sampling as configured is harmful. Leave `--start-pool-frac` at 0.

### 3.3 Does it help where it trained? — YES, and it still is not worth it

**Question.** The pool arm spent 10% of rollouts on turn-8 starts and the control none, so it
should be stronger there. Is it, or is the whole mechanism broken?

**Setup.** 1,000 salvageable turn-8 positions harvested from *pool-arm* self-play at training
temperature, each replayed from both sides (2,000 games) via
`BatchMatchRunner.play_parallel_matchup(start_states=...)`.

**Result.** Pool arm **53.4% ± 2.2%** (z = +3.0), 1068W-918L-14D, near-identical from each
side (US 53.5%, USSR 53.3%).

**Verdict.** The mechanism works and the local gain is real. The allocation is what is wrong:
+3.4 points in a regime that occurs in **5.6%** of self-play episodes, paid for with -17.3
points from the opening. If revisited, weight the turn mix toward positions that actually
occur rather than a flat 10%.

---
