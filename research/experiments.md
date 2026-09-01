# Experiment Log

Running record of experiments actually run against this codebase, what they measured, and
what the result was. Companion to [`ideas_and_plans.md`](ideas_and_plans.md), which holds
the design intent; this file holds what happened when it was tried.

Only measurements taken in this repository belong here. An entry states what was compared,
how it was measured, and the number that came out, so that a later reader can tell a settled
question from an open one without rerunning anything.

---

## How to maintain this file

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

**Distrust a measurement before trusting a result.** Three separate diagnostics in this repo
reported confident numbers that were wrong (§1). Every one of them looked plausible. When an
experiment produces a surprising result, the cheapest first hypothesis is that the instrument
is broken — check that before building on the finding.

---

## 1. Measurement bugs found (read this before trusting any older number)

Four defects in the evaluation and diagnostic code, three of them the same defect in three
different files. All are fixed; all invalidate numbers logged before their fix.

### 1.1 Survivorship bias in batched diagnostics — three instances

The pattern: run N parallel envs, stop once `num_episodes` episodes have completed, with
`num_episodes` far below `num_envs`. Short games finish first, so the sample is the *fastest
N of num_envs* — and because envs auto-reset, a quick env can be counted repeatedly while a
slow one is never counted at all.

| file | called with | effect |
|:---|:---|:---|
| `ai/eval/position_diagnostics.py` | 30 episodes / 128 envs | mean final turn 3.30 vs 6.67 true; `frac_reaching_turn9` 0.000 vs 0.266; every late-game metric pinned to zero |
| `ai/eval/decisive_probe.py` | 30 episodes / 128 envs | instant-win take rate 89.5% vs 76.6% true |
| (both) | — | fixed by measuring the **first episode of every env**, draining until all finish, so sample size is `num_envs` and membership cannot depend on game length |

Both files had already had a *partial* fix that buffered positions per env and folded them in
on completion. That corrected attribution within an episode and left the stopping rule intact,
and both docstrings then claimed the length bias was handled. The decisive probe's docstring
even explained why the bias mattered — "decisive positions cluster near the end of a game" —
directly above the code that kept it.

Regression tests assert `num_games == num_envs` / `episodes == num_envs`, which can only hold
if each env contributed exactly one episode.

### 1.2 A policy was choosing its own dice in evaluation

At a `ROLL_DIE` chance node `ctx().decision_player` is `NONE`.
`TournamentEvaluator.play_matchup` fell back to `phasing_player` and asked an agent to pick an
action there, resolving it through `step_flat` — **134 such nodes per 20 games**, about 6.7 a
game. The vectorized runner resolves chance nodes inside the engine and never exposes one.

The two evaluation paths therefore disagreed by more than 25 points on the same deterministic
matchup, and the sequential path scored a *newer* snapshot at 0.34 against an older one, which
is what gave it away. Fixed by draining chance nodes; the paths are now bit-identical on
identical seeds. This is the same bug previously fixed in the decisive probe, in a second
place.

**Every in-run snapshot evaluation logged before this fix used the buggy path.**
`tools/tournament.py` was always on the correct path.

### 1.3 Evaluation consumed most of a training run

Snapshot evaluation used the one-game-at-a-time path (1.82 games/sec) rather than the
vectorized one (53.7 games/sec on the same 100 games, 30x). Cost also grew with run length,
because every snapshot was appended to the opponent list permanently.

Measured on the 3-hour A/B in §3.1: evaluation took **37%** of one arm's wall clock and
**61%** of the other's. The final evaluation faced 14 opponents and took 957s against a 900s
snapshot interval, leaving about one training iteration per interval.

Fixes: batched evaluation, `--eval-max-snapshot-opponents` (default 4), and evaluation
excluded from the `--duration-seconds` budget. After the fix, both arms of the §3.2 rerun
showed **zero** gaps over 20s and evaluation overhead of about 4%.

---

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

## 4. Current agent deficiencies (control, `sp2_pool_off/snapshot_final.pt`)

Measured with the corrected diagnostics of §1. arch v2, 78M steps, ~3h.

**Decisive decisions** (512 games, 196,130 decisions):

| | rate |
|:---|---:|
| instant wins taken | **76.6%** (301 of 393) — nearly a quarter of forced wins missed |
| avoidable losses walked into | **5.6%** (241 of 4,290) |

**Map coverage** — battlegrounds empty in late positions (turn >= 6), 512 games:

| | empty |
|:---|---:|
| Algeria | 97.7% |
| Saudi Arabia | 95.5% |
| Libya | 90.7% |
| Nigeria | 58.0% |
| Zaire | 53.8% |
| **West Germany** | **48.6%** |
| Mexico | 47.2% |
| **France** | **42.5%** |
| India | 39.9% |

Empty battlegrounds *plateau* at ~7.5 from turn 8 through turn 10: the agent stops expanding
once its early cards are spent. Two top-value Europe battlegrounds sit empty in nearly half of
late positions.

**Game shape.** Mean final turn 6.50; only 37.9% of games reach turn 8, 11.7% reach turn 10.
Five sample games are in `data/replays/92001-92005.tslog.json` — four are USSR 20-VP runaways
ending turns 4-6, one is a turn-3 self-inflicted DEFCON 1.

**Reading.** Missing a forced win is not an exploration failure: the winning action is legal,
in the mask, and one ply from terminal. It is a credit-assignment failure, and the same
blindness plausibly explains both the unclaimed battlegrounds and the early VP runaways.

### 4.1 Does more training fix it? — NO, it plateaus

Instant-win take rate across the control's own 13 snapshots (256 games each), which is the
cheapest available test of whether the deficiency is a matter of training budget:

| snapshot (s) | 0 | 900 | 1780 | 2661 | 3563 | 4449 | 5361 | 6169 | 7110 | 8092 | 9061 | 10008 | 10938 |
|:---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| take rate | 0.569 | 0.683 | 0.727 | 0.697 | 0.777 | 0.801 | 0.703 | 0.700 | 0.744 | 0.723 | 0.764 | 0.727 | 0.717 |
| avoid rate | 0.947 | 0.897 | 0.885 | 0.940 | 0.954 | 0.960 | 0.936 | 0.944 | 0.952 | 0.937 | 0.923 | 0.931 | 0.942 |

Take rate rises from 0.57 to roughly 0.75 within the first ~30M steps and then oscillates in
0.70-0.80 with **no trend over the remaining 45M steps**. Loss-avoidance is flat at ~0.94 from
initialization onward and never improves.

**This is the single most important number in this file for planning.** More of the same
training does not fix the deficiency, so a longer run of the current algorithm is not worth
buying, at any architecture, as a way of finding out.

### 4.2 Why forced wins are missed — not rarity, and not sampling

Three candidate explanations, measured on the same checkpoint (512 games each).

**Is the opportunity too rare to learn from? No.**

| | greedy | temperature 0.1 |
|:---|---:|---:|
| instant-win opportunities | 348 | 382 |
| share of all decisions | 0.175% | 0.196% |
| per game | 0.68 | 0.75 |
| games with at least one | 55.9% | 59.2% |
| implied count over a 78M-step run | **~137,000** | ~153,000 |

Opportunities are spread evenly across turns 2-10 (36-54 each), so they are not confined to
a phase the agent rarely reaches. ~137k instances in a single run is not a data-starved
regime.

For scale, avoidable losses arise about **11x more often** (4,290 per 512 games, 8.4 a game)
and are handled at 94.4% against 80.5% for forced wins. Frequency and competence do correlate
across the two, so rarity is not irrelevant -- but 137k opportunities is far from too few.

**Is it sampling noise from evaluating at temperature 0.1? No.**
Greedy take rate is **80.5%** against 79.3% when sampling. The failures survive argmax.

**It is low advantage.** The policy is not blind in general -- it puts a median **0.980**
probability mass on the winning action, above 0.5 in 77.0% of opportunities, against a
uniform baseline of 0.285 over ~6.7 legal actions. Only 4.6% of opportunities see it assign
under 0.01.

The misses are concentrated by *critic optimism*:

| critic v_win at the opportunity (+1 = certain win) | mean |
|:---|---:|
| win **was taken** (n=280) | +0.419 |
| win was **missed** (n=68) | **+0.626** |

The agent skips the immediate win precisely when it already believes it is winning
comfortably. If the critic says +0.63, ending the game now is worth only ~0.37 more than
playing on, so there is little gradient pressure to prefer it -- and the critic is
systematically overconfident in exactly those positions, since "probably winning" is not
"won".

**Consequence for the roadmap.** The fix is not more data and not more capacity. It is making
the terminal consequence visible where the critic is optimistic: one ply of lookahead reveals
it directly, which is what search buys. This also predicts the same failure in the unclaimed
battlegrounds -- a critic that cannot distinguish "winning" from "won" will not distinguish
"comfortable" from "needs another battleground" either.

---

## 4.3 Length-scaled terminal reward (`--decisiveness-turns`) — SETTLED, adopt K=40

**Question.** With gamma = 1 and terminal-only rewards the objective is indifferent to *when*
you win. Does making a result on turn T worth `1 - T/K` fix the forced-win floor?

**Setup.** arch v2, `--train-steps 78000000` (all arms landed on exactly 1,191 iterations),
512 envs. Control is `sp2_pool_off` from §3.2, reusable because training never touches
`classify_legal_actions`, so the §1 diagnostic fixes do not invalidate it.

**Result** (tournament 1,500 games/pair; take rate engine-verified on all three arms with
current code):

| | control | K=40 | K=20 |
|:---|---:|---:|---:|
| vs control | — | **54.6%** (z~3.6) | 43.5% |
| Elo | 1766.5 | **1818.8** | 1719.6 |
| vs HeuristicBot | 79.8% | **88.9%** | 76.9% |
| forced-win take | 81.2% | 72.4% | 69.2% |
| self-inflicted DEFCON-1 | 0.247 | 0.178 | 0.121 |
| final-scoring endings | 0.045 | 0.069 | 0.091 |

**Verdict.** Adopt K=40. K=20 suppresses blunders harder but loses games, so the slope
matters and K=40 is near the useful end of it. The guardrail did not trigger: final-scoring
wins rose rather than collapsed.

**The pre-registered prediction failed and the reason matters.** Take rate *fell* (81.2% ->
72.4%) while strength rose. §4.4 explains why that is not a regression.

## 4.4 Declining a forced win is usually free — which is why the metric misleads

**Question.** When a forced win is declined, does the declining player actually lose?

**Setup.** 400 games per configuration, engine-verified wins only (12 die streams per
candidate), split by side.

| configuration | declines | -> lost | cost of declining |
|:---|---:|---:|---:|
| control self-play | 55 | 10 | **18.2%** |
| K=40 self-play | 87 | 9 | **10.6%** |
| K=40 vs control: control | 36 | 6 | 16.7% |
| K=40 vs control: K=40 | 44 | 2 | **4.5%** |

**A declined win is still won 82-95% of the time.** Games actually thrown away this way are
~2.5% of the total (control: 10 of 400). So the ~20-30% miss rate is worth roughly *two
points* of win rate, not ten -- which is why the metric does not track strength, why
PI-MCTS improved win rate 85% without moving it, and why K=40 got stronger while its take
rate fell.

**K=40's lower take rate is better judgement, not worse play.** Its declines cost the game
4.5-10.6% against the control's 16.7-18.2%: it declines when it can afford to. The direction
is consistent across all three configurations, though each comparison alone is underpowered
(z ~ 1.2-1.75).

**Consequence.** Stop treating forced-win take rate as an optimisation target. It is a floor
check, worth watching for gross regressions, but it is close to uncorrelated with strength.

## 4.5 Side imbalance — USSR wins 60-65%

Measured incidentally in §4.4, and large enough to matter:

| configuration | US win rate | US / USSR forced-win opportunities |
|:---|---:|:---|
| control self-play | **35.0%** | 92 / 158 |
| K=40 self-play | 38.3% | 147 / 145 |
| K=40 vs control | 40.2% | — |

Real Twilight Struggle is close to balanced with a slight USSR edge, so 60-65% is well beyond
what the game explains: the agent plays US materially worse than USSR. The opportunity counts
say the same thing more sharply -- under the control, USSR gets 158 winning chances to the
US's 92, while K=40 evens that to 147/145 and lifts the US win rate by 3.3 points. Worth
investigating on its own; a US-side weakness this large is a bigger strength gap than
anything in §4.2.

---

## 5. Open questions

- **Does more *capacity* fix the forced-win misses?** More *training* does not (§4.1). The
  plateau shape argues the bottleneck is the learning signal rather than model size, but this
  was measured on v2 only, so a larger network having a higher plateau is not ruled out. The
  informative version of this test is to retrain a larger architecture *after* changing the
  training signal, not before: a v4 run under the current algorithm would most likely
  reproduce the plateau at some cost in hours.
- **Search.** One-ply lookahead would fix forced-win misses immediately but is exactly the
  hand-coding to avoid, and would not touch positional blindness. AlphaZero-style MCTS with the
  policy as prior addresses both, and gives the value head better targets.
- **Opponent league.** Raised alongside start-position sampling and never tested. Self-play
  against only the current policy lets both sides tacitly agree to ignore Algeria forever;
  nothing punishes it. PFSP against past snapshots does, and unlike start-position sampling it
  does not spend rollouts on states that occur in 5% of games.
- **Reward shaping** is deliberately out of scope: the aim is for the model to learn these
  decisions rather than have them encoded.

---

## 6. Method notes

- **Budget A/B arms by `--train-steps`, never by wall clock.** Steps/sec is policy-dependent,
  so a time budget hands the arms different amounts of training (§3.1).
- **Run arms in parallel.** Contention is symmetric and roughly halves throughput for both;
  prior paired runs matched within 6%. Sequential doubles turnaround and fixes nothing.
- **Give every model a distinct filename in a tournament.** Two checkpoints both named
  `snapshot_final` collided in the Bradley-Terry fit and were reported with identical Elo.
- **Beware `harvest()` in analysis scripts.** It calls `retire_stale()`, which drops older
  generations by design, so positions must be taken out of the pool after each round or they
  are lost. This silently reduced a 1,000-position sample to 91.
