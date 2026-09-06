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

## 4.6 The US late-war recovery is missing

**Question.** Does the agent reproduce the standard arc -- USSR strong through the Early War,
US strengthening through Mid and Late War as its cards arrive?

**Setup.** 400 self-play games per arm, VP recorded at each turn boundary (US-positive).

Mean VP by turn:

| turn | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 |
|:---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| control | +0.15 | -1.02 | -1.87 | -3.03 | -3.58 | -3.91 | -3.11 | -2.49 | -4.47 |
| K=40 | +0.01 | -0.91 | -2.26 | -1.96 | -2.20 | -2.31 | -1.95 | -1.68 | -2.18 |

US win rate by how long the game ran:

| | ends turns 1-4 | ends turns 7-10 |
|:---|---:|---:|
| control | 44.8% | 39.5% |
| K=40 | 38.7% | 42.0% |

**Reproduced:** the USSR early lead. VP turns negative by turn 3 in both arms and the US is
ahead in only 35-40% of positions.

**Not reproduced:** the US comeback. The control gets *worse* late -- VP deepens to -3.91 by
turn 7 and long games favour the USSR more than short ones, the opposite of real play. K=40
shows a real but weak recovery (-2.31 at turn 7 to -1.68 at turn 9, long games favouring the
US). Neither arm reaches parity at any turn.

This is the same deficiency as the turn-8 battleground plateau (§4) seen from another angle:
the US late-war edge comes from cards that need board presence, and the agent has stopped
contesting the map by then. K=40 reaches turn 10 in 28% of games against the control's 14%,
so it gets twice the late-war experience -- plausibly part of why it is stronger, independent
of the blunder reduction.

**Caveat.** These are conditional-on-reaching statistics and the population shifts by turn:
US-winning games end sooner on 20 VP, so what survives to turn 10 skews USSR-favourable. The
within-arm trend and the ending-turn win rates are the more trustworthy signals.

**Setup rules are not the cause.** The optional +2 US bonus placement is implemented and
unconditional (`state_machine.cpp:467`), verified empirically: US makes 9 setup placements
(7 Western Europe + 2 bonus) to USSR's 6, on every seed.

## 4.7 The side imbalance is a symptom, and the encoding is not at fault

**Encoding audited, clean.** `extract_observation` is canonical (me vs opponent). Over 132
sampled positions, all seven my/opp board pairs -- influence 0/1, control 5/6, superpower
adjacency 8/9, can-place 19/20, can-coup 21/22, can-realign 23/24, control-deficit 26/27 --
swap exactly when perspective flips (77,616 comparisons, zero mismatches), and the signed
realignment column 2 negates correctly. The only unpaired perspective feature is column 7
(`is_coup_nuclear_hazard` for the mover), which is structurally identical for both sides.
The card block folds the opponent's hand into the unknown slot, so there is no
hidden-information leak. **The encoding does not explain the US deficit.**

**Nor does setup** -- the +2 US bonus placement is unconditional and verified (§4.6).

**Baselines locate the cause:**

| | US win rate | mean final turn |
|:---|---:|---:|
| random self-play | **51.3% +/- 5.1** | 2.73 |
| heuristic self-play | **22.5% +/- 5.0** | 8.23 |
| control (learned) | 37.4% | 6.28 |
| K=40 (learned) | 39.1% | 6.80 |

Random play is even, so there is no gross engine bias -- but random games end on turn 2.7,
before the asymmetry can express itself, so it is a weak control. The informative number is
the hand-written heuristic at **22.5%**: a non-learning player, in 8-turn games, is far more
USSR-skewed than either trained agent. The imbalance is therefore not an artifact of learned
policy; our agents are the *least* imbalanced players we have.

**Synthesis.** This is the same defect as §4.6 and the turn-8 battleground plateau. The USSR
early advantage is real and every agent reproduces it. The compensating US advantage is a
late-war one that must be *converted*, and none of these agents can convert it, so the USSR
edge stands unopposed. The ordering is consistent across three independent measurements:
K=40 has the best US win rate, the best late-war VP recovery, and reaches turn 10 most often.

**Not ruled out:** a subtle rules bug favouring the USSR that only shows in long games.
Random ends too early to test it and the heuristic is too weak to separate "engine bias" from
"cannot play the late war". Card behaviour is covered by the struggler differential tests, so
scoring or turn structure would be the place to look -- starting with whether the Mid/Late War
deck additions are introduced correctly.

## 4.8 Dominance suite: local decisions with a position-independent right answer

**Why.** Most evaluation here compares against "the better move", which depends on position, so
disagreement is not proof of error. Dominance pairs remove that: two options identical in
printed Ops, differing only in a way that is weakly better in *every* position. Preferring the
dominated one is wrong with no judgement to argue about. `ai/eval/dominance.py`; measurement
only, never fed to the agent.

**Rules.** At equal printed Ops, (a) discarding an opponent *recurring* event under Quagmire /
Bear Trap beats discarding your own or a neutral card, and (b) the same for what you spend on
the space track. Excluded from the dominant side: **Five Year Plan** (#5, the one recurring
event whose firing can help its non-owner), **one-time starred events** (removing them
permanently is a different and stronger argument), scoring cards, and the China Card. Eight
tests pin each exception.

**There is no rule about playing an opponent card for its Event** -- the engine already makes
that illegal, verified over 971 play-mode decisions on opponent cards with EVENT legal in none.

**Results** (200 self-play games each):

| | control | K=40 |
|:---|---:|---:|
| Quagmire: chose a dominated card | 37.5% | **30.7%** |
| Quagmire: pairs ranked wrongly | **64.5%** | 52.7% |
| Bear Trap: chose a dominated card | 32.8% | **18.2%** |
| Bear Trap: pairs ranked wrongly | 51.2% | 35.4% |
| Space: chose a dominated card | 18.1% | **12.6%** |

**Normalise by opportunity.** Only ~20% of space plays (498 of 2,356) offer an equal-Ops
opponent alternative at all. Against all space plays the violation rate reads 2.4%, which is
the wrong denominator and badly understates the error; against decidable cases it is 12.6-18.1%.

**Two findings.** The control's Quagmire pair ranking is **64.5% wrong -- worse than a coin
flip**, so it is not merely ignoring the relation but actively inverting it. And K=40 is better
on *every* dominance measure, which says the decisiveness reward improved local decision
quality and not only endgame behaviour -- a broader effect than §4.3 alone suggested.

Bear Trap is handled better than Quagmire in both models, matching the US-side weakness in
§4.5 and §4.7 on the same architecture and the same bit.

**Capacity is not the blocker.** Flipping the trap bit moves the policy by TV 0.13 (Quagmire)
to 0.36 (Bear Trap), so the network plainly conditions on it; `extract_features` fuses board,
card and global streams into one trunk and `policy_head` scores all 212 actions from it, so any
global feature can reach any card logit. It has the capacity and has not learned what to do
with it -- the same conclusion as §4.2 and §4.4, and the strongest argument yet for
demonstrations, since one human game shows the reversal that RL needs thousands to notice.

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
- **Replicate before believing a small gap.** A tournament is not reproducible from its
  configuration: deals are seeded, but the agents sample, so two identical 6,000-game runs differ
  by ~1.5 points on a matchup (§7.2). Treat that as the noise floor at 1,000 games a pair, not the
  binomial SE, which assumes away exactly this source of variation.
- **Enable `--auto-advance` freely.** It is outcome-neutral, verified bit-exact under a
  position-derived policy on the vectorized path (§7.1). It is also a smaller speed win than it
  looks (3.3% fewer batched steps).
- **Beware `harvest()` in analysis scripts.** It calls `retire_stale()`, which drops older
  generations by design, so positions must be taken out of the pool after each round or they
  are lost. This silently reduced a 1,000-position sample to 91.


---

## 7. Re-anchor after the engine changes (E1) — SETTLED

**Question.** Six `fix(engine)` commits (per-card headline decision frames, Defectors, Shuttle
Diplomacy, the scoring-card trap rule, deck refill) landed after every number in §3 and §4 was
taken. Do the checkpoint rankings survive, and are the old Elo anchors still usable?

**Setup.** `tools/tournament.py`, 4 models, 500 games per side per pair (6,000 games), RTX 4090,
`--auto-advance`. Engine verified current via `tools/scripts/check_engine_fresh.sh` (368 C++ tests
pass, fuzzer clean over 2,000 games). Both checkpoints were staged under distinct filenames first:
they are both named `snapshot_final`, which is the exact collision §6 warns about. Report:
`research/e1_reanchor_report.md`.

**Result** (Bradley-Terry, HeuristicBot anchored at 1500):

| Rank | Model | Elo | vs HeuristicBot | overall |
|:---|:---|---:|---:|---:|
| 1 | `dec_turns40` (K=40) | **1880.4** | 90.2% | 82.0% |
| 2 | `sp2_pool_off` (control) | 1836.5 | 87.2% | 76.9% |
| 3 | HeuristicBot | 1500.0 | — | 39.9% |
| 4 | RandomBot | 898.9 | 2.9% | 1.2% |

K=40 beats the control head-to-head 56.0% (1,000 games).

**Verdict.** The §4.3 ordering survives the engine changes: K=40 > control > heuristic > random,
and K=40's win rate over HeuristicBot is 90.2% against the 88.9% recorded pre-change. Old
checkpoints load and run forward passes on the current 4,293-dim observation unchanged, so they
remain valid opponents and Elo anchors. **What does not carry over is anything measured through
self-play distribution** — the §4 deficiency tables, ending mixes and battleground counts were
taken on the old engine and must be re-measured before being quoted again.

### 7.1 Auto-advance does not change outcomes

`Engine::step(..., auto_advance)` resolves unattended die rolls, single-choice masks and a few
deterministic multi-step events (Suez <= 4, Muslim Revolution <= 2, East European Unrest <= 3,
Truman, Independent Reds) inside the engine. Enabling it must be a pure speed change or every
tournament number taken with it is incomparable to one taken without.

**Bit-exact where bit-exactness is possible.** `tests/training/test_auto_advance_outcome_equivalence.py`
plays 128 vectorized games under a policy that is a pure function of the mask, and asserts that
terminal utility, victory points and final turn are identical with the flag off and on. The policy
has to be position-derived rather than RNG-driven: with the flag on the engine asks for fewer
actions, so a policy consuming a shared random stream would diverge for reasons unrelated to the
flag. This joins the existing single-state suite (`tests/training/test_auto_advance_integration.py`,
plus `engine/tests/test_auto_advance.cpp`).

**It removes little.** Under that policy the flag cut batched step calls only from 672 to 650
(3.3%), and wall-clock at that scale was inconclusive. It is not the speed lever it looks like.

### 7.2 Tournament results are NOT reproducible run to run — noise floor ~1.5 points

Found while trying to verify 7.1 at tournament scale. Three runs of the *identical* 6,000-game
command, differing only in the flag:

| matchup | auto-advance ON | OFF run 1 | OFF run 2 |
|:---|---:|---:|---:|
| K=40 vs control | 56.0% | 53.6% | 55.2% |
| K=40 vs Heuristic | 90.2% | 89.6% | 89.5% |
| control vs Heuristic | 87.2% | 87.0% | 85.5% |
| Heuristic vs Random | 97.1% | 97.2% | 97.3% |

**Two identical OFF runs differ by 1.6 points on the headline matchup and 1.5 on another** — as
much as the ON/OFF difference. So the ON/OFF gap is not evidence about the flag, and the flag is
not the source of the variation. `BatchMatchRunner` seeds deals deterministically from `base_seed`,
but the agents sample, so a tournament is not reproducible from its configuration alone.

**Consequence for reading every A/B in this file.** At 1,000 games a matchup, differences below
roughly **1.5-2 points are inside the run-to-run envelope** and mean nothing on a single pair of
runs. The binomial SE (1.6% at n=1,000) understates it, because it assumes the only variation is
sampling from a fixed distribution. Either fix the agent sampling seed, or replicate a run before
believing a small gap. The §4.4 comparisons flagged as "underpowered (z ~ 1.2-1.75)" sit exactly
in this band.