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
- **Run arms in parallel — for comparability, not for speed.** Measured on the 4090 (arch v2,
  512 envs): one arm alone runs at **14,761 steps/s**; two arms together run at **7,044 and
  7,833**, so combined throughput is **14,877** — total throughput is conserved and the wall clock
  to finish both is the same either way. The earlier claim here that "sequential doubles
  turnaround" was wrong. What parallel actually buys is that both arms meet identical machine
  conditions, which removes a time-varying confound; what sequential buys is the first arm's
  result at T instead of 2T, which matters if a run may be abandoned early.
  Note the two arms differed by 11% (7,044 vs 7,833), so contention is *not* symmetric — which is
  another reason a wall-clock budget cannot be used for parallel arms. `--train-steps` gives both
  the same training regardless.
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

---

## 8. The human corpus reproduces the US late-war recovery (E2) — SETTLED

**Question.** §4.7 left one hypothesis open that it could not test: a rules bug favouring the USSR
that only surfaces in long games. Random play ends at turn 2.73, before the asymmetry can express
itself, and the heuristic is too weak to separate "engine bias" from "cannot play the late war".
The 300-game human corpus is the missing control — a strong player on *this* engine.

**Setup.** `ai/eval/human_corpus.py`. All 300 corpus files: 9 are empty cached downloads (verified
by reading them, not converter failures), 9 are skipped for non-standard handicaps, **282 convert
with 0 failures**; 131 reach a terminal state and 151 are fragments whose recording stops. 144,844
decisions classified, none excluded. The VP sign convention was verified empirically rather than
assumed (`victory_points = +20` -> `get_terminal_utility = +1.0`).

**Result — mean VP by turn (US-positive), against the §4.6 arms:**

| turn | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 |
|:---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| control | +0.15 | -1.02 | -1.87 | -3.03 | -3.58 | -3.91 | -3.11 | -2.49 | -4.47 |
| K=40 | +0.01 | -0.91 | -2.26 | -1.96 | -2.20 | -2.31 | -1.95 | -1.68 | -2.18 |
| **human** | +0.21 | -1.09 | -1.58 | -1.80 | -1.34 | -0.19 | -0.62 | -0.01 | **+2.36** |
| human N | 267 | 253 | 235 | 210 | 193 | 167 | 145 | 107 | 94 |

Human US win rate over the 131 finished games: **48.9%** (64/65/2, 95% CI ±8.6pp), and **50.0%**
in games ending turns 7-10 (N=124), against 35-40% for both agents.

**Verdict.** The corpus reproduces the standard arc in full. The USSR early lead appears at about
the agents' magnitude (trough -1.80 at turn 5), then the human curve climbs monotonically, crosses
zero by turn 9 and ends **+2.36**, while the control ends at -4.47 and K=40 at -2.18 without ever
reaching parity. **The US late-war edge is fully expressible on this engine**, so the §4.7 rules-bug
hypothesis is not supported: §4.5's side imbalance and §4.6's missing recovery are properties of our
agents, not of the simulation. Training work on the US-side weakness is unblocked.

**Caveats.** This clears the engine only of a bug big enough to erase the late-war edge, not of
smaller ones. It is not a matched comparison — skill, game length and ending mixes all differ. The
per-turn population shifts as in §4.6 but for a different reason: human attrition (267 -> 94) is
mostly the *recording* stopping rather than games ending, and whether that truncation is
outcome-neutral was not tested. The turn 1-4 bucket is empty for humans, so §4.6's short-game row
has no counterpart. And none of this says *why* the agents fail to convert the late war.

### 8.1 Humans decline forced wins twice as often as the agents — WITHDRAWN, see 8.3

| | value |
|:---|---:|
| instant-win opportunities | 149 (0.103% of decisions; agents 0.175-0.196% in §4.2) |
| games with at least one | 95 of 282 (33.7%) |
| **take rate** | **47.7%** (71 of 149, ±8.0pp) |
| declines with a settled outcome | 73 (5 more fell in fragments, excluded) |
| **cost of declining** | **20.5%** (the decliner still won 58 of 73, ±9.3pp) |

This strengthens §4.4 rather than complicating it. A corpus that is 48.9% balanced overall takes
forced wins at **47.7%**, roughly half the control's 81.2% and K=40's 72.4% — so a low take rate is
plainly compatible with strong play, and the metric cannot be an optimisation target. The
opportunity split is also 93 US / 56 USSR, the opposite tilt to the control's 92/158.

### 8.2 §4.2's critic optimism does not reproduce on human positions

`dec_turns40/snapshot_final.pt`'s value head over 84,073 decisions in finished human games:

| v_win bin | N | mean predicted | mean actual | gap |
|:---|---:|---:|---:|---:|
| [-0.75,-0.50) | 11,951 | -0.61 | -0.43 | -0.18 |
| [-0.50,-0.25) | 18,428 | -0.37 | -0.15 | -0.22 |
| [-0.25,0.00) | 15,458 | -0.13 | -0.03 | -0.10 |
| [0.00,+0.25) | 15,987 | +0.13 | +0.17 | -0.04 |
| [+0.25,+0.50) | 14,945 | +0.36 | +0.23 | **+0.13** |
| [+0.50,+0.75) | 6,431 | +0.59 | +0.54 | +0.05 |
| **all** | 84,073 | -0.06 | +0.01 | -0.07 |

Globally the critic is mildly **pessimistic** here, and its largest errors are in the negative bins
— it overstates how lost a losing-looking human position is. Optimism appears only in
[+0.25,+0.50), which is exactly the band §4.2's missed forced wins sit in, and it is smaller than
the pessimism elsewhere. So §4.2's finding is narrower than stated: not a global optimism, but a
miscalibration in one band, measured on a state distribution the agent generates itself. Off its own
distribution the sign flips.


### 8.3 The human take rate was pooling two different things — 8.1 withdrawn

**8.1's 47.7% is not a fact about human play.** Splitting the 149 opportunities by *how* the win
arrives takes the surprise out of it entirely:

| | opportunities | taken | take rate |
|:---|---:|---:|---:|
| all, as 8.1 reported | 149 | 71 | 47.7% |
| — win by VP threshold or scoring | 117 | 68 | 58.1% |
| — win by the game reaching DEFCON 1 | 32 | 3 | 9.4% |
| **excluding the last action round of turn 10** | **64** | **32** | **50.0%** |
| — win by VP threshold or scoring | 35 | 31 | **88.6%** |
| — win by the game reaching DEFCON 1 | 29 | 1 | **3.4%** |

**On ordinary wins humans are better than either agent** — 88.6% against the control's 81.2% and
K=40's 72.4%. The headline was dragged down by a second category humans essentially never take.

Two separate corrections are folded in above:

**The last action round of turn 10 is an artifact.** Final scoring fires after it no matter what,
so every action whose forced continuation ends the game in your favour is labelled a win. 46 of
78 declines sit there and **all 46 decliners won the game anyway**: they were choosing among
winning moves, not missing one. This inflates the human sample specifically, because humans reach
turn 10 far more often than our agents do (§8 records 94 games at turn 10; §4 has agents reaching
it in 11.7%). The metric is therefore not comparable across populations with different game
lengths without this exclusion.

**The DEFCON-1 category is under query — see 8.4.** Humans take 1 of 29. That is not caution; it
is what you would expect if the move is a blunder that the classifier has labelled a win.

### 8.4 The DEFCON-1 "wins" are illegal moves the engine offers — FIXED

My first reading of these, that DEFCON-1 losses were attributed to the wrong player, was **wrong**.
`resolve_defcon_one_loss` (`engine/include/ts/defcon.hpp:28`) makes the *phasing* player lose
regardless of who drove DEFCON down, which is the rule. The engine is right about that.

The actual defect is narrower and worse. **Two events run their own free-coup target lists and
never consult `Operations::can_coup`:**

| card | site | what it validates |
|:---|:---|:---|
| #91 Ortega Elected in Nicaragua | `card_dispatcher.cpp:1431` | adjacency to Nicaragua only |
| #107 Che | `card_dispatcher.cpp:1402` | region, non-battleground, not visited |

`can_coup_or_realign` refuses a country the opponent has no influence in
(`engine/src/ops.cpp:119`), and `get_coup_target_mask` is built on it, so an ordinary Ops coup is
filtered correctly. These two bypass it, and so offer coups the rules forbid — along with,
presumably, the DEFCON regional restrictions, NATO and The Reformer, which live in the same
function.

**Replay 139, turn 9, action round 2.** The US played Ortega — a USSR card — for Ops, so its event
fired and handed the USSR a free coup. The engine offered **Cuba**. The log records Cuba as
`inflUS 0 / inflUSSR 3` at *every* entry of turn 9, and the engine state agrees exactly, so the
reconstruction is correct and the board is not in doubt. With no US influence there, the USSR
cannot coup Cuba. But Cuba is a battleground, so the offered coup took DEFCON 2 → 1 and ended the
game against the phasing player, the US. That is why it scored as a USSR "win".

Same shape at **replay 16 T9 AR3**, **replay 165 T9 AR3**, **replay 245 T8 AR1** — all Ortega, all
Cuba, all `US 0 / USSR 3`.

`tests/engine_logic/test_free_coup_target_legality.py` reproduces both synthetically: Ortega offers
`[67, 68, 71]` including Cuba with zero US influence, and Che offers 26 countries without checking
influence at all. A third test confirms the ordinary Ops path filters correctly, so the defect is
in the two event handlers, not in the coup rule. **Fixed** (approved): both handlers now call `Operations::can_coup(state, Player::USSR, i)`, at
the target mask in `get_event_action_mask` and again where the chosen target is applied. The
forced win at replay 139 T9 AR2 is gone, all 368 C++ tests pass, the fuzzer is clean over 3,000
games, and all 282 corpus games still convert with 0 failures. One existing C++ test,
`OrtegaElected_CanCoupCuba_AndAdjacentCountries`, asserted the old behaviour -- it gave Cuba US
influence but left Costa Rica and Honduras empty and expected them offered anyway -- and now sets
up influence in those two and additionally asserts that an adjacent country with none is refused.

**Why this matters beyond the metric.** `classify_legal_actions` reads the engine's terminal
utility, and so does every reward. A policy trained against this learns that an opponent's Ortega
is a free win whenever a battleground sits next to Nicaragua — a move the rules do not permit.

### 8.5 The forced-win metric under-detects as well as over-detects

Three further corrections, from a review of the individual cases, all pointing the same way: the
metric is not measuring what §8.1 claimed.

**It misses wins that need a choice, and then scores them as declines.**
`classify_legal_actions` follows only *forced* continuations, which its docstring is explicit
about. So when a position has several winning lines and the human takes one the walk cannot see,
it is recorded as declining. **Replay 104 T9 AR7** is exactly this: the US had more than one path
to a forced win, and playing How I Learned to Stop Worrying won just as Duck and Cover would
have. It is counted as a decline; it is a win taken. The metric should ask whether the action the
player chose also wins, not whether it is in the classifier's set.

**A headline is not an action round, and cannot be judged as one.** **Replay 224 T7 AR0** and
**replay 259 T8 AR0** are headline decisions. Both players choose simultaneously and neither
knows the other's card, so a line that is forced *given the board* is not available information
to the player. Declining it is ordinary play under uncertainty, not an error. Headline decisions
should be excluded from this metric entirely.

**And at least one labelled win does not reach 20 VP.** At **replay 259** the US was on 17 VP and
KAL-007 moves them to 19 — not a win. The engine nonetheless drives that line to a terminal state
it scores +1.0 for the US, so there is a second defect here, distinct from 8.4, in whatever
terminal the forced walk arrives at. Not yet diagnosed.

**Where this leaves 8.1.** Of the 64 non-endgame opportunities, 29 are the illegal Ortega/Che
coups of 8.4, and an unknown further number are headline decisions or wins-taken-by-another-line.
The remaining sample is too small and too contaminated to support any statement about how humans
treat forced wins. **8.1 is withdrawn and not replaced.** The instrument needs fixing first: skip
headlines, test the chosen action for a win rather than set membership, exclude the last action
round of turn 10, and re-run once the free-coup handlers filter.


### 8.6 Replay 259 is VP drift, not a card bug — and drift is corpus-wide — SUPERSEDED BY 8.7

**The user's arithmetic was right and the engine's starting number was wrong.** At replay 259 the
engine holds `victory_points = 18` when it labels the KAL-007 headline a win, and KAL takes it to
20. The log records 17 at the end of turn 7 and **16** at the turn 8 headline, and it also shows
KAL actually played at **AR1, not the headline**, moving the score 16 → 17. So the "win" rests on
a starting VP two points above what the game had.

**This is not specific to 259.** `Conversion.vp_drift` counts entries where the engine's score
disagreed with the log's, and across the 282 converted games:

| | |
|:---|---:|
| games with **zero** drift | 53 (18.8%) |
| games with some drift | **229 (81.2%)** |
| drift per game | mean 8.1, median 5, p90 20, max 56 |
| `log_miscounts` / `scores_forced` across the corpus | 1 / 1 |

`_reconcile_scalars` resyncs `state.victory_points` to the score the log narrates whenever an
entry states one, so drift is corrected at entry boundaries and does not accumulate — which is why
these games still convert cleanly. But **within** an entry the engine's VP is its own, and that is
what a decision sample sees.

**Consequence for §8, which is not yet resolved.** The human VP-by-turn arc — the whole basis for
saying humans reproduce the US late-war recovery — is read from the engine's VP at decision time
(`obs.vps`), not from the log's narrated score. Turn boundaries are taken at the first decision of
turn T+1, which is close to a resync, so the arc may well survive; but "may well" is not
"measured". **Re-derive the §8 arc from the log's own narrated scores before relying on it.** That
is a cheap check and it is the one that matters, because §8 is currently the evidence that the
engine is not USSR-biased.

Separately, this is a second reason the forced-win metric cannot be trusted on human data (§8.5):
a labelled win can rest on a VP the game never had.


### 8.7 Most of the "VP drift" was the log's score field, not the engine — 8.6 corrected

**`Conversion.vp_drift` is not a measure of engine error.** It compares the engine against
`entry.score`, a running field that lags the log's own narration -- `_narrated_score`'s docstring
already records the two disagreeing in 531 of 6,602 places. The converter resyncs from the
*narration*, which is the authoritative statement, so a lagging field produces a counted "drift"
with nothing wrong.

**Replay 219 is the clean demonstration.** At turn 8 AR1 the log reads:

```
Turn 8, USSR AR1: South America Scoring: Event: South America Scoring
USSR gains 10 VP. Score is even.
```

The score was US +10, the USSR gains 10, so it is even -- and the engine says 0, matching the
narration exactly. The `score` field still reads 10, and goes on reading 10 for five more entries
before catching up at AR5. Every one of those was counted as drift. The engine was right
throughout.

**Re-measured against the narration**, over midgame entries (excluding the terminal ±20 marker,
and turn 10 AR7/AR8 where the engine's final scoring and the log's bookkeeping legitimately
differ):

| | entries |
|:---|---:|
| engine disagrees with the score **field** | 4,500 |
| — engine matches the **narration** (field lags; engine correct) | 666 |
| — entry narrates no score, so nothing to check against | 3,822 |
| — **engine disagrees with the narration** | **12**, in 7 games |

Real disagreements are **12 entries across 7 games, magnitude 1-2**. Not 194 games, and not mean
2.2 VP. §8.6's "81% of converted games carry drift" was measuring the log's field lag.

**This largely clears §8.** The human VP arc is read from the engine's VP, and the engine agrees
with the log's narration nearly everywhere it can be checked. Re-deriving the arc from narrated
scores is still worth doing, but as confirmation rather than as repair. The one caveat that stands
is the terminal marker: `victory_points` becomes ±20 when a game ends, so any turn-boundary sample
taken after a terminal reads the result rather than the score.

### 8.7.1 Asia Scoring under Shuttle Diplomacy — RESOLVED: the log is wrong, not the engine

Six of the twelve real disagreements are one game, **replay 259**, where the engine sits **+1**
above the narration from turn 7 AR3 onward -- through turns 7, 8 and 9. It starts here:

```
Turn 7, USSR AR3: Asia Scoring: Event: Asia Scoring
Shuttle Diplomacy is no longer in play.
USSR gains 2 VP. Score is US 17.
```

The engine awards the USSR **1**, the log **2**. Working the region from the log's own board
(USSR holds North Korea, South Korea, Japan and Pakistan; the US holds India; the USSR also holds
Afghanistan, the US eight more non-battlegrounds):

* Neither side dominates -- the US has more countries, the USSR more battlegrounds -- so both
  score Presence, 3 each.
* Shuttle Diplomacy removes one USSR battleground. Taking Japan removes the battleground, the
  country, *and* the USSR's superpower-adjacency bonus, since Japan is the Asian country adjacent
  to the US.
* USSR 3 + 3 battlegrounds + 0 adjacency = 6; US 3 + 1 = 4. Net **USSR 2**, which is what the log
  says.

The engine's 1 is what you get from **two** battlegrounds coming off rather than one. The
adjustment lives at `engine/src/scoring.cpp:70-85` and decrements the battleground count, the
country count and (in Asia) the adjacency in one block, guarded by
`SHUTTLE_DIPLOMACY_ACTIVE`; the flag is cleared in two places, `scoring.cpp:182-184` and
`scoring.cpp:287-289`. Double application is the obvious candidate and is **not yet verified** --
the arithmetic above establishes the symptom and which side is right, not the mechanism.

Per invariant 11 this is reported, not fixed.


### 8.7.2 The Shuttle/Japan divergence is a log fault, and is now corrected by rule

Settled by the project owner: this is a known, reproducible property of the ts-replayer logs. When
Shuttle Diplomacy is in play, Asia is scored, and the USSR holds Japan, **the log keeps the USSR's
bonus for controlling a country adjacent to the United States** — which the card has just removed
along with Japan — and pays the USSR 1 VP too many. **The engine is right; the log is not.** My
arithmetic in 8.7.1 reached the wrong conclusion from the same numbers.

It was already handled for the one game it had been diagnosed in: `_LOG_MISCOUNTED` carried
`259: {(7, "AR3", "USSR"): 1}`, so the six "disagreements" 8.7 attributed to replay 259 were an
artefact of my measurement, which compared the engine against the narration without applying the
offset the converter applies. **The real count of engine/log disagreements is 6, not 12.**

Now recognised by rule rather than by entry (`_shuttle_japan_asia_miscount`): Shuttle Diplomacy
flag set, the entry's events name Asia Scoring, and the USSR controlling Japan — all read before
the entry is driven, since scoring consumes the flag. Where it fires, the engine's score stands
and every score the log states afterwards is compared against its own number plus the offset,
which is the existing `_LOG_MISCOUNTED` machinery.

**Generalising found a second game.** The rule fires on **replay 127** as well as 259. 127 was
never listed, so until now it converted by taking the log's score — carrying a USSR VP total 1 too
high, and the board that does not pay it, into the training data. That is the whole argument for a
rule over a list: a list only covers the games already downloaded.

`_LOG_MISCOUNTED` is now empty and kept for anything genuinely particular to one game. All 282
games still convert with 0 failures.


### 8.7.3 The log's score is what training data carries, and what is left after that

**Corrected intent.** 8.7.2 kept the engine's rules-correct score and offset the log. That is
backwards for this corpus. The players were reading the app's score, not the rulebook: they played
the slightly wrong game, and every decision after the Shuttle/Japan Asia scoring was made against
the number the log shows. Training data has to be the position the human actually saw, so the
converter now **adopts the logged score** and carries it forward; only the assertion for that one
entry is dropped, and no offset is accumulated. `_LOG_MISCOUNTED` stays empty.

**What is left.** Sweeping all 282 converted games, comparing the engine against the log's
*narration* (never the lagging score field) at the point the converter compares:

| | entries |
|:---|---:|
| engine disagrees with the narration | 21 |
| — terminal ±20 result marker, not a score | 10 |
| — Shuttle/Japan Asia scoring, log adopted | 1 |
| — **unexplained** | **10**, in 8 files |

Of the 10 unexplained, four are end-of-game, where the engine's final scoring and the log's
bookkeeping legitimately differ:

| replay | diff | where |
|---:|---:|:---|
| 117 | −14 | T10 AR7, The China Card |
| 234 | −10 | T10 AR7, South America Scoring |
| 16, 43 | +6 | T10 AR8, Mideast Scoring |

That leaves **six midgame entries in five distinct games** (16 and 43 are duplicate downloads of
one game), all of magnitude 1–2:

| replay | diff | where |
|---:|---:|:---|
| 16, 43 | −2 | T4 AR7, Alliance For Progress |
| 28 | −2 | T4 AR7, Special Relationship |
| 57 | −2 | T3 AR6, Arab-Israeli War |
| 71 | +2 | T3 AR6, Mideast Scoring |
| 65 | +1 | T6 AR7, Che |

Four of the six sit at turn 3 AR6 or turn 4 AR7 and are all worth exactly 2 on US entries, which
is suggestive of one shared cause rather than five unrelated ones. None is diagnosed. This is the
whole remaining VP disagreement between the engine and 282 human games — down from the "194 games,
mean 2.2 VP" of 8.6, which was measuring the log's own bookkeeping lag.


### 8.7.4 The midgame cluster is the military operations penalty — diagnosis confirmed

The six midgame disagreements of 8.7.3 are not engine errors. Traced at **replay 16, turn 4 AR7**:

* The log narrates `US gains 3 VP. Score is USSR 5.` — the score **after** the play and
  **before** the turn is cleaned up.
* The engine, at the point the converter compares, is already on **turn 5** at **−7 VP**, with
  military operations reset.
* Entering the round the USSR held 5 military operations to the US's 0 at DEFCON 2, which is a
  2 VP penalty to the US. −5 − 2 = −7.

Both numbers are right; they are taken at different moments. The log books the penalty on a later
entry — at turn 5 AR1 it narrates `Score is USSR 6`, which is the engine's −7 plus the 1 VP that
entry awards, and the two agree from there on. That accounts for the whole cluster: every one of
the six sits on a turn's last action round, and the magnitudes (2, 2, 2, 2, 1) are military
operations shortfalls.

**The obvious repair does not work, and the reason is worth recording.** The converter currently
skips the comparison whenever an entry crosses a turn boundary (`crossed_turn`), which leaves one
entry in eight unchecked. Comparing against the score as it stood before the turn moved would
restore that coverage — but there is no such observable moment. Wrapping `Engine.step` to capture
the score the instant before the turn number changes yields **−8**, not the −5 the log states,
because Alliance For Progress's own +3 and the turn cleanup are applied **within a single engine
step**. Between the event's VP and the penalty there is no step boundary to read.

Two ways to get the check, neither done:

1. **Engine-side**: record the score at the start of cleanup (or make cleanup its own step), so
   the pre-penalty value can be read. Per invariant 11 this needs the owner's approval.
2. **Log-side**: derive the expected penalty from the log, which prints both military operations
   totals and DEFCON, and check the engine's post-cleanup score against
   `narrated + min(ussr_ops, defcon) − min(us_ops, defcon)`. Self-contained, but duplicates a
   rule the engine already implements, so it is a differential check rather than a reconciliation.

Until one of them exists, a turn's last action round remains unverified for score, and 8.7.3's
six "unexplained" midgame entries should be read as explained.


### 8.7.5 The turn-ending score is checked now, but only to within the cleanup

`DecisionContext.pending_roll_type` exposes which chance node is pending, so the converter can
find the `TURN_CLEANUP` pause and read the score on both sides of it. A turn's last action round
is no longer skipped: **212 such entries in the first 60 games** are now checked where they were
not.

**The check is weaker than intended, and the reason is in the logs.** Which moment the narration
describes is not something the log states:

| replay | entry ends with | moment |
|---:|:---|:---|
| 16 T4 AR7 | `Score is USSR 5` | **before** cleanup |
| 182 T1 AR6 | `Turn 9, Cleanup: US gains 3 VP. Score is even` | **after** cleanup |
| 221 T2 AR6 | `USSR gains 2 VP. Score is US 2` | **after** cleanup, unlabelled |

The word "Cleanup" cannot separate them — replay 221 is a cleanup award with nothing marking it as
one, and keying on the word failed that game (and its duplicates 265, 299) while passing 182. So
the assertion is that the engine's score equals **one of the two moments** the narration could be
describing, and it fails when it is neither. Anything off by more than that turn's Military
Operations deficit is caught; an error the size of the deficit is not.

Pinning the exact moment would need the log to say which it means, and it does not. Worth
revisiting only if a real disagreement is ever found hiding in that gap.

All 282 games still convert with 0 failures.


---

## 9. Human-corpus BC warmup (E3, first pair) — NEGATIVE

**Question.** §4.8 argued that "one human game shows the reversal that RL needs thousands to
notice". Does warming up on the 280-game human corpus instead of on self-play demonstrations
produce a better agent?

**Setup.** arch v2, `--train-steps 80000000` (both arms exactly 80M), 512 envs, `blunder_aware`
+ K=40, run in parallel on one 4090, snapshots every 2M steps. One flag apart: the BC warmup
checkpoint. Arm A warmed on 5,000 self-play games from `dec_turns40` (79.2% top-1 after 2 epochs);
arm B on the human corpus, 144,844 samples with value targets masked on the 149 games whose
recording stops (45.4% top-1). Engine as of `68f155b` minus the Independent Reds fix, which landed
mid-run and applies to neither arm.

**Result** (tournament, 1,000 games per pair):

| | Elo | vs the other arm | vs `dec_turns40` | vs HeuristicBot |
|:---|---:|---:|---:|---:|
| arm A, self-play warmup | **1852.9** | **57.4%** | 56.8% | 84.1% |
| `dec_turns40` (previous best) | 1824.7 | — | — | **90.5%** |
| arm B, human warmup | 1810.3 | 42.6% | 49.2% | 85.8% |

**Game shape and map coverage** (256 self-play games each, temperature 0.1):

| | mean final turn | reaches turn 9 | empty BG turn 5 | empty BG turn 8 |
|:---|---:|---:|---:|---:|
| arm A, self-play | 6.58 | 0.270 | **10.83** | **6.70** |
| arm B, human | 6.93 | 0.336 | 11.93 | 7.99 |
| `dec_turns40` | **7.42** | **0.398** | 11.51 | 7.06 |

**Verdict: the hypothesis is not supported.** Arm A beats arm B head-to-head 57.4% over 1,000
games, which is far outside the ~1.5 point run-to-run envelope of §7.2. Arm B is also *worse* on
the measure the corpus was supposed to fix: it leaves **7.99** battlegrounds empty at turn 8
against arm A's 6.70, where §4 identifies unclaimed battlegrounds as the deficiency. The one thing
arm B does better than its control is game length — 6.93 turns and 33.6% reaching turn 9 against
6.58 and 27.0% — which is the §4.6 axis, but it does not convert into strength.

**A confound that matters, and that this pair cannot separate.** The two warmups do not start the
arms from equally good policies: 79.2% top-1 against 45.4%, and arm B's first snapshot beat
HeuristicBot 37.5% against arm A's 80.0%. So this compares "human data" and "a much weaker
initialisation" at once, and 80M steps may simply not be enough for arm B to close a gap it began
with. What it establishes is narrower than the question: *at 80M steps, warming on the human corpus
alone is worse than warming on self-play demonstrations*. It does not establish that human data is
unhelpful.

**The remaining arms are now the interesting ones.** The synthetic→human fine-tune gets the strong
initialisation *and* the human data, which is exactly the combination this pair could not test; and
the no-warmup arm would say how much either warmup is worth at all. Only two arms fit in 24 GB, so
they were always a second round.

**Also worth noting: `dec_turns40` did not lose its crown cleanly.** It beats both new arms against
HeuristicBot by 4-6 points and leads both on game length and turn-9 reach, while losing to arm A
head-to-head 43.2%. Beating the model that beats the heuristic more, while beating the heuristic
less, is a real intransitivity and a reminder that a single opponent is not a ranking.

**Caveat.** One seed per arm. §7.2's noise floor covers tournament measurement, not run-to-run
training variance, which this repository has never measured. A 57.4% head-to-head is comfortably
outside measurement noise; it is not known to be outside seed noise.


### 9.1 The human prior washes out in 2% of training — which explains §9

**Question.** §9 could not separate "human data is worse" from "the human arm started from a much
weaker policy". Measuring whether the prior survives at all separates them, and costs nothing: the
checkpoints already exist.

**Setup.** Top-1 agreement with human moves on a fixed probe of 20,000 corpus decisions, evaluated
at all 42 snapshots of both arms. Arm A never saw a human game and is the floor.

| | after BC | 1st snapshot (~2M steps) | minimum | final (80M) |
|:---|---:|---:|---:|---:|
| arm B, human warmup | **45.7%** | 33.1% | 29.0% | **32.5%** |
| arm A, self-play warmup | 33.6% | 31.4% | 30.6% | 32.6% |

**The prior is gone inside the first 2M steps — 2.5% of the run — and never returns.** From there
the two arms agree with human play *identically*, both sitting at 31-34% for the remaining 78M
steps and finishing within 0.1 points of each other. Roughly 32% is what RL converges to whatever
it was initialised from.

**So §9 was not testing what it looked like it was testing.** Past the first snapshot there was no
human prior left to test; the arms differed only in where they started, which is precisely the
confound §9 flagged as unresolvable from that pair. It is now demonstrated rather than suspected.

**Consequence.** A warmup variant cannot answer this question -- not a longer one, not a mixed
synthetic+human one, not a fine-tune. Anything delivered as an initialisation decays to the same
attractor within 2M steps. Human data has to be applied as something that *persists*, which is
what E4's pinned `π_ref` is: a KL anchor held throughout training rather than a starting point.

### 9.2 How long to train the human BC — about 8 epochs, and it matters only for E4

**Question.** The E3 warmup ran 2 epochs and reached 45.4% top-1. Would training it longer help?

**Setup.** Split by **game**, not by sample: positions within a game are heavily correlated, so a
sample split scores the model on positions it has effectively already seen. 224 games train,
56 held out (117,025 / 27,819 samples), fresh v2 net, agreement after each epoch.

| epoch | 0 | 1 | 2 | 4 | 6 | 8 | 10 | 12 |
|:---|---:|---:|---:|---:|---:|---:|---:|---:|
| train | 18.7 | 40.9 | 41.5 | 45.3 | 47.0 | 49.0 | 52.9 | 57.5 |
| **held out** | 19.0 | 41.0 | 41.6 | 44.8 | 45.9 | **46.9** | 47.5 | **47.8** |

**Two epochs was undertrained** -- held-out agreement climbs from 41.6% to about 47% by epoch 8, so
the E3 warmup left roughly five points on the table. **But it plateaus there**: epochs 8 to 12 buy
0.8 points of held-out agreement while the training score gains 8.5, so the model is memorising
games from that point on. About 8 epochs is the useful end.

**It would not have changed §9.** By 9.1 a 48% initialisation decays to the same ~32% attractor as
a 45% one, and just as fast. Where it does matter is E4: there the human policy is the anchor
rather than the starting point, it persists for the whole run, and its quality is the experiment.
Build that net at ~8 epochs.
