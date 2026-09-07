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


### 9.3 Do the humans respect the dominance relations? — TRAP NUMBERS WRONG, SUPERSEDED BY 9.5

**Question.** §4.8 called the dominance result "the strongest argument yet for demonstrations,
since one human game shows the reversal that RL needs thousands to notice". That is a claim about
the corpus which had never been checked against the corpus. Before engineering any way to keep
human data in the policy, it is worth knowing whether the data carries the signal.

**Setup.** `ai/eval/human_dominance.py`, over all 280 converted games. For a policy §4.8 can ask
how the pair is *ranked*, because there is a distribution; for a human there is only the move, so
the measure is the stricter half — did they choose the dominated card while a dominant alternative
of equal Ops sat in the same hand.

| | humans | control | K=40 |
|:---|---:|---:|---:|
| Quagmire, chose dominated | **24.3%** (9/37) | 37.5% | 30.7% |
| Bear Trap, chose dominated | **21.1%** (16/76) | 32.8% | **18.2%** |
| Space race, chose dominated | **0.4%** (3/712) | 18.1% | 12.6% |

**On the space race the corpus is emphatic.** Humans spend the opponent's card on the track
essentially always: 3 errors in 712 decidable plays, against our agents' 12.6-18.1%. That is a
factor of 30 to 45, on the largest sample of the three, and it is exactly the kind of local,
position-independent preference a demonstration can teach and RL evidently does not find.

**On the traps it is much weaker than §4.8 implies.** Humans are better than the control on both,
but **K=40 already beats them on Bear Trap** (18.2% against 21.1%) and is close on Quagmire. So on
trap discards there is little left for the corpus to teach the current best agent.

**Verdict.** The premise is *partly* confirmed, and narrower than the rhetoric it was based on.
There is one clear, well-evidenced class where human play is strongly better and ours is not, and
two where our best model has already caught up. Set against §9 -- where the human-warmed arm was
weaker overall despite inheriting these preferences -- the case for spending GPU time on a
mechanism to retain human data rests on that one class.

**A denominator caution, in the same family as §4.8's own.** The first version of this counted
every opponent-card space play as a correct decision and reported 0 errors in 265 "decidable"
plays. Most of those were not decisions: no equal-Ops own card was in hand, so nothing could have
been chosen differently. Requiring both kinds of card at the same Ops cut the denominator to 712
across the corpus and is what makes the 0.4% meaningful. §4.8 records the same trap from the other
direction — measuring against all space plays reads 2.4% and badly understates the agents' rate.


### 9.4 Half the humans' apparent trap mistakes are ours, not theirs — PARTLY WRONG, SUPERSEDED BY 9.5

**Question.** §9.3 put humans at 22.1% dominated on trap discards, better than the control but not
clearly better than K=40. Are those human mistakes, or does our hand reconstruction hand them a
card they never held? A dominance pair says "you discarded X when Y was available"; if Y is the
solver's padding, we invented the alternative and the error is ours.

**Setup.** Each of the 25 dominated discards checked against the log's *own* per-turn hand list --
the only independent record of what a player held.

| | count |
|:---|---:|
| dominant alternative **is** in the log's hand list | **13** |
| dominant alternative is **not** — reconstruction supplied it | **12** |

**Recomputed over pairs where the log evidences both cards:**

| | as §9.3 measured | log-evidenced pairs only |
|:---|---:|---:|
| trap discards, dominated | 22.1% (25/113) | **12.1%** (12/99) |
| — Quagmire | 24.3% | **6.7%** (2/30) |
| — Bear Trap | 21.1% | **14.5%** (10/69) |

Against §4.8's agents that reverses the reading: Quagmire **6.7%** against the control's 37.5% and
K=40's 30.7%; Bear Trap **14.5%** against 32.8% and 18.2%. Humans are better than both agents on
both traps, where §9.3 had them level with K=40.

**But the log's hand lists are not complete either, so this is a bound, not a value.** Measured
over all 33,511 human card decisions: **3.5%** of the time the card the human is *recorded playing*
is itself absent from that turn's hand list. A list that omits cards the player demonstrably held
cannot be treated as ground truth. So 12.1% is a **lower** bound on the human error rate and 22.1%
an **upper** one. The useful conclusion survives either way: at the upper bound humans beat the
control, at the lower bound they beat K=40 as well.

### 9.4.1 The wider consequence: a quarter of the corpus's offered actions are unevidenced

The same measurement, applied to every card decision rather than to dominance pairs:

| | |
|:---|---:|
| human card decisions measured | 33,511 |
| cards offered across them | 187,458 |
| **not in the log's hand list** | **44,247 (23.6%)** |
| decisions offering at least one such card | 27,267 (**81.4%**) |

Part of that is the solver padding hands the log underdetermines, and part is the log's own
incompleteness — the 3.5% above proves the second exists and the two cannot be separated with what
the log records. Either way, **the human corpus offers the policy a choice among cards there is no
evidence the player held, at 81% of its card decisions**, and those cards are in the action masks
the BC warmup trains against.

That is a data-quality finding rather than a bug: nothing is *forced*, and the converter's own
rules are intact. But it bounds what the corpus can teach about card selection, it is a plausible
contributor to §9's negative result, and it means any future measurement conditioned on hand
contents needs the same log-evidenced restriction applied here.


### 9.5 The humans never once chose a dominated trap discard — 9.3 and 9.4 corrected

**Both earlier readings were wrong, and the second error was mine rather than the corpus's.** The
project owner checked two of the cases §9.4 called genuine and found the log says otherwise:

* **replay 63, turn 5.** The USSR *headlined* Che and discarded **Duck and Cover** at AR1. The log:
  `USSR Headlines Che`, then `Turn 5, USSR AR1: Duck and Cover: USSR discards Duck and Cover`.
  Duck and Cover is a US card, so that discard is the *dominant* choice.
* **replay 71, turn 6.** The USSR *headlined* Quagmire; the Bear Trap discards were Duck and Cover
  and then Camp David Accords.

**The converter is right in both.** Driving replay 63 gives headline USSR → Che, US → Red
Scare/Purge, then AR1 USSR → Duck and Cover, matching the log line for line. **The corpus does not
need reconstructing.**

**The fault was in the dominance driver.** A headline play is a `SELECT_CARD` decision and the trap
flag is set in that state too, so a headline made while trapped was scored as a trap discard — and
scored as an error nearly every time, because the rule says "discard the opponent's recurring
event" while a headline is where you play your own best card. **14 of 15 decidable headline
decisions were counted as errors**, against 11 of 98 in the action rounds where the rule belongs.

**Corrected, with both fixes applied:**

| | trap discards dominated |
|:---|---:|
| §9.3, as first measured | 22.1% (25/113) |
| headline plays excluded | 11.2% (11/98) |
| **and restricted to pairs the log evidences on both sides** | **0.0% (0/87)** |

Bear Trap 0/59, Quagmire 0/28. **Every remaining apparent mistake involved a dominant alternative
the log does not list** — a card the reconstruction supplied, or one the log omits. On decisions
where the log records both cards, the humans in this corpus never chose the dominated one.

**The bound still applies in the other direction.** The log's hand lists are incomplete (§9.4:
3.5% of the cards humans are recorded *playing* are absent from them), so restricting to
log-evidenced pairs is conservative and may discard genuine decisions. The true rate lies between
0% and 11.2%. At either end it beats every agent we have measured: traps 18.2-37.5%, space race
12.6-18.1%.

**Verdict, replacing §9.3's.** The corpus carries the dominance signal on all three classes, not
one. §4.8's argument for demonstrations stands as written; §9.3's doubt about it was an artefact of
this driver. What that means for §9's negative result is unchanged — the corpus contains the signal
and BC warmup still failed to deliver it, which is a statement about the mechanism, not the data.

**Method note.** Both of §9.4's headline examples were presented as the *genuine* cases, the ones
left after the padding correction. They were the least reliable in the set. A filter that admits a
decision type it was not written for will do most of its damage in the cases that look cleanest.


### 9.6 The dominance relation as evidence about the hand

**Idea (owner's).** §9.5 established that humans never take the dominated side of a trap discard
where the log records both cards -- 0 of 87. So the discard is evidence about the rest of the hand:
if a player gave up their own or a neutral card to Quagmire or Bear Trap, an opponent recurring
event of that printed Ops was very probably not in their hand. The solver can use that.

**Implementation.** `tools/lib/ts_replayer_hands.py` records which card went to the trap
(`GameFacts.trap_discard`) and adds a soft clause against holding any equal-Ops opponent recurring
event that turn. Deliberately soft, and deliberately a *second* clause on the same literal so its
weight adds to that card's existing hold cost rather than replacing it. It is a statement about how
people play, not about what the rules permit, so everything the log establishes stays in the hard
model and outranks it. Five Year Plan, the China Card, scoring cards and one-time events are
excluded, matching `ai/eval/dominance`.

**Result.** Apparent dominated trap discards across the corpus:

| | dominated | rate |
|:---|---:|---:|
| before | 11/98 | 11.2% |
| **after** | **7/94** | **7.4%** |
| — Bear Trap | 2/61 | 3.3% (was 7.8%) |
| — Quagmire | 5/33 | 15.2% (was 17.6%) |

Restricted to pairs the log evidences on both sides it stays **0/87**, as it already was -- those
cases were never the problem. What moved is the residue: four hands that previously held a
dominating card the log does not list no longer do.

**The remaining seven are not the solver's to fix.** In those the hard constraints pin the
dominating card into the hand, so the log itself implies it was held even though that turn's hand
list omits it. That is the log's incompleteness (§9.4: 3.5% of cards humans are recorded playing
are missing from their own hand lists), not a preference the solver got wrong, and forcing it would
be exactly the guessing the corpus rules forbid.

**Unchanged by the rebuild:** 282 games convert with 0 failures, 144,844 samples, 84,073 with a
value target. The dataset was rebuilt on the new hands.


### 9.7 There is no game where a human provably held the dominating card

The seven trap discards §9.6 still scores as dominated were checked one by one for what the log
says about the alternative. In **all seven** the dominating card is absent from that turn's hand
list *and* is never played by that side during that turn. None of them is evidence of a human
passing over a card they demonstrably held.

Where those cards do appear is the pattern:

| replay | turn | alternative the solver placed | the log has it at |
|---:|---:|:---|:---|
| 71 | 5 | Duck and Cover | T1 (played), T6 (hand list, played AR1) |
| 80 | 6 | Liberation Theology | T7 (hand list, played AR7) |
| 179 | 4 | The Voice of America | T6 (hand list, played AR1) |
| 184 | 7 | Arab-Israeli War | T3, T9 |
| 198 | 5 | Socialist Governments | T2, T7 |
| 283 | 5 | Arab-Israeli War | T2, T6 |

Every one is a card the log places in that player's hands **in other turns**, which the solver has
carried into the turn in question -- one to two turns before the log first lists it. Carrying a card
over is ordinary and the hand lists are demonstrably incomplete (§9.4), so the placements are not
illegal; they are simply unevidenced, and each manufactures the appearance of a mistake.

**So the corpus contains zero proven violations of the dominance relation.** §9.5's "0 of 87 where
the log records both cards" is not a restriction that hides the counterexamples -- there are none
to hide. Across all 94 decidable trap discards, every apparent human error rests on a card the log
does not put in that hand at that time.

**A refinement this suggests, not made.** The solver could pay a cost for holding a card in turns
before the log first lists it, which is what all six distinct cases have in common. It would want
care: carry-over is real, the lists are incomplete, and a hard version would contradict the corpus
rule against forcing what the log does not state. Worth trying as another soft clause if the
residue matters.


### 9.8 The Junta counterfactual, and two bugs it found in §9.6's heuristic

**Question (owner's).** In replay 80, would the objective be better with **Junta** in the turn 6 US
hand instead of Liberation Theology? And carry-over should not be penalised (§9.7's suggestion),
because carry-over is the primary way a hand is reconstructed at all.

**Junta specifically: no, the log forbids it.** Pinning Junta into that hand is **unsatisfiable**.
It appears in this game only in the turn 7 US hand list and is never played, so at turn 6 the hard
constraints place it elsewhere.

**But the probe found the real problem.** Liberation Theology is *not* forced either — pinning it
out is satisfiable, at total soft cost **152** against the chosen **151**. A card responsible for
two of the apparent human mistakes was being decided by a margin of **one**. It should have been
carrying §9.6's +120 dominance penalty, and it was not, for two reasons:

1. **Only the last trap discard of a turn was kept.** A trap holds until the escape roll lands, so
   one turn can force several discards. At turn 6 the US discarded three times — Warsaw Pact
   Formed, Our Man in Tehran, Nuclear Subs — and `trap_discard` recorded only the third.
2. **The two sides of the pair were given the same eligibility.** One-time (starred) events are
   excluded as the *dominant* side, because discarding one removes it from the game permanently
   and that is a different argument. They are perfectly ordinary as the *discarded* side, and
   `ai/eval/dominance` splits exactly there. Treating them alike meant all three of the US's
   starred discards were ignored and the turn contributed no evidence at all.

Both fixed: every discard in the turn is kept, and `_discard_excluded` is now separate from
`_dominance_excluded`.

**Effect on the corpus:**

| | dominated | rate |
|:---|---:|---:|
| §9.3, with the headline bug | 25/113 | 22.1% |
| headlines excluded (§9.5) | 11/98 | 11.2% |
| first heuristic (§9.6) | 7/94 | 7.4% |
| **heuristic corrected** | **3/90** | **3.3%** |
| — Bear Trap | 1/60 | 1.7% |
| — Quagmire | 2/30 | 6.7% |

282 games still convert with 0 failures. The rebuilt dataset is 144,845 samples with 84,074 value
targets, one more than before — a hand changed somewhere and a decision came with it.

**Carry-over is not penalised**, per the owner: it is how hands are reconstructed in the first
place, and costing it would attack the mechanism rather than the error. The three remaining cases
stand as they are.


### 9.9 Why the log allows Liberation Theology at turn 6 but not Junta

Both cards appear in replay 80 only in the turn 7 **US** hand list, so on the face of it the solver
should treat them alike. It does not, and the reason is a line outside the hand lists.

| | Junta | Liberation Theology |
|:---|:---|:---|
| turn 7 US hand list | listed | listed |
| who actually spent it | **USSR** — `USSR Headlines Junta` | **US** — space race at AR7 |

The hand list and the play disagree about Junta, and the play wins: this is one of the 4
`hand_reattributions` in this game, the mechanism `tools/README.md` describes for lists that give a
card to the wrong side. So Junta is the USSR's at turn 7.

From there the constraint follows without any preference being involved:

1. The USSR spent Junta at turn 7, so Junta was in the USSR's hand at turn 7.
2. **All seven** of the US's turn 6 action rounds are recorded — Warsaw Pact Formed, Our Man in
   Tehran, Nuclear Subs, The Voice of America, Shuttle Diplomacy, How I Learned To Stop Worrying,
   SALT Negotiations — and Junta is not among them.
3. A card in a hand that is not played or discarded stays there, so Junta in the US hand at turn 6
   would still be in the US hand at turn 7.

Which contradicts (1). There is a reshuffle before turn 7, so a discard-and-redraw route exists in
principle — but (2) closes it, because the US's turn is fully recorded and Junta was not discarded.

Liberation Theology has none of that: the US spent it at turn 7, so holding it from turn 6 is
consistent, and only the soft preferences decide whether it was. §9.8 established that they decide
it by a single unit of cost, which is why it looked arbitrary.

**No fault found.** The asymmetry is entailed by the log, and the evidence for it is in the headline
line rather than in the hand lists the question naturally looks at.


### 9.10 Two corrections to the hold cost, from the owner

**1. A one-time card of your own, in a quiet turn, is not evidence against holding.** `_hold_cost`
charged a flat 40 for holding any card of your own side, on the reasoning that a hand is for
spending. That is right for a recurring event and wrong for a one-time one: it is played once, at
a moment that suits it, and saving it is ordinary. §9.9 showed the charge doing real work — it was
the whole of the +38 that rejected Sadat Expels Soviets, a starred US card the US demonstrably held
the following turn.

The charge is now dropped for a one-time card of your own **when that side fired none of the
opponent's events that turn**. A turn with no opponent event fired is consistent with a hand that
simply held no opponent cards, so nothing about it argues against having kept your own. Where an
opponent event *was* fired the side was holding opponent cards and had a choice about what to keep,
so the charge stands. **CIA Created** and **"Lone Gunman"** keep it always: they are played to see
the opponent's hand, and nobody sits on them.

**2. Holding the opponent's card while giving your own to a trap is now heavily penalised, at any
Ops.** §9.6's clause required the two cards to print the same Ops, mirroring the dominance
*measurement*. Under a trap that is too narrow: you cannot play anything while trapped, so the Ops
of what you give up buys nothing, and keeping the opponent's event is worse whatever it prints.
The clause now applies to any opponent card except Five Year Plan, the China Card and scoring
cards, and the weight goes 120 → **240**, above the top of `_hold_cost`'s range.

One-time opponent events are no longer excluded either. `ai/eval/dominance` excludes them so that
every *measured* pair is strictly defensible; as a claim about what a hand held the direction is
the same and stronger, since nobody keeps the opponent's one-time event while giving up their own.
The measurement module is unchanged — only the solver's preference is widened.

**Effect.** 282 games still convert with 0 failures.

| | before | after |
|:---|---:|---:|
| dominated trap discards | 3/90 (3.3%) | 3/93 (3.2%) |
| decidable space plays | 712 | 749 |
| dataset samples | 144,845 | 144,839 |

The trap figure barely moves because §9.8 had already taken it near the floor; what changed is the
hands themselves, which is what the dataset rebuild reflects. 961 tests pass.


### 9.11 Agreement with human play, counted without the order of a placement

**The measure was wrong, and by construction.** A card played for Operations spends its points one
at a time and the engine asks a separate `POINT_NODE` question for each, so the log's order is
whatever the recording happened to write. Placing two Influence in Angola and one in Zaire is the
same play in any order, and scoring each point against the index the human's sequence happened to
hold marked the model wrong for reordering a play it agreed with. The same holds for every event
that spreads or removes several points -- Decolonization, De-Stalinization, Colonial Rear Guards,
Ussuri River Skirmish, Puppet Governments, COMECON, Marshall Plan, The Reformer, and for removals
Socialist Governments and East European Unrest.

`ai/eval/agreement.py` groups consecutive point decisions belonging to one play and scores the
group on the multiset of countries rather than the sequence. The model is teacher-forced along the
human's trajectory, so its own earlier choices cannot take it somewhere the human never went, and
each point still contributes exactly one comparison -- the two figures are directly comparable and
only permutations are forgiven. Two points into one country are two entries, so agreeing on the
country but not the weight still costs.

**It matters less than expected.** Over 60,670 decisions from 120 replays, of which **34% sit in
multi-point plays**:

| | ordered | unordered | gain |
|:---|---:|---:|---:|
| BC on the human corpus | 46.04% | 46.34% | +0.29 |
| BC on self-play | 33.74% | 34.34% | +0.60 |
| E3 arm B (human) final | 32.41% | 32.96% | +0.56 |
| E3 arm A (self-play) final | 32.18% | 32.93% | +0.76 |

**So the ordering artefact was worth about half a point, not the several it might have been.** The
reason is teacher forcing: at the second point of a play the model already sees the board after the
human's first placement, so where it disagrees it is usually disagreeing about *which* countries,
not about the order. Every figure quoted earlier in §9 was pessimistic by roughly this much, which
changes no conclusion in it -- §9.1's washout still lands at ~32-33% either way.

The correct measure is now the one to use, and `play` is stored as a dataset column so it can be
applied without re-running conversion, which is the expensive part.


### 9.11.1 Coups and realignments are excluded from reordering

Not every run of point decisions is order-free, and §9.11 treated them all as if they were. The
board changes between points wherever a die is involved: a **realignment** roll is made against the
influence the last one left, so a different order is a different sequence of odds, and the same
holds for **coups** — in particular **Che**, whose second coup is offered only if the first removed
influence, so the pair is a sequence and not a set.

Those are now scored strictly. Implemented as a **blacklist** rather than a whitelist of the
order-free cases, per the owner: spreading Influence is the ordinary case, and a card that spreads
it in some new way should be handled without anyone having to remember to add it. Detection needed
`DecisionContext.op_mode`, which was not exposed to Python.

**It changes the numbers barely at all.** Grouped decisions fall from 34.0% to **31.8%** of the
total, and the correction each model gets is unchanged to within 0.01 points:

| | ordered | unordered | gain |
|:---|---:|---:|---:|
| BC on the human corpus | 46.04% | 46.32% | +0.28 |
| BC on self-play | 33.74% | 34.33% | +0.59 |
| E3 arm B (human) final | 32.41% | 32.96% | +0.55 |
| E3 arm A (self-play) final | 32.18% | 32.93% | +0.75 |

Which is worth knowing in itself: the reordering credit was never resting on coups and
realignments being wrongly forgiven, so §9.11's figures stand as measured. The measure is now right
for the right reason rather than by luck.


### 9.11.2 Agreement is now the reported figure everywhere

`ai/eval/agreement.evaluate_dataset` takes either dataset and returns both figures, so nothing has
to re-implement the measure. A directory is the human corpus, which stores the play grouping as a
column; a file is the self-play set, whose loader gained `stream_with_plays` and recovers the
grouping while replaying, since that format keeps only a seed and the actions.

BC warmup now reports it every epoch, for both datasets:

```
Epoch  2/ 2 COMPLETED | Loss: 1.8802 | Strict Acc: 44.81% |
    Agreement: 47.48% (ordered 47.31%, 20,000 decisions)
```

Three numbers because they answer different questions. **Strict Acc** is the running in-batch
figure, computed on shuffled batches while the weights are still moving, and is what the trainer
always printed. **Agreement** is the measure: an unshuffled pass after the epoch, scoring a play on
the multiset of countries. **ordered** is that same pass scored strictly, so the gap between the
last two is exactly what reordering costs and nothing else.

The pass is capped at 20,000 decisions. The self-play format rebuilds its observations by replaying
from a seed, so a full pass over 2.1M samples would take minutes per epoch; 20,000 gives a figure
stable to about a tenth of a point.

Note the in-batch and post-epoch numbers differ by a few points (44.81% against 47.31% here) and
should: one averages over an epoch of changing weights, the other measures the weights the epoch
ended with.


### 9.12 Injection frequency vs alignment — quick arms, INCONCLUSIVE

**Question.** §9.1 showed a BC warmup washes out in ~2M steps. Does interleaving supervised steps on
human data during RL hold alignment up, and how does the frequency matter?

**Setup.** 30-minute budget. Four arms, all from the *same* synthetic BC init (35.0% agreement),
4M steps each, two at a time; one flag apart -- `--inject-every` 0/16/4/1 at weight 1.0, a separate
AdamW at lr 1e-4 on human batches of 512 with value targets masked by `has_outcome`. Agreement
measured per snapshot on a fixed 20,000-decision probe (§9.11's measure).

All four completed 4M steps with ten snapshots each, so the endpoints are at equal step counts:

| `--inject-every` | trajectory | end |
|:---|:---|---:|
| 1 | 35.0 → 34.5, range 32.8-35.8 | **34.5** |
| 4 | 35.0 → 33.9, range 33.2-35.2 | 33.9 |
| 0 (control) | 35.0 → 33.9, range 33.7-35.0 | 33.9 |
| 16 | 35.0 → 32.7, range 32.0-35.0 | **32.7** |

**The endpoints order monotonically with frequency**, and injecting *rarely* lands below not
injecting at all. That is the direction predicted when this was proposed: a dose rarer than the
~2M-step washout perturbs the policy without establishing anything, and RL spends the interval
walking back, so the arm oscillates rather than holds. Weak support, but it is the predicted sign.

**Verdict: still inconclusive.** The spread across all four arms is 1.8 points and the control alone
ranges 1.3 across its own snapshots. One seed at 4M steps cannot separate a monotone ordering from
four samples of the same wobble.

**The design flaw is the starting point.** Every arm began from the synthetic BC init at 35.0%,
which is already at the ~32-35% attractor RL converges to (§9.1). There was almost no alignment to
preserve, so the experiment measured whether injection can *raise* agreement rather than whether it
can *prevent* the washout it was built for. The informative version starts from the **human** BC
init at ~47% and asks whether injection stops the decay to 32%. That is one flag different and the
right thing to run next.

**Also worth noting:** at weight 1.0 even every-iteration injection did not pull agreement toward
the BC ceiling of ~47%. Either the weight is too low against the RL updates, or RL actively pulls
away from human play. Those are different problems and the human-init arm distinguishes them: if
injection holds ~47% there, the weight is fine and the synthetic start was the issue; if it decays
anyway, the pull is real and the weight has to rise.


### 9.12.1 The warmup nets are unharmed by the corpus rebuilds — reference figures

The human BC net was trained against an earlier build of the corpus, before the solver changes of
§9.6/§9.8/§9.10 and before the `play` column, and the measure itself changed in §9.11. Checked
against the corpus and the measure as they now stand, over **all 144,839 decisions** rather than a
probe:

| net | ordered | agreement |
|:---|---:|---:|
| `e3_warmup_human` (BC on the corpus) | 45.94% | **46.23%** |
| `e3_warmup_synth` (BC on self-play) | 33.64% | 34.23% |
| `inj_none` snapshot_0s | 33.64% | 34.23% |
| `inj_every1` snapshot_0s | 33.64% | 34.23% |

**Nothing broke it.** The human warmup still sits where it did — the training-time figure was 45.4%
strict in-batch, and the earlier probes read 46.0-46.8% on subsets; 46.23% over the whole corpus is
the authoritative number and should be quoted in preference to those.

**And the arms started where they were meant to.** Both `snapshot_0s` files agree with
`e3_warmup_synth` to the decimal, which confirms independently that §9.12's four arms all began
from the same synthetic initialisation and that `--warmup-checkpoint` loads what it says. That was
assumed rather than checked when those arms were described.


### 9.13 BC on human games from a self-play net — it works, and it works *better*

**Question.** §9.2 measured behaviour cloning from a fresh network: held-out agreement climbs to
about 47-48% by epoch 8 and then memorises. Does a net already trained on self-play demonstrations
reach the same place, and does that start help or hinder?

**Setup.** Split by game (224 train / 56 held out, 117,021 / 27,818 samples), the same seed as
§9.2, agreement measured with §9.11's measure on the held-out games after every epoch.

| epoch | 0 | 1 | 2 | 4 | 6 | 8 | 10 | 12 |
|:---|---:|---:|---:|---:|---:|---:|---:|---:|
| **from the self-play net** | 34.18 | 45.58 | 47.15 | 47.58 | 48.61 | **49.33** | 50.04 | **49.99** |
| from scratch | 20.56 | 42.32 | 43.49 | 44.52 | 46.26 | 47.08 | 47.91 | 48.24 |

**Yes, and it dominates the from-scratch curve at every epoch.** Starting from the self-play net is
ahead by 3.3 points after one epoch and still ahead by 1.8 at twelve, finishing at **50.0%** against
48.2%. The self-play pretraining is not something the human data has to overcome; it is a better
starting representation, and the human data builds on it.

That is worth its own line: **the best human-aligned net available is not the one trained on human
games alone.** `e3_warmup_human`, BC from scratch, measures 46.23% over the full corpus (§9.12.1).
Two epochs of the same data on top of the self-play net beats it, and eight epochs reach 49.3%.

**Consequence for the arm that was never run.** The synthetic-then-human fine-tune was proposed as
round two of E3 and postponed for VRAM. This is direct evidence its *initialisation* is the best of
the three tried -- better aligned than human-only BC and far better than self-play alone -- which
removes the confound §9 could not escape: a human-data arm no longer has to start from a weaker
policy than its control.

**Caveat.** Both curves are still rising slowly at twelve epochs, so neither ceiling is established;
what is established is the gap between them, which is stable across every epoch measured. And the
from-scratch numbers here run about half a point above §9.2's on the same split, which is the
corpus having been rebuilt since (§9.6, §9.8, §9.10) -- the comparison inside this table is
like-for-like, comparisons across sections are not.


### 9.14 Injection from a human start — it halts the washout

**Question.** §9.12's arms all began from the self-play init, already at the ~32-35% attractor, so
they could only ask whether injection *raises* agreement and answered inconclusively. This is the
version they should have been: same four-million-step budget, same settings, but starting from the
human BC net at **47.2%**, where there is something to lose.

| snapshot | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 |
|:---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| control, no injection | 47.2 | 43.0 | 39.0 | 36.7 | 34.8 | 34.0 | 34.4 | 32.1 | 30.9 | **30.9** |
| `--inject-every 1` | 47.2 | 42.9 | 40.8 | 39.6 | 37.8 | 38.2 | 37.0 | 37.7 | 38.3 | **38.3** |

**Injection does not prevent the decay, it stops it.** Both arms fall for the first two snapshots
and then separate: the control keeps going all the way to 30.9%, reproducing §9.1's washout, while
the injected arm flattens at 37-38% and stays there for the second half of the run. The gap at the
end is **+7.4 points**, against a run-to-run range of about 1.3 (§9.12). This is not noise.

**It also settles what §9.12 could not.** The 0.6-point difference there was not evidence that
injection is weak; it was evidence that an arm starting at the attractor has nothing to preserve.
Same mechanism, same weight, same budget -- only the starting point differs, and the effect goes
from 0.6 to 7.4.

**What it does not do is hold the BC level.** 47.2% → 38.3% is still nine points lost, so at weight
1.0 injection is fighting the RL gradient to a draw somewhere below where it started rather than
holding its ground. Weight is the obvious next knob, and the one this pair does not vary.

**One oddity worth recording.** The control ends at 30.9%, *below* the 33.9% a self-play-initialised
control reached on the identical budget (§9.12). Starting closer to human play and then training
away from it appears to overshoot past where you would have been having never started there. One
seed, so it may be nothing -- but it is the opposite of what you would guess.


## 10. Injection weight and frequency (E5) — frequency is everything, weight is nothing

**Setup.** All nine arms start from the same checkpoint: self-play BC, then 8 epochs of human BC on
the **224 train games only**, reaching **49.33%** agreement on the 56 held-out games. That split
matters — a first attempt fitted the warm start on all 280 games and read 65% "held out", which was
memorisation, so it was thrown away and rebuilt. Every figure below is on games no arm has been
fitted to. 4M steps each, two at a time, `blunder_aware` + K=40, one flag apart.

### 10.1 Weight, at every iteration — no effect across 16x

| `--inject-weight` (every=1) | end |
|:---|---:|
| 0.25 | **40.4%** |
| 1 | 40.2% |
| 4 | 39.2% |
| control, no injection | 35.6% |

A sixteenfold range of weight spans 1.2 points, while the gap to the control is 4.6. Injection is
close to a switch: whether it happens matters, how hard barely does.

### 10.2 Frequency, at fixed weight — monotone, and gone by every 4th iteration

| `--inject-every` (weight=1) | end | over control |
|:---|---:|---:|
| 1 | **40.2%** | +4.6 |
| 2 | 38.8% | +3.2 |
| 4 | 36.6% | +1.0 |
| control | 35.6% | — |

The benefit decays steadily as the dose gets rarer and is nearly gone by every fourth iteration.

### 10.3 Equal dose, different schedule — rarity cannot be bought off with weight

Holding `weight / every` at 1.0, so every arm applies the same total supervised signal:

| schedule | end |
|:---|---:|
| w=1, every=1 | **40.2%** |
| w=4, every=4 | 35.9% |
| w=16, every=16 | 34.7% |
| control | 35.6% |

**Same total dose, 5.5 points apart, and the two rare schedules are at or below the control.**
w16e16 is *worse than not injecting at all*.

### 10.4 What this means

**Frequency is the parameter; weight is not.** The signal has to be applied continuously or it does
nothing, and a bigger dose delivered less often is not a substitute — it is worse than nothing,
because each large correction knocks the policy somewhere RL then has to walk back from, and the
interval is long enough for it to do so. That is the oscillation predicted when injection was
proposed, now with matched-dose evidence rather than a guess.

The practical setting is **every iteration, at whatever weight is convenient** — 0.25 works as well
as 4. Anything rarer than every second iteration is not worth running.

**What it still does not do is hold the BC level.** The best arm decays 49.3% → 40.2%: injection
halves the washout (the control loses 13.7 points, injection loses 9.1) but does not stop it. Since
weight does not help, the remaining levers are elsewhere — a lower RL learning rate, or injecting
against a frozen copy rather than the live policy.

**Caveats.** One seed per cell, 4M steps, and a control that moved 1.3 points across its own
snapshots in §9.12. The weight ladder's 1.2-point spread is inside that; the frequency and
matched-dose effects, at 4.6 and 5.5 points, are not.


## 11. The gate: what does the space-race dominance error actually cost? — ~3 points, and that changes the plan

**Question.** §9.3 found the largest behavioural gap between humans and our agents: spacing a card
while holding an equal-Ops opponent recurring event, where humans err 0.4% of the time and our
agents 12.6-18.1%. §10 then spent nine arms learning how to transfer human behaviour. None of it
asked what the behaviour is worth. §4.4 is the warning -- forced-win take rate looked like a 20-30%
failure and was worth about two points, because declining was usually free.

**Method** (`ai/eval/dominance_cost.py`). The fork has to be at card *selection*: by the time the
engine asks how to play a card the card is chosen, and which card went to the track is the decision.
Positions are taken at `SELECT_CARD` where the agent's own greedy continuation goes on to space its
own or a neutral card with an equal-Ops opponent recurring event in hand. Both branches -- the
agent's choice, and the opponent's card to the track instead -- are then continued by the same
policy over twelve die streams each, counting wins for the player who chose.

**Result** (`dec_turns40`, two collections pooled, 148 positions):

| branch | wins | rate |
|:---|---:|---:|
| dominated (what the agent wanted) | 766/1768 | 43.33% |
| dominant (the opponent's card instead) | 823/1772 | **46.44%** |
| **cost of the error** | | **+3.12 points** (approx SE 1.67, z ≈ 1.9) |

The two collections gave +5.15 (39 positions) and +2.39 (109 positions); the spread between them is
itself a fair warning about how noisy this is.

**Verdict: the error is not free, but the ceiling is low, and that is the finding.** Roughly three
points of win rate, in the same range as §4.4's forced wins. It clears the gate in the sense that
the behaviour matters -- but it puts a hard ceiling on the whole line, because three points is what
you would get for fixing the error **completely**.

**And E6 as planned cannot measure a fraction of three points.** §10's best arm transfers about half
the alignment it starts with (49.3% → 40.2%), so the expected strength gain is one to two points.
§7.2 put the tournament noise floor at ~1.5 points on 1,000 games a matchup, and training seed
variance has never been measured at all. A two-arm, one-seed E6 would return a number
indistinguishable from noise whichever way it came out, and we would not be able to tell which had
happened.

**What that argues for.** Either run E6 with several seeds per arm and accept the multiplied cost,
or stop treating strength as the target for this line and use the corpus for what it demonstrably
is -- the only reference we have for what good local play looks like, and an evaluation asset. The
one thing not worth doing is the cheap version of E6, which would produce a number nobody should
believe.

**Caveats.** The SE above treats games as independent; they are not -- twelve rollouts share a
position, and the two branches are paired -- so it is approximate in both directions. One
checkpoint (`dec_turns40`); a differently-trained agent may pay a different price for the same
mistake.

## 12. Battlegrounds: the critic prices *control*, not the road to it — SETTLED

§4 records that battlegrounds sit empty in late positions and that the count plateaus from turn 8.
The tidy explanation was a self-reinforcing loop: the agent never holds battlegrounds → never sees a
scoring card pay out on them → the critic never learns they are worth anything → the policy has no
reason to take them. If that were right, the fix would be exploration.

It is not right. Measured on `ai/eval/battleground_value.py` against the 160M run
(`data/checkpoints/run_v2_20260907_long160M`, snapshots at 0 / 35M / 70M steps), turn-stratified
self-play positions, 30 per turn bucket, perturbing the board one way at a time and reading Δ`v_win`
from the mover's side.

### 12.0 Which snapshots these are

Everything below is measured on three checkpoints from one run,
`data/checkpoints/run_v2_20260907_long160M` — `--arch v2`, `--reward-scheme blunder_aware`,
`--decisiveness-turns 40`, `--num-envs 512`, `--train-steps 160000000`,
`--snapshot-every-steps 5000000`, `--inject-dataset data/datasets/human_corpus --inject-every 1
--inject-weight 1.0`, launched from a dirty tree on `5159bfe` (the dirt being the
`--snapshot-every-steps` support itself).

| name used here | file | what it is |
|---|---|---|
| **warm start** | `snapshot_0s.pt` | zero RL steps — verified tensor-for-tensor identical to `data/checkpoints/warmup_synth_then_human_train.pt`, the §9.13 synthetic-then-human BC init fitted on the 224-game train split (49.33% held-out agreement). Pure behaviour cloning; no self-play has touched it. |
| **35M steps** | `snapshot_35061760steps.pt` | 35.06M env steps of NashPG with continuous human injection |
| **70M steps** | `snapshot_70057984steps.pt` | 70.06M env steps, same |

So "warm start → 70M" is a trajectory *within one run*, not a comparison across configurations,
and the human corpus is being injected throughout — the washout in 12.3 happens *despite*
injection, not in its absence. The run was still training when these were taken; later snapshots
exist and 12.3's open question is whether the collapse is monotone across all of them.

### 12.1 Control is priced; the road to it is not

At 70M steps, Δ`v_win` for the mover, each row adding the same 2 Influence except *control*, which
adds exactly enough to flip the country:

| perturbation | T2 | T3 | T5 | T8 |
|---|---|---|---|---|
| control a battleground | +0.061 | +0.074 | +0.060 | +0.046 |
| presence in a battleground (no control) | +0.024 | +0.024 | +0.028 | +0.011 |
| access: adjacent to a battleground | +0.025 | +0.034 | +0.026 | +0.011 |
| plain influence, no battleground near | +0.020 | +0.023 | +0.019 | +0.007 |
| opponent controls a battleground | −0.073 | −0.078 | −0.096 | −0.043 |

Standard errors run 0.003–0.016, so:

* **Control is real.** Three times a plain Influence point, and symmetric — losing a battleground to
  the opponent costs about what taking one gains. The critic is not blind to battlegrounds.
* **Presence is worth nothing extra.** Two Influence into a battleground that does *not* reach
  control reads the same as two Influence into a backwater with no battleground anywhere near it:
  +0.024 against +0.020 at T2, inside one standard error at every turn.

That is the whole finding. The critic has learned the *step function* — a country is worth
something once it flips and nothing before — and a battleground almost always takes two plays to
take. Every intermediate instalment of the investment is priced at zero, so the policy sees a
two-play sequence whose first play is free money spent for nothing. This is a credit-assignment
gap, not a knowledge gap, and exploration bonuses do not touch it.

The same step function is written into the one shaping potential the repo already has:
`Scoring::compute_useful_actions_potential` (`engine/src/scoring.cpp:326-338`) builds its
battleground term from `get_country_control`, counting controlled battlegrounds and nothing else.
Switching the 160M run from `blunder_aware` to `useful_actions` would therefore reward exactly the
same last-point-only shape. (That run used `blunder_aware`, so the potential was not in play; the
step function above is what the critic learned from terminal outcomes on its own.)

### 12.2 Access *is* priced, once the comparison is honest

Contrasting "I control Thailand and the opponent is shut out of every neighbour" against "…and the
opponent has 1 next door" showed a large effect — but that contrast **deletes** the opponent's
Influence, and the critic prices raw Influence loss regardless of where it was. The matched form
*relocates* one Influence point instead: a Thailand neighbour versus a non-adjacent,
non-battleground country in the same region. Both boards carry identical totals for both players,
and adjacency is the only difference.

At 70M steps, Δ`v_win`:

| contrast | T2 | T3 | T5 | T8 |
|---|---|---|---|---|
| the opponent's access to Thailand costs me | +0.010 | +0.016 | +0.030 | +0.043 |
| my access to a Thailand the opponent holds is worth | +0.038 | +0.028 | +0.024 | +0.018 |

Positive in 8 of 8 cells here, and in 8 of 8 at 35M — and one relocated Influence point moves the
value by as much as two points placed anywhere. **The critic does model access.** The Thailand
intuition is in the network already; it does not need to be taught.

### 12.3 What RL does to the early Americas

Per-Influence-point value of a USSR foothold in each region's battlegrounds, against the
plain-backwater baseline:

| region | warm start T2 | warm start T3 | 70M T2 | 70M T3 |
|---|---|---|---|---|
| Europe | +0.055 | +0.055 | +0.013 | +0.024 |
| Asia | +0.015 | +0.057 | +0.017 | +0.018 |
| Middle East | +0.045 | +0.058 | +0.019 | +0.022 |
| Africa | −0.010 | +0.015 | +0.008 | +0.011 |
| Central America | +0.012 | +0.104 | **−0.005** | +0.003 |
| South America | +0.053 | +0.105 | +0.007 | +0.015 |
| *(plain backwater)* | +0.015 | +0.047 | +0.005 | +0.006 |

Absolute magnitudes shrink everywhere as the value head sharpens, so the ratio to the backwater
baseline is the comparison that means anything. On that basis the human-BC warm start puts South
America among its **best** early destinations at T2 (3.6× backwater) and Central America among its
best at T3 (2.2×). By 70M steps both have collapsed to the **bottom** of the table — South America
1.4× at T2, Central America *negative* — while Europe, Asia and the Middle East hold at 3–4×.

**Part of that ordering is correct.** Europe, Asia and Middle East Scoring are `WarEra::EARLY`
(`engine/src/card_data.cpp:12-14`) and are in the deck from turn 1; Central America, South America
and Africa Scoring are `WarEra::MID` (lines 37, 79, 81) and cannot be drawn before the mid-war deck
is shuffled in. A turn-2 influence point in Europe can be cashed this turn and a turn-2 point in
Brazil cannot, so the early-war regions *should* rank above the Americas at T2. The table's top
half is not the defect.

The defect is the floor. Central America at −0.005 is below a backwater with no battleground
anywhere near it — the critic prefers spending the point in a country that can never be scored for
control or presence over a Central American battleground. South America at 1.4× backwater is barely
distinguishable from the same. Those regions are still worth something at turn 2: the mid-war deck
arrives on turn 4, positions built early are cheap because they are uncontested, and influence
placed there is what makes the region contestable when the scoring card does appear. A defensible
critic ranks them below Europe and above nothing.

The warm start clears that floor comfortably — South America at 3.6× backwater at T2, Central
America at 2.2× at T3. If anything it goes further than the deck argument supports: at T2 it ranks
South America (3.6×) alongside Europe (3.7×) and above the Middle East (3.1×), which is a stronger
claim for the early Americas than "worth something" and may be the human prior overshooting. Either
way the direction of travel is the point. RL does not fail to learn that early Americas presence
matters; it learns it from the human prior and then unlearns it, past a defensible ordering and
down through the floor. This is the §9 washout showing up in the critic rather than the policy.

### 12.4 What this changes

The deadlock in §4 was framed as exploration. It is not:

1. The critic prices control (3× a plain point) and access (a relocated point moves 1–4 points of
   win rate). Both halves of the board understanding are present.
2. What is missing is any value for **partial progress toward control** — and that is precisely the
   quantity a policy needs a gradient on, because taking a battleground costs two plays.
3. The early-Americas valuation exists at the warm start and is destroyed by RL, so injection is
   holding the wrong thing in place: it slows policy washout (§10) while the critic drifts anyway.

The lever that follows is a potential-based shaping term that is **continuous in distance to
control** rather than a step at control — φ rising with each Influence point that shortens the gap.
Potential-based shaping is policy-invariant (Ng, Harada & Russell), so it cannot change the optimal
policy, only the credit path to it. That is a change to `compute_useful_actions_potential` in the
C++ engine and therefore needs approval before anything is written.

Open: whether the step function is also present in `v_vp` (only `v_win` was measured), and whether
the collapse in 12.3 is monotone across all 32 snapshots or happens at a particular point in
training.

## 13. De-Stalinization and Decolonization on turn 2 — the model will not fire its own event

§12 perturbs the board. This perturbs the *hand*, at real human decisions rather than self-play
positions, using a new `on_decision` seam on `convert_game` that hands over the `GameState` as the
human faced it. `ai/eval/card_probe.py` then edits the hand and re-asks the node of either side.

Six corpus games where the USSR played one of the two cards on turn 2 — replays 142, 101, 143
(Decolonization) and 191, 158, 16 (De-Stalinization). In all six the human fired the **event**.
Positions are taken at the USSR's first turn-2 Action Round with the card in hand; headline nodes
are excluded, because neither answer under test exists at a headline. Snapshots as in §12.0.

### 13.1 As the USSR, holding its own card

Given that it plays the card, how does the policy want to play it? Share of the `SELECT_PLAY_MODE`
distribution on **event**, the human's choice in every one of these games:

| replay | card | warm start | 35M | 70M |
|---|---|---|---|---|
| 142 | Decolonization | 0.055 | 0.176 | 0.082 |
| 101 | Decolonization | 0.058 | 0.462 | 0.268 |
| 143 | Decolonization | 0.024 | 0.482 | 0.183 |
| 191 | De-Stalinization | 0.223 | 0.062 | 0.197 |
| 158 | De-Stalinization | 0.201 | 0.014 | 0.068 |
| 16 | De-Stalinization | 0.037 | 0.126 | 0.323 |

The rest is almost entirely **ops**. Not once, at any snapshot, does the event carry the
distribution. Played greedily to the end of turn 2 the USSR spends the card for Ops in 4 of 6 games
at the warm start and 3 of 6 at 70M.

A USSR card played by the USSR for Ops does not fire its event, so this is not a trade — it is
Decolonization bought as a 2-Ops filler and De-Stalinization, a one-time card, bought as 3 Ops and
gone. Decolonization places four Influence across Africa and South-East Asia and De-Stalinization
relocates four; two Ops of Influence placement is not a substitute for either. This is the §12
finding wearing different clothes: the agent will not spend a play on board presence whose payoff is
a scoring card several turns away, and here it declines even when the card hands that presence over
for free.

### 13.2 As the US, holding the same card

Same games, same turn-2 Action Round, the card swapped into the **US** hand in place of their
highest-Ops non-scoring card, so the swap cannot be read as having handed them a weaker hand.

The right answer is to hold it past the turn; failing that, to space it. Ops is the worst available
choice, not a neutral one — an opponent's card played for Operations still owes its Event
(`engine/src/state_machine.cpp:271`), so the US pays a play, hands the USSR the full event, and
keeps only the Ops. What actually became of the card by the end of turn 2:

| replay | warm start | 35M | 70M |
|---|---|---|---|
| 142 | ops | **held** | **held** |
| 101 | ops | *space* | *space* |
| 143 | ops | **held** | *space* |
| 191 | (game ended inside turn 2) | ops | *space* |
| 158 | ops | ops | ops |
| 16 | ops | ops | ops |

Nothing acceptable at the warm start, 3 of 6 at 35M, 4 of 6 at 70M. The mode distribution moves the
same way: space is worth 0.001–0.052 at the warm start and reaches 0.83–0.89 by 70M on the replays
it gets right.

**So RL is teaching this one, and the human prior is not.** That is the opposite of §12.3, where the
prior held the early-Americas valuation and RL destroyed it. The two are consistent under one
reading: self-play punishes handing the opponent a free event within the same game, quickly and
legibly, and it does not punish a thin position in Brazil until a scoring card that may never come.
RL learns what its reward can see.

The two failures at 70M are the sharp ones. In replay 16 the US wants the card *first* — p(select)
0.938, rank 1 of 6 — and then plays it for Ops, which is the most expensive way to hold a card it
should simply not have touched. Replay 158 is the same shape. Both are turn-2 positions where the
US has better Ops available and spends the opponent's card anyway.

### 13.3 Caveats

Six games, one greedy rollout each, one die stream: this locates a behaviour, it does not measure a
rate. The §9.3 dominance suite is the pattern to follow if a rate is wanted — the same question
asked across the whole corpus with a denominator of positions where the choice was actually
available. The USSR result in 13.1 is the one worth that treatment, since it is uniform across every
snapshot rather than trending.

Also unmeasured: whether firing the event would in fact have been better here, as opposed to merely
being what the human did. §11 is the warning — a behavioural gap is not a cost until the cost is
measured, and the fork-and-play-out method in `ai/eval/dominance_cost.py` transfers to this question
directly.

## 14. The same question asked of the whole corpus, with a measured human baseline

§13 asked six hand-picked games and reported a behaviour. This asks every early-war position in the
corpus (turns 1–3, Action Rounds), and — the part §13 was missing — measures what the humans
actually did, rather than asserting it. `ai/eval/early_war_cards.py`; snapshots as in §12.0, plus
15M and 100M.

Denominators: Decolonization, 276 USSR positions holding it (270 with a mode node) and 400 US
positions with the card swapped in; De-Stalinization, 451 (445) and 400.

### 14.1 What humans do — and it is nearly absolute

Read from the raw logs, so it covers games that do not fully convert:

| | n | event | ops | space |
|---|---:|---:|---:|---:|
| USSR, Decolonization | 91 | **100.0%** | 0 | **0** |
| USSR, De-Stalinization | 121 | **97.5%** | 2.5% (3) | **0** |
| US, Decolonization | 159 | 0.6% (1) | 0.6% (1) | **98.1%** |
| US, De-Stalinization | 47 | 6.4% (3) | 4.3% (2) | **89.4%** |

**The USSR never spaces either card in the early war: 0 of 212 plays.** It plays the event in 209
of 212. The three exceptions are all De-Stalinization for Ops — replays 292 T1 AR5, 58 T1 AR3,
319 T3 AR3.

For the US both *event* and *ops* mean the USSR got the event, since an opponent's card played for
Operations still owes it. That is 7 of 206 early-war plays, 3.4%. Checked one at a time:

* **replays 273 (T3, both cards), 253 (T1), 80 (T1)** — the US held more USSR cards than it had
  space-and-hold slots to absorb. In 273 the US hand carries Socialist Governments, Decolonization
  *and* De-Stalinization; one space and one hold cannot cover three. Structurally forced.
* **replays 315 and 316 (T1 AR6)** — the same game recorded twice (see 14.4). The US never spaced
  at all that turn, so spacing was available and unused. One genuine deviation, double-counted.
* **replay 176 (T1 AR4)** — the space race was already spent on Suez Crisis at AR2, but a hold slot
  was open and went to a US card instead. A genuine deviation.

So the standard the user stated holds, with two real exceptions in 206 plays and the rest explained
by having more opponent cards than slots.

### 14.2 The model, against that baseline

Greedy mode share conditional on playing the card at that node. Human column from 14.1.

**USSR — event is the right answer (human 100% / 97.5%)**

| snapshot | Decol event / ops / space | De-Stal event / ops / space |
|---|---|---|
| warm start | 20.7% / 79.3% / 0.0% | 18.0% / 82.0% / 0.0% |
| 15M | 13.7% / 73.3% / 13.0% | 12.8% / 71.5% / 15.7% |
| 35M | **9.3%** / 87.8% / 3.0% | **5.6%** / 91.2% / 3.1% |
| 70M | 23.3% / 43.0% / 33.7% | 12.4% / 51.9% / 35.7% |
| 100M | 37.8% / 45.6% / 16.7% | 31.7% / 49.4% / 18.9% |

**US — ops is the only wrong answer (human 1.3% / 10.6% let the event fire)**

| snapshot | Decol ops / space | De-Stal ops / space |
|---|---|---|
| warm start | 93.0% / 7.0% | 89.1% / 10.9% |
| 35M | 85.2% / 14.8% | 82.6% / 17.4% |
| 100M | 48.4% / 51.6% | 40.1% / 59.9% |

Three things this settles that §13 could not:

1. **The BC warm start never learned it either.** It is behaviour cloning on a corpus where the
   USSR fires the event 100% of the time, and it reproduces that choice 20.7% of the time. The gap
   is not created by RL; RL inherits it. §13 read the trajectory as RL degrading USSR handling,
   which was reading a dip as a trend — see the trough below.
2. **The trajectory is a dip, not a slide.** USSR event share falls to 9.3% / 5.6% at 35M and then
   recovers to 37.8% / 31.7% at 100M, *above* the warm start. On the US side the movement is
   monotone and large: 93% ops down to 48%.
3. **The model uses a mode the humans never use.** The USSR spaces these cards in up to 33.7% of
   positions at 70M, against 0 of 212 human plays. Spacing is legal in 85–98% of positions, so
   this is a preference, not an artefact of what was available.

### 14.3 What it does with the event when it fires it

Both cards allow an early stop, so firing and using are separate questions. Firing is the only
problem: every snapshot spends the whole event — 4.00 of 4 placements for Decolonization (3.94 at
100M), and the full 4 removals plus 4 placements for De-Stalinization.

Where it sends them is the second failure, and it tracks §12.3 exactly:

* **Decolonization**, warm start: Angola\*, Algeria\*, Nigeria\*, Zaire\*, Thailand\* — battlegrounds
  first. At 100M the top destination is **Cameroon ×92**, then Indonesia ×62, Zaire\* ×53,
  Tunisia ×47.
* **De-Stalinization**, warm start: **into** Venezuela\* ×95, Brazil\* ×62, Chile\* ×54,
  Argentina\* ×20, **out of** Romania ×74 and Finland ×45. That is the textbook plan — shed cheap
  Eastern European filler, buy the South American battlegrounds. At 100M it moves **out of East
  Germany ×259**, stripping a battleground it controls and needs for Eastern Europe, and scatters
  into Cameroon, Lebanon, Guatemala, Colombia.

So §12.3's early-Americas collapse is visible as behaviour and not only as a value number: the warm
start uses De-Stalinization to buy South America; by 100M the same card is used to gut its own
Eastern European battleground.

### 14.4 The corpus is 11% duplicates, and 4 held-out games leak

Found while checking the replay 315/316 exception: 315 and 316 have identical `all_turns` and differ
only in id and source URL. Fingerprinting every game's entries:

**300 files, 266 distinct games, 25 duplicated groups, 34 redundant copies (11.3%).** One game
appears nine times — ids 90, 214, 215, 216, 217, 218, 254, 298, 309.

Duplicates are weighted twice in the BC warm start and twice in every injection batch. Worse, the
train/held-out split is by replay id, and duplicates carry different ids: under the seed-7 split,
**3 duplicate groups straddle the boundary and 4 of the 56 held-out games (7.1%) were also trained
on** — held-out 125 is trained-on 118, held-out 216/217 are trained-on 90/214/215/218/254, held-out
235 is trained-on 232/233.

That is the same failure as the first synth+human warm start reading 65% "held out", in a milder
form: the 49.33% figure in §9.13 and every number derived from that split are inflated by whatever
7.1% memorisation is worth. Small, but it should be a fingerprint-based split, not an id-based one.

### 14.5 Bearing on injection

This run carries a human BC warm start **and** a human batch every single RL iteration at weight
1.0 (`--inject-dataset data/datasets/human_corpus --inject-every 1 --inject-weight 1.0`). The
injector trains on the 224-game train split, so roughly four fifths of the positions surveyed above
are in the injection stream every iteration, with the human's own action as the label.

Under that pressure the USSR still plays its own event in at most 37.8% of positions. Injection at
this strength does not hold the behaviour — but 14.2's first point says injection is not the whole
story either, because the pure BC init did not have the behaviour to hold. Before more injection is
tried, the thing to establish is why supervised learning on a 100%-consistent label reproduces it
20% of the time: a decision this uniform should be the easiest thing in the corpus to fit, and if
BC cannot fit it, the loss, the sampling, or the action encoding is where to look — not the
injection schedule.

## 15. Policy or critic? Asked on real held positions — and the answer differs by card

Two corrections to §14 first, both of which change numbers.

**Real positions, not constructed ones.** §14's US arm swapped the card into an arbitrary US node.
That answers a weaker question: the hand was never dealt to anyone, and the swap has to discard
something to make room, so the rest of the hand is wrong too. The corpus has real US-held positions
in quantity — **1029 for Decolonization, 636 for De-Stalinization** — and on those the model is
markedly better than the swap suggested:

| | swapped-in (§14.2) | really held |
|---|---|---|
| US spaces Decolonization, 70M | 49.2% | **68.5%** |
| US spaces Decolonization, 100M | 51.6% | **65.0%** |
| US spaces De-Stalinization, 100M | 59.9% | **67.0%** |

So §14 understated US competence by roughly 15 points. The USSR figures are unaffected — that arm
always used real hands — and 100M Decolonization cross-checks exactly: 102 of 270 is the 37.8%
already reported.

**The method.** From each real held position, resolve the card the human's way (event for the USSR,
space for the US) and the model's greedy way, with the model making every follow-on choice in
*both* branches so the mode is the only difference, then read `v_win` from the mover's own side.
`gap` is v(human) − v(model): positive means the critic prefers the human's line. The rightmost
column counts, among positions where the *policy* chose against the human, how often the *critic*
still preferred the human's. Snapshots to 140M.

### 15.1 Decolonization: the critic knows and the policy ignores it

USSR really holding it, 276 positions:

| snapshot | policy agrees | gap | critic sides with human |
|---|---|---|---|
| warm start | 21% | **−0.276** ± 0.051 | 50% |
| 35M | 9% | +0.140 ± 0.006 | 92% |
| 70M | 23% | +0.029 ± 0.005 | 69% |
| 100M | 38% | +0.079 ± 0.011 | 73% |
| 140M | 38% | **+0.225** ± 0.013 | **93%** |

At 140M the critic prefers firing the event in **93% of the positions where the policy chose not
to**, by a wide and well-determined margin — and the policy plays it for Ops anyway in 62% of
positions. For this card the two heads have come apart: **the value function has learned the right
answer and the policy is not following it.** That is the encouraging case, because it is what
policy imitation, or simply more of the same RL, can close.

Note also the warm start: gap −0.276, critic siding with the human only 50% of the time. The
behaviour-cloned init has *neither* head right, which corroborates §14.2's finding that BC never
learned this decision at all.

### 15.2 De-Stalinization: the critic does not know

USSR really holding it, 451 positions:

| snapshot | policy agrees | gap | critic sides with human |
|---|---|---|---|
| warm start | 18% | −0.176 ± 0.043 | 51% |
| 35M | 6% | +0.028 ± 0.004 | 65% |
| 70M | 12% | −0.018 ± 0.003 | 42% |
| 100M | 32% | −0.040 ± 0.007 | **32%** |
| 140M | 32% | +0.039 ± 0.008 | 59% |

The gap changes sign three times and at 100M the critic actively *prefers* the Ops line, siding
with the human in under a third of disagreements. **Both heads are wrong here**, and no amount of
policy imitation will hold a behaviour the critic scores as a mistake.

**The likely reason is not that the critic misprices the mode — it is that the model cannot execute
the card.** De-Stalinization asks for eight choices, four removals and four placements, against
Decolonization's four placements into one restricted region. §14.3 shows the execution: at 100M the
model fires De-Stalinization by stripping **East Germany ×259**, a battleground it controls and
needs for Eastern Europe, and scattering into Cameroon, Lebanon and Guatemala. A critic that scores
*that* below taking three Ops is not obviously wrong. Both branches here are resolved by the model,
so what the comparison shows is the value of the event **as this model would play it**, and for
De-Stalinization that is genuinely bad.

This makes the two cards a clean pair rather than a contradiction: Decolonization is hard to botch,
so the critic can price the mode and the policy is the only thing lagging; De-Stalinization is easy
to botch, the model botches it, and the critic prices the botched version accurately.

### 15.3 The US side: both heads improving, together

| | policy agrees (space) | gap | critic sides with human |
|---|---|---|---|
| Decolonization, warm start | 8% | −0.074 ± 0.020 | 44% |
| Decolonization, 140M | 55% | +0.063 ± 0.005 | 75% |
| De-Stalinization, warm start | 11% | −0.001 ± 0.026 | 51% |
| De-Stalinization, 140M | 59% | +0.055 ± 0.007 | 75% |

Both start with the critic mildly preferring the Ops line and end with it preferring space three
times out of four, while the policy moves from ~10% correct to ~55–59%. This is the one place in
§12–§15 where RL improves both heads steadily and in the same direction, and it is also the only
decision whose cost lands inside the same game — handing the opponent a free event is punished
immediately, which is exactly the credit-assignment argument §12.4 makes.

### 15.4 What to do with this

* **Decolonization is a policy-side fix.** The critic's own ranking already carries the answer at
  140M. Anything that makes the policy follow its own value estimate more closely — imitation,
  a sharper advantage, more training — should move it.
* **De-Stalinization is an execution fix first.** Teaching the policy to fire an event it plays
  badly makes things worse, and the critic is right to say so. The prerequisite is the §12.4
  shaping question: a value that is continuous in distance-to-control would price the removals out
  of East Germany correctly, and only then is firing the event worth imitating.
* The confound is stated above and is not removable by this method: both branches are resolved by
  the model, so a negative gap cannot distinguish "the critic misprices the mode" from "the critic
  correctly dislikes the model's execution". Separating them needs the human's own targets replayed
  from the log, which the converter has and this measurement does not yet use.

## 16. The human's own board against the model's round — the critic is wrong about the card

§15.2 found the critic scoring De-Stalinization below Ops and offered an excuse for it: both
branches there were resolved by the model, and the model executes that card badly (§14.3, East
Germany stripped 259 times), so a critic that dislikes the result might be right about the
execution rather than wrong about the card. **That excuse does not survive the test.**

**Method.** A second seam on `convert_game`, `on_entry`, hands over the board *after* a human entry
has been replayed and checked against the log's own next position — the humans' actual removals and
placements, not a model's replay of their mode choice. From the identical pre-round board the model
then plays that Action Round however it likes: its own card, its own mode, its own targets, with no
obligation to touch De-Stalinization. Both boards are scored by the same value head from the USSR's
side. `ai/eval/round_counterfactual.py`, 274 deduplicated corpus games, early war.

*Comparability check:* in all 105 rounds both boards end on the same turn, the same `action_round`,
and in `Phase.ACTION_ROUND` — so the gap measures the play, not where the two branches stopped.

### 16.1 De-Stalinization: 105 real human rounds

| snapshot | v(human) | v(model) | gap | critic prefers human | model played it too |
|---|---|---|---|---|---|
| warm start | −0.263 | −0.241 | −0.022 ± 0.051 | 51% | 87/105 |
| 35M | +0.055 | +0.110 | −0.055 ± 0.010 | 32% | 36/105 |
| 70M | −0.157 | −0.100 | −0.057 ± 0.009 | 28% | 25/105 |
| 100M | −0.316 | −0.158 | **−0.158** ± 0.022 | **21%** | 7/105 |
| 140M | −0.010 | +0.087 | −0.098 ± 0.016 | 28% | 23/105 |

**Every snapshot prefers its own round to a real human De-Stalinization**, and the margin is widest
at 100M, where the critic sides with the human in 21% of rounds. With the humans' own competent
execution on the board, the critic still says its own line is better. So this is not the model
mispricing its own bad targeting. **The critic is wrong about the card.**

### 16.2 What it prefers instead, and this is the alarming part

The card the model chooses instead is not the whole story — spacing an opponent's card is correct
play, and only firing its event is an error — so the mode has to be read too. In those same 105
rounds:

| | 100M | 140M |
|---|---|---|
| **US card played for Ops** (fires the US event) | **38.1%** | **34.3%** |
| US card spaced (correct) | 21.9% | 8.6% |
| USSR card played as event | 7.6% | 12.4% |
| USSR card played for Ops | 6.7% | 22.9% |

Most common single choices at 100M: Truman Doctrine [US] for Ops ×16, Five Year Plan [US] spaced
×11, Five Year Plan [US] for Ops ×8, Special Relationship [US] for Ops ×6, Defectors [US] for Ops
×5. At 140M the top pick is its own De-Stalinization played for **Ops** ×16, then Truman Doctrine
[US] for Ops ×13 and Five Year Plan [US] for Ops ×10.

So in the modal case the USSR declines its own strongest early event in order to play a *US* card
for Operations — firing Truman Doctrine or Five Year Plan against itself — and the value head rates
the resulting board above the human's. §14 found the US doing this with USSR cards; the error is
symmetric, it runs in both directions, and the critic endorses it.

### 16.3 This refines §15, it does not contradict it

§15 asked a within-card question: *given* that you play this card, is event better than Ops? On
that question the critic is right about Decolonization (93% of disagreements at 140M, gap +0.225).
§16 asks a between-card question: is playing the card at all better than what else the model would
do? There the same critic is near-neutral on Decolonization —

| snapshot | gap | critic prefers human |
|---|---|---|
| warm start | −0.015 ± 0.050 | 41% |
| 35M | +0.012 ± 0.012 | 57% |
| 70M | −0.008 ± 0.013 | 47% |
| 100M | −0.035 ± 0.017 | 49% |
| 140M | +0.041 ± 0.019 | 63% |

— drifting positive only by 140M, and much weaker than the within-card result. Both readings are
true at once: the critic has learned how to play a card it has decided to play, and has not learned
which card to play. That is a narrower and more useful statement than either section alone.

### 16.4 Consequences

* **§15.4's split was wrong about De-Stalinization.** I proposed execution as the prerequisite. It
  is not sufficient: even with human execution on the board the critic prefers its own line, so
  fixing targeting alone would leave the value function still steering away from the card.
* **Opponent-card discipline is the bigger error and it is bidirectional.** A third of rounds at
  both 100M and 140M are a US card played for Ops by the USSR. §11 measured what the space-race
  dominance error costs (~3 points); this one has never been costed and looks larger, since it
  hands over a full event rather than a dominated space.
* **The critic, not the policy, is the thing to fix here.** Imitation, injection and longer RL all
  push the policy toward the human line while the value function pushes back. §12.4's shaping
  proposal is aimed at the same organ, and this is a second, independent reason to take it up.
* Open, and now the cheapest thing to measure: the cost of playing an opponent's card for Ops,
  using the fork-and-play-out method in `ai/eval/dominance_cost.py`. If it is worth several points,
  it outranks everything in §9–§11.

## 17. Playing both boards out — the on-policy defence is half right, and §16 was over-stated

§16 concluded "the critic is wrong about the card" from value gaps alone. That inference has a hole:
`v_win` is an **on-policy** estimate. If this policy will not defend or build on De-Stalinization's
spread, the spread really is worth less to it than to a human, and a critic reporting that is
accurate rather than miscalibrated. §12 gives every reason to expect exactly that.

The only way to separate the two is to finish both boards under the model's own policy and see which
actually wins. 12 rollouts per board, temperature 0.1, `ai/eval/round_counterfactual.py`.

**Paired per position.** Pooling 12 rollouts of 105 positions and quoting a binomial error over
~1260 games overstates precision badly: twelve rollouts of one position are not twelve independent
games, and the unit that repeats is the position. Differencing the two arms position by position
also removes the position's own difficulty, which dominates the variance. All figures below are
per-position means with a standard error over positions.

### 17.1 De-Stalinization, 105 positions

| snapshot | realized gap | critic predicted | critic error |
|---|---|---|---|
| 100M | **−0.91** ± 2.56 | −7.89 ± 1.08 | **−6.98** ± 2.47 |
| 140M | **+5.46** ± 2.75 | −4.88 ± 0.80 | **−10.34** ± 2.88 |
| 160M final | **+5.11** ± 2.83 | −3.42 ± 0.73 | **−8.53** ± 2.87 |

Two findings, and they point in different directions.

**The on-policy defence holds at 100M.** The human's De-Stalinization board is worth −0.91 ± 2.56 to
that policy — indistinguishable from nothing. A human's competent spread genuinely bought the 100M
model no wins at all. The critic's *sign* was right, and the intuition behind it is right: this
policy could not use the position.

**It stops holding after that, and the critic never notices.** By 140M and 160M the same human
boards are worth **+5.46** and **+5.11** win-rate points — the policy learned to capitalize — while
the critic still predicts −4.88 and −3.42. Its error is −6.98, −10.34 and −8.53 points, every one of
them 2.8σ or more. So the critic is not merely reporting a weak policy; it is **systematically
over-pessimistic about human boards by 7–10 win-rate points**, and it did not update when the policy
improved underneath it.

For scale: §11 measured the space-race dominance error at ~3.12 points and treated that as the
ceiling on a whole line of work. A human's De-Stalinization round is worth **+5.11 points** to the
finished model, and the model plays that card as the human would in a minority of positions.

### 17.2 Decolonization, 79 positions — the critic is roughly right

| snapshot | realized gap | critic predicted | critic error |
|---|---|---|---|
| 100M | +2.33 ± 2.25 | −1.76 ± 0.84 | −4.09 ± 2.39 |
| 140M | +1.34 ± 2.61 | +2.04 ± 0.96 | **+0.70** ± 2.74 |
| 160M final | +3.01 ± 2.56 | +0.44 ± 0.86 | −2.57 ± 2.65 |

By 140M the critic's error is +0.70 ± 2.74 — calibrated. **The miscalibration is card-specific**,
concentrated on De-Stalinization, which is the card with eight choices and the one §14.3 shows the
model executing worst.

### 17.3 Correcting §16.2 — the "opponent card for Ops" bucket was too crude

§16.2 reported that 38.1% of rounds at 100M were "a US card played for Operations by the USSR,
firing the US event against itself", and presented the whole bucket as error. That is wrong, and the
largest component of it is not an error at all. Measuring what each play actually costs the USSR
(before minus after, so positive means Influence lost):

| card | mode | n (100M) | Europe Influence lost | total | cards lost |
|---|---|---:|---:|---:|---:|
| Truman Doctrine [US] | ops | 16 | **+1.06** | +0.50 | 1.00 |
| Five Year Plan [US] | space | 11 | 0.00 | 0.00 | 1.00 |
| Five Year Plan [US] | ops | 8 | 0.00 | −2.75 | **2.00** |
| Special Relationship [US] | ops | 6 | −0.67 | −0.67 | 1.00 |
| Defectors [US] | ops | 5 | −0.20 | −1.80 | 1.00 |
| De-Stalinization [USSR] | ops | 4 | 0.00 | −3.00 | 1.00 |

**Truman Doctrine for Ops costs the USSR 1.06 Influence in Europe** (1.08 at 140M) and buys a
placement back — the card removes all USSR Influence from one non-US-controlled European country,
and the model is picking a country where it holds one. That is a reasonable price for the Ops, not a
blunder, and it is 16 of the 40 plays §16.2 counted at 100M and 13 of 36 at 140M. Those should never
have been in the error column.

Two entries do belong there, for reasons an Influence count does not show:

* **Five Year Plan for Ops** gains 2.75 Influence but loses **two** cards — the card played plus the
  random discard its event forces. The cost is the discard, not the board.
* **Five Year Plan spaced** costs no Influence at all, which is exactly why the Influence metric
  missed it: what it spends is the turn's space attempt, on a card whose Ops the USSR could have
  had. §16.2 scored these 11 plays as *correct* ("US card spaced"), and that was wrong in the other
  direction.

So the honest statement is narrower than §16.2's: the model does decline its own strongest early
event, and some of what it does instead is a real error, but the bucket cannot be scored by side and
mode alone. Each card needs its own adjudication, and two of the three largest components were
mis-scored — one as error when it is sound, one as sound when it is error.

### 17.4 Where this leaves it

* §16's headline stands for De-Stalinization but for a narrower reason than it gave: the critic is
  over-pessimistic about human boards by 7–10 points, significantly, and it is stale — the policy
  improved from 140M and the value function did not follow.
* The on-policy objection is real and was worth raising: at 100M it is the correct account, and any
  conclusion drawn from value gaps alone, in §15 or §16, is unsafe without a rollout behind it.
* Decolonization is calibrated, so this is not a general property of the critic. It is the card the
  model cannot execute that it also cannot price.
* The 160M run reached its budget (`snapshot_final.pt`, 160,038,912 steps) and its own final
  diagnostics say the §12 problem is untouched: `mean_final_turn` 6.4, `frac_reaching_turn9` 0.22,
  `empty_battlegrounds_turn8` 8.1. Longer RL did not rediscover the strategy. It did, however, move
  the realized value of a human De-Stalinization board from ~0 to +5 points, which is the policy
  learning to use a position it still will not create.

## 18. The finished 160M run: stronger agreement, no more strength

### 18.1 What the critic is actually predicting

The question is settled in the engine, not by measurement. Every abrupt ending writes ±20 into
`victory_points`:

* DEFCON-1 and Cuban Missile Crisis suicide — `ops.cpp:200,216`, `victory_points = ±20`, `GAME_OVER`;
* a held scoring card at turn end — `state_machine.cpp:574-576`, the same;
* a 20 VP win — the cap by definition;
* final scoring — the only ending that leaves VP interior.

`Engine::get_terminal_utility` (`engine.cpp:174`) is then `sign(victory_points)`. So the game does
reduce to *final VP, with DEFCON suicide and held scoring normalised to a ±20 result*, and `v_win`
regresses onto the sign of that.

An attempt to measure whether the learned head behaves more like win-probability or like VP margin
**failed to discriminate, and is reported as such**: at roughly 92% of terminals VP sits at ±20, so
the two comparators coincide. corr(v_win, win) = 0.421 against corr(v_win, VP/20) = 0.419 at 160M is
not evidence for either reading. Separating them needs the final-scoring subset alone, which is
~7% of games.

What the same run did establish:

| | warm start | 70M | 160M final |
|---|---|---|---|
| Brier against the actual win | **0.378** | 0.218 | **0.207** |
| corr(v_win, outcome) | 0.119 | 0.357 | 0.421 |
| share of games ending by DEFCON-1 | **70%** | 57% | **40%** |

The BC warm start's value head is *worse than always predicting even* (0.25) — behaviour cloning
fits the policy and leaves the critic actively misleading. RL repairs it. And 40–70% of self-play
games end in mutual destruction, against `mean_final_turn` 6.4.

### 18.2 Why more RL does not rediscover De-Stalinization

§17 showed the policy learning to *use* a human's De-Stalinization board (+5.11 points by 160M)
while still refusing to *create* one. The natural expectation is that the second follows the first
eventually. 18.1 says why it does not, at least not here.

In this model's own game distribution games end around turn 6 by DEFCON-1. Regional scoring that
would pay for spread Influence arrives at turns 8–10 and mostly never arrives at all. The critic is
not being irrational about the card; it is fitted to a world where positional investment is rarely
collected. That closes a loop: short games → positional value seldom realised → critic prices it low
→ policy never invests → games stay short and decided by coups and DEFCON.

This predicts that more of the same RL will not fix it, and the run agrees: 160M steps left
`empty_battlegrounds_turn8` at 8.1 and `frac_reaching_turn9` at 0.22. It also predicts where to
intervene — anything that makes games last (DEFCON discipline, the §12.4 shaping term) should move
the card play as a side effect, and is worth more than teaching the card directly.

### 18.3 Tournament: tied with the previous best, not ahead of it

500 games per pair (250 per side), `--auto-advance`, Bradley-Terry MLE Elo anchored on
HeuristicBot = 1500. Report at
`data/checkpoints/run_v2_20260907_long160M/vs_prior_runs.md`.

| rank | model | Elo | overall win rate |
|---:|---|---:|---:|
| 1 | `dec_turns40` | **1956.6** | 84.4% |
| 2 | **`run_v2_20260907_long160M`** | **1908.4** | 80.0% |
| 3 | `run_v2_blunder_aware_9h` | 1725.1 | 60.0% |
| 4 | `g_w1e1` | 1695.0 | 56.4% |
| 5 | `inj_every1` | 1633.5 | 49.1% |
| 6 | HeuristicBot | 1500.0 | 34.3% |
| 7 | `hum_inj1` | 1497.2 | 34.0% |
| 8 | RandomBot | 937.9 | 1.7% |

**Head to head the two leaders are tied**: `dec_turns40` takes 50.8% of 500 games against the 160M
run. The standard error on a 500-game win rate is 2.2 points, so 50.8% is indistinguishable from
even. The 48-point Elo gap comes from the rest of the matrix — `dec_turns40` beats the weaker field
harder (84.6% against `blunder_aware_9h` where the 160M run manages 73.2%).

Two things worth noting about that comparison. `dec_turns40` is the checkpoint that generated the
synthetic warmup set the 160M run was initialised from, so they are not independent lineages. And
`hum_inj1`, the most human-weighted injection arm, finishes *below HeuristicBot*.

### 18.4 Agreement rose and strength did not follow

Measured on 47 distinct genuinely-unseen games, 23,191 decisions. **The first attempt at this was
wrong and is worth recording**: scoring against a fresh seed-7 split of the deduplicated dataset gave
the warm start 61.45%, because these models were fitted under the split of the *old* 280-id dataset
and 58% of the new held-out set had been trained on. The valid evaluation set is the old split's own
held-out games, minus duplicate-leaked and duplicate-copy ids.

| checkpoint | agreement |
|---|---|
| warm start (BC) | **48.61%** |
| 5M | 38.80% |
| 35M | 38.27% |
| 100M | **32.29%** |
| 140M | 35.47% |
| **160M final** | **37.43%** |
| `dec_turns40` | 33.73% |
| `run_v2_blunder_aware_9h` | 29.86% |
| `inj_every1` | 34.66% |

48.61% reconciles with the 49.33% recorded in §9.13, which validates the measurement. The shape is
the familiar washout — 48.6% down to 32.3% at 100M — followed by a recovery to 37.4%.

**The finished run agrees with humans more than any prior checkpoint (37.43% against
`dec_turns40`'s 33.73%) and is not stronger than it.** That is the cleanest statement yet of the
problem §9–§11 kept circling: human agreement is not a proxy for strength on this axis. 3.7 points
of extra agreement bought nothing measurable in Elo, which is consistent with §11's finding that the
behaviours being imitated are worth a few points at most and with §17's finding that the largest
mispricing is in the critic rather than the policy.

*(Minor correction to §14.4: of the four held-out games named as leaking into training, two — 216
and 217 — are empty downloads carrying no decisions. The real leak is two games.)*
