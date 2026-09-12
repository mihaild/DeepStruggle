# Measurement

What the numbers in [`experiments.md`](experiments.md) can bear: how each measure is defined,
what a tournament result is reproducible to, how much of a run's rating is just where it
stopped, and the instruments that reported confident numbers while measuring nothing.

`experiments.md` records what was learned about training, design and data. This file records
whether it could have been learned -- the two are separated because they are read at different
moments, and because a result and the doubt about it should not have to be read together.

**Section numbers are the ones these entries were first written under** in `experiments.md`, so
every reference that names one still resolves. A gap is a section that stayed behind.

> **Distrust a measurement before trusting a result.** Seven separate diagnostics in this repo
> reported confident numbers that were wrong. Every one of them looked plausible. When an
> experiment produces a surprising result, the cheapest first hypothesis is that the instrument
> is broken -- check that before building on the finding.

---

## 1. Measurement bugs found (read this before trusting any older number)

Four defects in the evaluation and diagnostic code, three of them the same defect in three
different files. All are fixed; all invalidate numbers logged before their fix.

Three more of the same character were found later and are below under the numbers of the
experiments that turned them up: **§8.5**, where the forced-win metric both over- and under-counts,
and **§23.1**, where a model was fed the wrong observation layout and misread it without complaint,
a tournament nearly rated one checkpoint twice, and a snapshot sort returned the wrong four
snapshots for one arm while returning the right four for the other.

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


### 23.1 Three measurement faults found while running this, all silent

Each would have produced a plausible wrong number rather than an error, which is §1's pattern.

**A model reads fixed slices, so a wrong-width observation is misread, not rejected.** Arm E's
in-training evaluations fed the v2.1 network legacy 4,293-wide observations: the board came out
right by luck, the card block read 1,430 floats spanning cards plus globals, and the global slice
landed in the legacy history region and was all zeros. Reported 2.0% against `HeuristicBot` and a
mean final turn of 1.37 while the run's own rollouts averaged turn 6.7 — the contradiction is what
exposed it. The same 35M snapshot, given its own layout, beats `HeuristicBot` 81.3%. `NeuralAgent`
now derives the layout from the model and `_assert_width` raises on a mismatch. **Every arm E
evaluation logged before that fix is void.**

**`snapshot_final.pt` is not a distinct model.** It is weight-identical to the last step snapshot
(97/97 tensors; only the file hash differs). Listing both in a tournament enters one player twice
and lets it accumulate a rating partly against its own duplicate. Checked before the 240M run and
excluded; the 80M comparison had used explicit step snapshots and was unaffected.

**Sorting snapshot paths numerically does not sort them numerically.** `sort -t_ -k2 -n` over full
paths reads field 2 of a path that is itself full of underscores — `E` in `arm_E_cont_80to240` — so
every line compares equal and `ls`'s lexical order survives. Lexical equals numeric only while every
step count has the same digit count: arm D's are all 9 digits and came out correct, while arm E's
span 85000192 to 240058368, so the 8-digit names sorted last and `tail -4` selected 85M, 90M and 95M
as that arm's "late" snapshots. Caught before the tournament ran, by printing the selection. **A
check that passes on one arm by arithmetic accident is not a check** — the selection is now made by
extracting the step count and sorting on it, and verified to return matched lists for both arms.

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
- **`--auto-advance` is outcome-neutral but not free, and it redefines a training step.**
  Re-measured on the current engine. Outcomes are identical -- 256 of 256 games end on the same
  turn, action round, VP, DEFCON and result -- and a game takes **4.2% fewer batched steps**,
  because forced decisions are resolved inside the engine instead of being handed to the policy.

  The speed is the surprise. `auto_advance_step` runs after *every* action on *every* env and
  scans for its auto-resolvable cases even when there are none, and that costs more than the
  round-trips it saves when nothing else is in the loop: engine-only, 512 envs, order
  alternated, auto-advance is **6.3% slower in wall time** despite the 4.2% fewer steps. Put a
  real v2.3 forward pass in the loop and it turns around, because the network is most of the
  cost: **+2.5% per step, 4.2% fewer steps, net +1.8% wall time for the same amount of game**.
  So it is worth having where a network drives the loop, and a small loss where one does not.

  **The catch for training.** `steps_collected = buffer_size * num_envs`, so a "step" is one
  decision the policy was *asked about*, and auto-advance removes the forced ones. An 80M-step
  budget with it on therefore covers ~4.2% more game than the same budget with it off -- about
  +3 Elo by the budget curve (§20), which is inside the ±16-24 Elo noise floor but is a real
  shift against every arm measured so far. It belongs in `engine_config` and a fresh baseline,
  not switched on mid-programme for 1.8%.

  One more caveat on "outcome-neutral": that is per decision stream, verified with a
  deterministic policy. A *sampling* policy consumes RNG at every decision it is asked about, so
  removing the forced ones shifts every later draw. Games are then statistically equivalent, not
  bit-identical -- fine for a tournament, not a basis for reproducing a specific run.

  Current state: training does **not** use it; `tools/tournament.py` defaults it off; five
  `ai/eval/` probes (dominance_cost, battleground_value, critic_calibration,
  round_counterfactual, input_ablation) pass `auto_advance=True` on the batch runner.
- **The Python chance-drain loop is not worth moving into C++.** A *chance node* is a point
  where the engine has stopped for a die nobody chooses: `ctx().decision_player` is `NONE` and
  `decision_type` is `ROLL_DIE`. Draining is stepping it with `MicroAction(ROLL_DIE, 0, 0, 0)`
  until it is gone; the zero means "roll it yourself", since `execute_coup` and friends do
  `forced > 0 ? forced : Prng::roll_d6(state.rng_state)` (a non-zero payload is how the replay
  converter forces a recorded die). If nobody drains, the next agent is handed the roll: the
  single-state loop falls back to `phasing_player` when `decision_player` is `NONE`, which had a
  policy network picking its own dice, and made that path disagree with the batched one by more
  than 25 points.

  `VectorizedBatchRunner::step_flat_all` already drains inside C++ *whether or not*
  `auto_advance` is set, so the batched path never crosses the binding boundary twice. Only the
  single-state paths drain in Python (`tournament_evaluator`, `position_diagnostics`,
  `blunders`), and moving that into C++ **loses 6.7% of wall time**, measured over 120 real
  games with the order alternated. The reason is the same one that makes `auto_advance` a
  per-step cost: `auto_advance_step` scans after *every* action, while chance nodes are only
  **10.1 per game against 249 real decisions** — 3.9% of stops (WAR_EVENT 41%, TURN_CLEANUP 36%,
  OLYMPIC_GAMES 15%, TRAP_ESCAPE 6%, SUMMIT 3%). The Python check is cheaper than the scan that
  would replace it.

  A first attempt measured this as a 16% *win*, because it drove the engine with
  `get_legal_action_indices` — a different index space that `step_flat` rejects, so the loop
  spun on an invalid action for 4,000 iterations and never played a game. Both modes stalled
  identically, and the drain check ran 4,000 times against nothing. Drive the engine with
  `ActionMask.generate_flat_mask` and check that `step_flat` returned true.

- **Measure game length in plies, not turns.** A ply is one player's single opportunity to act
  — one headline, or one action round for one side — numbered continuously from the start of the
  game, so ply 1 is the USSR's turn-1 headline and **154 is a game that played all ten turns
  out** (`ai/game_length.py`). The turn counter is wrong for this in two ways. It is too coarse:
  a game abandoned at turn 7 AR1 and one that ran to turn 7 AR7 are both "turn 7", which is 14
  plies apart. And it has an artefact at the top of its range — `finish_end_turn` increments the
  turn and only *then* tests `turn <= 10` before calling `execute_final_scoring`, so a completed
  game terminates holding turn **11**, while a human replay log numbers that same game turn 10.
  Comparing a model's mean turn against a corpus mean turn therefore compares two different
  scales, and the error runs in the flattering direction. Logged as `mean_ply` / `median_ply`,
  reported as `avg_ply` by both tournament paths.

- **Beware `harvest()` in analysis scripts.** It calls `retire_stale()`, which drops older
  generations by design, so positions must be taken out of the pool after each round or they
  are lost. This silently reduced a 1,000-position sample to 91.

---

## 1.4 A Wargames ending in turn 10 was reported as final scoring

`classify_game_ending_reason` tested `state.turn < 10` for Wargames and `state.turn >= 10` for
final scoring. Real final scoring terminates at turn **11** (above), so the `>= 10` branch was
only ever reachable by a game that ended *inside* turn 10 — which, with `abs(VP) < 20` and
`GAME_OVER`, is a Wargames. Every turn-10 Wargames was therefore counted as final scoring in
`ending_frac_*` and in tournament reports. 3 of the 119 finished human games end exactly that
way. The bound is now `turn <= 10`; `tests/engine_logic/test_game_invariants.py` had encoded the
same misconception and asserted a turn-10 state was final scoring, so it was corrected against
40 driven heuristic games, all of which terminate final scoring at turn 11 / AR 0 and never at
turn 10.

The effect on the arms is nil in practice — they play Wargames essentially never (0-1 games in
1,000) — but it mattered for the human baseline, where Wargames is **18.5%** of finished games.

---

## 1.5 Human game length, and what the corpus can and cannot say about it

The corpus is 274 distinct logs: **119** reach a terminal state, **146** are fragments whose
recording stops, and 9 are empty. Only the 119 are a game length. Two traps:

- The fragments' mean *last* turn is 6.77, which happens to land right on top of the arms'
  6.5-7.2. Averaging all 265 usable logs gives 7.95. Neither is a game length; both flatter the
  models by measuring when ts-replayer users stopped recording.
- `game_ended` must come from the converter's terminal test, not from the log's own fields. The
  log's `defcon` on the last entry holds the value *before* the ending resolved, so classifying
  by it found 2 DEFCON-1 endings where there are 21.

The 119, classified from the terminal position the converter now records (`final_turn`,
`final_action_round`, `final_defcon`, `final_cmc_suicide`, `final_defcon_provoked`):

| | mean ply | share |
|:---|---:|---:|
| final scoring | 154.0 | 56.3% |
| 20 VP / held scoring | 129.9 | 23.5% |
| wargames | 121.7 | 18.5% |
| DEFCON 1 (own) | 153.0 | 0.8% |
| DEFCON 1 (provoked) | 137.0 | 0.8% |
| **all** | **142.2** | 92.3% of a full game |

67.2% of finished human games play all ten turns out. 20 of the 119 contain an AR8, which the
ply scheme counts at its nominal 16 and so under-counts by 2 apiece.

### 1.4.1 The position-diagnostics probe is reporting nonsense

`diag/mean_final_turn` reads 1-2 across the whole of arm H2 while the run's actual mean terminal
turn is ~6.8; `diag/frac_reaching_turn9` reads 0.0 throughout, and `diag/empty_battlegrounds_turn8`
reads 0.0 -- which would mean every battleground is contested by turn 8. Walking 60 live self-play
games from the same checkpoint gives **6.15 of 29 battlegrounds still completely untouched** at
turn 8, with Saudi Arabia empty in 33 of the 33 games that got there, India in 30, Algeria in 27.

**Cause: the fourth instance of the wrong-layout bug.** `profile_self_play_batched` built its
`TsVectorizedEnv` without passing a layout, and the constructor defaults to `legacy`. A v2.3
model handed legacy observations does not raise -- the width assertions live elsewhere -- it just
reads the wrong floats and plays at random. The games were dying in turn 1, so nothing reached
turn 8 and `empty_battlegrounds_turn8` averaged over an empty set. `measure_decisive_batched` had
the identical line, so the `decisive/` group was wrong the same way.

Both now derive the layout from the model's own width via `bindings.ts_env.layout_for_model`,
which raises rather than guessing, and both take the run's `obs_flags` so a flagged run is not
probed with flags off. After the fix, on the same H2 @160M checkpoint: `mean_final_turn` 7.19
(was 1-2, actual ~7.1) and `empty_battlegrounds_turn8` **6.258** against the 6.15 measured
independently above -- two separate code paths agreeing is what says it is fixed.

Every `positions/` and `decisive/` number recorded before this is void.
`tests/training/test_probe_observation_layout.py` pins the class: it captures the layout each
probe constructs its env with, and fails if it is not the model's.

### 1.5.3 Game shape varies between seeds by as much as it varies between arms

Arm H2 is arm H's configuration on a second seed (20260921, v2.3, corrected engine). At an equal
80M budget the two are the same strength -- pooled over four late snapshots a side and all 16
pairings, 3,200 games, **50.4%, +3 Elo**. A clean replication.

Their *games* are not the same shape:

| self-play, 1,000 games | mean ply | 20 VP | final scoring | DEFCON 1 | wargames |
|:---|---:|---:|---:|---:|---:|
| H @80M | 107.0 | 40.9% | 15.7% | 43.2% | 0.2% |
| H2 @80M | 98.6 | 47.4% | 9.6% | 42.8% | 0.2% |
| H2 @160M | 100.8 | 50.4% | 10.6% | 37.7% | 1.3% |
| (arm D, legacy, 80M) | 99.0 | 46.1% | 11.0% | 42.9% | 0.0% |
| (arm E, v2.1, 80M) | 100.1 | 46.0% | 9.6% | 44.2% | 0.2% |

Two runs of identical configuration and indistinguishable strength differ by **8.4 plies** and by
6 points of final-scoring share -- more than H differed from the pre-fix arms D and E. H2 lands
squarely on top of D and E, not on H.

**This retires the claim that the corrected engine lengthens games.** That was read off arm H
alone (§25 draft, and reported as "the first arm to move final scoring at all"), and it does not
replicate: H is the outlier of the three v2.3-or-earlier runs at 80M, not the start of a trend.
Game length and ending mix are seed-noisy at this budget and cannot carry a conclusion from a
single run, exactly as Elo cannot (§20.7). Nothing here contradicts the starred-card fix being
*correct* -- it is a rules bug either way -- only the evidence offered that it changed play.

What does survive is a budget effect, consistent across two independent comparisons:

- H2 at 160M vs H at 80M: **61.0%, +78 Elo**
- H2 at 160M vs H2 at 80M: **60.9%, +77 Elo**
- against the anchor: 83.0% (H @80M), 84.2% (H2 @80M), **88.0% (H2 @160M)**

and, at 160M, the first movement on the two human gaps that is visible in both the binned
training log and self-play: DEFCON 1 falls 42.8% -> 37.7% and wargames rises 0.2% -> 1.3%. Set
against ITS's 11.7% and 14.9% those are still a factor of three and a factor of eleven away.

A caution on reading the per-iteration monitor: it reports a mean over the last 40 logged
iterations, which overlaps heavily between consecutive reports and made H2's USSR win rate look
like a monotone late-run climb to 66%. Binned by 20M it oscillates 48.5-58.0% for the whole run
with no drift. Bin before believing a trend in it.

### 1.4.2 The per-region scoring scalars are not architecturally connected to their countries

`global_features[64..69]` carry the live net VP differential per region, from
`Scoring::evaluate_region`, so presence, domination, control, battlegrounds and superpower
adjacency are all folded in. Measured on Europe: empty 0.000, domination **+0.500**, control
**+1.000**. Control is forced to ±20 because its raw `net_delta` is *8* against domination's
*10* -- Europe's `control_vp` is 0, so without the special case the observation would rank a won
position below a merely dominated one.

Each country also carries a 6-way region one-hot in `board_features[10..15]`, plus Western
Europe / Eastern Europe / South-East Asia flags at 16..18. So the country-to-region *mapping* is
an explicit input; nothing has to learn it from data.

**What is missing is the binding between the two.** In `ColdWarNetV2.extract_features` the board
branch runs two GraphConv layers over the 84 countries and then **mean- and max-pools across all
of them** into 128 floats before the global block is ever seen. The region scalars arrive
through a separate `global_proj`, and the two meet only in the fused trunk. The policy head is
then `Linear(hidden -> 256) -> Linear(256 -> 212)` off that fused vector: there is no
per-country output path at all.

So there is no architectural route from "the Europe scalar rose" to "because of France". The
association has to be discovered statistically, through a trunk that has already discarded which
country is which. That is a plausible reason the critic values regional position poorly, and it
is not fixed by adding features -- the information is already there and correctly scaled.

Compounding it: arm H2 reaches Europe control in **0.0%** of games, so the top of that ramp is
essentially unvisited and whatever the critic predicts at 1.0 is extrapolation from the 0.5
neighbourhood. The feature makes control *learnable*; visitation is what would make it learned.

### 1.5.1 The ts-replayer figure is biased long; the ITS results database is the better baseline

`/workspace/data/itsc-games` is a scrape of the ITS Junta results table at twilight-struggle.com
— **47,928** digital (Playdek, Deluxe) games, 44,136 of them with a rules ending, against
ts-replayer's 119. It is results-only: one row per game with `endTurn` and `endMode` and no
moves, so it can give length and ending mix and **cannot** replace ts-replayer as a training
corpus.

Its `endTurn` uses the same convention the engine does — 12,787 of its 12,792 Final Scoring games
are recorded at turn 11 — which is independent confirmation of the turn-11 artefact above, from a
source that has never seen this code.

It disagrees with ts-replayer sharply, and in the direction that says ts-replayer is the biased
one:

| | ts-replayer (119) | ITS (44,136) |
|:---|---:|---:|
| went the distance | 67.2% | **29.7%** |
| mean end turn (engine scale) | 9.99 | **8.33** |
| mean ply | 142.2 | **~119** (bounded [112, 123]) |
| final scoring | 56.3% | 29.0% |
| 20 VP | 23.5% | 43.1% |
| wargames | 18.5% | 14.9% |
| DEFCON 1 | 1.7% | **11.7%** |
| held scoring | (folded into 20 VP) | 1.4% |

ts-replayer's 119 are the subset of 274 logs whose *recording* completed, and that filter is not
independent of how the game ended: a game that blows up at turn 6 leaves a log that stops
mid-turn and lands in the 146-game fragment pile, while a game that goes the distance gets
recorded to the end. The DEFCON-1 row is the tell — 1.7% against 11.7% is not sampling noise at
these sizes. Use ITS for length and ending-mix baselines; use ts-replayer where moves are needed.

The ITS ply figure is an estimate, not a measurement: a turn pins the ply only to a 14- or
16-wide interval, so the point value applies ts-replayer's mean *within-turn* offset per ending
kind (20 VP lands late in a turn, 13.7 of 16; Wargames early, 5.5). Report it with its interval.

### 1.5.2 Side balance and per-side asymmetry in the ITS corpus

Human play is almost exactly balanced: **49.90% USSR** over 43,685 decided games (451 ties,
1.02%). Every arm so far has wandered well outside that — H2 has ranged 41.9% to 54.8% within a
single run — so the pooled figure is a target the runs are not obviously converging on.

The two sides do not win the *same way*, and a pooled ending mix hides it:

| | US wins | USSR wins |
|:---|---:|---:|
| n | 21,888 | 21,797 |
| mean end turn | 8.62 | 8.00 |
| mean ply | ~124 | ~115 |
| went the distance | 33.4% | 25.2% |
| 20 VP | 36.2% | **50.8%** |
| final scoring | **32.1%** | 25.0% |
| wargames | 14.5% | 14.9% |
| DEFCON 1 | **15.6%** | 8.1% |
| held scoring | 1.6% | 1.1% |

The USSR takes half its wins on the VP track and wins faster; the US wins later, more often by
scoring the board out. Read the other way, within each ending: the USSR wins 58.3% of 20 VP
games, while the US wins **65.9%** of games that end at DEFCON 1 — that is, the USSR is far more
often the side that walks into the war it loses to. Ties are almost entirely final scoring
(67.4%) and Wargames (32.4%), which is what a 6 VP Wargames swing landing on zero looks like.

This is why the trainer now splits length and ending mix by winning side (`game_won_us/`,
`game_won_ussr/`): a run can hit the pooled human numbers while getting both halves wrong.

### The gap this exposes

Self-play at temperature 0.1, 1,000 games each, measured through the batched match runner:

| | mean ply | of 154 | 20 VP | final scoring | DEFCON-1 | wargames |
|:---|---:|---:|---:|---:|---:|---:|
| RandomBot | 40.3 | 26.1% | 47.4% | 0.6% | 50.5% | 1.5% |
| arm D (legacy, 80M) | 99.0 | 64.3% | 46.1% | 11.0% | 42.9% | 0.0% |
| arm E (v2.1, 80M) | 100.1 | 65.0% | 46.0% | 9.6% | 44.2% | 0.2% |
| arm H (v2.3, corrected engine, 80M) | 106.7 | 69.3% | 38.0% | 14.4% | 47.5% | 0.1% |
| arm H2 (same config, second seed, 80M) | 98.6 | 64.0% | 47.4% | 9.6% | 42.8% | 0.2% |
| HeuristicBot | 114.6 | 74.4% | 74.6% | 25.4% | 0.0% | 0.0% |
| **humans (ITS, 44,136)** | **~119** | **77.4%** | **43.1%** | **29.0%** | **11.7%** | **14.9%** |
| humans (ts-replayer, 119) | 142.2 | 92.3% | 23.5% | 56.3% | 1.7% | 18.5% |

H and H2 are the same configuration on two seeds and the same strength to +/-3 Elo, yet differ by 8.4 plies (§1.5.3): read the pair, never H alone.

Against the ITS baseline the length gap is modest -- the arms at 99-107 against ~119, and
HeuristicBot at 114.6 is essentially at human length -- and the arms match humans on 20 VP
endings almost exactly (38-46% against 43.1%). The deficit is in *how* games end, in two
specific ways:

- **DEFCON 1 is four times too common**: 43-48% of arm games against 11.7% of human ones.
  HeuristicBot, which carries an explicit instant-loss safety layer, never does it at all, and
  that alone is most of its length advantage over the arms.
- **Wargames is never played**: 0-0.2% of arm games against 14.9% of human ones. It is a real
  human resource for closing out a won position at DEFCON 2, and no arm has found it.

Held scoring is folded into the 20 VP column for the arms and for ts-replayer: our classifier
only separates it on the training path, where ts_env supplies the flag.

Arms F, F2 and G cannot be re-measured: they are observation layout v2.2, which is retired, so
their checkpoints cannot be loaded. Their turn-only training figures remain the only record.

---

## 7. What a tournament number is reproducible to

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

## 20. Run-to-run variance, and how much of it is just where you stopped

§18 compared single final checkpoints and read differences off them. §19's encoding plan assumed a
"slight positive effect" would be visible at one run per arm. Both rested on a variance estimate
that had never been made, and when it was made it was much larger than assumed -- and then, on a
closer look, largely removable.

### 20.1 Seeding had to be added before variance could be measured at all

The environment seed was the literal `12345` and torch was left unseeded, so two runs of one
configuration saw the same deals and the same dice and differed only in initialisation. A "replicate"
under that arrangement measures a fraction of the thing. `--seed` now sets both together (`4f2b23b`),
or neither when omitted, which keeps existing behaviour.

`--resume` was added alongside (`4f2b23b`), because snapshots are bare `state_dict`s and a
`--warmup-checkpoint` restart silently drops the optimiser moments, the reference policy and the
step counter. The tests assert the distinction rather than assume it: ten steps must equal five,
save, resume, five more, **and** a weights-only restart must *not* match. If that negative control
ever passes, the resume file is pointless and says so.

### 20.2 The first variance number was wrong, and it was the one everything rested on

Arm A against the 160M run's own 80M snapshot -- same configuration, different seed -- came out
**5 Elo apart, 50.2%/49.8% on 1,000 games**. That was reported as the noise floor, with the
transfer to other configurations flagged as an assumption.

It does not transfer. On the synth-only configuration, C against C2 is **59.6 Elo apart, 63.9%
head to head**. A third seed gave SD 64.2 over three final snapshots. The assumption was wrong by
an order of magnitude, and several earlier claims went with it -- "the synthetic warm start is
worth +199 Elo" and "the human BC layer costs 128 Elo" were both single-draw differences smaller
than the spread they were measured against.

### 20.3 Most of that spread is *when you stopped*, not *which seed you drew*

Rating the last four snapshots (65M, 70M, 75M, 80M) of every arm rather than the final one:

| arm | 65M | 70M | 75M | 80M | mean | within-run SD |
|---|---:|---:|---:|---:|---:|---:|
| C s1 | 1728.4 | 1773.4 | 1764.3 | **1847.7** | 1778.4 | 50.1 |
| C2 s2 | 1729.6 | 1731.9 | 1736.5 | 1760.0 | 1739.5 | 14.0 |
| C3 s3 | 1754.2 | 1778.5 | 1732.2 | 1736.8 | 1750.4 | 21.0 |
| B s1 | 1680.7 | 1634.7 | 1640.7 | 1642.8 | 1649.7 | 20.9 |
| B2 s4 | 1689.2 | 1632.6 | 1741.1 | 1722.9 | 1696.4 | 47.7 |

Mean within-run oscillation is **30.7 Elo**, and C s1 spans 1728–1848 with its *final* snapshot at
the peak -- reporting it as 1848 was reading the top of a wobble.

| how a run's number is taken | synth-only SD | cold-start SD |
|---|---:|---:|
| final snapshot only | 58.5 | 56.6 |
| **mean of the last four** | **20.1** | **33.0** |

**Averaging four snapshots cuts the between-run SD by 2.9x on the synth-only configuration, for no
extra compute.** Cold start gains less because B2 itself oscillates 47.7, and two seeds cannot
average that away.

This is the cheapest variance reduction available and it should be standard: *rate the last four
snapshots, report the mean*. Every single-final-snapshot comparison earlier in this document is
inflated by roughly 30 Elo of stopping-point noise.

### 20.4 What survives

| configuration | mean of per-run means | seeds |
|---|---:|---:|
| synth-only warm start | **1756.1** | 3 |
| cold start | **1673.1** | 2 |
| `dec_turns40` | **1959.3** | 1 |

The warm-start effect is **+83.0 Elo** against a between-run SD of 20–33, roughly 2.5–3σ. Real,
and less than half the +179 claimed from single finals.

The ordering is not clean at the level of individual runs: the weakest synth-only seed (C3) finishes
*below* the stronger cold start (B2) and splits 50.9%/48.9% with it. Configuration means separate;
individual runs overlap.

### 20.5 Consequences for how experiments get run here

* **Rate four snapshots, not one.** Free, and nearly triples effective precision.
* **Two or three seeds per arm** now suffices for effects above ~40 Elo. The encoding experiment
  §19 planned is viable on that basis; at the 60 Elo single-final floor it was not.
* **The oscillation has a likely cause worth removing at the source.** The learning rate is
  constant for the entire run -- there is no decay schedule -- which is exactly what leaves a policy
  wandering at the end rather than settling. Linear or cosine decay, or a weight EMA evaluated
  instead of the live policy, would remove the 30.7 Elo oscillation rather than averaging over it.
  Neither has been tried.
* **`dec_turns40` is still ~200 Elo clear of everything trained since**, which is the subject of
  §21.

---

### 20.6 Continuation variance, measured directly — the seed is worth ~15 Elo, and a leg's *gain* twice that

§20.3 measured the spread between *runs of a configuration*. It never measured the spread between
*continuations of one run*, which is what every within-lineage number in §23 is, and those were
being quoted without an error bar.

**Setup.** Two continuations of arm D from the identical 160M resume state — same weights, same
optimiser moments, same reference policy — differing only in seed (20260916 against 20260918), each
to 240M. Four late snapshots per arm, 400 games a side. There is no better arm here; the output is a
spread.

| | 225/230/235/240M | mean | SD | gain over the 160M start |
|:---|:---|---:|---:|---:|
| seed 20260916 | 1880, 1893, 1887, 1886 | 1886.2 | 5.4 | **+86.9** (62.25%) |
| seed 20260918 | 1897, 1894, 1828, 1858 | 1869.1 | 32.7 | **+54.7** (57.81%) |

Difference in mean **17.1 Elo**; pooled head-to-head over all 16 pairings, 12,800 games, **51.80%
±0.87**, implying **12.5 Elo**. Consistent with §20.3's ~20 Elo between-run SD, now confirmed for
the continuation case specifically.

**The consequential number is the second column, not the first.** The same 80M leg, measured the
same way, was worth +86.9 Elo on one seed and +54.7 on the other. A leg's gain therefore carries
roughly **±16 Elo**, which is larger than most of the differences §23 was reading as a trend.

**Three practices follow.**

* **Quote a leg's gain with ±16, or do not quote it as a trend.** §23's +74.1 then +52.7 for arm D
  is not evidence of compressing returns; the two are indistinguishable.
* **Never compare Elo across tournaments.** The identical comparison — D's late four against its own
  160M final, the same eight files — read 60.50% in the 240M pool and 62.25% here, 1.4σ apart on
  3,200 games each from sampling alone. Direct head-to-heads carry about ±12 Elo before any seed
  effect. Within one tournament the numbers are comparable; across two they are not.
* **A single cell is not a comparison, twice over.** The final snapshots of the two replicates meet
  at 59.1%, which would put seed variance at ~64 Elo; pooling the sixteen pairings gives 51.80% and
  ~13. The same trap produced a spurious "E beats D 54.9%" at 240M and a spurious "+52 Elo for E's
  last leg" at 320M.

**And snapshot averaging damps the stopping point without taming it.** Within-run SD across the last
four was 5.4 on one seed and 32.7 on the other — a sixfold difference between two runs of one
configuration, with no visible cause. Four snapshots is the practice, not a guarantee.

---

### 20.7 A single snapshot cannot measure an effect smaller than a run's oscillation

20.3 established that rating four late snapshots instead of one cuts between-run SD from 58.5 to
20.1, and it was adopted as practice. 24.1 is what happens when the practice is skipped.

The `staged_cards` flag was first measured with one snapshot per budget: 800-game cells at about
+/-24 Elo, against a mean within-run oscillation of 30.7. It read **+52.8 Elo at 80M and +4.7 at
160M**, and the 80M figure stood as the headline until the pooled version replaced it with **+29.7
and -11.4** -- a different sign at one budget and half the size at the other.

The arm that looked better was also the noisier one: G's within-run SD was 27.0 against F's 7.5, at
the same seed with one flag between them, which is unexplained and was invisible from a single
checkpoint. Picking its final snapshot flattered it.

**So the error bar on a single-cell comparison is the run's oscillation, not the binomial SE of the
games played.** 12,800 games across sixteen pairings gives +/-0.87pp; 800 games in one cell gives
+/-3.4pp, and the snapshot choice on top of that is worth another 30. Two arms compared one
snapshot each cannot resolve anything below roughly 50 Elo, which is larger than most effects worth
arguing about.

---

## 21. What an 80M arm actually spends its budget learning

Measured on `p1_scalar_nofilter`, binned by snapshot (eval rows carry `iteration`, not
`total_steps`; 80,019,456 steps at iteration 1221 is 65,536 steps/iteration):

| | 5M | 30M | 60M | 80M | 90% of the run's movement by |
|:---|---:|---:|---:|---:|---:|
| anchor win rate | 32% | 85% | 84% | 84% | **30M** |
| empty battlegrounds, turn 8 | 11.91 | 9.49 | 7.62 | 7.60 | **60M** |
| uncontrolled battlegrounds, turn 8 | 20.88 | 17.95 | 16.47 | 16.33 | **60M** |
| DEFCON-suicide blunder rate | 9.8% | 6.2% | 7.3% | **10.0%** | **flat** |

Two things follow.

**Most of the anchor number is bought in the first 30M**, about 38% of the budget, and the
remaining 50M moves it by a few points inside the noise band. The strategic diagnostics are the
opposite: they improve steadily and are still moving at 60M. So the anchor metric saturates long
before the behaviour does — another reason (§20.7) not to rate an arm by it.

**The DEFCON-suicide rate does not improve over a whole run.** It starts at 9.8%, wanders between
4.6% and 10.7%, and ends at 10.0%. This is the mistake the project goal names first, and 80M
steps of self-play do not touch it. That is a finding about the *reward*, not about the budget: a
warm start cannot save re-learning something nothing learns in the first place.

> **The taxonomy changed underneath this metric.** `blunder_defcon_suicide_with_alternative`
> was redefined (battleground-conditioned influence, Tear Down This Wall added, Nuclear Subs made
> one-sided, Ortega and Star Wars added, We Will Bury You / How I Learned / Junta removed). The
> table above is the **old** definition. Runs from the second P1 leg onward use the new one, so
> the series is **not continuous across that boundary** — a step change there is the definition
> moving, not the policy. Re-measure before comparing across it.

### 21.1 The critic does not see a provoked DEFCON-1 coming, at any node

Traced through the `h2_480M_provoked_*` replays by replaying their recorded flat actions through
a fresh engine on the replay's own seed and reading `v_win` from the **losing** side's
perspective. (Perspective checked: the two sides sum to +0.05 at a sampled node, so these are
the loser's own values, not a sign error.)

Every value is read on the state *before* that node's action is applied, so the cost of a single
choice is the difference between the node where it is made and the node after it. Quoting the
value at a node against the value at the *previous* node measures the opponent's intervening
moves as well, and is not attributable to the choice.

In 7107, 7115 and 7118 the card was played **Ops first, event later**, so `PlayMode: OPS` on an
opponent card is the point of no return. In 7119 Grain Sales was **unspaceable**, so the card
selection is. Both were tested:

| game | v at the card node | delta from selecting the fatal card | delta from choosing OPS |
|:---|---:|---:|---:|
| 7107 Star Wars | −0.770 | **+0.004** | **+0.006** |
| 7115 Lone Gunman | −0.052 | **−0.003** | **−0.004** |
| 7118 Grain Sales | −0.691 | **−0.008** | **+0.007** |
| 7119 Grain Sales (unspaceable) | −0.711 | **−0.007** | **−0.014** |

**It never reacts.** Selecting the fatal card is worth at most 0.008 to the critic and choosing
to spend it for Operations at most 0.014; two of the eight deltas are positive. At the last
decision before DEFCON 1 it still reports −0.40 to −0.75 rather than anything near −1, and in
7115 the losing side sits at −0.05 -- essentially even -- four micro-actions from death.

(7122 is excluded: Missile Envy is a different case. It is neutral, usually safe to play, and
the information that would make it unsafe -- the opponent's highest-Ops card -- is hidden, so
there is no comparable "point of no return" to test.)

Two consequences.

**This is not only a credit-assignment problem.** The critic cannot represent the conjunction, so
sharpening the policy's credit will not by itself teach it -- but that is the argument *for*
windowing the provoked case rather than against it. Inside a blunder window the advantage is
`-1 - v_t`, which never consults the critic, so the window is precisely the mechanism that works
when the critic is blind. Outside one, the -1 has to flow back through a value function that
prices the position at -0.7 and rising, and is absorbed rather than attributed.

**And it raises the value of the auxiliary DEFCON-risk head**, a direct supervised signal for
what `v_win` demonstrably does not encode. Its label is the same `defcon_blunder` flag that
excludes provoked endings, so the one-line label fix is a prerequisite for either.

### 21.2 Windowing a provoked DEFCON-1 moves exactly the class it targets

`--window-provoked-defcon` credits a provoked DEFCON-1 to the player who played the card, using
the existing turn-scoped blunder window, instead of letting the -1 propagate back as an ordinary
loss. One seed, 80M steps, against `p1_scalar_nofilter` at 80M -- same recipe, filtering off in
both, the window the only difference.

| ending | control | window | delta |
|:---|---:|---:|---:|
| DEFCON-1 total | 49.4% | **27.0%** | −22.4 |
| — provoked | 36.2% | **13.7%** | **−22.5** |
| — self-inflicted | 13.1% | 13.3% | **+0.1** |
| final scoring | 6.9% | **16.7%** | +9.8 |
| 20 VP | 39.6% | 53.9% | +14.3 |

**The self-inflicted share does not move.** That is the result: the intervention targets provoked
endings alone, and provoked endings alone changed, by 62% of their own value, while the
neighbouring class in the same metric family stayed put to a tenth of a point. It is as close to
a placebo control as a training change gets here, and it rules out the reading that the arm just
made every DEFCON-1 rarer by playing more timidly.

Games also got longer -- mean ply 93.4 to 103.1, final scoring 6.9% to 16.7% -- moving toward the
human distribution, where about 30% of games go the distance against this control's 7%.

**The behaviour replicates on a second seed; the strength cost does not, yet.**

| arm | provoked | self | final scoring | USSR | anchor |
|:---|---:|---:|---:|---:|---:|
| control | 36.2% | 13.1% | 6.9% | 47.0% | 80.6% |
| window ...921 | 13.7% | 13.3% | 16.7% | 63.7% | 71.4% |
| window ...922 | **10.7%** | 12.1% | 16.7% | 58.3% | **81.8%** |

Both seeds cut provoked endings by about two thirds and leave the self-inflicted share alone, so
the behavioural result is solid. But the two seeds sit **10.4 anchor points apart**, straddling
the control -- the same pattern the filtering arm produced, and the reason the anchor cannot
settle anything here.

**Both seeds lose, and by more than the first one suggested.** Pooled over sixteen snapshot
pairings each:

| | vs control | Elo |
|:---|---:|---:|
| window ...921 | 1,331/3,200 = 41.6% [39.9, 43.3] | **−59** |
| window ...922 | 1,250/3,200 = 39.1% [37.4, 40.8] | **−77** |
| pooled | 2,581/6,400 = **40.3%** [39.1, 41.5] | **−68** |

So the strength cost is real and two-seed confirmed at about **−68 Elo**.

**And the anchor was wrong about both seeds, in opposite directions.** Seed ...921 read 71.4%
against the control's 80.6% and is 59 Elo weaker; seed ...922 read **81.8%**, slightly *above*
the control, and is **77 Elo weaker** -- the largest single divergence this project has recorded
between the anchor and a pooled head-to-head. An earlier draft of this section argued the anchor
corroborated the 59 Elo to within 8; that was one seed, and it was luck. Take nothing from the
anchor win rate that a head-to-head has not confirmed.

The USSR share rose in both (63.7% and 58.3% against 47.0%), which is a move away from the human
49.9% and is measured over the whole run rather than 500 eval games.

**The likely cause is that the window is far wider than the mistake.** The blunder window is
*turn-scoped* -- `rollout_buffer` pins the blunderer for the whole turn the blunder happened in.
Measured (§21.1), the fatal card play sits **3-9 micro-actions** from the loss. A turn is up to
seven action rounds. So the window sets the advantage to `-1 - v_t` across a long stretch of play
that was mostly fine, and the policy learns to avoid far more than the one card choice that was
wrong. The rising USSR share is consistent with that: the arm is not making one behaviour rarer,
it is distorting a whole turn's worth of play.

That reading is testable and the fix is narrow: scope the window to the action round, or to the
segment from the card play to the terminal, instead of the turn. The turn-scoped form was built
for *unprovoked* suicides, where the mistake is the final move and the window's width costs
nothing because the game ends immediately after. For the provoked case the game also ends
immediately -- but the credited stretch reaches backwards over everything else the player did
that turn.

**The over-reach is 5.6x, counted.** On the five generated replays, the turn the game ended in
holds **124** of the losing player's decisions, of which only **22** are at or after the fatal
choice. Every one of the 124 gets `-1 - v_t`. So 82% of the credited decisions had nothing to do
with the loss, and the worst case (7122) pins 30 decisions for a mistake two decisions deep.

**And the critic still does not see it.** Both checkpoints were run over the *same* stored
positions from those replays, so the comparison is paired and any difference is the critic rather
than the games it happened to play. The delta from choosing to spend the card for Operations:

| | control critic | windowing critic |
|:---|---:|---:|
| 7107 Star Wars | −0.023 | **−0.001** |
| 7115 Lone Gunman | −0.026 | **+0.050** |
| 7118 Grain Sales | +0.024 | **−0.002** |
| 7119 Grain Sales | −0.036 | **+0.008** |

No more reactive than the control, and in two of four less. That is not a surprise on reflection:
a window's advantage is `-1 - v_t` *by construction*, so it teaches the policy while routing
around the value function entirely. The behaviour moved and the understanding did not.

Which is the best available explanation for the Elo: the policy learned a blunt avoidance over a
whole turn's play rather than the one conjunction that is actually fatal, because nothing in this
arm taught it the conjunction. It also makes the auxiliary DEFCON-risk head the natural next step
rather than a narrower window alone -- the head is the only piece on the table that would make
the network *represent* the danger instead of avoiding a region of the game.

**The opening got better, so that is not where the Elo went.** The setup probe on the same two
checkpoints, 1,500 games each:

| target | control | window |
|:---|---:|---:|
| USSR Poland ≥ 3 | 84.8% | **97.9%** |
| US Italy ≥ 2 | 54.6% | **82.3%** |
| US Iran ≥ 2 | 1.2% | **46.0%** |
| US West Germany ≥ 4 | 0.0% | 0.0% |

Three of the four targets improve substantially and none regresses, which localises the loss to
mid and late play -- exactly where a DEFCON-2 turn lives, and so consistent with the over-reach.
(Both arms place West Germany ≥ 4 in 0.0% of games. That is a property of this 80M lineage, not
of the window: H2 at 240M manages 73.2%. Do not read it across lineages or budgets.)

**Verdict: do not adopt as it stands.** The behavioural target is reachable -- 13.7% against the
human 11.7% is the closest this project has come -- but not at this price, and the next attempt
should narrow the window before anything else.

### 21.3 Rate arms against H2 @480M, not against HeuristicBot

Every P1 and windowing arm was re-rated in **one pool** with H2 @480M, four late snapshots each,
50 games a side. Elo is not comparable across tournaments, so a common scale requires a common
pool.

| arm | vs H2 @480M | Elo gap | vs the 80M control | Elo |
|:---|---:|---:|---:|---:|
| control 80M | 26.6% | −176 | — | — |
| control 160M | 39.8% | −72 | 63.7% | +98 |
| categorical 80M | 21.4% | −226 | **46.0%** [43.6, 48.4] | **−28** |
| filter 921 / 922 80M | 26.0% / 31.1% | −182 / −138 | 52.9% / 53.6% | +20 / +25 |
| filter 921 / 922 160M | 41.9% / 41.3% | −57 / −61 | 65.4% / 67.2% | +111 / +125 |
| window 921 / 922 80M | 19.7% / 21.3% | −244 / −227 | 41.8% / 36.1% | −58 / −99 |
| H2 @480M | — | — | 73.1% | +174 |

**The weak anchor inverts orderings that the strong one gets right.** Against HeuristicBot,
windowing seed ...922 scored **81.8%** against the control's 80.6% — above it. Against H2 @480M
it scores 21.3% against 26.6% — correctly below. The head-to-head against the control always knew
(−99 Elo); the point is that the *anchor rate*, which is what a training run logs live and what a
monitor shows, is not merely noisy but can rank a materially weaker arm first. Rate against
H2 @480M.

**And it changed a verdict.** The categorical head was recorded as a null on 80.0% against
HeuristicBot versus the control's 80.6%. Head-to-head in this pool it is **−28 Elo** with the
interval excluding 50%: mildly harmful, not neutral. One seed, so the magnitude is soft, but the
sign is no longer in doubt.

That arm had never been rated at all, and could not have been: `NeuralAgent.from_checkpoint`
built a scalar `ColdWarNetV2` unconditionally, so a categorical checkpoint failed to load with
`value_dist_head` unexpected and `val_vp_head` missing. Every number previously reported for it
came from the training loop's own evaluation, which builds the model itself and so never hit the
path the tournament and every probe use. The head is now detected by weight name.

## Agreement with human play

The corpus is the only strategy prior available, so how closely a policy reproduces it is a
reported figure everywhere -- per BC epoch, per snapshot, and in the tournament reports. It is
not accuracy: a play that spends several points is one decision made several times, and the
order the log happens to record is not part of it.

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

