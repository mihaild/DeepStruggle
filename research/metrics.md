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

> **Naming, and one thing the old names hid.** Runs are named `<engine><attempt>-<steps>` --
> see [`run_nomenclature.md`](run_nomenclature.md). H2 is **E2**: it predates the Aldrich Ames
> and Star Wars mandatory-choice fixes, so it was trained on a different game from every P1 arm,
> which are **E3**. Every "against H2" figure below is therefore a *cross-engine* comparison,
> with the reference playing a game it never trained on. The handicap is small -- two rare cards
> -- but it runs against the reference, so it flatters the E3 arms slightly. In the new naming
> the control at 80M is `E3-03-80M` and the reference is `E2-02-480M`.

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

### 21.4 The model cannot tell 86% of cards apart

Every card is described to the network by 8 slots saying where it is, **5 properties -- Ops,
side relative to the viewer, era, one-time, is-scoring** (`engine/src/observation.cpp:213-219`)
-- and a flag saying this decision is about it. Nothing about what the card *does*: no target,
no effect class, no "this hands the opponent Operations". Identity exists only as position in
the 110x14 block, and v2's card branch applies one shared MLP per token then mean+max pools, so
position is discarded.

Put every card in one hand, so only its own properties can separate it:

**110 cards collapse to 46 distinct signatures. 95 of them -- 86% -- share a vector with another
card.** Groups run to six. The trunk difference between a hand holding Marshall Plan and the same
hand holding US/Japan Pact is **5e-6**.

Every card this project's failures turn on is ambiguous:

| card | indistinguishable from |
|:---|:---|
| Grain Sales (hands the US your Operations) | Colonial Rear Guards, **The Voice of America** |
| Tear Down this Wall (a free US coup in Europe) | Iron Lady, North Sea Oil, **Chernobyl**, An Evil Empire, AWACS |
| Star Wars (takes a card out of the discard) | Reagan Bombs Libya, Solidarity |
| Junta | **Missile Envy**, Latin American Death Squads, One Small Step |
| Cuban Missile Crisis (a coup here loses the game) | **SALT Negotiations** |
| Olympic Games (the boycott ends the game) | Indo-Pakistani War |
| **Nasser** | **Blockade, Romanian Abdication** |
| UN Intervention | *unique* |

So the Nasser question -- can it know the card targets Egypt rather than some other 2-stability
Middle Eastern battleground -- does not get as far as Egypt. It cannot tell Nasser from Blockade.

**This is the common cause behind most of §21.** The DEFCON blunder rate that never moves in 80M
steps, the UN Intervention misrouting, the Olympic Games rule, coups under Cuban Missile Crisis:
each needs the model to know *which card* it holds, and it does not. Meanwhile the things it does
competently -- spacing a card whose Ops clear the box, the setup probe's targets -- are decidable
from **properties alone**. Property-level competence with identity-level blindness fits every
measurement in this section.

It also retires the reading of §21.1's linear probe. AUC 0.953 on "the card being committed is
one I must not play" cannot have been reading card identity, because there is none; it was
reading the board half of the conjunction -- DEFCON 2 plus an exposed battleground plus the
card's side and Ops -- which gates most of the danger set and ranks well without ever separating
Grain Sales from The Voice of America.

`--identity-dim` adds a learned embedding indexed by position, 5,152 parameters at width 16, and
is model-side: the observation is untouched. It makes the distinction *learnable*. It does not
make it known -- there is still no card-to-effect or card-to-target encoding, so the association
between a card and what it does must come from games in which it was played.

### 21.5 The structured backbone is worth ~110 Elo, and the anchor said the opposite

An MLP control -- the graph convolution, per-card encoder and cross-attention replaced by two
dense layers over the flat observation, same heads, same recipe, 2.6x the parameters, 4.2x the
throughput -- scores **34.5%** and **35.2%** against the v2 control over 3,200 games each:
**−112 and −106 Elo**, two seeds agreeing.

So structure is worth about 110 Elo, and it is not capacity: the control has *more* parameters
and loses. Note the MLP keeps card identity for free, by position, and still loses -- identity is
not what makes v2 good, and the two findings are complementary rather than competing.

**The anchor read the MLP at 88.2% against the control's 80.6%** -- better, by a wide margin.
That is the third inversion in this session:

| arm | anchor | pooled head-to-head |
|:---|---:|---:|
| windowing seed ...922 | 81.8% (above control's 80.6%) | **−77 Elo** |
| MLP backbone | 88.2% (above control's 80.6%) | **−110 Elo** |
| filtering | ordering flips between 80M and 160M | +24 at both |

The bias is one-directional: the anchor **overrates arms that are weaker**. HeuristicBot is a
fixed script, and a differently-trained policy can exploit its habits without being stronger.
This is systematic, not noise, and it means no live training metric can rank arms. Rate against
H2 @480M, pooled over snapshots.

### 21.6 Identity embeddings are worth ~115 Elo

`--identity-dim 16` adds a learned embedding indexed by position to each card and country token.
5,152 parameters, model-side, observation untouched. Two seeds, 80M steps, pooled over sixteen
snapshot pairings:

| | vs E3-01 control 80M | vs E2-01 H2 80M |
|:---|---:|---:|
| **E3-13** (seed 20260921) | 65.3% [63.6, 66.9] → **+110** | 59.7% → +68 |
| **E3-14** (seed 20260922) | 66.6% [65.0, 68.2] → **+120** | 61.2% → +79 |
| E3-01 control | — | 42.0% → −56 |

Five times the effect of advantage filtering, the largest single change measured in this project,
and the two seeds agree to 10 Elo. It also turns the control's deficit against the E2 baseline at
matched budget into a substantial lead.

The size is what the diagnosis predicted rather than a surprise (§21.4): 86% of cards share a
feature vector, so most of what a player knows about its own hand was unavailable, and the MLP
control (§21.5) had already priced the backbone at ~110 Elo. Behaviour moves with it -- DEFCON-1
endings 49.4% → 32.9-37.3%, provoked 36.2% → 20.9-22.7%, final scoring 6.9% → 13.3-16.7%,
explained variance +0.841 → +0.862/+0.889.

**One thing goes the wrong way in both seeds.** The USSR win rate rises 47.0% → 51.7% and 57.6%,
against a human 49.9%; the control was the most balanced arm measured. Windowing pushed the same
direction. Not disqualifying at this size of gain, but it should be tracked at longer budgets
rather than assumed to wash out.

**Adopt, and re-baseline.** Every E3 comparison to date used a control without identity
embeddings, so the recipe line moves and the control has to be re-run with them before the next
factor is screened.

### 21.7 What the trunk actually encodes, read off linearly

`ai/eval/state_readout.py` fits a **linear** probe from the frozen 512-float trunk to facts about
the position. Linear on purpose: if a fact is not linearly available, no head can condition on it
either, and a deeper probe would only show it is recoverable in principle.

| arm | board R² | hand AUC | **which twin** (chance 0.36) | tracks R² |
|:---|---:|---:|---:|---:|
| E3-01-21-80M control | 0.234 | 0.861 | **0.512** | 0.781 |
| E3-09-21-80M mlp | **0.396** | 0.884 | **0.717** | 0.630 |
| E3-10-21-80M identity | 0.230 | 0.864 | **0.628** | 0.739 |
| E2-02-21-480M | 0.259 | 0.883 | **0.546** | 0.856 |

**The aggregate hand AUC answers nothing.** Every arm scores ~0.86, including ones that provably
cannot identify a card, because most cards *are* separable by properties: "a 3-Ops US early-war
card" narrows 110 to about six and lifts AUC far above chance without identity.

**The within-collision-group test is the real one.** Restricted to positions where exactly one
member of a same-feature group is in hand, the question is which -- and properties cannot help.
Identity embeddings take it from 0.512 to 0.628 and the positional MLP to 0.717, against a chance
rate of 0.36. That is the mechanism behind §21.6's +110 Elo, measured directly rather than
inferred. (Caveat: the non-identity arms sit above chance because the group's other members are
visibly in the discard or deck in a real position, which is a cue that is not identity. It is
shared by every arm, so the ordering holds.)

**And the unexpected result: the trunk barely encodes per-country influence.** Board R² is
0.23-0.40 everywhere. The board branch pools mean+max over 84 country tokens, so which country
holds what is largely gone by the time any head sees it -- the same destruction as for cards, and
the MLP scores highest (0.396) for the same reason it wins the twin test, by reading positionally.
This bears directly on the empty-battleground failure, and it is untouched by identity embeddings,
which address the card side only.

Tracks read well everywhere (DEFCON 0.85-0.92, victory points 0.93-0.97), so nothing is wrong
with the trunk in general -- it is specifically per-entity information that pooling removes.

### 21.8 Identity embeddings hold ~110 Elo across a doubling

Matched budget, same engine, two seeds, pooled over sixteen snapshot pairings against
`E3-01-21-160M`. The first comparison in this work that needs no caveat about engine or budget.

| arm | vs control at 160M | at 80M |
|:---|---:|---:|
| E3-10-21 identity | 63.1% [61.4, 64.7] → **+93** | +110 |
| E3-10-22 identity | 65.8% [64.1, 67.4] → **+114** | +120 |
| E3-07-21 filter | 54.6% [52.9, 56.3] → +32 | +20 |
| identity vs filter, head to head | **58.8% [57.0, 60.4] → +61** | — |

Mean +115 at 80M and +104 at 160M, inside the between-seed spread. Under the log-linear law
(`references.md` §7) that is an **intercept shift**: identity reaches a given strength sooner and
does not change the rate, so it is worth about 110 Elo at any budget rather than compounding.
Filtering has the same shape at a fifth of the size (+20, +24, +32 across three measurements).

**Adopt, and re-baseline.** Every E3 comparison to date used a control that could not identify
its own cards, so `--identity-dim 16` becomes part of the recipe line and the control has to be
re-run with it before the next factor is screened. Filtering's +32 was measured on top of a
blind policy and is not established on top of identity.

**A caution that applies to this section's behavioural numbers generally.** The control's own
DEFCON-1 share falls 49.4% → 32.4% between 80M and 240M with no intervention at all, and its
USSR share rises 47.0% → 65.4%. At 160M the identity arms' behavioural advantage is much smaller
than it looked at 80M -- DEFCON-1 33.9% against 38.7%, final scoring level, USSR imbalance now
*worse* than the control's. So part of what §21.2 and §21.6 credited to interventions is what
longer training does anyway. The Elo comparisons were all at matched steps and stand; the
behavioural ones need a matched-budget control before they mean what they appear to.

### 21.9 Dropping the 1,364 constant observation slots buys nothing, and could not have

1,364 of the 3,824 observation floats never change value in any position -- per-country and
per-card properties (stability, region membership, Ops value, era) that the structured encoders
in §21.5 need, because a graph convolution and a per-card MLP see a *token* and have no other way
to know which country or card it is. An MLP reading a flat vector does not: it recovers identity
from the offset. So the question was whether those slots are dead weight for E3-09.

E3-11 is E3-09 with them masked out: 2,460 inputs, 6.6M parameters against 8.0M, two seeds at 80M.

| | vs E3-09-21-080M | vs E3-01-21-080M control |
|:---|---:|---:|
| E3-11-21-080M | 44.3% [42.6, 46.0] → **−40** | 35.1% [33.4, 36.7] → −107 |
| E3-11-22-080M | 50.0% [48.3, 51.7] → **−0** | 39.0% [37.3, 40.7] → −78 |

**Not adopted**, and the mean of −20 is inside a seed spread of 40. Throughput was unchanged
(64.4k against 65.5k steps/s), which is the practical answer on its own: the saving was supposed
to be speed and there was none.

The stronger statement is that a difference here *cannot* be information. A constant input
contributes `w·c` to every unit, which the bias already spans, so the two networks have the same
function class and the masked one loses nothing it could have used. Any real residual would be
optimisation-side: `nn.Linear` initialises `U(±1/√fan_in)`, and fan_in moving 3,824 → 2,460
rescales the init of *every* first-layer weight, the varying ones included. That is a reason to
expect small noise, not a reason to expect a loss.

**The mask was verified rather than assumed**, because the first attempt at it was wrong. Across
16,800 positions, **0 of the 1,364 claimed-constant dimensions takes a second value** -- counted
by distinct values, not by standard deviation, which is what had previously mislabelled a slot
that varies over a tiny range as constant.

### 21.10 One Elo scale for E3, and identity is worth a doubling of compute

Everything below is on **one scale, anchored at `E3-01-21-080M` = 0**, from 228,000 games in three
pools of four late snapshots per arm-budget. Both of the first two pools contain the anchor, so
nothing here is stitched across pools through a third model.

| arm | vs anchor | 95% CI | **Elo** | BT | snapshot spread |
|:---|---:|:---:|---:|---:|---:|
| `E3-01-21-080M` control | — | — | **0** | 0 | 30 |
| `E3-01-21-160M` control | 64.9% | [63.7, 66.0] | **+106** | +99 | 17 |
| `E3-01-21-240M` control | 67.5% | [66.4, 68.7] | **+127** | +135 | 50 |
| `E3-10-21-080M` identity | 65.4% | [64.2, 66.6] | **+111** | +118 | 67 |
| `E3-10-22-080M` identity | 67.2% | [66.1, 68.4] | **+125** | +118 | 45 |
| `E3-10-21-160M` identity | 75.2% | [74.1, 76.3] | **+193** | +194 | 54 |
| `E3-10-22-160M` identity | 75.7% | [74.6, 76.7] | **+197** | +203 | 33 |
| `E3-11-21-080M` mlp-static | 34.3% | [33.2, 35.5] | **−113** | −115 | 48 |
| `E3-11-22-080M` mlp-static | 38.2% | [37.1, 39.4] | **−83** | −76 | 67 |

`BT` is the Bradley-Terry fit over the whole pool, recentred on the anchor's four snapshots. It
agrees with the pooled win rate within 8 Elo everywhere, so the pool is transitive and neither
column is doing hidden work.

**Identity at 80M is level with the control at 160M.** Measured directly rather than inferred
through the anchor -- `E3-10-21-080M` scores **52.4% [51.2, 53.6]** and `E3-10-22-080M` **49.9%
[48.7, 51.1]** against `E3-01-21-160M`. So 5,152 embedding parameters buy what doubling the
budget buys, which is the sharpest form of the intercept-shift claim in §21.8 and the reason
identity is in the recipe rather than on the list of things to try.

It does not buy the *next* doubling as well: the same pool puts `E3-01-21-240M` at **+38** over
the 160M control, above identity@80M.

**The control's own returns are collapsing.** +106 for the first doubling, then +38 for the 1.5x
from 160M to 240M -- 36 per doubling against 106. Two consequences: an arm measured only at 80M
is measured on the steep part of the curve, and the 240M control is a much harder reference than
its step count suggests.

**Two cross-checks, both passed.** `E3-01-21-240M` vs `E3-01-21-160M` reads **+38 in two
independent pools**. And identity-vs-control at matched budget reads **+92 / +109** at 160M here
against **+93 / +114** measured separately in §21.8, and **+111 / +125** at 80M against
**+110 / +120** -- four figures reproducing across a different pool composition.

**Elo from a pooled win rate is not additive, and the matched pair is the number to quote.** The
160M control is +106 on the anchor's scale and the 240M control +127, but played against each
other directly the gap is +38, not +21. Nothing is wrong: converting each pooled win rate
separately compresses differences between two arms that are both far from the anchor. For a
specific comparison, re-anchor and measure that pair.

**The snapshot spread column is why §20.7 exists.** Four snapshots of one arm-budget span 17 to
80 Elo. Every effect in the table except identity's is smaller than the largest of those spreads,
so a single-snapshot version of this table would have been noise dressed as a result.

### 21.11 What the 160M arms actually do, and where identity's Elo does *not* show up

Three probes at temperature 0.1 over **all four** late snapshots of each arm-budget -- the
denominator that matters is the run's own oscillation, and the numbers below show why. Spreads
are max − min over the four snapshots.

| share of all games | ctrl 160M | ctrl 240M | id·21 160M | id·22 160M |
|:---|---:|---:|---:|---:|
| **DEFCON-1 total** | 35.3% ±5.7 | 28.5% ±0.7 | 28.2% ±8.5 | 22.6% ±6.5 |
| own goal | 3.9 ±3.3 | 2.8 ±2.5 | 3.3 ±2.0 | 3.5 ±3.0 |
| bad bet | 0.6 ±0.5 | 0.6 ±1.3 | 0.6 ±0.5 | 0.3 ±1.0 |
| forced trap | 10.6 ±5.0 | 7.4 ±3.3 | 6.8 ±7.0 | 6.6 ±4.2 |
| **unforced trap** | 15.0 ±2.8 | 12.4 ±4.5 | 12.7 ±3.0 | 8.2 ±3.0 |
| unclassified | 1.8 ±0.5 | 2.2 ±2.0 | 1.1 ±1.7 | 1.8 ±1.7 |
| in headline | 3.5 ±0.5 | 3.1 ±2.2 | 3.8 ±3.3 | 2.2 ±1.7 |

| | ctrl 160M | ctrl 240M | id·21 160M | id·22 160M |
|:---|---:|---:|---:|---:|
| empty battlegrounds, turn 5 (of 29) | 11.45 ±0.67 | 11.16 ±0.71 | 10.85 ±0.78 | 10.65 ±0.41 |
| empty battlegrounds, turn 8 | 7.26 ±0.63 | 6.94 ±0.70 | 6.67 ±1.21 | 6.46 ±0.90 |
| uncontrolled battlegrounds, turn 8 | 15.58 ±0.98 | 15.61 ±1.37 | 15.35 ±1.46 | 14.84 ±1.02 |
| mean final turn | 7.13 ±0.77 | 6.94 ±0.64 | 7.30 ±0.72 | 7.22 ±0.46 |
| **blunder rate, pooled** | 2.0% ±0.4 | 1.9% ±1.1 | **3.0% ±1.0** | **2.5% ±1.2** |
| · spaced own or neutral | 9.7% ±3.7 | 7.9% ±4.6 | **15.9% ±8.3** | **15.7% ±13.1** |
| · DEFCON suicide with an alternative | 1.8% ±0.8 | 1.9% ±1.2 | 2.3% ±1.1 | 1.4% ±0.7 |
| · Olympic Games at DEFCON 2 | 0.3% ±0.5 | 0.3% ±0.5 | 0.2% ±0.3 | 0.2% ±0.2 |

**Most of this is not resolvable, and saying so is the finding.** Both identity seeds end fewer
games at DEFCON 1 than the control at the same budget, but by 7.1 and 12.7 points against
snapshot spreads of 5.7 to 8.5, and the two identity seeds differ from each other by 5.6. The
direction is consistent; the magnitude is not established. `E3-01` has only one seed at 160M, so
there is no control seed pair to compare that spread against -- a gap worth closing before any of
these behavioural numbers carries an argument.

**The 240M control matches identity@160M on almost every behavioural line** -- 28.5% against
28.2% DEFCON-1, 6.94 against 6.67 empty battlegrounds. §21.8 warned that part of what was
credited to interventions is what longer training does anyway; at matched *behaviour* rather than
matched steps, that is exactly what this shows. Identity's 92-109 Elo over the 160M control is
real (§21.10) and is not visible in these aggregates.

**Identity blunders more, not less.** Pooled rate 3.0% and 2.5% against the control's 2.0%,
driven almost entirely by `spaced_own_or_neutral` -- 15.9% and 15.7% against 9.7%, the same
direction in both seeds and roughly a doubling. The other three rules are level. So the arm that
gained ~100 Elo also spends its own and neutral cards on the space race considerably more often.
Either the rule is mis-specified for a policy that can now tell its cards apart -- spacing a
*specific* low-value own card may be correct where the rule reads only "own or neutral" -- or
identity bought its Elo somewhere else and paid here. This is worth resolving before the rule is
used to judge another arm, and it is the one place in this table where the spread does not
swallow the effect.

#### Which battlegrounds, not how many

The per-country table is where identity is legible, because "8 of 29 empty" is a claim about
*which* eight. Late-game empty rate, mean of four snapshots:

| battleground | ctrl 160M | ctrl 240M | id·21 160M | id·22 160M |
|:---|---:|---:|---:|---:|
| **India** | **98.3%** | 87.7% | **54.4%** | **55.2%** |
| Algeria | 88.3% | 91.4% | 85.5% | 83.9% |
| Saudi Arabia | 82.1% | 69.7% | 93.9% | 78.6% |
| Libya | 79.2% | 42.7% | 59.7% | 56.7% |
| Argentina | 68.7% | 70.8% | 41.0% | 67.9% |
| Nigeria | 51.5% | 30.0% | 35.9% | 35.6% |
| Brazil | 43.6% | 49.5% | 30.9% | 15.7% |
| Cuba | 31.0% | 27.5% | 53.0% | 42.9% |
| **Pakistan** | **23.6%** | 14.1% | **2.6%** | **7.6%** |
| **West Germany** | **15.2%** | 29.3% | **36.8%** | **35.3%** |
| France | 4.5% | 15.4% | 13.2% | 15.6% |
| Italy | 4.5% | 2.5% | 12.1% | 14.2% |

**India stops being invisible.** 98.3% empty in the control -- effectively never touched, the
finding `position_diagnostics` has reported since it was written -- against 54.4% and 55.2% in
both identity seeds. Pakistan moves with it, 23.6% to 2.6% and 7.6%. Two seeds agreeing on a
40-point move is not snapshot noise, and the 240M control only reaches 87.7%, so this is not
simply what more training does.

**It is a reallocation, not an improvement.** West Germany goes the other way, 15.2% to ~36%, and
France and Italy roughly triple. Identity did not learn to contest more of the board; it learned
to contest a *different* part of it. Whether trading Western Europe for South Asia is right is a
question about the game, not about the probe -- but a 5-point battleground region where three
countries are now emptier deserves an answer before this is called a win.

**Four battlegrounds are still untouched in every arm**: Algeria (84-91%), Saudi Arabia (70-94%),
Libya (43-79%) and, in three of four arms, Argentina. Identity did not help there. §21.7's trunk
read-out found per-country influence is the thing the representation destroys (R² 0.23-0.40,
unchanged by identity), and this is the behavioural face of the same gap.

### 21.12 Exact influence and control, read off the trunk per country

§21.7 reported per-country influence at R^2 0.23-0.40 and called it the thing the representation
destroys. R^2 is the wrong scale for the question actually being asked -- *does the trunk know
the position* -- so this measures the two numbers that answer it: the share of held-out positions
where a linear read-out names the **exact** influence, and the share where it calls **control**
correctly, each against the best-constant baseline.

**Three methodology notes. All of them changed the answer more than any arm difference did, and
the first two were caught only after being written up as findings.**

*The estimator, twice.* Rounding a least-squares fit scores **below** the constant baseline -- 61%
against 69% on battlegrounds -- because influence is 0 in most countries most of the time with an
occasional 3 or 4, so the MSE-optimal fit sits between the two and rounds to neither. Replacing it
with least-squares onto one-hot *class* columns looked like the fix and was not: that estimator
**masks intermediate classes** once there are three or more (ESL 4.2) and shrinks rare ones out of
the argmax. Handed a *noiseless* `influence / 10` column -- board slot 0, verbatim -- it recovered
79% against a 72% baseline, closing **24% of the gap with the answer in front of it**. An earlier
version of this section reported that ceiling as "the trunk closes about a fifth".

The probe is now **multinomial logistic regression**, still linear, and it is gated: any estimator
used for this question must first clear `tests/training/test_state_readout_probe.py`, which feeds
it the noiseless column and requires near-perfect recovery *and* requires pure noise to score
exactly the constant baseline. The current one gets 100%, 99.3% buried in 25 noise columns, and
baseline on noise.

*The aggregate.* A mean of per-country ratios is not a usable summary. India has 4% headroom, so
`(acc - base) / 0.04` turns two points of probe noise into -431%, and the average over countries
is then decided by whichever near-constant country wobbled. Every headline below is
`sum(acc - base) / sum(headroom)`, which weights each country by what it had to give.

*The split.* The probe held out whole environments rather than permuting positions, since
successive samples from one env are the same game six steps apart. It was worth doing and it
barely mattered: battleground exact-match 62.0% held-out against 63.2% permuted, about a point.
Kept because it is free, reported because the concern was real and the effect was not.

#### What the trunk actually recovers

Raw accuracy flatters a country that is empty in 97% of positions, so the headline is the share of
the **recoverable gap** closed: 1.0 is a perfect read-out and 0.0 is a representation adding
nothing a constant already gave. Negative means the probe fits noise.

| | ctrl 80M | ctrl 160M | ctrl 240M | id 80M | id 160M |
|:---|---:|---:|---:|---:|---:|
| **influence, gap closed (battlegrounds)** | 4.7% | 7.7% | **10.3%** | 3.1% | 10.0% |
| influence, gap closed (all 84) | 4.0% | 2.0% | 4.8% | 1.7% | 3.3% |
| **control, gap closed (battlegrounds)** | 24.0% | 40.8% | **43.4%** | 37.2% | 43.0% |
| control, gap closed (all 84) | 29.7% | 35.4% | 35.3% | 34.0% | 36.3% |
| exact influence, BG (raw accuracy) | 70.3% | 69.5% | 71.3% | 67.9% | 69.9% |
| · best-constant baseline | 68.8% | 66.9% | 68.0% | 66.8% | 66.5% |
| control correct, BG (raw accuracy) | 88.5% | 90.5% | 91.1% | 89.4% | 90.0% |
| · best-constant baseline | 85.5% | 83.8% | 84.7% | 82.3% | 82.4% |

**Exact influence is almost absent from the trunk.** Four to ten percent of the recoverable gap on
battlegrounds. The raw accuracy column is what makes this concrete: 71.3% against a 68.0% constant
-- three points for 512 floats of representation.

**Control is a different story: a quarter to well over 40%, and it improves sharply with compute**
where influence barely moves. 24.0% → 43.4% across the control's own run against 4.7% → 10.3%. The
trunk is learning *who holds what* and not *by how much* -- the right priority for scoring, and
the wrong one for knowing whether a coup or a placement flips a country.

**Identity helps control, not influence, and most at low budget.** 37.2% against 24.0% at 80M, a
13-point gain; by 160M the control has caught up (43.0 against 40.8). Influence is unmoved at both.
Identity embeddings name the *country*; they do not carry its *number*.

#### Where it is lost: pooling, measured

The numbers above say the trunk does not have it; they do not say where it went. Tapping one
rollout at four points localises it. `raw` is country i's own 26 observation floats -- slot 0 is
literally `my_influence / 10`, so it is also the check that the probe works at all.

Battlegrounds, share of the influence gap closed, whole games held out, **penalty selected per
stage on a validation split**:

| stage | ctrl 240M | id 160M |
|:---|---:|---:|
| `raw` -- country i's 26 observation floats | **96.8%** | 98.5% |
| `gconv1` -- after one graph convolution | 62.9% | 64.5% |
| `gconv2` -- after the second, still pre-pooling | **65.8%** | 65.1% |
| `trunk` -- the 512 floats every head reads | **−0.5%** | 6.9% |

*Tuning the penalty per stage is not a detail.* A single `l2`, fitted to a synthetic one-feature
problem, held the `raw` rung to 85% -- with the answer in slot 0. The same constant is far too
weak for 512 features and far too strong for one, and a sweep moved the trunk rung between 8.9%
and 28.0% depending only on that choice. Each rung now gets its own penalty, chosen on games held
out of training and never on the test games. `raw` lands at 96.8%, and at exactly 100% for most
individual countries, which is what makes the rest of the column readable.

**The trunk holds essentially nothing about exact influence.** Not "about a fifth", which was the
broken probe, and not 12%, which was the under-tuned one: **−0.5% and 6.9%**, at or below what a
constant gives.

**Pooling is where it goes: 66% → 0%.** The pre-pooling token holds two thirds of the recoverable
gap and the 512 floats hold none of it. Mean- and max-pooling over 84 countries is the step that
destroys the board.

**The graph convolution costs a third before that**, 96.8% → 62.9%, and the second layer adds
nothing back, so an attention read-out over `gconv2` tokens caps near 66% rather than 97%.

*Two controls, because "the trunk holds nothing" is the kind of claim a broken probe also makes.*
On the same checkpoint and the same pipeline the trunk gives **tracks R^2 0.847** and **hand AUC
0.874** -- so the representation and the probe are both working, and it is specifically
per-country influence that is absent. And the graph loss tracks node **degree**: correlation
+0.38 with the raw-to-`gconv1` loss, and +0.43 between `gconv1` recovery and the self-loop weight
`1/(deg+1)`.

Per country, the same ladder (control 240M, headroom in brackets):

| battleground | raw | gconv1 | gconv2 | trunk |
|:---|---:|---:|---:|---:|
| Japan (49%) | 99% | 92% | 97% | **79%** |
| South Africa (52%) | 98% | 86% | 93% | **71%** |
| North Korea (67%) | 76% | 65% | 64% | 42% |
| Poland (47%) | 86% | 50% | 45% | 35% |
| Iraq (55%) | 90% | 65% | 67% | 32% |
| South Korea (66%) | 85% | 58% | 58% | 14% |
| Iran (48%) | 80% | 41% | 45% | 8% |
| Israel (38%) | 100% | 33% | 57% | 4% |
| Italy (52%) | 83% | 40% | 45% | **−13%** |
| Egypt (34%) | 82% | 64% | 63% | **−17%** |
| Pakistan (38%) | 66% | 45% | 45% | **−22%** |
| France (33%) | 81% | 63% | 46% | **−41%** |

Only Japan and South Africa survive pooling intact. France, Pakistan, Egypt and Italy are
recoverable from their own token at 45-63% and are **worse than a constant** in the trunk.

#### The graph convolution is the wrong operator for this quantity

`GraphConvLayer` is `A_norm @ (W x) + b`, with `A_norm = D^-1/2 (A + I) D^-1/2` -- textbook GCN.
**One weight matrix is applied to a country and to its neighbours alike**, and the only thing
keeping a country's own value is the self-loop, whose weight is `1/(deg+1)`. So a country's own
influence is attenuated in proportion to how many neighbours it has, and mixed with theirs
through a transform that cannot tell the two apart. That is a low-pass filter, and exact
per-country influence is the high-frequency part of the signal.

The prediction that follows is that survival through `gconv1` should track degree, and it does:

| country | neighbours | self-loop weight | `raw` | `gconv1` |
|:---|---:|---:|---:|---:|
| Australia | 1 | 0.50 | 100% | **100%** |
| Canada | 1 | 0.50 | 99% | **99%** |
| Cuba | 2 | 0.33 | 95% | 95% |
| Panama | 2 | 0.33 | 94% | 96% |
| South Africa | 2 | 0.33 | 100% | 98% |
| Egypt | 3 | 0.25 | 99% | 81% |
| East Germany | 4 | 0.20 | 100% | 66% |
| Israel | 4 | 0.20 | 98% | 60% |
| France | 5 | 0.17 | 98% | **60%** |
| West Germany | 5 | 0.17 | 96% | **50%** |
| Italy | 5 | 0.17 | 100% | **34%** |

Degree-1 countries pass through untouched; the five-neighbour countries of Western Europe lose
half to two thirds. The correlation is +0.38 over the 33 countries with real headroom and is not
the whole story -- Austria (degree 4) loses 86% and Vietnam (degree 2) loses 70%, so traffic
matters too -- but the mechanism is visible and it is the one the operator implies.

**This is fixable without abandoning the map.** The defect is not that adjacency is modelled, it
is that self and neighbour share a transform. `h_i = W_self x_i + W_neigh * mean_j(x_j)`
(GraphSAGE-style), or simply a residual `h = GCN(x) + W x`, gives the network the option of
keeping a country's own value at full strength and costs one more weight matrix per layer.
Adjacency is genuinely part of this game -- placement legality, realignment, superpower adjacency
-- so the relation is worth keeping; what is wrong is being forced to average across it.

#### It tracks stability

Correlating gap-closed against country properties over the battlegrounds with at least 15%
headroom, **stability is the one that lines up** -- r = **+0.63**, monotone at every step:

| stability | mean gap closed | battlegrounds |
|---:|---:|:---|
| 1 | **−30.7%** | Angola, Nigeria, Zaire |
| 2 | **−11.9%** | Brazil, Egypt, Iran, Italy, Mexico, Pakistan, Panama, Thailand, Venezuela |
| 3 | **+17.0%** | Cuba, East Germany, France, Iraq, North Korea, Poland, South Africa, South Korea |
| 4 | **+23.3%** | Israel, Japan, West Germany |

By region: Asia +18.5%, Europe +1.1%, Middle East +0.6%, Africa −6.8%, Central America −7.9%,
South America −21.3%.

The obvious reading -- low-stability countries change hands more, so they are harder to track --
is contradicted by the other correlation in the same fit: gap-closed rises with *headroom*
(r = +0.72), so the countries whose influence varies most are read **better**, not worse, as a
share of what is there. Volatility alone does not explain it.

**A lead, not a conclusion.** n is 23 after the headroom filter, and stability is confounded with
region and with how often a country is contested at all -- and now also with **degree**, which
has a mechanism behind it where stability does not. Western Europe is both high-stability and
high-degree, so the two hypotheses are not separated by this data.

(An earlier version of this note said adjacency could not be tested because the country info
exposed no adjacency field. It does: the key is `neighbors`, not `adjacent`.)

#### What this licenses

An attention block reading the pre-pooling tokens has something real to find -- five times what
the trunk carries -- so the architecture proposal is aimed at the measured defect rather than a
guessed one. Two things the ladder adds to it:

1. **The tokens are already damaged.** Attention over `gconv2` caps near 61%. A skip from the raw
   26 per-country floats into the attention keys and values, or a residual around the graph
   layers, is what recovers the other quarter -- and it is cheap.
2. **It predicts what should change.** If the block works, recoverability at the decision point
   should move from ~12% toward 61%, and toward 85% with the skip. That makes the arm
   falsifiable by something other than whether Elo happened to go up.


### 21.13 Fixing the graph layer works; attention-pooling into the trunk does not

Two architecture changes, both aimed at §21.12, both with their predicted effect written down in
`run_nomenclature.md` before the runs. One prediction held exactly and the other failed, which is
the more useful of the two outcomes.

* **E3-12** (`--self-transform`) gives each graph layer a second weight matrix applied to the node
  itself, so a country need not be averaged with its neighbours.
* **E3-13** (`--self-transform --attn-readout 64`) adds an attention read-out after the residual
  trunk: the state vector queries the 84 country and 110 card tokens, each concatenated with its
  raw observation slots, and the result is folded back into the trunk.

Share of the recoverable exact-influence gap, battlegrounds, 80M, seed 21, games held out:

| stage | E3-10 control | E3-12 self-transform | E3-13 + read-out |
|:---|---:|---:|---:|
| `raw` | 98.0% | 98.1% | 98.7% |
| `gconv1` | 60.1% | **94.3%** | **93.8%** |
| `gconv2` | 63.5% | **89.2%** | **88.5%** |
| `trunk` | 2.8% | **14.3%** | **14.2%** |

**The self-transform does what it was built to do.** The graph layer went from destroying 38
points of per-country influence to destroying 4: `gconv1` recovery 60.1% → 94.3%, against a raw
ceiling of 98%. The mechanism in §21.12 -- one weight matrix for a country and its neighbours,
self-loop weight `1/(deg+1)` -- was the right diagnosis, and a second weight matrix is the whole
fix. Per country the effect is uniform rather than concentrated: South Korea 97%, Iraq 100%,
Italy 94%, Poland 95%, all of which sat between 40% and 67% in the control.

**The attention read-out bought nothing.** E3-13's trunk is 14.2% and E3-12's, with no read-out
at all, is 14.3%. The entire trunk improvement over the control comes from the tokens being less
damaged before pooling; the attention block contributes zero.

That is a design error worth naming precisely, because it was mine. **A single-query read-out is
still a pooling operation.** One query vector over 84 countries returns one weighted average --
attention pooling instead of mean/max pooling. It is a better average, which is presumably part
of why 2.8% became 14%, but it cannot be more than an average, and the ladder now shows the
information is *there*: the token holds 89% and the trunk holds 14%.

**So the bottleneck is the single 512-float vector itself, not how it is filled.** No read-out
that ends in one fixed-size summary can carry 84 countries' worth of exact influence to a head.
That is the case for **per-entity output heads** -- `logit_i = f(token_i, trunk)` for each of the
110 card and 84 country actions -- which was deferred as the riskier change and is now the
indicated one. The ladder gives it a sharp prior: the tokens it would read carry 89%.

#### And it shows up in Elo

Both seeds of both arms, pooled over four late snapshots each, 200 games a side, 24 models,
anchored on the identity control:

| arm | vs anchor | 95% CI | **Elo** | snapshot spread |
|:---|---:|:---:|---:|---:|
| `E3-10-21-080M` control | — | — | **0** | 61 |
| `E3-10-22-080M` control, other seed | 47.4% | [46.2, 48.6] | −18 | 52 |
| **`E3-12-21-080M` self-transform** | 57.5% | [56.3, 58.7] | **+53** | 37 |
| **`E3-12-22-080M` self-transform** | 61.8% | [60.6, 63.0] | **+83** | 94 |
| `E3-13-21-080M` + attention read-out | 48.0% | [46.8, 49.2] | −14 | 60 |
| `E3-13-22-080M` + attention read-out | 46.8% | [45.5, 48.0] | −23 | 70 |

**The representation fix is worth +53 and +83 Elo**, both seeds positive against a control seed
pair 18 apart. Against `E3-12-21` directly, the two controls sit at −56 and −61.

**The read-out costs the entire gain**, on both seeds: −14 and −23 against the control, and −67
and −91 measured directly against `E3-12-21`. It adds no information (above) and it ends the
trunk in a non-residual `Linear → LayerNorm → GELU` after the residual blocks, which breaks the
identity path the stack was built around. Both reasons point the same way, and the ladder rules
out the charitable one.

The seed-22 ladder agrees with seed 21 throughout: `gconv1` 93.4% and 92.1% against the control's
60.1%, and trunks of 6.1% and 8.0% -- low everywhere, and not systematically higher with the
read-out than without it.

**This answers the question the arm was posed to answer.** The prediction registered before the
runs allowed that Elo might not move at all, which would have said per-country influence is not
what limits play. It moved, on both seeds. Exact per-country influence is worth most of a budget
doubling -- §21.10 puts a doubling at +106 and this is +53 and +83 -- for one extra weight matrix
per graph layer and **no measurable throughput cost**: median 15,125 steps/s against the control's
15,097 over the runs themselves. An earlier figure of ~17% came from a 2M-step smoke run and was
startup-dominated.

**Adopt `--self-transform`; do not adopt `--attn-readout` in this form.**

### 21.14 Does a network that can see the board contest more of it?

§21.13 gave the self-transform arm a representation that holds per-country influence where the
control's lost it. The behavioural question that follows is whether it *uses* it: eight of 29
battlegrounds sit empty from turn 8 in the control, always the same ones (§21.11). Position
diagnostics, four late snapshots each, temperature 0.1:

| | E3-10 control | E3-12-21 | E3-12-22 | E3-13-21 |
|:---|---:|---:|---:|---:|
| empty battlegrounds, turn 5 | 11.88 ±1.25 | 11.72 ±1.16 | **8.90 ±1.06** | 11.59 ±0.88 |
| empty battlegrounds, turn 8 | 7.05 ±1.07 | 7.61 ±1.40 | **4.71 ±1.26** | 8.38 ±0.59 |
| uncontrolled, turn 8 | 15.39 ±0.98 | 15.32 ±0.88 | 13.91 ±1.55 | 15.61 ±0.74 |
| mean final turn | 7.38 ±0.62 | 7.52 ±0.29 | 7.61 ±0.68 | 6.91 ±0.62 |

**The answer is "it can, not it does".** `E3-12-22` contests three more battlegrounds at turn 5 and
two and a half more at turn 8 -- far outside the ±1.2 snapshot spread. `E3-12-21`, the same
architecture with a different seed, is indistinguishable from the control. The architecture makes
the board *available*; which policy is found is still down to the seed.

It is suggestive that the seed which used it is the stronger one -- `E3-12-22` is the +83 Elo arm
and `E3-12-21` the +53 -- but that is one pair, and it is exactly the shape of correlation that
needs more seeds before it is a claim.

| battleground | E3-10 control | E3-12-21 | E3-12-22 | E3-13-21 |
|:---|---:|---:|---:|---:|
| **India** | 97.8% | **37.0%** | **61.5%** | 97.6% |
| Saudi Arabia | 92.9% | 77.3% | 94.2% | 95.2% |
| Algeria | 71.7% | 98.1% | **24.5%** | 96.8% |
| Libya | 68.0% | 97.2% | **14.2%** | 56.0% |
| Brazil | 52.1% | 89.2% | 27.6% | 99.6% |
| Argentina | 41.2% | 16.2% | 27.6% | 94.3% |
| Mexico | 19.0% | 96.6% | 13.5% | 64.5% |
| France | 6.7% | 20.6% | 4.3% | 22.8% |

**India is the one country both self-transform seeds agree on**, 97.8% → 37.0% and 61.5%, and the
attention arm leaves it at 97.6%. India was the flagship never-touched battleground in §21.11 and
the one identity embeddings had already halved; the representation fix takes it further, and it is
the one behavioural change that tracks the architecture rather than the seed.

**Everything else is seed-divergent, and wildly.** Seed 21 *abandons* Africa and Central America
-- Algeria 71.7% → 98.1%, Libya 68.0% → 97.2%, Mexico 19.0% → 96.6% -- while seed 22 takes them
up: Algeria to 24.5%, Libya to 14.2%, Mexico to 13.5%. Two runs of one configuration, differing
only in seed, found opposite policies about two whole regions.

**A design limit worth stating.** Only `E3-10-21` was probed as the control, so the control's own
seed variance on these numbers is unmeasured -- and given how far the two E3-12 seeds are apart,
that variance is exactly what would be needed to attribute any of this to the architecture. The
India result survives because both arms agree and the third disagrees; nothing else here does.

`E3-13-21`, the arm whose read-out cost its Elo, is also the arm that contests least: 8.38 empty
at turn 8 against the control's 7.05, and South America effectively abandoned (Brazil 99.6%,
Argentina 94.3%). Its representation and its play agree with each other.

### 21.15 Per-entity policy heads, built wrong: −219 Elo, and why

§21.13 argued that no read-out ending in one fixed-size summary can carry 84 countries to a head,
and that the remaining move was to stop routing the board through the trunk: compute a country's
logit from that country's own token. E3-14 does that. Actions 0-109 (cards) and 119-202
(countries) get a per-entity head; the 18 that name no entity stay dense.

| | vs `E3-12-21` | 95% CI | **Elo** |
|:---|---:|:---:|---:|
| `E3-12-21` self-transform | (anchor) | — | **0** |
| `E3-12-22` self-transform | 54.9% | [53.7, 56.1] | +34 |
| `E3-10-21` control (no self-transform) | 43.0% | [41.8, 44.2] | −49 |
| **`E3-14-21` per-entity heads** | 22.1% | [21.1, 23.1] | **−219** |

Worse than the control it was built on top of, by a margin no snapshot spread explains.

**The cause is in the implementation, not the idea.** The head was written as

```python
self.pe_trunk = nn.Linear(hidden_dim, d)      # 512 -> 64
ctx = self.pe_trunk(h)
country_in = cat([country_token, country_raw_slots, ctx])
```

so **the only path from the trunk to any action logit became 64 floats**, where the dense head
read all 512. Each logit gained its own country's detail and lost seven eighths of its view of
the situation. That trade is what −219 measures, and it is why the arm lands below a control that
at least kept the full trunk.

**The representation went the other way**, which is the tell. Battlegrounds, share of the
recoverable exact-influence gap:

| stage | `E3-10` control | `E3-12-21` | `E3-12-22` | `E3-14-21` |
|:---|---:|---:|---:|---:|
| `gconv1` | 60.1% | 94.3% | 93.4% | **95.4%** |
| `gconv2` | 63.5% | 89.2% | 84.3% | 88.6% |
| `trunk` | 2.8% | 14.3% | 6.1% | **26.9%** |

The trunk holds *more* per-country influence than in either E3-12 seed, plausibly because it is
now under pressure to make that information survive a 64-dim projection. So the arm improved the
representation and halved the play. **A representation probe moving the right way is not evidence
the change helped** -- which is worth having measured, since §21.12-13 leaned on those probes to
choose what to build.

**The prediction was wrong in both halves.** It said the trunk should not move and Elo should
rise. The trunk moved and Elo collapsed. Registering predictions did its job anyway: it made the
result legible instead of something to rationalise.

#### What to build instead

A **residual** form, which is strictly better than what was built:

```python
logit_i = dense_logit_i + per_entity_correction_i
```

At initialisation the correction is near zero, so the network starts *exactly* as the dense
baseline and learns a per-entity refinement on top of it. It keeps the full 512-dim context, it
cannot be worse than dense at initialisation, and it makes the per-entity path an addition rather
than a replacement. E3-14 instead replaced the baseline outright and cold-started from a
bottleneck.

The open question §21.13 raised is therefore still open. Nothing here shows that per-country
information cannot help the policy; it shows that paying 448 floats of global context for it is a
bad trade.

### 21.16 The same idea built two ways spans 413 Elo

E3-15 is E3-14's per-entity heads as a **residual** on the dense logit, with the correction's
output layers zero-initialised so the network starts as the dense baseline exactly.

| arm | vs `E3-12-21` | 95% CI | **Elo** | snapshot spread |
|:---|---:|:---:|---:|---:|
| `E3-12-21` self-transform | (anchor) | — | **0** | 57 |
| `E3-12-22` self-transform | 55.2% | [54.0, 56.4] | +36 | 83 |
| `E3-14-21` per-entity, replacing | 20.8% | [19.8, 21.8] | **−232** | 93 |
| **`E3-15-21` per-entity, residual** | 73.9% | [72.8, 75.0] | **+181** | 53 |

+181 against a snapshot spread of 53, and against a seed gap of 36 in the arm it is anchored on.

**The gap between the two builds is 413 Elo, and the only difference is whether the dense term
survives.** Both compute a per-entity correction from the same tokens with the same 64-float
context. E3-14 returned the correction alone, making that 64-float projection the sole path from
the trunk to any logit; E3-15 adds it to the dense logit, which still reads all 512. The idea was
never the problem.

**The representation is unchanged, so the gain is in the heads.** Battlegrounds: `gconv1` 95.0,
`gconv2` 90.8, `trunk` 14.8 -- E3-12-21's 94.3 / 89.2 / 14.3 to within noise. E3-15 reads the same
trunk as its baseline and plays 181 Elo better, which is the direct confirmation of §21.13's
claim: the information was present and stranded, and the fix was a path to it rather than more of
it.

**Both registered predictions held** -- it could not start worse than the dense baseline, and the
floor was E3-12. Recorded before the run, as with E3-14, whose predictions both failed.

A caution kept from §21.15: E3-14 moved the trunk ladder *up* while halving play, so the ladder is
reported here and not used to argue the arm is good. The Elo is the argument.

**One seed.** The effect is 3.4x the snapshot spread and the two preceding architecture changes
were both legible from one seed, so this is reported rather than held -- but E3-15 now becomes the
recipe everything downstream is measured against, and a baseline resting on one seed is the kind
of thing §20.7 exists to warn about.

### 21.17 The architecture progression on one scale

Every arm of the progression, all seeds that exist, four late snapshots each, 28 models in one
pool anchored on the E3 control:

| arm | vs anchor | 95% CI | **Elo** | snapshot spread |
|:---|---:|:---:|---:|---:|
| `E3-01-21-080M` control | — | — | **0** | 33 |
| `E3-01-21-240M` control, 3x the **steps** | 66.9% | [65.7, 68.0] | **+122** | 55 |
| `E3-10-21-080M` identity | 64.9% | [63.8, 66.1] | +107 | 59 |
| `E3-10-22-080M` identity | 66.0% | [64.9, 67.2] | +115 | 63 |
| `E3-12-21-080M` + self-transform | 72.0% | [70.9, 73.1] | +164 | 51 |
| `E3-12-22-080M` + self-transform | 73.8% | [72.7, 74.8] | +179 | 92 |
| **`E3-15-21-080M` + per-entity residual** | 85.2% | [84.3, 86.0] | **+304** | 67 |

**Identity embeddings alone, at 80M, are worth about what tripling the step budget is worth** --
+107 and +115 against the 240M control's +122. The full stack at 80M clears it by roughly 180 Elo,
for 5,152 embedding parameters, one extra weight matrix per graph layer, and a zero-initialised
correction head.

**In wall clock the comparison is 2.4x, not 3x**, and the difference matters when the claim is
about compute rather than steps. Measured medians over the runs: control 15,097 steps/s, the
self-transform 15,125 (free), the residual heads 12,159. So 240M of control is about 4.4 hours
against 1.8 for 80M of E3-15. The architecture is still ahead on equal wall clock, by a smaller
margin than the step counts suggest.

**Quote the anchored column for the picture and the paired measurement for an increment.** Elo
converted from pooled win rates is not additive: this pool puts E3-15-21 164 above E3-12-21 by
subtraction, while the two measured head to head give **+181** (§21.16). Both are right about
what they measure; the subtraction compresses differences between two arms that are each far from
the anchor (§21.10).

**Each step came from a measurement, not a guess.** Identity followed from 86% of cards sharing a
feature vector; the self-transform from a graph layer that attenuated a country's own influence in
proportion to its degree; the residual heads from information that was present in the tokens at
89-91% and stranded before the heads at 6-14%. The two failures in the sequence -- the attention
read-out and the replacing form of the per-entity heads -- were diagnosed by the same instruments
that chose the successes.

**E3-15 rests on one seed**, and it is now the recipe. That is the standing caveat on this table.

### 21.18 Does the network price a country for the operation it is performing?

Elo says a policy is better; it never says what it understands. Three probes built against the
engine's own answer key, so none of them needs a human judgement.

#### The same card, the same board, Event or Operations

The sharpest of the three, because the two branches differ in exactly one decision. Placement
costs **2 Ops per point in a country the opponent controls** and 1 elsewhere
(`Operations::get_influence_cost`), while event placement never consults cost at all
(`Operations::place_influence`). So free-placement events should want the countries ordinary Ops
should avoid, and a removal event should want them most.

All four cards are held by their own side, so playing for Ops does not fire the event and the
pairing is exact. Probability mass on opponent-controlled countries:

| card | E3-15 Event | E3-15 Ops | **gap** | E3-01 Event | E3-01 Ops | **gap** |
|:---|---:|---:|---:|---:|---:|---:|
| The Voice of America (removal) | 80.1% | 13.4% | **+66.7** | 76.9% | 26.8% | +50.1 |
| Ussuri River Skirmish | 40.3% | 16.0% | **+24.3** | 30.8% | 31.6% | **−0.8** |
| Colonial Rear Guards | 28.9% | 17.3% | **+11.6** | 17.3% | 26.8% | **−9.5** |
| Decolonization | 39.5% | 26.0% | **+13.5** | 11.7% | 20.5% | **−8.8** |

**E3-15 has the gap positive on all four. The control has it backwards on all three
free-placement events** -- it puts *less* weight on opponent-held countries when placement is free
than when it costs double. Voice of America is the easy case for both, because its legal mask
already restricts to countries holding Soviet influence (uniform baseline 60.1%); the
discriminating cases are the placement events, where the uniform baseline is 6-18%.

E3-15's Ops branch sits at 13-26% and its Event branch at 29-80% **on the same boards**. That is
one network conditioning its country choice on the operation, which is what the per-entity heads
were for, and it is invisible to Elo.

#### Where it aims, by operation

Opponent influence in the chosen country against the mean over that node's own legal set, so the
mask cannot manufacture the result:

| operation | legal | chosen | uniform | lift | battleground | unif |
|:---|---:|---:|---:|---:|---:|---:|
| event: Ussuri River Skirmish | 14.5 | 3.97 | 1.17 | **+2.80** | 0.81 | 0.38 |
| event: Brush War | 51.8 | 2.79 | 0.62 | **+2.17** | 0.94 | 0.29 |
| event: The Voice of America | 16.4 | 4.00 | 2.67 | **+1.33** | 0.91 | 0.64 |
| ops: influence | 38.0 | 2.00 | 0.49 | +1.51 | 0.79 | 0.39 |
| **ops: coup** | 9.0 | 2.09 | 2.11 | **−0.02** | **0.59** | **0.68** |
| ops: realign | 6.7 | 2.32 | 2.21 | +0.10 | 0.89 | 0.78 |
| event: Comecon | 7.5 | 0.10 | 0.07 | +0.03 | 0.15 | 0.21 |

**Couping is undiscriminating**: no opponent-influence lift and a *below-uniform* battleground
rate. Couping battlegrounds is what earns military operations and what moves regional scoring, so
this is a specific, named weakness rather than a general one.

#### Picking the best placement

Scored against the engine's true marginal regional VP for every legal country: chosen **1.038**
against 0.272 for a uniform legal pick and 3.162 for the best available -- **lift +0.767**, the
single best target taken **31.7%** of the time, average **93rd percentile** of the legal set. It
reliably finds a good country and often not the best one.

#### What did not work, and why it is recorded

The first version of this probe asked whether regional VP is recoverable from the trunk. It is
not a test: `global_features[64..69]` already carry the live per-region differential, so it asks
whether six floats can be copied. The counterfactual replacement was better and still weak --
board slot 24 is `my_deficit`, "how many Ops to reach control", which is most of the threshold
answer, so a linear probe on the raw input already scores 0.886 AUC and the encoder adds 0.010.
The lesson generalises: **a probe is only a test of understanding if its answer key is absent
from the observation**, and this observation is rich in precomputed per-country facts.

### 21.19 Does it know which region a scoring card scores?

The one association the observation cannot supply. The card block carries properties and
location, never identity, and the six scoring cards collapse into **two feature groups** -- early
war (Asia, Europe, Middle East) and mid war (Central America, Africa, South America). Era
separates the groups; nothing separates within one. So this is a direct test of the identity
embedding.

Both arms are evaluated on **one shared set of boards**. An earlier run let each arm sample its
own self-play positions and produced the opposite conclusion on one of the two measures; that is
the same confound §21.18 records for the marginal-value probe, reintroduced and caught.

#### Location: moving the card, holding the board fixed

Placement mass into the card's own region, minus the mean shift into the other regions, so a
generic "a scoring card is in hand" reflex cancels:

| card | E3-15 | E3-01 control |
|:---|---:|---:|
| Asia Scoring | **+12.3%** | **−11.6%** |
| Europe Scoring | **+12.2%** | +7.0% |
| Middle East Scoring | **+30.3%** | +16.5% |
| Central America Scoring | **+7.8%** | −0.2% |
| Africa Scoring | **+2.7%** | +8.7% |
| South America Scoring | **+4.6%** | +2.6% |

**E3-15 responds in the right direction in all six regions; the control has Asia backwards and
Central America flat.** Moving Europe Scoring into hand raises the model's appetite for Europe
specifically -- the card-to-region association, visible in behaviour.

#### Ordering: which scoring card to play

| | E3-15 | E3-01 control |
|:---|---:|---:|
| took the best scoring card (US / USSR) | 27.4% / 34.4% | **59.7% / 55.7%** |
| rank correlation with true VP | +0.39 / +0.47 | **+0.50 / +0.56** |
| value captured | 74.2% / 71.8% | **89.3% / 88.7%** |

Chance is 16.7% and group-level guessing 33%, so both beat both -- but **the control is markedly
better**, on identical inputs. E3-15 knows *where* a scoring card points and is worse at deciding
*which* to play. That is a negative worth carrying: the arm that wins by 304 Elo loses this
comparison clearly.

#### A blind spot both share

Putting a scoring card in the **opponent's** hand moves neither model: the same difference in
differences runs −1.3% to +1.1% across all six cards and both arms. An opponent holding Europe
Scoring is a reason to defend Europe, and nothing in either policy reacts to it. Hand knowledge
of the opponent is partial, but the card's location *is* in the observation, so this is not an
information limit.

**Caveats.** The shared boards come from E3-15's self-play, so the population is one arm's
distribution even though both models see identical inputs. E3-15 is one seed.

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

