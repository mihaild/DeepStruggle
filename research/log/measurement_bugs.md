# Measurement bugs — every instrument that reported confident nonsense

The full history of defects found in this project's evaluation and diagnostic code: what each one
was, how it showed up, what it invalidated, and what now stops it recurring.
**Update discipline: append-only.** Nothing here is rewritten or removed, including the readings
that were retracted and the fixes that turned out to be partial — a bug that was fixed twice is
the most useful entry in the file. Add an entry whenever an instrument is found to have been
lying, even if the number it produced was never quoted. The short checklist distilled from these
entries, which is the thing to read before trusting a number, is
[`../method/measurement_pitfalls.md`](../method/measurement_pitfalls.md).

These entries were written under the section numbers of the original `research/experiments.md`,
by way of the retired `research/metrics.md` (see [`README.md`](README.md) for where the rest of
`experiments.md` went). The numbers are recorded here so older references still resolve; new
entries get a descriptive heading and no number.

| written as | now |
|:---|:---|
| §1.1 | Survivorship bias in batched diagnostics — three instances |
| §1.2 | A policy was choosing its own dice in evaluation |
| §1.3 | Evaluation consumed most of a training run |
| §8.5 | The forced-win metric under-detects as well as over-detects |
| §1.4 | A Wargames ending in turn 10 was reported as final scoring |
| §23.1 | Three measurement faults found while running the 240M continuation |
| §1.4.1 | The position-diagnostics probe is reporting nonsense |

Two more instrument faults are recorded elsewhere because they belong to a longer story: the
tournament that is not reproducible from its own configuration (`metrics.md` §7.2) is in
[`variance_and_noise.md`](variance_and_noise.md), and the three methodology faults in the
per-country influence probe (`metrics.md` §21.12) are in
[`P9_architecture.md`](P9_architecture.md).

---

## Read this before trusting any older number

Four defects in the evaluation and diagnostic code, three of them the same defect in three
different files. All are fixed; all invalidate numbers logged before their fix.

Three more of the same character were found later and are below under their own headings: the
forced-win metric, which both over- and under-counts; the session in which a model was fed the
wrong observation layout and misread it without complaint, a tournament nearly rated one
checkpoint twice, and a snapshot sort returned the wrong four snapshots for one arm while
returning the right four for the other; and a turn-10 Wargames counted as final scoring. The
position-diagnostics probe then supplied a fourth instance of the wrong-layout bug.

## Survivorship bias in batched diagnostics — three instances

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

## A policy was choosing its own dice in evaluation

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

## Evaluation consumed most of a training run

Snapshot evaluation used the one-game-at-a-time path (1.82 games/sec) rather than the
vectorized one (53.7 games/sec on the same 100 games, 30x). Cost also grew with run length,
because every snapshot was appended to the opponent list permanently.

Measured on the 3-hour A/B in [`early_training_signal.md`](early_training_signal.md) §3.1:
evaluation took **37%** of one arm's wall clock and
**61%** of the other's. The final evaluation faced 14 opponents and took 957s against a 900s
snapshot interval, leaving about one training iteration per interval.

Fixes: batched evaluation, `--eval-max-snapshot-opponents` (default 4), and evaluation excluded
from the `--duration-seconds` budget. After the fix, both arms of the `early_training_signal.md`
§3.2 rerun showed **zero** gaps over 20s and evaluation overhead of about 4%.


## The forced-win metric under-detects as well as over-detects

Three further corrections, from a review of the individual cases, all pointing the same way: the
metric is not measuring what
[`engine_reanchor_and_human_control.md`](engine_reanchor_and_human_control.md) §8.1 claimed.

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
it scores +1.0 for the US, so there is a second defect here, distinct from
`engine_reanchor_and_human_control.md` §8.4, in whatever terminal the forced walk arrives at. Not
yet diagnosed.

**Where this leaves `engine_reanchor_and_human_control.md` §8.1.** Of the 64 non-endgame
opportunities, 29 are the illegal Ortega/Che coups of its §8.4, and an unknown further number are
headline decisions or wins-taken-by-another-line. The remaining sample is too small and too
contaminated to support any statement about how humans treat forced wins. **§8.1 is withdrawn and
not replaced.** The instrument needs fixing first: skip headlines, test the chosen action for a
win rather than set membership, exclude the last action round of turn 10, and re-run once the
free-coup handlers filter.


## A Wargames ending in turn 10 was reported as final scoring

`classify_game_ending_reason` tested `state.turn < 10` for Wargames and `state.turn >= 10` for
final scoring. Real final scoring terminates at turn **11** (see *measure game length in plies,
not turns* in [`../method/running_experiments.md`](../method/running_experiments.md)), so the `>=
10` branch was only ever reachable by a game that ended *inside* turn 10 — which, with `abs(VP) <
20` and `GAME_OVER`, is a Wargames. Every turn-10 Wargames was therefore counted as final scoring
in `ending_frac_*` and in tournament reports. 3 of the 119 finished human games end exactly that
way. The bound is now `turn <= 10`; `tests/engine_logic/test_game_invariants.py` had encoded the
same misconception and asserted a turn-10 state was final scoring, so it was corrected against 40
driven heuristic games, all of which terminate final scoring at turn 11 / AR 0 and never at turn
10.

The effect on the arms is nil in practice — they play Wargames essentially never (0-1 games in
1,000) — but it mattered for the human baseline, where Wargames is **18.5%** of finished games.


## Three measurement faults found while running the 240M continuation, all silent

Each would have produced a plausible wrong number rather than an error, which is this file's
pattern.

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

## The position-diagnostics probe is reporting nonsense — the fourth wrong-layout instance

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

## `runner.get_state(i)` returns a live view, not a snapshot

`ai/eval/held_scoring.py` captured each env's pre-step state to read back what was in hand when a
game ended, and every held-scoring ending came back reading **turn 1** with a hand that often held
no scoring card at all. The handle is not a copy: it points at the batch runner's slot, so it
follows the env through `step()` *and* through auto-reset. What was being read was the freshly
dealt turn-1 game the slot had already been rewound to, not the game that had just ended.

Confirmed directly -- hold a handle, step 40 times, and its `turn`/`action_round` track the live
state exactly. `GameState.clone()` is the copy; there is no `__copy__`.

It reads as a plausible result, which is what makes it dangerous: "held-scoring losses are a
turn-1 phenomenon" is a claim someone would believe, and the card histogram built on top of it
named **Asia Scoring at 1.76x its exposure**. With `.clone()` the turns spread across 1-10 and the
Asia over-representation drops to 0.89-0.90 on the other two arms -- it was an artifact of reading
reset hands.

The codebase already knew: `ai/training/start_pool.py` clones for exactly this reason, with a
comment. The other ~40 `get_state` call sites use the handle inside the same iteration, before any
step, which is safe -- audited, and no other site holds one across a step.

## `phasing_player` is not the player making a setup decision

`ai/eval/forced_setup.py` scripts the fifteen opening placements, and keyed which side was placing
off `state.phasing_player`. The USSR is the phasing player for the whole of setup, so all nine US
placements were dispatched against the USSR script, which had already run out -- the US placed
"first legal action in the mask" instead, in every game.

Caught only because the probe counts substitutions: **4608 off-script = 2048 x 9 exactly**, one
per US placement. A count that lands on an exact multiple of the games is an unambiguous signal;
without it the run reported a forced opening that was half junk, and it reported it *plausibly*
(US win rate down, Europe-control endings up -- the story one would expect). `get_decision_players()`
is the correct, vectorized source. The probe now raises if any placement goes off-script, since a
partly-forced setup measures neither opening.

## A run watcher tracked "some training process", not its own run

`tools/scripts/watch_run.py` exists because "silence and success look identical" — it watches
whether the step counter is advancing and emits a terminal event either way. Its liveness check
was `pgrep -f <pattern>` with `--pattern` defaulting to `tools/train.py`, which is true whenever
**any** run is on the box.

Found on 2026-09-16, the same day as the E3-22-28 relaunch. The first E3-22-28 attempt was killed
at 47M steps and a corrected run launched in its place. The old watcher, still pointed at the
killed run — whose directory had by then been *renamed* — reported

```
STALL E3-22-28_20260916_121202: alive but stuck at 43,515,904 steps for 1800s
```

It was not stuck. It was dead, and its directory was gone. The watcher was reading the liveness of
the **replacement** run and attributing it to the dead one.

The script's own docstring had worried about a neighbouring version of this — that `pgrep -f
<run-name>` also matches the watcher itself, making liveness permanently true — and filtered the
self-match. It did not consider that a *different run* could supply the same false liveness, which
is the likelier case, since relaunching after a fault is exactly when two runs coexist.

Severity is moderate rather than high: the wrong event still fires, so the failure is loud, and a
STALL prompts the same investigation a CRASH would. The danger is the opposite pairing — a run
that dies while a sibling lives reads as merely stuck, so a crash can be mistaken for a hang and
waited out.

**Fixed by making the run identify itself.** `train_pipeline` writes its PID to `run.pid` in the
run directory before anything can fail, and the watcher checks that exact process with
`os.kill(pid, 0)`. The pattern match survives only as a fallback for runs started before the file
existed — including the E3-22-28 relaunch itself, which was already running when this was written.
`tests/training/test_watch_run_liveness.py` pins the regression: a dead run with a live sibling
must report CRASH and must not report STALL.

A second, smaller thing the test caught about its own design: a watcher on a *healthy* run never
exits, because it exits only on a terminal state. A test that waits for exit therefore hangs. The
helper now takes whatever the watcher printed within a deadline and kills it, which is also how a
human should read it — the absence of a terminal event is the good news.

## Reading a saturated anchor as a null result

Asked whether the training logs show the opponent pool working, I computed external side balance
and win rate against `HeuristicBot` for all eight 4x4 arms, averaged over the final 40M — the
analysis [`../findings/training/pooling.md`](../findings/training/pooling.md) lists as never done —
and got:

| | mean \|gap\| over 120–160M | win rate vs HeuristicBot |
|:---|---:|---:|
| no pool | 8.8 pp (spread 2.3–15.5) | 93.9% (89.5–96.6) |
| pooled | 8.0 pp (spread 1.5–17.0) | 92.1% (80.2–97.5) |

Fully overlapping, and I reported it as a null before noticing why. **The metric is saturated.**
By 120M every arm beats `HeuristicBot` above 89% and `RandomBot` above 99%, so neither can express
a difference that the 20-model peer tournament separates completely (pooled 5.4 pp against unpooled
33.2 pp). A ceiling cannot show you a gap.

This was already on the checklist in two forms and I used the metric anyway: *an endpoint read
from self-play alone can be satisfied without getting stronger*, and *the anchor overrates weaker
arms … no live training metric can rank arms*. What neither entry said explicitly is that the
anchor also **stops discriminating at all** once the arms outgrow it, which is a different failure
from overrating — overrating produces a wrong ordering, saturation produces no ordering. The
checklist entry now says so.

The correct instrument for pooled-vs-unpooled strength is peer Elo from one tournament field, which
exists (`arena_p12`, 76,000 games). The honest statement of what the training logs can contribute
is: **nothing**, on this question, after roughly 120M steps.

The two live metrics that remain useful are the ones that do not depend on an opponent's strength:
`critic_auc` / `critic_brier_skill` (a property of the value head against realised outcomes) and
the blunder rates (violations of the rules of good play, independent of who is across the table).
