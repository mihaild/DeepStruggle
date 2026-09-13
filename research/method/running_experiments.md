# Running an experiment so its result means something

Living reference for how an arm is budgeted, run and rated here, and for the engine flags that
change what a "step" is. **Update discipline: rewritten in place.** When a practice changes,
change it here and leave the history of *why* in
[`../log/variance_and_noise.md`](../log/variance_and_noise.md) and
[`../log/measurement_bugs.md`](../log/measurement_bugs.md); this file should never accumulate
superseded advice. What belongs here: how to budget an arm, how many seeds and snapshots a
comparison needs, what to rate against, and the measured cost and outcome-neutrality of the
options the runner exposes. What does not: the experiment that established a number (log), and
the checklist of ways an instrument lies
([`measurement_pitfalls.md`](measurement_pitfalls.md)).

These notes were `metrics.md` §6 and §7.1.

---

## How to rate an arm

The short form of what
[`../log/variance_and_noise.md`](../log/variance_and_noise.md) and the architecture programme
([`../log/P9_architecture.md`](../log/P9_architecture.md)) cost to learn. Every line here is a
practice, not a finding; the measurements behind them are in those files.

* **Rate the last four snapshots and report the mean, never the final one.** Free, and it cuts
  between-run SD from 58.5 to 20.1 on the synth-only configuration. A single-snapshot comparison
  carries the run's own oscillation (~30 Elo) on top of its binomial SE, so it cannot resolve
  anything below roughly 50 Elo.
* **Two or three seeds per arm** for effects above ~40 Elo. Below that, one seed decides nothing.
* **Quote a leg's gain with ±16 Elo, or do not quote it as a trend.** The same 80M leg from the
  same resume state was worth +86.9 on one seed and +54.7 on another.
* **Never compare Elo across tournaments.** Within one pool the numbers are comparable; across
  two they are not, and Elo converted from pooled win rates is not additive — for a specific
  comparison, re-anchor and measure that pair directly.
* **Rate against a strong pooled reference, not against `HeuristicBot`.** The anchor
  systematically *overrates weaker arms* — it read the MLP backbone and one windowing seed above
  a control that beat both by 77–110 Elo head to head. No live training metric can rank arms.
* **Budget by `--train-steps`, never by wall clock** (below), and give every model in a
  tournament a distinct filename.

---

## Method notes

- **Budget A/B arms by `--train-steps`, never by wall clock.** Steps/sec is policy-dependent,
  so a time budget hands the arms different amounts of training
  ([`../log/early_training_signal.md`](../log/early_training_signal.md) §3.1).
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
  by ~1.5 points on a matchup
  ([`../log/variance_and_noise.md`](../log/variance_and_noise.md)). Treat that as the noise floor at 1,000 games a pair, not the
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
  +3 Elo by the budget curve ([`../log/variance_and_noise.md`](../log/variance_and_noise.md)), which is inside the ±16-24 Elo noise floor but is a real
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

## What a tournament number is reproducible to

A tournament is **not** reproducible from its configuration: two identical 6,000-game runs
differ by about 1.5 points on a matchup, which is the noise floor at 1,000 games a pair. The
measurement and its consequences are in
[`../log/variance_and_noise.md`](../log/variance_and_noise.md).

### Auto-advance does not change outcomes

`Engine::step(..., auto_advance)` resolves unattended die rolls, single-choice masks and a few
deterministic multi-step events (Suez <= 4, Muslim Revolution <= 2, East European Unrest <= 3,
Truman, Independent Reds) inside the engine. Enabling it must be a pure speed change or every
tournament number taken with it is incomparable to one taken without.

**Bit-exact where bit-exactness is possible.**
`tests/training/test_auto_advance_outcome_equivalence.py` plays 128 vectorized games under a
policy that is a pure function of the mask, and asserts that terminal utility, victory points and
final turn are identical with the flag off and on. The policy has to be position-derived rather
than RNG-driven: with the flag on the engine asks for fewer actions, so a policy consuming a
shared random stream would diverge for reasons unrelated to the flag. This joins the existing
single-state suite (`tests/training/test_auto_advance_integration.py`, plus
`engine/tests/test_auto_advance.cpp`).

**It removes little.** Under that policy the flag cut batched step calls only from 672 to 650
(3.3%), and wall-clock at that scale was inconclusive. It is not the speed lever it looks like.
