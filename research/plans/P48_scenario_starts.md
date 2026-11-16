# P48 — starting a share of games from given situations: many continuations of undecided positions (2026-11-16)

**Status:** proposed (owner, 2026-11-16). P48a, a check that trains nothing, first: it answers how to find
undecided positions. P48b is the training arm.
**Gate:** P48b waits for P48a's detector and for the owner's go.
**Needs approval:** none for the engine, bindings or observation (the engine already starts a game from a
saved state and reseeds it); trainer-side changes to `--start-pool-*` only.

## Goal

Less noise in what the network learns about the middle and late game: a share of training games start from
positions the run itself reached, each continued several times with a fresh deal, so that a late-war
decision is learned from many results a few dozen moves away and not from one result behind two hundred
decisions. The owner's emphasis: **positions that are not yet decided** -- and the open question of how to
find them, which P48a measures.

## Why

* **A late position is seen once a game, through a long chain.** With lambda 0.98 over ~300 decisions the
  result reaches an early decision's target with weight ~0.2%; everything before the last ~50 decisions
  learns from the critic. A game started at turn 8 has its result inside that horizon from the first move.
* **Late turns are under-sampled.** A game from turn 8 is about a third as long, so the same steps hold
  about three times as many late-war decisions.
* **Several continuations of one position** are what makes a value target for it precise: K results, not one.
* **The critic decides what is learned, and it is only as good as the positions it has seen.** P44-P47's
  lesson: once the critic stopped marking down unfamiliar play, the policy found the human setups by itself
  within 400-1,600M. The same may hold late in the game; nothing has measured the critic there.
* **Feedback from play:** the models do not protect Europe well, the USSR in the late war in particular;
  people take Europe Control from the models' USSR in 9 of 50 reported wins against 1.3-1.4% in self-play.
* **Tried once, far from here.** `--start-pool-frac` (an E1-era architecture, 78M steps): -115 Elo head to
  head, with half the restarts at turn 1 and a policy that changed completely between refreshes; the same
  log found it did help in the turns it trained ([log](../log/early_training_signal.md) §3). Never run on a
  network past its first hundred million steps.
* **Outside:** exploring starts (Sutton and Barto); TRPO's vine sampling (Schulman et al. 2015), several
  rollouts from one state for a lower-variance advantage; Go-Exploit (Trudeau and Bowling 2023, arXiv
  2302.12359), self-play from an archive of states, better value generalisation; KataGo's forked positions
  (Wu 2019); Prioritized Level Replay (Jiang et al. 2021), which replays the starts where the critic was
  most wrong.

## What exists

`ai/training/start_pool.py` (`--start-pool-frac`, `--start-pool-capacity`, `--start-pool-episodes`): positions
are harvested from separate self-play games at every snapshot evaluation, **at the last decision before a
turn rolls over, so the next deal has not happened**; a restart reseeds the state, so each continuation
draws fresh hands and dice; the pool is tagged by the snapshot that made it and refreshed. "Undecided" is a
hand-written rule (`is_salvageable`: |VP| <= 10 and the regions' net score <= 20), and the turn mix is fixed
at 50 / 15 / 15 / 10 / 10 over turns 1 / 4 / 6 / 8 / 10.

## P48a — which positions are undecided, and can a cheap test tell? (no training)

**Network:** the no-pool run's newest SWA (`E8.1-A8-R132-S45`, level with J1's best at 6,000M and 61.5% on the
old reference at 6,800M; P47), whose critic never learned the pool's style bias.

**Positions:** 2,000 turn-boundary positions (pre-deal) for each of turns 3 to 10, from self-play at T = 1.
**Continuations:** 16 of each, fresh deal and dice, both sides the same network at T = 1: 256,000 partial
games. The 16 are split 8 / 8: the first eight label a position, the second eight score every test, so no
test is judged on the results it was built from.

**Reads:**

1. **How decided is the game, by turn?** The spread of a position's true win probability (a beta-binomial fit
   to the 16 results): the share of positions with it inside 0.2-0.8, and the share of the result's variance
   already fixed by the position. This is the curve that says where restarts are worth their games.
2. **The candidate tests for "undecided"**, each scored on how well it picks the positions whose label is
   inside 0.2-0.8 (AUC, and the share of restarts it would waste on decided positions at 80% recall):
   * the critic's win probability, its distance from one half (the two seats' views averaged);
   * the existing rule (VP and region net), and VP with the turn alone;
   * the critic's own spread, where the value head has a distribution;
   * the two best together.
3. **Where the critic is wrong:** its error against the label by turn and by how open the position is, and
   how concentrated the error is (the tenth of positions holding the most of it). Positions the critic calls
   decided that are not are the ones a critic-based test would never restart.
4. **Europe as the USSR:** among turn 7-10 positions, the share where the US wins by Europe Control in at
   least one continuation, and the critic's value there. It sizes the feedback in self-play terms.

**Decision.** A state-only test with AUC 0.85 or more at every turn from 6 -> P48b uses it (a test that reads
only the position does not bias the targets). Otherwise P48b labels positions at harvest by a few probe
continuations whose results are **not** trained on. Either way a fifth of restarts stay unfiltered, so
positions the test wrongly calls decided still get played. Read 1 sets the turn mix: turns where less than a
third of positions are open get no restarts.

## P48b — the arm

A branch of the clean base (the no-pool run at its newest state), 800M, beside that run's own continuation:

* **A quarter of environments start from the pool**; the rest play whole games, so the opening keeps most
  of its training.
* **Turns by P48a's read 1**, weighted to the late war (default if it allows: turns 6-10, most at 8).
* **Undecided positions by P48a's test**, a fifth unfiltered.
* **Four continuations of each position** before it is retired; positions from the newest snapshot only.
* Everything else as the base. Restarted games are ordinary self-play on positions the policy itself
  reaches, so the policy and value losses need no correction; they are logged apart from whole games (ending
  mix, length, the side split), which would otherwise read as a change in play.

**Built for it:** the turn mix and the continuations per position as flags, the chosen test in place of
`is_salvageable`, harvest every 10M as now, the separate logging; the checks in `tests/training/test_start_pool.py`
extended (a restart never repeats a deal; the share, the turn mix and the unfiltered fifth hold; whole-game
statistics exclude restarts).

**Reads**, at +400M and +800M beside the continuation: the old reference and J1's 13,600M SWA; head to head;
the critic's error by turn on P48a's held-out continuations (the direct read); how games end, per seat, and
the US's Europe Control wins; the forced-win take rate (`ai/eval/decisive_probe.py`); setups and the critic's
error in turns 1-3, which must not get worse.

**Prediction.** The critic's error in turns 8-10 falls by a third or more by +400M; the old reference is 2
points or more above the continuation at +800M; turns 1-3 unchanged within a point.
**Rule.** Both -> into the base recipe, and the from-scratch question is next. The critic's error falls but
strength does not move -> the late critic was not what limited play; the arm stays as the starting point for
the follow-ups below and is not adopted. Turns 1-3 worse by 2 points or more -> the share is halved once.

## Follow-ups

* **Opponents that test one ending, started from these positions:** an exploiter paid for Europe Control as
  the US against the main's USSR, and its dual, the defender, in the main's games from late-war positions.
  A defender does not change the main's reward; it makes the main meet the situations where that ending is
  on offer (owner, 2026-11-16). P45's seat-true league and P46's prior are the machinery.
* **Restarts ranked by the critic's error** (Prioritized Level Replay), if P48a's read 3 shows the error
  concentrated in a small share of positions.
* **A value target from the continuations' mean** for the restarted position, if the plain form moves the
  critic slowly.

## Runs (filled in while running)
