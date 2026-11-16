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

### P48a (2026-11-16): the game stays open late; the critic is the test, near its ceiling to turn 7

B6's 7,920-8,000M SWA (`E8.1-A8-R132-S45`), both sides, T = 1. 2,000 turn-boundary positions for each of turns 3-10,
captured as `start_pool.harvest` does (the state at the last decision before the turn rolls over, so pre-deal), one
per game per turn; 16 reseeded continuations each (256,000 partial games). Scripts `p48a_games.py`, `p48a_read.py`,
`p48a_oracle.py` (the session's tmp); data `data/reports/p48a_games.npz`, full read `data/reports/p48a_read.txt`.
Win = the US's result, a draw 1/2. Deviations agreed with the plans session: the value head has no spread, so (c)
is the VP head's predicted margin; every test is read on the half of positions not used to fit (d), against the
label of continuations 9-16 (B), and (d) is fitted on continuations 1-8 (A).

**Read 1 -- how decided the game is** (beta-binomial fit to the 16 results, which removes their noise):

| turn | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 |
|:---|---:|---:|---:|---:|---:|---:|---:|---:|
| open: true US win p inside 0.2-0.8 | 80.8% | 76.9% | 69.7% | 65.4% | 59.9% | **56.0%** | 53.0% | **43.7%** |
| result variance fixed by the position | 18% | 21% | 26% | 29% | 33% | 36% | 38% | 47% |
| mean US win | 0.45 | 0.47 | 0.48 | 0.52 | 0.54 | 0.55 | 0.56 | 0.57 |

Most of the game is still open late: over half the positions at turn 8 and 44% at turn 10, and even at turn 10 the
position fixes under half the variance of the result. Every turn is worth restarting from; turns 8-10 hold the most
open positions per game step (a game from turn 8 is about a third of a whole one).

**Read 2 -- tests for 'undecided'** (AUC / share of restarts wasted on decided positions at 80% recall; target: the
B label inside 0.2-0.8). The ceiling is the AUC of a test that knew each position's true win probability, against the
same noisy 8-result label, simulated from read 1's fit; the 8-result A label itself scores well below it (0.64-0.82),
so it is not a ceiling.

| turn | open (B) | ceiling (true p) | **(a) critic** | (b1) `is_salvageable` | (b2) \|VP\| | (b3) regions | (c) VP head | (d) best two |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 3 | 70% | 0.797 | **0.782** / 16% | 0.567 / 27% | 0.615 / 24% | 0.661 / 23% | 0.625 / 25% | 0.779 / 16% |
| 4 | 68% | 0.817 | **0.789** / 17% | 0.614 / 24% | 0.651 / 24% | 0.590 / 26% | 0.667 / 22% | 0.789 / 17% |
| 5 | 61% | 0.834 | **0.820** / 20% | 0.594 / 33% | 0.620 / 34% | 0.628 / 34% | 0.631 / 33% | 0.816 / 19% |
| 6 | 58% | 0.850 | **0.820** / 23% | 0.614 / 36% | 0.628 / 37% | 0.654 / 35% | 0.641 / 35% | 0.819 / 23% |
| 7 | 52% | 0.864 | **0.843** / 22% | 0.618 / 39% | 0.669 / 38% | 0.592 / 42% | 0.690 / 36% | 0.843 / 22% |
| 8 | 50% | 0.873 | **0.829** / 23% | 0.652 / 38% | 0.691 / 37% | 0.619 / 42% | 0.703 / 35% | 0.828 / 22% |
| 9 | 45% | 0.887 | **0.821** / 33% | 0.611 / 49% | 0.641 / 49% | 0.567 / 53% | 0.667 / 48% | 0.810 / 34% |
| 10 | 39% | 0.905 | **0.843** / 34% | 0.586 / 55% | 0.607 / 55% | 0.607 / 58% | 0.643 / 53% | 0.843 / 35% |

* **The critic is the test.** It beats the existing rule by 0.17-0.26 AUC at every turn and wastes a third to a half
  as many restarts. Nothing adds to it: the best pair (critic with the VP head, or with VP) is the critic alone.
* **Within 0.03 of the ceiling at turns 3-7** (0.015-0.030), not at turns 8-10 (0.044, 0.066, 0.062) -- the turns
  P48b would weight most. Against the plan's absolute 0.85 it falls short everywhere (0.78-0.84); judged against the
  ceiling, it is near-optimal to turn 7 and falls short late.
* `is_salvageable` is barely better than chance (0.57-0.65), worse than VP alone.

**Read 3 -- where the critic is wrong** (against the mean of all 16 results; points of US win probability):

| turn | critic - result | RMS beyond 16-result noise | the worst tenth of positions holds | open positions the critic calls decided (outside 0.2-0.8) |
|---:|---:|---:|---:|---:|
| 3 | -1.7 | 4.9 | 44% | 61 (4.2% of open) |
| 4 | -1.6 | 4.3 | 46% | 97 (7.0%) |
| 5 | -1.0 | 5.8 | 49% | 85 (6.8%) |
| 6 | -1.4 | 6.3 | 51% | 107 (9.2%) |
| 7 | -1.0 | 6.3 | 49% | 107 (10.3%) |
| 8 | -1.2 | 6.9 | 53% | 146 (14.4%) |
| 9 | -1.0 | 7.5 | 54% | 114 (12.6%) |
| 10 | -0.5 | 9.2 | 57% | 124 (15.9%) |

The critic's error beyond noise doubles from turn 3 to turn 10 (4.9 -> 9.2 points), is concentrated (the worst tenth
holds about half of it), and is larger on open positions (RMS 17-18 against 13-14 on decided ones, turns 6-10). It is
mildly underconfident: about +2 points where the US is losing badly and -2 where it is winning. A critic-based test
would never restart 4-10% of the open positions at turns 3-7 and 13-16% at turns 8-10.

**Read 4 -- Europe as the USSR** (turns 7-10): 6.9% of positions see a US Europe Control win in at least one of 16
continuations, 0.66% of continuations end that way, and those are positions the critic rates the US behind in (US
win p 0.42 against 0.55 elsewhere) -- Europe Control is a US comeback route from losing positions, and rare in
self-play.

**Against the registered expectation:** more of the game is open than expected (turn 8: 56%, not under half; turn
10: 44%, not under a third); the critic beats the VP-and-region rule by far more than 0.05-0.10 (0.17-0.26) but does
not reach 0.85 absolute; it misses under 10% of open positions only to turn 6, 10-16% from turn 7.
