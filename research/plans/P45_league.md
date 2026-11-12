# P45 — a league on the J1 recipe: members in their own seat, one member for the whole game (2026-11-12)

**Status:** proposed (owner, 2026-11-12: "write league tests and approach"). L1 runs on the code as it is; L2
onward wait for the pool change below.
**Gate:** L2-L5 need the pool change and its checks merged.
**Needs approval:** none for the engine, bindings or observation (trainer, pool and driver only). L6 and L7
change what every run's pool does and are the owner's call.

## Goal

A stronger base network, read on the old reference (`E7-A8-R1-S44@6400M+R34@7200M+R48@9170M+A13@12730..12800M`,
greedy): J1's continuation scores about 61% against it, and **the target is 75% or higher** (owner,
2026-11-12). The league is the first thing that moved J1's plateau; this plan says how to run it properly and
which runs tell whether it keeps paying.

## Why

* **Step 0a's branch is stronger than J1's own continuation** after the same 800M: 64.2% against 60.9% on the
  old reference (4,000 games each), 54.7% ± 0.6 head to head, more as US than as USSR
  ([log](../log/P44_checks_2026-11-12.md) §1). One branch; not yet attributed to the exploiters.
* **It got that with the league seated wrongly.** The pool draws the learner's seat 50/50 whoever the opponent
  is, so R108 (trained as US only) and R109 (USSR only) played their untrained seat -- 3-8% against J1 -- in
  half their games; and it draws one opponent per 128-step iteration, so a game meets about five opponents
  and no member ever played one from start to finish (log §2). 15% of games were league games; 7.5% had a
  member in its own seat. The owner decided "only in that seat" on 2026-10-09 ([P24](P24_league.md)); it was
  never built, and 0a ran without it. **There is no reason to seat a member where it was not trained
  (owner, 2026-11-12): this plan treats it as a defect, not a variant.**
* **Every earlier league verdict was read on those mechanics** (P24's table: level from a plateau on E7,
  positive only with setup credit on E5/E6), and from plateaus where the exploiter published once in 16
  generations. They do not bound a league seated correctly.
* **A fresh exploiter found nothing against 0a's result in 400M** (margins -0.8 and -0.5; P44, *Runs*), so a
  second generation trained from the main's own state may add opponents without adding exploits. Whether
  such members still help is L3's question, and the reason L3 continues the exploiters' lineages.

## Approach

**Roles.** One main agent (the run being improved) and, per seat, an exploiter lineage: a network that trains
in one seat only (`--opponent-lock-side`) against a frozen, greedy copy of the main (its newest 80M SWA),
R108 / R109's recipe unchanged. No league exploiter and no supervised reset: there is no human prior to
reset to (P24).

**Seating (the defect fixed).** A league member has a seat: the one its run was locked to, read by the pool
from that run's `metadata.json` (`opponent_lock_side`), never declared at launch. In a league game the
learner sits opposite. A member whose run was not locked may take either seat. The learner's seat is drawn
50/50 first, then a member of the opposite seat; with no member for that seat the game is an ordinary
own-pool game, so a league with one seat filled has half the share and never a wrong seat.

**One member per game.** A league game is assigned its member when the game starts and keeps it to the end.
The opponent's rows are evaluated per member (the active members' networks on their own rows), the learner's
pass over the whole batch as now. Own-pool games keep today's one-opponent-per-iteration draw, so a run
without league members executes exactly the code it does today.

**Share.** League games are `--league-frac` of the pool's 30% of games, 0.5 as in 0a: 15% of all games, now
all of them with a member in its seat. The banner still reads `[opponent pool] ... frac=0.3` (invariant 14).

**Members play as they trained:** sampled from their own policy at T = 1; no temperature flag on the main.

**Generations.** Every 800M of the main: the US exploiter for 400M, then the USSR exploiter for 400M, each
against the main's SWA at the generation's start, in the second slot beside the main. An exploiter is
**continued from its own previous generation** (its counter-play carries over; a fresh one found nothing in
400M) and restarted from the main's state only if its margin has gone negative twice. **Every generation is
published**, with its margin recorded (its greedy rate in its seat against the target minus the target's own
self-play rate there, 1,000 deals) -- a record, not a gate: P24's publish bar kept the league empty. Capacity
four members a seat, the oldest out first.

**What the main is judged on:** the old reference, not the league. The exploiter margin is a reading of
exploitability and decides nothing about adoption.

## The pool change and its checks

Code (no engine or observation change): `ai/training/opponent_pool.py` (a member's seat; per-environment
league assignment held to the episode's end; the learner's seat forced opposite), the rollout's opponent
pass in `ai/training/nash_pg.py` (rows grouped by member; the compiled-graph path falls back to eager for
league rows), the member's seat read where `--league-dirs` is scanned, `tools/scripts/league.py` (one
exploiter per seat, continued lineages, publish-always with the margin in `events.jsonl`), and two metrics
(`opp_league_games_<seat>` per member, `opp_league_wrong_seat_games`).

Checks, in `tests/training/test_league_pool.py` and `test_league_driver.py` (CPU, built states, no retries):

1. **A seat-locked member never plays its other seat:** a pool with a US-locked and a USSR-locked member over
   2,000 episode ends; in every league environment the learner's seat is the member's opposite.
2. **A league game keeps its member:** across iterations inside an episode the environment's opponent does
   not change; it changes only at an episode end.
3. **Own-pool games are untouched:** with no league members, and with `--league-frac 0`, the sequence of
   (opponent, learner seat, mixed) for a fixed seed equals the sequence recorded from the code before the
   change.
4. **Shares:** the league's share of mixed episodes is `--league-frac` within tolerance, members of one seat
   are drawn alike, and the learner's seats in league games are 50/50 when both seats have members.
5. **One seat filled:** with only a US-locked member, league games occur only with the learner as USSR and the
   rest fall back to the own pool.
6. **The seat comes from the member's run:** a league directory holding a snapshot of a locked run gives that
   seat, an unlocked run gives both, a snapshot with no readable metadata is refused.
7. **Each environment's opponent rows come from its own member:** two members with distinguishable constant
   policies; the actions taken in each league environment are that member's, and the learner mask is right.
8. **A resume carries no assignment:** every environment starts a new game, seats are read again, and the
   own-pool statistics round-trip as today.
9. **End to end:** two iterations of the real trainer on a tiny network with two seat-locked members;
   `opp_league_wrong_seat_games` is 0 and each member's games are all in its seat.
10. **The driver:** a generation trains US then USSR against the SWA at its start, continues each lineage,
    publishes both with their margins, and evicts the oldest member of a seat at capacity.

The same commit records the league's settings in `metadata.json`, so `launch_flags.py` and `run_codes.py`
see them.

## The league tests

Each is a branch of J1 from its 15,210M state (the plateau, where 0a ran), 800M unless said, one change from
its neighbour. Recipe codes are allocated at launch after checking `main` and `experimental-main`. T0 is 0a.

| | members | mechanics | question | prediction on the old reference at 16,000M |
|:---|:---|:---|:---|:---|
| T0 (done) | R108, R109 (15,610M) | as the pool is | does a plateau main gain from exploiters in its pool? | 64.2% against the continuation's 60.9% |
| **L1** | J1's own 15,610M snapshot (one member; the share is `--league-frac`, not the count) | as the pool is (0a's flags, another league directory) | was it the exploiters, or any current-strength frozen opponent? | at the continuation's level if exploiters matter; at T0's if recency does |
| **L2** | R108, R109 (15,610M) | seat-true, one member a game | does seating the league properly raise the gain? | above T0: 65.5% or more, and above 52% against T0 head to head |
| **L3** | L2's, plus a generation every 800M (continued lineages) | L2's | does the gain keep growing? | +1.5 points per 800M on average over 2,400M |
| L4 | L3's | L2's, the league at 30% of games with the own pool kept at 30% (`--opponent-frac 0.6 --league-frac 0.5`) | dose | more than L3's first leg; a departure from invariant 14's banner, owner's call |
| L5 | fresh exploiters of the head-norm run's own 4,000M state | L2's | does a league help before the plateau, so that it belongs in the from-scratch recipe? | ahead of that run's plain continuation after 2,000M |
| L6 | none | own-pool opponents held for the whole game | is one opponent per iteration costing every run? | owner's call; no prediction |
| L7 | none | own-pool draws weighted to recent snapshots | only if L1 reads at T0's level | owner's call |

**Order and slots.** At most two trainers. L1 needs no code and runs first, in the control's slot (the control
paused at a tagged state; it has been level with J1 head to head since 4,800M), beside the head-norm arm:
about 4.5 hours. L2 follows when the pool change is merged. L3 takes both slots (the main and an exploiter),
so it starts when L2 has read and the head-norm arm is at a read where it can pause. L4-L7 by their results.

## Measure

* **Primary:** the 80M SWA at each 400M against the old reference, greedy, three seed sets (10000, 50000,
  70000; 6,000 games, ± 0.6), the same sets for every arm and for J1's continuation at the same step.
* **Beside it:** head to head against the matched no-league continuation and against T0 at the same step;
  J1's 13,520-13,600M SWA; per seat from the JSON.
* **Exploitability:** each generation's margins as published; once per plan, a fresh 400M exploiter per seat
  against the final main (P44's measure 2).
* **Health:** the league's games by member and seat, `opp_league_wrong_seat_games` = 0, the learner's rate
  against each member (a member the learner beats 90% of the time is dead weight and is reported).

## Decision rule

* **L1:** within 1.5 points of T0 on the reference -> the lever is recent frozen opponents; L7 before more
  exploiters. Within 1.5 of the continuation -> exploiters; L2 as planned. Between -> both, L2 first.
* **L2:** the seat-true league is the league from here whatever it reads (the old seating is a defect). Above
  T0 by 1.5 points or more -> the fix was worth strength, and L4's dose question is live.
* **L3:** continue while each 800M adds a point or more on the reference; stop after two legs under a point.
  The league enters the base recipe when its main is 2 points or more above the matched continuation on
  6,000 games each and not below it head to head.
* **Toward 75%:** the league main's score on the old reference is reported at every read beside J1's 61%.

## Follow-ups

* Exploiters by win condition (P41: a payoff for one ending). Europe Control against the main's USSR is the
  first: it is how people beat the models' USSR (9 of 50 reported wins as US against 1.3-1.4% of US wins in
  self-play; log §5).
* The league on the head-norm recipe from scratch, if L5 passes.
* A second main lineage in the league (the control or J1') once feature bits allow: members must have the
  learner's exact bits.

## Runs (filled in while running)
