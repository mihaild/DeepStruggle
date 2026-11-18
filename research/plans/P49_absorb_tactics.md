# P49 — a main that takes in a tactic it cannot find by itself, with no other lineage in its training (2026-11-18)

**Status:** proposed (owner, 2026-11-18). P49a trains nothing; P49b is the search for an exploiter that needs no
foreign lineage; P49c is the absorption; P49d the loop.
**Gate:** P49c waits for a P49b arm that passes (or the owner's marked exception, below).
**Needs approval:** each launch; trainer-side only. Builds are named where they are needed.

## Goal

A network that plays the no-pool run (`E8.1-A8-R132-S45`, "B6") about as well as its J1-started exploiter does
-- 68% as the US against 49% in B6's self-play -- **and is not worse than B6 against anything else** (owner,
2026-11-18). Twilight Struggle may be somewhat intransitive and an equilibrium solver may be wanted one day;
at this level the holes are one-sided mistakes that a strictly better network repairs.

**The constraint (owner):** nothing started from J1, or from any other finished lineage, enters the new main's
training. An exploiter that needs a fresh 10,000M lineage every few hundred million steps is not a method.
The J1-started exploiter (`E8-A8-R102-S44@9990M+R136`, **X** below) is kept out of training entirely and used
as the yardstick: the known answer P49b's search is compared with, and the examination P49c's main sits
without having met it.

## Why

Numbers and instruments: [`../log/P49_exploiter_cross_table.md`](../log/P49_exploiter_cross_table.md).

* **X found a tactic, not a blunder.** As the US it gives up Europe and races to 20 VP: the US wins by 20 VP in
  45% of games (24% in B6's self-play) and B6's USSR takes Europe Control in 23% (3.3%) and still loses the
  rest. Against the four networks it was not aimed at X's US scores 44-55%, about what their own US does: the
  line costs nothing elsewhere, so a general network can simply play it. (J1's exploiter is the opposite case:
  66-68% inside the blunder-aware family, 11-12% outside it; that hole is the reward's and pure win/loss
  already closes it.)
* **B6 cannot find it from where it stands.** Its own exploiter, started from B6 at 11,600M, reaches 52% in 800M;
  X, started from J1 at 9,990M -- 42% against B6 before any exploiter training -- reaches 63% in 800M and 69%
  in 1,600M. What X had was a different place to start from, not strength.
* **The defence is learnable on this architecture and recipe:** the pooled control is not exploited by X's line
  (46.8%).
* **Once the USSR defends, self-play stops rewarding the line,** so the ability to punish the old weakness has
  to be kept by something other than self-play.
* **Part of any exploit of a frozen greedy network is that snapshot's:** X at 10,800M scores 61.6% against the
  state it trained on and 55.0% against B6 800M later. The prize is nearer 7-13 points than 20, and an
  exploiter should not train against one frozen network.
* **The machinery exists:** the seat-true league and its driver (P45), the opponent prior with a learned gain
  (P46c), one opponent per game (P47), the restart pool (P48b).

## P49a — the target and the signature (CPU, no training)

1. **The robust margin.** X's newest SWA (11,520-11,600M) as the US against B6 at 10,800, 11,200, 11,600, 12,000
   and 12,400M and against B6's soup, 2,000 games each: the margin that survives a change of snapshot is the
   number P49b and P49c are held to, not 68%.
2. **X's signature**, from those games, as four numbers by turn: the share of US wins by 20 VP, the share of
   USSR wins by Europe Control, VP at the end of turns 3 and 5, and the US's Europe battlegrounds at the end of
   turn 3 -- beside B6's self-play. It says whether a P49b exploiter found X's tactic or another one; either
   is acceptable, and the difference is recorded.
3. **The other seat.** B6's own USSR exploiter read -1.6 and J1's USSR exploiter of B6 was stopped after one
   read (46.1% as the USSR at 410M); whether B6's US has a hole of this size is not known. No run is planned
   for it here; P49d's second seat takes it up.

## P49b — an exploiter of B6 from inside its own lineage

**Target, for every arm:** B6's eight snapshots of 11,530-11,600M and their SWA, greedy, one drawn per game
(the fixed-opponent list drawn per game is the one build if `--opponent-per-game` does not yet cover it), in
place of one frozen network. The US seat, X's recipe otherwise (R136's flags), 1,600M, reads every 400M.

| arm | the exploiter starts from | why it might leave B6's basin | build |
|:---|:---|:---|:---|
| E1 | B6's own state at 4,020M | an earlier era of the same run plays differently and scores about what J1 did against B6 before training; it is free | none |
| E2 | B6 at 11,600M, paid only for the ending X wins by: +1 a US win by 20 VP, 0 any other US win, -1 a loss (P41's restricted payoff) | the reward points at the line; a test with a known answer | an ending-restricted reward for exploiter runs |
| E3 | B6 at 11,600M with fresh policy heads on its trunk | the policy is relearned against the target instead of adjusted | none if the head reset of R87 still runs |
| E4 | a new no-pool seed trained to 2,000M | a different run's early style; costs 2,000M more | none |

E1 and E2 first, one in each slot (about nine hours); E3 next; E4 only if the first three fail.

**Reads** at each 400M, as the US: against the target (X's curve: 47 / 63 / 67 / 69%); against B6 at 12,400M,
which no arm trains on; against the pooled control and J1's best, as the cost elsewhere; P49a's signature.

**Pass:** 60% or more against the target by 1,600M, 55% or more against B6 at 12,400M, and not under 44%
against the control and J1. **All arms fail ->** the reading "a different lineage is needed" stands, P49c runs
once with X itself as a marked exception only if the owner says so (to learn whether absorption works at
all), and the search moves to two no-pool mains of different seeds trained side by side, each the other's
source of unfamiliar play.

## P49c — absorption: defence, attack, retention

The new main is B6 from its 11,600M state, the no-pool recipe (R132), with three additions and nothing from
another lineage. 800M, then 800M.

| share of games | what | what it teaches |
|:---|:---|:---|
| 70% | self-play, as now | everything else |
| 15% | **the P49b exploiter, in its seat**, one opponent a game, the prior with a learned gain | the USSR meets the line from the first move: defence |
| 10% | **restarts from the league games' own positions** (the turn boundaries of turns 2-6, where the line's shape is on the board), the main playing both sides from there | the critic learns what those positions are worth, and the main's own US learns to steer into them: attack, with no imitation term |
| 5% | **B6's 11,600M state as a fixed opponent**, both seats, the prior on | punishing the old weakness stays rewarded after the main's own USSR has stopped showing it: retention |

* The restart pool takes its positions from the league games in this arm (a source switch on `restart_pool`),
  admitted without the critic's openness test for the first 200M -- the critic misjudging these positions is
  the starting condition, not a reason to skip them -- and by P48b's put-back rule after.
* If the main's US has not moved by the first read (below), a small pull of its US play toward the exploiter's
  on the league games' positions, fading to zero over 400M, is the one addition; it is not in the first run.

**Reads** at +400M, +800M and +1,600M, per seat from the JSON, beside B6's own continuation:

1. the main as the US against B6 at 11,600M and at 12,400M (now 49%; the target is P49a's robust margin);
2. the main as the USSR against the exploiter it trains with (now about 32%; back to its self-play rate);
3. **the main as the USSR against X, which it never met** (X's US now 68%; under 55% is the pass);
4. the control, J1's best and the old reference (not below B6's continuation at the same step);
5. setups, the handover check, and the share of the main's own US games that carry P49a's signature.

**Prediction.** Read 2 within 3 points of the self-play rate by +400M; read 3 under 55% by +800M; read 1 at half
the robust margin by +800M and at it by +1,600M; read 4 level.
**Rule.** Reads 2-4 met and read 1 not -> the defence is in and the attack is not: add the pull once. Read 3 not
met while read 2 is -> the main learned its sparring partner and not the line; the exploiter's lineage must be
continued before it is a teacher (P49d). Read 4 down by 2 points or more -> the shares are halved once.

Afterwards, one at a time and only if the arm worked: without the restarts, and without the retention
opponent, to learn which of the three additions carried it.

## P49d — the loop, and when it stops

* **A generation** is 800M of the main beside 800M of an exploiter in the other slot: the P49b exploiter
  **continued** against the new main (its base is paid for once), then the same for the USSR seat with an E1-
  type start. The newest two exploiters of a seat stay in the league; older ones leave.
* **X is continued 800M against the final main and never fed back:** what it still finds is the result.
* **Stop** after two generations in which no exploiter gains 3 points or more over the main's self-play rate
  in its seat and the references do not move.

## Cost

| | steps | about |
|:---|---:|:---|
| P49a | none | a few CPU minutes |
| P49b, E1 and E2 in parallel | 3,200M | 9 hours |
| P49c, two legs, beside the control leg | 1,600M + 1,600M | 9 hours |
| P49d, one generation | 1,600M | 4.5 hours |

## Follow-ups

* Two no-pool mains of different seeds as each other's opponents, if P49b says a lineage cannot exploit
  itself.
* Conditional exploitation (a line played only against an opponent who has shown the weakness) needs the
  opponent's earlier moves in the input, which is the owner's memory decision; nothing here depends on it.

## Runs (filled in while running)
