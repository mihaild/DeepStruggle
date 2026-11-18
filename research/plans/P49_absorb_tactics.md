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
  (P46c), one opponent per game (P47), the restart pool (P48b). **The prior is used in every game the main plays
  against an opponent that is not itself** (P49c says how, and why the restarts are needed beside it).

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

The new main is B6 from its 11,600M state, the no-pool recipe (R132), with three kinds of game added and
nothing from another lineage. 800M, then 800M.

**The critic prior is used: in every game against the exploiter and against the old state** (P46c's form, the
gain learned, as built for R128). It is zero in self-play and in the restarts.

| kind of game | share | who plays | the value the critic predicts | what it teaches |
|:---|---:|:---|:---|:---|
| self-play | 70% | the main, both seats | `sigmoid(c(s))` -- no prior | everything else |
| **against the exploiter** | 15% | the P49b exploiter in its seat (the US), the main in the other; one opponent for the whole game | `sigmoid(c(s) +- g(s) * beta)`, **`beta` the prior**: the main's running score against this exploiter in this seat, in log-odds against its self-play rate there; + on the main's rows, - on the exploiter's | the USSR meets the line from the first move: defence |
| restarts from those games | 10% | the main, both seats, from the turn boundaries of turns 2-6 of the games against the exploiter | `sigmoid(c(s))` -- no prior, the exploiter is no longer playing | what the line's positions are worth, and the main's own US steering into them: attack, with no imitation term |
| **against B6's 11,600M state** | 5% | the old state in either seat, the main in the other; one opponent a game | `sigmoid(c(s) +- g(s) * beta)`, **`beta` the prior** for that state and seat | punishing the old weakness stays rewarded: retention |

`c(s)` is the critic's win logit and `g(s)` its gain head, both reading the position only; `beta` is the one
number from outside the network, kept by the trainer per opponent and seat as a running average. The shifted
value is the one stored for GAE, bootstrapped from and fitted in the value loss, on every row of the game; the
policy loss takes the main's rows only, as in every game against a fixed opponent.

### Why the prior, and why it is not enough by itself

P44-P46's trap was a critic that read "plays unlike the current self" as "loses", because the unfamiliar side
(old pool members) lost. Here the unfamiliar side wins. Without a prior the critic would learn "the US is
playing this line, so the US wins", and would then credit the main's own US for positions that are good only
in the exploiter's hands.

* **With the prior**, the exploiter's edge is carried by `g(s) * beta`, and `c(s)` is left to say what the
  position is worth between two copies of the main.
* **The prior alone does not guarantee that**, because an exploiter's strength is not one number: it is strong
  where its line is on the board and weak elsewhere. `g(s)` reads the position, so it can carry that -- but in
  the exploiter's games alone nothing says how to split a line position's value between `c` and `g`.
* **The restarts make the split determinate.** They are the same positions with the main on both sides and
  `beta = 0`, so they fit `c(s)` directly, and what is left over in the exploiter's games can only go into the
  gain. They are therefore part of the critic's correctness and not only of the attack: their share is not cut
  below half the league's.

So: the prior on every game against an opponent, **and** the restarts, together.

**The check, at every read:** the handover test from the exploiter -- the exploiter plays the US's first three
turns, the main then plays both sides; the critic's value at that point against sixteen continuations. The
critic should be within 3 points. **The fallback:** over 3 points in the US's favour -> the rows of the games
against the exploiter leave the value loss (the form of `--value-selfplay-only`, for those games only; R124
showed the policy tolerates it), the prior stays for their advantages, and the arm continues.

### The restarts

The restart pool takes its positions from the games against the exploiter in this arm (a source switch on
`restart_pool`), admitted without the critic's openness test for the first 200M -- the critic misjudging these
positions is the starting condition, not a reason to skip them -- and by P48b's put-back rule after.

If the main's US has not moved by the first read, a small pull of its US play toward the exploiter's on those
positions, fading to zero over 400M, is the one addition; it is not in the first run.

**Reads** at +400M, +800M and +1,600M, per seat from the JSON, beside B6's own continuation:

1. the main as the US against B6 at 11,600M and at 12,400M (now 49%; the target is P49a's robust margin);
2. the main as the USSR against the exploiter it trains with (now about 32%; back to its self-play rate);
3. **the main as the USSR against X, which it never met** (X's US now 68%; under 55% is the pass);
4. the control, J1's best and the old reference (not below B6's continuation at the same step);
5. the critic: the handover test from the exploiter (above; within 3 points), the prior the trainer holds for
   the exploiter and for the old state, and the learned gain on the line's positions beside its value elsewhere;
6. setups, and the share of the main's own US games that carry P49a's signature.

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

### P49a, 2026-11-18 (CPU, greedy, 2,000 deals a cell, each seat its own rule bits; `data/reports/p49a.json`)

X is R136's newest SWA, **11,920-12,000M** (the run's last; it ran to 12,000M, past the 11,520-11,600M named above),
with the 11,520-11,600M SWA on the target as the bridge to the cross-table's 67.8%. VP is + for the US; "Europe
battlegrounds" are the US's controlled ones of France, East Germany, Italy, Poland and West Germany at the first
state of turn 4 (of the games that reach it). ± is one SE.

| X as the US against | US score | US 20 VP (in T6-8) | USSR Europe Control (by T5) | VP end T3 | VP end T5 | US Europe BGs end T3 |
|:---|---:|---:|---:|---:|---:|---:|
| B6 10,800M (−800M) | **68.8** ± 1.0 | 54.8 (28.9) | 14.9 (11.3) | −2.2 | +2.4 | 0.00 |
| B6 11,200M (−400M) | **65.5** ± 1.1 | 49.1 (25.1) | 18.2 (14.1) | −3.1 | +1.5 | 0.00 |
| B6 11,600M, its target | **67.5** ± 1.0 | 50.0 (26.9) | 18.0 (13.6) | −2.2 | +2.1 | 0.00 |
| B6 12,000M (+400M) | **61.3** ± 1.1 | 41.6 (20.3) | 19.2 (13.1) | −3.5 | +0.5 | 0.00 |
| B6 12,400M (+800M) | **56.1** ± 1.1 | 37.1 (18.4) | 19.9 (13.2) | −4.4 | −0.4 | 0.00 |
| B6's soup | **68.3** ± 1.0 | 54.3 (27.8) | 17.5 (13.9) | −2.3 | +2.4 | 0.00 |
| (X at 11,600M) B6 11,600M | 68.8 ± 1.0 | 52.6 (27.0) | 16.9 (13.0) | −1.6 | +2.3 | 0.00 |
| *B6 11,600M against itself* | *49.9* ± 1.1 | *22.4 (10.8)* | *3.2 (1.7)* | *−1.6* | *−1.1* | *2.00* |

1. **The robust margin is not one number: it is the snapshot's age.** Against B6's *earlier* states and the soup X
   scores as on its target (65.5-68.8%); against B6's *later* ones it falls, 61.3% at +400M and 56.1% at +800M.
   Over B6's own 49-50% that is +16 to +19 points backwards and +11 / +6 forwards. The registered expectation
   (57-60% at 400-800M away) holds forwards only. B6 kept moving after 11,600M in a direction that partly
   defends this line (its 12,400M state is also where its J1 read was lowest), while its soup -- made of
   11,520-11,600M states -- has the hole whole.
2. **The signature is the 10,800M one, stronger**, and the same against every B6 state: US 20 VP wins 37-55% of
   games (B6 itself 22%), about half of them in turns 6-8; USSR Europe Control 15-20% (B6 3%), three quarters of
   it by turn 5; VP behind at the end of turn 3 (−2 to −4) and ahead or level at the end of turn 5. **X controls
   no European battleground at the end of turn 3 -- in 1 to 3 games of ~1,745 a cell -- where B6 controls 2.0.**
   The reason is its setup: X's US places nothing in West Germany or France (58% of setups: Canada 2,
   Spain/Portugal 2, Turkey 2, Denmark 1, Greece 1, Italy 1; every one of its 19 distinct setups is of that kind),
   and the 13% of its games that end before turn 4 are the USSR's Europe Control in 84% of cases (211 of 251 on the target). A P49b exploiter
   that finds "the 20 VP race" without that setup has found a different tactic.
3. **The other seat**, from what exists: B6's own USSR exploiter (R135) read +1.4 / −1.6 over its 51.0% bar at
   12,000 / 12,400M; J1's USSR exploiter of B6 (R137) read 46.6% at 10,400M against a 43.0% bar (+3.6) and was
   stopped there. No USSR-seat exploiter has found a hole in B6's US of the US seat's size; none has been run
   long from outside B6's lineage.
