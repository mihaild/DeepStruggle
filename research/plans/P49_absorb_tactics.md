# P49 — a main that takes in what exploiters find, in both seats, with no other lineage in its training (2026-11-18)

**Status:** proposed (owner, 2026-11-18); **revised the same day** after P49a: both seats throughout (owner: "we
need something symmetrical, that improves both sides"), the search aimed at setups (P49a found the exploit is
one), and the absorption followed by a low-rate leg. P49a is done (*Runs*); P49a' is two more CPU checks; P49b
is the search for exploiters that need no foreign lineage; P49c the absorption; P49d the loop.
**Gate:** P49c waits for a P49b exploiter that passes in at least one seat.
**Needs approval:** each launch; trainer-side only. Builds are named where they are needed.

## Goal

A network that is **not worse than the no-pool run (`E8.1-A8-R132-S45`, "B6") against anything, and that an
exploiter of either seat gains little against** -- including exploiters it never met. For the one hole known
so far, B6's USSR against the J1-started exploiter **X** (`E8-A8-R102-S44@9990M+R136`): X scores 68% as the US
against B6's 50% in self-play, and the owner's aim is a network that plays B6 about as well as X does and
loses nothing elsewhere.

**The constraint (owner):** nothing started from J1, or from any other finished lineage, enters the new main's
training; an exploiter that needs a fresh 10,000M lineage every few hundred million steps is not a method.
Exploiters started from other lineages are yardsticks only: the known answer the search is compared with, and
the examination the main sits without having met them.

**Both seats (owner):** every step below is run for the US seat and for the USSR seat alike. No part of the
training is fixed to a side; where the two seats end up treated differently it is because the search found
something in one seat and not in the other, and the plan says what happens then.

## Why

Numbers and instruments: [`../log/P49_exploiter_cross_table.md`](../log/P49_exploiter_cross_table.md) and
P49a under *Runs*.

* **X's exploit is a setup.** X's US places nothing in West Germany or France (all 19 of its setups; the
  commonest Canada 2, Spain/Portugal 2, Turkey 2, Denmark 1, Greece 1, Italy 1) and holds no Europe
  battleground at the end of turn 3. B6's USSR, handed Europe, takes Europe Control in 15-20% of games and
  loses the VP race in the rest (US wins by 20 VP in 37-55%, against 22% in B6's self-play).
* **The weakness is B6's USSR not converting an abandoned Europe.** USSRs that do punish hold the line to
  slightly under par (X's US 44% against J1, 47% against the pooled control; their own US 47-49%). So a USSR
  that punishes it is a pure gain; a US that adopts the setup trades 2-3 points against everyone else for the
  gain against B6-like opponents, because a setup is chosen before the opponent is known.
* **It explains why B6's own exploiter found nothing** (52% in 800M): leaving Europe empty is a jump at the
  setup, which step-by-step learning from B6's own play does not make. X started from J1 at 9,990M (42%
  against B6 untrained) and reached 63% in 800M: it had a different place to start, not strength.
* **The margin is partly the snapshot's, forwards only:** X scores 65.5-68.8% against B6's earlier states and
  its soup, 61.3% against B6 400M later and 56.1% 800M later, though B6 never met it.
* **The USSR seat has not been tested at the budget that worked.** B6's own USSR exploiter read +1.4 / -1.6;
  J1's USSR exploiter of B6 was stopped after 410M, before the step at which X took off (X read 47% at 410M
  and 63% at 810M). Whether B6's US has a hole of this size is unknown.
* **Part of B6's plateau is noise:** continued at lr 1e-4 it reads 68.2% on the old reference after 400M (64.4%
  at the full rate; its three-seed soup 68.8%). A lower rate sharpens what a network knows and slows what it
  can take in, so absorption comes first and the low-rate leg last.
* **The machinery exists:** the seat-true league and its driver (P45), the opponent prior with a learned gain
  (P46c), one opponent per game (P47), the restart pool (P48b). **The prior is used in every game the main plays
  against an opponent that is not itself** (P49c says how, and why the restarts are needed beside it).

## P49a — the target and the signature (CPU; done, see *Runs*)

The robust margin, X's signature and what is known of the other seat.

## P49a' — two more checks before any training (CPU)

1. **How much of X is the setup?** B6's own US forced into X's commonest setup (and into its three next
   commonest), B6 playing both sides after the setup, 2,000 deals each, beside B6's own setup. Well above 50%
   -> the exploit is the setup, and a search over setups can find it; near 50% -> it is X's play after the
   setup, and a setup alone will not. The mirror cell: X's US forced into B6's own setup, against B6's USSR.
2. **Does the low rate keep the hole?** X as the US against B6's two low-rate continuations at 12,000M (R139,
   R140), beside 61.3% against the plain run at that step. About 67% -> the low rate freezes the hole in, and
   absorbing first is confirmed; about 61% -> the order matters less.

## P49b — exploiters of B6 from inside its own lineage, one for each seat

**Every arm is run in both seats, the pair in the two slots:** a US exploiter against B6's USSR and a USSR
exploiter against B6's US, the same start, recipe and budget. **Target, for every arm:** B6's eight snapshots
of 11,530-11,600M and their SWA, greedy, one drawn per game (the fixed-opponent list drawn per game is the one
build if `--opponent-per-game` does not yet cover it). X's recipe otherwise (R136's flags, the full rate),
1,600M, reads every 400M.

| arm | the exploiter | why it might leave B6's basin | build |
|:---|:---|:---|:---|
| **S** | starts from B6 at 11,600M; in half its games its own setup is drawn at random from the legal ones, as part of the environment (not credited to the policy), and the critic trains on those games | P49a: the known exploit is a setup, and a setup is what a policy does not explore; once the critic prices setups against this opponent, the policy's own setup follows it, as the setups did in P44 | a random legal setup for the learner's seat as environment (the forced-opening path, with a random draw) |
| E1 | starts from B6's own state at 4,020M | an earlier era of the same run plays differently and scores about what J1 did against B6 before training | none |
| E2 | starts from B6 at 11,600M, paid only for one ending (P41's restricted payoff): for the US, 20 VP; for the USSR, Europe Control, then 20 VP | the reward points at a line | an ending-restricted reward for exploiter runs |
| E3 | starts from B6 at 11,600M with fresh policy heads | the policy is relearned against the target | none if R87's head reset still runs |
| E4 | a new no-pool seed trained to 2,000M | a different run's early style; 2,000M more | none |

**Order:** S in both seats first (about nine hours for the pair); E1 in both seats next; E2-E4 only for a seat
in which neither passed.

**The yardstick for the USSR seat (owner's go needed; it trains a yardstick, not the main):** J1's USSR
exploiter of B6 (R137), continued in place from 10,400M to 11,600M, so that the USSR seat has what X is for
the US seat. Without it a USSR arm that finds nothing cannot be told from a US seat with nothing to find.

**Reads** at each 400M, in the exploiter's seat: against the target (for the US, X's curve 47 / 63 / 67 / 69%;
for the USSR, R137's if it is run); against B6 at 12,400M, which no arm trains on; against the pooled control
and J1's best, as the cost elsewhere; the signature (endings by turn, VP at the end of turns 3 and 5, Europe
battlegrounds at the end of turn 3, the setups it plays).

**Pass, per seat:** 10 points or more over B6's self-play rate in that seat against the target by 1,600M, 5 or
more against B6 at 12,400M, and not more than 4 under the control's and J1's own rates in that seat.
**No arm passes in either seat ->** "a different lineage is needed" stands; P49c runs once with X as a marked
exception only if the owner says so, and the search moves to two no-pool mains of different seeds trained
side by side. **One seat only ->** P49c runs with that seat's exploiter (below).

## P49c — absorption: defence, attack, retention, in both seats

The new main is B6 from its 11,600M state, the no-pool recipe (R132) at the full rate, with three kinds of
game added and nothing from another lineage. 800M, then 800M, then the low-rate leg.

**The critic prior is used: in every game against an exploiter and against the old state** (P46c's form, the
gain learned, as built for R128). It is zero in self-play and in the restarts.

| kind of game | share | who plays | the value the critic predicts | what it teaches |
|:---|---:|:---|:---|:---|
| self-play | 70% | the main, both seats | `sigmoid(c(s))` -- no prior | everything else |
| **against an exploiter** | 15%, half to each seat's exploiter | the P49b exploiter of a seat in that seat, the main in the other; one opponent for the whole game | `sigmoid(c(s) +- g(s) * beta)`, **`beta` the prior**: the main's running score against this exploiter in this seat, in log-odds against its self-play rate there; + on the main's rows, - on the exploiter's | the main's USSR meets the US exploiter's line from the first move, and its US the USSR exploiter's: defence |
| restarts from those games | 10%, from both seats' games alike | the main, both seats, from the turn boundaries of turns 1-6 of the games against the exploiters | `sigmoid(c(s))` -- no prior, the exploiter is no longer playing | what each line's positions are worth, and the main's own play in the exploiter's seat steering into them or not: attack, with no imitation term |
| **against B6's 11,600M state** | 5% | the old state in either seat, the main in the other; one opponent a game | `sigmoid(c(s) +- g(s) * beta)`, **`beta` the prior** for that state and seat | punishing the old weaknesses stays rewarded: retention |

`c(s)` is the critic's win logit and `g(s)` its gain head, both reading the position only; `beta` is the one
number from outside the network, kept by the trainer per opponent and seat as a running average. The shifted
value is the one stored for GAE, bootstrapped from and fitted in the value loss, on every row of the game; the
policy loss takes the main's rows only, as in every game against a fixed opponent.

**If only one seat has an exploiter,** the league holds that one at half the share (7.5%), the other half
stays self-play, and the restarts come from its games; the main's seats then meet different opponents because
of what the search found, and the next generation's search (P49d) starts with the empty seat.

### Why the prior, and why it is not enough by itself

P44-P46's trap was a critic that read "plays unlike the current self" as "loses", because the unfamiliar side
(old pool members) lost. Here the unfamiliar side wins. Without a prior the critic would learn "this side is
playing the exploiter's line, so it wins", and would then credit the main's own play for positions that are
good only in the exploiter's hands.

* **With the prior**, an exploiter's edge is carried by `g(s) * beta`, and `c(s)` is left to say what the
  position is worth between two copies of the main.
* **The prior alone does not guarantee that**, because an exploiter's strength is not one number: it is strong
  where its line is on the board and weak elsewhere. `g(s)` reads the position, so it can carry that -- but in
  the exploiter's games alone nothing says how to split a line position's value between `c` and `g`.
* **The restarts make the split determinate.** They are the same positions with the main on both sides and
  `beta = 0`, so they fit `c(s)` directly, and what is left over in the exploiter's games can only go into the
  gain. They are therefore part of the critic's correctness and not only of the attack: their share is not cut
  below half the league's.

So: the prior on every game against an opponent, **and** the restarts, together.

**The check, at every read, for each exploiter:** the handover test -- the exploiter plays its seat's first
three turns, the main then plays both sides; the critic's value at that point against sixteen continuations.
Within 3 points. **The fallback:** over 3 points in the exploiter's seat's favour -> the rows of the games
against that exploiter leave the value loss (the form of `--value-selfplay-only`, for those games only; R124
showed the policy tolerates it), the prior stays for their advantages, and the arm continues.

### The restarts

The restart pool takes its positions from the games against the exploiters in this arm (a source switch on
`restart_pool`), turn 1's boundary included because the known line is a setup; admitted without the critic's
openness test for the first 200M -- the critic misjudging these positions is the starting condition, not a
reason to skip them -- and by P48b's put-back rule after.

If the main's play in an exploiter's seat has not moved by the first read, a small pull toward that
exploiter's play on those positions, fading to zero over 400M, is the one addition; it is not in the first run.
For a line that is a trade (X's setup costs 2-3 points against opponents that punish it) the main is not
expected to adopt it, and not adopting it is not a failure: read 1 below is then reported, not required.

### Absorb, then the low rate

After the two legs at the full rate, the arm and its control (B6's own continuation over the same steps) each
get the same 400M at lr 1e-4 (R139's recipe). **The product is the arm after that leg, and what it has to beat
is the control after the same leg**, not the plateau network: the rate alone is worth about 4 points on the
old reference.

**Reads** at +400M, +800M, +1,600M and after the low-rate leg, per seat from the JSON, beside the control:

1. the main in each seat against B6 at 11,600M and at 12,400M (now 50% as the US, 50% as the USSR);
2. the main against each exploiter it trains with, in the seat opposite it (for the US exploiter now about
   32%; back to the main's self-play rate in that seat);
3. **the main against the held-out yardsticks:** as the USSR against X (X's US now 68%; 55% or under is the
   pass), and as the US against R137's continuation if it was run;
4. the pooled control, J1's best and the old reference (not below the control arm at the same step);
5. the critic: the handover test from each exploiter, the priors the trainer holds, and the learned gain on
   the lines' positions beside its value elsewhere;
6. setups in both seats, and the share of the main's own games that carry an exploiter's signature.

**Prediction.** Read 2 within 3 points of the self-play rate by +400M; read 3 at 55% or under by +800M; read 4
level; read 1 above 50% only for lines that are not a trade.
**Rule.** Reads 2-4 met -> the method works for the seats it was run in; P49d. Read 3 not met while read 2 is ->
the main learned its sparring partner and not the line; the exploiter's lineage is continued before it is a
teacher (P49d). Read 4 down by 2 points or more -> the shares are halved once.

Afterwards, one at a time and only if the arm worked: without the restarts, and without the retention
opponent, to learn which of the three additions carried it.

## P49d — the loop, and when it stops

* **A generation** is 1,600M of the main beside two exploiter legs in the other slot, 800M for each seat: each
  seat's exploiter **continued** against the new main (its base is paid for once), or, for a seat that has
  none, the next P49b arm. The newest two exploiters of a seat stay in the league; older ones leave.
* **The yardsticks are continued 800M against the final main and never fed back:** what X (and R137) still
  find is the result.
* **Stop** after two generations in which no exploiter of either seat gains 3 points or more over the main's
  self-play rate there and the references do not move; then the low-rate leg.

## Cost

| | steps | about |
|:---|---:|:---|
| P49a' | none | a few CPU minutes |
| P49b, arm S in both seats | 3,200M | 9 hours |
| P49b, the USSR yardstick (R137 to 11,600M) | 1,200M | 3.5 hours alone |
| P49c, two legs beside the control's | 1,600M + 1,600M | 9 hours |
| the low-rate leg, arm and control | 400M + 400M | 2.5 hours |
| P49d, one generation | 1,600M + 1,600M | 9 hours |

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
