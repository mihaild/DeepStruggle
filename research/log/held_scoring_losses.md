# Held-scoring losses and the zero-avoiding term (2026-11-06)

**Conclusion.** The zero-avoiding term (R102, J1 and J5) roughly doubles held-scoring losses in training games, and it does
so by sampling noise at the forced decision, not by misjudgement. J1's twin without the term (R95) loses this way in 2.3-3.4%
of training games at every matched step, J1 in 4.1-6.3%. At the action-round card choices where the scoring cards held
equal the rounds left, R95 puts 100.00% on a scoring card at the median and below 99% in 11% of them; J1 and J5 sit at
99.7-99.8% at the median and below 99% in ~19% -- the term's floor on the other cards. J5's losses come with 82% on the
scoring card at the holder's last choice (a sample of the remaining 18% loses the game); R95's come at 46%, half of them
deliberate. Holding more scoring cards at the turn's start raises the risk (~0.3% / 1% / 2% per player-turn for 1 / 2 / 3+),
but half the losses start the turn with a single scoring card. In 70-87% of losses the held card would have cost the holder
VP (median -3 to -5), and in 14-19% scoring it would have lost by 20 VP anyway.

## Training games at matched steps

Held-scoring share of all training games (`training_metrics.jsonl`, T=1 self-play and pool games):

| steps | J1 (R102, term) | R95 (J1 without the term) | A8 control (R1) | J5 (A35-R102, term) |
|:---|---:|---:|---:|---:|
| 0-100M | 8.3% | 5.8% | 6.1% | 5.6% |
| 300-400M | 4.7% | 3.6% | 4.0% | 5.1% |
| 700-800M | 4.9% | 3.2% | 3.1% | 4.9% |
| 1,900-2,000M | 6.3% | 3.4% | 2.5% | |
| 3,900-4,000M | 4.9% | 2.4% | 2.5% | |
| 7,500-7,600M | 4.4% | 2.3% | 2.8% | |
| 12,000-12,400M | 4.5% | | 2.5% | |

## The decision (T=1 self-play, 2,048 games per network)

`data/reports/held_scoring_probe_J5_J1_R95.{json,txt}` (`held_probe2.py`; J5 at 830M, J1 and R95 at 2,000M). Rounds
left = 6 or 7 (8 with a space-station AR8; North Sea Oil not counted) minus the action round, the current one included.

| | J5 830M | J1 2,000M | R95 2,000M |
|:---|---:|---:|---:|
| held-scoring losses | 4.1% (US 50, USSR 33) | 6.0% (65 / 57) | 3.3% (41 / 27) |
| P(loss this turn) with 1 / 2 / 3+ scoring cards at the turn's start | 0.29 / 0.97 / 2.0% | 0.47 / 1.38 / 2.6% | 0.19 / 0.64 / 3.1% |
| losses by scoring cards at the turn's start, 1 / 2 / 3+ | 37 / 35 / 10 | 60 / 49 / 13 | 25 / 25 / 16 |
| held = rounds left: mean / median P(scoring) | 98.2 / 99.72% | 95.2 / 99.77% | 92.3 / 100.00% |
| held = rounds left: share of choices with P(scoring) < 99% | 18.8% | 19.5% | 11.0% |
| losses: P(scoring) at the holder's last choice | 81.6% | 54.2% | 45.9% |
| losses: VP the held card(s) would give the holder (median; share < 0) | -3; 70% | -5; 84% | -4; 87% |
| losses: scoring would have lost by 20 VP anyway | 14% | 19% | 18% |

Nearly every loss happens at the holder's last action round with one scoring card in hand (J5 82 of 83). R95 places a
scoring card with near certainty when it means to and its non-scoring choices at those nodes are deliberate (near-zero P),
often when the scoring would cost it; J1 and J5 keep a floor of probability on the other cards even when sure, and at T=1
that floor is drawn. Under greedy play the floor costs nothing; it costs training games, a few per hundred, whose outcome
is a sampling accident.

## Does the non-scoring probability match the term's floor?

The term c KL(uniform || pi) (c = 0.01) pulls every legal move's logit toward 1/n with gradient c(1/n - pi_b); against a policy
gradient pushing a bad move down at pi_b * Delta (Delta its advantage gap in normalised units) it settles at
pi_b ~ c / (n Delta), a total of ~c / Delta on the bad moves whatever n. `data/reports/held_scoring_floor_J5_J1_R95.{json,txt}`
(`held_floor.py`; 1,024 T=1 games per network): at the forced choices (scoring cards held >= rounds left), the probability of
each non-scoring legal card, against each network's own floor (the smallest legal-card probability at all its card choices):

| n legal cards | J5 floor / forced per-card median | J1 floor / forced per-card median | R95 floor / forced per-card median | c/n |
|---:|---:|---:|---:|---:|
| 2 | 12.4% / 0.198% | 10.0% / 0.183% | 0.174% / 0.000% | 0.50% |
| 3 | 3.73% / 0.167% | 2.62% / 0.123% | 0.006% / 0.000% | 0.33% |
| 4 | 2.49% / 0.127% | 1.70% / 0.133% | 0.005% / 0.000% | 0.25% |

* **The bulk matches the floor.** Without the term (R95) the forced non-scoring cards sit at 0.000% at the median; with it,
  at 0.13-0.20% each, falling with n as c/(n Delta) predicts, for an implied Delta ~ 2 (playing the scoring card against
  losing the game); the total median mass is 0.2-0.35%. The networks do push these cards far below their usual minimum
  (10-60x under the floor of their ordinary card choices); they cannot push them to zero.
* **It accounts for under one point.** ~2.6 forced choices per game x ~0.28% gives ~0.7% of games lost to the floor alone --
  J5 against R95 in the first probe is 4.1 against 3.3% (+0.8). J1's larger excess at 2,000M (6.0%) comes from a tail of
  forced choices with real non-scoring mass (mean mass 4.8%, 5.7% of them above 20%; J5 2.2%), and R95 has such a tail too
  (7.9% above 20%) yet loses least -- so part of what this classifies as forced is not (North Sea Oil's eighth round is not
  counted; an opponent's event may take the card; the game may end first), and the tail is not readable as risk.

## The clean case: the US's last action round, no eighth round

`data/reports/held_scoring_last_us_ar.{json,txt}` (`held_last_us.py`; 3,072 T=1 games per network): the US's AR6 (turns
1-3) or AR7 (turns 4-10, neither North Sea Oil nor the space-station AR8) holding a scoring card, so the scoring card must be
played now or the turn is lost. Quagmire is tagged: the engine's trap mask offers the scoring card when the scoring cards held
are at least the rounds left or nothing is worth 2 Ops to discard (`action_mask.cpp`), and it did at every trapped node.

| one scoring card, no Quagmire | J5 830M | J1 2,000M | R95 2,000M |
|:---|---:|---:|---:|
| nodes | 3,288 | 3,507 | 3,165 |
| P(scoring): mean / median | 98.24 / 99.717% | 96.83 / 99.737% | 98.16 / 100.000% |
| per non-scoring card P, median (n = 2 / 3 / 4) | 0.214 / 0.169 / 0.223% | 0.189 / 0.157 / 0.130% | 0.000 / 0.000 / 0.000% |
| non-scoring mass > 5% (n = 2 / 3) | 4.8 / 8.6% | 6.6 / 9.2% | 2.0 / 4.0% |
| played a non-scoring card | 1.76% | 3.11% | 1.71% |
| ... and lost the game that turn | 48 of 58 (1.46% of nodes) | 89 of 109 (2.54%) | 33 of 54 (1.04%) |
| lost after playing the scoring card | 0 of 3,230 | 0 of 3,398 | 0 of 3,111 |

The non-scoring plays that did not lose ended the game inside that round (R95 21 of 54 -- deliberate winning plays). With
Quagmire (19-27 nodes per network) the scoring card was legal every time; J5 discarded instead 3 times of 27, all lost.
Two or more scoring cards at the last round (10-18 nodes) lose whatever is played.

So in the clean case the term's floor is visible (0.13-0.22% per non-scoring card with the term, 0.000% without) and costs
J5 about 0.4 points per such node over R95 (1.46 against 1.04% lost) -- about the size of the floor's median mass
(0.21-0.37%). The rest of the losses, in all three networks, come from a few percent of nodes where the network puts real
weight on another card; J1 at 2,000M has the most of those.
