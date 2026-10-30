# P41 — Jaxcalibur's ingredients on the E8 base, one at a time, with training against search repeated on top (2026-10-30)

**Why.** [Jaxcalibur](https://jaxcalibur.github.io/) matched top human players at Gen 9 random-battle
Pokémon with our training algorithm -- pure PPO self-play with GAE, undiscounted win/loss reward, no
search in training, search only at play time (+100–150 Elo, "most of the strength comes from the
network") -- at about four times our compute per run (one H100 for a week, ~100M games) and four
times our parameters (8.5M). It has four ingredients we do not: a zero-avoiding regulariser beside
the entropy bonus, two auxiliary heads about the opponent (its next action, which "made the bot
substantially stronger", and its hidden attributes carried across turns), a token transformer with
position-type-specific MLPs, and the last 16 game events as tokens. Each maps onto a measured gap
here: policy saturation (19–33 nats, the gap penalty's crude target), a trunk that holds only public
information about the opponent's hand (P33f), a residual trunk that collapses where a transformer
over entities does not, and a memoryless policy. This plan tests them one at a time on the best E8
recipe, then repeats the one result of P40 that moved a plateau -- training against a searched
opponent -- on top of whatever they establish.

**Where P40 stands (2026-10-30), which sets the base.** Arm C (the deep network warm-started by
distillation) held its start and did not gain past it; stopped. Arm S (the rollout root as the
opponent in one game in eight, k4 w4 h1 at entry decisions, graphed, hold-and-batch) reads 51.6 /
53.4 / 54.3% at 16,400 / 16,800 / 17,200M from R94's plateau -- the one trend past the band -- and runs
alone. The recipe found on the way is the new from-scratch base: **A8 on R95** (16k minibatch, PPO
clip 0.1, µP's zeroed readouts and per-layer rates at A8's widths) reads **35.8% at 2,000M** against
A13+R48's SWA where the plain control read 27.7%; its control run `E8-A8-R95-S44` continues to its
plateau (10,000M budget) in the next free slot.

## The arms (from scratch on A8-R95, one change each, R codes at launch)

| arm | change | what it maps to here |
|:---|:---|:---|
| **J1** | **the zero-avoiding term**: KL(uniform ‖ π) over the legal moves, i.e. −mean_a log π(a), coefficient c, from step 0, beside the entropy bonus | saturation; the gap penalty (+7 late, −150–240 from scratch) is its crude form. Cells: c small (the entropy coefficient's scale), then 3×. Reads: median top-two logit gap and P(top) (the P37a saturation numbers), the rare-spot card rates (Nuclear Subs, One Small Step, Wargames declined), the Cuba trap suicide rate; strength at matched steps |
| **J2** | **the opponent's next-action head**: at the learner's decision, predict the opponent's next entry decision (its next card play, the flat action) from the observation; cross-entropy at weight w; the target is the rollout's own record | the auxiliary that "made the bot substantially stronger"; here an opponent model the trunk lacks (P33f: public information only) |
| **J3** | **the hidden-hand head**: predict the opponent's hand (sigmoids over the hidden pool) from the observation; the true hand from the engine state as the target (a recipe: the policy never reads it); weight w | their item/ability/move predictions as targets; ours is the belief the trunk does not hold. Read on its own target against the probes' numbers (prior 0.3914, public probe 0.3780, the policy-likelihood posterior 0.3363): does the trunk learn a belief when asked to |
| **J4** | J2 + J3 together | their setting; only if each is at least level alone |
| **J5** | **the token encoder on the R95 recipe**: two attention layers over country and card tokens (the `--ladder-token-layers` path, pre-norm) on A8's heads, with the per-entity-type MLPs the ladder already has (their position-type mixture of experts), µP per-layer rates, the stability checks at 60M | P38 rung 4 with its prior raised: a transformer over entities trained under PPO self-play is their whole architecture, and it did not need hyperspherical machinery |
| **J6** | **history tokens** (the owner's decision): the last 16 decisions of the game -- side, decision type, card or country -- as an appended `obs_features` block of integer codes (16 × 3 floats), embedded network-side; into J5's network, not A8's | their 16 event tokens; the cheapest memory, 48 floats |
| **J7** | **matchup features** (the owner's decision, lower priority): per held card, the engine's event and Ops summary against the board (P30's T2/T4: VP, DEFCON and regional deltas if the event fires, countries the Ops could control), ≤ 9 cards × ~10 floats as an appended block | their matchup tokens: pairwise structure handed over rather than learned |
| **J8** | **training against search, repeated on top**: Arm S's searched opponent (`--searched-opponent-spec 4:4:1:z2:earlier:own:keyed:aligned:box4:graphs`, one game in eight) (a) as a branch from the new base's plateau -- Arm S's form, replicated; (b) at the same share from the start of the best combination, gated as R69 was (search joins when the root beats its raw snapshot ≥ 55% in a 200-game check every 40M) | the one P40 result that moved a plateau; it must be shown to hold on the new base, and whether it helps from the start is untested |
| **J9** | **length**: the best combination run to twice the steps its control plateaued at if still rising | their ~100M games are ~45,000M of our steps; our plateaus are at 12,000M |

## Measure

* **Matched steps against the control** `E8-A8-R95-S44` at 400M windows (80M-window SWAs, raw, 1,000
  deals a side), per seat against the control's own self-play bar; the fixed reference A13+R48's SWA
  beside it; **the own-past gain tracked** (the drift read of P40) so a lineage-specific gain is seen
  for what it is.
* Each arm's own target where it has one: J1's saturation and rare-card rates; J2's next-action
  accuracy on held-out games; J3's hand log loss against the probes.
* The stability note's checks at 60M and the plasticity probe at 400M for every arm; the searched
  reads (under the corrected keyed root) at the plateau only.
* `P40a`, the Czechoslovakia opening priced under the network's own continuation, the searched root
  and the oracle, is read on the J2/J3 and J8 networks: the owner's example is the test of whether an
  opponent model or a searched opponent makes a setup fault cost.

## Decision rules

* An arm **passes** by the per-seat rule against the control at matched steps at 2,000M and holds it
  at 4,000M; a pass enters the base recipe for the arms after it (J1 → J5 → J6 → J2/J3 → J4).
* J1 passes only with the saturation numbers moved (median gap down, rare-card rates up) **and**
  strength not lower; a regulariser that moves the numbers and costs strength is tuned (c ÷ 3) once.
* J3 is read first on its own target: a head that does not beat the public probe (0.378) has not
  taught the trunk a belief, and J4 is skipped.
* J5 passes P38's rule (the gap to its control decreasing at 2,000M with rank ≥ 2× fresh at 60M);
  J6 and J7 go in only on J5's passing network and only with the owner's approval of the block.
* J8a passes if it reproduces Arm S's trend (above the base's plateau band on two consecutive
  400M reads); J8b if it is ahead of J8a's base at matched steps with the own-past gain not
  outrunning the reference gain.
* The from-scratch adoption run is the best passing combination, to its own plateau, J9's length.

## Sequencing (two slots; Arm S keeps one until it reads)

**Order (owner, 2026-10-30): J1 → J5 → J6 → J2 / J3 → J4 → J7 → J8 → J9.** Architecture comes
right after the regulariser because much is likely to depend on it; and **the opponent heads depend
on history**: without a history input the hand head can learn only the public-information prior
(the belief probes: public 0.3780 against 0.3363 for the posterior from this turn's opponent
decisions), and the next-action head only a style prior, so J2/J3 are not run before J6. Jaxcalibur
has both the 16 event tokens and predictions carried across turns; the heads are read on their own
targets only once the network can see the history they need.

| slot 1 | slot 2 |
|:---|:---|
| Arm S to its decision (P40), then the `E8-A8-R95-S44` continuation to its plateau | J1 (2,000M, ~a day beside another arm), then **J5** |
| J8a from the base's plateau once the control has one | **J6** on J5's passing network (the owner's block), then J2, J3, J4 |
| J8b on the best combination | J7 on the owner's word |

Builds, in order of need: the zero-avoiding term (a loss term; the gap penalty's code is the
pattern); µP for the token path and the health checks on it; the history block (16 × 3 integer
codes, embedded network-side, tokens in J5's encoder) once approved; then the two auxiliary heads
with their targets in the rollout buffer (the opponent's next action from the rollout's record; the
opponent's hand from the engine state, never into the observation).

## Not in this plan

Policy distillation of search (closed), value targets from playouts (closed), the belief root as a
product (not adopted), regret matching and HP binning (simultaneous moves; not our game).

## Runs

(filled in while running; mirrors [`../runs.md`](../runs.md))
