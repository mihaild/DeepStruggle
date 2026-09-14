# Europe control and held scoring, E3-15/E3-17

Two questions off the E3-17-22 (0 graph layers, 160M) TensorBoard: the USSR win rate rising to
90%, and held-scoring losses appearing to rise for the USSR while falling for the US.

## Held scoring is falling on both sides -- the per-side series is a share, not a rate

`won_us/endgame/ending_held_scoring` is the fraction of *US wins* that ended that way, so the
holder was the USSR. Read as shares the two sides do diverge, which is what the chart shows:

| steps (M) | share of US wins | share of USSR wins |
|:---|---:|---:|
| 60-80 | 3.93% | 5.78% |
| 100-120 | 5.20% | 4.78% |
| 140-160 | **5.76%** | **2.13%** |

Multiplied by each side's win rate, so that both become per-game rates, the divergence is gone:

| steps (M) | US win% | USSR win% | USSR strands | US strands |
|:---|---:|---:|---:|---:|
| 60-80 | 53.5% | 46.5% | 2.11% | 2.69% |
| 100-120 | 24.2% | 75.8% | 1.26% | 3.63% |
| 140-160 | 9.6% | 90.4% | **0.55%** | **1.92%** |

Both fall. The rising share is a denominator effect: US wins collapse from 53% to 9.6%, and
held-scoring becomes a larger slice of a much smaller set -- by 160M a USSR scoring blunder is
close to the only route the US has left to a win. **Per-side ending-mix series are shares of that
side's wins; they are not comparable across steps unless the win rate is held fixed.**

## Which scoring cards get stranded: no card-identity effect

2048 games per arm at temperature 1.0 (`ai/eval/held_scoring.py`), over-representation being the
share of strandings divided by the share of time the card spent in a hand.

| arm | rate | loser US/USSR | most over-represented | Europe | Asia | Mid East |
|:---|---:|:---|:---|---:|---:|---:|
| E3-15-21 80M | 15.0% | 89 / 219 | Europe 1.36 | 1.36 | 0.90 | 0.80 |
| E3-17-21 80M | 8.2% | 72 / 96 | SE Asia 1.96 (16 events) | 1.05 | 0.89 | 0.90 |
| E3-17-22 160M | 5.0% | 64 / 38 | Asia 1.76 | 0.63 | 1.76 | 0.74 |

Nothing replicates. Each arm's top card is a different card, and the three high-exposure cards sit
between 0.63 and 1.76 with no consistent ordering; the extreme values all sit on cards with a
handful of events. The hypothesis that the six scoring cards are confusable because they share
their feature vectors, leaving only the identity embedding to separate them, is **not** supported
by where the losses land. (The E3-17-22 Asia figure is 4.5 sigma within its own run, so it is real
for that arm -- it is simply not a property of the architecture.)

What does replicate: the rate tracks training maturity (15.0% -> 8.2% -> 5.0%), and **which side
strands flips with it**. The USSR strands far more early (219 vs 89 at E3-15-21 80M) and the US
more once trained (64 vs 38 at 160M) -- the same flip the per-game table above shows.

## The US does not lose West Germany; it never goes there

Every generated USSR Europe-control win has the same shape. Final European battlegrounds in the
three captured games (`data/replays/E3-17-22-europe-{206,208,214}.tslog.json`):

| | West Germany | France | Italy | East Germany | Poland |
|:---|:---|:---|:---|:---|:---|
| seed 206, turn 7 | **0/4** | 1/4 | 6/8 | 0/3 | 1/4 |
| seed 208, turn 8 | **0/7** | 5/8 | 4/6 | 0/3 | 0/3 |
| seed 214, turn 5 | **0/6** | 5/8 | 2/4 | 0/3 | 0/4 |

France and Italy are genuinely fought over. West Germany is US **0** in all three -- the region is
completed without a fight there, so it is the gateway.

### Forcing the standard human opening

`ai/eval/forced_setup.py` overwrites the fifteen setup decisions with USSR +1 East Germany, +4
Poland, +1 Yugoslavia / US +4 West Germany, +3 Italy, +2 Iran, then hands control back to the
policy. 2048 games per arm, E3-17-22 @160M, temperature 1.0:

| | free setup | forced human opening |
|:---|---:|---:|
| US win rate | 14.4% | **14.4%** |
| Europe-control endings | 11.2% | 10.2% |
| West Germany, US influence at end | 0.75 | **2.58** |
| US has 0 in West Germany | 57.1% | **32.8%** |
| US controls West Germany | 3.2% | **35.8%** |
| USSR controls West Germany | 38.7% | 21.7% |

**The US can hold West Germany when it is handed it** -- control at the end goes from 3.2% to
35.8% -- so this is not an inability to fight for the country. It is an opening blind spot: the
policy fights where it was placed and does not open a new front.

**And fixing the opening changes nothing that matters.** The win rate is identical to the decimal,
and Europe-control endings move 11.2% -> 10.2%, within noise at n=2048. So West Germany is a
symptom, not the cause: the USSR's edge is not located in the opening, and handing the US the
human setup does not recover it. Worth remembering against the intuition that the opening is where
a policy this lopsided must be going wrong.

Caveat on scope: one checkpoint, one arm, self-play against itself. A forced opening also puts the
policy off its own training distribution, which is an argument for reading the West Germany
columns (a direct behavioural question) rather than the win rate (a strength comparison between
distributions it was not trained on).

## How the US actually loses West Germany: Blockade, every time

The forced-opening numbers say the US *can* hold West Germany but that holding it does not change
the win rate. Watching the games it still loses says why. Four USSR Europe-control wins played
from the forced human opening (`--opening human`, seeds 305/308/314/316, temperature 0.1,
E3-17-22 @160M) -- `data/replays/E3-17-22-human-open-europe-*.tslog.json`.

In all four the US starts with 4 in West Germany, builds it as high as 6, and then loses **all of
it in a single step**. The step is always a US action, and the card is always the same one:

| seed | collapse | US influence | card the US played | US hand at that moment |
|:---|:---|---:|:---|:---|
| 305 | T1 AR6 | 4 -> 0 | Blockade (#10) | Blockade (1), Nasser (1) |
| 308 | T3 AR5 | 5 -> 0 | Blockade (#10) | Blockade (1), Independent Reds (2) |
| 314 | T2 AR6 | 4 -> 0 | Blockade (#10) | Blockade (1), Romanian Abdication (1) |
| 316 | T3 AR6 | 6 -> 0 | Blockade (#10) | Blockade (1), Olympic Games (2) |

Blockade is a 1-Ops USSR card: *"Unless the US immediately discards a card with an Operations
value of 3 or more, remove all US Influence from West Germany."* Playing an opponent's card for
Ops fires its event regardless, so the US is triggering this itself -- and in every one of the four
games its hand at that moment was two cards, neither of them 3+ Ops, so the ransom could not be
paid.

**The ransom was payable, repeatedly, and the policy waited until it was not.** Tracking every AR
at which the US held Blockade alongside a 3+ Ops card:

| seed | first held | ARs at which the discard was available | best discardable Ops |
|:---|:---|---:|---:|
| 305 | T1 AR0 | 6 | 4 |
| 308 | T2 AR6 | 4 | 3 |
| 314 | T1 AR6 | 7 | 4 |
| 316 | T1 AR6 | 11 | 4 |

So the policy treats Blockade as cheap end-of-turn filler -- a 1-Ops card to dump when nothing
better is left -- and by the time it dumps it, the hand can no longer pay. It is not being
outplayed for the country; it is handing it over, on its own action, with the counterplay in hand
for four to eleven action rounds beforehand.

**And it never comes back.** Across all four games, after the collapse the US takes exactly *one*
action naming West Germany (seed 308, T4 AR5) and that one removes the USSR's influence rather
than placing its own. US influence there is 0 for the remainder of every game, while the USSR
walks it up to 4-5 unopposed. This is the "fights where it was placed, never opens a new front"
behaviour in its sharpest form: the country is not merely unprioritised at setup, it is treated as
gone once lost.

Two things this points at, neither of them the opening:

* **Card-level: the event cost of an opponent's card is not being priced.** Blockade is 1 Ops and
  the policy plays it like 1 Ops. A card whose event costs a battleground is not a filler card,
  and the discard that cancels it is a decision the policy never makes.
* **Positional: influence already lost is not re-contested.** Zero rebuild attempts across four
  games is not a tuning issue.

Caveat: four games from one checkpoint, chosen *because* they ended in Europe control, so this
says what goes wrong in those games, not how often. The Blockade unanimity across four
independently sampled seeds is what makes it worth naming; the frequency is not measured here.

## What the critic says while Europe is being lost (probe-208)

`E3-17-22-europe-208.tslog.json`, USSR Europe-control win at turn 8, read with
`ai/eval/replay_critic.py`. The replay is re-driven through the engine from its seed and every
reconstructed state is checked against the snapshot the replay recorded -- **0 mismatches across
all 424 steps**, so the positions below are the positions that were played.

Both value heads are perspective-aligned, so the same state evaluated from both sides is a
consistency check with a known answer: a zero-sum critic must satisfy `v(US) = -v(USSR)`.

Note on units: `v_vp` is **normalised to [-1, 1]**, not real VP. The `forward()` docstring in
`ai/models/coldwar_net_v2.py` says `[-20, 20]`, which is stale -- `_value_scalars` divides by
`VP_LIMIT` and `rollout_buffer` stores `returns_vp` in [-1, 1]. Multiply by 20 for VP.

### The win head does not move while the game is decided

| step | T/AR | VP | USSR-held European BGs | v_win US | v_win USSR | residual | v_vp US |
|---:|:---|---:|:---|---:|---:|---:|---:|
| 25 | T1/1 | -2 | E.Ger, Pol | -0.834 | +0.803 | -0.031 | -1.3 |
| 125 | T2/6 | -9 | W.Ger, E.Ger, Pol | -0.846 | +0.865 | +0.019 | -9.7 |
| 225 | T4/3 | -9 | W.Ger, Ita, E.Ger, Pol | -0.771 | +0.814 | +0.043 | -8.7 |
| 300 | T5/4 | -9 | **all five** | -0.761 | +0.654 | **-0.108** | -9.6 |
| 350 | T6/4 | -15 | all five | -0.791 | +0.777 | -0.014 | -15.6 |
| 424 | T8/0 | -20 | all five | -0.772 | +0.802 | +0.030 | -19.4 |

At step 25 -- turn 1, VP -2, the USSR holding nothing but its own starting countries -- the critic
already reads -0.834. At step 424, with the USSR holding **every** European battleground and the
game ending on exactly that, it reads -0.772: *less* negative than on turn 1. From step 300 onward
the winning configuration is on the board for four turns and the win head never responds.

The VP head, by contrast, tracks the VP track closely (-9.7 vs -9, -15.6 vs -15, -19.4 vs -20).
It is a good readout of the score and not a forecast of the ending.

### Italy: the critic cannot resolve a European battleground

At step 209 the USSR has just broken Italy to 3/3 and the US holds Nixon Plays the China Card
(2 Ops). Italy is stability 2, so two points retake control. The policy played the event
(p = 0.516 EVENT, 0.224 OPS, 0.260 SPACE) -- and the China Card is in `ONGOING_EVENT`, held by
neither player, so the event's transfer clause does nothing here.

Holding the card and the play mode fixed and varying only where the two Ops go:

| line | Italy after | v_win US | v_vp US |
|:---|:---|---:|---:|
| 2 into Panama | 3/3 -- | **-0.808** | -9.4 |
| 2 into Thailand | 3/3 -- | -0.821 | -10.3 |
| 2 into Greece | 3/3 -- | -0.823 | -10.3 |
| 1 Italy + 1 France | 4/3 -- | -0.824 | -10.4 |
| 2 into France | 3/3 -- | -0.824 | -10.4 |
| **2 into Italy -> US control** | **5/3 US** | **-0.828** | -10.3 |

Taking control of a European battleground ranks **last of six**, and dumping two influence into
Panama ranks first. But the honest reading is not "the critic thinks Italy control is bad": the
whole spread is **0.020 v_win**, and the critic's own zero-sum residual has **sd 0.153** over
3,200 sampled self-play states. The spread is an order of magnitude below the model's own noise
floor. **The critic cannot resolve the value of a European battleground at all**; the sign is not
meaningful at this resolution.

### Is the win head collapsed? No, but it is concentrated

Over 3,200 states sampled from 256 self-play games:

* `v_win` US-perspective: mean -0.732, **sd 0.225**, range [-0.918, +0.910]
* **83.3%** of US-perspective values fall in [-0.9, -0.7]
* correlation of `v_win` with current VP: **+0.342**
* zero-sum residual `v(US) + v(USSR)`: mean +0.010, sd 0.153, max |1.676|

So the head has learned the base rate -- the US does lose about 90% of these games -- and
discriminates weakly within it. That is what makes the Italy result what it is: not a wrong
preference, an absent one.

### Asymmetry

The residual is centred (mean +0.010) with sd 0.153. The largest excursions in this game are at
steps 300 and 325 (-0.108, -0.102), exactly where the USSR completes all five European
battlegrounds: the USSR-perspective value *drops* to +0.654 while the US-perspective value stays
at -0.761. The winning side reading the winning configuration as worse is the right shape for the
blind spot above.

**But this is ~0.7 sd of the residual distribution, from one game.** It is suggestive and it is
consistent with the Italy result; it is not on its own evidence of a systematic side asymmetry.
Testing that needs the residual measured against European-control state across many games, which
has not been done.

## The policy wants Europe; the critic never prices it

The critic section above says the value heads cannot resolve a European battleground. The policy
is a separate head, and at the same decisions it is emphatic. From probe-208, each move's own
probability under the acting side, and the one-ply critic delta it buys for the mover:

| step | move | own p | rank | uniform | critic dv |
|---:|:---|---:|:---|---:|---:|
| 207 | USSR breaks Italy | 0.334 | 1 / 51 | 0.020 | -0.003 |
| 217 | USSR breaks France | 0.058 | 1 / 51 | 0.020 | **-0.038** |
| 219 | **US** recontrols France | 0.089 | **3 / 12** | **0.083** | -0.009 |
| 264 | USSR takes France control | **0.919** | 1 / 47 | 0.021 | -0.045 |
| 280 | USSR recovers East Germany | 0.104 | 1 / 49 | 0.020 | -0.105 |

Two things at once.

**The policy overrides the critic, and is right to.** Every USSR Europe move has a *negative*
one-ply critic delta -- at step 217 the alternatives all sit at +0.002 while France is at -0.038 --
and the policy plays France first anyway, at p=0.919 by step 264. This is not a contradiction:
the policy is updated by advantage accumulated over episodes, and GAE transmits the terminal
return whether or not V localises it. The policy learned from *outcomes* that taking Europe wins;
the critic never learned to *price* the position. Note also that most of these deltas are far
below the critic's own zero-sum residual (sd 0.153), so "the critic disagrees" is mostly the
critic having no opinion.

**The US at step 219 is the whole asymmetry in one row.** Recontrolling France ranks 3rd of 12
legal actions at p=0.089 against a uniform 0.083 -- the US policy is indistinguishable from
random about retaking a European battleground, while the USSR is at 0.919 about taking one.

### Where the asymmetry comes from (`ai/eval/europe_attention.py`)

Both sides are the same weights on a perspective-aligned observation, so this needs a mechanism.
Over ~40,000 POINT_NODE decisions from 256 self-play games:

**Not legality.** The US can legally place in West Germany in **58.1%** of its placement
decisions -- *more* often than the USSR (45.7%). It is not locked out of its own battleground.

**Not a late-game artifact.** Probability mass on re-contesting, when the opponent controls the
country, bucketed by turn (n in brackets):

| France | T1 | T2 | T3 | T4 | T5 | T7 |
|:---|---:|---:|---:|---:|---:|---:|
| US retaking | 0.021 (16) | 0.036 (85) | 0.033 (261) | 0.018 (259) | 0.017 (271) | 0.019 (208) |
| USSR taking | 0.172 (1179) | 0.136 (1163) | 0.152 (1021) | 0.100 (694) | 0.083 (449) | 0.068 (157) |

The gap is 4-8x at *every* turn, including turns 3-5 where the US has hundreds of such decisions
and plenty of game left. It is a preference, not a position.

**The country that decides these games is symmetric -- and symmetrically ignored.** West Germany,
same conditional: US 0.015-0.047 across turns, USSR 0.018-0.041. **Neither side re-contests it.**
Italy, by contrast, the US does fight for (0.151 at T1 against the USSR's 0.189).

So the dynamic is not "the USSR is more aggressive". It is:

1. Re-contesting an opponent-controlled battleground is uniformly unattractive to this policy --
   consistent with the 2-Ops-per-point cost and with a critic that cannot price control at all.
2. Therefore **whoever establishes control first keeps it**, and West Germany is decided early.
3. France and Italy, which the US *does* start with, are where the remaining asymmetry lives.

**A mechanism the observation makes available.** Board slots 8/9 (superpower adjacency) are
perspective-relative -- `superpower_adjacent == my_player` / `== opp_player` -- but slots 16-18
(`in_western_europe`, `in_eastern_europe`, `in_southeast_asia`) are **absolute**: Western Europe
reads identically to both players. A policy keyed on the absolute flag -- "put influence into
Western Europe" -- is aggression from the USSR and redundancy from the US, which usually already
holds those countries. That is consistent with `p | legal` being nearly equal for France
(US 0.131, USSR 0.126) while the *conditional on opponent control* diverges 6x.

This is a mechanism the representation permits, shown consistent with the numbers. It is not
established as the cause: that would need an intervention on those slots, which is an observation
change and therefore not mine to make.

## Correction: the regime split, and where the asymmetry really is

The section above conditioned on "the opponent **controls** the country". That is the wrong
regime for the question being asked. `get_influence_cost` charges 2 Ops per point only in a
country the opponent controls; once the USSR has *broken* Italy to 3/3 nobody controls it, and
the US pays the normal **1 Op per point** to take it back. So "the USSR pays double to break
while the US will not pay single to restore" is a comparison between two different regimes, and
the earlier `opp_control` numbers do not measure the second half of it.

The west/east Europe hypothesis is also withdrawn. `in_western_europe` / `in_eastern_europe` serve
a handful of card events and nothing in scoring -- Europe is one region for scoring -- and the
earlier data already argued against it: `p | legal` for France was 0.131 (US) against 0.126
(USSR), which is not what a side-specific regional pull looks like.

Re-measured with `measure_by_regime`, splitting contested into *empty* (opening a front) and
*both present* (restoring a control the opponent just broke), which is the situation at issue:

| country | side | I hold (1/pt) | empty (1/pt) | **both present (1/pt)** | opp holds (2/pt) |
|:---|:---|---:|---:|---:|---:|
| West Germany | US | 0.023 | 0.043 | **0.040** | 0.027 |
| West Germany | USSR | 0.008 | 0.093 | **0.151** | 0.028 |
| France | US | 0.093 | 0.194 | **0.099** | 0.021 |
| France | USSR | 0.030 | 0.138 | **0.206** | 0.132 |
| Italy | US | 0.046 | 0.193 | **0.240** | 0.098 |
| Italy | USSR | 0.029 | 0.349 | **0.480** | 0.182 |

In the **identical** regime -- both sides present, neither controls, 1 Op per point -- the USSR is
**2.0x (Italy), 2.1x (France) and 3.8x (West Germany)** more willing to take the country than the
US is. Same cost, same board condition, same weights.

The literal form of the claim, "USSR paying 2 Ops beats US paying 1 Op", holds for **France**
(0.132 breaking against 0.099 restoring) and not for Italy or West Germany. The same-regime
comparison above is the stronger and cleaner statement.

Note also the `I hold` column: the US consistently puts *more* mass on countries it already
controls than the USSR does (France 0.093 vs 0.030, West Germany 0.023 vs 0.008). The behavioural
signature is **the US consolidates, the USSR expands**.

### It is acquired, not structural

The same measurement across the run, as the ratio USSR/US in the both-present regime:

| country | 40M | 80M | 160M |
|:---|---:|---:|---:|
| West Germany | **0.43** | 0.69 | **3.82** |
| France | 0.70 | 0.74 | **2.14** |
| Italy | 1.02 | 1.29 | **1.94** |

At 40M the asymmetry runs the *other way* -- the US was 2.3x more willing than the USSR to take a
contested West Germany. It inverts between 80M and 160M, exactly the window in which the
Europe-control ending rate went from near-zero to 15%.

And the US side does not merely fail to keep up; it **regresses in absolute terms**:

| country | US mass, 40M | 80M | 160M |
|:---|---:|---:|---:|
| West Germany | 0.045 | 0.027 | 0.041 |
| France | 0.172 | 0.177 | **0.096** |
| Italy | 0.328 | 0.349 | **0.248** |

while the USSR's rises (West Germany 0.019 -> 0.019 -> **0.156**).

So this is not a representational blind spot and not a missing feature. It is a self-play
co-adaptation failure: one network, and at a 90% USSR win rate the advantage signal is dominated
by USSR-side trajectories, so US-side behaviour gets little useful gradient and drifts. That
reframes the earlier recommendation -- the question is not whether the US has had *enough* steps
to learn the counter, but whether a training signal in which it wins 10% of games can teach it
anything at all.

## The advantage signal collapses, and that is the actual pathology

Two hypotheses for the US-side regression were tested against a real rollout
(`RolloutBuffer.diagnostics`, now reporting per-side pre-normalisation statistics; 256 envs x
128 steps, ~33k transitions per checkpoint).

**Hypothesis 1, that the shared advantage normalisation biases the sides: refuted.**
`rollout_buffer.compute_gae` normalises with one mean and one std over both sides' transitions
together. Measured, the two sides are already centred and equally spread:

| checkpoint | US mean | USSR mean | US std | USSR std | std ratio |
|:---|---:|---:|---:|---:|---:|
| 40M | +0.023 | +0.024 | 0.252 | 0.269 | 1.07 |
| 80M | -0.009 | +0.020 | 0.192 | 0.213 | 1.11 |
| 160M | +0.013 | -0.003 | 0.049 | 0.051 | 1.04 |

So **Role-conditioned Advantage Estimation would fix nothing here.** RAE addresses a scalar EMA
baseline that cannot represent two roles; this codebase has a learned critic reading a
perspective-aligned observation, which already centres each role. The published result does not
transfer, and the measurement is what says so.

**Hypothesis 2, that the signal vanishes: confirmed, and it is severe.**

| checkpoint | explained variance | advantage std (pre-norm) |
|:---|---:|---:|
| 40M | 0.740 | 0.257 |
| 80M | 0.789 | 0.211 |
| 160M | **0.996** | **0.049** |

By 160M the critic explains **99.6%** of the return variance and the true advantage spread has
collapsed **5x**. With the outcome 90/10 the winner is predictable from early in the game, so
`A = R - V` goes to zero almost everywhere. The per-rollout normalisation then divides by that
tiny std and rescales everything back to unit variance -- so whatever residual *noise* the GAE
estimate carries is amplified to the magnitude the real signal had at 40M, and the policy update
consumes mostly noise.

This explains both halves of the behavioural finding. The USSR sits at a local optimum that keeps
winning and needs no gradient to stay there; the US has no gradient holding it in place, so it
drifts -- which is exactly the profile measured earlier (France 0.172 -> 0.177 -> 0.096).

**The existing diagnostic cannot see this.** `adv_frac_near_zero` is computed *after*
normalisation, so it reads 0.012, 0.012, 0.014 across the three checkpoints -- flat and healthy
looking -- while the underlying signal falls by 5x. `adv_std_raw` was already recorded and is the
number that shows it. Per-side means and stds are now recorded too.

### What this implies for the remedy

It reframes opponent-pool methods. Sampling opponents you lose to is pointless in pure self-play
against the current policy -- the US already faces the strongest USSR there is. The reason a
league or historical-snapshot pool would help is different: **opponent diversity makes the outcome
less predictable, which restores advantage variance.** A US that sometimes faces a 40M USSR
without the Europe attack has games it can win, and `A` becomes non-zero again.

That also predicts the cheapest check: if `adv_std_raw` recovers when a fraction of games are
played against older snapshots, the mechanism is right.

## Correction: the zero-sum residual is not "pure model error"

An earlier section called `v(s, US) + v(s, USSR)` pure model error. That is wrong as stated.
Twilight Struggle is an imperfect-information game: each perspective sees its own hand, so the
two evaluations condition on **different information sets**, and a nonzero residual is expected.
Holding a scoring card for a region you dominate genuinely raises your value without lowering the
opponent's estimate, because they cannot see it.

Whether that is what *this* residual is made of is measurable (`ai/eval/value_residual.py`,
10,240 sampled states from 256 self-play games, E3-17-22 @160M). If the residual were
information, it should grow with how much is hidden:

| total cards in both hands | n | mean residual | mean abs residual |
|---:|---:|---:|---:|
| 0-2 | 373 | +0.016 | 0.056 |
| 3-5 | 2043 | +0.014 | 0.059 |
| 6-8 | 2018 | +0.019 | 0.060 |
| 9-11 | 2037 | +0.019 | 0.060 |
| 12-14 | 2203 | +0.009 | 0.044 |
| 15+ | 1566 | +0.004 | 0.043 |

It does not. `corr(|residual|, hand total) = -0.046` -- flat, and if anything slightly *negative*,
the opposite of the prediction. Splitting by scoring-card asymmetry gives the same answer:
`corr(residual, scoring gap) = +0.034`, and mean |residual| is 0.049-0.061 across every gap from
-3 to +3. So the principle is right and the residual in this checkpoint does not appear to be
made of it: it sits at a roughly constant 0.054 regardless of how much is hidden.

This does not rescue the earlier over-claim, it only bounds it. The right statement is that the
residual is *mostly* error here, and that any reading of a single state's residual has to clear a
0.15 sd noise floor first.

## The two sides' gradients do conflict, and the conflict is acquired

Gradient surgery (PCGrad) is motivated only when task gradients genuinely oppose each other.
Treating "win as US" and "win as USSR" as two tasks sharing one network makes that measurable
(`ai/eval/side_gradient_conflict.py`): take the policy-gradient direction on each side's
transitions separately and compare. The surrogate is `-(logp * A)`, which is what PPO's gradient
reduces to on the first epoch while the importance ratio is still 1.

| checkpoint | cosine(US, USSR) | sd | rollouts conflicting | \|g_USSR\| / \|g_US\| |
|:---|---:|---:|---:|---:|
| 40M | **+0.157** | 0.118 | 1 of 4 | 0.60-0.76 |
| 160M | **-0.096** | 0.142 | **3 of 4** | 0.39-1.05 |

Early in training the two sides mostly **reinforce** each other; by 160M they mostly **oppose**.
The sign flips over the same window as the behavioural divergence and the advantage collapse.

Note the norms: the US gradient is not small -- it is the *larger* of the two at 160M in three of
four rollouts. So "the US has no gradient" is wrong; the US has a gradient that is being partly
cancelled. That is a different problem with a different fix, and it is the one PCGrad addresses.

**Caveat: 4 rollouts per checkpoint, sd ~0.13.** The two means are about two standard deviations
apart at n=4, which makes this suggestive rather than settled. It is the first of the three
hypotheses tested here to survive its own measurement, and firming it up is cheap -- more
rollouts, and intermediate checkpoints to see where the sign crosses.

## KataGo's mechanisms, and which of them transfer

From *Accelerating Self-Play Learning in Go* (Wu, 2019):

* **Adaptive outcome centering** is the score-utility re-centering: "at the start of each search,
  the utility is re-centered by setting x_0 to the mean of the neural net's predicted score
  distribution at the root node", with a utility that saturates far from 0 so the incentive stays
  on realistic marginal gains. It is a *search* mechanism, and this codebase's evaluation path is
  PIMCTS rather than search-in-training, so it does not transfer directly -- but the principle
  does, and it is the same principle the advantage collapse needs: re-center the objective on the
  currently-expected outcome so a decided game still produces gradient.
* **Komi randomization** -- "komi is randomized by drawing from a normal distribution with mean 7
  and standard deviation 1", and in handicap games komi is adjusted to compensate the weaker side.
  This is the closest published analogue to the 90/10 problem: perturb the starting balance so
  games stay near even and the value target stays informative. The Twilight Struggle analogue is a
  randomized starting VP or a randomized opening, and it is cheap.
* **Auxiliary ownership and score targets** -- extra heads predicting who ends up owning each
  point on the board, used only to sharpen credit assignment. The direct analogue here is a
  per-country *final control* head, which is exactly the signal the critic was measured to lack:
  it cannot price control of a European battleground at all. This is an auxiliary loss, not an
  observation change, so it costs no checkpoint compatibility.
* **Reduced visits in dominated positions** and downweighting resign-worthy positions: relevant to
  the same saturation problem, less directly applicable without search.

## The critic collapses to the base rate, and that is the whole mechanism

### `explained_variance` cannot be used to judge this critic

`rollout_buffer` builds the GAE return as `returns_win[t] = last_gae + v_t`, so `G - V = A`
**exactly** and therefore `EV = 1 - Var(A)/Var(G)`. A critic whose advantages collapse scores near
1.0 *by construction*. The 0.996 recorded at 160M is the advantage collapse restated, not
independent evidence of a good critic, and the two move together across the run because they are
the same quantity. Pinned by `test_explained_variance_is_circular_by_construction`.

### The non-circular test: does `v_win` predict who actually wins?

Against two baselines that require no learning -- the base rate (always predict the more frequent
winner) and the sign of the VP track.

| | 80M | 160M |
|:---|---:|---:|
| US win rate in the sample | 53.9% | 12.7% |
| base-rate accuracy | 53.9% | **87.3%** |
| VP-sign accuracy | 66.8% | 66.0% |
| **critic accuracy** | **79.2%** | **87.5%** |
| critic correlation with outcome | **+0.654** | **+0.264** |

At 80M the critic is genuinely good: 79.2% against a 53.9% base rate, and **66.2% at turn 1** --
it calls the winner from the opening position well above chance. At 160M it beats the base rate by
**0.2 percentage points**, and per turn it is 85.9 vs 85.8, 85.7 vs 85.8, 88.9 vs 88.9. It has
degenerated into "predict the USSR" and its correlation with the outcome has fallen by more than
half. **At 160M the critic has learned nothing the base rate does not already say.**

### Why that makes advantages vanish

With `V` near-constant at the base rate and the outcome 90/10, `delta_t = r_t + gamma*V' - V` is
near zero at every non-terminal step, and at the terminal step it is `r_T - V ~ -1 - (-0.87)`.
Small everywhere. So the collapse is self-consistent: a dominant strategy makes outcomes
predictable, a critic minimising loss on predictable outcomes needs no discrimination, and a
critic with no discrimination produces no advantage. Learning stops for **both** sides -- which is
what the per-side figures show (US std 0.048, USSR 0.037 at 140M) -- and the side sitting on a
winning strategy loses nothing by freezing, while the side that needs to change has nothing left
pushing it anywhere.

### Where it breaks

`adv_std_raw` across the run (192 envs x 128 steps per checkpoint):

| steps (M) | 5 | 20 | 35 | 50 | 65 | 80 | 95 | 110 | 125 | 140 | 155 | 160 |
|:---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| adv_std_raw | 0.247 | 0.248 | 0.179 | 0.180 | 0.289 | 0.210 | 0.179 | 0.269 | **0.111** | **0.043** | **0.030** | 0.052 |
| explained var | 0.65 | 0.65 | 0.87 | 0.74 | 0.82 | 0.77 | 0.90 | 0.84 | 0.98 | 0.997 | 0.999 | 0.996 |

Noisy but level through 110M, then a hard break: it falls 6x between 110M and 155M. That is the
same window as the Europe-control discovery and the USSR win-rate spike.

## Does the critic read the hand? Yes -- more than it reads the board

`ai/eval/card_sensitivity.py`. Each perturbation is applied to a clone and only the value head is
read. The null control must be exactly 0 and the board perturbation supplies the scale, without
which a `|dv|` is uninterpretable.

| perturbation | 80M | 160M |
|:---|---:|---:|
| null (nothing changed) | 0.0000 | 0.0000 |
| hand replaced with random draw-deck cards | 0.0977 | 0.0279 |
| all 7 scoring cards forced into hand | 0.2473 | 0.0497 |
| +4 influence in a European battleground | 0.0940 | 0.0198 |
| **ratio hand / board** | **1.04** | **1.41** |

So the answer to "does the critic look at cards" is an emphatic yes, and it distinguishes *which*
cards -- forcing all seven scoring cards into hand moves the value 2.5x as far as a random hand.
It is at least as sensitive to the hand as to a four-influence swing in West Germany, France or
Italy, and at 160M half again more so. This supports reading part of the zero-sum residual as
information rather than error, even though the residual itself was measured not to track hand size.

It also sharpens the earlier finding rather than contradicting it: the critic's *board* sensitivity
is genuinely the weaker of the two.

And note the column-wise collapse. Every perturbation moves the 160M critic about **5x less** than
the 80M one -- 0.098 to 0.028, 0.094 to 0.020. The value function has gone flat with respect to
everything, which is the same fact as the base-rate degeneration and the advantage collapse, seen
from a third direction.

## It reproduces on E3-15-22: different architecture, different exploit, same collapse

E3-15-22 is the 2-graph-layer arm (E3-17 has 0), trained independently. Its side balance goes the
same way at the same point:

| steps (M) | US win% | USSR win% |
|:---|---:|---:|
| 0-21 | 45.8% | 54.2% |
| 41-62 | 47.7% | 52.3% |
| 82-103 | 52.3% | 47.7% |
| **103-123** | **27.8%** | **72.2%** |

**And it is not Europe control.** That ending is 0.0-0.2% across the whole run; the USSR runs away
through the ordinary 20 VP route (61.0% of endings in the last band). So the winning strategy is a
different one, found by a different architecture, and the collapse is identical:

| steps (M) | adv_std_raw | expl var | base rate | critic acc | corr |
|:---|---:|---:|---:|---:|---:|
| 5 | 0.209 | 0.537 | 51.2% | 62.1% | +0.334 |
| 30 | 0.190 | 0.705 | 56.7% | 73.1% | +0.524 |
| **55** | **0.236** | 0.779 | 56.8% | **77.5%** | **+0.621** |
| 80 | 0.202 | 0.770 | 70.1% | 80.3% | +0.556 |
| 105 | 0.175 | 0.913 | 94.1% | 94.4% | +0.429 |
| **120** | **0.073** | **0.990** | 84.6% | **84.3%** | **+0.102** |

The critic peaks around 55M and then degenerates; by 120M it is **below** the base rate (84.3%
against 84.6%) and its correlation with the outcome has fallen from +0.621 to +0.102, while
`adv_std_raw` falls 3x and explained variance climbs to 0.990 exactly as the circularity predicts.

This settles what the pathology is not. It is not the Europe-control exploit, not the graph depth,
and not one unlucky seed. **It is the self-play dynamic**: whenever one side finds a strategy the
other has not answered, outcomes become predictable, the critic has no reason to discriminate, the
advantage vanishes and both policies freeze -- with the side that needed to adapt being the only
one that loses by it.

It also means the remedy should not be aimed at Europe control specifically, and that any fix can
be validated on either arm. The instrument is `adv_std_raw` plus critic accuracy against the base
rate, and both are now recorded.

## The tournament: both 160M arms are weaker than their own 80M selves

Round robin, 400 games per side per pair, plus the two baselines
(`data/tournaments/e3_12_15_17_report.md`). The report's model labels are positional, so the
mapping is given here once.

| arm | Elo | as USSR | as US | USSR - US |
|:---|---:|---:|---:|---:|
| E3-15-22 @80M | **2059.8** | 75.1% | 67.8% | +7.2 pp |
| E3-17-21 @80M | **2058.7** | 72.5% | 70.1% | **+2.4 pp** |
| E3-17-22 @80M | 2003.6 | 68.2% | 61.1% | +7.0 pp |
| E3-15-22 @160M | 1985.0 | 72.3% | 52.2% | **+20.1 pp** |
| E3-15-21 @80M | 1964.8 | 62.3% | 57.2% | +5.1 pp |
| E3-17-22 @160M | 1939.3 | 72.8% | 40.3% | **+32.5 pp** |
| E3-12-21 @80M | 1842.2 | 50.5% | 39.2% | +11.2 pp |
| HeuristicBot | 1500.0 | 20.4% | 15.0% | +5.4 pp |
| RandomBot | 895.8 | 0.7% | 0.2% | +0.5 pp |

**Both 160M checkpoints rank below their own 80M versions** -- E3-15-22 loses 74.8 Elo and
E3-17-22 loses 64.2 over the second half of training. Independent head-to-head confirmation of
the collapse, from outside the training loop.

**And the loss is entirely on one side:**

| arm | USSR 80M -> 160M | US 80M -> 160M |
|:---|---:|---:|
| E3-15-22 | 75.1% -> 72.3% (**-2.8**) | 67.8% -> 52.2% (**-15.6**) |
| E3-17-22 | 68.2% -> 72.8% (**+4.6**) | 61.1% -> 40.3% (**-20.8**) |

The USSR side is flat or slightly better; the US side drops 16 to 21 points. That is the exact
signature predicted from the internals -- the side with the working strategy freezes at no cost,
the side that needed to adapt drifts -- and it is measured here against a fixed external field
rather than against itself.

The side gap tells the same story: every 80M arm sits between +2.4 and +11.2 pp, while the two
160M arms are at +20.1 and +32.5. E3-17-22 @160M wins only **13.5%** of its games as US against
E3-15-22 @80M.

Worth noting separately: **E3-17-21 @80M is both near the top and the most side-balanced arm in
the field** (+2.4 pp), which makes it a reasonable reference point for what a healthy run looks
like on these numbers.
