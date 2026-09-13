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
