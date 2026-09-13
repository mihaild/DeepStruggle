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
