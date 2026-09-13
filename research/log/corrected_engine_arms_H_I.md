# Arms H, H2 and I on the corrected engine — tactics without strategy, the KL penalty, and 480M

Records §25–§27 of the original `experiments.md`: the first arms after the starred-card fix and the
first on observation layout **v2.3**. Nothing before §25 is on the same Elo ladder, because the fix
changes the decision stream. §25 covers arms H (80M, seed 20260920) and H2 (the same recipe on seed
20260921, run to 160M and then to 240M in §25.1) — the first model here that plays a recognisable
tactic, and still none with strategy; §26 is arm I, the `--eta 0` ablation of the NashPG KL penalty,
which finds no intransitivity for the penalty to prevent and keeps it anyway at 169 Elo; §27 doubles
H2 to 480M and finds it beating its own past and nothing else, with the two sides 22 points apart,
and §27.1 names the single mistake behind most DEFCON-1 endings. **This file is append-only
history** and is not edited to match current belief.

---

## 25. Arms H and H2 on the corrected engine — the first model with tactics, and still none with strategy

The first runs after the starred-card fix (a starred card spent for Operations was being deleted
from the game for 380 of the repo's 389 commits) and the first on observation layout **v2.3**.
Arm H is 80M steps, seed 20260920; arm H2 is the same recipe on seed 20260921, run to 160M.

**Nothing before this is comparable to it.** The fix changes the decision stream, so every Elo in
§1–§24 is on a different ladder. v2.2 is retired with its `temp_card_count` slot, so arms F, F2
and G cannot be re-rated even in principle — arm G's 1957.2 is not a number this can be measured
against, and reading H2's 1925 as a regression from it would be wrong.

### Elo, on the corrected-engine ladder

7 players, 500 games a side per pair, 21,000 games, Bradley-Terry MLE anchored on HeuristicBot at
1500 (a rule-based bot, and so the same player on any engine):

| model | Elo | vs HeuristicBot |
|:---|---:|---:|
| **H2 @160M** | **1925.4** | 89.2% |
| H2 @80M | 1872.1 | 87.3% |
| H @80M | 1867.4 | 89.2% |
| arm E (v2.1, 80M, pre-fix) | 1763.2 | 84.3% |
| arm D (legacy, 80M, pre-fix) | 1739.7 | 82.5% |
| HeuristicBot | 1500.0 | — |
| RandomBot | 866.1 | 2.6% |

**The seed is worth nothing measurable.** H2 @80M and H @80M differ by 4.7 Elo, and their direct
matchup is 50.8% — matching an independent pooled measurement over four late snapshots a side and
all 16 pairings (50.4%, +3 Elo). A clean replication, which is what a second seed was for.

**The budget is worth ~+55 Elo per doubling here**: 1867/1872 at 80M to 1925 at 160M, confirmed
twice by direct matchup (H2@160M beats H@80M 59.7%, and its own 80M self 60.9%).

**Game *shape* does not replicate, and that retires a claim.** H2 lands on the pre-fix arms D and
E, not on H — 98.6 plies against H's 107.0, 9.6% final scoring against 15.7%, a wider gap than H
had over D and E. The earlier reading that the corrected engine lengthens games came from H alone
and does not survive its own replicate. The fix is still correct — it is a rules bug either way —
but H was not evidence that it changed play. Details in [`variance_and_noise.md`](variance_and_noise.md).

### Tactics, for the first time

H2 @160M is the first model here that does something a human would recognise as a *tactic*: it
uses **UN Intervention to defuse an opponent-associated card and keep its Operations**. Across
five self-play games it played UN Intervention ten times, every one of them on an opponent card —
and in `h2_160M_selfplay_20260401`, turn 7 AR1, the USSR ran **Grain Sales to Soviets** through it
and couped with the Ops. Grain Sales is one of the DEFCON-suicide cards for the USSR
(`ai/eval/blunders.py`); playing it through UN Intervention is exactly the right handling of it,
and no earlier arm did this.

The blunder counters agree that the crude mistakes are gone: Olympic Games at DEFCON 2 fired 0 of
23 opportunities across the five games, and DEFCON-suicide-with-an-alternative 3 of 108 (2.8%).

### No strategy

**Battlegrounds stay empty, and the same ones every game.** Measured on 60 live self-play games
from the 160M snapshot, mean battlegrounds still completely untouched at the start of each turn:

| turn | 1 | 3 | 5 | 6 | 7 | 8 | 9 | 10 |
|:---|---:|---:|---:|---:|---:|---:|---:|---:|
| empty of 29 | 20.0 | 14.6 | 10.9 | 8.7 | 7.4 | **6.2** | 5.7 | 5.2 |

Of the 33 games that reached turn 8, **Saudi Arabia was empty in all 33**, India in 30, Algeria in
27. This is the §12 finding unchanged: the model does not contest a battleground it has no
influence adjacent to, and the same countries are conceded every game.

*(These are direct measurements. The trainer's own `diag/empty_battlegrounds_turn8` reports 0.0
for the whole run while `diag/mean_final_turn` reports 1–2 against an actual mean turn of 6.8 —
that probe is measuring something other than what it claims and should not be read. Filed as a
P0 item.)*

**And it cannot sequence a turn.** `h2_160M_selfplay_20260405`, turn 10, USSR to play, holding
both Tear Down this Wall and Grain Sales — two US cards — plus UN Intervention:

- AR3: spends **UN Intervention on Tear Down this Wall**, taking the Ops for influence.
- AR7: plays **Grain Sales to Soviets** raw, `EVENT_FIRST`, handing the US the event.
- The US coups Cuba on the same action round; DEFCON hits 1 and the USSR loses, +20 US.

The owner's read of the position: spacing Tear Down this Wall and holding UN Intervention for
Grain Sales instead wins the game for the USSR. The model had every piece of the right line and
picked the wrong order — it can play the tactic when the card is in front of it, and cannot plan
which card the tactic should be spent on. That is the gap between this and a mediocre human, and
it is what the P0 probes and P4 (setup / macro-action credit) exist to measure.

### 25.1 Continuing H2 to 240M: +16 Elo, and Wargames finally appearing

Arm H2 resumed from its own 160M state and ran to 240,058,368 steps (5,559s for the leg).
Same recipe, same seed, weights and optimiser moments restored.

**Strength is flattening.** Pooled over four late snapshots a side, 3,200 games:
H2 @240M beats H2 @160M **52.2%, +16 Elo**. The previous leg (80M → 160M, a doubling) was
worth +55; this one is 1.5x the budget and would have been ~+32 at that rate. Against the
anchor, 89.6% → **93.0%**.

**Game shape, self-play, 1,000 games each:**

| | mean ply | 20 VP | Europe Control | final scoring | DEFCON 1 | wargames |
|:---|---:|---:|---:|---:|---:|---:|
| H2 @160M | 100.9 | 50.9% | 0.0% | 13.2% | 34.2% | 1.7% |
| H2 @240M | 105.3 | 47.6% | 0.0% | 14.4% | 33.8% | **4.2%** |
| humans (ITS) | ~119.9 | 41.5% | 1.6% | 29.0% | 11.7% | 14.9% |

The one clear behavioural change is **Wargames, 1.7% → 4.2%** — two and a half times, and the
first sustained movement on a resource the arms had never used (§25 recorded 0-0.2% at 80M).
The binned training log agrees: it sits at 2-3% across the whole leg where it was under 1%
before. DEFCON 1 fell 46% → 37% over the leg in the training log, though self-play at
temperature 0.1 shows it flat at ~34%, so treat the decline as unconfirmed.

Everything else moved a little in the right direction and remains far off: ply 105 against ~120,
final scoring 14.4% against 29.0%, DEFCON 1 three times the human rate.

**Europe Control is 0.0% for both arms against 1.6% for humans.** That number could not be
measured at all before this run — the engine set ±20 VP and GAME_OVER for it, identical to any
other 20 VP win, so every such game was counted as `20vp`. `effect_bits::EUROPE_CONTROL_WIN`
makes it visible, and what it shows is that the models never win this way. Winning Europe
outright is a strategic plan the policy has no representation of, which is consistent with §25's
finding that it plays tactics and not strategy.

---

## 26. The NashPG KL penalty: there is no intransitivity to prevent, and removing it still costs 169 Elo

NashPG regularises the active policy toward a frozen reference with weight `eta`
(`policy_loss = ppo_loss + eta * kl_div - ent_coef * entropy`). The justification is anti-cycling
— stopping the policy beating what it just beat and losing to what came before. Nothing here had
ever checked that this game *has* cycles, and the term is a standing tax on exploration either
way.

**Arm I**: `--eta 0`, cold start, 80M steps, otherwise arm H2's recipe and its seed (20260921).
One variable.

### The penalty is not buying anti-cycling

Every pair on each run's own snapshot ladder, 7 snapshots from 20M to 80M, 21 pairs x 200 games:

| | 3-cycles among triples | later snapshot losing to an earlier one |
|:---|---:|---:|
| arm H2 (KL on) | **0** | **0 of 21** |
| arm I (KL off) | **0** | **5 of 21** |

**No intransitivity in either arm.** The hypothesis that motivated the term is not supported at
this budget: no triple anywhere cycles, with or without the regulariser.

### What it *is* buying is monotonicity

H2's ladder is perfectly ordered — every later snapshot beats every earlier one, all 21 pairs.
Arm I's is not. It peaks around 60M and then goes backwards: 80M scores **41% against its own
60M** and **38% against its own 70M**, and 70M scores 40% against 60M. (Two of the five
regressions, at 49%, are inside the ±3.5% noise of a 200-game cell; those three are not.)

So the failure mode without the penalty is not a cycle. It is a run that stops improving and
drifts, while still being totally ordered — it goes *down* a ladder rather than around a loop.

### And it costs

- **arm I vs arm H2, pooled over four late snapshots a side, 3,200 games: 27.5%, -169 Elo.**
- against the anchor: 86.1% (H2) against **79.4%** (arm I).

### No sign of the exploration it was supposed to be taxing

The premise was that the penalty suppresses exploration. It does not show up:

| self-play, 1,000 games | mean ply | 20 VP | Europe Control | final scoring | DEFCON 1 | wargames |
|:---|---:|---:|---:|---:|---:|---:|
| H2 @80M (KL on) | 97.1 | 50.8% | 0.0% | 8.1% | 41.0% | 0.1% |
| arm I @80M (KL off) | 91.7 | **62.0%** | 0.0% | 7.3% | 30.7% | 0.0% |
| humans (ITS) | ~119.9 | 41.5% | 1.6% | 29.0% | 11.7% | 14.9% |

Entropy ran 1.00-1.11 without the penalty against H2's 1.09-1.18 *with* it — no higher. Wargames
stayed at 0% and Europe Control at 0%, so none of the rare lines opened up. What did change is
that arm I plays a narrower game: 62% of its endings are VP-track wins against H2's 51%, and its
games are shorter. Unpenalised, the policy specialised rather than explored.

`kl_div` is still computed at `eta = 0`, and it ran 0.08-0.18 against H2's 0.02-0.06 — so the
policy did drift several times further from `pi_ref` when nothing pulled it back. The drift is
real; it simply did not buy anything.

**Keep `eta = 0.1`.** Not for the reason it was introduced — there are no cycles here to
prevent — but because it is worth 169 Elo as a stabiliser, and the exploration it was suspected
of costing is not visible.

---

## 27. 240M -> 480M: it beats its own past and stops beating anything else

Arm H2 resumed from 240M and ran a full doubling to 480,051,200 steps (31,886s). Same recipe and
seed throughout. The leg was meant to test whether returns were flattening — 80M→160M was worth
+55 Elo, 160M→240M only +16.

**They are not flattening.** Pooled over four late snapshots a side, 3,200 games:
**H2 @480M beats H2 @240M 59.2%, +65 Elo** — more than the earlier doubling. The +16 at 240M
was a plateau, not the start of a curve.

### But nothing else agrees that it got better

| | vs its own 240M | vs HeuristicBot |
|:---|---:|---:|
| H2 @240M | — | **92.2%** |
| H2 @480M | **59.2% (+65 Elo)** | **90.6%** |

Against a fixed external opponent it went *down* 1.6 points over 240M steps of training. That is
the signature of a policy improving against its own lineage rather than improving.

The internal ladder says the same thing. Seven snapshots spanning the leg, 21 pairs × 200 games,
and almost every cell sits between 42% and 56% — a 240M-step spread that barely separates. The
250M snapshot beats 290M, 330M, 410M and 450M, and holds 480M to 49%. Six of 21 pairs have a
later snapshot losing to an earlier one.

One 3-cycle appears (250M > 450M > 370M > 250M) — the first ever observed here. It should not be
read as intransitivity: with 21 cells clustered near 50% and 200 games each (±3.5%), one cycle
among 35 triples is what chance produces. §26 found none across two arms; this is not evidence
against that.

### The real damage: the sides came apart

Binned by 20M, the USSR win rate climbs monotonically and does not come back:

| window | 240-260 | 280-300 | 320-340 | 360-380 | 400-420 | 440-460 | 460-480 |
|:---|---:|---:|---:|---:|---:|---:|---:|
| USSR win % | 57.3 | 58.1 | 61.4 | 62.1 | 65.8 | 71.3 | **72.2** |

Against a human 49.9%. This is not the oscillation that fooled the monitor at 160M — it is
monotone across twelve consecutive windows. `experiments.md` §4.5 recorded a standing 60–65%
USSR imbalance historically; at 480M it is worse than it has ever been. Self-play against a
partner that is 22 points worse as the US is training both sides on a distorted distribution.

### Game shape barely moved

| self-play, 1,000 games | ply | 20 VP | Europe Ctl | final | DEFCON 1 | wargames |
|:---|---:|---:|---:|---:|---:|---:|
| H2 @240M | 104.2 | 47.1% | 0.0% | 11.8% | 36.4% | 4.7% |
| H2 @480M | 109.5 | 47.0% | 0.0% | 11.5% | 35.4% | **6.1%** |
| humans (ITS) | ~119.9 | 41.5% | 1.6% | 29.0% | 11.7% | 14.9% |

240M steps bought ~5 plies and 1.4 points of Wargames. DEFCON 1 is still three times the human
rate, final scoring still a third of it, Europe Control still never.

### Reading

Budget is no longer the binding constraint. The run is churning — moving in policy space, beating
what it just was, and not getting better against anything outside itself, while the two sides
drift 22 points apart. Doubling again is not the next experiment. The side imbalance is, because
a self-play equilibrium this lopsided is training both policies on a board neither would face
against a balanced opponent.

### 27.1 The DEFCON-1 endings are mostly one specific mistake, and it has a name

34.1% of 220 self-play games from the 480M snapshot end at DEFCON 1. Reading three of them by
hand suggested two different things were wearing one label, so the cause was counted.

**82.7% are *provoked*** — the loser was pushed into it rather than walking into it. The
mechanism is a single move: the model plays an **opponent-associated card for Operations while
DEFCON is 2**. An opponent's card fires its event when played for Ops; the event lowers DEFCON;
DEFCON reaches 1; and `resolve_defcon_one_loss` makes the *phasing* player the loser — which is
the player who just played it.

The card in play when DEFCON crossed 2 → 1:

| card | share of DEFCON-1 endings |
|:---|---:|
| Grain Sales to Soviets | **25.3%** |
| "Lone Gunman" | 14.7% |
| Duck and Cover | 12.0% |
| Olympic Games | 9.3% |
| Summit | 8.0% |
| CIA Created | 6.7% |
| Tear Down this Wall | 5.3% |
| Five Year Plan | 5.3% |

Every one of those except Summit and Tear Down this Wall is already on the DEFCON-suicide list in
`ai/eval/blunders.py`. So the blunder tracker is naming the right cards — the puzzle was why its
*rate* looks small while the endings look common.

**The rate is per opportunity, and opportunities are frequent.** Across six logged games the
tracker reported 1/32, 1/21, 0/15, 1/7, no chances, 1/13 — about 4 blunders in 88 opportunities,
or 3-8%. But that is 7 to 32 opportunities *per game*, so the absolute incidence is roughly one
every two games, and each one is frequently fatal. A low rate against a large denominator is
still a game-ending mistake most of the time it happens.

**Summit is the opposite case and should not be counted with them.** At 8% of these endings the
model is not dying, it is killing: the non-phasing player wins the Summit roll and chooses to
lower DEFCON, ending the game while the *opponent* is phasing. In `h2_480M_selfplay_20260503`
the US does exactly this at turn 9 AR7 and wins +20. That is a correct tactic, not a blunder,
and it means the 34.1% figure mixes a real error with a real skill.

Compare the human 11.7%. The gap is not that humans never play an opponent's card for Ops at
DEFCON 2 — it is that they check the card first.
