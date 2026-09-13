# Human play — the baseline the arms are measured against

Living reference for everything this project knows about how humans play this game and how
closely a policy reproduces them: what the two human corpora contain, which one is the right
baseline for which question, the length and ending-mix numbers themselves, and the definition of
the agreement measure. **Update discipline: rewritten in place.** Corrections to these numbers
replace them; the history of how each was got wrong first is in
[`../log/measurement_bugs.md`](../log/measurement_bugs.md) (the turn-10 Wargames misclassified as
final scoring, which mattered here and nowhere else) and
[`../log/variance_and_noise.md`](../log/variance_and_noise.md) (why a single arm's game shape
cannot be read against these numbers). What belongs here: corpus composition, reference numbers,
which corpus answers which question, and how agreement is defined and reported. What does not:
an arm's result against these numbers, which is an experiment entry.

How the corpus is converted to engine decisions — the log, the engine and the hands behind it —
is [`../log/P7_replayer_conversion.md`](../log/P7_replayer_conversion.md).

These sections were `metrics.md` §1.5, §1.5.1, §1.5.2, *The gap this exposes*, and §9.11–§9.11.2.

---

## Human game length, and what the ts-replayer corpus can and cannot say about it

The corpus is 274 distinct logs: **119** reach a terminal state, **146** are fragments whose
recording stops, and 9 are empty. Only the 119 are a game length. Two traps:

- The fragments' mean *last* turn is 6.77, which happens to land right on top of the arms'
  6.5-7.2. Averaging all 265 usable logs gives 7.95. Neither is a game length; both flatter the
  models by measuring when ts-replayer users stopped recording.
- `game_ended` must come from the converter's terminal test, not from the log's own fields. The
  log's `defcon` on the last entry holds the value *before* the ending resolved, so classifying
  by it found 2 DEFCON-1 endings where there are 21.

The 119, classified from the terminal position the converter now records (`final_turn`,
`final_action_round`, `final_defcon`, `final_cmc_suicide`, `final_defcon_provoked`):

| | mean ply | share |
|:---|---:|---:|
| final scoring | 154.0 | 56.3% |
| 20 VP / held scoring | 129.9 | 23.5% |
| wargames | 121.7 | 18.5% |
| DEFCON 1 (own) | 153.0 | 0.8% |
| DEFCON 1 (provoked) | 137.0 | 0.8% |
| **all** | **142.2** | 92.3% of a full game |

67.2% of finished human games play all ten turns out. 20 of the 119 contain an AR8, which the
ply scheme counts at its nominal 16 and so under-counts by 2 apiece.

## The ts-replayer figure is biased long; the ITS results database is the better baseline

`/workspace/data/itsc-games` is a scrape of the ITS Junta results table at twilight-struggle.com
— **47,928** digital (Playdek, Deluxe) games, 44,136 of them with a rules ending, against
ts-replayer's 119. It is results-only: one row per game with `endTurn` and `endMode` and no
moves, so it can give length and ending mix and **cannot** replace ts-replayer as a training
corpus.

Its `endTurn` uses the same convention the engine does — 12,787 of its 12,792 Final Scoring games
are recorded at turn 11 — which is independent confirmation of the turn-11 artefact above, from a
source that has never seen this code.

It disagrees with ts-replayer sharply, and in the direction that says ts-replayer is the biased
one:

| | ts-replayer (119) | ITS (44,136) |
|:---|---:|---:|
| went the distance | 67.2% | **29.7%** |
| mean end turn (engine scale) | 9.99 | **8.33** |
| mean ply | 142.2 | **~119** (bounded [112, 123]) |
| final scoring | 56.3% | 29.0% |
| 20 VP | 23.5% | 43.1% |
| wargames | 18.5% | 14.9% |
| DEFCON 1 | 1.7% | **11.7%** |
| held scoring | (folded into 20 VP) | 1.4% |

ts-replayer's 119 are the subset of 274 logs whose *recording* completed, and that filter is not
independent of how the game ended: a game that blows up at turn 6 leaves a log that stops
mid-turn and lands in the 146-game fragment pile, while a game that goes the distance gets
recorded to the end. The DEFCON-1 row is the tell — 1.7% against 11.7% is not sampling noise at
these sizes. Use ITS for length and ending-mix baselines; use ts-replayer where moves are needed.

The ITS ply figure is an estimate, not a measurement: a turn pins the ply only to a 14- or
16-wide interval, so the point value applies ts-replayer's mean *within-turn* offset per ending
kind (20 VP lands late in a turn, 13.7 of 16; Wargames early, 5.5). Report it with its interval.

## Side balance and per-side asymmetry in the ITS corpus

Human play is almost exactly balanced: **49.90% USSR** over 43,685 decided games (451 ties,
1.02%). Every arm so far has wandered well outside that — H2 has ranged 41.9% to 54.8% within a
single run — so the pooled figure is a target the runs are not obviously converging on.

The two sides do not win the *same way*, and a pooled ending mix hides it:

| | US wins | USSR wins |
|:---|---:|---:|
| n | 21,888 | 21,797 |
| mean end turn | 8.62 | 8.00 |
| mean ply | ~124 | ~115 |
| went the distance | 33.4% | 25.2% |
| 20 VP | 36.2% | **50.8%** |
| final scoring | **32.1%** | 25.0% |
| wargames | 14.5% | 14.9% |
| DEFCON 1 | **15.6%** | 8.1% |
| held scoring | 1.6% | 1.1% |

The USSR takes half its wins on the VP track and wins faster; the US wins later, more often by
scoring the board out. Read the other way, within each ending: the USSR wins 58.3% of 20 VP
games, while the US wins **65.9%** of games that end at DEFCON 1 — that is, the USSR is far more
often the side that walks into the war it loses to. Ties are almost entirely final scoring
(67.4%) and Wargames (32.4%), which is what a 6 VP Wargames swing landing on zero looks like.

This is why the trainer now splits length and ending mix by winning side (`game_won_us/`,
`game_won_ussr/`): a run can hit the pooled human numbers while getting both halves wrong.

## The gap this exposes

Self-play at temperature 0.1, 1,000 games each, measured through the batched match runner:

| | mean ply | of 154 | 20 VP | final scoring | DEFCON-1 | wargames |
|:---|---:|---:|---:|---:|---:|---:|
| RandomBot | 40.3 | 26.1% | 47.4% | 0.6% | 50.5% | 1.5% |
| arm D (legacy, 80M) | 99.0 | 64.3% | 46.1% | 11.0% | 42.9% | 0.0% |
| arm E (v2.1, 80M) | 100.1 | 65.0% | 46.0% | 9.6% | 44.2% | 0.2% |
| arm H (v2.3, corrected engine, 80M) | 106.7 | 69.3% | 38.0% | 14.4% | 47.5% | 0.1% |
| arm H2 (same config, second seed, 80M) | 98.6 | 64.0% | 47.4% | 9.6% | 42.8% | 0.2% |
| HeuristicBot | 114.6 | 74.4% | 74.6% | 25.4% | 0.0% | 0.0% |
| **humans (ITS, 44,136)** | **~119** | **77.4%** | **43.1%** | **29.0%** | **11.7%** | **14.9%** |
| humans (ts-replayer, 119) | 142.2 | 92.3% | 23.5% | 56.3% | 1.7% | 18.5% |

H and H2 are the same configuration on two seeds and the same strength to +/-3 Elo, yet differ
by 8.4 plies ([`../log/variance_and_noise.md`](../log/variance_and_noise.md)): read the pair,
never H alone.

Against the ITS baseline the length gap is modest -- the arms at 99-107 against ~119, and
HeuristicBot at 114.6 is essentially at human length -- and the arms match humans on 20 VP
endings almost exactly (38-46% against 43.1%). The deficit is in *how* games end, in two
specific ways:

- **DEFCON 1 is four times too common**: 43-48% of arm games against 11.7% of human ones.
  HeuristicBot, which carries an explicit instant-loss safety layer, never does it at all, and
  that alone is most of its length advantage over the arms.
- **Wargames is never played**: 0-0.2% of arm games against 14.9% of human ones. It is a real
  human resource for closing out a won position at DEFCON 2, and no arm has found it.

Held scoring is folded into the 20 VP column for the arms and for ts-replayer: our classifier
only separates it on the training path, where ts_env supplies the flag.

Arms F, F2 and G cannot be re-measured: they are observation layout v2.2, which is retired, so
their checkpoints cannot be loaded. Their turn-only training figures remain the only record.


---

## Agreement with human play

The corpus is the only strategy prior available, so how closely a policy reproduces it is a
reported figure everywhere -- per BC epoch, per snapshot, and in the tournament reports. It is
not accuracy: a play that spends several points is one decision made several times, and the
order the log happens to record is not part of it.

### Counting a placement without the order it was recorded in

**The measure was wrong, and by construction.** A card played for Operations spends its points one
at a time and the engine asks a separate `POINT_NODE` question for each, so the log's order is
whatever the recording happened to write. Placing two Influence in Angola and one in Zaire is the
same play in any order, and scoring each point against the index the human's sequence happened to
hold marked the model wrong for reordering a play it agreed with. The same holds for every event
that spreads or removes several points -- Decolonization, De-Stalinization, Colonial Rear Guards,
Ussuri River Skirmish, Puppet Governments, COMECON, Marshall Plan, The Reformer, and for removals
Socialist Governments and East European Unrest.

`ai/eval/agreement.py` groups consecutive point decisions belonging to one play and scores the
group on the multiset of countries rather than the sequence. The model is teacher-forced along the
human's trajectory, so its own earlier choices cannot take it somewhere the human never went, and
each point still contributes exactly one comparison -- the two figures are directly comparable and
only permutations are forgiven. Two points into one country are two entries, so agreeing on the
country but not the weight still costs.

**It matters less than expected.** Over 60,670 decisions from 120 replays, of which **34% sit in
multi-point plays**:

| | ordered | unordered | gain |
|:---|---:|---:|---:|
| BC on the human corpus | 46.04% | 46.34% | +0.29 |
| BC on self-play | 33.74% | 34.34% | +0.60 |
| E3 arm B (human) final | 32.41% | 32.96% | +0.56 |
| E3 arm A (self-play) final | 32.18% | 32.93% | +0.76 |

**So the ordering artefact was worth about half a point, not the several it might have been.** The
reason is teacher forcing: at the second point of a play the model already sees the board after the
human's first placement, so where it disagrees it is usually disagreeing about *which* countries,
not about the order. Every figure quoted earlier in
[`../log/P7_human_bc_warmup.md`](../log/P7_human_bc_warmup.md) §9 was pessimistic by roughly this
much, which changes no conclusion in it -- its §9.1 washout still lands at ~32-33% either way.

The correct measure is now the one to use, and `play` is stored as a dataset column so it can be
applied without re-running conversion, which is the expensive part.


### Coups and realignments are excluded from reordering

Not every run of point decisions is order-free, and the measure above treated them all as if
they were. The
board changes between points wherever a die is involved: a **realignment** roll is made against the
influence the last one left, so a different order is a different sequence of odds, and the same
holds for **coups** — in particular **Che**, whose second coup is offered only if the first removed
influence, so the pair is a sequence and not a set.

Those are now scored strictly. Implemented as a **blacklist** rather than a whitelist of the
order-free cases, per the owner: spreading Influence is the ordinary case, and a card that spreads
it in some new way should be handled without anyone having to remember to add it. Detection needed
`DecisionContext.op_mode`, which was not exposed to Python.

**It changes the numbers barely at all.** Grouped decisions fall from 34.0% to **31.8%** of the
total, and the correction each model gets is unchanged to within 0.01 points:

| | ordered | unordered | gain |
|:---|---:|---:|---:|
| BC on the human corpus | 46.04% | 46.32% | +0.28 |
| BC on self-play | 33.74% | 34.33% | +0.59 |
| E3 arm B (human) final | 32.41% | 32.96% | +0.55 |
| E3 arm A (self-play) final | 32.18% | 32.93% | +0.75 |

Which is worth knowing in itself: the reordering credit was never resting on coups and
realignments being wrongly forgiven, so the unordered figures above stand as measured. The measure
is now right for the right reason rather than by luck.


### Agreement is now the reported figure everywhere

`ai/eval/agreement.evaluate_dataset` takes either dataset and returns both figures, so nothing has
to re-implement the measure. A directory is the human corpus, which stores the play grouping as a
column; a file is the self-play set, whose loader gained `stream_with_plays` and recovers the
grouping while replaying, since that format keeps only a seed and the actions.

BC warmup now reports it every epoch, for both datasets:

```
Epoch  2/ 2 COMPLETED | Loss: 1.8802 | Strict Acc: 44.81% |
    Agreement: 47.48% (ordered 47.31%, 20,000 decisions)
```

Three numbers because they answer different questions. **Strict Acc** is the running in-batch
figure, computed on shuffled batches while the weights are still moving, and is what the trainer
always printed. **Agreement** is the measure: an unshuffled pass after the epoch, scoring a play on
the multiset of countries. **ordered** is that same pass scored strictly, so the gap between the
last two is exactly what reordering costs and nothing else.

The pass is capped at 20,000 decisions. The self-play format rebuilds its observations by replaying
from a seed, so a full pass over 2.1M samples would take minutes per epoch; 20,000 gives a figure
stable to about a tenth of a point.

Note the in-batch and post-epoch numbers differ by a few points (44.81% against 47.31% here) and
should: one averages over an epoch of changing weights, the other measures the weights the epoch
ended with.
