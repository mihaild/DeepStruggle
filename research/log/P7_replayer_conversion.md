# Replayer Conversion — the log, the engine, and the hands behind it

How the human ts-replayer corpus is turned into engine decisions, and every discrepancy found
between the two while doing it. Split out of the main experiment log, which is about
how well the agents play; this file is about whether the data they are measured against is read
correctly. The rule the pipeline is held to is in `AGENTS.md` §4 invariant 11 and
`tools/README.md` §6: a decision the log does not determine is a bug to diagnose, not a gap to
fill.

**Section numbers are the ones these entries were first written under**, in `experiments.md`.
They are kept rather than renumbered so that every existing cross-reference -- twenty-one inside
`experiments.md`, and several from source files -- still names the right entry. A gap in the
numbering here is a section that stayed behind because it is about play rather than conversion.

The maintenance rules are `experiments.md`'s: record what was compared, the numbers with their
sample size, and what the measurement cannot tell you; revise rather than delete when a later
finding invalidates an earlier one, and say what invalidated it. Instrument faults and what a
measurement is reproducible to are in [`measurement_bugs.md`](measurement_bugs.md) and
[`variance_and_noise.md`](variance_and_noise.md).

---

## Score: what the log says against what the engine says

The corpus's scores and the engine's disagree in twenty-one places out of 282 games. Every
one of them turned out to be the log's bookkeeping, a known log fault, or a difference of
*when* the score is read -- not an engine error. The sequence below is worth reading in
order, because the first measurement was wrong by a factor of thirty and the reason is
instructive.

### 8.6 Replay 259 is VP drift, not a card bug — and drift is corpus-wide — SUPERSEDED BY 8.7

**The user's arithmetic was right and the engine's starting number was wrong.** At replay 259 the
engine holds `victory_points = 18` when it labels the KAL-007 headline a win, and KAL takes it to
20. The log records 17 at the end of turn 7 and **16** at the turn 8 headline, and it also shows
KAL actually played at **AR1, not the headline**, moving the score 16 → 17. So the "win" rests on
a starting VP two points above what the game had.

**This is not specific to 259.** `Conversion.vp_drift` counts entries where the engine's score
disagreed with the log's, and across the 282 converted games:

| | |
|:---|---:|
| games with **zero** drift | 53 (18.8%) |
| games with some drift | **229 (81.2%)** |
| drift per game | mean 8.1, median 5, p90 20, max 56 |
| `log_miscounts` / `scores_forced` across the corpus | 1 / 1 |

`_reconcile_scalars` resyncs `state.victory_points` to the score the log narrates whenever an
entry states one, so drift is corrected at entry boundaries and does not accumulate — which is why
these games still convert cleanly. But **within** an entry the engine's VP is its own, and that is
what a decision sample sees.

**Consequence for §8, which is not yet resolved.** The human VP-by-turn arc — the whole basis for
saying humans reproduce the US late-war recovery — is read from the engine's VP at decision time
(`obs.vps`), not from the log's narrated score. Turn boundaries are taken at the first decision of
turn T+1, which is close to a resync, so the arc may well survive; but "may well" is not
"measured". **Re-derive the §8 arc from the log's own narrated scores before relying on it.** That
is a cheap check and it is the one that matters, because §8 is currently the evidence that the
engine is not USSR-biased.

Separately, this is a second reason the forced-win metric cannot be trusted on human data (§8.5):
a labelled win can rest on a VP the game never had.


### 8.7 Most of the "VP drift" was the log's score field, not the engine — 8.6 corrected

**`Conversion.vp_drift` is not a measure of engine error.** It compares the engine against
`entry.score`, a running field that lags the log's own narration -- `_narrated_score`'s docstring
already records the two disagreeing in 531 of 6,602 places. The converter resyncs from the
*narration*, which is the authoritative statement, so a lagging field produces a counted "drift"
with nothing wrong.

**Replay 219 is the clean demonstration.** At turn 8 AR1 the log reads:

```
Turn 8, USSR AR1: South America Scoring: Event: South America Scoring
USSR gains 10 VP. Score is even.
```

The score was US +10, the USSR gains 10, so it is even -- and the engine says 0, matching the
narration exactly. The `score` field still reads 10, and goes on reading 10 for five more entries
before catching up at AR5. Every one of those was counted as drift. The engine was right
throughout.

**Re-measured against the narration**, over midgame entries (excluding the terminal ±20 marker,
and turn 10 AR7/AR8 where the engine's final scoring and the log's bookkeeping legitimately
differ):

| | entries |
|:---|---:|
| engine disagrees with the score **field** | 4,500 |
| — engine matches the **narration** (field lags; engine correct) | 666 |
| — entry narrates no score, so nothing to check against | 3,822 |
| — **engine disagrees with the narration** | **12**, in 7 games |

Real disagreements are **12 entries across 7 games, magnitude 1-2**. Not 194 games, and not mean
2.2 VP. §8.6's "81% of converted games carry drift" was measuring the log's field lag.

**This largely clears §8.** The human VP arc is read from the engine's VP, and the engine agrees
with the log's narration nearly everywhere it can be checked. Re-deriving the arc from narrated
scores is still worth doing, but as confirmation rather than as repair. The one caveat that stands
is the terminal marker: `victory_points` becomes ±20 when a game ends, so any turn-boundary sample
taken after a terminal reads the result rather than the score.

### 8.7.1 Asia Scoring under Shuttle Diplomacy — RESOLVED: the log is wrong, not the engine

Six of the twelve real disagreements are one game, **replay 259**, where the engine sits **+1**
above the narration from turn 7 AR3 onward -- through turns 7, 8 and 9. It starts here:

```
Turn 7, USSR AR3: Asia Scoring: Event: Asia Scoring
Shuttle Diplomacy is no longer in play.
USSR gains 2 VP. Score is US 17.
```

The engine awards the USSR **1**, the log **2**. Working the region from the log's own board
(USSR holds North Korea, South Korea, Japan and Pakistan; the US holds India; the USSR also holds
Afghanistan, the US eight more non-battlegrounds):

* Neither side dominates -- the US has more countries, the USSR more battlegrounds -- so both
  score Presence, 3 each.
* Shuttle Diplomacy removes one USSR battleground. Taking Japan removes the battleground, the
  country, *and* the USSR's superpower-adjacency bonus, since Japan is the Asian country adjacent
  to the US.
* USSR 3 + 3 battlegrounds + 0 adjacency = 6; US 3 + 1 = 4. Net **USSR 2**, which is what the log
  says.

The engine's 1 is what you get from **two** battlegrounds coming off rather than one. The
adjustment lives at `engine/src/scoring.cpp:70-85` and decrements the battleground count, the
country count and (in Asia) the adjacency in one block, guarded by
`SHUTTLE_DIPLOMACY_ACTIVE`; the flag is cleared in two places, `scoring.cpp:182-184` and
`scoring.cpp:287-289`. Double application is the obvious candidate and is **not yet verified** --
the arithmetic above establishes the symptom and which side is right, not the mechanism.

Per invariant 11 this is reported, not fixed.


### 8.7.2 The Shuttle/Japan divergence is a log fault, and is now corrected by rule

Settled by the project owner: this is a known, reproducible property of the ts-replayer logs. When
Shuttle Diplomacy is in play, Asia is scored, and the USSR holds Japan, **the log keeps the USSR's
bonus for controlling a country adjacent to the United States** — which the card has just removed
along with Japan — and pays the USSR 1 VP too many. **The engine is right; the log is not.** My
arithmetic in 8.7.1 reached the wrong conclusion from the same numbers.

It was already handled for the one game it had been diagnosed in: `_LOG_MISCOUNTED` carried
`259: {(7, "AR3", "USSR"): 1}`, so the six "disagreements" 8.7 attributed to replay 259 were an
artefact of my measurement, which compared the engine against the narration without applying the
offset the converter applies. **The real count of engine/log disagreements is 6, not 12.**

Now recognised by rule rather than by entry (`_shuttle_japan_asia_miscount`): Shuttle Diplomacy
flag set, the entry's events name Asia Scoring, and the USSR controlling Japan — all read before
the entry is driven, since scoring consumes the flag. Where it fires, the engine's score stands
and every score the log states afterwards is compared against its own number plus the offset,
which is the existing `_LOG_MISCOUNTED` machinery.

**Generalising found a second game.** The rule fires on **replay 127** as well as 259. 127 was
never listed, so until now it converted by taking the log's score — carrying a USSR VP total 1 too
high, and the board that does not pay it, into the training data. That is the whole argument for a
rule over a list: a list only covers the games already downloaded.

`_LOG_MISCOUNTED` is now empty and kept for anything genuinely particular to one game. All 282
games still convert with 0 failures.


### 8.7.3 The log's score is what training data carries, and what is left after that

**Corrected intent.** 8.7.2 kept the engine's rules-correct score and offset the log. That is
backwards for this corpus. The players were reading the app's score, not the rulebook: they played
the slightly wrong game, and every decision after the Shuttle/Japan Asia scoring was made against
the number the log shows. Training data has to be the position the human actually saw, so the
converter now **adopts the logged score** and carries it forward; only the assertion for that one
entry is dropped, and no offset is accumulated. `_LOG_MISCOUNTED` stays empty.

**What is left.** Sweeping all 282 converted games, comparing the engine against the log's
*narration* (never the lagging score field) at the point the converter compares:

| | entries |
|:---|---:|
| engine disagrees with the narration | 21 |
| — terminal ±20 result marker, not a score | 10 |
| — Shuttle/Japan Asia scoring, log adopted | 1 |
| — **unexplained** | **10**, in 8 files |

Of the 10 unexplained, four are end-of-game, where the engine's final scoring and the log's
bookkeeping legitimately differ:

| replay | diff | where |
|---:|---:|:---|
| 117 | −14 | T10 AR7, The China Card |
| 234 | −10 | T10 AR7, South America Scoring |
| 16, 43 | +6 | T10 AR8, Mideast Scoring |

That leaves **six midgame entries in five distinct games** (16 and 43 are duplicate downloads of
one game), all of magnitude 1–2:

| replay | diff | where |
|---:|---:|:---|
| 16, 43 | −2 | T4 AR7, Alliance For Progress |
| 28 | −2 | T4 AR7, Special Relationship |
| 57 | −2 | T3 AR6, Arab-Israeli War |
| 71 | +2 | T3 AR6, Mideast Scoring |
| 65 | +1 | T6 AR7, Che |

Four of the six sit at turn 3 AR6 or turn 4 AR7 and are all worth exactly 2 on US entries, which
is suggestive of one shared cause rather than five unrelated ones. None is diagnosed. This is the
whole remaining VP disagreement between the engine and 282 human games — down from the "194 games,
mean 2.2 VP" of 8.6, which was measuring the log's own bookkeeping lag.


### 8.7.4 The midgame cluster is the military operations penalty — diagnosis confirmed

The six midgame disagreements of 8.7.3 are not engine errors. Traced at **replay 16, turn 4 AR7**:

* The log narrates `US gains 3 VP. Score is USSR 5.` — the score **after** the play and
  **before** the turn is cleaned up.
* The engine, at the point the converter compares, is already on **turn 5** at **−7 VP**, with
  military operations reset.
* Entering the round the USSR held 5 military operations to the US's 0 at DEFCON 2, which is a
  2 VP penalty to the US. −5 − 2 = −7.

Both numbers are right; they are taken at different moments. The log books the penalty on a later
entry — at turn 5 AR1 it narrates `Score is USSR 6`, which is the engine's −7 plus the 1 VP that
entry awards, and the two agree from there on. That accounts for the whole cluster: every one of
the six sits on a turn's last action round, and the magnitudes (2, 2, 2, 2, 1) are military
operations shortfalls.

**The obvious repair does not work, and the reason is worth recording.** The converter currently
skips the comparison whenever an entry crosses a turn boundary (`crossed_turn`), which leaves one
entry in eight unchecked. Comparing against the score as it stood before the turn moved would
restore that coverage — but there is no such observable moment. Wrapping `Engine.step` to capture
the score the instant before the turn number changes yields **−8**, not the −5 the log states,
because Alliance For Progress's own +3 and the turn cleanup are applied **within a single engine
step**. Between the event's VP and the penalty there is no step boundary to read.

Two ways to get the check, neither done:

1. **Engine-side**: record the score at the start of cleanup (or make cleanup its own step), so
   the pre-penalty value can be read. Per invariant 11 this needs the owner's approval.
2. **Log-side**: derive the expected penalty from the log, which prints both military operations
   totals and DEFCON, and check the engine's post-cleanup score against
   `narrated + min(ussr_ops, defcon) − min(us_ops, defcon)`. Self-contained, but duplicates a
   rule the engine already implements, so it is a differential check rather than a reconciliation.

Until one of them exists, a turn's last action round remains unverified for score, and 8.7.3's
six "unexplained" midgame entries should be read as explained.


### 8.7.5 The turn-ending score is checked now, but only to within the cleanup

`DecisionContext.pending_roll_type` exposes which chance node is pending, so the converter can
find the `TURN_CLEANUP` pause and read the score on both sides of it. A turn's last action round
is no longer skipped: **212 such entries in the first 60 games** are now checked where they were
not.

**The check is weaker than intended, and the reason is in the logs.** Which moment the narration
describes is not something the log states:

| replay | entry ends with | moment |
|---:|:---|:---|
| 16 T4 AR7 | `Score is USSR 5` | **before** cleanup |
| 182 T1 AR6 | `Turn 9, Cleanup: US gains 3 VP. Score is even` | **after** cleanup |
| 221 T2 AR6 | `USSR gains 2 VP. Score is US 2` | **after** cleanup, unlabelled |

The word "Cleanup" cannot separate them — replay 221 is a cleanup award with nothing marking it as
one, and keying on the word failed that game (and its duplicates 265, 299) while passing 182. So
the assertion is that the engine's score equals **one of the two moments** the narration could be
describing, and it fails when it is neither. Anything off by more than that turn's Military
Operations deficit is caught; an error the size of the deficit is not.

Pinning the exact moment would need the log to say which it means, and it does not. Worth
revisiting only if a real disagreement is ever found hiding in that gap.

All 282 games still convert with 0 failures.

---

## Hands: what the log evidences, and what the solver supplies

The log never states a hand in full, so the converter solves for one. These sections are
about the difference between a card the log *evidences* and a card the solver *supplied* to
satisfy its constraints -- a distinction that decides whether an apparent human mistake is
the human's or ours.

### 9.4.1 The wider consequence: a quarter of the corpus's offered actions are unevidenced

The same measurement, applied to every card decision rather than to dominance pairs:

| | |
|:---|---:|
| human card decisions measured | 33,511 |
| cards offered across them | 187,458 |
| **not in the log's hand list** | **44,247 (23.6%)** |
| decisions offering at least one such card | 27,267 (**81.4%**) |

Part of that is the solver padding hands the log underdetermines, and part is the log's own
incompleteness — the 3.5% above proves the second exists and the two cannot be separated with what
the log records. Either way, **the human corpus offers the policy a choice among cards there is no
evidence the player held, at 81% of its card decisions**, and those cards are in the action masks
the BC warmup trains against.

That is a data-quality finding rather than a bug: nothing is *forced*, and the converter's own
rules are intact. But it bounds what the corpus can teach about card selection, it is a plausible
contributor to §9's negative result, and it means any future measurement conditioned on hand
contents needs the same log-evidenced restriction applied here.

### 9.6 The dominance relation as evidence about the hand

**Idea (owner's).** §9.5 established that humans never take the dominated side of a trap discard
where the log records both cards -- 0 of 87. So the discard is evidence about the rest of the hand:
if a player gave up their own or a neutral card to Quagmire or Bear Trap, an opponent recurring
event of that printed Ops was very probably not in their hand. The solver can use that.

**Implementation.** `tools/lib/ts_replayer_hands.py` records which card went to the trap
(`GameFacts.trap_discard`) and adds a soft clause against holding any equal-Ops opponent recurring
event that turn. Deliberately soft, and deliberately a *second* clause on the same literal so its
weight adds to that card's existing hold cost rather than replacing it. It is a statement about how
people play, not about what the rules permit, so everything the log establishes stays in the hard
model and outranks it. Five Year Plan, the China Card, scoring cards and one-time events are
excluded, matching `ai/eval/dominance`.

**Result.** Apparent dominated trap discards across the corpus:

| | dominated | rate |
|:---|---:|---:|
| before | 11/98 | 11.2% |
| **after** | **7/94** | **7.4%** |
| — Bear Trap | 2/61 | 3.3% (was 7.8%) |
| — Quagmire | 5/33 | 15.2% (was 17.6%) |

Restricted to pairs the log evidences on both sides it stays **0/87**, as it already was -- those
cases were never the problem. What moved is the residue: four hands that previously held a
dominating card the log does not list no longer do.

**The remaining seven are not the solver's to fix.** In those the hard constraints pin the
dominating card into the hand, so the log itself implies it was held even though that turn's hand
list omits it. That is the log's incompleteness (§9.4: 3.5% of cards humans are recorded playing
are missing from their own hand lists), not a preference the solver got wrong, and forcing it would
be exactly the guessing the corpus rules forbid.

**Unchanged by the rebuild:** 282 games convert with 0 failures, 144,844 samples, 84,073 with a
value target. The dataset was rebuilt on the new hands.


### 9.7 There is no game where a human provably held the dominating card

The seven trap discards §9.6 still scores as dominated were checked one by one for what the log
says about the alternative. In **all seven** the dominating card is absent from that turn's hand
list *and* is never played by that side during that turn. None of them is evidence of a human
passing over a card they demonstrably held.

Where those cards do appear is the pattern:

| replay | turn | alternative the solver placed | the log has it at |
|---:|---:|:---|:---|
| 71 | 5 | Duck and Cover | T1 (played), T6 (hand list, played AR1) |
| 80 | 6 | Liberation Theology | T7 (hand list, played AR7) |
| 179 | 4 | The Voice of America | T6 (hand list, played AR1) |
| 184 | 7 | Arab-Israeli War | T3, T9 |
| 198 | 5 | Socialist Governments | T2, T7 |
| 283 | 5 | Arab-Israeli War | T2, T6 |

Every one is a card the log places in that player's hands **in other turns**, which the solver has
carried into the turn in question -- one to two turns before the log first lists it. Carrying a card
over is ordinary and the hand lists are demonstrably incomplete (§9.4), so the placements are not
illegal; they are simply unevidenced, and each manufactures the appearance of a mistake.

**So the corpus contains zero proven violations of the dominance relation.** §9.5's "0 of 87 where
the log records both cards" is not a restriction that hides the counterexamples -- there are none
to hide. Across all 94 decidable trap discards, every apparent human error rests on a card the log
does not put in that hand at that time.

**A refinement this suggests, not made.** The solver could pay a cost for holding a card in turns
before the log first lists it, which is what all six distinct cases have in common. It would want
care: carry-over is real, the lists are incomplete, and a hard version would contradict the corpus
rule against forcing what the log does not state. Worth trying as another soft clause if the
residue matters.


### 9.8 The Junta counterfactual, and two bugs it found in §9.6's heuristic

**Question (owner's).** In replay 80, would the objective be better with **Junta** in the turn 6 US
hand instead of Liberation Theology? And carry-over should not be penalised (§9.7's suggestion),
because carry-over is the primary way a hand is reconstructed at all.

**Junta specifically: no, the log forbids it.** Pinning Junta into that hand is **unsatisfiable**.
It appears in this game only in the turn 7 US hand list and is never played, so at turn 6 the hard
constraints place it elsewhere.

**But the probe found the real problem.** Liberation Theology is *not* forced either — pinning it
out is satisfiable, at total soft cost **152** against the chosen **151**. A card responsible for
two of the apparent human mistakes was being decided by a margin of **one**. It should have been
carrying §9.6's +120 dominance penalty, and it was not, for two reasons:

1. **Only the last trap discard of a turn was kept.** A trap holds until the escape roll lands, so
   one turn can force several discards. At turn 6 the US discarded three times — Warsaw Pact
   Formed, Our Man in Tehran, Nuclear Subs — and `trap_discard` recorded only the third.
2. **The two sides of the pair were given the same eligibility.** One-time (starred) events are
   excluded as the *dominant* side, because discarding one removes it from the game permanently
   and that is a different argument. They are perfectly ordinary as the *discarded* side, and
   `ai/eval/dominance` splits exactly there. Treating them alike meant all three of the US's
   starred discards were ignored and the turn contributed no evidence at all.

Both fixed: every discard in the turn is kept, and `_discard_excluded` is now separate from
`_dominance_excluded`.

**Effect on the corpus:**

| | dominated | rate |
|:---|---:|---:|
| §9.3, with the headline bug | 25/113 | 22.1% |
| headlines excluded (§9.5) | 11/98 | 11.2% |
| first heuristic (§9.6) | 7/94 | 7.4% |
| **heuristic corrected** | **3/90** | **3.3%** |
| — Bear Trap | 1/60 | 1.7% |
| — Quagmire | 2/30 | 6.7% |

282 games still convert with 0 failures. The rebuilt dataset is 144,845 samples with 84,074 value
targets, one more than before — a hand changed somewhere and a decision came with it.

**Carry-over is not penalised**, per the owner: it is how hands are reconstructed in the first
place, and costing it would attack the mechanism rather than the error. The three remaining cases
stand as they are.


### 9.9 Why the log allows Liberation Theology at turn 6 but not Junta

Both cards appear in replay 80 only in the turn 7 **US** hand list, so on the face of it the solver
should treat them alike. It does not, and the reason is a line outside the hand lists.

| | Junta | Liberation Theology |
|:---|:---|:---|
| turn 7 US hand list | listed | listed |
| who actually spent it | **USSR** — `USSR Headlines Junta` | **US** — space race at AR7 |

The hand list and the play disagree about Junta, and the play wins: this is one of the 4
`hand_reattributions` in this game, the mechanism `tools/README.md` describes for lists that give a
card to the wrong side. So Junta is the USSR's at turn 7.

From there the constraint follows without any preference being involved:

1. The USSR spent Junta at turn 7, so Junta was in the USSR's hand at turn 7.
2. **All seven** of the US's turn 6 action rounds are recorded — Warsaw Pact Formed, Our Man in
   Tehran, Nuclear Subs, The Voice of America, Shuttle Diplomacy, How I Learned To Stop Worrying,
   SALT Negotiations — and Junta is not among them.
3. A card in a hand that is not played or discarded stays there, so Junta in the US hand at turn 6
   would still be in the US hand at turn 7.

Which contradicts (1). There is a reshuffle before turn 7, so a discard-and-redraw route exists in
principle — but (2) closes it, because the US's turn is fully recorded and Junta was not discarded.

Liberation Theology has none of that: the US spent it at turn 7, so holding it from turn 6 is
consistent, and only the soft preferences decide whether it was. §9.8 established that they decide
it by a single unit of cost, which is why it looked arbitrary.

**No fault found.** The asymmetry is entailed by the log, and the evidence for it is in the headline
line rather than in the hand lists the question naturally looks at.


### 9.10 Two corrections to the hold cost, from the owner

**1. A one-time card of your own, in a quiet turn, is not evidence against holding.** `_hold_cost`
charged a flat 40 for holding any card of your own side, on the reasoning that a hand is for
spending. That is right for a recurring event and wrong for a one-time one: it is played once, at
a moment that suits it, and saving it is ordinary. §9.9 showed the charge doing real work — it was
the whole of the +38 that rejected Sadat Expels Soviets, a starred US card the US demonstrably held
the following turn.

The charge is now dropped for a one-time card of your own **when that side fired none of the
opponent's events that turn**. A turn with no opponent event fired is consistent with a hand that
simply held no opponent cards, so nothing about it argues against having kept your own. Where an
opponent event *was* fired the side was holding opponent cards and had a choice about what to keep,
so the charge stands. **CIA Created** and **"Lone Gunman"** keep it always: they are played to see
the opponent's hand, and nobody sits on them.

**2. Holding the opponent's card while giving your own to a trap is now heavily penalised, at any
Ops.** §9.6's clause required the two cards to print the same Ops, mirroring the dominance
*measurement*. Under a trap that is too narrow: you cannot play anything while trapped, so the Ops
of what you give up buys nothing, and keeping the opponent's event is worse whatever it prints.
The clause now applies to any opponent card except Five Year Plan, the China Card and scoring
cards, and the weight goes 120 → **240**, above the top of `_hold_cost`'s range.

One-time opponent events are no longer excluded either. `ai/eval/dominance` excludes them so that
every *measured* pair is strictly defensible; as a claim about what a hand held the direction is
the same and stronger, since nobody keeps the opponent's one-time event while giving up their own.
The measurement module is unchanged — only the solver's preference is widened.

**Effect.** 282 games still convert with 0 failures.

| | before | after |
|:---|---:|---:|
| dominated trap discards | 3/90 (3.3%) | 3/93 (3.2%) |
| decidable space plays | 712 | 749 |
| dataset samples | 144,845 | 144,839 |

The trap figure barely moves because §9.8 had already taken it near the floor; what changed is the
hands themselves, which is what the dataset rebuild reflects. 961 tests pass.

---

## Corpus composition

What the 300 downloaded files actually contain.

### 14.4 The corpus is 11% duplicates, and 4 held-out games leak

Found while checking the replay 315/316 exception: 315 and 316 have identical `all_turns` and differ
only in id and source URL. Fingerprinting every game's entries:

**300 files, 266 distinct games, 25 duplicated groups, 34 redundant copies (11.3%).** One game
appears nine times — ids 90, 214, 215, 216, 217, 218, 254, 298, 309.

Duplicates are weighted twice in the BC warm start and twice in every injection batch. Worse, the
train/held-out split is by replay id, and duplicates carry different ids: under the seed-7 split,
**3 duplicate groups straddle the boundary and 4 of the 56 held-out games (7.1%) were also trained
on** — held-out 125 is trained-on 118, held-out 216/217 are trained-on 90/214/215/218/254, held-out
235 is trained-on 232/233.

That is the same failure as the first synth+human warm start reading 65% "held out", in a milder
form: the 49.33% figure in §9.13 and every number derived from that split are inflated by whatever
7.1% memorisation is worth. Small, but it should be a fingerprint-based split, not an id-based one.
