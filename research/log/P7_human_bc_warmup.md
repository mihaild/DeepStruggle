# P7 (human data) — BC warmup on the human corpus, and what the corpus actually carries

Records §9 of the original `experiments.md` in full: experiment E3 and everything that came out of
it. This is the first half of the human-data programme that [`../plans/P7_human_data.md`](../plans/P7_human_data.md)
now queues — behaviour cloning from the ts-replayer corpus as an *initialisation*, measured on
`--arch v2` at 80M steps, 512 envs, `blunder_aware` + K=40, post-E1 engine, legacy observation. The
arc is: the human warmup loses (§9), the prior washes out inside 2% of training (§9.1), how long BC
should run (§9.2, §9.13), whether the corpus carries the dominance signal at all (§9.3 → §9.4 →
§9.5, where the first two readings are both wrong and the third corrects them), and the first
injection pilots (§9.12–§9.14). Conversion mechanics are in
[`P7_replayer_conversion.md`](P7_replayer_conversion.md); the injection dose experiments and the
ablation that killed injection are in
[`P7_human_injection_and_its_cost.md`](P7_human_injection_and_its_cost.md). **This file is
append-only history** — §9.3 and §9.4 keep their wrong numbers and their `SUPERSEDED` labels rather
than being corrected in place, and nothing is edited to match current belief.

---

## 9. Human-corpus BC warmup (E3, first pair) — NEGATIVE

**Question.** §4.8 argued that "one human game shows the reversal that RL needs thousands to
notice". Does warming up on the 280-game human corpus instead of on self-play demonstrations
produce a better agent?

**Setup.** arch v2, `--train-steps 80000000` (both arms exactly 80M), 512 envs, `blunder_aware`
+ K=40, run in parallel on one 4090, snapshots every 2M steps. One flag apart: the BC warmup
checkpoint. Arm A warmed on 5,000 self-play games from `dec_turns40` (79.2% top-1 after 2 epochs);
arm B on the human corpus, 144,844 samples with value targets masked on the 149 games whose
recording stops (45.4% top-1). Engine as of `68f155b` minus the Independent Reds fix, which landed
mid-run and applies to neither arm.

**Result** (tournament, 1,000 games per pair):

| | Elo | vs the other arm | vs `dec_turns40` | vs HeuristicBot |
|:---|---:|---:|---:|---:|
| arm A, self-play warmup | **1852.9** | **57.4%** | 56.8% | 84.1% |
| `dec_turns40` (previous best) | 1824.7 | — | — | **90.5%** |
| arm B, human warmup | 1810.3 | 42.6% | 49.2% | 85.8% |

**Game shape and map coverage** (256 self-play games each, temperature 0.1):

| | mean final turn | reaches turn 9 | empty BG turn 5 | empty BG turn 8 |
|:---|---:|---:|---:|---:|
| arm A, self-play | 6.58 | 0.270 | **10.83** | **6.70** |
| arm B, human | 6.93 | 0.336 | 11.93 | 7.99 |
| `dec_turns40` | **7.42** | **0.398** | 11.51 | 7.06 |

**Verdict: the hypothesis is not supported.** Arm A beats arm B head-to-head 57.4% over 1,000
games, which is far outside the ~1.5 point run-to-run envelope of §7.2. Arm B is also *worse* on
the measure the corpus was supposed to fix: it leaves **7.99** battlegrounds empty at turn 8
against arm A's 6.70, where §4 identifies unclaimed battlegrounds as the deficiency. The one thing
arm B does better than its control is game length — 6.93 turns and 33.6% reaching turn 9 against
6.58 and 27.0% — which is the §4.6 axis, but it does not convert into strength.

**A confound that matters, and that this pair cannot separate.** The two warmups do not start the
arms from equally good policies: 79.2% top-1 against 45.4%, and arm B's first snapshot beat
HeuristicBot 37.5% against arm A's 80.0%. So this compares "human data" and "a much weaker
initialisation" at once, and 80M steps may simply not be enough for arm B to close a gap it began
with. What it establishes is narrower than the question: *at 80M steps, warming on the human corpus
alone is worse than warming on self-play demonstrations*. It does not establish that human data is
unhelpful.

**The remaining arms are now the interesting ones.** The synthetic→human fine-tune gets the strong
initialisation *and* the human data, which is exactly the combination this pair could not test; and
the no-warmup arm would say how much either warmup is worth at all. Only two arms fit in 24 GB, so
they were always a second round.

**Also worth noting: `dec_turns40` did not lose its crown cleanly.** It beats both new arms against
HeuristicBot by 4-6 points and leads both on game length and turn-9 reach, while losing to arm A
head-to-head 43.2%. Beating the model that beats the heuristic more, while beating the heuristic
less, is a real intransitivity and a reminder that a single opponent is not a ranking.

**Caveat.** One seed per arm. §7.2's noise floor covers tournament measurement, not run-to-run
training variance, which this repository has never measured. A 57.4% head-to-head is comfortably
outside measurement noise; it is not known to be outside seed noise.


### 9.1 The human prior washes out in 2% of training — which explains §9

**Question.** §9 could not separate "human data is worse" from "the human arm started from a much
weaker policy". Measuring whether the prior survives at all separates them, and costs nothing: the
checkpoints already exist.

**Setup.** Top-1 agreement with human moves on a fixed probe of 20,000 corpus decisions, evaluated
at all 42 snapshots of both arms. Arm A never saw a human game and is the floor.

| | after BC | 1st snapshot (~2M steps) | minimum | final (80M) |
|:---|---:|---:|---:|---:|
| arm B, human warmup | **45.7%** | 33.1% | 29.0% | **32.5%** |
| arm A, self-play warmup | 33.6% | 31.4% | 30.6% | 32.6% |

**The prior is gone inside the first 2M steps — 2.5% of the run — and never returns.** From there
the two arms agree with human play *identically*, both sitting at 31-34% for the remaining 78M
steps and finishing within 0.1 points of each other. Roughly 32% is what RL converges to whatever
it was initialised from.

**So §9 was not testing what it looked like it was testing.** Past the first snapshot there was no
human prior left to test; the arms differed only in where they started, which is precisely the
confound §9 flagged as unresolvable from that pair. It is now demonstrated rather than suspected.

**Consequence.** A warmup variant cannot answer this question -- not a longer one, not a mixed
synthetic+human one, not a fine-tune. Anything delivered as an initialisation decays to the same
attractor within 2M steps. Human data has to be applied as something that *persists*, which is
what E4's pinned `π_ref` is: a KL anchor held throughout training rather than a starting point.

### 9.2 How long to train the human BC — about 8 epochs, and it matters only for E4

**Question.** The E3 warmup ran 2 epochs and reached 45.4% top-1. Would training it longer help?

**Setup.** Split by **game**, not by sample: positions within a game are heavily correlated, so a
sample split scores the model on positions it has effectively already seen. 224 games train,
56 held out (117,025 / 27,819 samples), fresh v2 net, agreement after each epoch.

| epoch | 0 | 1 | 2 | 4 | 6 | 8 | 10 | 12 |
|:---|---:|---:|---:|---:|---:|---:|---:|---:|
| train | 18.7 | 40.9 | 41.5 | 45.3 | 47.0 | 49.0 | 52.9 | 57.5 |
| **held out** | 19.0 | 41.0 | 41.6 | 44.8 | 45.9 | **46.9** | 47.5 | **47.8** |

**Two epochs was undertrained** -- held-out agreement climbs from 41.6% to about 47% by epoch 8, so
the E3 warmup left roughly five points on the table. **But it plateaus there**: epochs 8 to 12 buy
0.8 points of held-out agreement while the training score gains 8.5, so the model is memorising
games from that point on. About 8 epochs is the useful end.

**It would not have changed §9.** By 9.1 a 48% initialisation decays to the same ~32% attractor as
a 45% one, and just as fast. Where it does matter is E4: there the human policy is the anchor
rather than the starting point, it persists for the whole run, and its quality is the experiment.
Build that net at ~8 epochs.


### 9.3 Do the humans respect the dominance relations? — TRAP NUMBERS WRONG, SUPERSEDED BY 9.5

**Question.** §4.8 called the dominance result "the strongest argument yet for demonstrations,
since one human game shows the reversal that RL needs thousands to notice". That is a claim about
the corpus which had never been checked against the corpus. Before engineering any way to keep
human data in the policy, it is worth knowing whether the data carries the signal.

**Setup.** `ai/eval/human_dominance.py`, over all 280 converted games. For a policy §4.8 can ask
how the pair is *ranked*, because there is a distribution; for a human there is only the move, so
the measure is the stricter half — did they choose the dominated card while a dominant alternative
of equal Ops sat in the same hand.

| | humans | control | K=40 |
|:---|---:|---:|---:|
| Quagmire, chose dominated | **24.3%** (9/37) | 37.5% | 30.7% |
| Bear Trap, chose dominated | **21.1%** (16/76) | 32.8% | **18.2%** |
| Space race, chose dominated | **0.4%** (3/712) | 18.1% | 12.6% |

**On the space race the corpus is emphatic.** Humans spend the opponent's card on the track
essentially always: 3 errors in 712 decidable plays, against our agents' 12.6-18.1%. That is a
factor of 30 to 45, on the largest sample of the three, and it is exactly the kind of local,
position-independent preference a demonstration can teach and RL evidently does not find.

**On the traps it is much weaker than §4.8 implies.** Humans are better than the control on both,
but **K=40 already beats them on Bear Trap** (18.2% against 21.1%) and is close on Quagmire. So on
trap discards there is little left for the corpus to teach the current best agent.

**Verdict.** The premise is *partly* confirmed, and narrower than the rhetoric it was based on.
There is one clear, well-evidenced class where human play is strongly better and ours is not, and
two where our best model has already caught up. Set against §9 -- where the human-warmed arm was
weaker overall despite inheriting these preferences -- the case for spending GPU time on a
mechanism to retain human data rests on that one class.

**A denominator caution, in the same family as §4.8's own.** The first version of this counted
every opponent-card space play as a correct decision and reported 0 errors in 265 "decidable"
plays. Most of those were not decisions: no equal-Ops own card was in hand, so nothing could have
been chosen differently. Requiring both kinds of card at the same Ops cut the denominator to 712
across the corpus and is what makes the 0.4% meaningful. §4.8 records the same trap from the other
direction — measuring against all space plays reads 2.4% and badly understates the agents' rate.


### 9.4 Half the humans' apparent trap mistakes are ours, not theirs — PARTLY WRONG, SUPERSEDED BY 9.5

**Question.** §9.3 put humans at 22.1% dominated on trap discards, better than the control but not
clearly better than K=40. Are those human mistakes, or does our hand reconstruction hand them a
card they never held? A dominance pair says "you discarded X when Y was available"; if Y is the
solver's padding, we invented the alternative and the error is ours.

**Setup.** Each of the 25 dominated discards checked against the log's *own* per-turn hand list --
the only independent record of what a player held.

| | count |
|:---|---:|
| dominant alternative **is** in the log's hand list | **13** |
| dominant alternative is **not** — reconstruction supplied it | **12** |

**Recomputed over pairs where the log evidences both cards:**

| | as §9.3 measured | log-evidenced pairs only |
|:---|---:|---:|
| trap discards, dominated | 22.1% (25/113) | **12.1%** (12/99) |
| — Quagmire | 24.3% | **6.7%** (2/30) |
| — Bear Trap | 21.1% | **14.5%** (10/69) |

Against §4.8's agents that reverses the reading: Quagmire **6.7%** against the control's 37.5% and
K=40's 30.7%; Bear Trap **14.5%** against 32.8% and 18.2%. Humans are better than both agents on
both traps, where §9.3 had them level with K=40.

**But the log's hand lists are not complete either, so this is a bound, not a value.** Measured
over all 33,511 human card decisions: **3.5%** of the time the card the human is *recorded playing*
is itself absent from that turn's hand list. A list that omits cards the player demonstrably held
cannot be treated as ground truth. So 12.1% is a **lower** bound on the human error rate and 22.1%
an **upper** one. The useful conclusion survives either way: at the upper bound humans beat the
control, at the lower bound they beat K=40 as well.

### 9.4.1 The wider consequence: a quarter of the corpus's offered actions are unevidenced

23.6% of the cards offered across 33,511 human card decisions are absent from the log's own
hand list, and 81.4% of those decisions offer at least one such card. Part is the solver
padding hands the log underdetermines and part is the log's own incompleteness, and the two
cannot be separated from what the log records.

**It bounds what the corpus can teach about card selection**, is a plausible contributor to
§9's negative result, and means any measurement conditioned on hand contents needs the
log-evidenced restriction §9.4 applies. Detail and method in
[`P7_replayer_conversion.md`](P7_replayer_conversion.md).

### 9.5 The humans never once chose a dominated trap discard — 9.3 and 9.4 corrected

**Both earlier readings were wrong, and the second error was mine rather than the corpus's.** The
project owner checked two of the cases §9.4 called genuine and found the log says otherwise:

* **replay 63, turn 5.** The USSR *headlined* Che and discarded **Duck and Cover** at AR1. The log:
  `USSR Headlines Che`, then `Turn 5, USSR AR1: Duck and Cover: USSR discards Duck and Cover`.
  Duck and Cover is a US card, so that discard is the *dominant* choice.
* **replay 71, turn 6.** The USSR *headlined* Quagmire; the Bear Trap discards were Duck and Cover
  and then Camp David Accords.

**The converter is right in both.** Driving replay 63 gives headline USSR → Che, US → Red
Scare/Purge, then AR1 USSR → Duck and Cover, matching the log line for line. **The corpus does not
need reconstructing.**

**The fault was in the dominance driver.** A headline play is a `SELECT_CARD` decision and the trap
flag is set in that state too, so a headline made while trapped was scored as a trap discard — and
scored as an error nearly every time, because the rule says "discard the opponent's recurring
event" while a headline is where you play your own best card. **14 of 15 decidable headline
decisions were counted as errors**, against 11 of 98 in the action rounds where the rule belongs.

**Corrected, with both fixes applied:**

| | trap discards dominated |
|:---|---:|
| §9.3, as first measured | 22.1% (25/113) |
| headline plays excluded | 11.2% (11/98) |
| **and restricted to pairs the log evidences on both sides** | **0.0% (0/87)** |

Bear Trap 0/59, Quagmire 0/28. **Every remaining apparent mistake involved a dominant alternative
the log does not list** — a card the reconstruction supplied, or one the log omits. On decisions
where the log records both cards, the humans in this corpus never chose the dominated one.

**The bound still applies in the other direction.** The log's hand lists are incomplete (§9.4:
3.5% of the cards humans are recorded *playing* are absent from them), so restricting to
log-evidenced pairs is conservative and may discard genuine decisions. The true rate lies between
0% and 11.2%. At either end it beats every agent we have measured: traps 18.2-37.5%, space race
12.6-18.1%.

**Verdict, replacing §9.3's.** The corpus carries the dominance signal on all three classes, not
one. §4.8's argument for demonstrations stands as written; §9.3's doubt about it was an artefact of
this driver. What that means for §9's negative result is unchanged — the corpus contains the signal
and BC warmup still failed to deliver it, which is a statement about the mechanism, not the data.

**Method note.** Both of §9.4's headline examples were presented as the *genuine* cases, the ones
left after the padding correction. They were the least reliable in the set. A filter that admits a
decision type it was not written for will do most of its damage in the cases that look cleanest.


### 9.6-9.10 Hand reconstruction: the dominance clause, the hold cost, and two counterfactuals

**Moved to [`P7_replayer_conversion.md`](P7_replayer_conversion.md).**
How §9.5's result was fed back into the hand solver as a soft clause, why the seven remaining
apparent mistakes are the log's incompleteness rather than the solver's error, and two
corrections to the hold cost. Conversion mechanics; the finding about human play they rest on
is §9.5 above.

### 9.11-9.11.2 Agreement with human play — how the measure is defined

**Moved to [`../method/human_play.md`](../method/human_play.md).** A card played for Operations spends its points one at
a time and the log's order is whatever the recording happened to write, so agreement scores a
play on the multiset of countries rather than the sequence -- except for coups and
realignments, where the board changes between points and the order is real. Reordering is
worth about 0.2 points; the measure is reported alongside a strictly-ordered figure so the
difference is always visible.

Numbers measured *with* it stay in this file: §9.12 onward, §14 and §18.4.

### 9.12 Injection frequency vs alignment — quick arms, INCONCLUSIVE

**Question.** §9.1 showed a BC warmup washes out in ~2M steps. Does interleaving supervised steps on
human data during RL hold alignment up, and how does the frequency matter?

**Setup.** 30-minute budget. Four arms, all from the *same* synthetic BC init (35.0% agreement),
4M steps each, two at a time; one flag apart -- `--inject-every` 0/16/4/1 at weight 1.0, a separate
AdamW at lr 1e-4 on human batches of 512 with value targets masked by `has_outcome`. Agreement
measured per snapshot on a fixed 20,000-decision probe (§9.11's measure).

All four completed 4M steps with ten snapshots each, so the endpoints are at equal step counts:

| `--inject-every` | trajectory | end |
|:---|:---|---:|
| 1 | 35.0 → 34.5, range 32.8-35.8 | **34.5** |
| 4 | 35.0 → 33.9, range 33.2-35.2 | 33.9 |
| 0 (control) | 35.0 → 33.9, range 33.7-35.0 | 33.9 |
| 16 | 35.0 → 32.7, range 32.0-35.0 | **32.7** |

**The endpoints order monotonically with frequency**, and injecting *rarely* lands below not
injecting at all. That is the direction predicted when this was proposed: a dose rarer than the
~2M-step washout perturbs the policy without establishing anything, and RL spends the interval
walking back, so the arm oscillates rather than holds. Weak support, but it is the predicted sign.

**Verdict: still inconclusive.** The spread across all four arms is 1.8 points and the control alone
ranges 1.3 across its own snapshots. One seed at 4M steps cannot separate a monotone ordering from
four samples of the same wobble.

**The design flaw is the starting point.** Every arm began from the synthetic BC init at 35.0%,
which is already at the ~32-35% attractor RL converges to (§9.1). There was almost no alignment to
preserve, so the experiment measured whether injection can *raise* agreement rather than whether it
can *prevent* the washout it was built for. The informative version starts from the **human** BC
init at ~47% and asks whether injection stops the decay to 32%. That is one flag different and the
right thing to run next.

**Also worth noting:** at weight 1.0 even every-iteration injection did not pull agreement toward
the BC ceiling of ~47%. Either the weight is too low against the RL updates, or RL actively pulls
away from human play. Those are different problems and the human-init arm distinguishes them: if
injection holds ~47% there, the weight is fine and the synthetic start was the issue; if it decays
anyway, the pull is real and the weight has to rise.


### 9.12.1 The warmup nets are unharmed by the corpus rebuilds — reference figures

The human BC net was trained against an earlier build of the corpus, before the solver changes of
§9.6/§9.8/§9.10 and before the `play` column, and the measure itself changed in §9.11. Checked
against the corpus and the measure as they now stand, over **all 144,839 decisions** rather than a
probe:

| net | ordered | agreement |
|:---|---:|---:|
| `e3_warmup_human` (BC on the corpus) | 45.94% | **46.23%** |
| `e3_warmup_synth` (BC on self-play) | 33.64% | 34.23% |
| `inj_none` snapshot_0s | 33.64% | 34.23% |
| `inj_every1` snapshot_0s | 33.64% | 34.23% |

**Nothing broke it.** The human warmup still sits where it did — the training-time figure was 45.4%
strict in-batch, and the earlier probes read 46.0-46.8% on subsets; 46.23% over the whole corpus is
the authoritative number and should be quoted in preference to those.

**And the arms started where they were meant to.** Both `snapshot_0s` files agree with
`e3_warmup_synth` to the decimal, which confirms independently that §9.12's four arms all began
from the same synthetic initialisation and that `--warmup-checkpoint` loads what it says. That was
assumed rather than checked when those arms were described.


### 9.13 BC on human games from a self-play net — it works, and it works *better*

**Question.** §9.2 measured behaviour cloning from a fresh network: held-out agreement climbs to
about 47-48% by epoch 8 and then memorises. Does a net already trained on self-play demonstrations
reach the same place, and does that start help or hinder?

**Setup.** Split by game (224 train / 56 held out, 117,021 / 27,818 samples), the same seed as
§9.2, agreement measured with §9.11's measure on the held-out games after every epoch.

| epoch | 0 | 1 | 2 | 4 | 6 | 8 | 10 | 12 |
|:---|---:|---:|---:|---:|---:|---:|---:|---:|
| **from the self-play net** | 34.18 | 45.58 | 47.15 | 47.58 | 48.61 | **49.33** | 50.04 | **49.99** |
| from scratch | 20.56 | 42.32 | 43.49 | 44.52 | 46.26 | 47.08 | 47.91 | 48.24 |

**Yes, and it dominates the from-scratch curve at every epoch.** Starting from the self-play net is
ahead by 3.3 points after one epoch and still ahead by 1.8 at twelve, finishing at **50.0%** against
48.2%. The self-play pretraining is not something the human data has to overcome; it is a better
starting representation, and the human data builds on it.

That is worth its own line: **the best human-aligned net available is not the one trained on human
games alone.** `e3_warmup_human`, BC from scratch, measures 46.23% over the full corpus (§9.12.1).
Two epochs of the same data on top of the self-play net beats it, and eight epochs reach 49.3%.

**Consequence for the arm that was never run.** The synthetic-then-human fine-tune was proposed as
round two of E3 and postponed for VRAM. This is direct evidence its *initialisation* is the best of
the three tried -- better aligned than human-only BC and far better than self-play alone -- which
removes the confound §9 could not escape: a human-data arm no longer has to start from a weaker
policy than its control.

**Caveat.** Both curves are still rising slowly at twelve epochs, so neither ceiling is established;
what is established is the gap between them, which is stable across every epoch measured. And the
from-scratch numbers here run about half a point above §9.2's on the same split, which is the
corpus having been rebuilt since (§9.6, §9.8, §9.10) -- the comparison inside this table is
like-for-like, comparisons across sections are not.


### 9.14 Injection from a human start — it halts the washout

**Question.** §9.12's arms all began from the self-play init, already at the ~32-35% attractor, so
they could only ask whether injection *raises* agreement and answered inconclusively. This is the
version they should have been: same four-million-step budget, same settings, but starting from the
human BC net at **47.2%**, where there is something to lose.

| snapshot | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 |
|:---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| control, no injection | 47.2 | 43.0 | 39.0 | 36.7 | 34.8 | 34.0 | 34.4 | 32.1 | 30.9 | **30.9** |
| `--inject-every 1` | 47.2 | 42.9 | 40.8 | 39.6 | 37.8 | 38.2 | 37.0 | 37.7 | 38.3 | **38.3** |

**Injection does not prevent the decay, it stops it.** Both arms fall for the first two snapshots
and then separate: the control keeps going all the way to 30.9%, reproducing §9.1's washout, while
the injected arm flattens at 37-38% and stays there for the second half of the run. The gap at the
end is **+7.4 points**, against a run-to-run range of about 1.3 (§9.12). This is not noise.

**It also settles what §9.12 could not.** The 0.6-point difference there was not evidence that
injection is weak; it was evidence that an arm starting at the attractor has nothing to preserve.
Same mechanism, same weight, same budget -- only the starting point differs, and the effect goes
from 0.6 to 7.4.

**What it does not do is hold the BC level.** 47.2% → 38.3% is still nine points lost, so at weight
1.0 injection is fighting the RL gradient to a draw somewhere below where it started rather than
holding its ground. Weight is the obvious next knob, and the one this pair does not vary.

**One oddity worth recording.** The control ends at 30.9%, *below* the 33.9% a self-play-initialised
control reached on the identical budget (§9.12). Starting closer to human play and then training
away from it appears to overshoot past where you would have been having never started there. One
seed, so it may be nothing -- but it is the opposite of what you would guess.
