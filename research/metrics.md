# Measurement

What the numbers in [`experiments.md`](experiments.md) can bear: how each measure is defined,
what a tournament result is reproducible to, how much of a run's rating is just where it
stopped, and the instruments that reported confident numbers while measuring nothing.

`experiments.md` records what was learned about training, design and data. This file records
whether it could have been learned -- the two are separated because they are read at different
moments, and because a result and the doubt about it should not have to be read together.

**Section numbers are the ones these entries were first written under** in `experiments.md`, so
every reference that names one still resolves. A gap is a section that stayed behind.

> **Distrust a measurement before trusting a result.** Seven separate diagnostics in this repo
> reported confident numbers that were wrong. Every one of them looked plausible. When an
> experiment produces a surprising result, the cheapest first hypothesis is that the instrument
> is broken -- check that before building on the finding.

---

## 1. Measurement bugs found (read this before trusting any older number)

Four defects in the evaluation and diagnostic code, three of them the same defect in three
different files. All are fixed; all invalidate numbers logged before their fix.

Three more of the same character were found later and are below under the numbers of the
experiments that turned them up: **§8.5**, where the forced-win metric both over- and under-counts,
and **§23.1**, where a model was fed the wrong observation layout and misread it without complaint,
a tournament nearly rated one checkpoint twice, and a snapshot sort returned the wrong four
snapshots for one arm while returning the right four for the other.

### 1.1 Survivorship bias in batched diagnostics — three instances

The pattern: run N parallel envs, stop once `num_episodes` episodes have completed, with
`num_episodes` far below `num_envs`. Short games finish first, so the sample is the *fastest
N of num_envs* — and because envs auto-reset, a quick env can be counted repeatedly while a
slow one is never counted at all.

| file | called with | effect |
|:---|:---|:---|
| `ai/eval/position_diagnostics.py` | 30 episodes / 128 envs | mean final turn 3.30 vs 6.67 true; `frac_reaching_turn9` 0.000 vs 0.266; every late-game metric pinned to zero |
| `ai/eval/decisive_probe.py` | 30 episodes / 128 envs | instant-win take rate 89.5% vs 76.6% true |
| (both) | — | fixed by measuring the **first episode of every env**, draining until all finish, so sample size is `num_envs` and membership cannot depend on game length |

Both files had already had a *partial* fix that buffered positions per env and folded them in
on completion. That corrected attribution within an episode and left the stopping rule intact,
and both docstrings then claimed the length bias was handled. The decisive probe's docstring
even explained why the bias mattered — "decisive positions cluster near the end of a game" —
directly above the code that kept it.

Regression tests assert `num_games == num_envs` / `episodes == num_envs`, which can only hold
if each env contributed exactly one episode.

### 1.2 A policy was choosing its own dice in evaluation

At a `ROLL_DIE` chance node `ctx().decision_player` is `NONE`.
`TournamentEvaluator.play_matchup` fell back to `phasing_player` and asked an agent to pick an
action there, resolving it through `step_flat` — **134 such nodes per 20 games**, about 6.7 a
game. The vectorized runner resolves chance nodes inside the engine and never exposes one.

The two evaluation paths therefore disagreed by more than 25 points on the same deterministic
matchup, and the sequential path scored a *newer* snapshot at 0.34 against an older one, which
is what gave it away. Fixed by draining chance nodes; the paths are now bit-identical on
identical seeds. This is the same bug previously fixed in the decisive probe, in a second
place.

**Every in-run snapshot evaluation logged before this fix used the buggy path.**
`tools/tournament.py` was always on the correct path.

### 1.3 Evaluation consumed most of a training run

Snapshot evaluation used the one-game-at-a-time path (1.82 games/sec) rather than the
vectorized one (53.7 games/sec on the same 100 games, 30x). Cost also grew with run length,
because every snapshot was appended to the opponent list permanently.

Measured on the 3-hour A/B in §3.1: evaluation took **37%** of one arm's wall clock and
**61%** of the other's. The final evaluation faced 14 opponents and took 957s against a 900s
snapshot interval, leaving about one training iteration per interval.

Fixes: batched evaluation, `--eval-max-snapshot-opponents` (default 4), and evaluation
excluded from the `--duration-seconds` budget. After the fix, both arms of the §3.2 rerun
showed **zero** gaps over 20s and evaluation overhead of about 4%.


### 8.5 The forced-win metric under-detects as well as over-detects

Three further corrections, from a review of the individual cases, all pointing the same way: the
metric is not measuring what §8.1 claimed.

**It misses wins that need a choice, and then scores them as declines.**
`classify_legal_actions` follows only *forced* continuations, which its docstring is explicit
about. So when a position has several winning lines and the human takes one the walk cannot see,
it is recorded as declining. **Replay 104 T9 AR7** is exactly this: the US had more than one path
to a forced win, and playing How I Learned to Stop Worrying won just as Duck and Cover would
have. It is counted as a decline; it is a win taken. The metric should ask whether the action the
player chose also wins, not whether it is in the classifier's set.

**A headline is not an action round, and cannot be judged as one.** **Replay 224 T7 AR0** and
**replay 259 T8 AR0** are headline decisions. Both players choose simultaneously and neither
knows the other's card, so a line that is forced *given the board* is not available information
to the player. Declining it is ordinary play under uncertainty, not an error. Headline decisions
should be excluded from this metric entirely.

**And at least one labelled win does not reach 20 VP.** At **replay 259** the US was on 17 VP and
KAL-007 moves them to 19 — not a win. The engine nonetheless drives that line to a terminal state
it scores +1.0 for the US, so there is a second defect here, distinct from 8.4, in whatever
terminal the forced walk arrives at. Not yet diagnosed.

**Where this leaves 8.1.** Of the 64 non-endgame opportunities, 29 are the illegal Ortega/Che
coups of 8.4, and an unknown further number are headline decisions or wins-taken-by-another-line.
The remaining sample is too small and too contaminated to support any statement about how humans
treat forced wins. **8.1 is withdrawn and not replaced.** The instrument needs fixing first: skip
headlines, test the chosen action for a win rather than set membership, exclude the last action
round of turn 10, and re-run once the free-coup handlers filter.


### 23.1 Three measurement faults found while running this, all silent

Each would have produced a plausible wrong number rather than an error, which is §1's pattern.

**A model reads fixed slices, so a wrong-width observation is misread, not rejected.** Arm E's
in-training evaluations fed the v2.1 network legacy 4,293-wide observations: the board came out
right by luck, the card block read 1,430 floats spanning cards plus globals, and the global slice
landed in the legacy history region and was all zeros. Reported 2.0% against `HeuristicBot` and a
mean final turn of 1.37 while the run's own rollouts averaged turn 6.7 — the contradiction is what
exposed it. The same 35M snapshot, given its own layout, beats `HeuristicBot` 81.3%. `NeuralAgent`
now derives the layout from the model and `_assert_width` raises on a mismatch. **Every arm E
evaluation logged before that fix is void.**

**`snapshot_final.pt` is not a distinct model.** It is weight-identical to the last step snapshot
(97/97 tensors; only the file hash differs). Listing both in a tournament enters one player twice
and lets it accumulate a rating partly against its own duplicate. Checked before the 240M run and
excluded; the 80M comparison had used explicit step snapshots and was unaffected.

**Sorting snapshot paths numerically does not sort them numerically.** `sort -t_ -k2 -n` over full
paths reads field 2 of a path that is itself full of underscores — `E` in `arm_E_cont_80to240` — so
every line compares equal and `ls`'s lexical order survives. Lexical equals numeric only while every
step count has the same digit count: arm D's are all 9 digits and came out correct, while arm E's
span 85000192 to 240058368, so the 8-digit names sorted last and `tail -4` selected 85M, 90M and 95M
as that arm's "late" snapshots. Caught before the tournament ran, by printing the selection. **A
check that passes on one arm by arithmetic accident is not a check** — the selection is now made by
extracting the step count and sorting on it, and verified to return matched lists for both arms.

---

## 6. Method notes

- **Budget A/B arms by `--train-steps`, never by wall clock.** Steps/sec is policy-dependent,
  so a time budget hands the arms different amounts of training (§3.1).
- **Run arms in parallel — for comparability, not for speed.** Measured on the 4090 (arch v2,
  512 envs): one arm alone runs at **14,761 steps/s**; two arms together run at **7,044 and
  7,833**, so combined throughput is **14,877** — total throughput is conserved and the wall clock
  to finish both is the same either way. The earlier claim here that "sequential doubles
  turnaround" was wrong. What parallel actually buys is that both arms meet identical machine
  conditions, which removes a time-varying confound; what sequential buys is the first arm's
  result at T instead of 2T, which matters if a run may be abandoned early.
  Note the two arms differed by 11% (7,044 vs 7,833), so contention is *not* symmetric — which is
  another reason a wall-clock budget cannot be used for parallel arms. `--train-steps` gives both
  the same training regardless.
- **Give every model a distinct filename in a tournament.** Two checkpoints both named
  `snapshot_final` collided in the Bradley-Terry fit and were reported with identical Elo.
- **Replicate before believing a small gap.** A tournament is not reproducible from its
  configuration: deals are seeded, but the agents sample, so two identical 6,000-game runs differ
  by ~1.5 points on a matchup (§7.2). Treat that as the noise floor at 1,000 games a pair, not the
  binomial SE, which assumes away exactly this source of variation.
- **Enable `--auto-advance` freely.** It is outcome-neutral, verified bit-exact under a
  position-derived policy on the vectorized path (§7.1). It is also a smaller speed win than it
  looks (3.3% fewer batched steps).
- **Beware `harvest()` in analysis scripts.** It calls `retire_stale()`, which drops older
  generations by design, so positions must be taken out of the pool after each round or they
  are lost. This silently reduced a 1,000-position sample to 91.

---

## 7. What a tournament number is reproducible to

### 7.1 Auto-advance does not change outcomes

`Engine::step(..., auto_advance)` resolves unattended die rolls, single-choice masks and a few
deterministic multi-step events (Suez <= 4, Muslim Revolution <= 2, East European Unrest <= 3,
Truman, Independent Reds) inside the engine. Enabling it must be a pure speed change or every
tournament number taken with it is incomparable to one taken without.

**Bit-exact where bit-exactness is possible.** `tests/training/test_auto_advance_outcome_equivalence.py`
plays 128 vectorized games under a policy that is a pure function of the mask, and asserts that
terminal utility, victory points and final turn are identical with the flag off and on. The policy
has to be position-derived rather than RNG-driven: with the flag on the engine asks for fewer
actions, so a policy consuming a shared random stream would diverge for reasons unrelated to the
flag. This joins the existing single-state suite (`tests/training/test_auto_advance_integration.py`,
plus `engine/tests/test_auto_advance.cpp`).

**It removes little.** Under that policy the flag cut batched step calls only from 672 to 650
(3.3%), and wall-clock at that scale was inconclusive. It is not the speed lever it looks like.

### 7.2 Tournament results are NOT reproducible run to run — noise floor ~1.5 points

Found while trying to verify 7.1 at tournament scale. Three runs of the *identical* 6,000-game
command, differing only in the flag:

| matchup | auto-advance ON | OFF run 1 | OFF run 2 |
|:---|---:|---:|---:|
| K=40 vs control | 56.0% | 53.6% | 55.2% |
| K=40 vs Heuristic | 90.2% | 89.6% | 89.5% |
| control vs Heuristic | 87.2% | 87.0% | 85.5% |
| Heuristic vs Random | 97.1% | 97.2% | 97.3% |

**Two identical OFF runs differ by 1.6 points on the headline matchup and 1.5 on another** — as
much as the ON/OFF difference. So the ON/OFF gap is not evidence about the flag, and the flag is
not the source of the variation. `BatchMatchRunner` seeds deals deterministically from `base_seed`,
but the agents sample, so a tournament is not reproducible from its configuration alone.

**Consequence for reading every A/B in this file.** At 1,000 games a matchup, differences below
roughly **1.5-2 points are inside the run-to-run envelope** and mean nothing on a single pair of
runs. The binomial SE (1.6% at n=1,000) understates it, because it assumes the only variation is
sampling from a fixed distribution. Either fix the agent sampling seed, or replicate a run before
believing a small gap. The §4.4 comparisons flagged as "underpowered (z ~ 1.2-1.75)" sit exactly
in this band.

---

## 20. Run-to-run variance, and how much of it is just where you stopped

§18 compared single final checkpoints and read differences off them. §19's encoding plan assumed a
"slight positive effect" would be visible at one run per arm. Both rested on a variance estimate
that had never been made, and when it was made it was much larger than assumed -- and then, on a
closer look, largely removable.

### 20.1 Seeding had to be added before variance could be measured at all

The environment seed was the literal `12345` and torch was left unseeded, so two runs of one
configuration saw the same deals and the same dice and differed only in initialisation. A "replicate"
under that arrangement measures a fraction of the thing. `--seed` now sets both together (`4f2b23b`),
or neither when omitted, which keeps existing behaviour.

`--resume` was added alongside (`4f2b23b`), because snapshots are bare `state_dict`s and a
`--warmup-checkpoint` restart silently drops the optimiser moments, the reference policy and the
step counter. The tests assert the distinction rather than assume it: ten steps must equal five,
save, resume, five more, **and** a weights-only restart must *not* match. If that negative control
ever passes, the resume file is pointless and says so.

### 20.2 The first variance number was wrong, and it was the one everything rested on

Arm A against the 160M run's own 80M snapshot -- same configuration, different seed -- came out
**5 Elo apart, 50.2%/49.8% on 1,000 games**. That was reported as the noise floor, with the
transfer to other configurations flagged as an assumption.

It does not transfer. On the synth-only configuration, C against C2 is **59.6 Elo apart, 63.9%
head to head**. A third seed gave SD 64.2 over three final snapshots. The assumption was wrong by
an order of magnitude, and several earlier claims went with it -- "the synthetic warm start is
worth +199 Elo" and "the human BC layer costs 128 Elo" were both single-draw differences smaller
than the spread they were measured against.

### 20.3 Most of that spread is *when you stopped*, not *which seed you drew*

Rating the last four snapshots (65M, 70M, 75M, 80M) of every arm rather than the final one:

| arm | 65M | 70M | 75M | 80M | mean | within-run SD |
|---|---:|---:|---:|---:|---:|---:|
| C s1 | 1728.4 | 1773.4 | 1764.3 | **1847.7** | 1778.4 | 50.1 |
| C2 s2 | 1729.6 | 1731.9 | 1736.5 | 1760.0 | 1739.5 | 14.0 |
| C3 s3 | 1754.2 | 1778.5 | 1732.2 | 1736.8 | 1750.4 | 21.0 |
| B s1 | 1680.7 | 1634.7 | 1640.7 | 1642.8 | 1649.7 | 20.9 |
| B2 s4 | 1689.2 | 1632.6 | 1741.1 | 1722.9 | 1696.4 | 47.7 |

Mean within-run oscillation is **30.7 Elo**, and C s1 spans 1728–1848 with its *final* snapshot at
the peak -- reporting it as 1848 was reading the top of a wobble.

| how a run's number is taken | synth-only SD | cold-start SD |
|---|---:|---:|
| final snapshot only | 58.5 | 56.6 |
| **mean of the last four** | **20.1** | **33.0** |

**Averaging four snapshots cuts the between-run SD by 2.9x on the synth-only configuration, for no
extra compute.** Cold start gains less because B2 itself oscillates 47.7, and two seeds cannot
average that away.

This is the cheapest variance reduction available and it should be standard: *rate the last four
snapshots, report the mean*. Every single-final-snapshot comparison earlier in this document is
inflated by roughly 30 Elo of stopping-point noise.

### 20.4 What survives

| configuration | mean of per-run means | seeds |
|---|---:|---:|
| synth-only warm start | **1756.1** | 3 |
| cold start | **1673.1** | 2 |
| `dec_turns40` | **1959.3** | 1 |

The warm-start effect is **+83.0 Elo** against a between-run SD of 20–33, roughly 2.5–3σ. Real,
and less than half the +179 claimed from single finals.

The ordering is not clean at the level of individual runs: the weakest synth-only seed (C3) finishes
*below* the stronger cold start (B2) and splits 50.9%/48.9% with it. Configuration means separate;
individual runs overlap.

### 20.5 Consequences for how experiments get run here

* **Rate four snapshots, not one.** Free, and nearly triples effective precision.
* **Two or three seeds per arm** now suffices for effects above ~40 Elo. The encoding experiment
  §19 planned is viable on that basis; at the 60 Elo single-final floor it was not.
* **The oscillation has a likely cause worth removing at the source.** The learning rate is
  constant for the entire run -- there is no decay schedule -- which is exactly what leaves a policy
  wandering at the end rather than settling. Linear or cosine decay, or a weight EMA evaluated
  instead of the live policy, would remove the 30.7 Elo oscillation rather than averaging over it.
  Neither has been tried.
* **`dec_turns40` is still ~200 Elo clear of everything trained since**, which is the subject of
  §21.

---

### 20.6 Continuation variance, measured directly — the seed is worth ~15 Elo, and a leg's *gain* twice that

§20.3 measured the spread between *runs of a configuration*. It never measured the spread between
*continuations of one run*, which is what every within-lineage number in §23 is, and those were
being quoted without an error bar.

**Setup.** Two continuations of arm D from the identical 160M resume state — same weights, same
optimiser moments, same reference policy — differing only in seed (20260916 against 20260918), each
to 240M. Four late snapshots per arm, 400 games a side. There is no better arm here; the output is a
spread.

| | 225/230/235/240M | mean | SD | gain over the 160M start |
|:---|:---|---:|---:|---:|
| seed 20260916 | 1880, 1893, 1887, 1886 | 1886.2 | 5.4 | **+86.9** (62.25%) |
| seed 20260918 | 1897, 1894, 1828, 1858 | 1869.1 | 32.7 | **+54.7** (57.81%) |

Difference in mean **17.1 Elo**; pooled head-to-head over all 16 pairings, 12,800 games, **51.80%
±0.87**, implying **12.5 Elo**. Consistent with §20.3's ~20 Elo between-run SD, now confirmed for
the continuation case specifically.

**The consequential number is the second column, not the first.** The same 80M leg, measured the
same way, was worth +86.9 Elo on one seed and +54.7 on the other. A leg's gain therefore carries
roughly **±16 Elo**, which is larger than most of the differences §23 was reading as a trend.

**Three practices follow.**

* **Quote a leg's gain with ±16, or do not quote it as a trend.** §23's +74.1 then +52.7 for arm D
  is not evidence of compressing returns; the two are indistinguishable.
* **Never compare Elo across tournaments.** The identical comparison — D's late four against its own
  160M final, the same eight files — read 60.50% in the 240M pool and 62.25% here, 1.4σ apart on
  3,200 games each from sampling alone. Direct head-to-heads carry about ±12 Elo before any seed
  effect. Within one tournament the numbers are comparable; across two they are not.
* **A single cell is not a comparison, twice over.** The final snapshots of the two replicates meet
  at 59.1%, which would put seed variance at ~64 Elo; pooling the sixteen pairings gives 51.80% and
  ~13. The same trap produced a spurious "E beats D 54.9%" at 240M and a spurious "+52 Elo for E's
  last leg" at 320M.

**And snapshot averaging damps the stopping point without taming it.** Within-run SD across the last
four was 5.4 on one seed and 32.7 on the other — a sixfold difference between two runs of one
configuration, with no visible cause. Four snapshots is the practice, not a guarantee.

---

## Agreement with human play

The corpus is the only strategy prior available, so how closely a policy reproduces it is a
reported figure everywhere -- per BC epoch, per snapshot, and in the tournament reports. It is
not accuracy: a play that spends several points is one decision made several times, and the
order the log happens to record is not part of it.

### 9.11 Agreement with human play, counted without the order of a placement

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
not about the order. Every figure quoted earlier in §9 was pessimistic by roughly this much, which
changes no conclusion in it -- §9.1's washout still lands at ~32-33% either way.

The correct measure is now the one to use, and `play` is stored as a dataset column so it can be
applied without re-running conversion, which is the expensive part.


### 9.11.1 Coups and realignments are excluded from reordering

Not every run of point decisions is order-free, and §9.11 treated them all as if they were. The
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
realignments being wrongly forgiven, so §9.11's figures stand as measured. The measure is now right
for the right reason rather than by luck.


### 9.11.2 Agreement is now the reported figure everywhere

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

