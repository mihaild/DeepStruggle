# Variance and noise — what a number here is reproducible to

Append-only history of every attempt to put an error bar on this project's numbers: what a
tournament result is reproducible to, how much of a run's rating is just where it stopped, how
far two seeds of one configuration drift apart in strength and in the *shape* of the games they
play. **Update discipline: append-only.** The estimates here superseded each other — the first
noise floor was wrong by an order of magnitude and everything rested on it — and the superseded
ones stay, because the conclusions they invalidated are quoted in older plans and experiment
entries. The practices that came out of this file (rate four snapshots, quote a leg's gain with
±16, never compare Elo across tournaments) are maintained as a living reference in
[`../method/running_experiments.md`](../method/running_experiments.md), and the one-line versions
are in [`../method/measurement_pitfalls.md`](../method/measurement_pitfalls.md).

These entries were written under the section numbers of the original `research/experiments.md`,
by way of the retired `research/metrics.md`:

| written as | now |
|:---|:---|
| §7.2 | Tournament results are NOT reproducible run to run |
| §20 – §20.5 | Run-to-run variance, and how much of it is just where you stopped |
| §20.6 | Continuation variance, measured directly |
| §20.7 | A single snapshot cannot measure an effect smaller than a run's oscillation |
| §1.5.3 | Game shape varies between seeds by as much as it varies between arms |

`metrics.md` §7.1, the check that auto-advance does not change outcomes, is reference rather than
history and lives in [`../method/running_experiments.md`](../method/running_experiments.md).

---

## Tournament results are NOT reproducible run to run — noise floor ~1.5 points

Found while trying to verify the auto-advance outcome-equivalence claim
([`../method/running_experiments.md`](../method/running_experiments.md)) at tournament scale.
Three runs of the *identical* 6,000-game
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

**Consequence for reading every A/B in these logs.** At 1,000 games a matchup, differences below
roughly **1.5-2 points are inside the run-to-run envelope** and mean nothing on a single pair of
runs. The binomial SE (1.6% at n=1,000) understates it, because it assumes the only variation is
sampling from a fixed distribution. Either fix the agent sampling seed, or replicate a run before
believing a small gap. The
[`agent_deficiencies_and_decisiveness.md`](agent_deficiencies_and_decisiveness.md) §4.4
comparisons flagged as "underpowered (z ~ 1.2-1.75)" sit exactly in this band.


## Run-to-run variance, and how much of it is just where you stopped

[`critic_vs_policy_160M.md`](critic_vs_policy_160M.md) §18 compared single final checkpoints and
read differences off them. [`observation_layout.md`](observation_layout.md) §19's encoding plan
assumed a
"slight positive effect" would be visible at one run per arm. Both rested on a variance estimate
that had never been made, and when it was made it was much larger than assumed -- and then, on a
closer look, largely removable.

### Seeding had to be added before variance could be measured at all

The environment seed was the literal `12345` and torch was left unseeded, so two runs of one
configuration saw the same deals and the same dice and differed only in initialisation. A
"replicate" under that arrangement measures a fraction of the thing. `--seed` now sets both
together (`4f2b23b`), or neither when omitted, which keeps existing behaviour.

`--resume` was added alongside (`4f2b23b`), because snapshots are bare `state_dict`s and a
`--warmup-checkpoint` restart silently drops the optimiser moments, the reference policy and the
step counter. The tests assert the distinction rather than assume it: ten steps must equal five,
save, resume, five more, **and** a weights-only restart must *not* match. If that negative control
ever passes, the resume file is pointless and says so.

### The first variance number was wrong, and it was the one everything rested on

Arm A against the 160M run's own 80M snapshot -- same configuration, different seed -- came out
**5 Elo apart, 50.2%/49.8% on 1,000 games**. That was reported as the noise floor, with the
transfer to other configurations flagged as an assumption.

It does not transfer. On the synth-only configuration, C against C2 is **59.6 Elo apart, 63.9%
head to head**. A third seed gave SD 64.2 over three final snapshots. The assumption was wrong by
an order of magnitude, and several earlier claims went with it -- "the synthetic warm start is
worth +199 Elo" and "the human BC layer costs 128 Elo" were both single-draw differences smaller
than the spread they were measured against.

### Most of that spread is *when you stopped*, not *which seed you drew*

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

### What survives

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

### Consequences for how experiments get run here

* **Rate four snapshots, not one.** Free, and nearly triples effective precision.
* **Two or three seeds per arm** now suffices for effects above ~40 Elo. The encoding experiment
  `observation_layout.md` §19 planned is viable on that basis; at the 60 Elo single-final floor it was not.
* **The oscillation has a likely cause worth removing at the source.** The learning rate is
  constant for the entire run -- there is no decay schedule -- which is exactly what leaves a policy
  wandering at the end rather than settling. Linear or cosine decay, or a weight EMA evaluated
  instead of the live policy, would remove the 30.7 Elo oscillation rather than averaging over it.
  Neither has been tried.
* **`dec_turns40` is still ~200 Elo clear of everything trained since**, which is the subject of
  the architecture programme ([`P9_architecture.md`](P9_architecture.md)).

---

### Continuation variance, measured directly — the seed is worth ~15 Elo, and a leg's *gain* twice that

*Most of that spread is when you stopped* (above) measured the spread between *runs of a
configuration*. It never measured the spread between *continuations of one run*, which is what
every within-lineage number in [`observation_layout.md`](observation_layout.md) §23 is, and those
were being quoted without an error bar.

**Setup.** Two continuations of arm D from the identical 160M resume state — same weights, same
optimiser moments, same reference policy — differing only in seed (20260916 against 20260918), each
to 240M. Four late snapshots per arm, 400 games a side. There is no better arm here; the output is a
spread.

| | 225/230/235/240M | mean | SD | gain over the 160M start |
|:---|:---|---:|---:|---:|
| seed 20260916 | 1880, 1893, 1887, 1886 | 1886.2 | 5.4 | **+86.9** (62.25%) |
| seed 20260918 | 1897, 1894, 1828, 1858 | 1869.1 | 32.7 | **+54.7** (57.81%) |

Difference in mean **17.1 Elo**; pooled head-to-head over all 16 pairings, 12,800 games, **51.80%
±0.87**, implying **12.5 Elo**. Consistent with the ~20 Elo between-run SD above, now confirmed for
the continuation case specifically.

**The consequential number is the second column, not the first.** The same 80M leg, measured the
same way, was worth +86.9 Elo on one seed and +54.7 on the other. A leg's gain therefore carries
roughly **±16 Elo**, which is larger than most of the differences `observation_layout.md` §23 was
reading as a trend.

**Three practices follow.**

* **Quote a leg's gain with ±16, or do not quote it as a trend.** `observation_layout.md` §23's
  +74.1 then +52.7 for arm D
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

### A single snapshot cannot measure an effect smaller than a run's oscillation

*Most of that spread is when you stopped* established that rating four late snapshots instead of
one cuts between-run SD from 58.5 to 20.1, and it was adopted as practice. `observation_layout.md`
§24.1 is what happens when the practice is skipped.

The `staged_cards` flag was first measured with one snapshot per budget: 800-game cells at about
+/-24 Elo, against a mean within-run oscillation of 30.7. It read **+52.8 Elo at 80M and +4.7 at
160M**, and the 80M figure stood as the headline until the pooled version replaced it with **+29.7
and -11.4** -- a different sign at one budget and half the size at the other.

The arm that looked better was also the noisier one: G's within-run SD was 27.0 against F's 7.5, at
the same seed with one flag between them, which is unexplained and was invisible from a single
checkpoint. Picking its final snapshot flattered it.

**So the error bar on a single-cell comparison is the run's oscillation, not the binomial SE of the
games played.** 12,800 games across sixteen pairings gives +/-0.87pp; 800 games in one cell gives
+/-3.4pp, and the snapshot choice on top of that is worth another 30. Two arms compared one
snapshot each cannot resolve anything below roughly 50 Elo, which is larger than most effects worth
arguing about.

---


### Game shape varies between seeds by as much as it varies between arms

Arm H2 is arm H's configuration on a second seed (20260921, v2.3, corrected engine). At an equal
80M budget the two are the same strength -- pooled over four late snapshots a side and all 16
pairings, 3,200 games, **50.4%, +3 Elo**. A clean replication.

Their *games* are not the same shape:

| self-play, 1,000 games | mean ply | 20 VP | final scoring | DEFCON 1 | wargames |
|:---|---:|---:|---:|---:|---:|
| H @80M | 107.0 | 40.9% | 15.7% | 43.2% | 0.2% |
| H2 @80M | 98.6 | 47.4% | 9.6% | 42.8% | 0.2% |
| H2 @160M | 100.8 | 50.4% | 10.6% | 37.7% | 1.3% |
| (arm D, legacy, 80M) | 99.0 | 46.1% | 11.0% | 42.9% | 0.0% |
| (arm E, v2.1, 80M) | 100.1 | 46.0% | 9.6% | 44.2% | 0.2% |

Two runs of identical configuration and indistinguishable strength differ by **8.4 plies** and by
6 points of final-scoring share -- more than H differed from the pre-fix arms D and E. H2 lands
squarely on top of D and E, not on H.

**This retires the claim that the corrected engine lengthens games.** That was read off arm H
alone ([`corrected_engine_arms_H_I.md`](corrected_engine_arms_H_I.md) §25 draft, and reported as
"the first arm to move final scoring at all"), and it does not
replicate: H is the outlier of the three v2.3-or-earlier runs at 80M, not the start of a trend.
Game length and ending mix are seed-noisy at this budget and cannot carry a conclusion from a
single run, exactly as Elo cannot (above). Nothing here contradicts the starred-card fix being
*correct* -- it is a rules bug either way -- only the evidence offered that it changed play.

What does survive is a budget effect, consistent across two independent comparisons:

- H2 at 160M vs H at 80M: **61.0%, +78 Elo**
- H2 at 160M vs H2 at 80M: **60.9%, +77 Elo**
- against the anchor: 83.0% (H @80M), 84.2% (H2 @80M), **88.0% (H2 @160M)**

and, at 160M, the first movement on the two human gaps that is visible in both the binned
training log and self-play: DEFCON 1 falls 42.8% -> 37.7% and wargames rises 0.2% -> 1.3%. Set
against ITS's 11.7% and 14.9% ([`../method/human_play.md`](../method/human_play.md)) those are
still a factor of three and a factor of eleven away.

A caution on reading the per-iteration monitor: it reports a mean over the last 40 logged
iterations, which overlaps heavily between consecutive reports and made H2's USSR win rate look
like a monotone late-run climb to 66%. Binned by 20M it oscillates 48.5-58.0% for the whole run
with no drift. Bin before believing a trend in it.
