# P7 (human data) — injection dose, what the behaviour is worth, and the ablation that killed it

Records §10, §11, §20, §21 and §22 of the original `experiments.md`: the second half of the
human-data programme. §10 (E5) sweeps injection weight and frequency; §11 measures what the largest
human/agent behavioural gap is actually worth in win rate and finds a ceiling low enough to change
the plan; §20 is the run-to-run-variance stub whose corrected numbers §21–§22 rest on; §21
investigates why `dec_turns40` still beat everything trained after it; and §22 resolves it — human
injection was a 157–236 Elo handicap, and a cold start without it beats everything. Arms are
`--arch v2`: 4M-step pilots in §10, 78–160M-step arms in §21–§22, post-E1 engine, legacy
observation. §22.2 explicitly invalidates the *magnitudes* of §18, §20 and §9–§10, all of which were
measured inside the handicap; those entries are left standing where they were written. §20–§22 are
not numerically adjacent to §10–§11 in the original file — they are here because they are the same
question. **This file is append-only history** and is not edited to match current belief.

---

## 10. Injection weight and frequency (E5) — frequency is everything, weight is nothing

**Setup.** All nine arms start from the same checkpoint: self-play BC, then 8 epochs of human BC on
the **224 train games only**, reaching **49.33%** agreement on the 56 held-out games. That split
matters — a first attempt fitted the warm start on all 280 games and read 65% "held out", which was
memorisation, so it was thrown away and rebuilt. Every figure below is on games no arm has been
fitted to. 4M steps each, two at a time, `blunder_aware` + K=40, one flag apart.

### 10.1 Weight, at every iteration — no effect across 16x

| `--inject-weight` (every=1) | end |
|:---|---:|
| 0.25 | **40.4%** |
| 1 | 40.2% |
| 4 | 39.2% |
| control, no injection | 35.6% |

A sixteenfold range of weight spans 1.2 points, while the gap to the control is 4.6. Injection is
close to a switch: whether it happens matters, how hard barely does.

### 10.2 Frequency, at fixed weight — monotone, and gone by every 4th iteration

| `--inject-every` (weight=1) | end | over control |
|:---|---:|---:|
| 1 | **40.2%** | +4.6 |
| 2 | 38.8% | +3.2 |
| 4 | 36.6% | +1.0 |
| control | 35.6% | — |

The benefit decays steadily as the dose gets rarer and is nearly gone by every fourth iteration.

### 10.3 Equal dose, different schedule — rarity cannot be bought off with weight

Holding `weight / every` at 1.0, so every arm applies the same total supervised signal:

| schedule | end |
|:---|---:|
| w=1, every=1 | **40.2%** |
| w=4, every=4 | 35.9% |
| w=16, every=16 | 34.7% |
| control | 35.6% |

**Same total dose, 5.5 points apart, and the two rare schedules are at or below the control.**
w16e16 is *worse than not injecting at all*.

### 10.4 What this means

**Frequency is the parameter; weight is not.** The signal has to be applied continuously or it does
nothing, and a bigger dose delivered less often is not a substitute — it is worse than nothing,
because each large correction knocks the policy somewhere RL then has to walk back from, and the
interval is long enough for it to do so. That is the oscillation predicted when injection was
proposed, now with matched-dose evidence rather than a guess.

The practical setting is **every iteration, at whatever weight is convenient** — 0.25 works as well
as 4. Anything rarer than every second iteration is not worth running.

**What it still does not do is hold the BC level.** The best arm decays 49.3% → 40.2%: injection
halves the washout (the control loses 13.7 points, injection loses 9.1) but does not stop it. Since
weight does not help, the remaining levers are elsewhere — a lower RL learning rate, or injecting
against a frozen copy rather than the live policy.

**Caveats.** One seed per cell, 4M steps, and a control that moved 1.3 points across its own
snapshots in §9.12. The weight ladder's 1.2-point spread is inside that; the frequency and
matched-dose effects, at 4.6 and 5.5 points, are not.


## 11. The gate: what does the space-race dominance error actually cost? — ~3 points, and that changes the plan

**Question.** §9.3 found the largest behavioural gap between humans and our agents: spacing a card
while holding an equal-Ops opponent recurring event, where humans err 0.4% of the time and our
agents 12.6-18.1%. §10 then spent nine arms learning how to transfer human behaviour. None of it
asked what the behaviour is worth. §4.4 is the warning -- forced-win take rate looked like a 20-30%
failure and was worth about two points, because declining was usually free.

**Method** (`ai/eval/dominance_cost.py`). The fork has to be at card *selection*: by the time the
engine asks how to play a card the card is chosen, and which card went to the track is the decision.
Positions are taken at `SELECT_CARD` where the agent's own greedy continuation goes on to space its
own or a neutral card with an equal-Ops opponent recurring event in hand. Both branches -- the
agent's choice, and the opponent's card to the track instead -- are then continued by the same
policy over twelve die streams each, counting wins for the player who chose.

**Result** (`dec_turns40`, two collections pooled, 148 positions):

| branch | wins | rate |
|:---|---:|---:|
| dominated (what the agent wanted) | 766/1768 | 43.33% |
| dominant (the opponent's card instead) | 823/1772 | **46.44%** |
| **cost of the error** | | **+3.12 points** (approx SE 1.67, z ≈ 1.9) |

The two collections gave +5.15 (39 positions) and +2.39 (109 positions); the spread between them is
itself a fair warning about how noisy this is.

**Verdict: the error is not free, but the ceiling is low, and that is the finding.** Roughly three
points of win rate, in the same range as §4.4's forced wins. It clears the gate in the sense that
the behaviour matters -- but it puts a hard ceiling on the whole line, because three points is what
you would get for fixing the error **completely**.

**And E6 as planned cannot measure a fraction of three points.** §10's best arm transfers about half
the alignment it starts with (49.3% → 40.2%), so the expected strength gain is one to two points.
§7.2 put the tournament noise floor at ~1.5 points on 1,000 games a matchup, and training seed
variance has never been measured at all. A two-arm, one-seed E6 would return a number
indistinguishable from noise whichever way it came out, and we would not be able to tell which had
happened.

**What that argues for.** Either run E6 with several seeds per arm and accept the multiplied cost,
or stop treating strength as the target for this line and use the corpus for what it demonstrably
is -- the only reference we have for what good local play looks like, and an evaluation asset. The
one thing not worth doing is the cheap version of E6, which would produce a number nobody should
believe.

**Caveats.** The SE above treats games as independent; they are not -- twelve rollouts share a
position, and the two branches are paired -- so it is approximate in both directions. One
checkpoint (`dec_turns40`); a differently-trained agent may pay a different price for the same
mistake.

## 20. Run-to-run variance — moved to [`variance_and_noise.md`](variance_and_noise.md)

Two numbers from it are quoted throughout this file and are worth having to hand:

* **Rate the last four snapshots and report the mean.** Mean within-run oscillation is 30.7
  Elo and a single final snapshot samples it arbitrarily; averaging four cuts between-run SD
  from **58.5 to 20.1** for no extra compute. This is the standing practice.
* **~20 Elo is the between-run SD** on a snapshot-averaged mean. Two or three seeds separate
  configurations above roughly 40 Elo; below that, a single pair of runs cannot.

The earlier 5 Elo figure was wrong and had been load-bearing — it invalidated the *sizes* of
the synthetic warm start's "+199 Elo" and the human BC layer's "−128 Elo", both single draws
smaller than the spread they were measured against. The corrected warm-start effect is
**+83.0 Elo**. Cause and consequences in [`variance_and_noise.md`](variance_and_noise.md).

## 21. `dec_turns40` is still the strongest model, and nothing since has matched it

Across every pool it has appeared in, the checkpoint from §4.3's K-sweep beats everything trained
afterwards: 1959.3 in §20's pool, ~200 Elo above the best configuration mean, winning 65–88%
against the recent arms. That is worth explaining before more arms are run against a bar that may
not be what it looks like.

### 21.1 What it actually was, corrected

An earlier reading of this concluded `dec_turns40` was a cold start, from two pieces of evidence:
its `snapshot_0s.pt` matched neither modern warmup checkpoint, and its weight statistics looked
like a fresh initialisation. **Both were weak and the conclusion was wrong.**

The decisive test is to compare *sibling runs*. `dec_turns40`, `dec_turns20` and `sp2_pool_off`
were launched separately, and their `snapshot_0s` files agree on **93 of 97 tensors**. Random
initialisation would agree on none. Searching every checkpoint on disk for an exact match finds
`data/checkpoints/coldwar_net_v2_warmup.pt` (25 August), identical on **91 of 91** comparable
tensors.

The four that differ are `defcon_risk_head` only -- an auxiliary head added after that warmup was
built, so each run initialised it fresh.

That file is the common ancestor of most of the repository: `sp_pool_on/off`, `sp2_pool_on/off`,
`run_v2_anchored_1h`, the four `run_v2_20260825_*` runs, `dec_turns20` and `dec_turns40` all start
from identical weights. The modern lineage is unrelated to it --
`warmup_synth_then_human_train.pt` matches on **1 of 91** tensors.

**Method note.** Comparing a candidate against two guessed sources and then reaching for weight
statistics was the wrong shape of test. Sibling runs sharing an ancestor is a positive control that
either fires or does not, and it should have been the first thing tried.

### 21.2 What differs from the recent arms

Read from what each run logged, not from its description:

| run | steps | injection | mean_turn | 20VP endings | entropy | KL |
|---|---:|:---:|---:|---:|---:|---:|
| **`dec_turns40`** | 78.1M | **none** | **7.21** | 0.374 | **1.148** | 0.041 |
| C s1 | 80.0M | every iter | 5.80 | 0.466 | 1.339 | 0.024 |
| C2 s2 | 80.0M | every iter | 5.67 | 0.507 | 1.385 | 0.017 |
| C3 s3 | 80.0M | every iter | **4.99** | 0.511 | 1.322 | 0.019 |
| B s1 | 80.0M | every iter | 5.56 | 0.441 | 1.472 | 0.017 |
| B2 s4 | 80.0M | every iter | 6.08 | 0.384 | 1.441 | 0.020 |
| 160M run | 160.0M | every iter | 5.54 | 0.443 | 1.381 | 0.034 |

Everything controllable is matched -- arch v2, `blunder_aware`, K=40, 512 envs, ~78-80M steps. Three
things are not:

1. **Injection.** `dec_turns40` has none; every arm since injects human data every iteration at
   weight 1.0. `inject_loss` never appears in its metrics.
2. **The warm start.** A different ancestor, unrelated to the modern one.
3. **The engine.** 44 files and +3,832/-470 lines since its base commit `44504c16`, including
   changes that move the decision stream: Che/Ortega free-coup legality, turn cleanup as its own
   step, Independent Reds, per-card headline frames, Defectors, Shuttle Diplomacy, the China Card
   space race, Grain Sales, trap effective-Ops.

The correlate worth noticing is that **every injected run plays shorter games** -- 4.99 to 6.08
turns against 7.21 -- wins by 20 VP blowout more often, and ends with higher entropy, i.e. a less
settled policy. That is consistent with §18's finding that injection buys human agreement and no
Elo, and it raises a sharper possibility: injection may be a *cause* of the short games that
§12-§18 identified as the thing stopping positional value from ever paying off.

### 21.3 The reproduction

`repro_dec40_noinject`: the same warm start (`coldwar_net_v2_warmup.pt`), K=40, 512 envs, 78M
steps, **no injection**, on the current engine. Injection confirmed off from both the absent startup
line and `inject_loss == 0.0` across all logged iterations.

That leaves the engine as the only uncontrolled difference, so the result discriminates:

* **near 1900** -- the engine changes are innocent, and injection plus the warm-start swap account
  for the ~200 Elo;
* **near 1650** -- the engine changes made the game harder, or `dec_turns40` is partly fitted to a
  game that no longer exists, and its crown is an artifact rather than a target.

Two cautions on reading it. §20 measured between-run SD at 20-33 Elo on averaged snapshots, so a
single arm resolves a 200 Elo gap but not a 60 Elo one -- intermediate outcomes will not be
diagnostic. And `dec_turns40`'s own rating has moved between 1824 and 1959 across pools, so it must
be evaluated in a pool containing `dec_turns40` itself rather than against a remembered number.

## 22. Human injection was the handicap, and a cold start without it beats everything

§21 left `dec_turns40` ~200 Elo clear of every arm trained since, with three candidate causes:
injection, a different warm-start ancestor, and 3,832 lines of engine change. All three are now
resolved.

### 22.1 The engine was innocent

`repro_dec40_noinject` -- the same warm start (`coldwar_net_v2_warmup.pt`), K=40, 512 envs, 78M
steps, **injection off**, on the current engine -- reached **1921.8** against `dec_turns40`'s
1912.8 in the same pool, splitting **51.5%** head to head over 600 games. Injection confirmed off
from both the absent startup line and `inject_loss == 0.0` in all 563 logged iterations.

So the engine changes did not make the game harder and `dec_turns40` is not fitted to a game that
no longer exists. It is a reproducible target.

### 22.2 The 2x2: injection costs 157-236 Elo

Arm D completes the design -- cold start, no injection -- so both factors separate for the first
time. Means of four late snapshots:

| | injection ON | injection OFF | effect of removing it |
|---|---:|---:|---:|
| **warm start** | 1736.2 | 1893.5 | **+157.3** |
| **cold start** | 1612.3 | 1847.9 | **+235.7** |
| *effect of warm start* | +123.9 | +45.5 | |

Head to head with injection isolated -- same cold start, same K, same budget -- **D beats B 78.2%**.

Against §20's noise floor of 20-33 Elo this is not arguable. **Human injection has handicapped
every arm since it was introduced**, and it was never ablated at full scale on the current engine
before now.

**The warm start mostly stops mattering once injection is gone**, falling from +123.9 to +45.5.
Most of what looked like the synthetic prior helping was the prior partially shielding the policy
from injection damage.

Injected arms are also *less settled*: entropy 1.32-1.47 against 1.12-1.15, and snapshot-to-
snapshot SD 47.9 against D's 9.5. They play shorter games too, 4.99-6.08 turns against 6.5-7.21,
which is the §12-§18 complaint about positional value never being collected -- so injection is
plausibly a *cause* of the short games those sections diagnosed rather than a separate problem.

**This invalidates the magnitude of several earlier findings**, all measured inside the handicap:
§18's "agreement rose 3.7 points for zero Elo", §20's +83 Elo warm-start effect (that is the
injection-on column), and §9-§10's conclusion that injection was neutral-to-useful.

### 22.3 Equal budget, and the cold start wins outright

Warm-started arms carry inherited compute, so comparing them to a cold start at equal *steps*
understates the cold arm. Extending D to 160M by true resume:

| model | Elo | vs `dec_turns40` |
|---|---:|---:|
| **D @160M** | **1885.9** | **63.6%** |
| `dec_turns40` | 1805.6 | -- |
| D' @80M (second seed) | 1786.1 | 50.8% |
| D @80M | 1767.0 | 42.1% |

**The strongest model in the repository is now a cold start with no warm start and no human data
-- no inherited lineage at all.**

D' also gives the between-seed spread on this configuration: D and D' split **50.6%/48.8%**, 19
Elo apart, against **59.6 Elo** for two seeds of the injected synth-only arm. The best recipe is
also the most reproducible, so future comparisons on it need fewer seeds.

### 22.4 The trajectory, and a plateau that would have misled

16 checkpoints at 10M intervals, 61,200 games:

| Msteps | 10 | 30 | 50 | 70 | 90 | 110 | 130 | 150 | 160 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Elo | 1409.7 | 1614.4 | 1731.8 | 1784.4 | 1868.0 | 1865.8 | 1930.9 | 1926.9 | **1946.3** |
| vs `dec40` | 8.0% | 14.8% | 33.0% | 39.0% | 52.2% | 50.8% | 61.0% | 61.8% | 64.8% |

**+537 Elo across the run.** Gains decelerate from ~+100 per 10M to ~+17 over the final 40M but
never reach zero, and it crosses `dec_turns40` at 90M.

**There is a three-checkpoint plateau at 90-110M** (1868, 1867, 1866, with the `dec40` rate dipping
to 50.8%) followed by another +80 Elo. A run stopped at 110M would have been recorded as converged.
That is §20's stopping-point trap again, in a different guise: not oscillation around a mean, but a
flat stretch that resumes climbing.

**Caveat on the join.** 10-80M come from the original run and 90-160M from the resumed
continuation, which used a fresh seed, so the +58 at 90M is not cleanly separable from a
new-environment-stream effect. Similar jumps at 120M (+40) and 130M (+25) occur well after the
join, which argues against it being a resume artefact, but it is not proven.

### 22.5 What follows

* **Turn injection off** for everything until there is a reason to believe a different form of it
  helps. §"brainstorm" lists the candidate mechanisms; the cheapest discriminating test is
  policy-only injection with the value loss disabled, since the BC warm start's critic measured
  Brier 0.378 -- worse than always predicting even -- and a corrupted critic damages every gradient.
* **Keep extending D.** It has not saturated at 160M and `--resume` makes continuation a single
  command.
* The human corpus remains valuable for **evaluation** -- §14-§17 found real defects with it --
  which is consistent with it being poor training supervision.

---
