# Detecting a collapse from internal metrics

Win rate against a frozen opponent is ground truth but arrives only every 5M steps. What do the
training loop's own metrics say in between?

**Update discipline: living reference, rewritten in place.**

## Scope rule — read this first

A **lineage** is an engine revision plus an action space plus an architecture plus an observation
layout. E3 and E4 are different lineages: the P17 engine refactor, the 212→220 action repack and
the architecture flags all changed between them.

> **What E3 established is *which instrument sees which failure mode*. That transfers.**
> **No number measured on E3 is a threshold for E4, and E3/E4 numbers are never compared.**

This is not caution for its own sake — the evidence for it is in this file, under
[why numbers do not transfer](#why-numbers-do-not-transfer). Every trigger below is therefore
written **relative to the run's own baseline**, which is the only form that survives a lineage
change. E4's own reference envelope is built from E4 arms alone, at the bottom.

---

## What transfers: the general findings

These are structural claims about mechanisms and instruments, established on E3 and stated without
numbers.

**1. There is no single detector, because there is no single collapse.** Each failure mode has its
own clean signature and is invisible to the other's instrument. Watching one metric and declaring
a run healthy is the mistake this page exists to prevent.

**2. Opponent-pool starvation is a structural check, not a threshold.** Is `opp_pool_size`
present in the metrics at all, and does the pool *grow*? A pool that is absent, or that never gets
past a single opponent, **is the failure itself** rather than evidence of it. Needs no tuning and
no reference run.

**2a. Starvation is the pool failing to grow — not the pool being beaten.** "Beaten more than 90%
of the time" was the obvious trigger and it is a **false positive**: E3-37-31, one of the
healthiest arms in the record, sits above 0.9 for 947 of its 1,229 iterations while its pool grows
to 8. A strong policy beating its own older snapshots is what improvement looks like. Alarm on
`opp_pool_size` not rising, and treat `opp_win_rate_mean` as description.

**3. KL domination shows as a step change in `kl_div` against that run's own running level.** The
regularizer overwhelms the policy gradient. The transferable form is *orders of magnitude above
this run's own median*, never an absolute cutoff. Pool instrumentation is completely blind to it —
the arm that died this way had a *healthier* pool than its control.

**4. The two are mutually blind.** Within E3: `E3-19-23` spent a third of its iterations with one
side winning essentially every game while `kl_div` stayed at its normal level; `E3-31-28` blew up
`kl_div` with side balance never once flagging. Both instruments are required, and a run is healthy
only when both say so.

**5. Critic quality never separates, in either direction.** `critic_auc`, `critic_brier_skill`,
`explained_variance` and `value_loss` fail to distinguish collapsed from healthy, and by some
measures the collapsing run scores *better*. This is not a paradox: a degenerate policy is an
**easy prediction problem**. Critic quality measures the critic, not the policy.

**6. `critic_base_rate` is worse than useless.** It is `max(p, 1−p)` — always ≥ 0.5 and carrying no
direction at all, so it cannot name a side. E3 recorded that the answer it appeared to give was
*backwards*; a threshold on it fired three times during E4-02-01 and was wrong every time.

**7. Side balance is a flag, not a verdict — and not even an alarm.** One side genuinely improving
faster produces the same reading as one side degenerating. Stronger than that: swept over trailing
windows of 100/200/300/400 iterations, the pinned fraction **does not separate healthy from
degenerate at any threshold**. The clean 320M arm the round robin ranks first sustains 0.50–0.72
pinned; the heavily one-sided arm reaches 0.54–0.87. They overlap, so any latched trigger fires on
the best run in the record. Transient total pinning is normal in healthy self-play. Print it,
never alarm on it. Its one virtue is that it is internal to self-play and needs no external
opponent, unlike win rate against HeuristicBot — which is itself a bad health metric.

**8. Dynamics metrics describe, they do not trigger.** `entropy`, `clip_frac`, `adv_std_raw`,
`logratio_max` are worth printing because when something *does* go wrong they say what *kind* of
wrong. No threshold on any of them has survived contact with a second dataset.

**9. The arbiter is per-seat win rate against a frozen reference.** Everything above routes
attention. Only this confirms.

**10. A candidate signature must be validated out-of-sample before it is acted on** — and, per the
scope rule, out-of-sample now means *another run in the same lineage*, not the other lineage. Two
candidates have already died this way; see [below](#candidates-that-failed).

---

## Why numbers do not transfer

The same metrics, fitted on one lineage's collapse and checked against the other's:

| metric | fitted on E4's pair | checked on E3's pair | outcome |
|:---|:---|:---|:---|
| `entropy` | standardised difference **5.6** — the strongest detector in E4 | gap of **0.014** against trajectories swinging 0.45–0.91 | **noise** |
| `clip_frac` | collapsed run **lower** | collapsed run **higher** | **direction reverses** |
| `adv_std_raw` | collapsed run **lower** | collapsed run **higher** | **direction reverses** |

An earlier version of this page recommended watching entropy on the strength of that 5.6. It was
wrong, and it was wrong in the specific way the scope rule predicts: **a dynamics metric fitted to
one collapse describes that collapse**, and its scale, and sometimes its sign, belong to the
lineage it was measured in.

The mechanisms nevertheless recurred across both lineages. That asymmetry — mechanisms transfer,
magnitudes do not — is the whole basis for how this page is organised.

---

## The protocol

Every trigger is self-relative. None requires a number from another run.

**ALARM — `NOPOOL`, structural, iteration 1.** `opp_pool_size` absent from the metrics
entirely. The run has no opponent pool; that is the failure, not evidence of it.

**ALARM — `POOLSTUCK`, structural.** `opp_pool_size` never exceeding 1 over at least 100
iterations. The pool is not growing.

**ALARM — `KLSPIKE`, self-relative.** `kl_div` jumping 10x or more above *this run's own* median
(with an absolute floor, so an early near-zero median cannot fire it). Catches mode 2.

**DESCRIPTION, never an alarm.** `us_episode_frac` (excluding rows with
`episodes_completed == 0` — see [the trap](#a-measurement-trap-that-produced-a-false-finding)),
`entropy`, `clip_frac`, `adv_std_raw`, `opp_win_rate_mean`, `logratio_max`. These say what *kind*
of wrong, once something else has fired.

**NEVER.** Critic quality, `critic_base_rate`, side balance as a trigger, `steps_per_sec`, win
rate against HeuristicBot.

**TO CONFIRM BEFORE ACTING.** Per-seat win rate against a frozen reference.

`tools/scripts/watch_run.py` implements the three alarms and prints the description metrics.
Validated by replay over all 54 runs in the record: **23 fire, 31 are silent, and every alarm
lands on a run already known to be bad** — 20 NOPOOL on the genuinely unpooled arms, POOLSTUCK on
E3-24-28, and KLSPIKE on exactly the two known KL-domination collapses (56x and 1070x their own
medians). No false positives. Two triggers were discarded during that validation for firing on
healthy arms: `opp_win_rate_mean > 0.9` (finding 2a) and any one-sidedness threshold (finding 7).

---

## E3 provenance

Kept for *what was learned*, deliberately without carrying the numbers forward.

**Pool starvation.** Found via a matched pair — the same run replayed from one resume state with
one accidental difference, a `dirname` bug that starved the pool to a single opponent. The cleanest
labelled control in the record.
[`../archive/E3_ladder/log/P15_X4b_collapse_is_pool_starvation.md`](../archive/E3_ladder/log/P15_X4b_collapse_is_pool_starvation.md).

**KL domination.** Found on an arm launched three minutes after its control with identical flags,
which declined with a *healthier* pool than that control. The KL term was measured at hundreds of
times the policy gradient on half its iterations.
[`../archive/E3_ladder/log/P15_kl_domination.md`](../archive/E3_ladder/log/P15_kl_domination.md).
Both arms that hit this mode used search (`search_ce_coef 0.5`), which the E4 arms do not — so
**whether this mode reproduces on E4 is untested**, and is an open experiment rather than a settled
result.

**Per-seat control.** The record of why `critic_base_rate` was dropped: the answer it appeared to
give was backwards.
[`../archive/E3_ladder/log/P15_control_per_seat.md`](../archive/E3_ladder/log/P15_control_per_seat.md).

**`ref_update_freq` is not a signal.** An arm carrying the slow reference anchor collapsed anyway.
Held at the 200,000 default by decision, as an axis to ablate deliberately later —
[`../findings/training/ref_update_freq_open_ablation.md`](../findings/training/ref_update_freq_open_ablation.md).

**Known-bad source.** `E3-24-28` logs `episodes_completed = 0.0` on every row and **cannot be used
for side-balance analysis at all**.

---

## E4 reference envelope

Built from E4 arms only, and the *only* numbers on this page admissible as E4 baselines. Provisional
— it grows as arms complete.

| arm | start | pool | steps | max `kl_div` | `us_episode_frac` pinned | verdict |
|:---|:---|:---|---:|---:|---:|:---|
| `E4-02-01` leg 1 | warm | yes | 240M | 0.10 | 1.8% | healthy |
| `E4-02-01` leg 2 | warm | yes | 320M | 0.08 | 13.2% | healthy, round-robin #1 |
| `E4-01-01` leg 1 | warm | **no** | 240M | 0.10 | 0.7% | degenerate by construction |
| `E4-01-01` leg 2 | warm | **no** | 240M | 0.12 | 13.4% | oscillating, never terminal |
| `E4-04-01` | **cold** | yes | 80M | 0.09 | 0.0% | **healthy, completed** |

Two things this table already establishes **within E4**:

* **`kl_div` has stayed ≤ 0.12 across every E4 leg**, healthy and degenerate alike. E4 has
  therefore **not yet exhibited the KL-domination mode**, consistent with it needing search.
* **`us_episode_frac` pinned ~13% does not imply collapse in E4** — the clean 320M arm that the
  round robin ranks first sits there too. It routes attention and nothing more.

**The first cold-start E4 reference (2026-09-19).** `E4-01-01` and `E4-02-01` are both
warm-started, and a warm start holds entropy higher for longer, so neither was ever a valid
comparison for a cold arm's annealing — reaching for E3's band instead, as was briefly done, is
exactly the error the scope rule forbids. `E4-04-01` is now the first completed cold-start pooled
arm, and is therefore the reference:

| | `E4-04-01`, cold, pooled, 80M |
|:---|---:|
| entropy | min 0.763, max 1.921, final 1.012 |
| max `kl_div` | 0.095 |
| `opp_pool_size` | reaches capacity 12 and stays |
| `us_episode_frac` | mean 0.443, **0.0%** pinned |
| health alarms | none fired |

Note how much of that is unremarkable: side balance never pinned once across 1,221 iterations,
and `kl_div` stayed an order of magnitude below anything alarming. `E4-03-01@80M` is the matched
cold-start arm and will be the second entry.

---

## Candidates that failed

* **`adv_std_raw` < 0.01** (with `explained_variance` → 1.0, the critic trivially perfect because
  every game ends the same way). Marks a contiguous 23-iteration event at ~218–221M in E4-01-01's
  unpooled continuation, US win fraction 0.00 throughout, and never fires in the pooled arm.
  **Status: one event in one arm, n=1, not validated against a second E4 arm.** Not a detector
  until it is.
* **The entropy gap**, refuted as described [above](#why-numbers-do-not-transfer).

Both looked convincing in-sample. That is the argument for requiring out-of-sample validation
before a candidate reaches the protocol.

---

## A measurement trap that produced a false finding

An earlier version of this page claimed `E3-24-28` was 100% one-sided with healthy KL. **That was
an artefact of my own sweep.** The arm logs `episodes_completed = 0.0` on every row, and the sweep
computed `won_us / max(1, episodes_completed)` — so "no episodes recorded" was silently read as
"the US won none of them", manufacturing a perfect one-sidedness score from missing data.

**Require `episodes_completed > 0` and report the excluded rows**, rather than defaulting a
denominator. `us_episode_frac` in `watch_run.py` returns `None` for such rows by construction.

---

## What this still does not establish

* **Two modes found; there may be more.** Each was found by having a labelled control. A third
  cause would likely need a third instrument, and nothing here predicts which.
* **KL domination is untested on E4** — both arms that showed it used search.
* **`logratio_max` and `ratio_negadv_max` remain untested.** They were the earliest and cleanest
  detectors on E4's pair, firing early with no false positive, and should not be trusted until a
  second E4 arm exercises them.
