# Pooling — what we know about pooled vs non-pooled opponents

Everything in this repository that bears on *pooled versus non-pooled*, in one place, with the
current verdict. **Update discipline: rewritten in place.** The experiments themselves stay in
[`../../log/seed_variance_and_pooling.md`](../../log/seed_variance_and_pooling.md), which is
append-only; the arms are indexed in [`../../runs.md`](../../runs.md).

This file exists because the question was hard to answer from the record. The evidence was spread
over one log file, two plan files, three untracked tournament reports and nine `metadata.json`
descriptions, and the word "pool" names four unrelated things in this project.

## First: which pool

| the word, where it appears | what it means | verdict |
|:---|:---|:---|
| **opponent pool** — `--opponent-self-pool`, `opponent_self_pool` in metadata, arms E3-19 / E3-20 | a fraction of rollout environments play the current policy against a snapshot of *itself* from earlier in the same run | the subject of this file |
| **start pool** — `--start-pool-frac`, `ai/training/start_pool.py`, dirs `sp_pool_*` / `sp2_pool_*` | a fraction of episodes *begin* from a saved mid-game position instead of turn 1 | settled negative, [below](#the-start-state-pool-a-different-mechanism-settled-negative) |
| **pooled head-to-head** — "pooled over sixteen snapshot pairings", "pooled win rate" | an *evaluation* protocol: rate four late snapshots a side and pool the result, to average out where a run stopped | a measurement method, [`../../method/running_experiments.md`](../../method/running_experiments.md) |
| **the pooled trunk** — "pooled 512-float trunk", `data/checkpoints/_pool_stage/` | global average pooling inside the network / the staging directory for the evaluation protocol above | architecture, [`architecture.md`](architecture.md) |

Only the first is what follows. A search for "pool" returns all four, which is most of why this
was hard to find.

## Current verdict, in one line

**On side balance the opponent pool works and the effect is large; on strength it is not
established, because the spread between pooled arms is larger than the gap between conditions;
and the endpoint that was pre-registered for the deciding experiment was abandoned mid-flight,
for a good reason that was never written down until now.**

Longer: four pooled arms and four unpooled arms at 160M, one tournament, 76,000 games, separate
completely on side balance — every pooled arm inside 7.2 pp of even, every unpooled arm at least
28.4 pp USSR-favoured. The same tournament puts pooled ahead by ~99 Elo on arm means, but the
pooled arms span 221 Elo among themselves, which fails the credibility rule the experiment set
for itself in advance. Extending two pooled arms to 320M did not improve either.

## The experiments, in order

### 1. The first pooled arms — n=1 per condition, and unpaired

`arena80_160`, 8 models, 16,800 games (**untracked**: `data/checkpoints/arena80_160/report.md`).

| label in the report | arm | Elo | side gap |
|:---|:---|---:|---:|
| C-poolfrombalanced-160M | E3-19-22 — pool from the 80M resume | **2081.3** | +12.4 pp |
| B-scratchpool-160M | E3-20-22 — pool from scratch | 2037.2 | +10.0 pp |
| A-nopool-80M | E3-17-22 @80M | 1998.1 | +10.4 pp |
| A-nopool-160M | E3-17-22 @160M | 1938.9 | +27.7 pp |
| B-scratchpool-80M | E3-20-22 @80M | 1906.1 | +21.5 pp |
| D-poolfromlocked-160M | E3-19-23 — pool from 90M, after the lock | 1566.1 | +14.4 pp |

This produced the claims that pooling from a balanced checkpoint preserves balance, that pooling
after the side imbalance has locked in repairs nothing, and the Elo ordering across the four.
**All three were retracted** when a second unpooled seed landed: two no-pool arms differing only
in seed sat 0.1183 apart on the balance endpoint while pooled minus no-pool was 0.0169, seven
times smaller, with the pooled arm *between* the two unpooled ones
([`../../log/seed_variance_and_pooling.md`](../../log/seed_variance_and_pooling.md)).

Two confounds compound it. E3-20-22 ran with `seed=None` against E3-17-22's `--seed 20260922`, so
the original pair differed in the env stream and the network initialisation as well as the pool
(`E3-17-24/metadata.json`). And E3-19-23, the "seed replicate" of E3-19-22, resumed from 90M
rather than 80M because the 80M state had been pruned — so the arm that collapsed to 1566 was
also the one started from a different place.

### 2. Self-play balance cannot see what the pool does

Re-measuring the 80M → 160M US decay against **external** opponents, 250 games per pairing
([`../../log/seed_variance_and_pooling.md`](../../log/seed_variance_and_pooling.md)):

| arm | US vs external @80M | @160M | change | self-play said |
|:---|---:|---:|---:|---:|
| E3-17-22 (no pool) | 69.0% | 32.0% | −37.0 pp | −36.3 pp |
| E3-17-24 (no pool) | 49.2% | 38.2% | −11.0 pp | −29.4 pp |
| E3-20-22 (pooled) | 51.6% | **76.4%** | **+24.8 pp** | +0.5 pp |

Self-play win rate is a *relative* quantity between two co-evolving seats: it reported −29 for an
arm that lost 11 and +0.5 for an arm that gained 25. This matters here specifically, because a
pool of past selves is a defence against co-evolutionary drift and a self-play metric cannot see
that defence working, by construction. The pooling "null" in §1 was measured on a self-play
endpoint.

### 3. The 4 × 4 replication

Four seeds per condition — no pool E3-17-22/24/25/26, pooled E3-20-22/27/28/29 — at 80M and 160M
each, in one 20-model tournament, 76,000 games (**untracked**:
`data/checkpoints/arena_p12/report.md`; staged in `data/checkpoints/arena_p12/`). Prediction,
registered before the arms landed: **null**.

Side balance across all opponents, |as USSR − as US|:

| | no pool | pooled |
|:---|---:|---:|
| @80M | 7.4, 4.3, 17.3, 20.9 → mean **12.5 pp** | 9.5, 3.3, 19.2, 1.2 → mean **8.3 pp** |
| @160M | 32.9, 30.7, 40.8, 28.4 → mean **33.2 pp** | 5.6, 1.9, 6.7, 7.2 → mean **5.4 pp** |

At 80M the conditions overlap and are indistinguishable. At 160M they **separate completely**:
the worst pooled arm (7.2 pp) is better balanced than the best unpooled arm (28.4 pp), 4 against
4, with no ordering in common. Over that interval every unpooled arm's gap widens — by 9.8, 11.1,
25.5 and 36.5 pp — while three of the four pooled arms' gaps narrow and the fourth widens by 6.0.
This is the strongest pooling result in the record, and the prediction of a null was wrong.

Bradley-Terry Elo from the same tournament:

| | no pool | pooled | difference |
|:---|---:|---:|---:|
| @80M mean | 1994.4 | 1978.8 | −15.6 |
| @160M mean | 1962.5 | 2061.1 | **+98.7** |
| @160M within-condition spread | 83 Elo | **221 Elo** | |

**The Elo half does not clear its own bar.** The analysis plan fixed in advance says a condition
effect is credible only if it exceeds the spread between same-condition arms; the pooled arms
span 221 Elo (P27-160 at 2151.0 down to P29-160 at 1929.6) against a 98.7 Elo condition
difference. So: balance, yes, decisively; strength, not established by this experiment.

**The pre-registered primary endpoint was abandoned, for cause.** It was *mean
|ussr_win_rate − 0.5| over the final 40M*, read from the per-iteration training metric and chosen
because a final checkpoint samples wherever in an oscillation the budget happened to stop. It was
dropped when **self-play side advantage turned out to be uncorrelated with strength against a
fixed opponent** — an arm can drive its own USSR win rate to 0.5 by having both of its sides drift
together, which is a statement about the pair, not about either one. Balancing an agent against
itself and beating a third party are different quantities, and only the second is what the project
is for.

So the tournament figures above are not a substitute chosen after the fact to flatter the result;
they are the endpoint that replaced a metric found to be measuring the wrong thing. What the
substitution costs is still real and is the reason this section says *suggestive* rather than
settled: external side balance is read at a single final checkpoint, so it reintroduces exactly the
oscillation-sampling problem the original endpoint was designed to avoid. The honest version of
this experiment measures external side balance *averaged over the final 40M*, which nothing has
done yet.

**This generalises past pooling.** Any endpoint computed from self-play alone inherits the same
defect — it can be satisfied by both sides moving together. See
[`../../method/measurement_pitfalls.md`](../../method/measurement_pitfalls.md).

### 4. Extending to 320M

The two extreme pooled arms extended 160M → 320M, with E3-17-26 as the unpooled control — chosen
because it is the *strongest* unpooled arm, making it the conservative test. 10 models, 22,500
games (**untracked**: `data/checkpoints/arena_320/report.md`).

| arm | Elo @160M | @320M | side gap @160M | @320M |
|:---|---:|---:|---:|---:|
| E3-20-28 (pooled, most balanced) | 2150.4 | 2151.6 | −2.9 pp | −6.9 pp |
| E3-20-29 (pooled, least balanced) | 2015.6 | **1925.1** | +4.5 pp | **+14.8 pp** |
| E3-17-26 (no pool, control) | 2050.1 | 2047.6 | +27.4 pp | +29.6 pp |

**No arm gained.** Both pooled arms moved *away* from balance rather than toward it, in the same
direction, which under the rule fixed in advance ("regression pulls them together, real damping
moves them the same way") rules out regression to the mean as the explanation and rules out
damping as the finding: the imbalance is not converging, it is oscillating. The unpooled control
stayed where it was, roughly 28–30 pp USSR-favoured, at both budgets.

The pooled arms are still far better balanced than the unpooled control at 320M, so §3's result
survives the extension; what does not survive is the hope that more steps settle it.

### 5. The pool costs nothing in strength at 80M and the arms are not converged

Three facts that bound how much weight the above can carry, all from
[`../../log/seed_variance_and_pooling.md`](../../log/seed_variance_and_pooling.md):

* 160M is **not a converged state** — entropy 1.2–1.3, `clip_frac` 0.11–0.20, KL 0.026–0.035 are
  all healthy at the budget end and two of three arms measured were still recovering.
* 80M may be better than 160M on strength. The unpooled control ranks above its own 160M
  descendant (1998 vs 1939) in the `arena80_160` field, and `N26-80` ranks second of twenty in
  `arena_p12` while `N26-160` ranks eighth.
* Never compare Elo or side balance across tournaments. `E3-20-28@160M` reads −1.9 pp in the
  20-model `arena_p12` field and −2.9 pp in the 10-model `arena_320` field; `E3-17-26@160M` reads
  +32.9 and +27.4. Within one tournament the comparison holds; between two it drifts ~5 pp.

## The start-state pool: a different mechanism, settled negative

`--start-pool-frac` resumes a share of self-play episodes from saved mid-game positions. It is
unrelated to the opponent pool and its verdict is the opposite one
([`../../log/early_training_signal.md`](../../log/early_training_signal.md) §3).

* The first A/B (`sp_pool_on/off`) was **withdrawn as confounded** — the arms got 67.1M steps
  against 31.0M because the control stalled under evaluation cost.
* The step-budgeted repeat (`sp2_pool_on/off`, both arms exactly 1,190 iterations) is settled and
  negative: head-to-head 32.7% for the pool arm, Elo 1645.1 against 1760.2, more empty
  battlegrounds at turn 8, not fewer.
* The mechanism itself works — replayed on 1,000 turn-8 positions the pool arm wins 53.4% ± 2.2%.
  The *allocation* is what fails: +3.4 points in a regime that occurs in 5.6% of episodes, paid
  for with −17.3 points from the opening.
* It was then **demoted a second time** for the advantage-collapse problem specifically:
  `is_salvageable` filters on board balance, not on outcome uncertainty, which is the wrong
  quantity ([`../../plans/restoring_advantage_signal.md`](../../plans/restoring_advantage_signal.md)).

## What the record does not say

* **External side balance averaged over the final 40M**, for the 4 × 4. The original endpoint
  was abandoned because self-play balance does not track strength (§3); its replacement is read at
  a single checkpoint, which reintroduces the oscillation problem. Neither has been computed from
  the eight
  `training_metrics.jsonl` files, the headline result rests on an instrument that was chosen
  after the arms ran.
* **E3-18-22 has no writeup at all.** It is P10's experiment 1 — continue unchanged, does the
  critic recover on its own — and it is the control that says whether any intervention was needed.
  Its final checkpoint was rated in `arena_heads` and nothing was written down.
* **E3-19's own question was never answered.** E3-19 was judged on `adv_std_raw` and critic
  AUC/Brier staying up, per its metadata and [`../../plans/P10_opponent_sampling.md`](../../plans/P10_opponent_sampling.md);
  the only numbers anyone recorded for it are Elo and side balance from `arena80_160`, which are
  not those instruments.
* **What fraction, and how big a pool.** Every pooled arm ran frac 0.30 and capacity 12. The
  P10 plan queued frac 0.15 and 0.50 and P11 listed them as *to run*; neither exists on disk.
* **The recommendation the plan made if the result came back null** — move to `eta` rather than
  more pooling seeds — was written for a null. The result was not null on balance, so that
  branch was never taken and never retracted either.
* **Three of the five results above exist only in git-ignored files.** `arena80_160`,
  `arena_p12` and `arena_320` are `report.md` files under `data/checkpoints/`. The numbers quoted
  here are the tracked copy.
