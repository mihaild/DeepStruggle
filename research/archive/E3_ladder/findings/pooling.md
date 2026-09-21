# Pooling — what we know about pooled vs non-pooled opponents

Everything in this repository that bears on *pooled versus non-pooled*, in one place, with the
current verdict. **Update discipline: rewritten in place.** The experiments themselves stay in
[`../../../log/seed_variance_and_pooling.md`](../../../log/seed_variance_and_pooling.md), which is
append-only; the arms are indexed in [`../../runs.md`](../runs.md).

This file exists because the question was hard to answer from the record. The evidence was spread
over one log file, two plan files, three untracked tournament reports and nine `metadata.json`
descriptions, and the word "pool" names four unrelated things in this project.

> **Before reading any pooled arm's result, check the pool was actually there.** Until
> 2026-09-17, `--resume <run-directory>` silently rebuilt the opponent pool from the *parent* of
> the run directory, found no snapshots, and started with a pool of **one** — a frozen copy of the
> current policy, beaten ~99% of the time. The run logs normally and nothing announces it. This
> produced the entire "X4b collapse":
> [`../log/P15_X4b_collapse_is_pool_starvation.md`](../log/P15_X4b_collapse_is_pool_starvation.md).
> `opp_pool_size` and `opp_win_rate_mean` in `training_metrics.jsonl` are the check, and a
> resumed run now prints a warning when it hits that branch.

## First: which pool

| the word, where it appears | what it means | verdict |
|:---|:---|:---|
| **opponent pool** — `--opponent-self-pool`, `opponent_self_pool` in metadata, arms E3-19 / E3-20 | a fraction of rollout environments play the current policy against a snapshot of *itself* from earlier in the same run | the subject of this file |
| **start pool** — `--start-pool-frac`, `ai/training/start_pool.py`, dirs `sp_pool_*` / `sp2_pool_*` | a fraction of episodes *begin* from a saved mid-game position instead of turn 1 | settled negative, [below](#the-start-state-pool-a-different-mechanism-settled-negative) |
| **pooled head-to-head** — "pooled over sixteen snapshot pairings", "pooled win rate" | an *evaluation* protocol: rate four late snapshots a side and pool the result, to average out where a run stopped | a measurement method, [`../../../method/running_experiments.md`](../../../method/running_experiments.md) |
| **the pooled trunk** — "pooled 512-float trunk", `data/checkpoints/_pool_stage/` | global average pooling inside the network / the staging directory for the evaluation protocol above | architecture, [`architecture.md`](architecture.md) |

Only the first is what follows. A search for "pool" returns all four, which is most of why this
was hard to find.

## Current verdict, in one line

**On side balance the opponent pool works and the effect is large. On strength it is better
supported than this file used to say: pooled arms improve over the second 80M and unpooled arms
decay — 4 of 4 against 1 of 4, one-tailed p = 0.057 — and pooled wins 61.5% of 6,400
cross-condition games at 160M. It is still not conclusive at the arm level, where four seeds a
condition give p = 0.10 on the levels.** And the endpoint pre-registered for the deciding
experiment was abandoned mid-flight, for a good reason that was never written down until now.

*Revised 2026-09-16.* The previous verdict — "not established, because the spread between pooled
arms is larger than the gap between conditions" — applied a level comparison and a crude
credibility rule to data that supports a **paired** one. See
[§3a](#3a-the-paired-comparison-the-level-comparison-was-hiding) immediately below §3.

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
([`../../../log/seed_variance_and_pooling.md`](../../../log/seed_variance_and_pooling.md)).

Two confounds compound it. E3-20-22 ran with `seed=None` against E3-17-22's `--seed 20260922`, so
the original pair differed in the env stream and the network initialisation as well as the pool
(`E3-17-24/metadata.json`). And E3-19-23, the "seed replicate" of E3-19-22, resumed from 90M
rather than 80M because the 80M state had been pruned — so the arm that collapsed to 1566 was
also the one started from a different place.

### 2. Self-play balance cannot see what the pool does

Re-measuring the 80M → 160M US decay against **external** opponents, 250 games per pairing
([`../../../log/seed_variance_and_pooling.md`](../../../log/seed_variance_and_pooling.md)):

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
[`../../../method/measurement_pitfalls.md`](../../../method/measurement_pitfalls.md).

### 3a. The paired comparison the level comparison was hiding

`arena_p12` rates every arm at **both** 80M and 160M in one field, so the within-arm gain over the
second 80M is measurable. Pairing removes the arm-to-arm variance that makes the level comparison
inconclusive.

| arm | condition | @80M | @160M | Δ |
|:---|:---|---:|---:|---:|
| P22 | pooled | 1935.6 | 2067.8 | **+132.2** |
| P27 | pooled | 2051.1 | 2151.0 | **+99.9** |
| P28 | pooled | 2027.4 | 2096.1 | **+68.7** |
| P29 | pooled | 1901.1 | 1929.6 | **+28.5** |
| N22 | no pool | 2037.3 | 1952.3 | −85.0 |
| N24 | no pool | 1925.9 | 1912.8 | −13.1 |
| N25 | no pool | 1891.9 | 1989.0 | +97.1 |
| N26 | no pool | 2122.5 | 1995.7 | −126.8 |

**Pooled 4 of 4 improved, mean +82.3 Elo; unpooled 1 of 4, mean −32.0.** Exact Mann-Whitney on the
deltas, U = 14 of 16, one-tailed **p = 0.057** — against p = 0.100 for the same test on the levels.

The finding this supports is more specific than "pooled is stronger": **unpooled arms decay over
the second 80M and pooled arms do not.** That is what a pool of past selves is for, it is a
late-training phenomenon, and it is invisible before 80M — which is also why the level comparison
is the weaker instrument here. At 80M the conditions genuinely overlap; `N26-80`, an unpooled arm,
is the second-strongest model in the whole 20-model field.

Two further figures from the same field, both stronger than the Elo means:

* direct cross-condition head-to-head at 160M, all 16 pairings, ~6,400 games: pooled **61.5%**,
  winning **13 of 16** pairings. Per pooled arm against the four unpooled: P27 70.9%, P28 67.1%,
  P22 63.2%, P29 44.6%.
* **3 of 4** pooled arms rank above *every* unpooled arm. The 221-Elo pooled spread that the old
  verdict leaned on is one arm: drop P29 and the pooled spread is 83, the same as the unpooled one.

So the honest split is between units of analysis. Per *game* the result is decisive (SE ≈ 0.6 pp on
61.5%); per *arm* it is four against four and p = 0.10 on levels, 0.057 paired. One more seed a
condition would likely settle it, and is cheaper than anything else queued.

### 3b. A pooled game is played against five different snapshots

Measured 2026-09-16, from `episodes_completed` in the arms' own logs.

`start_iteration()` draws one opponent and every mixed environment uses it for that iteration --
`buffer_size` = 128 steps per environment. A game does not finish in 128 steps:

| arm | mean episode | opponents faced in one game |
|:---|---:|---:|
| E3-20-28 (pooled) | **626.5** micro-actions | **4.89** |
| E3-17-25 (no pool) | 370.7 micro-actions | n/a |

So **every pooled game this project has run was played against roughly five different frozen
snapshots in sequence** -- at about 7 turns a game, an opponent swap every ~1.4 turns. With
capacity 12, each game faces a random ~40% subset of the pool, consecutively.

"The learner played a pool snapshot" has therefore never been accurate. It played a *composite*
whose identity changes five times a game. In a ten-turn positional game no snapshot ever executes
a plan spanning turns; it is swapped out mid-plan, and what the learner faces is an average
opponent that exists as no policy.

**This is not what the module documents.** "Diversity accumulates across iterations instead of
within them" describes diversity across *environments* within an iteration. That an episode
outlives an iteration is never stated, and the within-episode switching is an unremarked
consequence of the two facts sitting side by side.

Three things follow.

* It is why the pool had no per-opponent win-rate instrument for so long. There was no stable
  opponent identity for a game to be attributed to. PFSP needed exposure-weighted attribution --
  splitting a result across the snapshots that actually played it -- before it could mean anything.
* It narrows what `E3-23-28` tests. PFSP changes which snapshot is drawn each iteration, but a game
  still faces about five of them, so the arm shifts the *composition of the mixture* rather than
  the identity of a per-game opponent. Still one factor, but a narrower claim than "prioritising
  who you play".
* Pooled games run **69% longer** than unpooled ones (626.5 against 370.7 micro-actions), which is
  consistent with §3's balance result: closer games last longer instead of ending in an early
  autowin. It also means a pooled arm sees fewer episodes per step than an unpooled one, so any
  per-episode statistic compared between the conditions is drawn from different sample sizes.

**Whether the switching is harmful is open.** Against it: the learner never meets a coherent
adversary. For it: it is within-episode opponent diversity, and diversity is the mechanism this
file credits for the balance result. Holding one opponent for a whole episode is a clean follow-up
arm and is as likely to be a regression as a fix -- it trades within-episode diversity for
coherence, and this file's own evidence is that diversity is what does the work.

### 3c. PFSP on the pool draw is a null (E3-23-28, 2026-09-16)

`--opponent-pfsp` weights the draw by the learner's smoothed win rate against each snapshot
instead of drawing uniformly, with `var` weighting -- x(1-x), peaking on evenly matched opponents
-- and a 0.10 uniform floor. One factor against `E3-20-28`: same seed, envs, budget, reward,
architecture, eta, pool frac and capacity, and the same `--snapshot-every-steps 5000000`, verified
by identical observed pool growth (1@0.1M, 2@5.1M, 3@10.1M, 4@15.1M, 5@20.1M on both).

**Strength, the pre-registered primary endpoint** (`data/tournaments/E3-23-28_vs_E3-20-28/`,
8 models, 45,000 games, anchored HeuristicBot = 1500):

| steps | uniform | PFSP | delta | PFSP head-to-head |
|---:|---:|---:|---:|---:|
| 20M | 1816.9 | 1834.8 | +17.9 | 51.3% |
| 40M | 2016.3 | 2068.4 | +52.1 | 60.0% |
| 80M | 2208.6 | 2206.9 | −1.7 | 49.5% |
| **160M** | **2284.0** | **2295.6** | **+11.6** | **57.1%** |

**+11.6 Elo at the endpoint is nothing.** §3 records a within-condition seed spread of 83 Elo
(unpooled) and 221 Elo (pooled) across four seeds in one field; a single-seed arm cannot resolve a
twelve-point difference, and the pattern across budgets is not monotone.

**Critic, secondary.** Over all 33 windows: `critic_auc` +0.0096, 95% CI [−0.005, +0.024];
`critic_brier_skill` +0.0085, CI [−0.028, +0.045] — null. Restricted to the 15 windows where both
runs were scored on comparable win/loss mixes: +0.0211, CI [+0.005, +0.038] and +0.0368, CI
[+0.005, +0.069] — positive. The all-window null is driven by one block, 115–140M, where the arm's
own base rate spiked to 0.82–0.87 against the baseline's 0.54–0.65. That subset analysis was
introduced at 75M, when the all-window result was already positive, so it was not built to rescue
anything; it still carries less weight than a pre-registered one, and the pre-registered comparison
is the null.

**Three things bound this null rather than settling the question.**

* **The treatment was mild by construction.** At capacity 12 the reweighting settled at 1.1–1.2×
  over uniform (`opp_pfsp_entropy` 0.98, `opp_pfsp_max_prob` 0.13 against a uniform 0.111). A null
  on a 1.2× nudge is not a null on prioritisation. The follow-up is a 0.0 uniform floor, not
  abandoning it.
* **One seed.** §3 needed four a condition and still reached only p = 0.057 paired.
* **`adv_std_raw` is untouched** — +0.0002, CI [−0.004, +0.004] over 22 windows. PFSP did not
  degrade the advantage signal the pool exists to protect.

**It also refutes this module's own prediction.** `ai/training/opponent_pool.py` argued
prioritisation should *hurt*, because the pool's mechanism is that weak old snapshots give the
learner's trailing side winnable games and prioritising strips those out. It did not hurt.

**The one concrete behavioural difference is worth following.** The arm's self-play base rate
spiked to 0.82–0.87 for roughly 25M steps around 115–140M and then recovered. Holding side balance
is the pool's strongest documented effect (§3), so a prioritiser that lets it slip for 25M steps is
the thread to pull — and it is exactly the oscillation
[`../plans/P15_breaking_the_cycle.md`](../plans/P15_breaking_the_cycle.md) exists to measure.

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

**Qualified 2026-09-16 by the frozen-anchor sweep**
([`../log/P15_X0_frozen_anchors.md`](../log/P15_X0_frozen_anchors.md)). The −6.9 pp above is
measured across that tournament's whole field. Against a *fixed* peer anchor the same checkpoint,
`E3-20-28@320M`, reads **−41.5 pp** — six times larger. Both numbers are correct and they answer
different questions: a field of the arm's own relatives lets a shared drift cancel, because every
opponent drifted too, while a frozen peer cannot drift and so does not cancel. The pooled arm is
better balanced *than the unpooled control*, and it is not balanced. It is also the arm that runs
away monotonically after 200M rather than oscillating, which the field-averaged number cannot
show.

### 5. The pool costs nothing in strength at 80M and the arms are not converged

Three facts that bound how much weight the above can carry, all from
[`../../../log/seed_variance_and_pooling.md`](../../../log/seed_variance_and_pooling.md):

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
([`../../../log/early_training_signal.md`](../../../log/early_training_signal.md) §3).

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
  quantity ([`../../../plans/restoring_advantage_signal.md`](../plans/restoring_advantage_signal.md)).

## What the record does not say

* ~~**External side balance averaged over the final 40M**, for the 4 × 4.~~ **Computed
  2026-09-16, and it cannot answer the question.** Averaged over 120–160M against `HeuristicBot`:
  no pool 8.8 pp (spread 2.3–15.5), pooled 8.0 pp (spread 1.5–17.0) — fully overlapping. The metric
  is **saturated**: every arm beats `HeuristicBot` above 89% and `RandomBot` above 99% by 120M, so
  neither can express a gap the peer tournament separates completely. The training logs contain no
  opponent strong enough to rate these arms, which means there is no in-run instrument for this
  question at all after ~120M. Recorded in
  [`../../../log/measurement_bugs.md`](../../../log/measurement_bugs.md).
* **E3-18-22 has no writeup at all.** It is P10's experiment 1 — continue unchanged, does the
  critic recover on its own — and it is the control that says whether any intervention was needed.
  Its final checkpoint was rated in `arena_heads`, where it **tops the field at 1978.0**
  ([`../../checkpoints.md`](../checkpoints.md)); that number is now recorded, but the experiment
  it belongs to still has no analysis.
* **E3-19's own question was never answered.** E3-19 was judged on `adv_std_raw` and critic
  AUC/Brier staying up, per its metadata and [`../../../plans/P10_opponent_sampling.md`](../plans/P10_opponent_sampling.md);
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
