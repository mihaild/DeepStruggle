# P15-X2 — does a slower reference anchor prevent the collapse on its own?

**Launched 2026-09-17, `E3-30-28_20260917_104226`. Interim; the arm is running to 240M.**

P15's own diagnosis of `ref_update_freq = 200000`: the anchor is refreshed ~400× per 80M leg, so
it *tracks* the cycle rather than anchoring it, and NashPG's convergence story assumes the inner
loop approaches the regularized fixed point **before** the reference moves. X2 proposes 5M and
20M. This is the 5M cell, with no search, from scratch.

Two collapses on 2026-09-17 motivated running it now rather than in plan order: the X4b search arm
lost 413 Elo in 5M steps with KL against π_ref spiking to 31.99
([`P15_X4b_search_during_rl.md`](P15_X4b_search_during_rl.md)), and its no-search control
went somewhere strange at ~65M further steps, with `critic_base_rate` reaching 0.96 and the critic
falling to a base-rate predictor. (`critic_base_rate` is the majority-class rate and carries no
direction, so it cannot name a side; the per-seat evidence that the control's USSR play degraded
comes from a frozen-field tournament, not from this trace.)

## The comparison

`E3-20-28` is the same recipe **from scratch** at `ref_update_freq = 200000`. Its `metadata.json`
matches this arm on arch, `num_envs`, pool fraction and capacity, reward scheme, η, identity dim,
per-entity heads, graph layers, self-transform and value-dist coefficient. Both start cold, so
`critic_base_rate` is comparable at matched step counts — unlike the resumed arms, whose counters
sit on top of 200M.

**They differ in seed** (20260928 against 20260930), and that is the load-bearing caveat below.

| from scratch, at | `E3-20-28` ref 200k | `E3-30-28` ref 5M |
|---:|---:|---:|
| base_rate 10M | 0.5420 | 0.5552 |
| base_rate 20M | 0.5888 | 0.5947 |
| base_rate 40M | 0.6482 | 0.6124 |
| base_rate 60M | 0.5258 | 0.6983 |
| base_rate 70M | 0.5433 | 0.7038 |
| **first > 0.80** | **44,564,480 steps** | **never — max 0.7395 by 72M** |
| entropy 10M | 0.7380 | **1.3194** |
| entropy 70M | 0.7349 | 0.7566 |
| `critic_auc` 10M | 0.7342 | 0.6418 |
| `critic_auc` 70M | 0.7915 | 0.7909 |

## What `critic_base_rate` can and cannot say here

Every base-rate number below is a **hint, not a result**. One-sidedness of self-play is a fact
about the pair: 0.80 is equally consistent with one side collapsing and with both sides improving
at different rates, and it cannot tell those apart
([`method/measurement_pitfalls.md`](../method/measurement_pitfalls.md)). The question "does a
slower anchor stop a side being given away" is answered by **per-side win rate against a frozen
opponent**, which is a tournament measurement, and the table below is not that.

It is kept because it is free, it is logged every iteration, and it is the cheapest available
signal of *when* to spend a tournament. It is not evidence.

## Reading, with the caveat first

**One seed each, and the seeds differ.** This project has a findings entry on seed variance for a
reason, and a single pair cannot separate a treatment effect from a seed draw. Everything below is
a reason to finish the arm, not a result.

With that said, two things are visible:

* **The base-rate comparison below is superseded** by the per-seat section that follows, which
  shows both of this arm's seats improving while its base rate climbed from 0.61 to 0.79. The two
  runs' base rates differ; that says their self-play mixtures differ and nothing about either
  side's strength. Kept only as a record of what the cheap signal looked like.
* **The slow anchor holds much more entropy early**: 1.32 against 0.74 at 10M, converging by 70M.
  That is the expected shape — a reference that lags pulls less hard toward whatever the policy
  has just become — and it costs early critic accuracy, AUC 0.64 against 0.73 at 10M, which is
  recovered by 60M.

So the slower anchor buys a slower, flatter start and a lower one-sidedness ceiling so far. Whether
it prevents the decline that `E3-20-28` shows past its 200M peak is a question only the full 240M
answers, and a second seed would be needed before believing any of it.

## The search arm at ref 5M: 86 Elo behind at 5M, which is the expected cost

`E3-31-28` is one factor against `E3-29-28` — same warmup checkpoint, seed, search configuration
and snapshot cadence, only `--ref-update-freq` differs. First matched pair, temperature 0, 300
games a side (`/workspace/data/tournaments/P15_X2_search_5M/`):

| model | Elo | vs control |
|:---|---:|---:|
| `arm_ref200k` @5M — the arm that later collapsed | **1756.6** | +229.4 |
| `arm_ref5M` @5M | **1670.6** | **+143.4** |
| `source_200M` | 1531.4 | |
| `control_nosearch` @5M | 1527.2 | — |
| `anchor_280M` | 1500.0 | |

**The slower anchor costs 86 Elo at this point.** It is still 143 Elo above the no-search control,
so the search term is doing its work; it is simply doing it more slowly, which is the same shape
the from-scratch pair shows — more entropy, slower early critic, lower ceiling on one-sidedness.

P15 X2 names this risk in its own text: *"a slow anchor over-regularizes fresh learning — which is
why the arms resume from 80M rather than start cold."* This arm resumes from 200M and pays the
cost anyway, in the first 5M.

### At 10M the deficit has more than halved

`P15_X2_search_10M`, temperature 0, 300 games a side:

| model | Elo | vs `frozen_200M`, USSR / US |
|:---|---:|---:|
| `ref200k`@10M | 1759.8 | 83.7% / 80.0% |
| **`ref5M`@10M** | **1727.8** | **81.7% / 80.7%** |
| `frozen_200M` | 1500.0 | — |
| `control`@10M | 1491.1 | |
| `anchor_280M` | 1481.9 | |

**32 Elo apart, down from 86 at 5M.** Per seat against the 200M start the two are effectively
indistinguishable — 81.7 / 80.7 against 83.7 / 80.0 — and both beat that start by 80 points or
more on each seat. Against the shared no-search control the slow-anchor arm is +236.7 and the fast
one +268.7.

So the slow anchor's cost is **transient rather than a standing handicap**: it starts slower and
catches up. That matters for reading the decisive 25M point, because a surviving arm would no
longer have to be discounted for being permanently weaker.

### Per seat, both arms lose the USSR seat at 15M-20M

Win rate against the frozen `p28_200M`, as USSR / as US, from the same matched tournaments:

| arm | 5M | 10M | 15M | 20M |
|:---|:---|:---|:---|:---|
| `ref5M` (slow anchor) | 72.7 / 69.3 | 81.7 / 80.7 | **82.7 / 83.0** | **67.7** / 79.7 |
| `ref200k` (collapsed at 25M) | 81.0 / 77.7 | 83.7 / 80.0 | 80.3 / 79.7 | **67.0** / 77.7 |
| control, no search | 43.0 / 49.3 | 38.0 / 58.0 | 39.3 / 55.3 | 53.7 / 46.3 |

**Both search arms lose the USSR seat between 15M and 20M** — −15.0 pp for the slow anchor and
−13.3 pp for the fast one — while both hold the US seat within about 3 pp. Same seat, same window,
nearly the same size.

This is the precursor recorded for `E3-29-28` before its 25M cliff
([`P15_X4b_search_during_rl.md`](P15_X4b_search_during_rl.md)), and **the slower anchor has not
prevented it.** Whatever drives the decline is not the reference refresh interval.

**The aggregate Elo hid it.** At 20M the table puts `ref5M` 20 points *ahead* of `ref200k`, which
reads as the anchor working. Per seat, both are losing the same seat at the same rate, and the
slow-anchor arm simply started the fall from a slightly higher place. That is the third time in
this session that an aggregate number was true and the inference drawn from it was not — after
`critic_base_rate` and after the field-averaged side split.

### By 15M the deficit is gone

| matched steps | `ref200k` | `ref5M` | gap |
|---:|---:|---:|---:|
| 5M | 1756.6 | 1670.6 | −86.0 |
| 10M | 1759.8 | 1727.8 | −32.0 |
| 15M | 1761.3 | 1755.5 | **−5.8** |

Per seat against `frozen_200M` at 15M the slow anchor is marginally ahead on both seats, 82.7 /
83.0 against 80.3 / 79.7, while the head-to-head still gives `ref200k` a sliver — a tie within
noise either way.

**The combination is what matters: equal strength at 15M, but entropy 0.665 against 0.481.** The
slow-anchor arm has reached the same playing level while staying much less sharpened, and
over-sharpening is what preceded the other arm's collapse — logits driven about 2e5 apart, so the
policy assigned effectively zero probability to actions the searcher still visited. It arrives at
15M in a materially different internal state, roughly 5M steps before `E3-29-28` began to fall
apart.

That is a reason to expect a different outcome at 25M. It is not evidence of one.

**The early cost is only worth paying if it buys the collapse back.** The arm it replaces read +224.4,
+249.6, +253.8, +165.6 and then **−291.1** against its control at 5M through 25M. The decisive
comparison is at 25M. A slower anchor that is 86 Elo behind at 5M and still standing at 25M is a
win; one that is behind *and* collapses on the same schedule says the anchor was never the
mechanism.

(The old arm reads +229.4 here against the +224.4 recorded earlier in a four-model field — a
useful check that the two tournaments agree to within 5 Elo on a shared quantity.)

## Per seat against frozen anchors: nothing is degrading, and the base rate was misleading

The right instrument, applied to this arm. `E3-30-28` at 40M / 80M / 120M against the two frozen
anchors, temperature 0, 300 games a side
(`/workspace/data/tournaments/P15_X2_per_seat/`):

**Against `frozen_200M`:**

| steps | as USSR | as US | `critic_base_rate` at that step |
|---:|---:|---:|---:|
| 40M | 7.7% | 4.7% | 0.6124 |
| 80M | 30.0% | 14.3% | 0.6593 |
| 120M | **50.0%** | **23.0%** | **0.7903** |

**Against `frozen_280M`:** 6.3 / 4.0 → 25.7 / 12.3 → 37.7 / 18.3, the same shape.

**Both seats improve monotonically** — USSR by 42.3 pp and US by 18.3 pp against `frozen_200M`.
Nothing is degrading. The arm is simply climbing from a cold start toward anchors that already
have 200M and 280M steps behind them, and it is climbing faster as USSR than as US.

**And `critic_base_rate` rose from 0.61 to 0.79 across exactly that span.** Earlier entries in
this file read that rise as "heading toward one-sidedness" and treated it as a warning sign. It
was measuring the *difference in improvement rates between the two seats* — which is the specific
thing the statistic cannot distinguish from a side collapsing
([`method/measurement_pitfalls.md`](../method/measurement_pitfalls.md)). Here the per-seat numbers
say plainly that the answer is uneven improvement, and the base rate would have had this arm
written off.

So the question this arm was launched to answer — *does a slower anchor stop a side being given
away?* — cannot yet be answered from it, because **this arm has not given a side away**. At 120M
from scratch it is still far below both anchors, as expected, and both seats are rising. The
comparison that matters is whether it eventually turns the way `E3-20-28` did, and that needs the
full 240M with per-seat ratings at the end, not a base-rate trace.

## Budget: the search arm stops at 40M, not 240M

`E3-31-28` was launched with `--train-steps 240000000`, but the question it answers is narrow —
*does it collapse the way `E3-29-28` did between 20M and 25M?* — and 240M of search-rate training
is about 35 hours. It is stopped at **40M**, 15M past the point where the previous arm had already
lost 413 Elo.

The large budget costs nothing in comparability, which was worth checking rather than assuming:
`train_steps` drives only loop termination, the metadata, the progress label and a curriculum
switch this recipe does not use. There is **no learning-rate or coefficient schedule tied to it**
— no scheduler at all — so a 240M budget and a 40M budget train identically up to the point they
stop, and `E3-31-28` stays one factor from `E3-29-28` despite the latter having run with
`--train-steps 20000000`.

## The KL magnitude question X2 raised

X2 warns that a 25× slower anchor makes the KL term larger and says the response, if it dominates
the loss, is the η = 0.3 cell rather than a silent rescale. Measured:

| arm | KL | η·KL |
|:---|---:|---:|
| the 200k runs | 0.02–0.04 | 0.002–0.004 |
| `E3-30-28` ref 5M | 0.111 (max 0.238) | **0.011** |
| `E3-31-28` ref 5M + search | 0.151 | **0.015** |

KL rises 3–5× as expected, but η·KL lands at 0.011–0.015 against a policy loss of order 0.01–0.1
and an entropy bonus near 0.011. **The KL term is not dominating**, so the η = 0.3 escalation is
not indicated.

## Result: the slower anchor does NOT prevent side degradation

`E3-30-28` finished its 240M. Rated per seat against `frozen_200M`, temperature 0, 300 games a
side (`/workspace/data/tournaments/P15_X2_per_seat/`):

| steps | as USSR | as US | Elo in field |
|---:|---:|---:|---:|
| 40M | 7.7% | 4.7% | 987.4 |
| 80M | 30.0% | 14.3% | 1275.9 |
| 120M | 50.0% | **23.0%** | 1406.4 |
| 160M | 60.7% | 19.0% | 1399.7 |
| 200M | 60.0% | 18.7% | 1413.4 |
| 240M | **70.3%** | **11.3%** | 1409.9 |

**USSR improves monotonically from 7.7% to 70.3%. US peaks at 23.0% around 120M and then halves to
11.3%.** Against a *fixed* opponent, so this is absolute degradation on that seat — not the uneven
improvement that the earlier 40M–120M window showed, and the distinction is exactly the one the
per-seat instrument exists to make.

**Aggregate Elo hides it entirely**: 1406.4, 1399.7, 1413.4, 1409.9 from 120M onward — flat within
noise. The arm looks stable while one seat falls apart. Its `critic_auc` did fall to 0.706 with
Brier skill −0.157 by the end, so the critic instruments caught *something*; they simply cannot
say which seat.

### It is a USSR specialist, and that is two instruments agreeing

Mirror self-play, the same checkpoint on both seats, 300 games a side, T=0
(`/workspace/data/tournaments/P15_selfplay_sides/`):

| | `E3-30-28`@240M | `p28_200M` |
|:---|---:|---:|
| mirror self-play, as USSR | **96.3%** | 46.7% |
| mirror self-play, as US | **3.7%** | 53.3% |
| vs `frozen_280M`, as USSR | 51.0% | 53.7% |
| vs `frozen_280M`, as US | **13.7%** | 55.7% |

The two instruments agree, and they are independent: a mirror match says which seat wins when both
are this policy, and a common-opponent rating says how good each seat is in absolute terms. Here
both say the same thing — **its USSR play is genuinely competitive** (51.0% against a common
opponent where the 200M model manages 53.7%) **and its US play is broken** (13.7% against 55.7%).

So "USSR specialist, incompetent as US" is a fair description, and it is safe to say *because* the
common-opponent number backs the mirror. The mirror alone could not have established it: a 96/4
split is a fact about the pair and is equally consistent with a strong USSR and a broken US, or
with both seats mediocre and one slightly less so.

**`p28_200M` is not the mirror image of this.** Its own mirror is 53.3% US against 46.7% USSR — a
mild lean, near balanced — and against `frozen_280M` it is 53.7 / 55.7. It is a *balanced* model
that only looks US-specialised beside an opponent whose US seat has collapsed.

One caution on the earlier per-seat matrix: it appeared to show USSR winning most cross-pairings,
which reads like a population-wide side advantage. It is not. Four of its six entries were
`E3-30-28` snapshots with broken US play, so anyone playing USSR against them won easily. Between
the two frozen anchors there is no such skew (44.0 / 46.3 and 53.7 / 55.7).

### Search does not repair the broken seat

If a seat is broken because its *policy* is bad, search should recover much of it at test time.
If the *value function* is what is broken, search inherits the fault, because it evaluates its
leaves with that same critic. `E3-30-28`'s critic ended at AUC 0.706 with Brier skill −0.157, so
the second was the expectation. `P15_search_on_e330`, T=0, 100 games a side:

| vs `p28_200M` | as USSR | as US |
|:---|---:|---:|
| `E3-30-28`@240M raw | 70.0% | **9.0%** |
| `E3-30-28`@240M **+ search** | 77.0% | **13.0%** |

**Search adds 7 points on the healthy seat and 4 on the broken one, leaving US unplayable.**
Overall it is worth **+14.6 Elo** on this checkpoint (1455.7 against 1441.1) where the same
searcher is worth **+101.4** on `p28_200M` — about a seventh as much.

Head to head on identical weights, search against raw, the split is **98% / 3%** by seat: whoever
draws USSR wins almost regardless of whether they are searching. The seat dominates the search.

So a degraded seat is **not** recoverable at test time by search, at least not when the critic has
degraded with it. That is worth knowing before treating search as a safety net for a damaged
checkpoint.

(Cross-check: raw against `p28_200M` reads 70.0 / 9.0 here and 70.3 / 11.3 in the 300-game field,
so the fields agree within sampling error.)

### The US seat has no policy, rather than a bad one

Five self-play games at T=0, full policy traces
(`/workspace/data/replays/e3_30_240M_selfplay/`), 1,541 decisions:

| seat | mean p | median p | policy entropy | mean `v_win` | p < 0.5 | p > 0.9 |
|:---|---:|---:|---:|---:|---:|---:|
| USSR | 0.733 | 0.845 | 0.822 | **+0.761** | 24.6% | 43.9% |
| US | **0.572** | **0.582** | **1.299** | **−0.759** | **41.8%** | 28.0% |

Same network, same game, comparable branching (12.1 legal actions against 13.5) — but as US the
policy is diffuse: 58% more entropy, a median chosen probability of 0.58 against 0.85, and **42% of
US decisions taken with under half the mass on the chosen move**. As USSR it is decisive.

**It is not blundering.** The blunder audit reads 0/12 on `defcon_suicide_with_alternative`, 0/8 on
`olympic_games_at_defcon2` and 0/1 on `spaced_own_or_neutral` — zero in every tracked category. The
critic meanwhile reads `v_win` ≈ −0.76 from US's opening placement onward, so the model evaluates
the US position as near-lost from move one. Games end on turns 2, 4, 5, 6 and 10.

So the failure is not tactical: self-play let one seat stop being a real opponent, and the other
seat's policy **dissolved rather than degrading into specific mistakes**. That is the shape to look
for in future — an entropy and confidence split by seat, not a blunder-rate spike.

### What this answers

The arm was launched to ask whether a 25× slower reference anchor stops the decline on its own.
**It does not.** The degradation arrives around 120M and is on the **US** seat.

It is the same seat as `E3-26-28`'s late failure — US 47.0% → 19.3% against the same frozen
opponent ([`P15_control_per_seat.md`](P15_control_per_seat.md)) — and that arm had the **fast**
anchor and a different starting point. Two arms, two anchor settings, two starting points, the
same seat degrading. That points at something in the recipe or the game rather than at the
regularisation schedule.

### What it does not answer

* **One seed.** Seed variance is large enough here to matter, and a single lineage cannot separate
  a treatment effect from a draw.
* **It does not condemn X2 for the search arm.** The anchor was proposed as a fix for the *search*
  arm's collapse — a policy blown away from π_ref within a refresh window — which is a different
  failure from a seat slowly degrading over 120M. `E3-31-28` is still the test of that, and this
  result does not prejudge it.
* **Whether the fast anchor from scratch is better or worse at 240M** is not measured; `E3-20-28`
  ran to 320M but has not been rated per seat on this scale.

### Earlier readings in this file that this supersedes

The 40M–120M window was reported as "both seats improving, so the rising base rate was uneven
progress". That was true *for that window* and is still the right reading of it. The window
120M–240M is a different story, and only the per-seat rating distinguishes them — the base rate
rose across both.
