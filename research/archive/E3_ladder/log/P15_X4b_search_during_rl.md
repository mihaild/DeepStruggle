# P15-X4b — the search signal present *during* RL

**Launched 2026-09-17, `E3-29-28_20260917_041110` (after two arms collapsed on an implementation bug; see below).
Pre-registered below before any result
existed.** X4a established that the search edge is expressible as a policy (+47.7 Elo) but does
not survive subsequent RL — it decays to a dead heat within 20M steps
([`P15_X4a_distillation.md`](P15_X4a_distillation.md)). The plan's branch for that outcome is
this arm: if the signal cannot be installed once and kept, it has to be present continuously.

## The arm

An honest searcher answers a subsample of decisions during training and its visit distribution
enters the loss as a cross-entropy term, on searched decisions only. **Search produces targets and
never acts**, so the state distribution is the policy's own and the arm varies one thing.

## The control already exists, and that is the point

`E3-26-28_20260917_025539` — run earlier the same night as X4a step 4's control — is this exact
configuration with the search term switched off:

| | control `E3-26-28` | arm `E3-29-28` |
|:---|:---|:---|
| warmup checkpoint | `p28_200M` | `p28_200M` |
| seed | 20260928 | 20260928 |
| steps | 20M | 20M |
| envs / pool / reward / η / coefs | identical | identical |
| snapshot cadence | every 5M | every 5M |
| **search CE** | **off** | **coef 0.5, 64 sims, `all`, 1-in-8** |

One factor, matched seed, matched cadence. The intermediate snapshots line up, so a truncated arm
is still comparable at 5M, 10M and 15M rather than being lost.

## Two deliberate deviations from the written spec

Both are evidence-driven and recorded in the arm's own `metadata.json`.

**1. Node filter is `all`, not `card_playmode`.** The spec follows P3's argument that card and
play-mode decisions are "the decisions that matter". That is a claim about strategic consequence,
not about where a searcher disagrees with *this* policy, and
[`P15_X4a_where_the_search_signal_is.md`](P15_X4a_where_the_search_signal_is.md) measured the
difference: across all 85,113 decisions of 200 games, top-1 agreement is uninformative
(90.5–97.8% for every decision type) while KL varies 7×, and `POINT_NODE` carries **71.8% of the
CE signal** against card/play-mode's 20.9%. The spec's filter points at the fifth of the signal
X4a had already extracted.

**2. 64 simulations, not the flag default of 32.** 32 has never been measured to carry an edge
here. X0 rated a 64-sim searcher at +129.2 Elo and X4a used 96 offline as the saturation point. A
teacher weaker than any configuration shown to work is a good way to manufacture a false null.

## Budget: one seed, 20M steps — a partial leg

The plan specifies +80M at 2 seeds. Measured throughput with search on is **1,609 steps/s against
the control's ~11,500** — an 8.8× slowdown — which puts 80M at about 17 hours per seed. This arm
is **one seed at 20M steps** and must not be reported as the two-seed arm the plan specifies.

What the reduction costs is power, not validity: the comparison is against a step-matched control
rather than against the plan's absolute expectations.

## Pre-registered reading

Rate the arm’s final checkpoint against `E3-26-28`'s in one field, 500 games a side.

The control **lost 44.1 Elo** over its 20M steps, because this lineage declines past its 200M peak
regardless of what is done to it. So:

* **arm loses materially less than the control, or gains** → the search signal helps when present
  during RL, and X4b is worth the full two-seed leg.
* **difference < 25 Elo** (roughly 2 SE at 1,000 games a pair) → a **null at this budget**. That
  says the term needs more steps, more weight, or a different coefficient — *not* that continuous
  distillation does not work. A 20M-step arm cannot rule out an effect that needs 80M.
* **arm loses more than the control** → the CE term is actively harmful at this weight, which
  would be a real finding and points at the coefficient rather than the idea.

`--search-ce-coef` is a guess: the plan never specified a weight and none has been swept. The
first attempt at 0.5 collapsed the policy outright, which is recorded next.

## Two collapsed arms, and what actually caused them

Both were killed early. Against the step-matched control, same seed and same start:

| at ~196k steps | control `E3-26-28` | coef 0.5 | coef 0.05 |
|:---|---:|---:|---:|
| `critic_auc` | 0.905 | 0.570 | 0.830 |
| `entropy` | 1.106 | 0.318 | 0.362 |
| `kl_div` vs π_ref | 0.036 | 18.9 | 12.2 |

* `E3-27-28_..._VOID_ce_coef_collapse` — coef 0.5, killed at 1.44M steps
* `E3-28-28_..._VOID_search_target_offbyone` — coef 0.05, killed at 196k steps

### The wrong diagnosis, and what corrected it

The first collapse looked like dosage, and the arithmetic supported it: CE(search ‖ policy) is
about 1.0 nats, so at coef 0.5 the term contributes ≈ 0.5 to the policy loss where the entropy
bonus contributes ≈ 0.011 and `η × KL` ≈ 0.003. On that reading the CE term was not regularising
the policy toward the searcher, it *was* the objective. So the arm was relaunched ten times
smaller.

**It collapsed the same way.** Coef 0.05 started healthier — entropy 0.75 against 0.29 at 131k,
KL 2.0 against 10.7 — so the term does respond to its weight, but by 196k both arms had converged
on the same ruin. A tenfold reduction that only *delays* a collapse is not a dosage problem, and
that is what sent me to the call site rather than to a third coefficient.

### The bug

`collect_rollouts` took the search targets **after** `env.step`. The runner holds s_{t+1} by then
— the bootstrap comment a few lines above says so in as many words — while `buffer.add` files the
answer alongside `obs_t`. The CE term was training π(·|s_t) toward the searcher's answer at
**s_{t+1}**.

It fails in the worst available way. The targets do not become *illegal*, because consecutive
decisions share most of their legal set, so no mask catches them; they become legal and wrong.
Nothing raises, nothing warns, no tensor is malformed. The only symptom is on the training curve.

Fixed by moving the call above `env.step`, with three tests: two exercising `_search_targets`
against a real vectorised env, and one guarding the call-site ordering, which is where the bug
actually lived. The ordering guard was verified by reintroducing the bug and confirming that it,
and only it, fails.

### What this costs, and what it does not

Both collapsed arms are void and neither says anything about X4b's hypothesis — they measured an
implementation error. The coefficient question is **reopened**, not settled: the 0.5-is-too-large
arithmetic above was reasoning about a term that was pointed at the wrong state, and with aligned
targets the right weight has to be re-established rather than inherited from that analysis.

It also leaves one honest loose end. That the off-by-one *fully* explains the collapse is not
established — it is the cause of a real defect that was certainly harming learning, and the
relaunched arm's trace against the control is what confirms or refutes it.

## The arm that ran, and why at coef 0.5

With the targets aligned, a 400k-step smoke at **the original coef 0.5** — the weight the first
collapse was blamed on — tracks the control closely:

| step | entropy, arm / control | `critic_auc`, arm / control | KL, arm / control |
|---:|---:|---:|---:|
| 131k | 1.013 / 1.069 | 0.917 / 0.938 | 0.023 / 0.030 |
| 262k | 0.987 / 1.118 | 0.916 / 0.907 | 0.043 / 0.046 |
| 458k | 0.913 / 1.068 | 0.877 / 0.869 | 0.034 / 0.040 |

Compare the same coefficient before the fix: KL against π_ref was **3.16 at the very first
iteration** and 18.9 by 196k. It is now 0.023, inside the control's range.

Entropy sits about 15% below the control and drifts down slowly, which is what a CE term toward a
sharper teacher should do — and it is settling near **the searcher's own target entropy of 0.96
nats**, not falling toward zero. The critic tracks the control almost exactly.

So `E3-29-28` runs at coef 0.5: the original value, re-justified by measurement rather than
inherited from the analysis that turned out to be about a misaimed term.

## Result: +165.6 Elo over the step-matched control at 20M

Complete. `/workspace/data/tournaments/P15_X4b_verdict/`, 500 games a side, **temperature 0.0**:

| model | Elo | overall | vs control |
|:---|---:|---:|---:|
| **`x4b_arm` @20M** | **1666.9** | 71.4% | **+165.6** |
| `source_200M` — where both arms started | 1518.0 | 44.8% | |
| `control_no_search` @20M | 1501.3 | 41.7% | — |
| `anchor_280M` | 1500.0 | 41.6% | |

**Pre-registered branch:** *"arm loses materially less than the control, or gains → the search
signal helps when present during RL, and X4b is worth the full two-seed leg."* The bar was 25 Elo.
The arm cleared it by 6.6×, and finished **149 Elo above the checkpoint both runs resumed from**
while the control ended level with the anchor.

**X4b works.** With the caveat in the next section, which is not a small one.

### The trajectory, which is the part that matters

X4a's lesson was that a gain measured at one checkpoint can be gone 20M steps later, so the
matched pairs are rated as they appear. All at temperature 0.0, same four-model field, same anchor:

| matched steps | arm | control | **gap** | arm vs its own start |
|---:|---:|---:|---:|---:|
| 5M | 1752.0 | 1527.6 | **+224.4** | +214.2 |
| 10M | 1763.5 | 1513.9 | **+249.6** | +239.2 |
| 15M | 1749.8 | 1496.0 | **+253.8** | +223.2 |
| **20M** | **1666.9** | 1501.3 | **+165.6** | +148.9 |

The control slides steadily — 1527.6, 1513.9, 1496.0, 1501.3 — the post-peak decline X0 measured,
ending level with the 280M anchor. The arm holds near 1750 through 15M.

**Then the gap falls by 88 Elo in the final 5M.** It is reported here rather than folded into the
headline, because the headline number alone would say the arm simply won and the shape says
something more specific: the advantage peaks around 10–15M and is eroding by 20M. Whether it
levels off well above the control or continues toward X4a's dead heat is **not answered by a
20M arm**, and it is the single most important thing the 80M leg would settle.

The drop is far too large to be tournament noise — the earlier rows are separated by 25 Elo or
less while this is 88 — but it is one seed and one 5M segment, so it is a signal to chase, not an
established decay curve.

(Separate tournaments, so the scales are not identical; the field and the anchor are the same in
both, which is what makes the two rows comparable at all. The gap within a row is the number to
read, not the change in either column across rows.)

### The confound that had to be ruled out first

The arm's policy entropy is 0.56 against the control's 1.17, and the tournament's default is to
*sample* at temperature 0.1. A sharper policy sampled at a fixed temperature plays closer to its
own argmax, so it can win on sharpness rather than on strength — which would have made the whole
result an artefact of the treatment's side effect.

Re-rated at **temperature 0.0**, where every agent plays its argmax and entropy cannot matter by
construction, the gap is +224.4 Elo against +239.6 when sampling. The confound is worth about 6%
of the effect; the rest is policy strength.

`tools/tournament.py` gained a `--temperature` flag for this, and the setting is now recorded in
the report header and the JSON, because a rating is not interpretable without it.

### What is and is not established

**Established:** over 20M matched steps, with one seed, a searcher supplying CE targets during RL
produces a policy 165 Elo stronger than the identical run without it, measured argmax-on-argmax,
with the advantage present at every 5M checkpoint along the way.

**Not established:**

* **Where it settles.** The gap peaked at +253.8 and fell to +165.6 in the last 5M. X4a's +47.7
  also looked solid at its own checkpoint and was a dead heat 20M steps later. This arm is not
  that — it ends 165 Elo up, not level — but the final segment points the same direction, and only
  a longer leg distinguishes "settles high" from "slower washout".
* **One seed.** No variance estimate.
* **That it beats the searcher it learned from.** X0 rated search over all nodes at +129.2 Elo,
  which is a smaller number than +224, but in a different field against different opponents. Elo
  does not travel between tournaments and the two must not be subtracted.
* **Why entropy keeps falling.** It is at 0.56 and still drifting down, through the searcher's own
  target entropy of 0.96 rather than settling there. The benign reading is that the teacher is
  bimodal — mean 0.96 nats but 29.9% of targets carry >0.9 mass — so pulling toward the sharp ones
  sharpens the policy overall. The critic is mildly degraded too (`critic_auc` 0.835 against the
  control's 0.898). Neither is collapse, both are unexplained, and the 20M trace is what shows
  whether they stabilise.

## It improves the side RL was giving away

From the same final tournament:

| model | as USSR | as US | USSR − US |
|:---|---:|---:|---:|
| **`x4b_arm`** | **61.9%** | 81.0% | −19.1 pp |
| `source_200M` | 43.1% | 46.5% | −3.4 pp |
| `control_no_search` | 34.7% | 48.7% | −13.9 pp |
| `anchor_280M` | 29.9% | 53.3% | −23.4 pp |

X4a reported that 20M steps of this recipe gives away the USSR side, within-arm and in two
independent runs ([`P15_X4a_distillation.md`](P15_X4a_distillation.md)). The control here does it
again: USSR 34.7% against the source's 43.1%.

The X4b arm plays USSR at **61.9%** — 27 points above its own control and 19 above the checkpoint
they both started from. It improves *both* seats, and the larger US gain is why its USSR − US gap
still reads −19.1 pp. **The side gap is the wrong instrument here**: a model winning 61.9% and
81.0% is not "imbalanced" in any sense that matters, which is the argument
[`method/measurement_pitfalls.md`](../method/measurement_pitfalls.md) already makes about that
metric.

So the answer to X4a's USSR finding may be that the weak side was never a balance problem but a
search-depth one.

## What to run next

1. **The 80M two-seed leg the plan specifies.** The one question this arm raises and cannot answer
   is where the gap settles after its 15M peak. That is what the full leg is for, and it now has a
   measured throughput (1,609 steps/s) and a matched control recipe to reuse.
2. **A coefficient sweep.** 0.5 was never swept; it was a guess that survived a wrong diagnosis.
3. **`card_playmode` as the contrast.** This arm used `all` on the evidence that `POINT_NODE`
   carries 71.8% of the CE signal. Running the spec's filter would test that reasoning directly
   rather than leaving it inferred.

## How much search still adds along the way

[`P15_X4b_search_headroom.md`](P15_X4b_search_headroom.md) runs the searcher on top of each 5M
snapshot against that same snapshot raw. Across 5M-20M the margin is **flat at ~58%** (+50 to +77
Elo) with KL ~0.036 and agreement ~97%. The policy is not progressively catching its teacher —
which is what expert iteration looks like when the teacher is the student plus search and improves
with it. It also means the process has not saturated at 20M.

## The 80M extension collapsed the arm, and the +165.6 Elo was measured just before the cliff

**Stopped at 26.3M of 80M.** The arm did not degenerate the way the control did — it fell apart.

Rated at temperature 0, 300 games a side, `/workspace/data/tournaments/P15_X4b_25M/`:

| model | Elo | overall |
|:---|---:|---:|
| **arm @20M** (end of the original run) | **1667.0** | 76.9% |
| control @25M | 1544.8 | 57.8% |
| `anchor_280M` | 1500.0 | 50.5% |
| **arm @25M** | **1253.7** | **14.6%** |

**413 Elo lost in 5M steps**, ending 291 Elo *below* its own step-matched control. The instruments
agree: `critic_auc` fell 0.87 → 0.59, Brier skill to 0.010 — chance — and `kl_div` against π_ref
spiked to 14.9, 5.8 and 31.99 in consecutive iterations. `critic_base_rate` stayed near 0.51, so the failure does not even have the
shape of the control's, whatever that turns out to be; the policy is simply being blown apart.
(Used here only to say the two failures differ, which is a comparison the statistic can support —
not to characterise either one.)

### Per seat against the 200M start, the decline begins on the USSR side

Extracted from the per-side matrices of the same verdict tournaments, so no new games. Win rate
against `p28_200M`, the checkpoint both arms resumed from, at temperature 0:

| steps | arm as USSR | arm as US | control as USSR | control as US |
|---:|---:|---:|---:|---:|
| 5M | 80.2% | 76.4% | 42.2% | 52.4% |
| 10M | **83.0%** | 77.6% | 40.4% | 59.2% |
| 15M | 79.2% | 77.8% | 39.4% | 54.2% |
| 20M | **67.2%** | 76.8% | 50.4% | 46.4% |

**The search arm beats its own starting checkpoint on both seats throughout** — 67–83% — while the
no-search control never does, at 39–50% as USSR and 46–59% as US. The +165.6 Elo is therefore not
an artefact of a weak control: the arm improves on what it started from and the control
essentially does not.

**And the collapse has a per-seat precursor.** The arm's USSR rate falls 83.0 → 79.2 → 67.2 between
10M and 20M while its US rate is flat at 77.6 → 77.8 → 76.8. That is visible at 15M and pronounced
at 20M, **before** the Elo cliff at 25M — so a per-seat rating against a frozen opponent would have
flagged this arm one checkpoint earlier than the aggregate Elo did, which is a practical argument
for making it routine rather than occasional.

Note the seat: it is USSR here, the opposite of the control's own late failure, which is on the US
seat ([`P15_control_per_seat.md`](P15_control_per_seat.md)). Whatever drives the two failures, it
is not one shared side-specific weakness.

The 25M field did not include `p28_200M`, so the split at the collapse point itself is not
available without new games.

### It was not the resume, and it was not sudden

The first suspicion was the continuation itself, because KL was 10–20× the original run's from its
first iteration (0.12–0.87 against 0.02–0.04), where the control's continuation rose only ~2×. But
the original arm run **ended** at KL 0.08–0.40, having started at 0.02. The elevation was
inherited, not introduced: the continuation resumed an already-rising trend.

The reference policy *is* saved and restored (`generic_trainer.py:1147`, `:1176`), and the searcher
holds `active_net` by reference while resume loads weights in place, so the teacher is not stale
either. Both hypotheses were checked and neither holds.

**So the mechanism is the arm's own: search CE at coef 0.5 drives the policy steadily away from
π_ref, and past roughly 20M steps that divergence goes critical.** The η·KL term at η = 0.1 cannot
hold it: at KL 30 that term contributes 3.0 to the loss, which is large, but by then the policy has
already left.

### What this does to the result

**The +165.6 Elo at 20M stands as measured** — it was rated against a step-matched control, at
temperature 0, and the checkpoint is on disk and still rates 1667.0 in a fresh field. But it is now
known to be **a reading taken shortly before a cliff**, which changes what it licenses. The gap by
matched step reads:

| matched steps | gap vs control |
|---:|---:|
| 5M | +224.4 |
| 10M | +249.6 |
| 15M | +253.8 |
| 20M | +165.6 |
| **25M** | **−291.1** |

The decline from +253.8 to +165.6 in the last segment of the first run — flagged at the time as
"where it settles is open" and "the direction of that last segment is the same" as X4a's — was
**the leading edge of this collapse, not a fluctuation.** That caution is now cashed in.

### Measured: the CE term is more than half the update

`search_ce_grad_frac` — the CE term's share of the raw gradient norm, measured with a separate
backward on that term alone — was added after the collapse, because nothing had measured the CE
term at all. Its **first logged value on a fresh arm at coef 0.5 is 0.561**, with
`search_ce` = 0.904.

**At coef 0.5 the CE term is 56% of the gradient from the first iteration.** Gradients are
globally norm-clipped at 1.0, so this does not mean a larger step — it means a step whose
direction is more than half CE, with the PPO advantage and the value loss splitting what is left.
That is the concrete form of "the critic's signal was right and was ignored": nothing about the
advantage estimate is wrong, it is simply outvoted in every update.

It also gives the collapse a monitorable precursor. If the share climbs toward 1.0 as the policy
sharpens — and sharpening is what makes a soft target expensive, since `-log π` grows without
bound as the student's probability on a teacher-favoured action goes to zero — the update becomes
pure CE. A tripwire on `search_ce_grad_frac > 0.75` now runs alongside the arm, so a future
collapse can be stopped while a healthy checkpoint still exists rather than 5M steps later.

**Caveat:** 0.561 is one iteration of one arm at one coefficient. Whether the share is what rises
before a collapse is a hypothesis this instrumentation was built to test, not something it has
shown yet.

### Correction: the CE gradient does not explode, and an earlier explanation here was wrong

Running `E3-31-28` with the instrumentation on shows `search_ce` alternating between ~0.75 and
spikes of 5.8e3, 1.7e4, 2.0e5 — while `search_ce_grad_frac` sits flat at 0.48–0.53 and entropy,
KL and `critic_auc` barely move across the same iterations.

That rules out the mechanism proposed above. For softmax cross-entropy the gradient with respect
to the logits is `π − p_target`, **bounded in [−1, 1] however large the loss becomes**. A CE of
2e5 therefore produces an ordinary update, which is precisely what the steady gradient share
shows. The earlier reasoning — that `−log π` grows without bound as the student sharpens, and that
this is what destabilises the arm — was right about the loss and wrong about the gradient, which
is the only part that moves weights.

What the spikes *do* diagnose is real: the policy has driven some logits about 2e5 below the
maximum, so it assigns effectively zero probability to actions the searcher still visits. Entropy
cannot see this — those actions contribute nothing to it — which is why entropy sat at a
comfortable 0.79 throughout.

Two consequences. **The mean `search_ce` logged here is outlier-dominated and a poor summary**; a
median, or the max alongside it, would be the better statistic and is what a future arm should
record. And the candidate mechanism for the collapse narrows to the **sustained ~50% share of the
update direction** at coef 0.5 — a large, persistent pull toward the teacher — rather than to any
spike.

### What is worth running next

1. **A coefficient sweep is no longer optional.** 0.5 was a guess that survived a wrong diagnosis
   and a 458k-step smoke test; it is now known to be unstable over 20M. The obvious arm is a
   resume from the healthy **20M checkpoint** at a much lower coefficient — 0.05 or 0.1 — which
   keeps the +165.6 Elo starting point and tests directly whether the instability is dosage.
2. **Or a KL guard.** The divergence is visible for millions of steps before it goes critical, so
   an arm that anneals `search_ce_coef` when KL exceeds a threshold would be a cheap fix and is
   measurable against the same control.
3. The control's own 80M snapshots are intact and it degenerated on a different schedule and in a
   different way (side collapse at ~65M, i.e. ~265M cumulative), so the pair still supports the
   60M comparison for anything that reaches it.

## Extension to 80M, the standard leg — stopped

Launched 2026-09-17: both runs resumed from their 20M resume states, which sat at exactly
**20,054,016 steps each**, and continue to **80M cumulative**.

* arm — `E3-29-28_20260917_074357`, search CE unchanged at coef 0.5, 64 sims, `all`, 1-in-8
* control — `E3-26-28_20260917_074533`, the identical recipe with search **off**

Same seed as the originals, so each continues as if uninterrupted rather than diverging.

**The control is continued too, and that is not optional.** X0 measured this lineage declining
past its 200M peak, so an 80M rating with nothing to compare against cannot separate the treatment
from the decline — it would be the same mistake the X4a step-4 control caught. Search-free
training runs about 7× faster, so the control costs roughly 1.5h against the arm's ~9h; the pair
is worth far more than the arm alone.

Snapshots every 5M on both, so the gap is measured along the way rather than inferred from the
endpoints.

### The control degenerates at ~65M, which changes how the 80M pair must be read

Noticed while the continuations were running, and recorded before the result exists so it cannot
be reached for afterwards.

| control steps | `critic_base_rate` | `critic_auc` | Brier skill | entropy |
|---:|---:|---:|---:|---:|
| 20M | 0.65 | 0.92 | 0.44 | 1.20 |
| 51M | 0.52 | 0.90 | 0.46 | 1.09 |
| 60M | 0.61 | 0.93 | 0.56 | 1.25 |
| **65M** | **0.81** | 0.93 | 0.56 | 1.34 |
| 70M | 0.93 | 0.73 | 0.18 | 1.48 |
| 74M | 0.96 | 0.69 | **−0.08** | 1.43 |

`critic_base_rate` first crosses 0.80 at **65,273,856 steps** and reaches 0.96 by 74M: roughly
**96% of self-play games are won by one side**.

> **Correction: `critic_base_rate` cannot show this, and the wording above was wrong twice over.**
> It is `max(p, 1 − p)` (`critic_tracker.py:140`), the majority-class rate — so it has **no
> direction** and cannot name a side at all, a point
> [`seed_variance_and_pooling.md`](seed_variance_and_pooling.md) already records against earlier
> reports that read it as "US-leaning". And as a magnitude it is a fact about the *pair*: 0.96 is
> equally consistent with one side collapsing and with both sides improving at different rates.
> See [`method/measurement_pitfalls.md`](../method/measurement_pitfalls.md).
>
> The conclusion survives, but on a different instrument. The per-side matrix in
> [`P15_X4a_distillation.md`](P15_X4a_distillation.md), rated **against a frozen field**, puts this
> control at **USSR 34.7% against US 48.7%** where its own starting checkpoint was 43.1 / 46.5 —
> a per-seat measurement against fixed opponents, which is the instrument for the question. The
> `critic_base_rate` trace below is the cheap hint that prompted looking; it is not the evidence,
> and the 80M checkpoint has not yet been rated per seat.

**Those step counts are run-local.** `--warmup-checkpoint` loads weights but resets the counter,
so step 0 of this control is `p28_200M`, which is itself `snapshot_200015872steps.pt`. The onset
is therefore **65.3M steps after the source, or ~265.3M cumulative** for the weights. Read it as
the weights' total training exposure rather than as a continuation of `E3-20-28`'s exact
trajectory: this is a fresh run from those weights, with the opponent pool, reference policy and
optimiser moments all restarted. The critic falls with it, to AUC 0.69 and negative
Brier skill — it has become a base-rate predictor, which is the collapse mode this project has
seen before and which `checkpoints.md` records for `E3-21-28`. Entropy *rises* to ~1.43 while this
happens, so whatever this is, it is not an entropy collapse.

It looked like the X4a finding run to completion. **It is not, and the side was named wrongly.**
Rated per seat against frozen anchors
([`P15_control_per_seat.md`](P15_control_per_seat.md)), this control's **US** win rate collapses —
47.0% at 60M to 19.3% at 80M against `frozen_200M` — while its USSR play stays flat, 57.0% to
54.0%. The X4a figures of −17.3 pp and −11.6 pp were side balance *averaged across a tournament
field*, and that is not a property of a model: this same checkpoint reads −13.9 pp in X4a's field
and +2.8 pp in the per-seat field. Both instruments relied on earlier — a directionless base rate
and a field-averaged split — were wrong for the question, and happened to agree.

It also extends X0's curve. [`P15_X0_frozen_anchors.md`](P15_X0_frozen_anchors.md) measured this
lineage declining past its 200M peak — 2193.7 at 200M against 2168.0 at 240M — and this says where
that decline ends up: by roughly 265M cumulative the weak side is gone entirely.

**So a straight arm-against-control rating at 80M risks measuring a control that has gone
somewhere strange rather than the arm's strength** — provisionally, since what exactly has gone
wrong with it needs a per-seat rating against the frozen anchors to establish. Beating a degenerate opponent is not the claim X4b is making. The pair
is rated at **60M as well as 80M** — both snapshot every 5M — and 60M is the honest comparison,
being the last matched point where the control's instruments are still healthy.

The arm is at 24M as this is written, with `base_rate` drifting up from 0.60 to about 0.78. **No
claim is made that search CE prevents this**; it is behind the control on the same axis and has
not yet reached the steps where the control turned. Whether it degenerates at 65M too is a real
question the 80M run answers, and it may turn out to be the more interesting output of this arm
than the Elo number.

### Pre-registered reading of the 80M pair

Rated at temperature 0, for the entropy reason above. The 20M leg gave +224.4, +249.6, +253.8,
+165.6 — a peak near 15M and 88 Elo shed in the final segment. So:

* **gap ≥ 100 Elo** → the advantage is durable at the standard budget, and the late-20M dip was a
  fluctuation rather than the start of a decay. **Read at 60M, not 80M**, unless the arm turns out
  to have degenerated on the same schedule as the control, in which case the 80M pair compares two
  collapsed policies and says nothing.
* **gap 25–100** → real but eroding; the early peak was the best of it, and the mechanism buys a
  transient rather than a new level.
* **gap < 25** → it washed out the way X4a's offline distillation did, and the 20M result was a
  transient measured at a flattering moment.

The third outcome is a live possibility, not a formality: this project has already been wrong
once in exactly that shape tonight.

## Result

**+165.6 Elo over the step-matched control at 20M, one seed** — the mechanism works. The 80M pair
is running and decides whether the advantage is a new level or a transient.
