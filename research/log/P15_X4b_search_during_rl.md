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

## The arm that is running, and why at coef 0.5

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

## Interim result: +224 Elo at 5M, +250 at 10M, over the step-matched control

The arm is still running to 20M. Both runs snapshot every 5M, so the first matched pair can be
rated without waiting, and it is emphatic. `/workspace/data/tournaments/P15_X4b_verdict/`,
500 games a side, **temperature 0.0**:

| model | Elo | overall | vs control |
|:---|---:|---:|---:|
| **`x4b_arm` @5M** | **1752.0** | 78.8% | **+224.4** |
| `source_200M` — where both arms started | 1537.8 | 43.0% | |
| `control_no_search` @5M | 1527.6 | 41.1% | — |
| `anchor_280M` | 1500.0 | 36.4% | |

The control drifted slightly *below* the checkpoint it resumed from, which is the decline X0
measured past this lineage's peak. The arm went **+214 Elo above that same starting point in 5M
steps**.

The pre-registered bar was 25 Elo. This is nine times it.

### The trajectory, which is the part that matters

X4a's lesson was that a gain measured at one checkpoint can be gone 20M steps later, so the
matched pairs are rated as they appear. All at temperature 0.0, same four-model field, same anchor:

| matched steps | arm | control | **gap** | arm vs its own start |
|---:|---:|---:|---:|---:|
| 5M | 1752.0 | 1527.6 | **+224.4** | +214.2 |
| 10M | 1763.5 | 1513.9 | **+249.6** | +239.2 |

The control slides — 1527.6 to 1513.9 — which is the post-peak decline X0 measured. The arm holds
and edges up. **The gap is growing, not decaying**, which is the opposite of what X4a did.

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

**Established:** at 5M matched steps, with one seed, a searcher supplying CE targets during RL
produces a policy 224 Elo stronger than the identical run without it, measured argmax-on-argmax.

**Not established:**

* **That it holds to 20M.** X4a's +47.7 Elo looked solid at its own checkpoint and had decayed to
  a dead heat 20M steps later. The same could happen here, and this project has now been wrong
  once tonight in exactly that way. The arm runs to 20M for that reason.
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

## Result

**Running to 20M.** The 5M pair above is interim and is not the verdict.
