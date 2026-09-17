# P15-X4b — the search signal present *during* RL

**Launched 2026-09-17, `E3-28-28_20260917_035813` (after a first attempt collapsed; see below).
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

| | control `E3-26-28` | arm `E3-28-28` |
|:---|:---|:---|
| warmup checkpoint | `p28_200M` | `p28_200M` |
| seed | 20260928 | 20260928 |
| steps | 20M | 20M |
| envs / pool / reward / η / coefs | identical | identical |
| snapshot cadence | every 5M | every 5M |
| **search CE** | **off** | **coef 0.05, 64 sims, `all`, 1-in-8** |

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

## First attempt: `--search-ce-coef 0.5` destroys the policy in its first few updates

`E3-27-28_20260917_034351_VOID_ce_coef_collapse`, killed at 1.44M of 20M steps. Against the
step-matched control, same seed and same start:

| at 1.44M steps | control `E3-26-28` | arm at coef 0.5 |
|:---|---:|---:|
| `critic_auc` | 0.874 | **0.505** — chance |
| `entropy` | 1.105 | **0.169** |
| `kl_div` vs π_ref | 0.029 | 0.299, having peaked at **18.9** |
| `value_loss` | 0.027 | 0.092 |

The damage is done immediately, not gradually. KL against the reference policy spiked to 10.7,
18.9 and 14.5 over the first three logged iterations while entropy fell from 0.878 to 0.286 in
one. `critic_auc` never reached even 0.58 and decayed monotonically to chance from there. The
control, from the same weights and seed, sits at 0.87 with entropy 1.10 and KL 0.03 across the
identical span and stays there.

**The weight is the cause, and the arithmetic is not subtle.** CE(search ‖ policy) measures about
**1.0 nats** ([`P15_X4a_where_the_search_signal_is.md`](P15_X4a_where_the_search_signal_is.md)),
so at coef 0.5 the term contributes ≈ 0.5 to the policy loss. The entropy bonus contributes
`0.01 × 1.1 ≈ 0.011` and the KL regulariser `η × KL = 0.1 × 0.03 ≈ 0.003`. The CE term outweighs
both by more than an order of magnitude, so it is not regularising the policy toward the searcher
— it *is* the objective, and it drives the policy to near-determinism. The critic follows because
the trunk is shared.

This is the third branch of the pre-registered reading above — *"points at the coefficient rather
than the idea"* — reached before spending the arm rather than after.

### What it says about the mechanism

Nothing bad, and one useful thing. A CE term strong enough to flatten entropy 6.5× in a single
iteration is a term with plenty of gradient to give; the problem is dosage. It also means **X4b is
sensitive to a coefficient the plan never specified and nobody has swept**, which is worth knowing
before reading any X4b result as a verdict on continuous distillation.

## Second attempt: coef 0.05

`E3-28-28_20260917_035813`, ten times smaller, chosen to put the CE contribution in the same
regime as the other regularisers rather than an order of magnitude above them. Everything else
unchanged, and still matched to `E3-26-28`.

The control's healthy trace at matched steps is the live check: **if entropy falls far below ~1.05
within the first 500k steps, this weight is also too large** and the arm gets killed rather than
run to completion.

## Result

**Running.** Nothing claimed until the tournament lands.
