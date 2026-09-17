# P15-X4b — the search signal present *during* RL

**Launched 2026-09-17, `E3-27-28_20260917_034351`. Pre-registered below before any result
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

| | control `E3-26-28` | arm `E3-27-28` |
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

Rate `E3-27-28`'s final checkpoint against `E3-26-28`'s in one field, 500 games a side.

The control **lost 44.1 Elo** over its 20M steps, because this lineage declines past its 200M peak
regardless of what is done to it. So:

* **arm loses materially less than the control, or gains** → the search signal helps when present
  during RL, and X4b is worth the full two-seed leg.
* **difference < 25 Elo** (roughly 2 SE at 1,000 games a pair) → a **null at this budget**. That
  says the term needs more steps, more weight, or a different coefficient — *not* that continuous
  distillation does not work. A 20M-step arm cannot rule out an effect that needs 80M.
* **arm loses more than the control** → the CE term is actively harmful at this weight, which
  would be a real finding and points at the coefficient rather than the idea.

`--search-ce-coef 0.5` is a guess. The plan never specified a weight and none has been swept.

## Result

**Running.** Nothing claimed until the tournament lands.
