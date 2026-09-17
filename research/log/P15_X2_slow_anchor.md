# P15-X2 — does a slower reference anchor prevent the collapse on its own?

**Launched 2026-09-17, `E3-30-28_20260917_104226`. Interim; the arm is running to 240M.**

P15's own diagnosis of `ref_update_freq = 200000`: the anchor is refreshed ~400× per 80M leg, so
it *tracks* the cycle rather than anchoring it, and NashPG's convergence story assumes the inner
loop approaches the regularized fixed point **before** the reference moves. X2 proposes 5M and
20M. This is the 5M cell, with no search, from scratch.

Two collapses on 2026-09-17 motivated running it now rather than in plan order: the X4b search arm
lost 413 Elo in 5M steps with KL against π_ref spiking to 31.99
([`P15_X4b_search_during_rl.md`](P15_X4b_search_during_rl.md)), and its no-search control
degenerated into one-sidedness at ~65M further steps, `critic_base_rate` reaching 0.96.

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

## Reading, with the caveat first

**One seed each, and the seeds differ.** This project has a findings entry on seed variance for a
reason, and a single pair cannot separate a treatment effect from a seed draw. Everything below is
a reason to finish the arm, not a result.

With that said, two things are visible:

* **The slow anchor has not crossed `base_rate` 0.80 by 72M**, where the fast anchor crossed at
  44.6M. Note that the fast anchor's crossing was a **transient spike** — it peaked at 0.8204 and
  was back to 0.5433 by 70M — so this is not yet the sustained 0.96 degeneration seen in the
  resumed control. What it shows is a lower ceiling, not a prevented collapse.
* **The slow anchor holds much more entropy early**: 1.32 against 0.74 at 10M, converging by 70M.
  That is the expected shape — a reference that lags pulls less hard toward whatever the policy
  has just become — and it costs early critic accuracy, AUC 0.64 against 0.73 at 10M, which is
  recovered by 60M.

So the slower anchor buys a slower, flatter start and a lower one-sidedness ceiling so far. Whether
it prevents the decline that `E3-20-28` shows past its 200M peak is a question only the full 240M
answers, and a second seed would be needed before believing any of it.

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

## Result

**Running.** Nothing claimed until 240M, and not from one seed.
