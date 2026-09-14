# Seed variance, and what it does to the pooling result

*2026-09-14.*

## Summary

The opponent-pool result does not survive a second seed. Two no-pool arms differing only in seed
are **further apart than pooled is from no-pool**, so on current evidence pooling has no detectable
effect on side imbalance. Separately, per-entity attention heads look associated with roughly
seven times the seed spread, which — if it holds — matters more than any single ablation, because
it is a multiplier on the cost of every future experiment.

## The endpoint

Runs oscillate. E3-17-22 measured 1.2 pp self-play imbalance at 80M and 79.7 pp at 160M — the same
run. A final-checkpoint number therefore samples *where in a swing the budget ran out*, which is
variance on top of seed variance.

So the endpoint is **mean |ussr_win_rate − 0.5| over the final 40M**, read from the per-iteration
metric that is already logged. 0 is balanced, 0.5 is one side winning everything. It is not immune
— a run drifting monotonically at the end still partly reports where the drift got to — and if
results stay ambiguous the more robust form is *fraction of training spent within X of balance*,
which is not anchored to the end at all.

## Pooling: no detectable effect

| arm | condition | endpoint | final \|gap\| | USSR @ end |
|:---|:---|---:|---:|---:|
| E3-17-22 | no pool | 0.3458 | 0.4136 | 0.914 |
| E3-17-24 | no pool | 0.2275 | 0.2703 | 0.770 |
| E3-20-22 | pooled | 0.2698 | 0.1692 | 0.669 |

Same-condition spread (no pool, two seeds): **0.1183**.
Pooled minus mean of no-pool: **−0.0169**, seven times smaller.

The pooled arm sits *between* the two no-pool arms — what a null effect looks like. The trajectory
shows why a single pair was never safe: all three stay near balance to ~80M and then diverge
unpredictably, E3-17-24 even crossing to US-favoured (0.377) at 100M before swinging back.

| steps (M) | E3-17-22 | E3-17-24 | E3-20-22 |
|---:|---:|---:|---:|
| 60 | 0.500 | 0.518 | 0.768 |
| 80 | 0.512 | 0.527 | 0.765 |
| 100 | 0.661 | **0.377** | 0.766 |
| 140 | 0.857 | 0.780 | 0.738 |
| 160 | **0.915** | 0.634 | 0.713 |

This retroactively weakens several earlier claims, all of which rested on single arms of an
oscillating quantity: that pooling from a balanced checkpoint "preserves" balance, that pooling
after a lock "repairs nothing", and the Elo ordering across those arms. A 4 × 4 replication is
running to settle it; the honest prediction, recorded before the arms land, is that it comes back
null.

## Per-entity heads and seed spread

Every same-configuration seed set we hold, restricted to the 160M budget — runs mostly diverge
after 80–100M, so an 80M arm is not comparable:

| heads | config | n | spread |
|---:|:---|---:|---:|
| 64 | E3-17 (no pool) | 2 | 0.118 |
| 64 | E3-19 (pooled) | 2 | 0.256 |
| none | p1_identity | 2 | 0.042 |
| none | p1_scalar_filter | 2 | 0.008 |

Consistent direction across two independent groups per level, ~7× wider with heads. **It is not
yet a finding**, for three reasons:

1. **n=2 within each group.** A spread from two samples has a very wide confidence interval, and
   comparing two such spreads is weaker still. This is the trap that already forced the
   graph-depth withdrawal.
2. **The groups differ in more than heads.** `p1_*` is a separate lineage (categorical value and
   advantage filtering), not a heads ablation.
3. **A mean–variance confound.** `p1` endpoints sit at 0.08–0.19 and E3-17 at 0.23–0.35. |gap| is
   bounded below by zero, so a run that stays balanced *cannot* vary much while a diverging one
   can. Lower variance may be a consequence of staying balanced rather than a cause of it.

It is worth testing deliberately because of what it costs if true: seven times the spread is
roughly forty-nine times the arms for equal statistical power, which would make ablation at
`heads=64` effectively unaffordable.

The cheap first question is not "how much variance" but **"does `heads=64` buy strength at all?"**
Existing final checkpoints from both levels can be played against each other in one tournament,
with no new training. If it buys nothing, the variance question is moot and the rollback is free.

## Method notes carried out of this

* **`critic_base_rate` has no direction.** It is `max(p, 1−p)` (`critic_tracker.py:140`), the
  majority-class rate, so it is always ≥ 0.5 and says nothing about *which* side leads. Earlier
  reports that read it as "US-leaning" were wrong. As a magnitude it is fine.
* **Pairing seeds does not work here, at training or evaluation.** Trajectories diverge at the
  first differing action; and in this game even a fixed shuffle does not fix the *deal*, because
  one extra discard or a China card play changes who receives what from then on. Fixing the seed
  buys the opening deal and nothing after. For evaluation variance the levers are more games and
  common opponents.
* **Report distributions, not points.** Where arms allow it, IQM with stratified bootstrap
  intervals (Agarwal et al. 2021) rather than means, so a single runaway arm does not dominate.


---

## Per-entity heads: measured, and the variance claim narrowed

A tournament of existing finals — 3 `heads=64` arms against 5 `heads=None` arms, 250 games per
side, 22,500 games, no new training. A screen rather than an ablation: the `p1_*` and `arm_*`
lineages differ from E3-17 in more than the heads.

`heads=64` took the top three Elo places (1978, 1967, 1945) against 1940, 1936, 1865, 1845, 1836
— group means 1963 vs 1884. But the groups overlap: the best `heads=None` arm is 5 Elo below the
worst `heads=64` arm, which is nothing. The margin rests on the weaker `p1_scalar`/`armH2` arms.

**Restricting to cross-group games and reading each side separately:**

| | heads=64 | heads=None | diff |
|:---|---:|---:|---:|
| as USSR (vs the other group's US) | 83.4% | 62.1% | **+21.3 pp** |
| as US (vs the other group's USSR) | 37.9% | 16.6% | **+21.3 pp** |

Attention lifts **both sides by the same 21.3 pp**, to one decimal across 3,750 games per
direction. It is not a USSR-specialist architecture. `heads=64` playing US still loses to
`heads=None` playing USSR (37.9%), but that is the game's side advantage at this strength — it
applies to HeuristicBot (+9.9 pp) and even RandomBot (+0.6 pp) — not a defect of attention.

**So the earlier "heads=64 has ~7x the seed variance" needs narrowing.** Seed spread *on Elo*
within genuinely same-config pairs is comparable across levels: E3-17 22 Elo, p1_scalar 20,
p1_identity 4. The 7x applies only to the **side-imbalance endpoint**, and since `heads=64` arms
sit further from balance to begin with, that extra spread is plausibly a consequence of distance
from the bound at zero rather than an independent defect.

**Conclusion: do not roll back to E3-12.** Attention buys a uniform, symmetric +21.3 pp and ~79
Elo with no penalty in strength consistency. Side imbalance should be attacked directly, not by
reverting the backbone.

## The US side degrades from 80M to 160M, reproducibly

Self-play US win rate at tau=1.0, n=384, for every arm holding both an 80M snapshot and a final
(E3-18-22 has no 80M snapshot and is absent rather than approximated):

| arm | US% @80M | US% @160M | change |
|:---|---:|---:|---:|
| E3-17-22 (no pool) | 49.3% | 13.0% | **−36.3 pp** |
| E3-17-24 (no pool) | 50.0% | 20.6% | **−29.4 pp** |
| E3-20-22 (pooled) | 24.5% | 25.0% | +0.5 pp |

Both no-pool arms start at **near-perfect balance** — 49.3% and 50.0%, which is not a coincidence
worth ignoring — and lose about 30 pp of US strength over the following 80M. Same direction, large
magnitude, two independent seeds. This is the most reproducible result in this programme, and it
is more specific than "imbalance oscillates": the *US* side is what decays, from a balanced
starting point, during the second half of training.

The pooled arm stayed flat, but it was already at 24.5% at 80M, so "did not degrade" may be a
floor effect rather than protection. One pooled arm cannot separate those.

Two consequences worth acting on:

* **80M is not a waypoint to 160M; it may be better than 160M.** The unpooled control also ranks
  above its own 160M descendant on Elo (1998 vs 1939) in an earlier arena. Longer is not
  monotonically better on this axis, so a run's budget should not be assumed benign.
* **The endpoint should probably be read against 80M**, not only in absolute terms, since both
  arms pass through balance on the way to imbalance.
