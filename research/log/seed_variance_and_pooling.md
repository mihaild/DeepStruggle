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
