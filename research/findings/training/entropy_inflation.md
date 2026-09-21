# Entropy inflation — a second failure mode, and the detector is blind to it

First observed 2026-09-21 in `E4-12-01`, the P21 anchor continued from 80M to 160M. It lost
**~330 Elo** while every collapse indicator this repository has stayed healthy.

Companion to [`side_collapse.md`](side_collapse.md), which describes the *other* failure mode.
Measurements: `data/reports/P21_anchor_decline_trajectory.json`,
`data/reports/P21_matched_steps_160M.json`.

## What happened

`E4-03-01` rated 2093.3 at 80M. Continued to 160M as `E4-12-01` it rated 1737.3 — level with the
collapsed arms and with M1, two rungs below where it started. Rated again in a separate field
against its own intermediate snapshots, the same fall appears as −330.5.

The decline is **a cliff, not a drift**:

| steps | Elo | vs 80M | entropy | kl_div | opp_win_rate | adv_std_raw |
|---:|---:|---:|---:|---:|---:|---:|
| 85M | 2147.2 | −12.4 | 1.281 | 0.0427 | 0.806 | 0.2514 |
| 95M | 2136.4 | −23.2 | 1.491 | 0.0383 | 0.762 | 0.2622 |
| **105M** | **1840.5** | **−319.1** | 1.803 | 0.0306 | 0.638 | 0.2300 |
| 115M | 1721.2 | −438.4 | 1.941 | 0.0270 | 0.525 | 0.2171 |
| **125M** | **1609.0** | **−550.6** | 1.959 | 0.0277 | 0.398 | 0.2454 |
| 135M | 1770.8 | −388.8 | 1.866 | 0.0327 | 0.359 | 0.2588 |
| 145M | 1731.3 | −428.3 | 1.855 | 0.0280 | 0.443 | 0.2362 |
| 160M | 1829.1 | −330.5 | 1.730 | 0.0340 | 0.503 | 0.2598 |

**296 Elo in one 10M interval**, a trough at 125M, then partial recovery. Elo is a near-perfect
inverse mirror of policy entropy across the whole leg.

## Why nothing caught it

Every indicator built for the side collapse stayed healthy:

* `adv_std_raw` never left the clean band — 0.217 to 0.262 against the collapsed range of
  0.0036–0.0313. **The detector cannot fire on this.**
* `us_episode_frac` finished at 0.301 and never approached either tail. The arm's verdict from
  `collapse_check.py` is **CLEAN**, and that verdict is correct — it is not a side collapse.
* `explained_variance` stayed ordinary.

This is a distinct failure: the policy does not become one-sided, it becomes **indecisive**.

## It is not a configuration error

Checked before the result was believed, because a 330 Elo fall in a freshly launched arm is more
likely to be a mistake than a discovery:

* `launch_flags.py --diff` against the source run reports exactly **one** difference,
  `--train-steps`. Nothing else.
* The opponent pool restored correctly: 11 of 12 recorded members, spanning 75M, `frac=0.3`,
  capacity 12. Not pool starvation (invariant 14).
* **No schedule is derived from `train_steps`.** The curriculum was inactive (no `[CURRICULUM]`
  line, `reward_scheme=blunder_aware`), so `curriculum_switch_at` is infinite, and the rollout
  "temperature schedule" is a fixed per-environment band assignment rather than a function of
  progress. This matters beyond this arm: had any schedule keyed off `train_steps`, **every M2d
  extension would have been confounded too**. None does.

## The signature, and a hypothesis about mechanism

Across the four arms with both legs measured, sustained inflation separates the failures from the
healthy arms, where a rise is transient:

| arm | entropy start → mid → end | outcome |
|:---|:---|:---|
| anchor `E4-03-01`, 0–80M | 1.337 → 1.172 → 1.197 | healthy |
| **anchor `E4-12-01`, 80–160M** | 1.273 → **1.957** → 1.724 | **−330 Elo** |
| M2d s13 `E4-11-13`, 80–160M | 1.258 → 1.480 → 1.261 | +160 Elo |
| M2d s3 `E4-11-03`, 80–160M | 1.197 → 1.343 → 1.331 | +123 Elo |
| **M2d s14 `E4-11-14`, 80–160M** | 1.346 → 1.682 → **1.874** | **side collapse** |

The side-collapsed arm shows the same sustained inflation as the anchor. That suggests entropy
inflation is the **more general indicator**, with the side collapse as one downstream
manifestation — and it would have caught both failures where `adv_std_raw` caught one.

**Mechanism, unverified:** `kl_div` *falls* (0.043 → 0.027) while entropy *rises*. The active
policy is therefore tracking its reference closely while the reference itself drifts flatter —
the shape of a ratchet through `π_ref` refreshes. NashPG regularises against a frozen snapshot
refreshed every `ref_update_freq` steps; if a slightly flatter policy is captured at a refresh,
the KL term then pulls toward that flatter target, and the next refresh bakes in more of it.
`ref_update_freq` is 200,000 here, and was already recorded as an open ablation in
[`ref_update_freq_open_ablation.md`](ref_update_freq_open_ablation.md). This gives that ablation a
specific prediction to test rather than a general curiosity.

## What this does NOT establish

* **n = 1 for the anchor.** One arm, and it ran with `seed = None`, so there is no replicate. "The
  late-E3 architecture degrades past 80M" is *not* a supported claim yet; this could be one seed.
  A second anchor arm to 160M is ~3.9 GPU-h and is what would settle it.
* **The entropy signature is four arms.** It is a hypothesis worth instrumenting, not a finding.
* **The mechanism is inferred from two metrics moving in opposite directions**, not demonstrated.

## Consequence for the M2d comparison

[`P21_M2d_160M_slope.md`](../../log/P21_M2d_160M_slope.md) can now be completed, and one framing
in it must be avoided. At 160M the gap between M2d and the anchor is +465.8 Elo — but M2d did not
pull ahead by that much, **the anchor fell**. The fair statement compares M2d at 160M against the
anchor in its best measured state:

> M2d at 160M rates 2203.1 against the anchor's best (2093.3 at 80M): **+110 Elo, at twice the
> steps and 46% of the wallclock.**

The anchor's instability past 80M is a separate finding and should be reported as one, not folded
into M2d's margin.

It also puts a caveat under the ladder itself: the P21 anchor was chosen as a fixed reference, and
a reference that loses 330 Elo when trained further is not fixed. Rungs compared at 80M are
unaffected — that is where the anchor was measured and where it is healthy.
