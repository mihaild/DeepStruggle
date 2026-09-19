# Which side actually degraded? The control, rated per seat against frozen anchors

**Measured 2026-09-17**, after the owner pointed out that one-sidedness of self-play cannot answer
this question. It cannot, and the answer it appeared to give was **backwards**.

## Why this had to be measured rather than read off the trace

`critic_base_rate` is `max(p, 1 − p)` (`critic_tracker.py:140`) — the majority-class rate. It is
always ≥ 0.5 and carries **no direction**, so it cannot name a side; and as a magnitude it is a
fact about the pair, equally consistent with one side collapsing and with both sides improving at
different rates. Both points are now in
[`../../../method/measurement_pitfalls.md`](../../../method/measurement_pitfalls.md), and the first was already
recorded in [`../../../log/seed_variance_and_pooling.md`](../../../log/seed_variance_and_pooling.md) against earlier reports
that made the same mistake.

The instrument for "is a side degrading" is **win rate as each side against a frozen opponent**.

## The measurement

**Which "control".** `E3-26-28`: resumed from `p28_200M`, `search_ce_coef = 0.0`,
`--ref-update-freq 200000`, pooled, seed 20260928 — the plain recipe. Original run 0→20M
(`_20260917_025539`), continued 20→80M (`_20260917_074533`); its step counter is therefore *on top
of* the source's 200M. It is the step-matched no-search partner for the X4b search arm `E3-29-28`
and the X4a step-4 control, which is why it is called "the control" throughout those entries.

It is **not** `E3-30-28`, the no-search arm at `--ref-update-freq 5000000` started from scratch,
which is the control for the *anchor* question in
[`P15_X2_slow_anchor.md`](P15_X2_slow_anchor.md) rather than for the search question.

Rated at 20M / 60M / 80M against the two frozen anchors. Temperature 0, 300 games a side,
`/workspace/data/tournaments/P15_control_per_seat/`.

**Against `frozen_200M`:**

| control | as USSR | as US |
|:---|---:|---:|
| @20M | 53.7% | 46.3% |
| @60M | 57.0% | 47.0% |
| **@80M** | **54.0%** | **19.3%** |

**Against `frozen_280M`:**

| control | as USSR | as US |
|:---|---:|---:|
| @20M | 38.3% | 64.3% |
| @60M | 49.3% | 67.7% |
| **@80M** | **30.3%** | **40.3%** |

Elo in the same field: control_60M 1535.1, control_20M 1526.9, frozen_200M 1523.1,
frozen_280M 1500.0, **control_80M 1417.1**.

## The answer, and the correction it forces

**The control's US play collapsed; its USSR play did not.** Against a frozen opponent its US win
rate falls from 47.0% at 60M to **19.3%** at 80M, while USSR goes 57.0% → 54.0% — essentially
flat. Against the other anchor the same shape: US 67.7% → 40.3%, USSR 49.3% → 30.3%.

This is the **opposite** of what was written in
[`P15_X4b_search_during_rl.md`](P15_X4b_search_during_rl.md) and
[`P15_X4a_distillation.md`](P15_X4a_distillation.md), where the story was "RL in this recipe gives
away the USSR side". That claim came from a **side-balance column averaged across a tournament
field** — and a field-averaged side balance is not a property of the model. The same checkpoint,
`E3-26-28` @20M, reads **−13.9 pp (USSR 34.7 / US 48.7)** in X4a's field and **+2.8 pp (USSR 55.8 /
US 53.1)** in this one. Nothing about the checkpoint changed; the opponents did.

So two instruments were wrong for the question and agreed with each other by accident:
`critic_base_rate`, which has no direction, and a field-averaged side split, which floats with the
field. The per-seat rating against a *fixed* opponent is the one that answers it.

## What survives

* **The control does degrade badly by 80M** — 1417.1 Elo, below both frozen anchors and below its
  own 20M self. That part was right, and it does not depend on the side question.
* **Its degradation is concentrated in one seat**, which is the thing the base-rate trace was
  gesturing at without being able to say. It is the **US** seat.
* The X4a within-arm side-balance observation (−17.3 pp and −11.6 pp over 20M in two arms) is
  measured the same field-averaged way and is **suspect for the same reason**. It has not been
  re-measured per seat against frozen anchors, and should not be cited as a side-specific claim
  until it is.

## Caveat

Two anchors, one control lineage, 300 games a side. The direction is consistent across both
anchors and large (a 28 pp drop in US win rate against `frozen_200M`), so it is unlikely to be
noise, but it is one lineage.
