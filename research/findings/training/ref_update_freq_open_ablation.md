# `ref_update_freq`: an open ablation, not a drift

**2026-09-19.** Surfaced by `tools/scripts/launch_flags.py` while diffing E3-37-31 against
E4-02-01, which reported `--ref-update-freq 5000000` on one side and the 200,000 default on the
other. The first reading -- that E4 had silently reverted an anti-collapse intervention -- is
wrong, and the full survey says why.

## 200,000 is the ladder's baseline; 5,000,000 was a late experiment

| setting | arms | which |
|:---|--:|:---|
| `200000` (default) | 43 | all of E3-12 … E3-29, E3-34 … E3-35, and every E4 arm |
| `5000000` | 8 | E3-30, E3-31, E3-32, E3-33 (x3), E3-36, E3-37 |

E4 is therefore **continuous** with the ladder on this flag, not divergent from it. The diff
against E3-37-31 looked alarming only because that reference is one of the eight.

## The slow anchor did not prevent collapse

**E3-31-28 ran `ref_update_freq = 5000000` and collapsed** -- it is the arm whose `kl_div` rises
from ~0.1 to 47-107, the KL-domination signature in
[`../../method/detecting_collapse.md`](../../method/detecting_collapse.md). That is the direct
refutation: `P15_X2_slow_anchor.md` asked whether a slower reference anchor prevents the collapse
on its own, and an arm carrying it collapsed regardless.

Pointing the other way, E4-02-01 ran the *fast* anchor for a clean 320M. Neither setting predicts
the outcome on the evidence available, which is the definition of no signal.

## Decision (owner, 2026-09-19)

> "Keep ref update freq at 200k. Unlike other changes, it didn't gave clear signal, so should stay
> as experimental option to ablate later."

So it stays at the 200,000 default, and E4-03-01 / E4-04-01 keep it -- which also keeps that pair
a clean single-variable architecture A/B. **This flag is deliberately held constant, not
corrected.** When the collapse sweep gets more seeds it is a candidate axis to vary on purpose,
with both settings run against the same seeds rather than inferred across lineages.

## Contrast with the three real drifts

The opponent pool, the architecture flags and the warm start were drifts: each departed from what
*every* recent arm did, and none was chosen. This one was chosen, is the majority setting, and has
no measured effect. The distinction matters because
[`e4_architecture_discontinuity.md`](e4_architecture_discontinuity.md) treats an unexplained flag
difference as a defect -- correct there, wrong here. A diff tells you a flag differs; only the
survey tells you whether that difference is a mistake.
