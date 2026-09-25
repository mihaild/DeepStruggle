# Long runs to 800M (E4-57-43, E4-57-44), 2026-09-25

**Setup:**
* **Recipe:** today's defaults, the E4-57 configuration: the E4-08 recipe (M2d, λ 0.98,
  pool 0.3/12), TF32, snapshots every 10M, pool every 5M.
* **Seeds:** 43 and 44, from scratch to 800M in one run each, as a pair from 00:06 UTC.
* **Flag check:** `launch_flags.py --diff` against E4-57-40 shows only the seed and
  `--train-steps`.
* **Question:** what training time alone buys ([`../runs.md`](../runs.md)).

## E4-57-43 diverged at 232M and was stopped

**Before the divergence:**
* The run leaned USSR at 0.86–0.93 from 40M. It pinned for one 5M bucket at 125–130M
  (`adv_std_raw` 0.11) and was balanced again by 160–200M.
* Through 230M every training signal was normal: approximate KL rollout → update ≈ 0.011 per
  seat, KL to π_ref 0.03–0.04, clip fraction 0.22, critic explained variance 0.89.

**The runaway:** from 232.06M the KL to π_ref ran away over eight iterations (0.5M steps),
although π_ref is refreshed every ~3 iterations:

| iteration | steps | KL to π_ref | approx KL us / ussr | clip | value loss |
|---:|---:|---:|:---|---:|---:|
| 3539 | 232.00M | 0.04 | 0.016 / 0.012 | 0.20 | 0.048 |
| 3540 | 232.06M | 2.05 | 0.035 / 0.081 | 0.28 | 0.058 |
| 3541 | 232.13M | 3.45 | 0.373 / 0.240 | 0.36 | 0.069 |
| 3542 | 232.19M | 10.4 | 0.100 / 0.809 | 0.48 | 0.199 |
| 3544 | 232.33M | 99.8 | 0.437 / 0.385 | 0.40 | 0.210 |
| 3547 | 232.52M | 685 | 2.957 / 0.156 | 0.43 | 0.536 |
| 3548 | 232.59M | 463,659 | 2.013 / 0.307 | 0.57 | 1.052 |
| 3549 | 232.65M | 1,003,800 | 0.092 / 0.150 | 0.46 | 0.447 |

**After the runaway:**
* Critic explained variance went to 0.
* Both seats went near-uniform (entropy 2.37).
* Approximate KL ran at 10–1e7 through 240M.
* The legal logits reached ~1e8 (pool member at 235M) and ~9e10 (snapshot at 240M).

No weight was ever non-finite. The run was stopped at ~243M (exit 143, a kill). Its
`resume_210042880steps.pt` is the last saved state before the divergence.

**No other run has come near this.** The largest KL to π_ref per 40M window, over every
iteration:
* E4-57-44 (TF32) to 360M: ≤ 0.144;
* E4-08-36 and E4-08-37 (fp32) to 240M: ≤ 0.13.

## Logit level drift (every run)

`data/logs/long/logit_scale.py` (CPU) reads the largest legal logit per position on 1,280 rollout
positions. The table gives median / maximum.

| run | 40M | 80M | 160M | 240M | 360M |
|:---|:---|:---|:---|:---|:---|
| E4-08-36 (fp32) | 22 / 90 | 219 / 592 | 362 / 1,302 | 363 / 2,950 | |
| E4-57-44 (TF32) | 15 / 92 | 65 / 248 | 415 / 2,592 | 423 / 4,175 | 1,405 / 5,351 |
| E4-57-43 (TF32) | 26 / 85 | 67 / 308 | 249 / 826 | 590 / 3,840 at 230M | ~1e8 at 235M |

The spread between legal logits stays at ~20–25. What drifts is the common level. The softmax is
invariant to that level, so nothing in the loss bounds it. It grows in fp32 and in TF32 alike.

## What is and is not established

* **Established:** the level drift is unbounded in every run. E4-57-43's runaway ended with the
  level at 1e8–1e11. The runaway began in the KL-to-π_ref term, not in the value head or the
  advantages.
* **Not established: that TF32 caused it.** The first hypothesis was that TF32's 10-bit mantissa
  turns logits in the thousands into nat-sized rounding noise between the rollout and update
  forwards. E4-57-44, also TF32, has run from 240M to 360M with larger logits (maximum 5,351)
  and a steady approximate KL of 0.010. The trigger is unknown. The run cannot be replayed
  exactly, because a resume deals fresh games.
* **Also not established: that fp32 is safe past 240M.** No fp32 run from scratch has gone past
  240M in one run.

**Candidate remedies, not applied (they need the owner's decision):**
* **z-loss.** A small penalty on the log-normaliser keeps the level near 0, and so bounds the
  drift.
* **A divergence stop.** Abort loudly when the KL to π_ref exceeds a threshold (it is ≤ 0.15 in
  every healthy iteration measured), so a run is resumed from its last healthy snapshot instead
  of training on for 10M steps after dying.

E4-57-44 continues on TF32, watched by a per-iteration alarm on approximate KL > 0.05
(`data/logs/long/kl_alarm.py`).
