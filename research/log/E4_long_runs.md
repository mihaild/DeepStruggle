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

## E4-57-44 diverged at 596M and was stopped

**Before the divergence:** the run was clean from 240M to 590M. It had approximate KL
0.009–0.011, KL to π_ref ≤ 0.15, USSR share 0.48–0.64 at every 40M readout, and no pin.

**The divergence** was caught by the per-iteration alarm at 596.6M and read from
`training_metrics.jsonl`. It began in the **critic**, not in the KL:

| steps | KL to π_ref | approx KL us / ussr | clip | explained variance | value loss |
|---:|---:|:---|---:|---:|---:|
| 596.05M | 0.18 | 0.006 / 0.033 | 0.30 | 0.97 | 0.026 |
| 596.18M | 0.06 | 0.007 / 0.048 | 0.33 | 0.95 | 0.050 |
| 596.25M | 0.40 | 0.013 / 0.068 | 0.41 | 0.75 | 0.166 |
| 596.31M | 0.75 | 0.037 / 0.023 | 0.38 | 0.22 | 0.303 |
| 596.44M | 0.11 | 0.134 / 0.080 | 0.49 | 0.02 | 0.292 |
| 596.64M | 0.37 | 0.965 / 0.604 | 0.52 | −0.09 | 0.424 |
| 597.75M | 4.66 | 1.4e6 / 1.2e6 | 0.57 | −0.01 | 1.428 |

The run was stopped at ~598.6M (exit 143, a kill). `resume_state.pt` (590.0M) is kept as
`resume_590020608steps_pre_divergence.pt`.

**The logit level had taken off beforehand.** Largest legal logit, median / p99 / max:

| step | median | p99 | max |
|:---|---:|---:|---:|
| 400M | 825 | 3,485 | 4,643 |
| 480M | 370 | 4,791 | 6,029 |
| 560M | 1,748 | 5,227 | 5,854 |
| 590M | 2,293 | 37,469 | 38,901 |

The 590M state, the last one kept, already carries that jump.

**Both long runs have now died**, E4-57-43 at 232M and E4-57-44 at 596M, and both through the
unbounded logit level. Both ran with TF32. No fp32 run has gone past 240M from scratch, so
whether fp32 would survive is unknown.

## What time bought before the divergence

`data/reports/long_runs_to_590M.{md,json}`: 17 players, 100 games per side per pair,
temperature 0, HeuristicBot at 1500.

| step | E4-57-44 | E4-57-43 | E4-08-36 (fp32, reference) |
|:---|---:|---:|---:|
| 40M | 1940 | | |
| 80M | 2046 | 1966 | 2037 |
| 160M | 2177 | 2226 | 2238 |
| 230–240M | 2234 | 2260 (230M) | 2317 |
| 320M | 2286 | | |
| 400M | 2350 | | |
| 480M | 2352 | | |
| 560M | 2367 | | |
| 590M | 2367 | | |

E4-08-37@240M rates 2179 in the same field.

* **E4-57-44 gains up to about 400M, then flattens.**
  * +104 over 160–240M;
  * +116 over 240–400M;
  * +17 over 400–590M.
* **E4-57-44@560–590M is the strongest M2d measured: 2367, +50 over E4-08-36@240M.** Head to
  head, 590M wins 57% as USSR and 60% as US against E4-08-36@240M. Against its own 240M
  snapshot it wins 75 / 56.
* **Measured within each seed, the gain from 240M to ~560M is +133 Elo.** The flat stretch from
  400M on could be a plateau, or the drift already costing strength. The logit level took off
  across that stretch.

## Relaunched with z-loss (E4-59-43, E4-59-44), 2026-09-25

**The change:** `--z-loss-coef 1e-4` (`69d8b01`) adds `1e-4 · mean(logsumexp(policy logits)²)` to
the update. Everything else is the E4-57 configuration, with TF32, on the same seeds, 43 and 44.
`launch_flags.py --diff` against E4-57-43/44 shows only `--z-loss-coef`.

**The 3M smoke test:**
* the log-normaliser fell from 2.16 to 1.22, against 2.2 → 2.9 with z-loss off;
* z-loss was ~3e-4, against a policy loss of ~0.07;
* entropy and throughput were unchanged.

**Launch:** 06:33 UTC. The readouts now carry `logit_lse_mean` and `logit_lse_absmax`, and the
per-iteration KL alarm is armed.

### E4-59 stopped at ~199M; E4-60-43 with z-loss 1e-5

**What E4-59 showed:** z-loss at 1e-4 held the level. The mean log-normaliser stayed at
0.36–0.64 and the largest single value at 18–66, against thousands in the twins. But both runs
carried visibly higher entropy than their twins, above all on the US seat (1.7–1.95 against
1.25–1.46 at 80–160M), and both leaned USSR:
* **E4-59-43:** pins at 40M and 110M, both recovered.
* **E4-59-44:** 0.87–0.89 at 80M and 160M.

**Why the entropy rises:** the penalty's gradient on a logit is `2c · lse · p`, so it pushes the
likeliest actions down hardest, and that flattens the policy. The owner stopped both at
199M / 196M.

| run | USSR share by 5M, 0–195M |
|:---|:---|
| E4-59-43 | .59 .64 .80 .84 .83 .88 .90 .96 .94 .89 .86 .79 .78 .63 .66 .88 .63 .44 .49 .69 .79 .96 .94 .81 .72 .56 .52 … |
| E4-59-44 | .53 .68 .65 .80 .82 .71 .74 .75 .86 .78 .59 .58 .74 .77 .79 .87 .89 .73 .71 .85 .86 .92 .75 .70 .94 .85 .92 … |

**E4-60-43** is the same configuration with `--z-loss-coef 1e-5`, seed 43, solo, launched
07:45 UTC. Seed 43's twin without z-loss died earliest, at 232M.
