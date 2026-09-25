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

## E4-60-43 reached 800M, but z-loss costs a great deal of strength (2026-09-25)

**Stability.** E4-60-43 (z-loss 1e-5, seed 43) ran 0 → 800M without diverging.
* The mean log-normaliser stayed at 1.8–2.3 throughout.
* The largest single value moved between 55 and 520, spiking only during a pin.
* Approximate KL stayed at 0.01–0.02.
* The KL alarm fired once, at 684.4M. It was a single iteration (0.057) inside a pin, not a
  divergence.
* It pinned twice: at 140M (one bucket), and at ~655M, a deep pin (USSR ~100% of self-play) of
  roughly 40–60M that recovered by 720M.

**Strength.** `data/reports/long_z5_800M.{md,json}`: 23 players, 100 games per side per pair,
temperature 0, HeuristicBot at 1500.

| step | E4-60-43 (z 1e-5) | E4-59-43 (z 1e-4) | E4-59-44 (z 1e-4) | E4-57-43 (none) | E4-57-44 (none) |
|:---|---:|---:|---:|---:|---:|
| 80M | 1886 | | | | |
| 160M | **1725** | 2013 | 1926 | **2185** | **2142** |
| 190M | | 2012 | 2002 | | |
| 230–240M | 1803 | | | 2218 | 2208 |
| 320M | 1895 | | | | |
| 400M | 1921 | | | | 2323 |
| 480M | 1937 | | | | |
| 560M | 1918 | | | | 2334 |
| 640M | 1982 | | | | |
| 720M | 1991 | | | | |
| 800M | 2036 | | | | |

E4-57-44@590M rates 2341 and E4-08-36@240M rates 2279 in the same field.

**The z-loss runs are far weaker at matched steps. The deficit is large and the same in every
run:**
* at 160M, E4-60-43 is −460 against its no-z-loss twin, E4-59-43 is −170 against the same
  twin, and E4-59-44 is −216 against its twin;
* E4-60-43 got weaker from 80M to 160M (1886 → 1725);
* at 800M it rates 2036, below E4-57-44@160M, and loses 87% to E4-08-36@240M.

The runs' own training evaluations against HeuristicBot tell the same story:

| run | win rate vs HeuristicBot |
|:---|:---|
| E4-57-43 / E4-57-44 (no z-loss) | reach 97–100% by 160–240M |
| E4-60-43 (1e-5) | 73–96% throughout, to 240M |
| E4-59-43 / E4-59-44 (1e-4) | 61–96% |

**All three z-loss runs also carried higher entropy than their twins, on the US seat above
all.** The earlier reading that this entropy was run-to-run noise is withdrawn. It tracks the
presence of z-loss rather than its size, and so does the strength deficit.

**A candidate mechanism, not tested.** The softmax gives the shift direction of the logits
exactly zero gradient from every other loss term, so the z-loss is the only gradient there.
Adam normalises each parameter's step by that parameter's own gradient scale. For parameters
that carry mostly the shift direction, a z-loss of any coefficient is therefore amplified to a
full-sized step, and its size stops mattering. That would explain why 1e-5 hurt as much as
1e-4. The level drift itself, meanwhile, was not measured to cost strength. E4-57-44 grew to 2341
with its level in the thousands, and only its divergence was the problem.

**Status:** z-loss, as implemented (penalty on every row, from step 0), is not a fix. It stays in
the code, off by default.

## Does TF32 precision matter at these logit scales? (2026-09-25)

`data/logs/long/tf32_at_scale.py` (GPU) runs 4,096 rollout positions through each snapshot in
fp32, and in TF32 at batch 4,096 (update-sized) and at batch 512 (rollout-sized). The table
gives the largest |Δ log p| over legal actions.

| snapshot | largest logit, median / max | fp32 vs TF32: mean KL | fp32 vs TF32: max \|Δ log p\| | TF32 b4096 vs b512: max \|Δ log p\| |
|:---|:---|---:|---:|---:|
| E4-57-44@80M | 58 / 267 | 1.4e-7 | 0.09 | 0.03 |
| E4-57-43@230M (2M before it diverged) | 500 / 3,850 | 2.4e-7 | 1.0 | 0.44 |
| E4-57-44@400M | 809 / 4,960 | 2.9e-7 | 1.4 | 0.85 |
| E4-57-44@560M | 1,260 / 6,100 | 2.4e-7 | 1.7 | 1.1 |
| E4-57-44@590M (6M before it diverged) | 2,129 / 39,800 | 1.2e-6 | 8.5 | 3.0 |
| E4-60-43@800M (z-loss) | 16 / 64 | 3.0e-7 | 0.11 | 0.06 |

* **TF32 does not overflow.** It has fp32's exponent range and loses mantissa bits.
* **On average it barely moves the policy.** Mean KL stays at or below 1e-6 at every scale.
* **For unlikely actions the noise grows with the level.** The shift is ~1 nat at logits in the
  thousands and 3–8.5 nats once the tail reached ~40,000. The PPO ratio reads exactly those
  log-probs for a sampled action, so the mechanism is real.
* **But E4-57-44 trained healthily for ~300M steps with ~1 nat of it,** from 240M to 560M.
  E4-57-43 diverged at about the same level of noise (1.0 / 0.44). So TF32 precision does not
  explain E4-57-43's death on its own. It may have finished E4-57-44 once the tail jumped from
  6,000 to 40,000.

**The test is an fp32 long run: E4-56-43**, the same recipe with `--no-tf32`, seed 43, solo to
800M, launched 10:46 UTC. `launch_flags.py --diff` against E4-57-43 shows only `--tf32`.

## E4-56-43 (fp32) reached 800M without diverging (2026-09-25)

**Stability.** E4-56-43, the fp32 control, ran 0 → 800M without a divergence.
* Approximate KL stayed at 0.010–0.02 throughout. The KL alarm fired once, at 213.6M: a single
  iteration of 0.057 inside the pin.
* **The logit level drifted as it does in every run without z-loss.** The mean log-normaliser
  went 6 → 31 through 270M and then 87 (320M), 180 (400M), 311 (480M), and ~330–390 from 520M.
  The largest single value reached 4,000 by 400M and 5,300–8,900 over 520–800M.
* **So fp32 trained through the same logit scales at which both TF32 runs died.**
  * E4-57-43 died at 232M with a largest logit of 3,850.
  * E4-57-44 was fine to 560M at ~6,000 and died at 596M after its tail jumped to ~40,000.
  * No such jump happened in fp32.

**Its one long episode was a pin from ~125M to ~265M**: USSR ~98–99.6% of self-play,
`adv_std_raw` down to 0.074 at 200M. It recovered without intervention. The logit level grew
fastest during the pin, 11 → 31 in mean. E4-60-43's tail also spiked during its pin.

**Strength.** `data/reports/long_fp32_800M.{md,json}`: 23 players, 100 games per side per pair,
temperature 0, HeuristicBot at 1500. Elo is field-relative, so these numbers do not compare
directly with the earlier fields.

| step | E4-56-43 (fp32, seed 43) | E4-57-44 (TF32, seed 44) |
|:---|---:|---:|
| 80M | 1896 | |
| 120–240M (inside the pin) | 1626–1689 | 2067 (160M), 2130 (240M) |
| 280M | 1983 | |
| 320M | 2041 | |
| 400M | 2128 | 2248 |
| 480M | 2120 | |
| 560M | 2140 | 2263 |
| 640M | 2150 | |
| 720M | 2120 | |
| 800M | 2145 | |
| 590M | | 2253 |

References in the same field: E4-08-36@240M 2212, E4-57-43@230M 2158, E4-60-43 (z-loss 1e-5)
1861 @400M and 1940 @800M.

**Reading:**
* **Survival:** fp32 survived to 800M on 1 of 1 run. TF32 died on 2 of 2, at 232M and 596M.
  With the logit scale matched, and with the measured TF32 noise on unlikely actions (1–8.5 nats
  at these scales), TF32 is now the likely *contributor* to the long-run divergences. The drift
  is common to both, and in fp32 alone it did no measured harm to 800M. This is one fp32 run;
  n = 1 against 2.
* **Strength from time alone levels off at ~400M in both surviving curves.**
  * E4-56-43 gains +145 over 280–400M and then +17 over 400–800M.
  * E4-57-44 gains about +15 over 400–590M.
  * Past ~400M steps, time alone buys little on M2d with this recipe.
* **E4-56-43's plateau (~2,130–2,150) is ~110 below E4-57-44's (~2,250).** That is within the
  ~100 Elo seed spread, and seed 43 lost ~140M steps to its pin. It is not a precision effect
  that can be read from one pair.
* **z-loss is confirmed as costly** in this field too: E4-60-43@800M rates below E4-56-43@320M.

## Where the logit level lives, and centred per-entity heads (2026-09-25)

`data/logs/long/logit_anatomy.py` (CPU) splits each legal logit of M2d into `policy_head(h)` and
the per-entity correction. The level column is the median over positions of |mean over legal
actions|.

| snapshot | legal-logit level | `policy_head(h)` part | per-entity (country) part |
|:---|---:|---:|---:|
| E4-57-44@80M (TF32) | 59 | 1.9 | 60 |
| E4-57-44@400M | 813 | 1.4 | 807 |
| E4-57-44@590M | 2,266 | 2.7 | 2,259 |
| E4-56-43@800M (fp32) | 280 | 3.2 | 282 |
| E4-60-43@800M (z-loss) | 5.6 | 3.6 | 1.7 |

**All of the drift is in `pe_country`.** It is the one head whose output is shared by every
country: a single MLP applied per country. In E4 a decision's legal set never mixes countries with
non-country actions (confirmed by the owner). So a shift common to every country logit is
invisible to the policy, gets no gradient, and random-walks as the head's weights move for other
reasons.

**The fix, `--ladder-head-center` (`1e820df`), centres `pe_country`'s hidden features across the
84 countries before its final projection.** The tests confirm three things:
* every country-only distribution is unchanged;
* the correction's mean over countries equals the final bias exactly;
* that bias gets zero gradient, so it never moves.

It adds no loss term, unlike z-loss, so Adam has nothing to amplify. It is recorded as a
`pe_center` buffer and refused with `--merged-influence`. The 3M smoke test showed normal
training and speed.

**Stage 1, divergence (owner's plan):** E4-61-43/44 are the E4-57 configuration (TF32) plus
`--ladder-head-center`, seeds 43 and 44, as a pair to 800M, launched 14:36 UTC.
`launch_flags.py --diff` against E4-57-43/44 shows only `--ladder-head-center`.
**Stage 2, quality,** follows.

### E4-56-43 past 400M: a treadmill, not convergence or a cycle (2026-09-25)

**Head to head among its own late snapshots** (`data/reports/long_fp32_800M.json`, 100 games per
side per pair). Each cell is the row's win share, overall, as USSR / as US.

| | vs 400M | vs 480M | vs 560M | vs 640M | vs 720M |
|:---|---:|---:|---:|---:|---:|
| 480M | 56 (52/59) | | | | |
| 560M | 56 (57/54) | 52 (50/55) | | | |
| 640M | 58 (62/55) | 58 (53/64) | 59 (65/53) | | |
| 720M | 53 (62/44) | 55 (62/48) | 51 (64/38) | 53 (61/45) | |
| 800M | 55 (61/49) | 57 (62/53) | 62 (73/51) | 52 (63/41) | 59 (75/43) |

* **The later snapshot wins 15 of 15 pairs, and there are no cycles.** No triple has every leg
  above 55%.
* **The edge does not accumulate.**
  * 80M newer: 52–59%.
  * 400M newer: 55%.
* **Against outside references it is flat.** Its Elo is 2120–2150 at every late snapshot.
  * Against E4-08-36@240M it went from 39/30 at 400M to 52/29 at 800M.
  * Against E4-57-44@590M it went from 36/44 to 43/35.
* **The edge over older selves comes mostly from the USSR seat** (800M against 720M: 75% as USSR,
  43% as US). Against outside references the US seat stays weak, at 20–40%.

**Reading:** the run keeps moving and keeps beating its recent past, but gains nothing in general
strength. It has not converged to a fixed point, and it is not cycling intransitively at an 80M
resolution. It is a self-play treadmill on a plateau. Cycles faster than ~10M would not show at
this spacing; a tournament over the 10M snapshots would.
