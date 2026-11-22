# P54 -- compile alone and bf16 alone, from scratch against B6 (2026-11-21)

**Conclusion.**

* **Compiling the update costs nothing measurable: adopted for new launches.** R167 (B6's recipe with
  `--compile-update max-autotune` only) against B6 at matched steps reads 51.5 / 49.6 / 40.3 / 54.5 / 52.3% at
  400 / 800 / 1,200 / 1,600 / 2,000M (+-1.1). The one low read is 1,200M, where B6's SWA is unusually strong (its
  old-reference read is 42.5 there against 36.4 at 1,600M; P53's arms read 33-36% against it too); the other four
  read 49.6-54.5%, none below even by more than 0.4. On the references R167 ends level with B6 (2,000M: 44.4 / 36.2 against 45.3 / 35.7).
* **The compiled update's gradient is 0.13% from eager (cosine 0.999999), its differences unbiased across
  batches:** compile changes the sample, not the direction ([P51/P52](P51_P52_rollout_halves_and_bf16.md)).
  Alone it ran at ~150k steps/s from scratch (the logged median from 140M on, the GPU to itself), against 114.1k
  for the eager update in the 25M-step probe.
* **bf16 alone (R168) was stopped at ~139M with no read.** P53's deficit (3-7 points at 800M and later) is therefore attributed to bf16
  by elimination, backed by its measured numerics (4.3% of rows outside PPO's clip band, gradient 31% off), not by
  a bf16-only run. bf16 is postponed.

**Arms** (owner, 2026-11-21), both from scratch on seed 45, launched side by side for 8 h:

| code | run | recipe | reached |
|:---|:---|:---|---:|
| R167 | `E8.1-A8-R167-S45_20261121_095317` | R132 + `--compile-update max-autotune` | ~2,010M (14:40 UTC) |
| R168 | `E8.1-A8-R168-S45_20261121_095342` | R132 + `--update-bf16` | ~139M (stopped 10:24 UTC; reason not recorded -- minutes before the bf16 numerics measurement was committed, `2c333b4`) |

From-scratch probes (25M steps, alone): plain 114.1k, compile 129.1k (+13%), bf16 132.0k (+16%) steps/s (`1a6ebc5`).

**Instrument.** As P53's: greedy SWAs of the last 80M, head to head with B6 at the same step 1,000 deals a side,
the two references 1,000 a side. `data/reports/p54_compile_vs_b6_*.json`,
`recipe_E8.1-A8-R167-S45_*.json` (untracked).

| step | R167 vs B6 (US / USSR) | old reference: R167 (B6) | J1's best: R167 (B6) |
|---:|:---|:---|:---|
| 400M | 51.5 (50.3 / 51.1) | 23.8 (24.5) | 16.4 (16.4) |
| 800M | 49.6 (47.8 / 50.1) | 34.9 (34.9) | 26.6 (27.5) |
| 1,200M | 40.3 (37.7 / 41.9) | 31.7 (42.5) | 25.2 (32.1) |
| 1,600M | 54.5 (52.6 / 54.7) | 38.9 (36.4) | 30.5 (28.7) |
| 2,000M | 52.3 (50.0 / 53.9) | 44.4 (45.3) | 36.2 (35.7) |

**Decision** (owner, 2026-11-21, `01fae3e`): compile is adopted for new launches; bf16 is postponed (to be retried
only with the old/new log-prob precision fixed, audit M-7, and with fp32 heads). R167 is the compiled control of
[P55](P55_critic_everywhere.md).

**Caveats.** One seed, and B6 is itself one seed of the eager recipe, so "level" means within one recipe's
read-to-read noise, which is unmeasured (P56's C' was to measure it at +800M on R169's line).
