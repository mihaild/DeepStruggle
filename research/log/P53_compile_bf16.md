# P53 -- both extra critics, on the fast update (compile + bf16), from scratch (2026-11-20/21)

**Conclusion.**

* **The extra critics cost nothing and gained nothing measurable in 2,000M:** the arm (R166, both extra critics at
  weight 1.0) against its control (R165) at matched steps reads 53.2 / 49.5 / 49.2 / 50.3 / 48.0% at 400 / 800 /
  1,200 / 1,600 / 2,000M (+-0.8), and the two are level on both references.
* **Both arms trail B6, the same recipe without compile and bf16, at matched steps:** the control 48.8 / 45.5 /
  33.1 / 46.7 / 44.7%, the arm 52.3 / 47.3 / 36.3 / 47.1 / 43.3% (+-1.1). Apart from 1,200M (where B6's SWA is
  unusually strong: P54's compile-only arm also reads 40.3% there) that is 3-7 points behind at 800, 1,600 and
  2,000M (about 4 on average), and the references say the same (2,000M: old reference 40.2 / 40.4% against B6's 45.3).
* **The fast update was the suspect, and P54 split it:** compile alone is level with B6, so the cost is bf16's
  ([P54](P54_compile_alone.md); bf16's numerics in [P51/P52](P51_P52_rollout_halves_and_bf16.md)). bf16 is
  postponed; the extra critics were carried into P55 on compile alone ([P55](P55_critic_everywhere.md)).

**Arms** (owner, 2026-11-20), both from scratch on seed 45, 8 h side by side, read every 400M:

| code | run | recipe | reached |
|:---|:---|:---|---:|
| R165 (control) | `E8.1-A8-R165-S45_20261120_230117` | R132 (B6's recipe, pure self-play) + `--compile-update max-autotune --update-bf16` | ~2,723M |
| R166 (arm) | `E8.1-A8-R166-S45_20261120_230137` | R165 + `--afterstate-value-coef 1.0 --afterstate-frac 1.0 --predeal-value-coef 1.0` | ~2,167M |

Probed first (25M steps each from scratch): alone 166.7k / 129.1k steps/s, together 95.4k / 78.8k; the logged
medians while sharing the GPU from 140M on were ~108k / ~87k. The extra critics' coefficients are total weights
relative to the main value loss, each a mean over its own rows: at 1.0 an afterstate row weighs ~2.7x a decision
row and a pre-deal row ~31x (audit T-2, documented in `a18694c`).

**Instrument.** `tools/tournament.py`, greedy, SWAs of each run's last 80M (`@<step-80M>..<step>M`). Head to
head with B6 (`E8.1-A8-R132-S45`) at the same step, 1,000 deals a side; arm against control 2,000 a side; the old
reference (A13+R48's 12,730-12,800M SWA) and J1's best (`E8-A8-R102-S44@13520..13600M`) 1,000 a side.
`data/reports/p53_*.json`, `recipe_E8.1-A8-R16{5,6}-S45_*.json` (untracked).

| step | arm vs control | control vs B6 | arm vs B6 | old reference: control / arm (B6) | J1's best: control / arm (B6) |
|---:|---:|---:|---:|:---|:---|
| 400M | 53.2 | 48.8 | 52.3 | 24.1 / 22.6 (24.5) | 15.2 / 17.3 (16.4) |
| 800M | 49.5 | 45.5 | 47.3 | 32.8 / 32.4 (34.9) | 24.7 / 27.2 (27.5) |
| 1,200M | 49.2 | 33.1 | 36.3 | 25.7 / 30.5 (42.5) | 19.6 / 19.9 (32.1) |
| 1,600M | 50.3 | 46.7 | 47.1 | 33.8 / 34.7 (36.4) | 27.2 / 26.5 (28.7) |
| 2,000M | 48.0 | 44.7 | 43.3 | 40.2 / 40.4 (45.3) | 31.5 / 30.1 (35.7) |

Per seat (US / USSR) against B6 at 2,000M: control 40.5 / 47.3, arm 41.0 / 44.9. No seat ran away in either arm.

**Caveats.** One seed. The two arms shared the GPU, so their step counts per hour are not each one's speed alone.
Under `--update-bf16` the old log-probs are fp32 and the new ones bf16 (audit M-7), which may be part of the cost.
