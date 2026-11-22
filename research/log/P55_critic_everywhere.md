# P55 -- the critic everywhere, with compile, from scratch (2026-11-21)

**Conclusion.**

* **Both extra critics on the compiled update (R169) are not weaker than compile alone (R167) or than B6, and R169
  became the baseline** (P56 branches from its 2,010M state). Against R167 at matched steps: 52.6 / 49.5 / 58.3 /
  54.0 / 49.0% at 400 / 800 / 1,200 / 1,600 / 2,000M (+-0.8). Against B6: 52.7 / 47.5 / 47.6 / 53.8 / 50.3%
  (+-1.1). On both references R169 ends level with B6 (2,000M: 45.8 / 36.0 against 45.3 / 35.7).
* **No gain is shown either:** against R167 one read is clearly above (58.3% at 1,200M, where R167 itself is low
  against everything) and the last is level; against B6 two reads are ~2 SE under even (800 and 1,200M), two above (400 and 1,600M) and the last level.
* **It costs about 18% of throughput:** ~123k steps/s against R167's ~150k, both alone from scratch (logged medians,
  140M to 1,000M).
* **The coefficients are total weights, not per-row weights** (audit T-2): each extra critic's loss is a mean over
  its own rows, scaled by its coefficient relative to the main value loss, so at 1.0 an afterstate row weighs ~2.7x
  a decision row and a pre-deal row ~31x, and the value terms together carry ~3x `vf_coef`. The flag's help text
  said "rows weighted by this x --vf-coef"; the owner chose to document the total-weight meaning, and the help
  text now says so (`a18694c`). R169 was
  judged with the weights as they were.
* **Everything here, and the whole chain since B6, is one seed (S45).** The replicate that was to measure one
  recipe's seed noise, P56's C' (`E8.1-A8-R169-S45@2010M+S46`: R169 continued from 2,010M with seed 46), was stopped
  at ~2,328M (about +318M of its planned +800M; last metrics 10:32 UTC) on 2026-11-22 when training was paused
  (owner). **One recipe's seed noise at +800M is still unmeasured**, so P55's "not weaker" and P56's first margin
  (T1 53.7% against C) have no seed-noise bar beside them.

**Arm** (owner, 2026-11-21, `01fae3e`): R169 = R132 (B6's recipe, pure self-play) + `--compile-update max-autotune`
(adopted after [P54](P54_compile_alone.md)) + the afterstate critic at every learner decision and the pre-deal
critic, both at weight 1.0 (`--afterstate-value-coef 1.0 --afterstate-frac 1.0 --predeal-value-coef 1.0`, as
[P53](P53_compile_bf16.md)'s arm without bf16). From scratch on seed 45 to 2,000M
(`E8.1-A8-R169-S45_20261121_150423`), read every 400M against R167 at matched steps; the rule: the baseline if it
trains fast enough and is not weaker. Continued in place from 2,010M to 2,810M as P56's C.

**Instrument.** Greedy SWAs of each run's last 80M. R169 against R167 2,000 deals a side; against B6
(`E8.1-A8-R132-S45`) and the two references (the old reference, A13+R48's 12,730-12,800M SWA; J1's best,
`E8-A8-R102-S44@13520..13600M`) 1,000 a side. `data/reports/p55_critics_vs_r167_*.json`,
`p55_critics_vs_b6_*.json`, `recipe_E8.1-A8-R169-S45_*.json` (untracked).

| step | R169 vs R167 (US / USSR) | R169 vs B6 (US / USSR) | old reference: R169 / R167 / B6 | J1's best: R169 / R167 / B6 |
|---:|:---|:---|:---|:---|
| 400M | 52.6 (53.5 / 50.4) | 52.7 (50.9 / 53.1) | 24.2 / 23.8 / 24.5 | 15.2 / 16.4 / 16.4 |
| 800M | 49.5 (49.1 / 48.5) | 47.5 (48.6 / 44.9) | 34.4 / 34.9 / 34.9 | 27.7 / 26.6 / 27.5 |
| 1,200M | 58.3 (56.5 / 59.0) | 47.6 (46.9 / 47.0) | 41.9 / 31.7 / 42.5 | 31.8 / 25.2 / 32.1 |
| 1,600M | 54.0 (53.8 / 52.8) | 53.8 (55.0 / 51.3) | 40.0 / 38.9 / 36.4 | 32.9 / 30.5 / 28.7 |
| 2,000M | 49.0 (49.5 / 47.3) | 50.3 (50.6 / 48.6) | 45.8 / 44.4 / 45.3 | 36.0 / 36.2 / 35.7 |

The R169-against-B6 reads (the matched-step comparison with the eager, critic-free recipe) were measured during
P55 but not written up until this file (audit R-3).

**Throughput** (`steps_per_sec` medians from `training_metrics.jsonl`):

| run | 140-1,000M | 1,000-2,000M |
|:---|---:|---:|
| R169 (compile + both critics) | ~123k | ~114k |
| R167 (compile only) | ~150k | ~150k |

**The chain is one seed.** B6 (R132), R139, R155, R164 and R165-R171 all run seed 45 (`seed` 45 in every
`metadata.json`); B6's soup used seed branches S46 / S47 for 440M, but no recipe comparison on this line has a
second seed. P56 planned C' for exactly this; at its stop it had reached ~2,328M, so no read at +800M exists.

**Caveats.** One seed. Read-to-read swings of 5-10 points against one opponent (R167 at 1,200M) are larger than
the matches' own error, which is what a seed or snapshot noise bar would have to size.
