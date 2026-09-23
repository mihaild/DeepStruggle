# E4-28-03 — online search distillation on M2d

**Running; updated per snapshot.** P15-X4b, re-run on E4: an honest searcher answers 1 in 8 of
all decisions during RL, and its visit distribution enters the loss as a cross-entropy term.
**Search produces targets and never acts**, so rollouts are the policy's own.

## Design

* **Resumed from `E4-08-03@160M` on the same seed 3.** E4-08-03's own 160→240M leg
  (`E4-08-03_20260922_202204`) is therefore an exact **step-matched control**.
  `launch_flags.py --diff` against it shows only `--search-ce-coef 0.5 --search-sims 64
  --search-node-filter all` and the step budget, which no schedule depends on. Both legs restore the
  same 10-member opponent pool.
* **Configuration is E3's X4b exactly**: coef 0.5, 64 simulations, every decision type eligible,
  subsample 0.125, determinized. It is the one configuration with a measured gain, +165.6 Elo over
  its control within 20M ([`../archive/E3_ladder/log/P15_X4b_search_during_rl.md`](../archive/E3_ladder/log/P15_X4b_search_during_rl.md)).
  E3's later collapse of that arm was opponent-pool starvation from a resume path bug, not the CE
  term ([`../archive/E3_ladder/log/P15_X4b_collapse_is_pool_starvation.md`](../archive/E3_ladder/log/P15_X4b_collapse_is_pool_starvation.md)).
* **Why now:** honest 96-sim search beats `E4-08-03@240M` 64% ([`search_cost_and_coverage.md`](search_cost_and_coverage.md) §10),
  and the training-dynamics levers both failed ([`E4_dynamics_ref_lambda.md`](E4_dynamics_ref_lambda.md)).
* **Cost:** ~1,200 steps/s while sharing the GPU, against ~45,000 without search. Search is
  the whole bottleneck: ~8,000 searched decisions per 65,536-step iteration.
* **Tripwires**, from E3: `search_ce_grad_frac` > 0.75, `adv_std_raw` < 0.05 or `kl_div` > 1.0
  over the last 20 iterations. In a smoke test the CE term was 55% of the gradient from the first
  iteration, matching E3's 0.561, with zero illegal targets.

## Step-matched ratings

Each snapshot is rated in its own small field against the control at the same step, the shared
160M start and the anchor, 100 games per seat per pair, temperature 0. Reports:
`data/reports/E4-28-03_at<N>M.{md,json}`. Win rates are the row's **as USSR / as US**.

| steps | E4-28-03 Elo | control Elo | Δ | vs control | vs 160M start | vs anchor |
|---:|---:|---:|---:|---:|---:|---:|
| 165M | 2238.9 | 2178.6 | **+60.3** | 69 / 58 | 77 / 64 | 72 / 73 |
| 170M | 2225.6 | 2155.8 | **+69.8** | 61 / 62 | 75 / 64 | 77 / 66 |
| 175M | 2248.7 | 2194.8 | **+53.9** | 59 / 61 | 75 / 54 | 83 / 74 |
| 180M | 2222.6 | 2156.6 | **+66.0** | 66 / 62 | 70 / 61 | 80 / 72 |

**Four of four snapshots beat the step-matched control on both seats**, by +54 to +70 Elo.

## The whole leg in one field

All four E4-28-03 snapshots and the control at every step, including its 200M dip and its 240M
end, rated together (`data/reports/E4-28-03_leg.{md,json}`):

| # | model | Elo | vs E4-08-03@180M | vs E4-08-03@240M | vs anchor |
|---:|:---|---:|---:|---:|---:|
| 1 | **E4-28-03@180M** | **2278.8** | 66 / 62 | **59 / 66** | 80 / 72 |
| 2 | E4-28-03@165M | 2265.6 | 63 / 53 | 66 / 66 | 72 / 73 |
| 3 | E4-28-03@175M | 2264.4 | 56 / 61 | 60 / 72 | 83 / 74 |
| 4 | E4-28-03@170M | 2255.1 | 65 / 56 | 55 / 64 | 77 / 66 |
| 5 | E4-08-03@165M | 2202.5 | 61 / 44 | 59 / 53 | 80 / 69 |
| 6 | E4-08-03@175M | 2199.9 | 62 / 46 | 49 / 46 | 80 / 67 |
| 7 | E4-08-03@180M | 2190.7 | — | 52 / 50 | 68 / 80 |
| 8 | E4-08-03@240M | 2177.6 | 50 / 48 | — | 62 / 81 |
| 9 | E4-08-03@170M | 2175.7 | 49 / 41 | 53 / 49 | 83 / 67 |
| 10 | E4-08-03@160M | 2134.8 | 52 / 29 | 47 / 39 | 70 / 69 |
| 11 | E4-08-03@200M | 2089.9 | 50 / 31 | 45 / 32 | 46 / 52 |
| 12 | E4-03-01@80M | 2041.1 | 20 / 32 | 19 / 38 | — |
| 13 | HeuristicBot | 1500.0 | 3 / 0 | 0 / 2 | 1 / 1 |

* **Every search snapshot outranks every control snapshot**, including the control's 240M end.
* **20M steps with search beat 80M steps without it.** `E4-28-03@180M` beats `E4-08-03@240M`
  on both seats, 59% / 66%.
* **The gain is a level, not a slope.** It appears within the first 5M and then holds: the
  search snapshots span 2255–2279, and the control's 165–240M snapshots span 2176–2203 apart from
  the 200M dip. Mean over the matched 165–180M snapshots: **+73.8 Elo**.
* **Smaller than E3's +165.6 at the same 20M**, which matches the smaller search headroom on E4
  (64% against E3's 75%, [`search_cost_and_coverage.md`](search_cost_and_coverage.md) §10).
* In this field the control's 240M snapshot ranks below its own 165–180M snapshots. The 160→240M
  leg is flat within ~30 Elo outside the dip. The +31.9 of [`P21_M2d_240M.md`](P21_M2d_240M.md)
  was 240M against 160M in a smaller field. Field-relative ratings move by tens of Elo; head-to-head
  (240M beats 160M 61% as USSR / 53% as US, identically in both fields) is the steadier read.

## Training-side observations

* Entropy fell from 1.01 at the resume to ~0.75–0.78 within 5M and then held, below the
  control's ~1.25 over the same leg. The search targets are sharper than the policy
  (`search_target_entropy` ~1.1 against a policy at ~1.25 at the start).
* Self-play side balance sits near 0.5, and games run longer than at the resume point (mean turn
  ~7.7).
* `adv_std_raw` 0.21–0.23 and `kl_div` 0.014–0.03 throughout; no tripwire has fired.
