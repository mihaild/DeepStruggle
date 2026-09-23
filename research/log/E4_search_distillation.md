# E4-28 — online search distillation on M2d

Two arms, `E4-28-03` (seed 3, from 160M) and `E4-28-05` (seed 5, from 80M), run overnight
2026-09-22/23 and still training when this was written. P15-X4b, re-run on E4: an honest searcher answers 1 in 8 of
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
* **Correction (written 06:20, after the arms were launched): E3's configuration collapsed even
  with a healthy pool.** When choosing it I read only the first half of
  [`../archive/E3_ladder/log/P15_X4b_collapse_is_pool_starvation.md`](../archive/E3_ladder/log/P15_X4b_collapse_is_pool_starvation.md),
  where the *early* collapse is attributed to pool starvation from a resume path bug. Its appended
  sections retract the rest. `E3-35-28`, the same configuration with a working pool, peaked at 90%
  against a frozen anchor ~25M into its leg, began declining at ~30M, and reached ~11% by 36.5M,
  on both seats. The archive's verdict is that search CE at coefficient 0.5 with a 200k reference
  anchor *"collapses completely, with or without a healthy opponent pool. The pool changes when."*
  [`../questions.md`](../questions.md) had it right. **The collapse window is 30–36M into the
  leg, and E4-28-03 reached it overnight**; see *The E3 collapse window* below.
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
| 185M | 2241.3 | 2116.0 | **+125.3** | 74 / 70 | 71 / 64 | 80 / 83 |
| 190M | 2239.1 | 2044.9 | **+194.2** | 77 / 75 | 80 / 67 | 82 / 68 |
| 195M | 2228.7 | 2060.4 | **+168.3** | 72 / 73 | 77 / 63 | 75 / 85 |

**Seven of seven snapshots beat the step-matched control on both seats.** The 185–190M gaps are wider because the control is sliding into its 200M dip: `E4-08-03@190M` rates below its own 160M
start. Against the fixed 160M start the arm holds or improves, 77 / 64 at 165M and 80 / 67 at 190M.
Search distillation rides through the control's dip.

### The E3 collapse window

E3's healthy-pool arm was at 90% ~25M into its leg, declining by ~30M and gone by 36.5M. E4-28-03
at the same distance, against its fixed 160M start (which, unlike the control, does not move):

| steps into the leg | snapshot | vs 160M start (USSR / US) |
|---:|:---|---:|
| 25M | 185M | 71 / 64 |
| 30M | 190M | 80 / 67 |
| 35M | 195M | 77 / 63 |

**No decline yet at 35M.** But E3's arm went from its peak to ~11% in about 6M steps, and this
arm stops at 07:00 at ~196M, before 36.5M. The window is reached but not cleared. Extending
E4-28-03 is the single most important next step.

## Headroom left after 20M

96-sim honest search on top of `E4-28-03@180M` still wins **59.0%** (+63 Elo), against 64.0% on
top of the undistilled `E4-08-03@240M` ([`search_cost_and_coverage.md`](search_cost_and_coverage.md)
§10). The overall drop is within noise; the US-seat gain fell from 69% to 56% while the USSR seat
held. Most of the headroom remains, so a longer leg or a second round has something to take,
which fits a gain that arrived within 5M and then plateaued at this coefficient.

## Seed 5: a second replicate, from 80M

`E4-28-05` repeats the arm on seed 5 from `E4-08-05`'s 80M resume state (`E4-28-05_20260923_031034`);
E4-08-05's 80→160M leg (`E4-08-05_20260921_033333`) is its step-matched control. `launch_flags.py
--diff` shows only the search flags and `--resume-every-steps`. It runs alongside E4-28-03, so
both are at ~1,000–1,200 steps/s.

| steps | E4-28-05 Elo | control Elo | Δ | vs control | vs 80M start | vs anchor |
|---:|---:|---:|---:|---:|---:|---:|
| 85M | 2179.5 | 2022.7 | **+156.8** | 71 / 72 | 79 / 74 | 62 / 55 |
| 90M | 2207.0 | 1976.7 | **+230.3** | 82 / 77 | 85 / 73 | 66 / 59 |
| 95M | 2222.8 | 2012.8 | **+210.0** | 81 / 79 | 88 / 72 | 52 / 66 |
| 100M | 2145.6 | 1913.5 | **+232.1** | 89 / 77 | 80 / 77 | 56 / 59 |

**Four of four seed-5 snapshots beat the control on both seats.** The control sits below its own
80M start at 90M and 100M, which inflates the margin; against the fixed 80M start the arm holds
at 79–88% as USSR and 72–77% as US from 85M on.

At 90M part of the margin is the control's own dip: `E4-08-05@90M` rates below its 80M start
(47 / 37 against it). Against the shared 80M start, which is not a moving reference, the arm
goes 79 / 74 at 85M to 85 / 73 at 90M.

A larger gain than seed 3's at the same 5M into the leg, from an earlier and weaker starting point.
It is the first seed-5 M2d snapshot to beat the anchor on both seats.

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

## What this suggests next

Proposals, not decisions:

1. **Extend E4-28-03 past E3's collapse window before adopting anything.** It is the first
   intervention on E4 that beats plain M2d, on two seeds and on both seats, but E3's identical
   configuration collapsed completely 30–36M into its leg with a healthy pool, and E4-28-03 stopped
   at ~36M, inside that window. Rate it against the fixed 160M start every 2.5M from 195M to ~220M.
   If it holds, adopt it, priced in GPU-hours: ~40x per step (~1,200 against ~45,000 steps/s), with
   20M search steps (~4.6 h) beating 80M plain ones (~0.5 h). If it collapses, the gain is a
   transient worth harvesting by distilling for ~20M and stopping — the E3 record's shape too.
2. **Find out why the gain plateaus while headroom remains.** Search still beats the distilled
   net 59%, yet the arm stops improving after ~5M at this coefficient. Candidates to vary one at
   a time: the searched fraction (1 in 8), simulations (64, while play saturates at 96), and the
   coefficient (0.5; the CE term is ~55% of the gradient).
3. **Make it cheaper before making it bigger.** Search is the entire bottleneck. The searched
   fraction and the node filter trade cost against signal directly, and E3 found most of the
   signal outside card/play-mode nodes, so which decisions to search is a measurable choice.
4. **Lower coefficient or a KL-guarded anneal**, which E3's X4b log proposed and never ran, are
   the obvious hedges if the extension collapses. None of the three tripwires fired in ~50M search
   steps across both arms, and E3's collapse was also missed by live instruments until it showed
   against a frozen anchor, so frozen-anchor ratings are the monitor that matters.
