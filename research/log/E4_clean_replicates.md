# Clean M2d replicates to 240M (E4-08-36, E4-08-37), 2026-09-24

Two E4-08 replicates (M2d, λ 0.98, opponent pool 0.3/12) on new seeds, 36 and 37. Each ran from
scratch straight to 240M in a single run. Nothing resumed, so the drained-pool confound
([`P25_pool_resume_bug.md`](P25_pool_resume_bug.md)) that affected every earlier E4-08 leg past
80M is absent. They ran on the fixed CUDA-graph code (`9bfceb1`) and the clang engine.
`launch_flags.py --diff` against E4-08-03's first leg shows the seed and `--train-steps` only.
Both finished without a crash.

## Strength

`data/reports/e4-08_clean_240M.{md,json}`: 24 players, 100 games per side per pair, temperature 0,
HeuristicBot anchored at 1500.

| step | E4-08-36 | E4-08-37 | E4-08-03 (drained past 80M) | E4-08-05 (drained past 80M) |
|:---|---:|---:|---:|---:|
| 40M | 1769 | 1799 | | |
| 80M | 1988 | 1945 | 2051 | 1989 |
| 120M | 2090 | 1953 | | |
| 160M | 2199 | 2048 | 2180 | 2104 |
| 200M | 2230 | 2052 | | |
| 240M | **2294** | **2114** | 2216 | 1853 (collapsed) |

Reference points in the same field:
* E4-48-03@240M (seed 3, fixed pool from 160M): 2265
* E4-48-05@240M: 2148
* E4-48-05-160M.11@240M: 2148
* E4-31-03@240M (slow π_ref from 160M): 2323, the top of the field
* E4-03-01@80M (anchor): 2053

* **E4-08-36@240M is the strongest plain M2d measured.** It is +78 over E4-08-03@240M, winning
  62% as USSR and 67% as US head to head, and +29 over the fixed-pool E4-48-03. It is 29 under
  E4-31-03 (slow π_ref from 160M, one seed, drained pool).
* **Growth continues to 240M.** E4-08-36 gains +219 over 40–80M, +211 over 80–160M and +95 over
  160–240M.
* **The seed spread is 180 Elo at 240M.** That is larger than any lever measured on the recipe.

## Self-play balance, and what it cost

| run | 0–240M, by 5M (self-play USSR share) | peak |
|:---|:---|---:|
| E4-08-36 | .46 .65 .67 .58 .75 .85 .75 .72 .59 .57 .56 .72 .68 .67 .64 .64 .47 .69 .61 .55 .53 .67 .54 .53 .54 .45 .48 .41 .54 .55 .58 .68 .68 .61 .55 .57 .55 .48 .51 .59 .58 .59 .62 .57 .46 .52 .44 .52 | 0.85 |
| E4-08-37 | .56 .70 .83 .77 .78 .75 .80 .73 .85 .71 .79 .77 .82 .74 .72 .79 .91 .92 .88 .84 .83 .75 .76 .89 .77 .84 .91 .92 .89 .88 .85 .85 .90 .85 .77 .83 .73 .87 .87 .79 .88 .80 .71 .69 .81 .82 .80 .78 | 0.92 |

Neither run ever pinned (≥ 0.95).

**E4-08-37 leaned USSR for the whole run, and its flat stretches line up with its episodes.** It
gained only +8 over 80–120M, which contains the 0.91–0.92 episode at 80–90M, and +4 over
160–200M, after the 130–140M episode. It then gained +62 to 240M.

The weakness is on its US seat:
* against E4-08-36@240M it wins 38% as USSR but 16% as US;
* its 160M snapshot wins 8% as US against its own 240M.

On the production recipe, then, a chronic one-sided run **with no pin** can cost two stalls of
about 40M steps. That fits the owner's criterion ("delayed for a long time") without ever
touching 0.95. A pin threshold alone would not count it. A collapse measure for the recipe
should look at progress: per-seat Elo against the run's own earlier snapshots or frozen anchors.

This is one seed, so it shows what can happen, not how often.
