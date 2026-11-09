# Catastrophic-blunder census: swa_13520-13600M.pt

* model `swa_13520-13600M.pt` (sha256 `243ec4b375c5...`), screen search `gumbel:32:4`, validation Gumbel k=8 at 256 simulations, continuations `greedy`, 64 pairs, screen seed 0, validation seed 0.

## Stage 1 -- screening

* 175004 decisions with two or more legal moves in 400 self-play games; 51911 triggered the cheap filter and 6147 control positions (rate 0.05, weight 20.02) did not.
* triggers: forced-loss 44, missed-forced-win 56, g32-disagrees 51884.

## Stage 2 -- validation

* 58058 positions validated: 1669 candidates and 110 control positions confirmed as catastrophic, 689 candidates refuted, 49553 unresolved.

## The figures

| figure | value |
|:---|---:|
| candidate count | 51911 |
| confirmation rate (candidates) | 3.2% |
| confirmed catastrophic blunders per 1,000 decisions | 22.12 |
| &nbsp;&nbsp;-- exact terminal mistakes (engine-proven) | 0.57 |
| &nbsp;&nbsp;-- estimated strategic regret (playouts, upper bound) | 21.55 |
| share of confirmed mistakes the cheap screening found | 43.1% |
| &nbsp;&nbsp;-- exact terminal mistakes | 100.0% |
| &nbsp;&nbsp;-- estimated strategic regret | 41.6% |
| share rescued by 256-simulation search | 58.6% |
| share unresolved even under the stronger search | 95.5% |

Exact terminal mistakes (the engine's own labels) are counted separately from the strategic regret the paired continuations estimate; the per-1,000 figure sums both and adds the control sample's confirmed mistakes at their weight, standing in for the positions the screening never looked at.

**Noise floor.** The estimated class is an upper bound. With 4809 control rows carrying an alternative to compare, 110 clear the confirmation bar (z >= 2 on the best of their alternatives, 2.3%), against 1586 of 37769 candidates (4.2%). A max over a handful of paired comparisons clears z >= 2 by chance alone a few percent of the time, so the control's confirmed rows are consistent with noise plus a few real misses; the exact class and the candidates' excess over the control are the signals.

### missed a forced win (56 in the census, 3 shown)

| id | side | turn/ar | decision | greedy | search | verdict | regret (SE) | replay |
|:---|:---|:---|:---|:---|:---|:---|:---|:---|
| `deee4b6b43ec0e05` | USSR | 10/0 | headline | Missile Envy | Missile Envy | exact-missed-win | +0.219 (0.052) | game seed 183, decision 521, pos `deee4b6b43ec0e05` |
| `9e0118e785c2fc9c` | USSR | 8/0 | headline | Iranian Hostage Crisis | Iranian Hostage Crisis | exact-missed-win | +0.156 (0.051) | game seed 33, decision 392, pos `9e0118e785c2fc9c` |
| `fe60e45e0601359e` | US | 8/1 | card | “Ask Not What Your Country Can Do For You…” | “Ask Not What Your Country Can Do For You…” | exact-missed-win | +0.125 (0.042) | game seed 278, decision 406, pos `fe60e45e0601359e` |

### took a forced loss (VP threshold / scoring card / coup line) (43 in the census, 3 shown)

| id | side | turn/ar | decision | greedy | search | verdict | regret (SE) | replay |
|:---|:---|:---|:---|:---|:---|:---|:---|:---|
| `12db1ca36b1e2e47` | US | 6/7 | card | Central America Scoring | The China Card | exact-forced-loss | +0.102 (0.025) | game seed 100, decision 350, pos `12db1ca36b1e2e47` |
| `24f9d16749227607` | US | 5/7 | card | Africa Scoring | Panama Canal Returned | exact-forced-loss | +0.070 (0.022) | game seed 155, decision 277, pos `24f9d16749227607` |
| `9b161910c9a5530a` | US | 5/7 | card | Central America Scoring | “Lone Gunman” | exact-forced-loss | +0.062 (0.021) | game seed 105, decision 275, pos `9b161910c9a5530a` |

### played Duck and Cover at DEFCON 2 (its event degrades DEFCON) (1 in the census, 1 shown)

| id | side | turn/ar | decision | greedy | search | verdict | regret (SE) | replay |
|:---|:---|:---|:---|:---|:---|:---|:---|:---|
| `c20b1fd2e2d7000d` | USSR | 8/6 | select_play_mode | Ops: influence | Ops: realign | exact-forced-loss | +0.000 (0.000) | game seed 92, decision 450, pos `c20b1fd2e2d7000d` |

### strategic regret, confirmed by searched continuations (card) (626 in the census, 3 shown)

| id | side | turn/ar | decision | greedy | search | verdict | regret (SE) | replay |
|:---|:---|:---|:---|:---|:---|:---|:---|:---|
| `25b378b1e11b51f0` | US | 5/3 | card | Quagmire | “Lone Gunman” | confirmed-regret | +0.961 (0.023) | game seed 303, decision 266, pos `25b378b1e11b51f0` |
| `72b47d7fa4a9fa1a` | US | 8/3 | card | East European Unrest | “Lone Gunman” | confirmed-regret | +0.906 (0.037) | game seed 151, decision 404, pos `72b47d7fa4a9fa1a` |
| `de057618cbe5e7bb` | US | 8/5 | card | De Gaulle Leads France | “Lone Gunman” | confirmed-regret | +0.891 (0.039) | game seed 151, decision 422, pos `de057618cbe5e7bb` |

### strategic regret, confirmed by searched continuations (point_node) (739 in the census, 3 shown)

| id | side | turn/ar | decision | greedy | search | verdict | regret (SE) | replay |
|:---|:---|:---|:---|:---|:---|:---|:---|:---|
| `b1e6cd1839d56dc9` | US | 7/2 | point_node | PointNode #20 (Bulgaria) | PointNode #18 (Yugoslavia) | confirmed-regret | +0.953 (0.027) | game seed 251, decision 388, pos `b1e6cd1839d56dc9` |
| `701e98b429383936` | US | 7/2 | point_node | PointNode #74 (Colombia) | PointNode #18 (Yugoslavia) | confirmed-regret | +0.922 (0.034) | game seed 251, decision 387, pos `701e98b429383936` |
| `f4b6fb23f6d2700b` | US | 10/6 | point_node | PointNode #39 (Indonesia) | PointNode #63 (South Africa) | confirmed-regret | +0.797 (0.051) | game seed 104, decision 557, pos `f4b6fb23f6d2700b` |

### strategic regret, confirmed by searched continuations (select_play_mode) (211 in the census, 2 shown)

| id | side | turn/ar | decision | greedy | search | verdict | regret (SE) | replay |
|:---|:---|:---|:---|:---|:---|:---|:---|:---|
| `c009506ca5f03a09` | USSR | 10/7 | select_play_mode | Ops: coup | Ops: influence | confirmed-regret | +0.844 (0.046) | game seed 74, decision 579, pos `c009506ca5f03a09` |
| `e2ea9b209df1bae7` | USSR | 8/1 | select_play_mode | event | space | confirmed-regret | +0.625 (0.061) | game seed 44, decision 396, pos `e2ea9b209df1bae7` |

### strategic regret, confirmed by searched continuations (select_op_mode) (32 in the census, 2 shown)

| id | side | turn/ar | decision | greedy | search | verdict | regret (SE) | replay |
|:---|:---|:---|:---|:---|:---|:---|:---|:---|
| `9b34916b18c931c6` | USSR | 10/7 | select_op_mode | Ops: influence | Ops: realign | confirmed-regret | +0.703 (0.058) | game seed 329, decision 584, pos `9b34916b18c931c6` |
| `0435b070db9ff8d9` | USSR | 10/4 | select_op_mode | Ops: realign | Ops: realign | confirmed-regret | +0.367 (0.097) | game seed 159, decision 560, pos `0435b070db9ff8d9` |

### strategic regret, confirmed by searched continuations (headline) (70 in the census, 2 shown)

| id | side | turn/ar | decision | greedy | search | verdict | regret (SE) | replay |
|:---|:---|:---|:---|:---|:---|:---|:---|:---|
| `a82b73b13812b8c3` | USSR | 8/0 | headline | Colonial Rear Guards | Africa Scoring | confirmed-regret | +0.336 (0.085) | game seed 65, decision 409, pos `a82b73b13812b8c3` |
| `7690651a92083332` | US | 3/0 | headline | Asia Scoring | Containment | confirmed-regret | +0.281 (0.085) | game seed 93, decision 109, pos `7690651a92083332` |

### strategic regret, confirmed by searched continuations (event-choice) (1 in the census, 1 shown)

| id | side | turn/ar | decision | greedy | search | verdict | regret (SE) | replay |
|:---|:---|:---|:---|:---|:---|:---|:---|:---|
| `915c67a95d101206` | US | 9/5 | event-choice | Region #1 (Asia) | Region #5 (South America) | confirmed-regret | +0.062 (0.030) | game seed 395, decision 479, pos `915c67a95d101206` |

Every failure's exact board is its `pos=` token in the screening JSONL (`pos`), openable in the workbench (`?pos=<token>`) and regenerable from its `game.seed` and `decision`.

## Runtime

* stage 1 (screening): 100 games in 162.1s local wall (screen_blunder/summary-part-1.json); 100 games in 130.3s local wall (screen_blunder/summary-part-2.json); 100 games in 164.3s local wall (screen_blunder/summary-part-3.json); 100 games in 118.8s local wall (screen_blunder/summary-part-4.json); 
