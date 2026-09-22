# M2d s3 160M→240M — the US setup drops West Germany, and flips back and forth

Observed in self-play of `E4-08-03@240M`. At setup the US leaves West Germany empty in every game,
putting the influence into Canada, Italy and France instead. `E4-08-03@160M` stacked 4 influence there.

## Measurement

Every 5M snapshot from 160M to 240M (`E4-08-03_20260921_023139` for 160M,
`E4-08-03_20260922_202204` for the rest), self-play at temperature 0.1 with `--trace`, seeds
101–105. `tools/play_match.py`; replays in `data/replays/wg_scan/`. "Max P(WG)" is the highest
probability the policy gave West Germany at any of the US setup placements.

| steps | WG influence placed per game (seeds 101–105) | max P(WG) at any placement | typical US setup |
|---:|:---|:---|:---|
| 160M | 4 4 4 4 4 | 1.000 1.000 0.999 1.000 1.000 | West Germany 4.0, France 3.0, Italy 2.0 |
| 165M | 2 2 2 2 2 | 1.000 0.999 0.999 1.000 0.999 | France 3.0, Italy 2.0, West Germany 2.0, South Korea 2.0 |
| 170M | 4 4 4 4 4 | 1.000 1.000 1.000 1.000 1.000 | West Germany 4.0, France 3.0, Italy 2.0 |
| 175M | 4 4 2 4 4 | 0.998 0.998 0.999 0.997 0.998 | West Germany 3.6, France 3.0, Italy 2.0, South Korea 0.4 |
| 180M | 4 4 0 4 4 | 0.999 1.000 0.082 1.000 1.000 | West Germany 3.2, France 3.0, Italy 2.0, Canada 0.4, Iran 0.2 |
| **185M** | **0 0 0 0 0** | 0.000 0.000 0.012 0.001 0.004 | France 2.4, Italy 2.0, Canada 2.0, Iran 1.0, South Korea 1.0 |
| 190M | 0 0 0 0 0 | 0.079 0.109 0.114 0.072 0.111 | France 3.0, Italy 2.0, Canada 2.0, South Korea 2.0 |
| 195M | 0 0 0 0 0 | 0.024 0.034 0.076 0.030 0.039 | France 3.0, Italy 2.0, Canada 2.0, South Korea 2.0 |
| 200M | 0 0 0 0 0 | 0.056 0.056 0.068 0.055 0.055 | France 3.0, Italy 2.0, Canada 2.0, South Korea 2.0 |
| 205M | 0 0 0 0 0 | 0.023 0.033 0.052 0.035 0.032 | France 3.0, Italy 2.0, Canada 2.0, South Korea 2.0 |
| 210M | 0 0 0 0 0 | 0.010 0.008 0.012 0.012 0.003 | France 3.0, Italy 2.0, Canada 2.0, South Korea 2.0 |
| 215M | 0 0 0 0 0 | 0.050 0.052 0.056 0.051 0.053 | France 3.0, Italy 2.0, Canada 2.0, Iran 1.0, Panama 1.0 |
| 220M | 0 0 0 0 0 | 0.011 0.032 0.066 0.011 0.049 | Italy 2.0, Canada 2.0, Turkey 2.0, Iran 1.4, France 1.0 |
| 225M | 0 0 0 0 0 | 0.034 0.051 0.055 0.025 0.012 | Italy 2.0, Canada 2.0, Turkey 2.0, France 1.0, Iran 1.0 |
| **230M** | **4 4 4 4 4** | 0.997 0.998 0.997 0.997 0.998 | West Germany 4.0, Italy 2.0, South Korea 1.6, France 1.0, Iran 0.4 |
| 235M | 0 0 0 0 0 | 0.003 0.001 0.018 0.002 0.001 | France 3.0, Canada 2.0, Italy 2.0, Iran 1.0, South Korea 1.0 |
| 240M | 0 0 0 0 0 | 0.003 0.003 0.001 0.004 0.001 | France 3.2, Canada 2.2, Italy 2.0, Iran 1.0, South Korea 0.6 |

In all five 240M games of this scan, the USSR targets West Germany on T1 (AR1 in four, AR2 in one).

## Reading

* **The switch happens between 180M and 185M,** after partial wobbles at 165M and 175–180M. From
  185M the US never places in West Germany, except at 230M, which fully reverts for one snapshot.
* **It is not indecision.** At every snapshot the policy is near-certain: P(WG) is either ≥0.997
  or ≤0.11. The opening is swapped wholesale between snapshots 5M steps apart, not drifted.
* **Hypothesis, unverified:** setup is the decision furthest from any reward (about 400 plies to
  the end of the game), so its advantage signal is the weakest in the game. A strongly committed
  choice with almost no gradient behind it is free to follow whatever shared-trunk changes push
  it. The 230M reversion fits that reading better than a learned preference.
* **Its cost is unmeasured.** `E4-08-03@240M` is still the top-rated checkpoint
  ([`P21_full_field.md`](P21_full_field.md)), but every opponent in that field is from the same
  lineage. Nothing here shows whether an opponent that contests West Germany well would punish the
  open setup.

**Side note:** the scan's 240M games with seeds 101–105 have different outcomes from the
`m2d_240M_selfplay_*` games with the same seeds. At temperature 0.1 the same `--seed` does not
reproduce a game (not yet traced whether the deals or the sampled actions differ), so these
replays cannot be reproduced from the seed alone.
