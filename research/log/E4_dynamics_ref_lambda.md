# E4-26 / E4-27 — a slower π_ref and a longer λ, phase 1: both lose on seed 3

Two one-flag arms on M2d, each against its seed-matched M2d arm, chosen because the setup-critic
readout ([`P21_M2d_setup_west_germany.md`](P21_M2d_setup_west_germany.md)) and the self-play side
balance both show the training signal oscillating rather than the network lacking information.

| arm | change | directory |
|:---|:---|:---|
| **E4-26-03** | `--ref-update-freq 5000000` (default 200,000): π_ref refreshed 25x less often | `E4-26-03_20260922_230829` |
| **E4-27-03** | `--gae-lambda 0.99` (default 0.98): credit horizon ~100 decisions instead of ~50 | `E4-27-03_20260922_235853` |

Seed 3, 80M, flags otherwise identical to `E4-08-03` (`launch_flags.py --diff` shows the one
change, plus `--resume-every-steps` and the explicitly recorded seed streams). `--gae-lambda` was
added for this (`4f8b492`); it was hard-coded at 0.98 before. Phase 2, seed 5 from scratch to
160M, is running (`E4-26-05`, `E4-27-05`).

## Strength

`data/reports/E4_dynamics_phase1.{md,json}`. 100 games per seat per pair, temperature 0. Win rates
are the row's **as USSR / as US**.

| # | model | Elo | vs E4-08-03@80M | vs E4-08-03@160M | vs anchor E4-03-01@80M |
|---:|:---|---:|---:|---:|---:|
| 1 | E4-08-03@240M | 2285.1 | 66 / 74 | 61 / 53 | 62 / 81 |
| 2 | E4-08-03@160M | 2242.8 | 68 / 66 | — | 70 / 69 |
| 3 | E4-03-01@80M | 2133.6 | 42 / 60 | 31 / 30 | — |
| 4 | **E4-08-03@80M** | **2118.6** | — | 34 / 32 | 40 / 58 |
| 5 | E4-08-03@60M | 1957.8 | 27 / 22 | 11 / 8 | 18 / 35 |
| 6 | **E4-26-03@80M** | **1950.6** | 30 / 33 | 16 / 10 | 26 / 27 |
| 7 | **E4-27-03@80M** | **1876.6** | 21 / 15 | 27 / 7 | 22 / 7 |
| 8 | E4-26-03@60M | 1802.7 | 13 / 9 | 8 / 2 | 5 / 23 |
| 9 | E4-26-03@40M | 1726.4 | 8 / 12 | 7 / 1 | 8 / 11 |
| 10 | E4-08-03@40M | 1724.6 | 7 / 6 | 10 / 3 | 8 / 7 |
| 11 | E4-27-03@60M | 1697.2 | 23 / 9 | 8 / 7 | 8 / 6 |
| 12 | E4-27-03@40M | 1621.3 | 11 / 2 | 12 / 0 | 6 / 0 |
| 13 | HeuristicBot | 1500.0 | 8 / 1 | 6 / 0 | 1 / 1 |

Against seed-matched M2d at matched steps:

| | 40M | 60M | 80M |
|:---|---:|---:|---:|
| E4-26-03, slow π_ref | +1.8 | −155.1 | **−168.0** |
| E4-27-03, λ 0.99 | −103.3 | −260.6 | **−242.0** |

**Both lose at 80M on both seats**, so neither can pass the adoption rule (a win at 80M on both
seeds) whatever seed 5 does. The losses are larger than M2d's ~100 Elo seed spread and consistent
across two snapshots each.

## Stability — what each was meant to fix

`us_win_rate` in self-play, averaged into 1M-step bins over 20–80M:

| arm | sd | range | mean \|Δ\| between bins | mean entropy |
|:---|---:|:---|---:|---:|
| E4-08-03 (control) | 0.160 | 0.04–0.61 | 0.069 | 1.369 |
| E4-26-03, slow π_ref | 0.122 | 0.20–0.72 | 0.105 | 1.566 |
| E4-27-03, λ 0.99 | 0.084 | 0.02–0.37 | 0.029 | 1.557 |

The setup-critic probe (`ai/eval/setup_critic.py`, reports `data/reports/E4-2{6,7}-03_setup_critic.json`):

* **E4-26-03** prefers the no-West-Germany setup on 0/8 deals at every snapshot from 20M, 2 sign
  flips in 5–80M against E4-08-03's 4. But the post-setup value still swings from −0.90 (40M) to
  +0.74 (60M).
* **E4-27-03** prefers West Germany from 5M to 70M, then flips hard at 75–80M (−0.68, −0.95).

**Neither fix is stability of the useful kind.**

* E4-26-03's side balance never falls below 0.20, but it moves between bins faster than the
  control, and entropy runs ~0.2 higher throughout. The ratchet mechanism proposed in
  [`../findings/training/entropy_inflation.md`](../findings/training/entropy_inflation.md)
  predicted the opposite: a slower anchor should *lower* entropy. That is evidence against it.
* E4-27-03's low sd is **one-sidedness, not stability**: the US won 2–4% of self-play games from
  35M to 80M. The tournament confirms it is a real US deficit. It wins 7% as US against the anchor
  and 15% against E4-08-03@80M, while its USSR seat is merely behind. `adv_std_raw` stayed at
  0.12–0.20 throughout, above the collapsed band (0.004–0.031), so the collapse detector did not
  and should not call it; this is a third shape, a sustained one-sided state with a live signal.

## Seed 5 at 80M: the slow anchor loses there too

`E4-26-05` passed 80M on its way to 160M and was rated in its own field
(`data/reports/E4_dynamics_s5_80M.{md,json}`):

| # | model | Elo | vs E4-08-05@80M | vs anchor E4-03-01@80M |
|---:|:---|---:|---:|---:|
| 1 | E4-08-05@160M | 2223.4 | 74 / 59 | 60 / 63 |
| 2 | E4-03-01@80M | 2173.6 | 52 / 65 | — |
| 3 | **E4-08-05@80M** | **2071.7** | — | 35 / 48 |
| 4 | **E4-26-05@80M** | **2003.5** | 47 / 40 | 14 / 31 |
| 5 | E4-26-05@40M | 1819.0 | 32 / 19 | 4 / 9 |
| 6 | HeuristicBot | 1500.0 | 6 / 0 | 1 / 1 |

**−68.2 Elo at 80M, losing both seats.** With seed 3's −168.0 the slow anchor loses at 80M on
both seeds, which rejects it under the adoption rule outright. Only its 160M snapshot is still to
come.

## Reading

Both levers hurt at 80M, and the one-flag framing makes that a clean statement about this seed.
The more useful result is negative in a narrower way. The oscillation measured in the setup
critic is not fixed by slowing the reference or lengthening the credit horizon. Slowing the
reference trades it for higher entropy; lengthening the horizon trades it for a stuck seat.

Seed 5 to 160M decides only whether either catches up late. A slow anchor that learns more slowly
early and ends higher is the one outcome still open.
