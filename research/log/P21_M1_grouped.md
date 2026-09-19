# P21 rung M1 — grouped input projections

**2026-09-19.** M1 gives board, card and global their own dense projection instead of reading all
3,824 floats through one layer. No weight sharing, no tokens, position preserved in both, same
trunk width (480) and matched parameters (3.148M against the anchor's 3.149M), so the rung
isolates exactly one thing: **does forbidding board↔card mixing in the first layer help?**

All numbers from one tournament, `data/reports/P21_M1_complete.md`, 100 games per side, τ=0.0.

| arm | Elo | steps/s | USSR all | US all | gap |
|:---|---:|---:|---:|---:|---:|
| `E4-03-01@80M` — anchor | 2170.0 | 11,732 | 91.0% | 92.8% | **−1.7** |
| `M1@160M` | 1863.4 | 59,943 | 70.4% | 55.8% | +14.6 |
| `M0@160M` | 1819.6 | 60,814 | 68.6% | 46.2% | +22.4 |
| `M1@80M` seed 2 | 1791.6 | 59,943 | 58.9% | 48.0% | +10.9 |
| `M1@80M` seed 1 | 1769.0 | 59,561 | 50.7% | 49.2% | +1.5 |
| `E4-04-01@80M` — defaults | 1719.1 | 14,057 | 48.6% | 39.2% | +9.4 |
| `M0@80M` seed 1 | 1667.2 | 63,187 | 49.9% | 23.1% | +26.8 |
| `M0@80M` seed 2 | 1666.0 | 60,814 | 49.8% | 23.4% | +26.4 |
| `HeuristicBot` | 1500.0 | — | 19.9% | 14.5% | +5.4 |

## Adopted

**+101.8 Elo on seed 1, +125.6 on seed 2**, against M1's own seed spread of 22.6 and M0's of 1.2.
Both seeds, margin far above spread. Head to head at 80M, M1 takes 70.0% as USSR and 51.0% as US.

The compute gate does not bite: M1 runs 59,561–59,943 steps/s against M0's 60,814–63,187, a 3–6%
difference, under the 10% threshold. No parity re-run needed.

## The gain is one seat, not two

| at 80M | USSR | US |
|:---|---:|---:|
| M0 | 49.9% | **23.1%** |
| M1 | 50.7% | **49.2%** |

**USSR is unchanged (+0.8 pp). US gains +26 pp.** M0 is close to a one-seat player and grouping
the input repaired the weak seat rather than lifting both. "+102 Elo" reports none of this, which
is why the standing table carries per-seat columns.

Note the against-columns say something different and weaker: M1 beats M0 "70% as USSR", but that
is largely M0 being poor *at US*. A rate against one opponent conflates the arm's skill in a seat
with the opponent's weakness in the other; the field-averaged `USSR all` / `US all` columns do not.

## Intercept, not slope — and M0 is closing

| | 80M | 160M | within-seed slope |
|:---|---:|---:|---:|
| M0 | 1667 | 1820 | **+147** |
| M1 | 1792 | 1863 | **+72** |
| M1 − M0 | **+102** | **+44** | |

M1's advantage **halves between 80M and 160M**, because its slope is half M0's. Extrapolating one
further doubling puts them level near 320M.

Treat that extrapolation as indicative only — two points per slope, one seed each, and slope over
three doublings is not something this record has ever been able to trust. But the *direction* is
measured, not inferred, and it is exactly the distinction the 80M+160M protocol exists to make.
**A single 80M comparison would have recorded "M1 wins by 102" with no hint that the margin
halves.**

This is the first evidence that the ladder's 80M readings may systematically flatter
early-training effects. If M2 shows the same shrinking margin, that is a pattern rather than a
quirk of this rung, and the adoption rule should be revisited before more rungs are built on 80M
alone.

## Side balance is seed-dependent here

M1's two 80M seeds sit at **+1.5 pp and +10.9 pp**. M0's two agree closely (+26.8, +26.4). So M1's
balance is not yet a stable property of the rung, and its 160M arm drifts further to +14.6. One
seed cannot settle the balance story even where it settles the Elo.

## Health

Both arms clean: pool to capacity 12 and held, `kl_div` peaking at 0.062, no alarm across 1,237
and 2,474 iterations.

## Standing

| rung | Elo @80M | slope | side gap @80M | verdict |
|:---|---:|---:|---:|:---|
| M0 flat MLP | 1667 / 1666 | +147 | +27 pp | floor |
| **M1 grouped** | **1769 / 1792** | **+72** | +1.5 / +10.9 pp | **adopted** |
| anchor | 2170 | — | −1.7 pp | target |

Both rungs now beat the E4 default architecture (1719) while running **4.3x faster** than it.
