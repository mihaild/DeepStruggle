# E3-20-28 @200M against itself — the strong anchor's own side split

**Measured 2026-09-17**, when `@200M` replaced `@160M` as P15's strong anchor
([`../plans/P15_breaking_the_cycle.md`](../plans/P15_breaking_the_cycle.md) X0). An anchor that
is itself lopsided would put its own bias into every number measured against it, so this asks the
narrow question: played against a copy of itself, does either seat win more?

500 games, 250 with each copy holding the US seat. Same weights on both sides, staged under two
names — `--self-play` lists one model twice under one name, which enters a single player twice in
the Bradley-Terry fit.

## The split

| | games | share |
|:---|---:|---:|
| **US wins** | 255 | **51.0%** |
| **USSR wins** | 243 | **48.6%** |
| draws | 2 | 0.4% |

Side gap **+2.4 pp** toward the US. 95% CI on the US rate is **46.6% – 55.4%**, which contains
50%: **no detectable side bias.** With 500 games the standard error is 2.2 pp, so a bias smaller
than about ±4.4 pp could not have been seen here — this rules out a large tilt, not a small one.

Consistency check, since both seats are the same weights: the A-copy won 52.0% of the 250 games it
played as the US and 49.2% of the 250 it played as the USSR. The two agree within noise, as they
must.

## Why it matters for the anchor choice

`@200M` is the peak of the pooled lineage (2193.7 Elo in the X0 round robin, against 2141.9 for
`@160M`) **and** it is even-handed against itself. Those are the two properties an anchor wants:
strong enough not to saturate, and unbiased enough that `anchor_side_gap` measures the *arm's*
tilt rather than the anchor's.

It also lines up with the X0 sweep, where `@200M` produced the most balanced cell in the whole
table — a −1.5 pp gap against the `@80M` anchor, against −20.5 at 240M and −48.0 at 280M. The
lineage passes through balance at 200M and runs away from it afterwards; the anchor is taken at
the point of passage.

## What this does not say

Self-play balance is **not** evidence of strength or of general even-handedness — an arm can be
balanced against itself and lopsided against everyone else, which is exactly why the 4×4
experiment's self-play endpoint was abandoned
([`../../../method/measurement_pitfalls.md`](../../../method/measurement_pitfalls.md)). Against the whole
24-model X0 field the same checkpoint reads 76.6% as USSR against 69.0% as US, a +7.6 pp tilt the
*other* way. Both numbers are correct and they answer different questions. The claim here is only
the one needed for the anchor: it does not hand a systematic advantage to whichever seat an
opponent is given.
