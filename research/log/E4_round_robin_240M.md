# E4 round robin at 240M — does the collapse show against a fixed reference?

**Measured 2026-09-19.** 21 entrants: the pooled arm `E4-02-01` at 20M–240M and the unpooled arm
`E4-01-01` at 20M–180M, every 20M. 210 pairs, **100 games per seat** (200 per pair, 42,000 total),
per-seat recorded. `/workspace/data/p17/e4_final_round_robin_g100/`.

The question is **not** whether the unpooled arm loses to a stronger model — it trained for fewer
steps, so it must, and an absolute rate against a stronger opponent is not evidence. The question
is whether a win rate against a **fixed** reference rises and then falls.

> A first pass at 30 games per seat is kept at `.../e4_final_round_robin/`. Two of its readings did
> not survive the larger sample and are corrected below. At 30 games a cell carries about ±9pp,
> which is not enough to rank neighbours; at 100 it is about ±5pp.

## The finding: one sharp, isolated US-seat dip at 200M in the pooled arm

Against `E4-01-01@120M`, an independent reference from the *other* arm:

| steps | as US | as USSR | overall |
|---:|---:|---:|---:|
| 140M | 35.0% | 64.0% | 49.5% |
| 160M | 35.0% | 58.0% | 46.5% |
| 180M | 39.0% | 67.0% | 53.0% |
| **200M** | **11.0%** | 67.0% | **39.0%** |
| 220M | 30.0% | 56.0% | 43.0% |
| 240M | 37.0% | 68.0% | 52.5% |

The US seat falls from 39% to **11%** and back to 30%, while the **USSR seat does not move at all**
(67% → 67%). It is a single-seat event, roughly 6σ at this sample size, and it **fully recovers** —
240M is the strongest model in the field.

It replicates across both sample sizes and both references (30 games: 36.7% → 6.7% → 43.3%), which
is what makes it a finding rather than a wobble. But it recovers, so it is an *excursion*, not a
collapse. Whether this is the same mechanism as a real collapse, caught early and survived, is the
open question.

## The unpooled arm declines late, mildly

Against the same reference: 43.5% at 100M, 49.0% at 140M, **55.0% at 160M**, 47.0% at 180M. A peak
at 160M and an 8pp fall by 180M — right where its self-play `us_win_rate` was beginning to slide.
That is the shape the question was asking about, but it is one point and 8pp is only ~1.5σ here.

**The catastrophic part is not in the field.** `E4-01-01`'s self-play US seat reached 0.006 at
~184M and its last snapshot is 180M, so this stops just before the steep section.

## Corrections to the 30-game pass

**`E4-01-01@120M` does not top the field.** At 30 games it ranked #1 on 1597.7 Elo, ahead of both
pooled endpoints, and I wrote that the pool had not bought peak strength. At 100 games:

| rank | model | Elo |
|---:|:---|---:|
| 1 | **E4-02-01@240M** | 1600.7 |
| 2 | E4-02-01@180M | 1596.1 |
| 3 | E4-01-01@160M | 1583.7 |
| 4 | E4-01-01@120M | 1580.4 |

The pooled arm's final snapshot is first. The earlier ordering was noise, and the claim built on it
is withdrawn. Ranks within ~25 Elo are still not separated at this sample size, so 1–2 and 3–4 are
each ties.

**The pooled arm does not keep declining after 200M.** Measured against its own `@240M`, 220M reads
31.5% and looks like a continuing fall; against the independent reference it reads 43.0% and is
clearly a recovery. The self-lineage number is the artefact — comparing a snapshot to its own
near-neighbour endpoint measures how much changed in 20M steps, not strength. **Use a reference
from the other arm.**

## What holds from the first pass

**The seat asymmetry is structural and belongs to both arms.** The USSR seat is stronger at every
budget in both — pooled US 10–39% against USSR 15–68%, unpooled US 3–43% against USSR 17–67%. This
matches the ~63/37 self-play lean and is the game's own asymmetry. A seat gap is evidence of
collapse only if it *widens* against a fixed opponent.

**Self-play side split badly overstated the collapse.** At 180M `E4-01-01`'s self-play
`us_win_rate` read ~0.02, while the same checkpoint scored **31% as US** against a fixed external
reference and 47% overall. Both are true of different things: its USSR seat had run far ahead of
*its own* US seat, which is what self-play reports, while the US seat stayed ordinarily competent
against opponents that were not its own overgrown mirror. This is the concrete case against side
balance as a health metric.

## Open

- **Does the 200M excursion recur?** One transient in one run. A re-seeded or longer run would say.
- **What does the steep section look like?** Needs the unpooled arm re-run past 184M with snapshots
  kept.
