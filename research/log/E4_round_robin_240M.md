# E4 round robin at 240M — does the collapse show against a fixed reference?

**Measured 2026-09-19.** 21 entrants: the pooled arm `E4-02-01` at 20M–240M and the unpooled arm
`E4-01-01` at 20M–180M, every 20M. 210 pairs, 30 games per seat per pair, per-seat recorded.
`/workspace/data/p17/e4_final_round_robin/`.

The question is **not** whether the unpooled arm loses to a stronger model — it trained for fewer
steps, so it must, and an absolute rate against a stronger opponent is not evidence. The question
is whether a win rate against a **fixed** reference rises and then falls. That shape is a collapse;
a low flat line is only a weaker model.

## The headline: no clean rise-then-fall in either arm

Overall win rate against two independent fixed references:

| steps | pooled vs `E4-02-01@240M` | pooled vs `E4-01-01@120M` | unpooled vs `E4-02-01@240M` |
|---:|---:|---:|---:|
| 20M | 10.0% | 18.3% | 11.7% |
| 40M | 21.7% | 16.7% | 31.7% |
| 60M | 23.3% | 33.3% | 43.3% |
| 80M | 53.3% | 33.3% | 30.0% |
| 100M | 45.0% | 53.3% | 35.0% |
| 120M | 38.3% | 38.3% | 53.3% |
| 140M | 50.0% | 43.3% | 51.7% |
| 160M | 53.3% | 45.0% | 51.7% |
| 180M | 56.7% | 50.0% | 45.0% |
| 200M | **35.0%** | **31.7%** | — |
| 220M | 46.7% | 51.7% | — |

Both arms climb and then oscillate in a band. Neither shows the monotone decline that would make
"collapse" the right word for what the round robin can see.

## The one real event: a US-seat dip at 200M in the pooled arm

It reproduces against both references, which is what makes it more than noise:

| reference | 180M | **200M** | 220M |
|:---|---:|---:|---:|
| `E4-02-01@240M`, as US | 50.0% | **13.3%** | 36.7% |
| `E4-01-01@120M`, as US | 36.7% | **6.7%** | 43.3% |

The USSR seat is untouched across the same window (63.3% → 56.7% → 56.7%). So this is a
single-seat event, it is large — 6.7% against a 30-game denominator is roughly 4σ from 36.7% — and
it **recovers by 220M**. A transient, not a collapse, but the only thing in the run that looks like
the beginning of one.

## The seat asymmetry is structural, and belongs to both arms

I previously read the unpooled arm's weak US seat as its collapse signature. Against fixed
references that reading is wrong: **the USSR seat is stronger in both arms at every budget.**

```
pooled   vs E4-01-01@120M:  US 6.7-43.3%   USSR 23.3-76.7%
unpooled vs E4-01-01@120M:  US 3.3-36.7%   USSR 23.3-76.7%
```

That matches the ~63/37 self-play lean measured across the whole run and is the game's own
asymmetry, not a pathology. A seat gap is only evidence of collapse if it *widens* against a fixed
opponent, and here it does not.

## Self-play side split badly overstated the collapse

At 180M, `E4-01-01`'s self-play `us_win_rate` was ~0.02 — its US seat almost never beat its own
USSR seat. Against fixed external references the same checkpoint scored **23.3% and 20.0% as US**,
and 45%/40% overall. Both are true and they measure different things: the policy's USSR seat had
run far ahead of its own US seat, which is what self-play reports, while the US seat remained
ordinarily competent against opponents that were not its own overgrown mirror.

This is the concrete case for the owner's standing objection to side balance as a health metric.
A number that reads 0.006 where an external measurement reads 0.23 is not measuring strength.

## The unpooled arm's peak is the strongest model in the field

`E4-01-01@120M` ranks **#1 by Elo (1597.7)**, ahead of `E4-02-01@180M` (1594.3) and
`E4-02-01@240M` (1580.5). Whatever the pool is worth, it is not peak strength at this budget —
and a 240M pooled run did not beat a 120M unpooled one.

Two caveats before that is quoted. The field is dominated by snapshots of these two lineages, so
the Elo is internal to them and not a claim about absolute strength. And 30 games a seat gives
roughly ±9pp per cell, so ranks within ~40 Elo of each other are not separated.

## What this does not answer

**The catastrophic part is not in the field.** `E4-01-01`'s self-play US seat fell to 0.006 at
~184M, and its last snapshot is 180M. The round robin therefore stops just before the steep
section, which is why it sees an oscillation rather than a fall.

**Whether the 200M pooled dip means anything.** It recovered. Watching whether it recurs in a
longer or re-seeded run is the follow-up; one transient in one run is not a finding.
