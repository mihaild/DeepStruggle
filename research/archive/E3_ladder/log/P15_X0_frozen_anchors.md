# P15-X0 — the frozen-anchor measurement

**Status:** measured 2026-09-16. The instrument P15 calls a prerequisite for everything else.

The problem it solves: **no live training metric can rate an arm after ~120M.** Every arm
beats `HeuristicBot` above 89% and `RandomBot` above 99% by then, so the anchors that exist
in-run are saturated and separate nothing
([`../../../log/measurement_bugs.md`](../../../log/measurement_bugs.md)). A *frozen peer* does not
saturate, and it does not move, so a win rate against it is comparable across budgets and
across runs.

## The anchors

`E3-20-28 @80M` (mid) and `E3-20-28 @160M` (strong) — a pooled arm, the strongest available
reference. Both are frozen: every number below is measured against the same two opponents.

## How to read a cell

Each cell is **the snapshot's own win rate against that anchor**, as `win% playing USSR /
win% playing US`, over 200 games a side. The anchor's score is the complement, so nothing is
lost by stating it from the snapshot's side. `—` is the anchor against itself.

A cell near `50 / 50` means the snapshot is the anchor's equal and even-handed. The gap
**between** the two numbers in a cell is external side balance against a fixed opponent —
the endpoint [`../findings/pooling.md`](../findings/pooling.md) says
nothing had computed, now available at every
budget rather than at a single final checkpoint.

## E3-20-28 (pooled)

| steps | vs anchor @80M | vs anchor @160M | Elo |
|---:|:---|:---|---:|
| 40M | 35.0% / 22.5% | 23.5% / 10.5% | 1915.3 |
| 80M | — | 28.5% / 44.0% | 2083.1 |
| 120M | 57.0% / 69.5% | 38.0% / 51.0% | 2133.3 |
| 160M | 56.0% / 71.0% | — | 2141.9 |
| 200M | 69.0% / 70.5% | 60.0% / 58.5% | 2193.7 |
| 240M | 50.5% / 71.0% | 46.0% / 69.5% | 2168.0 |
| 280M | 36.0% / 84.0% | 25.5% / 82.0% | 2126.4 |
| 320M | 39.0% / 80.5% | 29.0% / 78.0% | 2138.0 |

## E3-20-29 (pooled, weak seed)

| steps | vs anchor @80M | vs anchor @160M | Elo |
|---:|:---|:---|---:|
| 40M | 8.5% / 46.5% | 10.5% / 34.0% | 1930.9 |
| 80M | 22.0% / 39.0% | 18.0% / 37.0% | 1952.8 |
| 120M | 4.5% / 17.0% | 2.5% / 18.0% | 1723.0 |
| 160M | 34.5% / 42.0% | 19.5% / 29.5% | 1992.4 |
| 200M | 35.0% / 55.5% | 35.0% / 40.5% | 2057.9 |
| 240M | 12.5% / 22.5% | 11.5% / 13.5% | 1840.6 |
| 280M | 16.0% / 16.0% | 15.0% / 10.0% | 1851.3 |
| 320M | 24.0% / 29.5% | 15.5% / 18.5% | 1905.9 |

## E3-17-26 (no pool)

| steps | vs anchor @80M | vs anchor @160M | Elo |
|---:|:---|:---|---:|
| 40M | 13.5% / 45.0% | 12.5% / 28.0% | 1923.4 |
| 80M | 59.5% / 66.5% | 46.0% / 58.5% | 2172.6 |
| 120M | 67.5% / 74.0% | 54.0% / 66.0% | 2188.8 |
| 160M | 41.0% / 35.0% | 44.5% / 34.0% | 2031.1 |
| 200M | 44.5% / 23.5% | 50.0% / 17.5% | 2020.1 |
| 240M | 47.0% / 24.5% | 49.0% / 21.0% | 2010.3 |
| 280M | 24.0% / 41.5% | 16.5% / 37.5% | 1958.2 |
| 320M | 49.0% / 27.5% | 47.0% / 21.5% | 2034.3 |

## The combined table

Same numbers, one grid: three pairs of columns, one pair per run.

| steps | p28 @80M | p28 @160M | p29 @80M | p29 @160M | n26 @80M | n26 @160M |
|---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 40M | 35.0% / 22.5% | 23.5% / 10.5% | 8.5% / 46.5% | 10.5% / 34.0% | 13.5% / 45.0% | 12.5% / 28.0% |
| 80M | — | 28.5% / 44.0% | 22.0% / 39.0% | 18.0% / 37.0% | 59.5% / 66.5% | 46.0% / 58.5% |
| 120M | 57.0% / 69.5% | 38.0% / 51.0% | 4.5% / 17.0% | 2.5% / 18.0% | 67.5% / 74.0% | 54.0% / 66.0% |
| 160M | 56.0% / 71.0% | — | 34.5% / 42.0% | 19.5% / 29.5% | 41.0% / 35.0% | 44.5% / 34.0% |
| 200M | 69.0% / 70.5% | 60.0% / 58.5% | 35.0% / 55.5% | 35.0% / 40.5% | 44.5% / 23.5% | 50.0% / 17.5% |
| 240M | 50.5% / 71.0% | 46.0% / 69.5% | 12.5% / 22.5% | 11.5% / 13.5% | 47.0% / 24.5% | 49.0% / 21.0% |
| 280M | 36.0% / 84.0% | 25.5% / 82.0% | 16.0% / 16.0% | 15.0% / 10.0% | 24.0% / 41.5% | 16.5% / 37.5% |
| 320M | 39.0% / 80.5% | 29.0% / 78.0% | 24.0% / 29.5% | 15.5% / 18.5% | 49.0% / 27.5% | 47.0% / 21.5% |

## What it says

### 1. Nothing improves past ~200M, and the unpooled arm peaks at 120M

Elo across all 24 models in one field ([`P15_X0_round_robin.md`](P15_X0_round_robin.md)):

| run | best snapshot | Elo | at 320M | lost after the peak |
|:---|:---|---:|---:|---:|
| `p28` E3-20-28 (pooled) | **200M** | 2193.7 | 2138.0 | −55.7 |
| `n26` E3-17-26 (no pool) | **120M** | 2188.8 | 2034.3 | −154.5 |
| `p29` E3-20-29 (pooled, weak seed) | **200M** | 2057.9 | 1905.9 | −152.0 |

This sharpens P15's premise. "No arm progresses after ~160M" is close but generous: the unpooled
arm's peak is at **120M** and it gives back 154 Elo over the next 200M of training. The pooled arm
holds longer — peak at 200M, −56 by 320M — which is the same shape as
[`../findings/pooling.md`](../findings/pooling.md) §3a, extended past the budget
that finding could see.

### 2. The oscillation, finally with an amplitude

Side gap = win% playing USSR − win% playing US, against the frozen `@80M` anchor. Negative means
the snapshot is stronger as the US.

| steps | `p28` | `p29` | `n26` |
|---:|---:|---:|---:|
| 40M | +12.5 | −38.0 | −31.5 |
| 80M | — | −17.0 | −7.0 |
| 120M | −12.5 | −12.5 | −6.5 |
| 160M | −15.0 | −7.5 | **+6.0** |
| 200M | −1.5 | −20.5 | **+21.0** |
| 240M | −20.5 | −10.0 | **+22.5** |
| 280M | **−48.0** | +0.0 | **−17.5** |
| 320M | **−41.5** | −5.5 | **+21.5** |

| run | range | amplitude | sign flips |
|:---|:---|---:|---:|
| `p28` | −48.0 … +12.5 | **60.5 pp** | 1 |
| `p29` | −38.0 … +0.0 | 38.0 pp | 0 |
| `n26` | −31.5 … +22.5 | **54.0 pp** | **3** |

**The two arms fail differently, and the distinction matters for which remedy applies.**

`n26`, unpooled, **oscillates**: three sign flips, swinging from −31.5 to +22.5 to −17.5 to +21.5.
That is best-response cycling as P15 describes it — the imbalance does not converge, it orbits.

`p28`, pooled, does **not** oscillate late. It is balanced at 200M (−1.5, the most even cell in
the table) and then **runs away monotonically** toward the US: −1.5, −20.5, −48.0, −41.5. One
direction, no return. A pool of past selves damps the orbit and then loses to a drift, which is a
different failure and is not what "oscillates rather than converges" predicts.

### 3. A field-averaged side gap hides most of this

`arena_320` rates `P28@320M` at a side gap of **−6.9 pp**, measured across that tournament's whole
field. Against a *fixed* anchor the same checkpoint reads **−41.5 pp**. Both are correct; they are
different questions. Averaging over a field that contains the arm's own relatives lets a shared
bias cancel, because every opponent drifted too. A frozen peer cannot drift, so it does not
cancel, and the imbalance shows at six times the size.

This is the concrete reason P15 asks for frozen anchors rather than more field tournaments, and it
is a third entry for the list of endpoints that look fine and measure the wrong thing — after
self-play side balance and win rate against `HeuristicBot`
([`../../../method/measurement_pitfalls.md`](../../../method/measurement_pitfalls.md)).

### 4. `p29` was never competitive, and it is not a late failure

The weak pooled seed reads 4.5/17.0 against the `@80M` anchor at 120M — beaten from both sides by
an opponent 40M younger than itself. It is 1723.0 Elo at 120M, last of 24. `pooling.md` §3 treats
`P29` as the outlier dragging the pooled condition's spread to 221 Elo; this locates the damage
early rather than late, which is worth knowing before a fifth seed is spent.

### 5. One seat has an unanswered strategy and the other does not (X1)

X1 asks whether a response to the runaway exists — not which model is best on average. "Highest
win rate as the US" is true of *someone* by construction and says nothing. The question that has
content is asked per seat, over the whole field:

1. is there a model that, **playing US, beats every model playing USSR**?
2. is there a model that, **playing USSR, beats every model playing US**?

23 opponents each (HeuristicBot excluded — including a saturated opponent makes "beats
everything" easier to achieve and harder to interpret).

> **Corrected 2026-09-17 by [`P15_X1_frozen_exploiter.md`](P15_X1_frozen_exploiter.md).**
> Every cell below is 200 games a side, ±3.5 pp. Re-rated at 1000 games, `p28_280M`'s US
> against `p28_200M`'s USSR is **50.1%** [47.0, 53.2], not 52.0% — a dead heat. The claim
> that survives is narrower: `p28_280M`'s US beats **22 of 23** opponents and is held level
> by the 23rd. It has one answer, which already existed, rather than none.

**(1) Playing US: yes, two of them.**

| model | worst result as US | against |
|:---|---:|:---|
| `p28_280M` | **52.0%** | `p28_200M` |
| `p28_240M` | **51.0%** | `p28_200M` |
| `p28_320M` | 48.5% | `n26_120M` (loses or draws to 2 of 23) |

**(2) Playing USSR: no, none.**

| model | worst result as USSR | against |
|:---|---:|:---|
| `p28_200M` | 47.5% | `p28_280M` |
| `n26_080M` | 44.0% | `p28_240M` |
| `n26_240M` | 42.0% | `p28_240M` |

Every model in the field, given the USSR seat, is beaten by at least one model given the US seat.
The closest to an answer is `p28_200M` at 47.5% worst-case — and what beats it is `p28_280M`, the
same checkpoint that is unanswered in the other direction.

**Read at the strength the sample supports.** 200 games a side is ±3.5 pp, so 52.0% and 51.0% are
*non-losing records against all 23*, not significant wins over the nearest challenger. The precise
claim is: **no opponent in the population has a winning record against `p28_280M`'s US, and the
best anyone manages is a statistical tie.**

**What this means for X1.** The target is well defined — beat `p28_280M` playing US, from the USSR
seat, which requires better than 47.5%, the population's ceiling. Three cheap ways of finding that
response have already failed:

* the lineage's **own past and future selves** — `p28_200M` is its best answer at 47.5%, and every
  other `p28` checkpoint does worse;
* an **independently trained arm** — `n26_120M` reaches 40.5%;
* an arm that **specialised in the answering seat for 320M steps** — `n26_240M`, the field's most
  USSR-tilted model at +38.1 pp, reaches 42.0%.

So the response does not exist by accident, in a population of 24 models spanning three runs and
eight budgets. Whether it can be found **on purpose** is exactly what X1 proposes to test, and the
asymmetry above is the reason to run it in one direction rather than both: there is a US strategy
with no answer, and no USSR strategy in the same position.

## Caveats

* **200 games a side**, so each cell carries roughly ±3.5 pp. The amplitudes above are far larger
  than that; individual cells are not.
* **One anchor family, and it inflates `p28`'s side gap about twofold.** Both anchors are
  snapshots of `p28`, so `p28`'s rows are same-family comparisons — and an arm's side tilt is
  *shared with its own earlier selves*, so both seats of the comparison lean the same way and the
  gap compounds. Measured against the whole 25-model field instead
  ([`P15_X0_round_robin.md`](P15_X0_round_robin.md), per-side summary), `p28_280M` is **−22.8 pp**
  where the `@80M` anchor reads **−48.0 pp**, and `p28_320M` is **−15.8 pp** against −41.5. The
  §2 table is correct for what it measures — drift relative to the run's own past — but it is
  **not** a measure of how lopsided the arm is in general, and the headline numbers above should
  not be quoted as if it were.

* **Which makes the real contrast sharper, not softer.** Field-wide, `p28` is competent in both
  seats (58.6% as USSR, 74.4% as US at 320M) — tilted. `n26` is not: 69.4% as USSR against
  **37.3% as US**, a 32.1 pp gap, and 38.1 pp at 240M. The unpooled arm has not merely decayed in
  Elo, it has *lost a seat*. A head-to-head between two same-family checkpoints is then decided by
  the seat rather than by the 160M steps between them: `p28@320M` and `p28@160M` sit 3.9 Elo apart
  and whichever plays US wins 71–78%. Cross-family it inverts — `p28@320M` beats `n26@240M` 69.5%
  *as USSR* and only 52.0% as US, because what decides a matchup is whose weak seat is exposed.
  This is [`../findings/pooling.md`](../findings/pooling.md) §3a in a starker
  form, and it is an argument for a second anchor drawn from a *different* run before these
  numbers are used to compare runs rather than budgets.
* Elo here is not comparable to any other tournament in this project.

Search on top of `@200M`, against these same anchors, is in
[`P15_X0_search_on_200M.md`](P15_X0_search_on_200M.md).

Source: `data/tournaments/P15_X0_frozen_anchors/tournament_report.md`, full round robin in [`P15_X0_round_robin.md`](P15_X0_round_robin.md).
