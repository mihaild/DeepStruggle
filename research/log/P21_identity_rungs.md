# P21 — M2.5 / M2.5c / M2.5b: the identity vector is rejected

One tournament, 18 entrants, 100 games per side at `temperature 0.0`, run 2026-09-22.
`/workspace/data/reports/P21_six_rungs.{json,md}`.

Three rungs add a trainable per-entity identity vector to the head, and nothing else:

| rung | head input | `pe_country` | `pe_card` | params |
|:---|:---|---:|---:|---:|
| M2d | dynamic + constants + ctx | 90 | — | 3,184,960 |
| **M2.5** | + identity | 106 | — | 3,189,088 |
| **M2.5c** | + identity, + card head | 106 | 94 | 3,195,233 |
| **M2.5b** | + identity, − constants | 95 | — | 3,188,384 |

Head widths were verified by construction before launch and the trunk's board projection is
**1,260 inputs in every configuration** — identity reaches the heads and nothing else, which is
the design claim. The largest parameter delta is 10,273 of 3.18M (**0.32%**), so nothing here can
be a capacity effect.

Seeds 3 and 5 throughout, both clean for M2d at both budgets and already rated there, so every
comparison is within-seed.

## Result: all three lose at 80M, on both seeds

| rung | 80M s5 | 80M s3 | Δ s5 | Δ s3 |
|:---|---:|---:|---:|---:|
| M2d | 2005.4 | 2094.3 | — | — |
| M2b | 1951.5 | 2019.4 | −53.9 | −74.9 |
| **M2.5** | 1958.6 | 1971.8 | **−46.8** | **−122.5** |
| **M2.5c** | 1915.6 | 1912.5 | **−89.8** | **−181.8** |
| **M2.5b** | 1945.3 | 1916.1 | **−60.1** | **−178.2** |

The adoption rule is *beat the rung below at 80M on both seeds*. None does. **Identity is
rejected**, and M2d's head stands unchanged.

**M2b replicates.** It measured −49.1 / −73.1 in the earlier M2abc field and −53.9 / −74.9 here —
a different tournament, a different entrant list, the same answer inside 5 Elo. That is the
protocol working, and it is why the identity numbers can be believed at face value.

## Identity does not rescue the card head — it is the most expensive variant

M2e measured the card head at −7 *without* identity, and P21 argued it could not work without
one, since 95 of 110 cards are indistinguishable to a shared MLP from their own slots. M2.5c gives
it sight. It is the **worst** rung at 80M, −89.8 / −181.8.

A behavioural diagnostic pointed the other way first — M2.5c leaves 2.33 empty battlegrounds at
turn 8 against M2.5's 9.03 — and that reading was wrong. Recorded because the diagnostic is
otherwise a good instrument: it tracks Elo tightly across the whole ladder (M0 8.51 → M2d 1.29)
and it still caught the anchor's degradation independently. Here it disagreed with the rating, and
the rating is the arbiter.

## M2.5b collapses reliably — the ladder's first architecture-linked collapse

| arm | seed | onset | pinned | min `adv_std_raw` |
|:---|---:|---:|---:|---:|
| `E4-17-03` | 3 | 109,969,408 | 21.0% | 0.0104 |
| `E4-17-04` | 4 | 131,989,504 | 13.1% | 0.0223 |

Both seeds are ones on which **M2d itself runs clean to 160M** (`E4-11-03`, `E4-11-04`: zero
pinned rows each). Against a ~19%-by-160M base rate two collapses would be p ≈ 0.036, and that
overstates the chance, because these are not random draws — they are seeds selected for being
clean on the base architecture. A third seed (6, also clean for M2d) is running.

**Neither ingredient collapses alone.** M2b removes the constants without identity: stable, and
the second-strongest arm in the field. M2.5 and M2.5c add identity while keeping the constants:
stable. Only the combination fails. So identity does **not** subsume the hand-designed per-type
constants — removing them in identity's presence destabilises training, which answers M2.5b's
question in the opposite direction to the one it was asked in.

Both onsets are past 80M, so the ~80M snapshots predate them and the 80M numbers above are valid
measurements. What M2.5b has no valid measurement *of* is 160M.

## Two distinct failure modes, and identity is present in every instance

Within-lineage slopes, where the seed cancels because it is the same run:

| arm | identity | 80M → 160M | pinned | min `adv_std_raw` | ΔH | mode |
|:---|:---:|---:|---:|---:|---:|:---|
| M2d s3 | no | **+125.5** | 0.0% | 0.1617 | +0.134 | healthy |
| M2b s3 | no | **+147.0** | 0.0% | 0.1246 | −0.379 | healthy |
| M2.5c s3 | yes | +118.1 | 2.5% | 0.0717 | −0.025 | healthy |
| M2.5 s3 | yes | **−401.0** | 4.5% | 0.0664 | +0.220 | entropy inflation |
| M2.5b s3 | yes | **−280.9** | 21.0% | 0.0104 | +0.225 | side collapse |
| anchor | yes | **−362.7** | 0.0% | 0.0982 | +0.451 | entropy inflation |

Three observations, in decreasing order of confidence.

1. **The two failure modes are distinct and the per-side win rates separate them.** The collapsed
   M2.5b arm wins **0.0% as US** against M2d@160M and 1.0% against the anchor, against 8.0%/11.0%
   as USSR. M2.5 at 160M is **symmetric** — 4.0% USSR / 2.0% US — and so is the anchor
   (5.0%/5.0%). One arm went one-sided; the other two became indecisive while staying balanced.
   [`../findings/training/entropy_inflation.md`](../findings/training/entropy_inflation.md).

2. **Every arm that falls carries identity; neither arm without it falls.** Four instances against
   two controls. M2.5c is the counterexample within the identity group — it climbs +118.1 — so
   this is 3 of 4, not a law.

3. **Every identity arm has a depressed `adv_std_raw` floor**, 0.0664–0.0982 against the
   non-identity arms' 0.1246–0.1617, with no overlap — M2.5c included, despite its healthy slope.
   Two controls is too few to call this separation, but it is the sharpest version of the pattern
   and it is cheap to extend: every future rung records the number anyway.

This supersedes a hypothesis raised and withdrawn on 2026-09-21. It was withdrawn on M2.5c's
battleground number and a flat entropy read; the Elo reinstates the weaker form of it. It also
places the anchor's 356 Elo fall as an *instance* rather than a one-off, which is the first thing
that has explained it at all.

## Method notes

* Ratings are field-relative. M2d@80M s3 rates 2094.3 here and 2076.2 in the M2abc field,
  unchanged. Only deltas inside this JSON mean anything.
* The 80M point of a cold 160M arm is that arm's own ~80M snapshot, sharing a byte-identical
  prefix with its 160M sibling. M2d's 160M arm is a *continuation*, so its 80M point is
  `E4-08-03`'s final — the same lineage.
* `E4-17-03@final` is in the field labelled COLLAPSED and is **not** M2.5b's strength at 160M.
  Protocol amendment 1 bars reading a collapsed arm as a strength measurement; it does not bar
  rating one to price what a collapse costs, which is what that row is for.
* The anchor contributes two entrants, not three: it has one lineage and no second seed.
