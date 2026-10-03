# Star Wars when the US is ahead in space: event or Ops, across E7-02-44 (2026-10-03)

Owner: for E7-02-44, at 40M snapshot intervals, how often the US, ahead in space with Star Wars in
hand, plays it for the event versus for Ops.

Star Wars (US, 2 Ops, Late War): if the US is ahead on the Space Race Track, the US takes any
non-scoring card from the discard pile and plays it as an event. Level or behind, the event does
nothing; the engine offers EVENT in a round only when the US is ahead, which the probe confirms.

**Method.** `tools/scripts/star_wars_play.py`, 4,096 greedy self-play games per snapshot (seed
57,000), the snapshot nearest each 40M mark. E7-02-44 is E7-01-44 continued from 560M, so the
marks to 560M are E7-01-44's snapshots (the same lineage). A *play* is the US selecting Star Wars
at its own card choice: a headline (its event), or an action round followed by the play mode
(EVENT, SPACE, or Ops: influence, coup or realign). "Ahead" is US space marker strictly above the
USSR's at that choice. A *holding* (card entering the US hand until it leaves) counts as "ahead"
when the US was ahead at one of its card choices while holding it. Two selections in a 2,048-game
check were card choices that were not plays (not followed by Star Wars' play mode); they are set
aside. Dumps: `data/eval/star_wars/E7-02-44/<snapshot>.json`.

## Result

Plays while ahead in space:

| snapshot | plays while ahead | **as event** | headline | AR event | **for Ops** | space race | holdings ahead | holding evented | holding not played |
|:---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 40M (E7-01) | 406 | **52%** | 44% | 9% | **45%** | 3% | 535 | 40% | 21% |
| 80M (E7-01) | 504 | **45%** | 39% | 6% | **53%** | 1% | 558 | 41% | 7% |
| 120M (E7-01) | 249 | **8%** | 8% | 1% | **90%** | 2% | 489 | 4% | 45% |
| 160M (E7-01) | 402 | **22%** | 21% | 1% | **76%** | 2% | 501 | 18% | 15% |
| 200M (E7-01) | 294 | **8%** | 5% | 3% | **91%** | 1% | 385 | 6% | 17% |
| 240M (E7-01) | 423 | **3%** | 1% | 2% | **96%** | 1% | 518 | 3% | 12% |
| 280M (E7-01) | 405 | **6%** | 4% | 3% | **93%** | 0% | 468 | 6% | 7% |
| 320M (E7-01) | 311 | **12%** | 9% | 2% | **88%** | 0% | 379 | 9% | 12% |
| 360M (E7-01) | 402 | **11%** | 6% | 5% | **89%** | 1% | 450 | 10% | 9% |
| 400M (E7-01) | 352 | **12%** | 8% | 5% | **87%** | 0% | 403 | 11% | 8% |
| 440M (E7-01) | 445 | **10%** | 7% | 2% | **88%** | 2% | 493 | 9% | 5% |
| 480M (E7-01) | 473 | **12%** | 12% | 0% | **87%** | 0% | 520 | 11% | 5% |
| 520M (E7-01) | 357 | **13%** | 10% | 4% | **86%** | 1% | 434 | 11% | 12% |
| 560M (E7-01) | 444 | **3%** | 1% | 2% | **95%** | 2% | 520 | 3% | 10% |
| 600M | 453 | **10%** | 5% | 5% | **90%** | 0% | 513 | 9% | 8% |
| 640M | 373 | **6%** | 4% | 2% | **93%** | 1% | 450 | 5% | 10% |
| 680M | 379 | **11%** | 9% | 1% | **88%** | 2% | 447 | 9% | 11% |
| 720M | 430 | **15%** | 15% | 0% | **84%** | 1% | 502 | 13% | 10% |
| 760M | 340 | **11%** | 7% | 3% | **88%** | 1% | 416 | 9% | 12% |
| 800M | 445 | **14%** | 13% | 1% | **85%** | 1% | 492 | 12% | 7% |
| 840M | 432 | **23%** | 19% | 4% | **76%** | 1% | 503 | 19% | 11% |
| 880M | 411 | **15%** | 13% | 2% | **85%** | 0% | 460 | 13% | 8% |
| 920M | 397 | **15%** | 14% | 1% | **84%** | 1% | 487 | 12% | 14% |
| 960M | 441 | **19%** | 17% | 2% | **80%** | 1% | 497 | 17% | 7% |
| 1000M | 423 | **21%** | 21% | 0% | **79%** | 0% | 508 | 18% | 13% |
| 1040M | 428 | **17%** | 16% | 1% | **83%** | 0% | 509 | 14% | 12% |
| 1080M | 491 | **20%** | 19% | 1% | **79%** | 0% | 553 | 18% | 9% |
| 1120M | 467 | **16%** | 16% | 0% | **84%** | 0% | 543 | 14% | 10% |
| 1160M | 374 | **21%** | 20% | 2% | **79%** | 0% | 428 | 18% | 11% |
| 1200M | 354 | **24%** | 21% | 3% | **75%** | 1% | 411 | 21% | 10% |

How often a play of Star Wars is a headline, by space position at the choice (event works only
when ahead):

| snapshot | ahead | level | behind |
|:---|---:|---:|---:|
| 40M (E7-01) | 44% | 44% | 47% |
| 160M (E7-01) | 21% | 21% | 23% |
| 400M (E7-01) | 8% | 5% | 7% |
| 560M (E7-01) | 1% | 1% | 0% |
| 800M | 13% | 12% | 13% |
| 1000M | 21% | 15% | 17% |
| 1080M | 19% | 20% | 13% |
| 1200M | 21% | 19% | 15% |

(Every snapshot is in the tool's dumps; the pattern is the same at all 30.)

**Reading.**

* **Ahead in space, the US mostly plays Star Wars for Ops**: 75-96% of plays from 120M on. The
  event share falls from ~50% at 40-80M to 3-13% through 200-640M, then climbs slowly in
  E7-02-44's own range to ~15-24% from 840M (24% at 1,200M). Space race use is ≤ 3%.
* **Almost all of the event share is the headline.** In an action round, choosing EVENT over Ops
  happens in 0-5% of plays at every snapshot after 80M.
* **The headline does not depend on being ahead.** At every snapshot the US headlines Star Wars at
  about the same rate whether it is ahead, level or behind, though only when ahead does the event
  do anything (e.g. 1,200M: 21% / 19% / 15%; 560M: 1% / 1% / 0%). So the rising headline share is
  not a growing use of the event: the policy headlines the card as often when it is a dud. That
  reads as Star Wars being used as a headline throwaway (a 2-Ops US card whose headline costs no
  action round) rather than for its event, and as the space-track condition not entering the
  choice. At 1,200M, 82 of 502 plays while not ahead (16%) are headlines that do nothing.
* About 10% of "ahead" holdings never see the US play the card (game end, or discarded or taken
  by other events).

**Not measured here:** whether the event would be worth more than the Ops in these positions. That
depends on what is in the discard pile, and needs a counterfactual (play the event on a copy and
compare the value head or game results). That is the natural next step if the owner wants it.
