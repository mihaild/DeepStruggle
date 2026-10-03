# Star Wars when the US is ahead in space: event or Ops, across E7-02-44 (2026-10-03)

Owner: for E7-02-44, at 40M snapshot intervals, how often the US, ahead in space with Star Wars in
hand, plays it for the event versus for Ops. Follow-up: per snapshot, play until there are 1,000
such plays, split into headline / Ops in an action round / event in an action round, with a plot.

Star Wars (US, 2 Ops, Late War): if the US is ahead on the Space Race Track, the US takes any
non-scoring card from the discard pile and plays it as an event. Level or behind, the event does
nothing; the engine offers EVENT in a round only when the US is ahead, which the probe confirms.

**Method.** `tools/scripts/star_wars_play.py --target-plays 1000`: greedy self-play of each
snapshot in rounds of 4,096 games on fresh seeds (from seed 57,000) until there are at least 1,000
plays while ahead, then exactly the first 1,000 in game order (8,209-15,268 games per snapshot).
The snapshot nearest each 40M mark; E7-02-44 is E7-01-44 continued from 560M, so the marks to
560M are E7-01-44's snapshots (the same lineage). A *play* is the US selecting Star Wars at its own
card choice: a headline (its event), or an action round followed by the play mode -- EVENT, SPACE
(space race), or Ops (influence, coup or realign). "Ahead" is the US space marker strictly above
the USSR's at that choice. A selection at a card choice that was not a play (not followed by Star
Wars' play mode -- 2 in a 2,048-game check) is set aside. Dumps:
`data/eval/star_wars/E7-02-44_1000/<snapshot>.json`; the plot is drawn from them
(`--load --plot-svg ... --branch-at 560`).

## Result

![Star Wars played while the US is ahead in space, E7-01-44/E7-02-44 snapshots](star_wars_event_vs_ops_E7-02-44.svg)

Each row is 1,000 plays while ahead (95% interval about ±3 points at 20%, ±2 at 5%):

| snapshot | games played | plays while ahead | headline | Ops in a round | event in a round | space race |
|:---|---:|---:|---:|---:|---:|---:|
| 40M (E7-01) | 10,154 | 1,000 | 429 (42.9%) | 445 (44.5%) | 96 (9.6%) | 30 (3.0%) |
| 80M (E7-01) | 8,575 | 1,000 | 387 (38.7%) | 536 (53.6%) | 63 (6.3%) | 14 (1.4%) |
| 120M (E7-01) | 15,268 | 1,000 | 94 (9.4%) | 878 (87.8%) | 6 (0.6%) | 22 (2.2%) |
| 160M (E7-01) | 9,847 | 1,000 | 224 (22.4%) | 749 (74.9%) | 15 (1.5%) | 12 (1.2%) |
| 200M (E7-01) | 13,108 | 1,000 | 56 (5.6%) | 896 (89.6%) | 39 (3.9%) | 9 (0.9%) |
| 240M (E7-01) | 9,829 | 1,000 | 25 (2.5%) | 948 (94.8%) | 19 (1.9%) | 8 (0.8%) |
| 280M (E7-01) | 10,161 | 1,000 | 41 (4.1%) | 914 (91.4%) | 36 (3.6%) | 9 (0.9%) |
| 320M (E7-01) | 12,698 | 1,000 | 111 (11.1%) | 859 (85.9%) | 21 (2.1%) | 9 (0.9%) |
| 360M (E7-01) | 10,708 | 1,000 | 53 (5.3%) | 894 (89.4%) | 49 (4.9%) | 4 (0.4%) |
| 400M (E7-01) | 11,181 | 1,000 | 83 (8.3%) | 865 (86.5%) | 49 (4.9%) | 3 (0.3%) |
| 440M (E7-01) | 9,355 | 1,000 | 74 (7.4%) | 890 (89.0%) | 25 (2.5%) | 11 (1.1%) |
| 480M (E7-01) | 8,396 | 1,000 | 118 (11.8%) | 875 (87.5%) | 3 (0.3%) | 4 (0.4%) |
| 520M (E7-01) | 11,544 | 1,000 | 83 (8.3%) | 867 (86.7%) | 45 (4.5%) | 5 (0.5%) |
| 560M (E7-01) | 9,118 | 1,000 | 21 (2.1%) | 939 (93.9%) | 22 (2.2%) | 18 (1.8%) |
| 600M | 9,276 | 1,000 | 51 (5.1%) | 885 (88.5%) | 61 (6.1%) | 3 (0.3%) |
| 640M | 11,065 | 1,000 | 46 (4.6%) | 927 (92.7%) | 21 (2.1%) | 6 (0.6%) |
| 680M | 10,846 | 1,000 | 103 (10.3%) | 877 (87.7%) | 13 (1.3%) | 7 (0.7%) |
| 720M | 9,969 | 1,000 | 131 (13.1%) | 855 (85.5%) | 4 (0.4%) | 10 (1.0%) |
| 760M | 11,506 | 1,000 | 77 (7.7%) | 874 (87.4%) | 37 (3.7%) | 12 (1.2%) |
| 800M | 9,385 | 1,000 | 139 (13.9%) | 836 (83.6%) | 11 (1.1%) | 14 (1.4%) |
| 840M | 9,502 | 1,000 | 191 (19.1%) | 751 (75.1%) | 42 (4.2%) | 16 (1.6%) |
| 880M | 9,854 | 1,000 | 143 (14.3%) | 835 (83.5%) | 19 (1.9%) | 3 (0.3%) |
| 920M | 10,354 | 1,000 | 151 (15.1%) | 826 (82.6%) | 11 (1.1%) | 12 (1.2%) |
| 960M | 9,256 | 1,000 | 169 (16.9%) | 789 (78.9%) | 26 (2.6%) | 16 (1.6%) |
| 1000M | 9,399 | 1,000 | 207 (20.7%) | 788 (78.8%) | 3 (0.3%) | 2 (0.2%) |
| 1040M | 9,535 | 1,000 | 181 (18.1%) | 810 (81.0%) | 4 (0.4%) | 5 (0.5%) |
| 1080M | 8,209 | 1,000 | 200 (20.0%) | 776 (77.6%) | 11 (1.1%) | 13 (1.3%) |
| 1120M | 8,371 | 1,000 | 147 (14.7%) | 842 (84.2%) | 5 (0.5%) | 6 (0.6%) |
| 1160M | 10,424 | 1,000 | 186 (18.6%) | 790 (79.0%) | 21 (2.1%) | 3 (0.3%) |
| 1200M | 11,259 | 1,000 | 212 (21.2%) | 756 (75.6%) | 28 (2.8%) | 4 (0.4%) |

How often a play of Star Wars is a headline, by space position at the choice (all plays in the
same games; the event works only when ahead):

| snapshot | ahead | level | behind |
|:---|---:|---:|---:|
| 40M (E7-01) | 43% (429/1000) | 46% (150/326) | 46% (219/471) |
| 80M (E7-01) | 39% (387/1000) | 38% (121/318) | 41% (219/533) |
| 120M (E7-01) | 9% (94/1000) | 8% (30/378) | 10% (81/799) |
| 160M (E7-01) | 22% (224/1000) | 24% (95/395) | 22% (153/687) |
| 200M (E7-01) | 6% (56/1000) | 4% (20/522) | 4% (47/1283) |
| 240M (E7-01) | 2% (25/1000) | 2% (8/398) | 3% (24/767) |
| 280M (E7-01) | 4% (41/1000) | 3% (10/380) | 3% (28/954) |
| 320M (E7-01) | 11% (111/1000) | 7% (33/469) | 7% (73/984) |
| 360M (E7-01) | 5% (53/1000) | 6% (21/379) | 5% (39/723) |
| 400M (E7-01) | 8% (83/1000) | 8% (33/428) | 7% (66/910) |
| 440M (E7-01) | 7% (74/1000) | 6% (25/387) | 7% (45/653) |
| 480M (E7-01) | 12% (118/1000) | 14% (46/338) | 9% (57/661) |
| 520M (E7-01) | 8% (83/1000) | 7% (35/489) | 7% (65/971) |
| 560M (E7-01) | 2% (21/1000) | 2% (6/390) | 0% (2/789) |
| 600M | 5% (51/1000) | 5% (17/366) | 6% (40/681) |
| 640M | 5% (46/1000) | 5% (24/471) | 4% (49/1155) |
| 680M | 10% (103/1000) | 11% (47/434) | 9% (81/885) |
| 720M | 13% (131/1000) | 14% (56/400) | 13% (109/817) |
| 760M | 8% (77/1000) | 7% (32/428) | 8% (74/883) |
| 800M | 14% (139/1000) | 12% (53/433) | 13% (99/762) |
| 840M | 19% (191/1000) | 18% (63/358) | 17% (132/788) |
| 880M | 14% (143/1000) | 11% (43/391) | 10% (82/827) |
| 920M | 15% (151/1000) | 14% (61/431) | 11% (83/753) |
| 960M | 17% (169/1000) | 14% (56/394) | 15% (118/790) |
| 1000M | 21% (207/1000) | 17% (57/341) | 16% (114/720) |
| 1040M | 18% (181/1000) | 20% (71/352) | 15% (107/724) |
| 1080M | 20% (200/1000) | 19% (65/335) | 16% (83/522) |
| 1120M | 15% (147/1000) | 11% (31/274) | 11% (63/581) |
| 1160M | 19% (186/1000) | 18% (70/393) | 18% (150/830) |
| 1200M | 21% (212/1000) | 18% (82/467) | 15% (149/985) |

Pooled by period, headline rate ahead vs not ahead (level and behind together):

| period | ahead | level | behind | ahead − not ahead |
|:---|---:|---:|---:|---:|
| 40-80M | 40.8% | 42.1% | 43.6% | −2.2 pp (±3.2) |
| 120-840M | 9.1% | 8.3% | 7.8% | +1.1 pp (±0.5) |
| 880-1,200M | 17.7% | 15.9% | 14.1% | +3.0 pp (±1.0) |

**Reading.**

* **Ahead in space, the US mostly plays Star Wars for Ops in a round**: 75-95% of plays from 120M
  on (76% at 1,200M). The early snapshots split it (40M: 43% headline, 45% Ops, 10% event in a
  round).
* **The event, when it is used, is almost always the headline.** Choosing EVENT over Ops in an
  action round is 0.3-6% at every snapshot after 80M (2.8% at 1,200M). The space race is ≤ 3%.
* **The headline share falls to 2-12% through 200-640M, then climbs in E7-02-44's own range** to
  14-21% from 840M (21% at 1,200M).
* **That climb is not the event being valued.** The headline rate is almost the same whether the
  US is ahead, level or behind -- and only when ahead does the event do anything. The gap opens
  only late, and stays small: +3.0 points (17.7% vs 14.6%) pooled over 880-1,200M, +1.1 over
  120-840M, none at 40-80M. So most of the rise is the policy headlining Star Wars more as such --
  a weak 2-Ops US card whose headline costs no action round -- with a faint, late preference for
  doing so when the event works. At 1,200M, 231 of 1,452 plays while not ahead (16%) are headlines
  that do nothing.
* The first, 4,096-game-per-snapshot version of this probe (same seeds for the first 4,096 games)
  gave the same picture, and found about 10% of "ahead" holdings never played by the US (game
  end, or discarded or taken by other events).

**Not measured here:** whether the event would be worth more than the Ops in these positions. That
depends on what is in the discard pile, and needs a counterfactual (play the event on a copy and
compare the value head or game results). That is the natural next step if the owner wants it.
