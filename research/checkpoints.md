# Checkpoint catalogue — what exists on disk, and what it is worth

One row per rated checkpoint, with its measured strength and the field that measured it. The arms
themselves — what each one varied and why — are [`runs.md`](runs.md); this file answers the
narrower question "which model do I load, and how good is it".

**Update discipline: rewritten in place.** Written 2026-09-16, extended 2026-09-17.

## How to read a number here

**Elo is not comparable across tournaments.** `P28-160` reads 2096.1 in the 20-model `arena_p12`
field and 2150.4 in the 10-model `arena_320` field — the same checkpoint, 54 points apart, because
the field and the anchoring differ. Compare within one block below, never across blocks.

**Two live metrics cannot rank arms**, and both have been used here by mistake:

* **Win rate against `HeuristicBot`** saturates. By 120M every 4 × 4 arm beats it above 89% and
  `RandomBot` above 99%, so the anchor separates pooled from unpooled by 8.0 pp against 8.8 pp — a
  null — where the peer tournament separates them completely. A ceiling cannot show a gap.
* **Self-play side advantage** can be satisfied by both sides drifting together; it is a fact about
  the pair, not either side. It was pre-registered as the 4 × 4 primary endpoint and abandoned
  mid-flight for exactly this.

Both are on the checklist in [`method/measurement_pitfalls.md`](method/measurement_pitfalls.md).
What *is* usable live: `critic_auc` / `critic_brier_skill` and the blunder rates, neither of which
depends on an opponent's strength.

## The pooling question, answered as far as the data allows

`--opponent-self-pool` at `frac 0.30`, `capacity 12`, against no pool. Four seeds each, both rated
at 80M and 160M in one 20-model field (`arena_p12`, 76,000 games).

### Strength at 160M

| | arms | mean Elo | range | spread |
|:---|:---|---:|:---|---:|
| **pooled** | P22 2067.8, P27 2151.0, P28 2096.1, P29 1929.6 | **2061.1** | 1929.6–2151.0 | 221.4 |
| **no pool** | N22 1952.3, N24 1912.8, N25 1989.0, N26 1995.7 | **1962.5** | 1912.8–1995.7 | 82.9 |

* difference of means **+98.7 Elo** to pooled
* **3 of 4** pooled arms rank above *every* unpooled arm; the exception is P29
* direct cross-condition head-to-head, all 16 pairings (~6,400 games): pooled wins **61.5%**, and
  wins **13 of 16** pairings. Per arm against the four unpooled: P27 70.9%, P28 67.1%,
  P22 63.2%, P29 44.6%
* exact Mann-Whitney on the four-vs-four arm Elos: **U = 13 of 16, one-tailed p = 0.100**

So: the *games* are decisive and the *arms* are not. 6,400 games put pooled ahead 61.5% with a
standard error near 0.6 pp, but the unit of analysis is the arm, and four per condition gives
p = 0.10. `pooling.md` records this as "not established" on a cruder rule — condition difference
against within-condition spread — which is dominated entirely by P29. Drop P29 and the pooled
spread is 83, identical to the unpooled one.

### The sharper result: the second 80M

The same field rates every arm at both budgets, so the **within-arm** gain is measurable, and
pairing removes the arm-to-arm variance that blurs the level comparison.

| arm | condition | @80M | @160M | Δ |
|:---|:---|---:|---:|---:|
| P22 | pooled | 1935.6 | 2067.8 | **+132.2** |
| P27 | pooled | 2051.1 | 2151.0 | **+99.9** |
| P28 | pooled | 2027.4 | 2096.1 | **+68.7** |
| P29 | pooled | 1901.1 | 1929.6 | **+28.5** |
| N22 | no pool | 2037.3 | 1952.3 | −85.0 |
| N24 | no pool | 1925.9 | 1912.8 | −13.1 |
| N25 | no pool | 1891.9 | 1989.0 | +97.1 |
| N26 | no pool | 2122.5 | 1995.7 | −126.8 |

**Pooled: 4 of 4 arms improved, mean +82.3 Elo. No pool: 1 of 4, mean −32.0 Elo.** Exact
Mann-Whitney on the deltas: U = 14 of 16, one-tailed **p = 0.057**.

This is the strongest form of the pooling result in the record, and it says something more
specific than "pooled is better": **unpooled arms decay over the second 80M and pooled arms do
not.** That is what a pool of past selves is for — a defence against co-evolutionary drift, which
is a late-training phenomenon and therefore invisible in the first 80M. It also explains why the
level comparison is weaker than the paired one: at 80M the conditions are genuinely close
(N26-80 is the second-strongest model in the entire field), and the gap opens afterwards.

### And it stops there

160M → 320M bought nothing for either condition (`arena_320`): P28 2150.4 → 2151.6, N26
2050.1 → 2047.6, P29 2015.6 → 1925.1. Two of three flat, one down. Whatever pooling protects
against, it has finished protecting by 160M.

## Rated checkpoints, by field

### `arena_p12` — the 4 × 4 pooling field (20 models, 7,600 matches each)

| model | directory | steps | pool | Elo |
|:---|:---|---:|:---|---:|
| P27-160 | `E3-20-27` | 160M | yes | **2151.0** |
| N26-80 | `E3-17-26_20260914_234625` | 80M | no | **2122.5** |
| P28-160 | `E3-20-28` | 160M | yes | 2096.1 |
| P22-160 | `E3-20-22` | 160M | yes | 2067.8 |
| P27-80 | `E3-20-27` | 80M | yes | 2051.1 |
| N22-80 | `E3-17-22_20260913_164848` | 80M | no | 2037.3 |
| P28-80 | `E3-20-28` | 80M | yes | 2027.4 |
| N26-160 | `E3-17-26_20260914_234625` | 160M | no | 1995.7 |
| N25-160 | `E3-17-25` | 160M | no | 1989.0 |
| N22-160 | `E3-17-22_20260913_164848` | 160M | no | 1952.3 |
| P22-80 | `E3-20-22` | 80M | yes | 1935.6 |
| P29-160 | `E3-20-29_20260914_194500` | 160M | yes | 1929.6 |
| N24-80 | `E3-17-24_20260914_155634` | 80M | no | 1925.9 |
| N24-160 | `E3-17-24_20260914_155634` | 160M | no | 1912.8 |
| P29-80 | `E3-20-29_20260914_194500` | 80M | yes | 1901.1 |
| N25-80 | `E3-17-25` | 80M | no | 1891.9 |
| REFstrong | — | — | — | 1881.8 |
| REFweak | — | — | — | 1812.3 |
| HeuristicBot | — | — | — | 1500.0 (anchor) |
| RandomBot | — | — | — | 897.7 |

### `arena_320` — the 320M extension (10 models, 4,500 matches each)

| model | directory | Elo |
|:---|:---|---:|
| P28-320 | `E3-20-28_20260915_064054` | **2151.6** |
| P28-160 | `E3-20-28` | 2150.4 |
| N26-160 | `E3-17-26_20260914_234625` | 2050.1 |
| N26-320 | `E3-17-26` | 2047.6 |
| P29-160 | `E3-20-29_20260914_194500` | 2015.6 |
| P29-320 | `E3-20-29_20260915_064054` | 1925.1 |

### `E3-22-28_vs_E3-20-28` — the per-player GAE arm (10 models, 9,000 matches each)

Run 2026-09-16, `data/tournaments/E3-22-28_vs_E3-20-28/`. Every `arm_*` row is
`--per-player-gae`; see [`findings/training/value_bootstrap_perspective.md`](findings/training/value_bootstrap_perspective.md).

| model | directory | steps | Elo |
|:---|:---|---:|---:|
| base_80M | `E3-20-28` | 80M | **2274.8** |
| base_60M | `E3-20-28` | 60M | 2180.8 |
| base_40M | `E3-20-28` | 40M | 2076.3 |
| base_20M | `E3-20-28` | 20M | 1880.9 |
| arm_60M | `E3-22-28_20260916_132615` | 60M | 1839.3 |
| arm_80M | `E3-22-28_20260916_132615` | 80M | 1754.4 |
| arm_40M | `E3-22-28_20260916_132615` | 40M | 1720.7 |
| arm_20M | `E3-22-28_20260916_132615` | 20M | 1659.2 |

Note `base_80M` at 2274.8 against the same checkpoint's 2027.4 in `arena_p12` — the clearest
illustration on this page of why Elo does not travel between fields.

### `E3-23-28_vs_E3-20-28` — the PFSP arm (10 models, 9,000 matches each)

Run 2026-09-16, `data/tournaments/E3-23-28_vs_E3-20-28/`. `arm_*` rows are `--opponent-pfsp`;
result is a null, see [`findings/training/pooling.md`](findings/training/pooling.md) §3c.

| model | directory | steps | Elo |
|:---|:---|---:|---:|
| arm_160M | `E3-23-28_20260916_171033` | 160M | **2295.6** |
| base_160M | `E3-20-28` | 160M | 2284.0 |
| base_80M | `E3-20-28` | 80M | 2208.6 |
| arm_80M | `E3-23-28_20260916_171033` | 80M | 2206.9 |
| arm_40M | `E3-23-28_20260916_171033` | 40M | 2068.4 |
| base_40M | `E3-20-28` | 40M | 2016.3 |
| arm_20M | `E3-23-28_20260916_171033` | 20M | 1834.8 |
| base_20M | `E3-20-28` | 20M | 1816.9 |

### `P15_X0_frozen_anchors` — three runs to 320M on one scale (24 models, 200 games/side)

Run 2026-09-16 for P15-X0. `p28` = E3-20-28 (pooled), `p29` = E3-20-29 (pooled, weak seed),
`n26` = E3-17-26 (no pool), each at 40M intervals from 40M to 320M. Full field and the per-side
matrices in [`log/P15_X0_round_robin.md`](log/P15_X0_round_robin.md); the anchor table and its
reading in [`log/P15_X0_frozen_anchors.md`](log/P15_X0_frozen_anchors.md).

| model | Elo | | model | Elo |
|:---|---:|---|:---|---:|
| `p28_200M` | **2193.7** | | `p28_080M` | 2083.1 |
| `n26_120M` | **2188.8** | | `p29_200M` | 2057.9 |
| `n26_080M` | 2172.6 | | `n26_320M` | 2034.3 |
| `p28_240M` | 2168.0 | | `n26_160M` | 2031.1 |
| `p28_160M` | 2141.9 | | `n26_200M` | 2020.1 |
| `p28_320M` | 2138.0 | | `p29_160M` | 1992.4 |
| `p28_120M` | 2133.3 | | `p29_040M` | 1930.9 |
| `p28_280M` | 2126.4 | | `p29_120M` | 1723.0 |

**Each run peaks and then declines**: `n26` at 120M (−154 Elo by 320M), `p28` at 200M (−56),
`p29` at 200M (−152). No arm of any configuration is improving past 200M.

### `P15_X4a_distilled` — one offline round of expert iteration (4 models, 1000 games/side)

Distilling an honest searcher's visit distribution into the policy head, card/play-mode nodes
only. See [`log/P15_X4a_distillation.md`](log/P15_X4a_distillation.md).

| model | Elo | what it is |
|:---|---:|:---|
| `x4a_distilled_200M.pt` | **1666.4** | `p28_200M` after 2 epochs of CE toward a 96-sim searcher |
| `p28_200M` | 1618.7 | the source |
| `p28_280M` | 1585.7 | |
| `p28_080M` | 1500.0 | anchor |

**`/workspace/data/checkpoints/x4a_distilled_200M.pt` is a product, not just an arm** — +47.7 Elo
over its source for two epochs of offline SFT, no RL. It is also the most side-balanced checkpoint
this project has: −0.9 pp USSR−US where its own source is −3.3 and `p28_280M` is −17.7.

### `P15_X4a_washout_control` — does the distilled gain survive RL? (5 models, 500 games/side)

Both post-RL arms and the three checkpoints they came from, on one scale. Supersedes the 4-model
`P15_X4a_washout`, which lacked the control and produced a wrong reading.

| model | Elo | what it is |
|:---|---:|:---|
| `distilled` | **1548.8** | as above, pre-RL |
| `source_200M` | 1526.7 | pre-RL |
| `anchor_280M` | 1500.0 | anchor |
| `after_rl_distilled` | 1491.3 | `E3-25-28`, 20M steps from `distilled` |
| `after_rl_control` | 1482.6 | `E3-26-28`, the same 20M from `source_200M` |

**The edge does not survive: +22.1 Elo before RL, +8.7 after, and the two arms split 51.2% head to
head over 1,000 games.** The control also lost 44.1 Elo from an undistilled start, so the large
drop both arms show is the price of training past this lineage's peak, not washout.

Side balance moved **−17.3 pp and −11.6 pp toward US** in the two arms over the identical 20M
steps — within-arm, same seed. See [`log/P15_X4a_distillation.md`](log/P15_X4a_distillation.md).

### `P15_X4b_verdict` — search targets during RL (4 models, 500 games/side, temperature 0)

The arm against its step-, seed- and cadence-matched control. See
[`log/P15_X4b_search_during_rl.md`](log/P15_X4b_search_during_rl.md).

| model | Elo | what it is |
|:---|---:|:---|
| `x4b_arm` | **1666.9** | `E3-29-28`, 20M steps with search-CE at coef 0.5, 64 sims, all nodes |
| `source_200M` | 1518.0 | where both runs started |
| `control_no_search` | 1501.3 | `E3-26-28`, the identical 20M with the search term off |
| `anchor_280M` | 1500.0 | anchor |

**`E3-29-28`'s final checkpoint is the strongest model this project has produced against this
field** — +165.6 Elo on its own control, +148.9 on the checkpoint it resumed from, and USSR 61.9%
where the control manages 34.7%. Rated at temperature 0 because the arm's policy entropy (0.56)
differs from the control's (1.17); at the sampling default a sharper policy wins partly on
sharpness.

Gap by matched step: +224.4, +249.6, +253.8, +165.6. It peaks near 15M — do not assume the 20M
number is where it settles.

### `arena_heads` — the per-entity-head ablation (10 models, 4,500 matches each)

| model | Elo |
|:---|---:|
| H64-E3-18-22 | **1978.0** |
| H64-E3-17-22 | 1967.0 |
| H64-E3-17-24 | 1944.6 |
| NOH-p1ident-b | 1940.1 |
| NOH-p1ident-a | 1935.7 |
| NOH-p1scalar-a | 1864.9 |
| NOH-p1scalar-b | 1844.5 |
| NOH-armH2 | 1836.4 |

This field contains the **only** number ever recorded for `E3-18-22`, which
[`findings/training/pooling.md`](findings/training/pooling.md) flags as having no writeup at all:
it tops the field at 1978.0.

### `arena80_160` — retracted

Its three headline claims were withdrawn when a second unpooled seed landed, and two of its arms
carry confounds (`E3-20-22` ran with `seed=None`; `E3-19-23` resumed from 90M rather than 80M
because the 80M state had been pruned). Kept for provenance only; see
[`findings/training/pooling.md`](findings/training/pooling.md) §1.

## Unrated, and worth knowing about

| directory | steps | what it is | status |
|:---|---:|:---|:---|
| `E3-21-28_20260916_092205` | 40M | same-perspective bootstrap | negative, critic at chance; not rated |
| `E3-22-28_20260916_121202_VOID_thin_opponent_pool` | 47M | per-player GAE, first attempt | **void** — snapshot cadence made it a two-factor experiment |
| `E3-21-28_20260916_004620_VOID_refusal_crash` | — | died on ENG-3 | void |
| `E3-21-28_20260916_010023_VOID_eng3_crash` | — | died on ENG-3 | void |
| `E3-25-28_20260917_020529` | 20M | X4a step 4, RL from the distilled checkpoint | rated in `P15_X4a_washout_control` |
| `E3-26-28_20260917_025539` | 20M | X4a step 4 **control**, same recipe from the undistilled source | rated in `P15_X4a_washout_control` |
| `E3-27-28_..._VOID_ce_coef_collapse` | 1.4M | X4b first attempt, coef 0.5 | **void** — search targets off by one step |
| `E3-28-28_..._VOID_search_target_offbyone` | 0.2M | X4b second attempt, coef 0.05 | **void** — same bug |
| `E3-18-22` | 160–240M | P10 experiment 1, the do-nothing control | rated only in `arena_heads`, no writeup |
| `E3-19-22`, `E3-19-23` | 80–160M, 90–160M | pool-from-checkpoint arms | rated only in the retracted `arena80_160` |

## Two traps in the directory names

**The bare-name directory is sometimes the continuation.** `E3-17-26/` holds 160M → 320M while
`E3-17-26_20260914_234625/` holds the original 0 → 160M. Selecting a run by its bare name silently
picks the wrong arm — it produced an empty result in this catalogue's own first draft. The same
pattern holds for `E3-20-28/` (0–160M) against `E3-20-28_20260915_064054/` (160–320M), where the
bare name is the *original*. There is no rule; check the step range.

**Snapshots across two runs share basenames.** Both arms of a matched pair land on identical step
counts, so `snapshot_40042496steps.pt` exists in both, and `load_agent` names an agent from its
basename — two files entering one Bradley-Terry row. Stage symlinks with distinct names, as
`data/tournaments/E3-22-28_vs_E3-20-28/models/` does.
