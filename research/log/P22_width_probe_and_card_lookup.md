# The width probe and P22-a — neither beats M2d

Two arms of [`P22`](../plans/P22_card_lookup_attention.md), each on the ladder's working seed pair
(3 → 160M, 5 → 80M), each rated in its own field against seed-matched M2d, the P21 anchor and
`HeuristicBot`. 100 games per seat per pair, temperature 0.

| arm | what it changes | directories |
|:---|:---|:---|
| **width probe** `E4-23` | M2d with `--ladder-entity-proj-dim 512` instead of 256 (+26.7% params) — is the card path capacity-limited? | `E4-23-03_20260922_145550`, `E4-23-05_20260922_155603` |
| **P22-a** `E4-24` | M2d plus the identity-keyed card lookup: keys are identity + the full 14-float card row (location included), values the full row, query from the pre-fusion vector, output concatenated into fusion. 3,352,608 params | `E4-24-03_20260922_171610`, `E4-24-05_20260922_184942` |

Reports: `data/reports/P21_width_probe.{md,json}`, `data/reports/P22_card_lookup.{md,json}`.

## Results

Win rates are the row's, **as USSR / as US**. The M2d and anchor rows are byte-identical in both
fields (deterministic play), so the two tables share a reference.

**Width probe**, sorted by Elo:

| checkpoint | Elo | vs M2d s3@160M | vs anchor `E4-03-01@80M` | Δ vs seed-matched M2d |
|:---|---:|---:|---:|---:|
| `E4-08-03@160M` (M2d) | 2201.1 | — | 70 / 69 | |
| `E4-08-03@80M` (M2d) | 2076.0 | 34 / 32 | 40 / 58 | |
| `E4-03-01@80M` (anchor) | 2069.7 | 31 / 30 | — | |
| `E4-08-05@80M` (M2d) | 1998.0 | 31 / 22 | 35 / 48 | |
| **`E4-23-03@160M`** | 1910.3 | 25 / 6 | 35 / 25 | **−290.8** |
| **`E4-23-03@80M`** | 1886.4 | 21 / 4 | 29 / 29 | **−189.6** |
| **`E4-23-05@80M`** | 1869.3 | 21 / 1 | 22 / 18 | **−128.7** |

**P22-a**, sorted by Elo:

| checkpoint | Elo | vs M2d s3@160M | vs anchor `E4-03-01@80M` | Δ vs seed-matched M2d |
|:---|---:|---:|---:|---:|
| `E4-08-03@160M` (M2d) | 2206.9 | — | 70 / 69 | |
| **`E4-24-03@160M`** | 2153.7 | 41 / 29 | 60 / 68 | **−53.2** |
| `E4-08-03@80M` (M2d) | 2074.1 | 34 / 32 | 40 / 58 | |
| `E4-03-01@80M` (anchor) | 2073.5 | 31 / 30 | — | |
| `E4-08-05@80M` (M2d) | 1995.3 | 31 / 22 | 35 / 48 | |
| **`E4-24-05@80M`** | 1923.6 | 31 / 3 | 45 / 17 | **−71.7** |
| **`E4-24-03@80M`** | 1920.2 | 22 / 12 | 23 / 26 | **−153.9** |

## Verdict

**Both rejected** under the adoption rule (a win at 80M on both seeds): each loses on both.

* **Width is not the binding constraint — it is a cost.** Doubling the entity projection loses
  129–291 Elo, and the loss *grows* with budget on seed 3. By the gate in the P22 plan, a null or
  a loss means capacity is not what the card path lacks.
* **P22-a loses at 80M but closes most of the gap by 160M.** Seed 3 gains +233.5 from 80M to 160M
  against M2d's +132.8 on the same seed, which leaves it −53 behind. That is one seed, so it is a
  slower start and not a steeper slope until a second seed is measured past 80M.
* **Both arms' 80M US seat is weak.** Against M2d@160M they win 1–12% as US, against M2d's own
  22–32%. That is not a pinned side: `adv_std_raw` stays at or above 0.138 in both P22 arms and
  above 0.09 in `E4-23-05`. **`E4-23-03` did touch the collapsed band once**, with a minimum of
  0.0247, and was not screened further; its 160M rating may include an entry episode.
