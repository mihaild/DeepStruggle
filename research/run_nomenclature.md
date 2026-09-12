# Run nomenclature

Every number in this project is only meaningful relative to four things, and three of them have
silently changed under a comparison at least once:

- the **engine**, which decides what game is being played;
- the **observation**, which decides what the network is told;
- the **recipe** — reward, loss, trainer flags — and the **architecture**;
- the **budget** in steps, and the **seed**.

`arm_H2_cont_240to480` says none of them. It took a `git merge-base` against two commit hashes to
establish that H2 predates the Aldrich Ames and Star Wars fixes and so was trained on a different
game from everything in P1 — which makes every "rated against H2" number a cross-engine
comparison. That should have been legible from the name.

## The scheme

A run is named

    <engine><attempt>-<steps>

for example **`E3-07-80M`**: engine revision E3, its seventh configuration, 80 million steps. A
continuation keeps the attempt number and changes the budget: `E3-07-160M`.

- **engine letter** — bumped whenever `engine/` or `bindings/` changes the decision stream.
  Checkpoints from different engine letters may be *evaluated* together, but the result is a
  cross-engine comparison and must be labelled as one.
- **attempt number** — one row in the table below, fixing observation, architecture,
  intervention, recipe **and seed**. Numbered **from 01 within each engine letter**, so both
  halves of the prefix carry information. Two runs differing only by seed get two numbers,
  because the seed has repeatedly mattered more than the intervention.
- **steps** — the budget the checkpoint was taken at, never omitted. Nothing is comparable
  across budgets.

The table is the contract. A short name is only meaningful with it, so it lives in the repository
and is updated in the same commit as the run.

## Engine revisions

| letter | from | what changed in the decision stream |
|:---|:---|:---|
| **E1** | before `cff2344` | pre-starred-card fix: a starred card spent for Operations was deleted from the game. Arms A-G and everything older. Not runnable -- the observation layouts they used are gone -- and not numbered below. |
| **E2** | `cff2344`, `25d9b70` | starred-card fix, observation v2.3. Arms H, H2, I. |
| **E3** | `1a3b782`, `a18ceab` | the Aldrich Ames discard and the Star Wars pick made mandatory. Everything from P1 onward. |

## Attempt numbers restart with each engine

An attempt number is unique **within its engine letter**, not across the project. The first
arm on a new engine is `01`.

The first draft of this file numbered them globally, which put the E3 control at `E3-03` with
nothing before it on that engine -- and made the letter redundant, since a global `03` already
implies E3. Restarting per engine keeps both halves of the name carrying information.

### E2

| # | architecture | intervention | seed | directory | budgets |
|---:|:---|:---|---:|:---|:---|
| 01 | v2 | baseline (arm H) | — | `arm_H_v23_corrected` | 80M |
| 02 | v2 | baseline, second seed (arm H2) | 20260921 | `arm_H2_v23_seedB`, `arm_H2_cont_160to240`, `arm_H2_cont_240to480` | 80M, 160M, 240M, 480M |
| 03 | v2 | `eta` 0, no KL to the reference (arm I) | — | `arm_I_no_kl` | 80M |

E2-02 is one lineage across four budgets: a continuation keeps its attempt number and changes
only the step suffix, so `E2-02-480M` is the strongest checkpoint in the project.

### E3

Observation v2.3, `blunder_aware`, K=40, `eta` 0.1, 512 envs, cold start unless noted.
"Intervention" is what distinguishes the row from the control, E3-01.

| # | architecture | intervention | seed | directory | budgets |
|---:|:---|:---|---:|:---|:---|
| 01 | v2 | — (control) | 20260921 | `p1_scalar_nofilter` | 80M, 160M, 240M* |
| 02 | v2 | categorical head, target in normalised units | 20260921 | `p1_categorical_nofilter_VOID_unscaled_target` | **VOID** |
| 03 | v2 | categorical head, `v_vp` in real VP | 20260921 | `p1_categorical_nofilter_VOID_vp_scale` | **VOID** |
| 04 | v2 | categorical head, `--vf-coef` sweep | 20260921 | `p1_categorical_nofilter_VOID_vf_coef` | **VOID** |
| 05 | v2 | categorical head, derived `v_win` | 20260921 | `p1_categorical_nofilter_VOID_derived_baseline` | **VOID** |
| 06 | v2 | categorical head, `--value-dist-coef 0.02` | 20260921 | `p1_categorical_nofilter` | 80M |
| 07 | v2 | `--adv-filter-quantile 0.5` | 20260921 | `p1_scalar_filter_seed20260921` | 80M, 160M |
| 08 | v2 | `--adv-filter-quantile 0.5` | 20260922 | `p1_scalar_filter_seed20260922` | 80M, 160M |
| 09 | v2 | `--window-provoked-defcon` | 20260921 | `p1_window_provoked_seed20260921` | 80M |
| 10 | v2 | `--window-provoked-defcon` | 20260922 | `p1_window_provoked_seed20260922` | 80M |
| 11 | **mlp** | `--arch mlp`, backbone control | 20260921 | `p1_mlp_backbone_seed20260921` | 80M |
| 12 | **mlp** | `--arch mlp`, backbone control | 20260922 | `p1_mlp_backbone_seed20260922` | 80M |
| 13 | v2 | `--identity-dim 16` | 20260921 | `p1_identity_seed20260921` | 80M |
| 14 | v2 | `--identity-dim 16` | 20260922 | `p1_identity_seed20260922` | 80M |

\* E3-01-240M is training as this is written; its legs used fresh sampling seeds 20260931 and
20260941, which the table records rather than the name.

The four VOID rows are numbered rather than omitted. They exist on disk, they consumed 6.1 of
the session's 22 GPU-hours, and a table whose purpose is to identify a directory unambiguously
cannot leave four directories out of it. Each died to a units or scale bug; see the P1 log.

## What the naming buys

The session that produced this table rated nine E3 configurations against **E2-02-480M** and
called it "the strongest arm we have". That is still the right reference — it is the strongest —
but `E3-07-80M vs E2-02-480M` says on its face that the engine differs, where
`p1_scalar_filter_seed20260921 vs arm_H2_cont_240to480` did not.

Three rules follow, and all three were broken at least once before the table existed:

1. **Never compare across budgets.** The suffix makes a violation visible.
2. **Label cross-engine comparisons.** An E2 checkpoint evaluated on an E3 engine is playing a
   game it never trained on; the handicap is small for the two mandatory-choice cards but it is
   not zero, and it biases in favour of the E3 arm.
3. **A seed is a configuration, not a detail.** E3-07 and E3-08 differ only by seed and landed
   11.8 points apart on anchor win rate while being +3 Elo apart head to head.
