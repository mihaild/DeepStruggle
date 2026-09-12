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

for example **`E3-07-80M`**: engine revision E3, configuration 07, 80 million steps. A
continuation appends its own budget: `E3-07-160M`.

- **engine letter** — bumped whenever `engine/` or `bindings/` changes the decision stream.
  Checkpoints from different engine letters may be *evaluated* together, but the result is a
  cross-engine comparison and must be labelled as one.
- **attempt number** — one row in the table below, fixing observation, architecture,
  intervention, recipe **and seed**. Two runs differing only by seed get two numbers, because
  the seed has repeatedly mattered more than the intervention.
- **steps** — the budget the checkpoint was taken at, never omitted. Nothing is comparable
  across budgets.

The table is the contract. A short name is only meaningful with it, so it lives in the repository
and is updated in the same commit as the run.

## Engine revisions

| letter | from | what changed in the decision stream |
|:---|:---|:---|
| **E1** | before `cff2344` | pre-starred-card-fix: a starred card spent for Operations was deleted from the game. Arms A-G. Not runnable — the observation layouts they used are gone. |
| **E2** | `cff2344`, `25d9b70` | starred-card fix. Arms H, H2, I and everything up to the mandatory-choice fixes. |
| **E3** | `1a3b782`, `a18ceab` | Aldrich Ames discard and the Star Wars pick made mandatory. Everything in P1 onward. |

## Configurations

All E3 rows below are observation v2.3, `blunder_aware`, K=40, `eta` 0.1, 512 envs, cold start.
"Intervention" is what distinguishes the row from the control.

| # | engine | architecture | intervention | seed | old directory |
|---:|:---|:---|:---|---:|:---|
| 01 | E2 | v2 | — (baseline) | 20260921 | `arm_H2_v23_seedB` |
| 02 | E2 | v2 | continuation of 01 | 20260921 | `arm_H2_cont_160to240`, `_240to480` |
| 03 | E3 | v2 | — (control) | 20260921 | `p1_scalar_nofilter` |
| 04 | E3 | v2 | categorical VP head, `--value-dist-coef 0.02` | 20260921 | `p1_categorical_nofilter` |
| 05 | E3 | v2 | `--adv-filter-quantile 0.5` | 20260921 | `p1_scalar_filter_seed20260921` |
| 06 | E3 | v2 | `--adv-filter-quantile 0.5` | 20260922 | `p1_scalar_filter_seed20260922` |
| 07 | E3 | v2 | `--window-provoked-defcon` | 20260921 | `p1_window_provoked_seed20260921` |
| 08 | E3 | v2 | `--window-provoked-defcon` | 20260922 | `p1_window_provoked_seed20260922` |
| 09 | E3 | **mlp** | backbone control, `--arch mlp` | 20260921 | `p1_mlp_backbone_seed20260921` |
| 10 | E3 | **mlp** | backbone control, `--arch mlp` | 20260922 | `p1_mlp_backbone_seed20260922` |
| 11 | E3 | v2 | `--identity-dim 16` | 20260921 | `p1_identity_seed20260921` |
| 12 | E3 | v2 | `--identity-dim 16` | 20260922 | `p1_identity_seed20260922` |

Continuations keep their configuration number: run 03 exists at `E3-03-80M` and `E3-03-160M`,
and its second leg used a fresh sampling seed, which the table records rather than the name.

## What the naming buys

The session that produced this table rated nine E3 configurations against **E2-02-480M** and
called it "the strongest arm we have". That is still the right reference — it is the strongest —
but `E3-05-80M vs E2-02-480M` says on its face that the engine differs, where
`p1_scalar_filter_seed20260921 vs arm_H2_cont_240to480` did not.

Three rules follow, and all three were broken at least once before the table existed:

1. **Never compare across budgets.** The suffix makes a violation visible.
2. **Label cross-engine comparisons.** An E2 checkpoint evaluated on an E3 engine is playing a
   game it never trained on; the handicap is small for the two mandatory-choice cards but it is
   not zero, and it biases in favour of the E3 arm.
3. **A seed is a configuration, not a detail.** Runs 05 and 06 differ only by seed and landed
   11.8 points apart on anchor win rate while being +3 Elo apart head to head.
