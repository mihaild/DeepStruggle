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

    <engine>-<attempt>-<seed>-<steps>

for example **`E3-10-21-80M`**: engine revision E3, its tenth configuration, seed 21, 80 million
steps. A continuation keeps everything and changes the budget: `E3-10-21-160M`.

- **engine letter** — bumped whenever `engine/` or `bindings/` changes the decision stream.
  Checkpoints from different engine letters may be *evaluated* together, but the result is a
  cross-engine comparison and must be labelled as one.
- **attempt number** — one row in the table below, fixing observation, architecture,
  intervention and recipe. Numbered **from 01 within each engine letter**, so both halves of the
  prefix carry information. It does **not** include the seed.
- **seed** — its own field, the last two digits of the sampling seed (`21` is 20260921). Two
  runs of one configuration differ only here, so `E3-10-21` beside `E3-10-22` is visibly a seed
  pair while `E3-09-21` beside `E3-10-21` is visibly an intervention difference. The seed has
  repeatedly mattered more than the intervention, so it is worth being able to see at a glance
  which kind of difference a comparison is.
- **steps** — the budget the checkpoint was taken at, never omitted. Nothing is comparable
  across budgets.

The table is the contract. A short name is only meaningful with it, so it lives in the repository
and is updated in the same commit as the run.

## The directory carries the name

A short name that lives only in this file is one someone has to look up. The directory is what
every later command quotes, so it is where the name has to be:

    data/checkpoints/E3-12-21_20260912_181622

`tools/train.py --run-name E3-12-21` builds it, validates it against the scheme, and records it
in `metadata.json` as `run_name`. **The steps field is deliberately absent from the directory**:
one directory holds every budget of a lineage -- `p1_scalar_nofilter` holds 80M, 160M and 240M --
so a steps field in the directory name is a claim that goes stale the first time the run is
continued. Each snapshot's own filename carries its budget.

Passing `--run-name` together with an `--output-dir` whose basename does not contain it is an
error rather than a preference, because the quiet version of that writes one arm's weights into a
directory named for another.

The directories named in the tables below predate this and are left alone: renaming them would
break every path in `research/` that cites one, which is a worse failure than an unhelpful name.
The `directory` column is what maps them back.

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

| # | architecture | intervention | seeds | directory |
|---:|:---|:---|:---|:---|
| 01 | v2 | baseline (arm H) | — | `arm_H_v23_corrected` |
| 02 | v2 | baseline (arm H2) | 21 | `arm_H2_v23_seedB`, `arm_H2_cont_160to240`, `arm_H2_cont_240to480` |
| 03 | v2 | `eta` 0, no KL to the reference (arm I) | — | `arm_I_no_kl` |

`E2-02-21` is one lineage across four budgets, so the strongest checkpoint in the project is
**`E2-02-21-480M`**.

### E3

Observation v2.3, `blunder_aware`, K=40, `eta` 0.1, 512 envs, cold start unless noted.
"Intervention" is what distinguishes the row from the control, E3-01.

| # | architecture | intervention | seeds | budgets | directory |
|---:|:---|:---|:---|:---|:---|
| 01 | v2 | — (control) | 21 | 80M, 160M, 240M* | `p1_scalar_nofilter` |
| 02 | v2 | categorical head, target in normalised units | 21 | **VOID** | `..._VOID_unscaled_target` |
| 03 | v2 | categorical head, `v_vp` in real VP | 21 | **VOID** | `..._VOID_vp_scale` |
| 04 | v2 | categorical head, `--vf-coef` sweep | 21 | **VOID** | `..._VOID_vf_coef` |
| 05 | v2 | categorical head, derived `v_win` | 21 | **VOID** | `..._VOID_derived_baseline` |
| 06 | v2 | categorical head, `--value-dist-coef 0.02` | 21 | 80M | `p1_categorical_nofilter` |
| 07 | v2 | `--adv-filter-quantile 0.5` | 21, 22 | 80M, 160M | `p1_scalar_filter_seed2026092*` |
| 08 | v2 | `--window-provoked-defcon` | 21, 22 | 80M | `p1_window_provoked_seed2026092*` |
| 09 | **mlp** | `--arch mlp`, backbone control | 21, 22 | 80M | `p1_mlp_backbone_seed2026092*` |
| 10 | v2 | `--identity-dim 16` | 21, 22 | 80M, 160M | `p1_identity_seed2026092*` |
| 11 | **mlp** | `--arch mlp --drop-static` | 21, 22 | 80M | `p1_mlp_static_seed2026092*` |
| 12 | v2 + identity | `--self-transform` | 21, 22 | 80M | `E3-12-2*_<ts>` |
| 13 | v2 + identity | `--self-transform --attn-readout 64` | 21, 22 | 80M | `E3-13-2*_<ts>` |

Seed `21` is 20260921 and `22` is 20260922.

**E3-12 and E3-13 carry `--identity-dim 16`, and their control is E3-10, not E3-01.** Identity
is part of the recipe as of §21.8, so an arm that omitted it would be measuring identity again.

Both target what `research/metrics.md` §21.12 localised: a country's exact influence is
recoverable from its own raw observation slots 97% of the time, from its post-GraphConv token
63%, and from the pooled 512-float trunk essentially never. E3-12 gives each graph layer a second
weight matrix applied to the node itself, so a country need not be averaged with its neighbours;
E3-13 adds an attention read-out at the end of the trunk, where the state vector queries the 84
country and 110 card tokens -- each concatenated with its raw slots, because the token is itself
already damaged -- and folds the result back into the trunk.

Predictions were recorded before the runs, so the result could not be read backwards: E3-12
should lift the `gconv` rungs and leave the trunk near zero; E3-13 should lift the trunk toward
the token's 63%. Elo may not move at all, and that would itself be the finding.

**Outcome: the first held, the second failed, and Elo moved.** E3-12 lifted `gconv1` recovery from
60% to 93-94% on both seeds and is worth **+53 and +83 Elo**. E3-13's read-out left the trunk
where E3-12 already put it and cost the whole gain, **−14 and −23**. A single-query read-out is
still a pooling operation, so it could not have done otherwise. `--self-transform` is adopted;
`--attn-readout` is not. See `research/metrics.md` §21.13.

E3-11 is an ablation *of* E3-09 rather than of the control: it drops the 1,364 observation slots
that never vary (35.7% of the input, 6.6M parameters against E3-09's 8.0M), which a positional
reader should not need because it recovers entity identity from the offset. It is the row most
likely to be misread, so what it measured is written next to it: **-40 and -0 Elo** against E3-09
across its two seeds, a mean inside a seed spread of 40, at identical throughput (64.4k vs
65.5k steps/s). The static mask was verified independently -- 0 of the 1,364 claimed-constant
dimensions take a second value across 16,800 states, counted by distinct values rather than by
standard deviation. Not adopted.

Continuation legs use a fresh sampling seed so they diverge rather than replay: E3-01-21's legs
used 20260931 and 20260941, and E3-10's use 20260951 and 20260952. Those are properties of a leg
rather than of the configuration, so the table records them and the name does not.

The four VOID rows are numbered rather than omitted. They exist on disk, they consumed 6.1 of the
session's 22 GPU-hours, and a table whose purpose is to identify a directory unambiguously cannot
leave four directories out of it. Each died to a units or scale bug; see the P1 log.

## What the naming buys

The session that produced this table rated nine E3 configurations against **E2-02-480M** and
called it "the strongest arm we have". That is still the right reference — it is the strongest —
but `E3-07-21-80M vs E2-02-21-480M` says on its face that the engine differs, where
`p1_scalar_filter_seed20260921 vs arm_H2_cont_240to480` did not.

Three rules follow, and all three were broken at least once before the table existed:

1. **Never compare across budgets.** The suffix makes a violation visible.
2. **Label cross-engine comparisons.** An E2 checkpoint evaluated on an E3 engine is playing a
   game it never trained on; the handicap is small for the two mandatory-choice cards but it is
   not zero, and it biases in favour of the E3 arm.
3. **A seed is a field, not a detail.** `E3-07-21` and `E3-07-22` differ only there and landed
   11.8 points apart on anchor win rate while being +3 Elo apart head to head. Giving it its own
   field is what makes that visible without opening the table.
