# Run nomenclature — the naming scheme

**This file is the scheme and nothing else.** The arms themselves — what each one varied, its
seeds, its budgets, its checkpoint directory and where its result is written up — are in
[`../runs.md`](../runs.md), the registry. They were one file until 2026-09-16, and mixing them
meant that looking up what `E3-20` *is* required reading three pages of prose about what `E3-14`
*measured*.

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
- **attempt number** — one row in [`../runs.md`](../runs.md), fixing observation, architecture,
  intervention and recipe. Numbered **from 01 within each engine letter**, so both halves of the
  prefix carry information. It does **not** include the seed.
- **seed** — its own field, the last two digits of the sampling seed (`21` is 20260921). Two
  runs of one configuration differ only here, so `E3-10-21` beside `E3-10-22` is visibly a seed
  pair while `E3-09-21` beside `E3-10-21` is visibly an intervention difference. The seed has
  repeatedly mattered more than the intervention, so it is worth being able to see at a glance
  which kind of difference a comparison is.
- **steps** — the budget the checkpoint was taken at, never omitted. Nothing is comparable
  across budgets.

The registry is the contract. A short name is only meaningful with it, so it lives in the
repository and is updated in the same commit as the run.

## The directory carries the name

A short name that lives only in a table is one someone has to look up. The directory is what
every later command quotes, so it is where the name has to be:

    data/checkpoints/E3-12-21_20260912_181622

`tools/train.py --run-name E3-12-21` builds it, validates it against the scheme, and records it
in `metadata.json` as `run_name`. **The steps field is deliberately absent from the directory**:
one directory holds every budget of a lineage — `p1_scalar_nofilter` holds 80M, 160M and 240M —
so a steps field in the directory name is a claim that goes stale the first time the run is
continued. Each snapshot's own filename carries its budget.

Passing `--run-name` together with an `--output-dir` whose basename does not contain it is an
error rather than a preference, because the quiet version of that writes one arm's weights into a
directory named for another.

The directories predating the scheme are left alone: renaming them would break every path in
`research/` that cites one, which is a worse failure than an unhelpful name. The `directory`
column of [`../runs.md`](../runs.md) is what maps them back.

## Engine revisions

| letter | from | what changed in the decision stream |
|:---|:---|:---|
| **E1** | before `cff2344` | pre-starred-card fix: a starred card spent for Operations was deleted from the game. Arms A–G and everything older. Not runnable — the observation layouts they used are gone. |
| **E2** | `cff2344`, `25d9b70` | starred-card fix, observation v2.3. Arms H, H2, I. |
| **E3** | `1a3b782`, `a18ceab` | the Aldrich Ames discard and the Star Wars pick made mandatory. Everything from P1 onward. |

Each boundary is a *batch* rather than a commit — eight further engine and observation commits
landed with the starred-card fix on 2026-09-10 — and what each one changed, with what it
invalidated, is [`../findings/engine/engine_revisions.md`](../findings/engine/engine_revisions.md).
What a letter bump does and does not invalidate is
[`what_survives_an_engine_change.md`](what_survives_an_engine_change.md).

## Attempt numbers restart with each engine

An attempt number is unique **within its engine letter**, not across the project. The first
arm on a new engine is `01`.

The first draft of this file numbered them globally, which put the E3 control at `E3-03` with
nothing before it on that engine — and made the letter redundant, since a global `03` already
implies E3. Restarting per engine keeps both halves of the name carrying information.

## When the letter does *not* bump

A commit that touches `engine/` bumps the letter only if it changes the decision stream the
agent sees. That is a claim to be measured, not assumed, and the measurement belongs in the
registry row beside the arm that depends on it. The E3-20/E3-21 pairing is the worked example:
`b6874af` and `9f78026` changed `engine/` between the two runs, and the letter was held at E3
because no `ROLL_DIE` node is ever handed to an agent (0 in 879 single-env and 12,800 vectorized
env-steps) and the runner always forces die 0. See [`../runs.md`](../runs.md), *E3-21*.

P14 is the second worked example and the stronger one: the mask/step collapse and the Missile
Envy forced-play fix (`5938e52`) were replayed against a build of `a09e15a` over 1,068 games and
385,812 steps under four policies, comparing outcome, length, chosen action and the legal mask's
hash at every step, with zero divergences. The letter was held at E3 on that measurement. Diffing
the **mask** as well as the action is the part worth copying — an engine change can alter what is
offered without altering what a deterministic chooser picks, and a policy sees the offer. Method:
[`../findings/engine/engine_change_decision_stream.md`](../findings/engine/engine_change_decision_stream.md).

## What the naming buys

The session that produced the first version of the registry rated nine E3 configurations against
**E2-02-480M** and called it "the strongest arm we have". That is still the right reference — it
is the strongest — but `E3-07-21-80M vs E2-02-21-480M` says on its face that the engine differs,
where `p1_scalar_filter_seed20260921 vs arm_H2_cont_240to480` did not.

Three rules follow, and all three were broken at least once before the scheme existed:

1. **Never compare across budgets.** The suffix makes a violation visible.
2. **Label cross-engine comparisons.** An E2 checkpoint evaluated on an E3 engine is playing a
   game it never trained on; the handicap is small for the two mandatory-choice cards but it is
   not zero, and it biases in favour of the E3 arm.
3. **A seed is a field, not a detail.** `E3-07-21` and `E3-07-22` differ only there and landed
   11.8 points apart on anchor win rate while being +3 Elo apart head to head. Giving it its own
   field is what makes that visible without opening the registry.
