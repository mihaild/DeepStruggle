# Run nomenclature — the naming scheme

The scheme only; the arms themselves are [`../runs.md`](../runs.md). A number means something only
relative to the **engine**, the **observation**, the **architecture** and **recipe**, the **budget**
and the **seed** — and `arm_H2_cont_240to480` said none of them (it took a `git merge-base` to learn
that H2 was trained on a different game from everything in P1). The name has to say them.

## The scheme (since 2026-10-07): engine, architecture, recipe, seed, and the path taken

A run is named by the four things a number depends on, each a labelled field, and by every
branch point on the way to it:

    E7-A4-R1-S44                                   from scratch
    E7-A4-R1-S44@4390M+S45                         its 4,390M state, continued under seed 45
    E7-A4-R1-S44@4390M+A7-R23@4800M+R24            two changes in a row
    E7-A4-R1-S44-2                                 a same-seed replicate of the first

- **`E`** — the engine revision, as below (`E8` since 2026-10-20, with ENG-5's alternating deal folded in on 2026-10-23 -- owner -- so an E8 run launched before then dealt the old way; a minor version is `E4.1`, and `E8.1` is E8 with ENG-9's rule bit `opp_event_gated`, the default for a
  new network since 2026-11-06 -- owner -- which `tools/scripts/run_codes.py --run` checks against the
  run's recorded `obs_rules`). An
  engine revision can also move a training default: since E8 a new network reads
  `headline_reveal` unless its A code lists `--obs-features`.
- **`A`** — the **architecture**: the network's shape, what it reads (its observation features)
  and its policy heads. A code, defined in [`../architectures_and_recipes.md`](../architectures_and_recipes.md).
- **`R`** — the **recipe**: every other training setting -- reward, loss, opponents, schedules,
  forcing. A code in the same file. An auxiliary head trained only as a target (the card-event
  head) is recipe: the policy never reads it.
- **`S`** — the **seed** of the environment, sampling and opponent pool.

**Plans and their experiments use the same letters** (owner, 2026-10-15): a plan combining many
ideas is `P<n>` (`research/plans/`); an experiment under it that trains nothing is `P<n>a`,
`P<n>b`, …; one that trains a model is named by the code it introduces -- an `R` for a training
approach, an `A` for an architecture -- reserved in the plan and registered here at launch. `E`
is never an experiment label.

**Datasets, evaluators and two-network players** (owner, 2026-10-17; [P34](../plans/P34_search_centric_evaluator.md)):

- **`D`** -- a **labelling recipe**, the one letter added: the flag set of the labelling tool
  (source temperature, decisions kept, worlds, stop, label columns, device), registered like an
  R code ([`../datasets.md`](../datasets.md) until `run_codes.py` reads it).
- A **dataset** is a model under a labelling recipe, spelled like a player under a search:
  `<generator model>~D1(seed=44,games=20000)`. The generator played the positions and the
  playouts, so the generation is in the name.
- An **evaluator** -- a value-only network trained on a dataset -- is an ordinary run name: an `A`
  code for its shape, an `R` code for the supervised recipe, which pins the dataset the way R32
  pins a teacher checkpoint. From scratch `E7-A<n>-R<n>-S44`; fine-tuned from an RL network's
  weights `<RL model>+A<n>-R<n>`.
- A **two-network player** names its leaf network inside the search spec:
  `<policy model>~rollout(…,leaf=own,leaf_net=<evaluator model>)` -- `leaf=` is the side the leaf is
  read from, `leaf_net=` the network that reads it.

**A code is a set of flags, not a description.** `tools/scripts/run_codes.py` reads each run's
`metadata.json`, splits its non-default `tools/train.py` flags into architecture, recipe and
bookkeeping (output, cadence, evals, budget, seeds), and gives each distinct set a code --
recorded with its description in [`../run_codes.json`](../run_codes.json). Two runs with one code
therefore cannot quietly differ in a flag, which is how lineage-critical flags drifted four times
(CLAUDE.md, invariant 15). A and R are numbered across the whole project, not per engine:
`E6-A4-R1-S44` and `E7-A4-R1-S44` are visibly one recipe on two engines. What flags cannot
show -- a code change under the same flags, a league whose exploiter was not training -- is
recorded by hand as an `override` in the JSON and makes a recipe of its own.

**`@<n>M` is always an absolute step on the lineage's own clock**, never a length -- the clock the
snapshots already use, so a typo cannot silently shift every later number. In a run's name every
`@` is a **branch point**: `+` and the fields that change, in E, A, R, S order, naming only what
changed. A **continuation that changes nothing keeps its name and its directory**; a re-run of a
from-scratch run takes a replicate index `-2`. A branch of such a replicate keeps the index on its
root, before the branch points (`E7-A4-R1-S44-2@4390M+R10`, since 2026-10-25); an index after a
branch point still ends a name.

**A run is one directory (since 2026-10-07).** `tools/train.py --resume <run-dir> --run-name <its
name>`, from the run's newest state, writes into that directory: metrics and TensorBoard are
appended, snapshots join the earlier ones, the untrained `snapshot_0s.pt` is kept, and
`metadata.json` keeps the current leg's settings at the top level (what every tool reads) with every
leg -- commit, flags, description, `start_steps`, `train_steps` -- under `legs`. A branch (any change
of name), a resume from an earlier state than the newest, and an old run's link directory get a
new directory as before; `--new-directory` asks for one explicitly, and a run still writing is
refused. Runs launched before this have one directory per leg, all under one name.

| what happened | name |
|:---|:---|
| plain run, 0 → 2000M | `E7-A4-R1-S47` |
| a head added at 2000M | `E7-A4-R1-S47@2000M+A8` |
| two more seeds from 2600M of that | `E7-A4-R1-S47@2000M+A8@2600M+S45`, `…+S46` |
| a recipe change from 820M | `E7-A4-R1-S44@820M+R14` (E7-13-44) |
| an engine change | `E6-A1-R1-S44@560M+E7` (E7-06-44: the deep E6 run continued on E7) |
| back to the plain recipe | `E7-A4-R1-S44@4390M+A7-R23@4800M+R24@5200M+R1` (E7-32-44) |

### A model: one snapshot, an SWA, or a soup

A *label* names a model, so it ends in a **selector** -- `@4800M` for one snapshot, `@4720..4800M`
for the SWA of every snapshot in that range -- and may put a **group** after the last branch point,
which is a soup of the alternatives:

    E7-A4-R1-S44@4800M                                   one snapshot
    E7-A4-R1-S44@4720..4800M                             the SWA of that range
    E7-A4-R1-S44@4390M+(S44,45,46)@4800M                 the best model (2026-10-06): a soup
    E7-A4-R1-S44@4390M+(S44,45,46)@4720..4800M           the same soup made from SWAs
    E7-A4-R1-S44@870M+(S44,S45,R8,R9)@1200M              the first shallow soup: seeds and recipes
    E7-A2-R2-S5@800M+R3@1200M+(S6,R4-S7)@1500..1600M     two branches of 1200M, each an SWA
    (E7-A4-R1-S44,E7-A4-R10-S44)@1200M                   two runs with no common state

- An item that changes nothing (`S44` on a seed-44 parent) is the **parent's own continuation**;
  write it out, since an empty item is easy to miss. A bare number repeats the previous item's
  letter: `S44,45,46`.
- The selector applies to every item. Groups nest, and a top-level group lists whole runs.
- A soup has **one A and one E**: weights of different architectures cannot be averaged, and a
  soup across engines plays no one game. The parser refuses both.
- A soup of runs with no common state (the top-level group) parses, and
  `shares_a_state()` says it is suspect: independent runs are not aligned, and their average is
  usually broken. Every soup that has worked here branched from one checkpoint.
- **Parentheses, not braces.** Both shells expand `{a,b}` silently into separate words, and `[a]` is
  a filename pattern bash silently rewrites when a file matches; an unquoted `(` is a syntax error
  in bash and zsh -- loud, never wrong. Quote a label with a group on the command line. Directories
  never contain one: a run is a single path, so a soup or SWA is a model file named by its label.
- In a URL query string `+` decodes to a space: encode names there.

The grammar's one parser is `ai/training/run_name.py` (`parse_run_name`, `parse_label`,
`is_run_name`); `tools/train.py --run-name`, the league driver and `tools/lib/checkpoint_id.py`
use it, and a snapshot's label (`E7-A4-R1-S44@4390M+S45@4800M`) parses back to its run. A league
exploiter generation is a branch of the main agent at its reset state: pass the league driver
`--exploiter-name '<main>@{step}M+R3'` and each generation runs as `<main>@6410M+R3`,
`<main>@6460M+R3`, ...

### Launching a run

1. Derive the flags from a healthy run (`tools/scripts/launch_flags.py`, invariant 15).
2. Find the A and R codes those flags are in [`../architectures_and_recipes.md`](../architectures_and_recipes.md).
   A setting no code has is a new code: launch, then `tools/scripts/run_codes.py --update`
   assigns it and you write its description into `run_codes.json`.
3. Name the run from its root and branch points, and launch with `--run-name`.
4. After launch, `tools/scripts/run_codes.py --run <dir>` checks the name against the flags the run
   recorded and fails if they disagree.

### The old names

Directories launched before 2026-10-07 keep their old names, because research cites them by path;
[`../run_name_map.md`](../run_name_map.md) gives each one's name in this grammar, generated from
its metadata. Each also has a **link directory under its new name** -- `E7-A4-R1-S44_20261005_174535`
beside `E7-20-44_20261005_174535`, every file a relative symlink (`tools/scripts/run_codes.py
--link`) -- so a checkpoint is labelled, resolved and resumed by its new name; a run resumed through
one is still traced to the real directory. Soups and line SWAs kept outside run directories are
linked as `data/checkpoints/_models/<label>.pt` (the map is `models` in
[`../run_codes.json`](../run_codes.json)), and a file named by its label is labelled by it. The current tree (E6 and E7) is mapped; the archived E3–E5 ladders are not. The old
scheme, below, still parses for the lineages that carry it.

## The old scheme (before 2026-10-07): attempt numbers

Directories launched earlier keep their old names because research cites them by path;
[`../run_name_map.md`](../run_name_map.md) maps each onto the grammar above, and `LEGACY_RUN_NAME_RE`
(`ai/training/run_name.py`) still parses them. The old grammar was
`E<engine>-<attempt>-<seed>[-<replicate>][-<M>M.<seed>]...`, e.g. `E3-10-21`:

* **attempt** — one registry row fixing observation, architecture, intervention and recipe,
  numbered from 01 **within each engine letter**; seeds are the last field, so `E3-10-21` beside
  `E3-10-22` is visibly a seed pair and `E3-09-21` beside `E3-10-21` an intervention difference;
* **replicate** — a same-seed re-run appends `-2`, `-3`, never the next seed number (`E4-08-07/08`
  were launched as "seeds 7 and 8" when both were seed-1 re-runs, which would have read a 1-in-5
  collapse rate as 1 in 7; renamed `E4-08-01-2/-3` on 2026-09-22);
* **continuation** — keeps the name, one directory per leg (`E4-08-03_<timestamp>` for 0→80M,
  80→160M, 160→240M), snapshots read `E4-08-03@160M`; attempt numbers once spent on continuations
  (E4-11, 12, 19–22, 25) were renamed into their lineages and are retired;
* **branch** — a resume under a new seed: `E4-17-06-50M.11` (`.`, because `_` separates the
  timestamp);
* **minor version** — `E4.1` (P23) is E4 with an opt-in action view: same game, so E4 and E4.1
  numbers compare directly. `E8.1` is E8 with the rule bit `opp_event_gated` (ENG-9): the unseated
  engine is E8's, and an E8.1 network plays its own seat under the gated rule, so E8 and E8.1
  networks play each other directly.

## Rules that hold in both schemes

* **The engine letter bumps when `engine/` or `bindings/` changes the decision stream** — a claim to
  measure, not assume (P14 held at E3 after 385,812 steps with zero divergences, masks included;
  [`../findings/engine/engine_change_decision_stream.md`](../findings/engine/engine_change_decision_stream.md)).
  The letters and what each changed: [`../findings/engine/engine_revisions.md`](../findings/engine/engine_revisions.md);
  what a bump invalidates: [`what_survives_an_engine_change.md`](what_survives_an_engine_change.md).
* **Never compare across budgets; label cross-engine comparisons** (an E2 checkpoint on E3 plays a
  game it never trained on).
* **A checkpoint is named by its run, never by its filename.** Snapshot filenames collide by
  construction (`E4-01-01` and `E4-02-01` both hold `snapshot_150011904steps.pt`);
  `tools/lib/checkpoint_id.py` turns a path into `E4-02-01@150M`, labels `snapshot_final.pt` by the
  budget in `metadata.json`, and `unique_labels` raises on a clash rather than suffixing `#1`/`#2`.
* **The directory carries the name** (`<run-name>_<timestamp>`, no step field: budgets outlive a
  directory). `--run-name` with an `--output-dir` whose basename lacks it is an error.
