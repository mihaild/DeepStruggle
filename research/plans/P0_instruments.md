# P0 — Instruments

**Status:** queued (second revision 2026-09-10, after the starred-card fix and layout v2.3)
**Gate:** none. Everything after this is read through these, so they come first.
**Blocked in this checkout** until the engine is rebuilt — see *Before anything runs*.
**Needs approval:** none. No engine, bindings, observation or trainer change. One assertion is
added to `ColdWarNetV2.forward` (§1.3); it changes no layout, no width and no content.

## Goal

Five measurements that the later steps are judged by, taken once per checkpoint so every later
arm has a before/after. Three are new probes, two are existing instruments pointed at new states.

The baseline is **not** one checkpoint. A probe number is only comparable inside a
(layout, engine, step-budget) cell, so the baseline is a table (*Procedure*).

## Why

The goal is "no simple mistakes", and the mistakes §25 actually names are not measured by
anything in `ai/eval/`:

- setup that ignores Poland / West Germany;
- footholds left exposed to Voice of America;
- **and, new in §25, the one the owner calls the gap to a mediocre human: H2 @160M can play a
  tactic when the card is in front of it and cannot choose which card to spend it on.** It spent
  UN Intervention on Tear Down this Wall at AR3 and then played Grain Sales raw at AR7 and lost
  the game to the DEFCON that followed.

Elo does not move for any of these (`experiments.md` §4.4: a behaviour can be worth two points of
win rate and still be the thing a human notices first). And P2 spends an arm on chance-aware
targets; whether that is worth an arm depends on how much of the return variance is chance in the
first place, which nobody has measured.

## What is true now, and what it does to this plan

This file was written against v2.2 and arm G. Both are gone. The reset:

- **The starred-card bug** — a starred card spent for Operations was deleted from the game, for
  380 of the repo's 389 commits (`cff2344`, `25d9b70`) — changes the decision stream. Every Elo
  and every probe number in §1–§24 is on a different ladder.
- **v2.2 is retired and raises**; `staged_cards` is retired with it and does nothing. Arms F, F2,
  G and G2 cannot be loaded at all, so the previous version of this file's entire baseline table
  is void. The current layouts are `legacy` (4293), `v2.1` (3891), `v2.3` (3824).
- **The current arms are H (80M), H2 (80/160/240M) and I (80M, in flight).** H2 @240M is the
  strongest at 93.0% against the anchor (§25.1).
- `temp_cards` is gone (`49ed564`); the chance node is named (`ctx().pending_roll_type`,
  `ctx().roll_actor`) (`712bce4`). Probe 4 below used `temp_cards[1]` and no longer can.
- Europe Control is its own recorded ending (`430ba9b`), and game length is measured in **plies**,
  not turns (`53f9c1c`).
- `metrics.md` §1.5.3 retires the claim that the corrected engine lengthens games — H2 does not
  replicate H's game shape. Nothing here leans on it.

**Three things in the previous revision survive unchanged**, and they are the reason §1 is still
first:

1. The seven layout-blind call sites are still layout-blind (§1).
2. The batch runner still drains chance inside C++ *whether or not* `auto_advance` is set — now
   documented in `metrics.md` §"The Python chance-drain loop is not worth moving into C++" — so
   the pre-deal node is still not reachable through it, and probe 4 still needs its own driver.
3. The nested variance decomposition still needs no engine or bindings change: `rng_state` is
   still read/write (`ts_bindings.cpp:608`), `set_card_location` is still there (`:629`), and
   `get_state` still returns a mutable `reference_internal` (`:1061`).

## Before anything runs: this checkout cannot load the current arms

`build/release/ts_engine*.so` here is **pre-v2.3**: it exports `OBS_SIZE_V22`, has no
`OBS_SIZE_V23`, and contains no `"v2.3"` string at all. Loading the current baseline fails
outright:

```
NeuralAgent.from_checkpoint('data/checkpoints/arm_H2_cont_160to240/snapshot_final.pt')
AttributeError: module 'ts_engine' has no attribute 'OBS_SIZE_V23'
```

That one is loud, which is the good case. `tools/scripts/check_engine_fresh.sh` cannot fix it by
itself — it exits **2**, because `build/release/CMakeCache.txt` still records
`/home/mihaild/prog/ts_ai` as its source directory, so the rebuild it attempts fails on a
directory that does not exist here. Reconfigure first, which rewrites the cache:

```bash
cmake -B build/release -S . -DPython_EXECUTABLE=$(pwd)/.venv/bin/python3
tools/scripts/check_engine_fresh.sh          # rebuilds, stamps, exit 1; rerun for exit 0
```

**The live run is not affected, and this is checked.** `arm_I_no_kl` runs from the
`fix-profiler-bias` worktree, so its `PYTHONPATH=.:build/release` resolves to *that* worktree's
build — a v2.3 extension, built 21:38, stamped, and matching its own sources exactly. Only the
main checkout's build is stale. And the two share one engine: `engine/` and `bindings/`
fingerprint identically in both trees —

```
sources (fix-profiler-bias) = sources (main checkout) = b96d6883a0b6b671…8b4fa4b2
stamp on arm I's build      = b96d6883a0b6b671…8b4fa4b2   MATCH
stamp on the main build     = (none)
```

— so rebuilding the main checkout reproduces arm I's engine rather than a different one, and no
decision-stream comparison is needed beyond this. Do the rebuild in a worktree of your own rather
than in `build/release` under the live run.

## Change

### 1. Make the instruments read the observation the checkpoint was trained on

`ts.extract_observation` defaults to `layout="legacy"` and `ts.VectorizedBatchRunner` defaults to
`("legacy", flags=0)`. A model reads fixed slices at fixed offsets, so an observation *wider* than
the model expects does not raise — it returns a number computed from the wrong floats. Measured
on the v2.2 engine before the merge, over 162 sampled self-play states, `v_win` from the legacy
observation against `v_win` from the correct one: **corr +0.05, mean |Δ| 0.47, sign disagreeing
60% of the time**. Not a degraded measurement — an unrelated one.

`6220f32` fixed the two probes the trainer calls (`position_diagnostics`, `decisive_probe`) and
added `bindings.ts_env.layout_for_model`, which raises rather than guessing. **The seven
standalone probes were not part of that fix and are still blind today:**

`behavioral_suite.py:217`, `card_probe.py:108`, `battleground_value.py:89,146`,
`dominance_cost.py:77,162,212`, `round_counterfactual.py:219,290`, `input_ablation.py:63`,
`critic_calibration.py:92` — plus `tools/generate_dataset.py:70`, which builds datasets.

That would be the fifth instance of this bug class, and the fourth cost a published number: arm
H2 logged `mean_final_turn` 1–2 against an actual 6.8 and `empty_battlegrounds_turn8` 0.0 against
6.2 (§25, `metrics.md` §1.4.1).

**1.1 `ai/eval/handle.py` — one object that carries the configuration**, built on the two helpers
that already exist rather than a third mapping:

```python
@dataclass(frozen=True)
class EvalHandle:
    model: ColdWarModel
    layout: str          # bindings.ts_env.layout_for_model(model) -- raises, never guesses
    obs_flags: int       # tools.lib.engine_config.mask_for_checkpoint(path)
    device: torch.device
    name: str
    steps: int | None    # parsed from snapshot_<N>steps.pt, for the budget column
    base_commit: str | None   # from the run's metadata.json; see Procedure

    @classmethod
    def from_checkpoint(cls, path, device) -> "EvalHandle": ...
    def observe(self, state, perspective) -> np.ndarray: ...
    def runner(self, n, seed) -> ts.VectorizedBatchRunner: ...
    def value(self, states, perspectives) -> np.ndarray: ...
```

**1.2 Convert the seven call sites** to take an `EvalHandle` instead of a bare `model` + `device`.
Mechanical: `ts.extract_observation(state, mover)` becomes `h.observe(state, mover)`,
`ts.VectorizedBatchRunner(n, seed)` becomes `h.runner(n, seed)`. Their CLIs already take a
checkpoint path, so no CLI changes.

**1.3 Make the silent case loud** — the systemic fix, and the one that would have caught all four
previous instances at the call site instead of in a retraction. `ColdWarNetV2.forward` has no
width check today (`TOTAL_OBS_SIZE` is declared and never compared). Add to V1/V2/V3:

```python
if obs.shape[-1] != self.TOTAL_OBS_SIZE:
    raise ValueError(f"observation width {obs.shape[-1]} != {self.TOTAL_OBS_SIZE} this model reads")
```

Extraction-side width guards already exist (`1f38004`); the network side is the half that is
missing, which is exactly why `layout_for_model`'s docstring has to say "the widths of the
extraction and the network are checked at different places". This is not an observation change —
no slot moves, nothing is added or removed.

**1.4 Regression test**, `tests/training/test_eval_handle_layout.py`, beside the existing
`test_probe_observation_layout.py`: build a v2.3-shaped model, feed it a legacy-width observation,
assert it raises; assert `EvalHandle.from_checkpoint` recovers `layout == "v2.3"` and the run's
recorded flags.

**1.5 `ai/eval/rollout.py` — a lockstep driver that does not swallow chance.** M states stepped
together in Python, one batched forward per step, so it is fast without being the C++ runner:

```python
def rollout(h: EvalHandle, states: list[ts.GameState], *, temperature: float,
            on_chance: Callable[[ts.GameState], ts.MicroAction] | None = None,
            on_turn_start: Callable[[int, ts.GameState], None] | None = None,
            observe_node: Callable[[int, ts.GameState], None] | None = None,
            max_steps: int = 4000) -> list[float]:            # terminal utility, US perspective
```

`on_chance` defaults to `MicroAction(ROLL_DIE, 0, 0, 0)` — the zero means "roll it yourself".
Returning a non-zero payload forces the die: `primary_id` is the **acting player's** die and
`secondary_id` the opponent's, per-actor rather than per-side, which is the shape `712bce4` gave
it after per-side storage made a USSR realignment read its two dice swapped. Draining in Python
costs 6.7% of wall time against a C++ equivalent and is the cheaper side of that trade
(`metrics.md`); chance nodes are only 10.1 per game against 249 real decisions. Probes 1, 2, 3
and 5 stay on the C++ runner.

### 2. The probes

**Probe 1 — setup.** `ai/eval/setup_probe.py`.

`init_new_game` deals hands and then leaves the state in `Phase::SETUP` at a `POINT_NODE`
(`state_machine.cpp:148-156`): USSR places 6 in Eastern Europe, then US 7 in Western Europe, then
US 2 bonus in any country it already occupies. The probe is exactly 15 batched forwards from a
fresh runner — no rollout at all.

- n = 2,000 seeds (four batches of 512), temperature 0.1, matching self-play.
- Report: the 84-wide placement histogram per side; **P(Poland ≥ 3 | USSR)** (`cid 15`);
  **P(West Germany ≥ 4 | US)** (`cid 7`); the distribution over distinct 6- and 7-placements
  (top 10 plus a coverage count); placement entropy per side. Every rate as a Wilson band via
  `ai/stats.py`, per `71739f9` — a point estimate on 2,000 seeds invites a comparison the noise
  does not support.
- *Decided:* **yes, condition on Europe Scoring (`cid 2`) in hand.** It is free — the deal
  precedes setup, so `state.get_card_location(2)` is readable at the setup node — reported as a
  three-way split (USSR hand / US hand / deck) with each cell's n printed. It is the one hand
  feature that plausibly changes the answer, and a flat split is itself the result.
- Human yardstick: read the **first log entry's board** straight out of `ts_replayer_parse`, not
  out of a conversion, so every corpus game counts including the ~half whose recording stops
  (`ts_replayer_convert.py:1508-1512`). **Caveat to print with the number:** bid/handicap
  influence is folded into those setup totals (`ts_replayer_convert.py:199-206`), so the human
  Poland and West Germany figures are inflated relative to a no-bid game; split the yardstick by
  bid = 0 vs bid > 0.

**Probe 2 — VOA exposure.** `ai/eval/voa_exposure.py`. VOA is `cid 74` (`constants.hpp:189`).

- 500 self-play games per checkpoint on the C++ runner, plus the same statistic over the corpus.
- *Exposure*, at each USSR end-of-turn: VOA unplayed —
  `get_card_location(74) in {DRAW_DECK, HAND_US_UNKNOWN, HAND_US_KNOWN}` — **and** at least one
  country with `region != EUROPE`, `ussr_influence in (1, 2)`, and no USSR control. Report the
  mean count per USSR turn, P(≥ 1) as a band, and the breakdown by region.
- *Punished*: when VOA is later played as an event, the drop in USSR non-Europe influence across
  it, and the share falling on countries flagged at the preceding turn end. Read before and after
  via `observe_node`.
- The probe reads the **true** card location from the state, not the observation's `CardLocation`
  bit. The observation shows the USSR only what it is entitled to know; an evaluator is not bound
  by that, and "was it in fact still live" is what makes the exposure a mistake.
- **If fewer than 30 corpus games reach a played VOA, report the exposure rate alone and say so.**
  VOA is one card in 266 games; this is likely to bind.

**Probe 3 — chance-variance decomposition.** `ai/eval/chance_decomposition.py`.

The original design replayed forced dice through the batch runner. The runner drains chance in
C++ unconditionally and takes no forced rolls, so that route needs a bindings change and an
owner approval — and it is not needed. A nested decomposition by the law of total variance gets
the same shares from two controls that are already exposed:

- *Deals fixed* — at each turn boundary, force the newly dealt cards to the ones a recorded
  reference rollout drew, via `set_card_location` on the state `get_state(i)` returns.
- *Dice free / fixed* — `state.rng_state` is read/write; a fresh value re-rolls everything
  downstream.

Four conditions per saved state, nested deal ⊃ dice ⊃ policy:

| # | deals | dice | temperature | gives |
|:--|:--|:--|:--|:--|
| A | reference | reference | 0 | reproducibility check: variance must be exactly 0 |
| B | reference | free | 0 | `Var_dice` |
| C | reference | free | 0.1 | `Var_dice+policy` |
| D | free | free | 0.1 | `Var_total` |

Shares of `Var_total`: dice = `Var_B / Var_D`; policy = `(Var_C − Var_B) / Var_D`;
deal = `(Var_D − Var_C) / Var_D`. The outcome is the terminal utility from the saved state's
mover's perspective; variance is taken within a state over replicates, averaged over states, with
a bootstrap CI over states.

- *Decided:* **N = 16 replicates, 1,000 saved states** over turns 3–9, sampled with
  `position_diagnostics`' own sampler and filtered by `is_salvageable` — a decided position has
  no variance to decompose and would drag every share toward zero. 16 is enough for a share
  against a coarse 20%/40% threshold and not enough for a small difference between checkpoints;
  the decision rule only asks for the former.
- *Cost, measured not guessed:* 128 envs ran at ~9,200 env-steps/s on 4 CPU threads — 128 full
  games in ~10 s. 1,000 × 16 × 4 = 64,000 partial rollouts is **~1.5 h on 4 CPU threads**, less
  with more, and **it never touches the GPU**, so it runs beside a training job.
- *Methodological caveat, decided before running:* a forced draw can be unavailable, because the
  counterfactual rollout may already have played, discarded or removed a card the reference drew.
  Rule: force the reference's draw where the card is still in the counterfactual's draw deck,
  otherwise draw normally, and **report the fraction of forced draws that could not be honoured**
  beside every share. Below ~90% fidelity the deal share is a lower bound and must be reported
  as one.

**Probe 4 — pre-deal calibration.** Extend `ai/eval/critic_calibration.py`.

- New parameter `sample_nodes: Literal["decision", "pre_deal", "both"] = "decision"`, and the
  measurement moves onto `ai.eval.rollout`, without which the node is invisible.
- The pre-deal afterstate is the chance node with `ctx().pending_roll_type == RollType.TURN_CLEANUP`
  — a named field since `712bce4`; the `temp_cards[1]` this file cited in its first revision no
  longer exists. `decision_player` is `NONE` there, and `critic_calibration.py:120`'s
  `if side == 0: continue` is exactly what drops it, so evaluate **both perspectives** and store
  both rows.
- Sizing: chance nodes are 10.1 per game and TURN_CLEANUP is 36% of them (`metrics.md`), so ~3.6
  per game — 500 games gives ~1,800 nodes, ~3,600 rows.
- **`classify_ending` must be replaced, not extended.** It predates the corrected engine and
  cannot see Europe Control, wargames, held scoring, or self-inflicted versus provoked DEFCON-1 —
  the taxonomy the trainer already emits as `ending_frac_*`. Reuse that taxonomy so probe 4 and
  the training logs name the same endings.
- Report the calibration buckets, Brier and `corr(v_win, realised)` **split by node kind**, so
  P2's "the critic prices the board before it sees the deal" claim has a curve to move.

**Probe 5 — spending the tactic on the right card.** `ai/eval/sequencing.py`. New, and the reason
this step now has five probes: §25 names turn sequencing as *the* gap to a mediocre human, says
"it is what the P0 probes and P4 exist to measure", and none of probes 1–4 measure it.

Two halves, cheap and independent:

- **Constructed positions**, added to `ai/eval/claims.py` so they run inside the existing
  behavioural suite: USSR holds UN Intervention (`cid 32`) and two US-associated cards of
  different severity — the §25 position is Tear Down this Wall (`cid 96`) and Grain Sales
  (`cid 67`) — and the claim is that UN Intervention is spent on the card whose event is worse to
  hand over. Score it on the masked policy distribution, as the suite already does.
- **A self-play rate**: over 500 games, how often a DEFCON-suicide-class opponent card
  (`ai/eval/blunders.py`'s list) is played raw while UN Intervention is in hand and legal. This
  needs no constructed position and no judgement about which line is best — the card was in hand,
  the tool was in hand, and the tool was not used. Report as a Wilson band, with the count of
  opportunities.

Both are baselines here, not gates. The rate is the number a later strategy arm has to move.

### 3. Existing instruments, on the same checkpoints, after §1 lands

`battleground_value.py` (§12.1 perturbation probe), `position_diagnostics.py` (empty
battlegrounds at turn 8, ply distribution, DEFCON-1 share), forced-win take rate as a floor.
`position_diagnostics` was fixed by `6220f32` and can be trusted; the perturbation probe cannot
until §1.2.

### 4. One CLI

`tools/probe_suite.py --models <paths and baselines> --probes setup voa chance calibration
sequencing existing --output-json <path> --device cpu`, one row per model. One entry point, so a
later arm reports the same numbers by running the same command.

## Implementation order

Seven commits. Each one is separately reviewable and separately revertible, and each has an
acceptance check that fails loudly if it did not work. Nothing measures anything until step 2 has
landed, which is why it is not last.

**0. Build the engine in a worktree of your own** — not code, but nothing runs without it.

```bash
cmake -B build/release -S . -DPython_EXECUTABLE=$(pwd)/.venv/bin/python3
tools/scripts/check_engine_fresh.sh          # rebuilds and stamps, exit 1; rerun for exit 0
```

*Accept when:* the script exits 0, the stamp reads `b96d6883…`, `ts.OBS_SIZE_V23 == 3824`, and
`NeuralAgent.from_checkpoint('data/checkpoints/arm_H2_cont_160to240/snapshot_final.pt')` loads and
reports layout `v2.3`. ~10 min, mostly compile.

**1. The width guard, on its own** (§1.3). `ai/models/coldwar_net.py`, three `forward` methods.
Deliberately first and deliberately alone: it is four lines, it converts every remaining instance
of this bug class from silent to loud, and landing it separately means the next commit's
conversions are *verified* by it rather than merely intended.

*Accept when:* the backend suite passes — and note that a guard firing in an existing test is a
finding, not a regression to paper over. Expect one or two synthetic-observation call sites to
need their widths corrected.

```bash
PYTHONPATH=.:build/release .venv/bin/python -m pytest -q -n auto \
    tests/bindings tests/engine_logic tests/training
```

**2. `EvalHandle`, and the seven conversions** (§1.1, §1.2, §1.4). New `ai/eval/handle.py`
(~70 lines, wrapping `layout_for_model` and `mask_for_checkpoint`, adding nothing of its own);
edits to `behavioral_suite`, `card_probe`, `battleground_value`, `dominance_cost`,
`round_counterfactual`, `input_ablation`, `critic_calibration`, and `tools/generate_dataset.py`;
new `tests/training/test_eval_handle_layout.py`.

*Accept when:* the new test passes both directions — a legacy-width observation into a v2.3 model
raises, and `EvalHandle.from_checkpoint` recovers `v2.3` plus the run's flags — and
`.venv/bin/pyrefly check ai tools tests web bindings` is at 0 errors with **explicit paths**
(a bare `pyrefly check` in a worktree examines zero files and still exits 0). ~2–3 h.

**3. `ai/eval/rollout.py`** (§1.5), with `tests/training/test_rollout_driver.py`.

*Accept when:* driven greedily from the same seeds with chance drained the default way, the
driver's terminal utilities and action streams are **identical** to `VectorizedBatchRunner`'s over
64 games. That equivalence is the whole warrant for using it in probes 2 and 4, so it is the test
that matters; a driver that merely "looks right" reintroduces the 25-point disagreement
`metrics.md` records from the last time a single-state path diverged from the batched one. ~2 h.

**4. Probes 1 and 5** — setup and sequencing. `ai/eval/setup_probe.py`, `ai/eval/sequencing.py`,
additions to `ai/eval/claims.py`, and the corpus setup reader. These two come first among the
probes because they are the cheapest to run (15 batched forwards, and 500 games), they need only
the C++ runner, and they produce the two numbers that gate P4 — the step most likely to jump the
queue.

*Accept when:* `random` places roughly uniformly over the legal setup countries and `heuristic`
does not, which is the sanity check that the probe reads placements rather than noise; and the
human yardstick is stable across a re-run of the corpus reader. ~3 h.

**5. Probes 2 and 4** — VOA exposure and pre-deal calibration. Both need step 3's `observe_node`.
Includes replacing `classify_ending` with the trainer's `ending_frac_*` taxonomy.

*Accept when:* the ending distribution the probe reports on H2 @240M self-play reproduces §25.1's
table (mean ply ~105, DEFCON 1 ~34%, Europe Control 0.0%) — an independent path onto numbers that
are already published is the cheapest available check that the new taxonomy is wired correctly.
~3 h.

**6. Probe 3** — chance decomposition, including the draw-fidelity metric. Heaviest to write and
the only one with a real runtime.

*Accept when:* condition A returns **exactly zero** variance. If it does not, the replay is not
deterministic and every share below it is meaningless, so this is a hard gate rather than a
diagnostic. ~3 h to write, ~1.5 h to run on CPU.

**7. `tools/probe_suite.py`, the baseline run, and the write-up.** One CLI over all five probes
plus the existing instruments; run it over the *Procedure* table; write `experiments.md` §26 with
the baseline rows; add the columns to the standard eval row in `metrics.md`; `git rm` this file
and drop its row from `plans/README.md`.

Steps 0–3 are the ones that have to be right; 4–6 are independent of each other and can land in
any order, or in parallel. Total ~2 days including the write-up, none of it on the GPU.

## Procedure

### Which checkpoints

Rate the last four snapshots of each run and report the median (`metrics.md` §20); all four arms
below have a full 5M-step snapshot series.

| row | checkpoint | layout | steps | base commit |
|:---|:---|:---|---:|:---|
| v2.3 baseline | `arm_H_v23_corrected/snapshot_final.pt` | v2.3 | 80M | `32902a3` |
| v2.3 seed B | `arm_H2_v23_seedB` @ 80M and 160M | v2.3 | 80M / 160M | `32902a3` |
| **strongest** | `arm_H2_cont_160to240/snapshot_final.pt` | v2.3 | 240M | `71739f9` |
| KL off, in flight | `arm_I_no_kl` @ 80M | v2.3 | 80M | `c391f2e` |
| floor | `heuristic`, `random` | — | — | — |
| yardstick | human corpus; ITS results for game shape | — | — | — |

**The three base commits do not split the ladder — checked, not assumed.** The only engine or
bindings source to change between `32902a3` and today is `scoring.cpp`, and the change sets
`effect_bits::EUROPE_CONTROL_WIN` at a point where the game is already over: same VP, same phase,
no decision affected. So H, H2, H2-continued and I share a decision stream and are mutually
comparable, even though `check_engine_fresh.sh`'s content hash differs across them. Record each
row's `base_commit` in the output anyway — the field exists on every run's `metadata.json` now,
and it is what makes this checkable next time instead of arguable.

Comparisons that are licensed: H @80M vs H2 @80M (seed replication, known to be worth ~nothing —
4.7 Elo); H2 @80M vs @160M vs @240M (budget); **I @80M vs H2 @80M (the KL term, the one arm I is
running to answer)**. Not licensed: anything against arms A–G, on either ladder.

### The run that is training right now

`arm_I_no_kl` — v2.3, cold start, `--eta 0` (NashPG KL penalty off), 80M steps, seed 20260921,
otherwise H2's recipe, so the only difference from H2 @80M is the KL term. Started 22:04, at 40M
by 22:54, so **80M at roughly 23:45 today**.

1. Do not rebuild into `build/release` while it runs — that is the path its `PYTHONPATH` names.
   (See *Before anything runs* for why what is on that path is also the thing to resolve first.)
2. Probes run on **CPU**, so they do not contend with the 4090.
3. Develop and validate the whole suite against H and H2, whose runs are finished; take arm I's
   row from its last four snapshots once it stops.
4. Probes never write into a run directory. Output goes to `research/probe_baseline_20260911/`.

### Recording

Commit the probes with the baseline table in `experiments.md` in the same change, per this
directory's maintenance rule, and add the numbers to the standard eval row in `metrics.md`. Then
`git rm` this file and delete its row from `plans/README.md`.

## Measure

Per row: `P(Poland ≥ 3 | USSR)`, `P(West Germany ≥ 4 | US)`, setup entropy; VOA exposure rate and
punished share; dice / deal / policy shares of return variance with the draw fidelity; pre-deal
vs decision Brier and correlation; the sequencing claim pass rate and the raw-play rate; plus the
existing instruments. Rates as Wilson bands. No Elo.

## Decision rule

- Setup probe well below the human rate (expected): P4 stays in the queue and may move ahead of
  P2.
- **Dice + deal share = 1 − policy share.** `≥ ~40%`: P2 runs. `≤ ~20%`: P2 is demoted to reserve
  and its arm goes to P4/P5. Between: P2 keeps its place but behind P4.
- VOA exposure: baseline only; the acceptance number for the belief-head weighting decision in
  P5's follow-ups.
- Sequencing: baseline only. If the raw-play rate is high **and** the constructed positions pass,
  the model knows the tactic and cannot time it, which argues for P4's macro-action credit over
  P6's capacity.
- If the engine question in *Before anything runs* is unresolved, nothing above is decided by
  arm I's row.

## Follow-ups

- Every later step reports these numbers. Add them to the standard eval row in `metrics.md`.
- If the corpus is too small for a stable VOA yardstick, say so in the log and use the exposure
  rate alone.
- **Queue the BC warmup layout fix** as its own change before P1 and P7: `WarmupDataset`
  re-extracts observations with `ts.extract_observation(st, p)` at the legacy default
  (`ai/training/warmup_dataset_loader.py:39,99`) and `run_bc_warmup` passes no layout down
  (`generic_trainer.py:377-383`), while the RL path does (`:914,917`). Arms H, H2 and I are cold
  starts and are unaffected; a v2.3 warmup today would train on legacy floats, silently. Thread
  `obs_layout`/`obs_flags` through and assert the width against the model being warmed.
- `tools/generate_dataset.py:70` builds its runner at the legacy default too. Same fix, and it
  matters for P7.
- If §1.3's width assertion fires anywhere outside the tests, that call site was measuring noise;
  note where in the log entry.

## Runs

(none yet)
