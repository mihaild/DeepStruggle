# P0 — Instruments

**Status:** queued (design revised 2026-09-10, before first run)
**Gate:** none. Everything after this is read through these, so they come first.
**Needs approval:** none. No engine, bindings, observation or trainer change. One assertion is
added to `ColdWarNetV2.forward` (§1.3); it changes no layout, no width and no content.

## Goal

Four measurements that the later steps are judged by, taken once per checkpoint so every later
arm has a before/after. Two are new probes, two are existing instruments pointed at new states.

The baseline is **not** one checkpoint. The layout lineage now spans three observations and one
engine flag, and a probe number is only comparable inside a (layout, flags, step-budget) cell, so
the baseline is a table (§4).

## Why

The goal is "no simple mistakes", and the simple mistakes named so far — setup that ignores
Poland / West Germany, footholds left exposed to Voice of America — are not measured by anything
in `ai/eval/`. Elo does not move for them (`experiments.md` §4.4 shows a behaviour can be worth
two points of win rate and still be the thing a human notices first). And P2 spends an arm on
chance-aware targets; whether that is worth an arm depends on how much of the return variance is
chance in the first place, which nobody has measured.

## What changed since this file was written

Three things were found while making the design specific. The first is the reason §1 exists and
comes before every probe.

**1. Every instrument in `ai/eval/` that builds its own observation is layout-blind, and on a
v2.1 or v2.2 checkpoint it silently measures noise.** `ts.extract_observation` defaults to
`layout="legacy"` and `ts.VectorizedBatchRunner` defaults to `("legacy", flags=0)`. The model
reads fixed slices at fixed offsets, so a *wider* observation than the model expects does not
raise — legacy is 4293 floats, v2.1 is 3891, v2.2 is 3825, so every one of these call sites hands
a v2.x network 4293 floats and gets a number back. Measured on
`arm_F_v22_cold/snapshot_final.pt`, over 162 sampled self-play states, `v_win` computed from the
legacy observation against `v_win` computed from the correct one:

```
corr = +0.05      mean |delta| = 0.47      sign disagreement = 60%
```

That is not a degraded measurement, it is an unrelated one. Affected call sites:
`battleground_value.py:89`, `card_probe.py:108`, `dominance_cost.py:77,162,212`,
`critic_calibration.py:92`, `input_ablation.py:63`, `round_counterfactual.py:219,290`,
`behavioral_suite.py:217`. This is the same failure as 83b5c7b (`fix(self_play): the model was
reading the wrong observation layout`), in nine more places. Any probe number ever taken on arm
D/E/F/G with these modules is void.

`position_diagnostics.py` is clean — it takes a `select_action` callable, so a `NeuralAgent`
carries its own layout and flags in.

**2. The batch runner swallows chance nodes, so the pre-deal state is not reachable through it.**
`VectorizedBatchRunner::refresh_single` and `step_flat_all` both resolve `ROLL_DIE` in C++ until a
real decision is reached (`bindings/ts_bindings.cpp:876-882, 930-938`). The `TURN_CLEANUP` node
probe 4 wants is consumed inside that loop and never surfaces in Python. Probe 4 therefore needs
a rollout driver that stops at chance nodes (§1.5) — not a bindings change.

**3. Out of scope, but it blocks P1 and P7: BC warmup ignores `--obs-layout`.** `WarmupDataset`
replays `(seed, actions)` traces and re-extracts with `ts.extract_observation(st, p)` at the
legacy default (`ai/training/warmup_dataset_loader.py:39,99`), and `run_bc_warmup` does not pass
a layout down (`ai/training/generic_trainer.py:377-383`). The RL path does pass it
(`generic_trainer.py:914,917`), so arms F and G — cold starts — are unaffected. But
`--mode warmup --obs-layout v2.2` today trains a v2.2 net on legacy observations, silently, by
mechanism 1. **Recommended as a separate one-file fix before P1/P7**, not folded in here.

## Change

### 1. Make the instruments read the observation the checkpoint was trained on

This is the whole of the work that has to land before any number is taken.

**1.1 `ai/eval/handle.py` — one object that carries the configuration.**

```python
@dataclass(frozen=True)
class EvalHandle:
    model: ColdWarModel
    layout: str          # "legacy" | "v2.1" | "v2.2"
    obs_flags: int       # engine_config.mask_for_checkpoint
    device: torch.device
    name: str
    steps: int | None    # parsed from snapshot_<N>steps.pt, for the budget column

    @classmethod
    def from_checkpoint(cls, path, device) -> "EvalHandle": ...   # via NeuralAgent.from_checkpoint
    def observe(self, state, perspective) -> np.ndarray: ...      # extract_observation(..., layout, flags)
    def runner(self, n, seed) -> ts.VectorizedBatchRunner: ...    # (n, seed, layout, flags)
    def value(self, states, perspectives) -> np.ndarray: ...      # batched v_win
```

`NeuralAgent.from_checkpoint` already resolves both halves — layout from the weights, flags from
the run's `metadata.json` via `engine_config.mask_for_checkpoint` — so `EvalHandle` wraps it
rather than re-deriving anything.

**1.2 Convert the nine call sites** listed above to take an `EvalHandle` instead of a bare
`model` + `device`. Mechanical: `ts.extract_observation(state, mover)` becomes
`h.observe(state, mover)`, `ts.VectorizedBatchRunner(n, seed)` becomes `h.runner(n, seed)`. Their
CLIs already take a checkpoint path, so no CLI changes.

**1.3 Make the silent case loud.** Add to `ColdWarNetV2.forward` (and V1/V3):

```python
if obs.shape[-1] != self.TOTAL_OBS_SIZE:
    raise ValueError(f"observation width {obs.shape[-1]} != {self.TOTAL_OBS_SIZE} this model reads")
```

Slicing is why a wide observation gets through; an explicit width check is the only thing that
stops it. This is not an observation change — no slot moves, nothing is added or removed.

It does **not** catch a flag mismatch: `staged_cards` is same-width by construction, which is why
arms F/F2 and G/G2 are indistinguishable by shape and why `engine_config` exists. Nothing but
carrying the recorded flags catches that, so §1.1 carries them.

**1.4 Regression test**, `tests/training/test_eval_handle_layout.py`: build a v2.2-shaped model,
feed it a legacy-width observation, assert it raises; assert `EvalHandle.from_checkpoint` on a
fixture whose `metadata.json` sets `engine_config.staged_cards = true` reports
`obs_flags == OBS_FLAG_STAGED_CARDS`.

**1.5 `ai/eval/rollout.py` — a lockstep driver that does not swallow chance.** M states stepped
together in Python, one batched forward per step, so it is fast without being the C++ runner:

```python
def rollout(h: EvalHandle, states: list[ts.GameState], *, temperature: float,
            on_chance: Callable[[ts.GameState], ts.MicroAction] | None = None,
            on_turn_start: Callable[[int, ts.GameState], None] | None = None,
            observe_node: Callable[[int, ts.GameState], None] | None = None,
            max_steps: int = 4000) -> list[float]:            # terminal utility, US perspective
```

`on_chance` defaults to `MicroAction(ROLL_DIE, 0, 0, 0)` (the engine rolls); returning a non-zero
`primary_id`/`secondary_id` forces the die, which the engine already honours
(`state_machine.cpp:1235-1236`, `ops.cpp:257,343-344`). `observe_node` fires at every node
including chance, which is what probe 4 needs. Probes 1 and 3 stay on the C++ runner.

### 2. The four probes

**Probe 1 — setup.** `ai/eval/setup_probe.py`, CLI in `tools/probe_suite.py`.

`init_new_game` deals hands and then leaves the state in `Phase::SETUP` at a `POINT_NODE`
(`state_machine.cpp:148-156`): USSR places 6 in Eastern Europe, then US 7 in Western Europe, then
US 2 bonus in any country it already occupies. So the probe is exactly 15 batched forwards from a
fresh runner — no rollout at all.

- n = 2,000 seeds (four batches of 512), temperature 0.1, matching self-play.
- Report: the 84-wide placement histogram per side; **P(Poland ≥ 3 | USSR)** (`cid 15`);
  **P(West Germany ≥ 4 | US)** (`cid 7`); the distribution over distinct 6-placements and
  7-placements (top 10 plus a coverage count); placement entropy per side.
- *Decided:* **yes, condition on Europe Scoring (`cid 2`) in hand.** It is free — the deal
  precedes setup, so `state.get_card_location(2)` is readable at the setup node — and it is
  reported as a three-way split (in USSR hand / in US hand / in deck) with the n of each cell
  printed, because that is the one hand feature that plausibly changes the answer and a split
  that turns out to be flat is itself the result.
- Human yardstick: read the **first log entry's board** straight out of `ts_replayer_parse`,
  not out of a conversion. The setup is stated as a per-country total in that entry
  (`ts_replayer_convert.py:1508-1512`), so every corpus game counts, including the ~half whose
  recording stops later. **Caveat to print with the number:** bid/handicap influence is folded
  into the setup totals in these logs (`ts_replayer_convert.py:199-206`), so the human
  West Germany and Poland figures are inflated relative to a no-bid game; report the bid where
  the log states it and split the yardstick by bid = 0 vs bid > 0.

**Probe 2 — VOA exposure.** `ai/eval/voa_exposure.py`.

The Voice of America is `cid 74` (`constants.hpp:189`).

- 500 self-play games per checkpoint on the C++ runner, plus the same statistic over the corpus.
- *Exposure*, evaluated at each USSR end-of-turn: VOA unplayed —
  `state.get_card_location(74) in {DRAW_DECK, HAND_US_UNKNOWN, HAND_US_KNOWN}` — **and** at least
  one country with `region != EUROPE`, `ussr_influence in (1, 2)`, and no USSR control. Report
  the mean number of such countries per USSR turn, P(≥ 1), and the breakdown by region.
- *Punished*: when VOA is later played as an event, the drop in USSR non-Europe influence across
  that event, and the share of it that fell on countries flagged as exposed at the preceding turn
  end. Read from the state before and after the event resolves, via `observe_node`.
- Note on the original wording: the probe reads the **true** card location from the state, not
  the observation's `CardLocation` bit. The observation shows the USSR only what the USSR is
  entitled to know; an evaluator is not bound by that, and "was it in fact still live" is the
  quantity that makes the exposure a mistake.
- Human yardstick: same statistic on converted corpus games. **If fewer than 30 corpus games
  reach a played VOA, report the exposure rate alone and say so** — this is the follow-up the
  original file anticipated and it is likely to bind.

**Probe 3 — chance-variance decomposition.** `ai/eval/chance_decomposition.py`.

The original design isolates dice by replaying forced rolls through the batch runner. The runner
resolves chance in C++ and takes no forced rolls, so that route needs a bindings change and an
owner approval, and it is not needed: **a nested decomposition by the law of total variance gets
the same three shares with no engine or bindings change.**

Two controls, both already exposed:

- *Deals fixed* — at each turn boundary, force the newly dealt cards to be the ones a recorded
  reference rollout drew, by writing `set_card_location` on the state (`get_state(i)` returns a
  `reference_internal`, so the env's own state is what is edited).
- *Dice free / fixed* — `state.rng_state` is read/write (`ts_bindings.cpp:602`); a fresh value
  re-rolls everything downstream, the recorded value reproduces it.

Four conditions per saved state, nested deal ⊃ dice ⊃ policy:

| # | deals | dice | temperature | gives |
|:--|:--|:--|:--|:--|
| A | reference | reference | 0 | reproducibility check: variance must be exactly 0 |
| B | reference | free | 0 | `Var_dice` |
| C | reference | free | 0.1 | `Var_dice+policy` |
| D | free | free | 0.1 | `Var_total` |

Shares of `Var_total`: dice = `Var_B / Var_D`; policy = `(Var_C − Var_B) / Var_D`;
deal = `(Var_D − Var_C) / Var_D`. The outcome variable is the terminal utility from the saved
state's mover's perspective; variance is taken within a state over replicates and then averaged
over states, with a bootstrap CI over states.

- *Decided:* **N = 16 replicates, 1,000 saved states** spread over turns 3–9, sampled from
  self-play with `position_diagnostics`' own sampler and filtered by `is_salvageable` (a decided
  position has no variance to decompose and would dilute every share toward zero). 16 is enough
  for a share against a coarse 20%/40% threshold and is not enough for a small difference between
  two checkpoints; the decision rule only asks for the former.
- *Cost, measured not guessed:* 128 envs of the arm-F net run at ~9,200 env-steps/s on 4 CPU
  threads — a batch of 128 full games in ~10 s. 1,000 × 16 × 4 = 64,000 partial rollouts is
  **~1.5 h on 4 CPU threads**, less with more, and **it never touches the GPU**, so it can run
  beside a training job.
- *Methodological caveat, decided before running:* a forced draw can be unavailable — the
  counterfactual rollout may already have played, discarded or removed a card the reference drew.
  Rule: force the reference's draw where the card is still in the counterfactual's draw deck,
  otherwise draw normally, and **report the fraction of forced draws that could not be honoured**
  as a fidelity number beside every share. A fidelity below ~90% makes the deal share a lower
  bound, and the report must say so rather than quietly averaging.

**Probe 4 — pre-deal calibration.** Extend `ai/eval/critic_calibration.py`.

- New parameter `sample_nodes: Literal["decision", "pre_deal", "both"] = "decision"`, and the
  measurement moves onto `ai.eval.rollout` so the `TURN_CLEANUP` node is visible at all.
- The pre-deal afterstate is the `ROLL_DIE` node with
  `ctx().temp_cards[1] == RollType.TURN_CLEANUP` (`state_machine.cpp:1228-1232`), where
  `decision_player` is `NONE`. The current code's `if side == 0: continue`
  (`critic_calibration.py:120`) is exactly what drops it, so the fix is to evaluate the node
  **from both perspectives** and store both rows.
- Report the existing calibration bucket table, Brier and `corr(v_win, realised)` **split by node
  kind**, so P2's "the critic prices the board before it sees the deal" claim has a curve to
  move.

### 3. Existing instruments, on the same checkpoints, after §1 lands

`battleground_value.py` (§12.1 perturbation probe), `position_diagnostics.py` (empty
battlegrounds at turn 8, final-turn distribution, DEFCON-1 share), forced-win take rate as a
floor. All three re-run from scratch: any earlier number on a v2.x checkpoint is void by
mechanism 1.

### 4. One CLI

`tools/probe_suite.py --models <paths and baselines> --probes setup voa chance calibration
existing --output-json <path> --device cpu`, printing one row per model with the columns of §4
below. One entry point, so a later arm reports the same numbers by running the same command.

## Procedure

### Which checkpoints

Probe numbers are comparable only inside a `(layout, flags, budget)` cell, and the budget rule
says never compare across budgets. Rate the last four snapshots of each run, not the final one
(`metrics.md` §20), and report the median.

| row | checkpoint | layout | flags | steps |
|:---|:---|:---|:---|---:|
| legacy baseline | `dec_turns40/snapshot_final.pt` | legacy | — | 78M |
| v2.1 baseline | `arm_E_cont_240to320/snapshot_final.pt` | v2.1 | — | 320M |
| v2.2 | `arm_F_v22_cold/snapshot_final.pt` | v2.2 | — | 80M |
| v2.2 seed B | `arm_F2_v22_cold_seedB/snapshot_final.pt` | v2.2 | — | 80M |
| v2.2 + staged | `arm_G_v22_staged` @ 80M **and** final | v2.2 | staged_cards | 80M / 320M |
| v2.2 + staged seed B | `arm_G2_v22_staged_seedB` | v2.2 | staged_cards | in flight |
| floor | `heuristic`, `random` | — | — | — |
| yardstick | human corpus | — | — | — |

The F/G comparison is the one that must be step-matched: G has a snapshot at 80M, so the
staged-flag row is read at 80M against F, and G's 320M row is reported but not compared to F.

### The run that is training right now

`arm_G2_v22_staged_seedB` (v2.2 + `staged_cards`, seed 20260921, 160M steps, paired with F2) was
at 40M after 46 minutes, so **~160M at roughly 09:50 today**.

1. **Do not rebuild into `build/release` while it runs.** That is the `.so` the live process was
   loaded from.
2. `tools/scripts/check_engine_fresh.sh` **currently exits 2** on this machine: the stamp file
   `build/release/.engine_fingerprint` is absent and `CMakeCache.txt` still points at
   `/home/mihaild/prog/ts_ai`, so the rebuild it tries fails on a wrong source directory. The
   `.so` (07:33) is newer than every engine source (07:21), so this is almost certainly an
   unstamped build rather than a stale one — but "almost certainly" is not what invariant 10
   asks for.
3. So configure a **second** build directory and verify against it, which leaves the live run
   untouched. The freshness script already takes the directory as `$1`:

   ```bash
   cmake -B build/probe -S . -DPython_EXECUTABLE=$(pwd)/.venv/bin/python3
   cmake --build build/probe -j
   tools/scripts/check_engine_fresh.sh build/probe      # stamps it; expect exit 0 on the rerun
   ```

4. **Prove the two engines are the same game** before trusting any probe run against G2, by
   hashing a fixed-seed decision stream under each module: 100 games, greedy policy, seeds
   1000–1099, `sha256` of the concatenated flat action indices, once with
   `PYTHONPATH=.:build/release` and once with `PYTHONPATH=.:build/probe`. Equal hashes mean the
   probes and the live run share a decision stream. **Unequal hashes mean G2 is training against
   an engine that no longer matches the sources, and the right move is to stop and ask, not to
   measure it.**
5. Probes run on **CPU** (probe 3's timing above is CPU), so they do not contend with G2 for the
   4090. Develop and validate the whole suite against F and G while G2 finishes; take G2's row
   from its last four snapshots afterwards.
6. Probes never write into a run directory. Output goes to `research/probe_baseline_20260910/`.

### Recording

Commit the probes with the baseline table in `experiments.md` in the same change, per this
directory's own maintenance rule, and add the four numbers to the standard eval row in
`metrics.md`. Then `git rm` this file and delete its row from `plans/README.md`.

## Measure

Per row: `P(Poland ≥ 3 | USSR)`, `P(West Germany ≥ 4 | US)`, setup entropy; VOA exposure rate and
punished share; dice / deal / policy shares of return variance with the draw fidelity; pre-deal
vs decision Brier and correlation; plus the three existing instruments. No Elo.

## Decision rule

- Setup probe well below the human rate (expected): P4 stays in the queue and may move ahead of
  P2.
- **Dice + deal share = 1 − policy share.** `≥ ~40%`: P2 runs. `≤ ~20%`: P2 is demoted to reserve
  and its arm goes to P4/P5. Between: P2 keeps its place but behind P4.
- VOA exposure: baseline only; it is the acceptance number for the belief-head weighting decision
  in P5's follow-ups.
- If the two engine hashes in Procedure step 4 differ, nothing above is decided by this run.

## Follow-ups

- Every later step reports these four numbers. Add them to the standard eval row in `metrics.md`.
- If the human corpus is too small for a stable VOA yardstick (it may be — VOA is one card in
  266 games), say so in the log and use the exposure rate alone.
- **Queue the BC warmup layout fix** (§"What changed" 3) as its own change before P1 and P7:
  thread `obs_layout` and `obs_flags` from `run_bc_warmup` through `WarmupDataset` and
  `HumanCorpusDataset`, and assert the width against the model being warmed.
- If §1.3's width assertion fires anywhere outside the tests, that call site was measuring noise;
  note where in the log entry.

## Runs

(none yet)
