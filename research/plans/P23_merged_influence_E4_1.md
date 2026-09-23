# P23 — E4.1: the first influence placement replaces the influence op-choice

**Status:** running -- stages 1-4 done (implementation `d6c89ad`), stage 5 A/B from scratch in progress; see Runs.
**Gate:** the owner approves this plan (it touches `engine/` and `bindings/`, invariant 11).
**Needs approval:** engine and bindings changes, as below; no change to `GameState`, the rules or
the observation. Stage 5 is a separate observation proposal, per the owner's standing rule.

## Goal

An engine version **E4.1** in which an agent may choose "play this card's ops for influence, first
point in country X" as **one** decision, instead of choosing `OPS_INFLUENCE` and then X. It is
opt-in per agent, and **not a breaking change**: with the option off everything is bit-identical to
E4, and E4 and E4.1 agents can play each other in one game with no adapter.

## Why

* `research/log/E4_decision_stream_census.md`: ops-for-influence is 46% of card resolutions, and
  merging the choice removes **43.0 of 386 decisions per game (11.1%)**. That is ~11% more games per
  GPU-hour before the long run.
* **It suits M2d.** The architecture's one mechanism that matters is the per-country head
  (`research/log/P21_ladder_status.md`). Today the "influence or not" choice is made by the
  generic trunk *before* any country is scored. Merged, the country head scores the first placement
  at the same node where the op type is chosen, so the choice can weigh where it would go.
* The credit-assignment effect is small, about λ 0.98 → 0.983, and is not the reason.

## Design: composition, not a new rule

**The merged step is defined as the two E4 steps it replaces.** For a NODE slot `116 + X` offered
at an op-choice node:

    step_merged(s, 116 + X)  :=  step(s, OPS_INFLUENCE); step(s, 116 + X)
    mask_merged(s)            :=  mask(s) \ {OPS_INFLUENCE}  ∪  NODE slots of mask(clone(s) after OPS_INFLUENCE)

A `GameState` is 4 KB and trivially copyable, so the clone is one memcpy per op-choice node.
Equivalence then holds **by construction**: every modifier the first placement depends on (the
China Card's Asia bonus, Vietnam Revolts, Containment, Brezhnev, Red Scare/Purge, 2-op cost into
opponent-controlled countries, `start_influence_nodes`) is applied by the same E4 code, in the same
order. No influence rule is re-implemented.

This applies at both op-choice nodes: the `RESOLUTION` node after `SelectCard` (decision type 2)
and the deferred `OP_MODE` node after an opponent's event (type 4). The census counts both.

**What an agent sees.** The same observation at the same node: the decision-type one-hot says
"op choice" either way (`observation.cpp:342-361`). Only the mask differs, which is exactly the
per-agent flag. After a merged step, the next decision is the second placement, which is the same
state E4 reaches after its second step, so everything downstream is unchanged.

**Why no adapter is needed.** The flag belongs to the *agent*, not the state. In a mixed game the
harness generates each mover's mask with that mover's flag and steps with it. Both conventions
leave the game in states E4 can produce.

### The one open question, answered by census before coding

Is the post-commit placement mask ever anything *other* than NODE slots, e.g. a `CONFIRM_DONE`?
Or is it ever empty? Either would mean "influence" cannot be represented by NODE slots alone.
Measure over ~10k fuzz games with the E4 engine. If it never happens, `OPS_INFLUENCE` is removed
from merged masks, one index keeps one meaning, and an assertion makes the case fail loudly. If it
does happen, `OPS_INFLUENCE` stays legal at merged nodes **only** for exactly those states, and the
rule is documented. No silent fallback (the owner's rule against heuristic safety nets).

## Change

| area | change |
|:---|:---|
| `engine/` | `ActionMask::generate_flat_mask(state, merged_influence)` and `Engine::step_flat(state, idx, merged_influence)`, default `false`, implemented as the composition above. The state machine, `MicroAction` layer and `GameState` are untouched. `VectorizedBatchRunner::get_action_masks` / `step_flat_all` take a **per-env** flag array, so one batch can hold E4 and E4.1 movers. |
| `bindings/` | expose the flag; regenerate the stubs (never hand-edit them); `action_encoder.py` docstring still describes the 212 layout, so correct it to 220 while there |
| training | `tools/train.py --merged-influence`, recorded in `metadata.json` and in the checkpoint; `bindings/ts_env.py` passes it to the runner. The rollout buffer stores one transition per merged decision, and nothing else changes |
| agents | `NeuralAgent` reads the flag from the checkpoint, defaulting to off for every existing one. `tools/tournament.py`, `tools/play_match.py` and `tools/lib/self_play.py` pass each mover's flag, so mixed fields just work |
| probes | the ones that decode a play mode from the op-choice action (`blunders.py`, `positions.py`, `card_probe.py`, `event_vs_ops.py`, `sequencing.py`, `targeting.py`) treat a NODE slot at an op-choice node as ops-influence. `decisive_probe` and `setup_probe` need only the flag passed |
| search | `batched_mcts.py` / `heuristic_mcts.py` expand with the agent's flag |
| unchanged | the web UI and server (humans keep E4 semantics; a merged bot's action is replayed as the two E4 steps by `web/bot_client.py`), the human-corpus converter and its datasets (E4 streams; a merged-stream view for BC is a later, pure transformation), the observation |
| naming | engine version **E4.1**; runs named `E4.1-<attempt>-<seed>`. `RUN_NAME_RE` and `checkpoint_id` are extended to accept `E\d+(\.\d+)?`. Because rules and state are identical, **E4 and E4.1 numbers are directly comparable**, unlike a letter bump (`method/run_nomenclature.md` gets the minor-version rule) |

## Procedure

1. **Freeze** the E4 decision stream first: `tools/scripts/decision_stream_baseline.py --games 200`.
2. **Census** the open question above.
3. **Implement** with tests:
   * flag off: the frozen decision stream reproduces bit for bit;
   * flag on: over fuzzed states at both op-choice node types, the merged mask's NODE set equals
     the E4 placement mask after `OPS_INFLUENCE`, and a merged step leaves a `GameState` that
     `memcmp`s equal to the two E4 steps;
   * a mixed E4-vs-E4.1 game reaches the same terminal state as the equivalent pure-E4 action
     sequence;
   * `ts_fuzz` with the flag on; C++, `tests/bindings`, `tests/engine_logic`, `tests/training`,
     `tests/replayer`, pyrefly.
4. **Measure the step saving** on real self-play: decisions per game E4 vs E4.1 (expect ~11%) and
   steps/s.
5. **Strength A/B, warm-started, one variable.** The existing late-start arms make this cheap:
   `E4-31-03` is `E4-08-03@160M` with `--ref-update-freq 5000000` run to 240M. Run the **same
   resume with `--merged-influence`** (`E4.1-01-03`), and the seed-5 twin against `E4-31-05`
   (`E4.1-01-05`). Existing E4 weights load unchanged, since the width and observation are the
   same. The first merged decisions are untrained, so the continuation has to learn them. Rate at
   matched steps **and** matched wallclock, per seat, and run the goal probes
   (`tools/scripts/goal_probes.py`).
6. **Then the observation question (separate approval).** At a merged node the network does not
   see the ops budget it is spending (`PENDING_OPS_VALUE` is set at commit,
   `observation.cpp:348`). Neither does today's op-choice node. Proposal to bring back at this
   point, not before: populate the **existing** slot at op-choice nodes from
   `grant_ops_for_card` (pure, `ops.cpp:61`). Same width, a content change at a node where the slot
   reads 0 today. One A/B on E4.1 before any long run.

## Measure

Probes first: decisions/game and steps/s; equivalence tests green; goal probes. Elo last, per
seat, at matched steps and matched wallclock.

## Decision rule

* **Adopt E4.1 for the long run** if the warm-started E4.1 arm is not worse than its E4 twin at
  matched steps on either seed (per seat, beyond the ~±5 pp of 100 games a seat), since it is
  ~11% cheaper per game. Adopt it outright if it is better.
* **Kill** if either seed is worse at matched steps on both seats, and investigate before retrying.
* If the equivalence tests cannot be made to pass without special cases, stop and report.
  Composition is the whole safety argument.

## Follow-ups

* Stage 6 (ops budget in the observation), then the long run.
* The coup/realign target heads (census variant (b), 15.5%) need ~168 new action slots and head
  slicing, so they would break checkpoint continuity. Measure E4.1 first.
* A merged-stream view of the human corpus, if BC is ever used again.

## Cost

Engine and bindings about half a day, the Python plumbing about half a day, tests alongside.
P17 went from plan to validated strength in one day on a larger change.
The validation arms cost ~2 × 80M warm-started steps, about 1–2 GPU-hours.

## Runs (filled in while running)

**2026-09-23, stages 1–4 done.**

* **Frozen stream:** 200 games, 27,325 decisions (`/workspace/data/p23_e4_baseline.jsonl.gz`,
  digest `1c609f72`). With the view off, the E4.1 build reproduces it with **0 divergences**.
* **Census (2,000 random games):** after `OPS_INFLUENCE` the next decision is the same player's
  placement in 61,862 of 61,863 cases, always with at least one country. `CONFIRM_DONE` (an early
  stop) is offered there **every time**, so the merged view keeps "influence, place nothing" as
  `OPS_INFLUENCE` rather than losing it. The exception is a pre-existing E4 dead end, UN
  Intervention played for Ops (`engine/AGENTS.md` §9a), reported and unfixed. The merged view does
  not offer influence there and calls `report_anomaly`; it fires a few times a minute in training.
* **Implementation `d6c89ad`:**
  * `tests/bindings/test_merged_influence.py`: mask equality, and byte-identical states for more
    than 5,000 composed actions.
  * 378 C++ tests, 2,000 fuzz games and 1,772 backend tests pass; pyrefly 0 errors.
* **Warm start measured (`tools/scripts/merged_view_warmstart.py`):** on E4-08-03@160M the raw
  merged-view policy puts **1.00** of its op-choice mass on influence, against the **0.38** its own
  E4 policy implies. KL 16.4, top-1 agreement 22%. The owner predicted this.

**Change to stage 5: the A/B runs from scratch, not warm-started.** Warm-started, the arm would
first have to recover from that initial policy, and the comparison would measure the recovery as
much as the view. A distillation fix means a new target source in the trainer (no side scripts,
invariant 9). From scratch needs no new code and is the ladder's standard adoption test:
`E4.1-01-03` against `E4-08-03@80M` and `E4.1-01-05` against `E4-08-05@80M`. Their flags differ
only in `--merged-influence`, plus the explicitly recorded seeds and resume cadence. Converting
existing checkpoints waits on the verdict.

| arm | directory | status |
|:---|:---|:---|
| E4.1-01-03 | `E4.1-01-03_20260923_104252` | running |
| E4.1-01-05 | — | queued behind it |
