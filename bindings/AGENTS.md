# Python Bindings & Environment Wrappers (`bindings/`)

The Python extension module `ts_engine` (built with **nanobind**), the WebAssembly entry point, and
the environment wrappers and action codec that bridge the C++20 core with Python. Section numbers
are cited from elsewhere; keep them.

---

## 1. File Overview

- [`CMakeLists.txt`](CMakeLists.txt): the nanobind module target, linked against `ts_engine_core`
  (under `emcmake` it builds `ts_engine_wasm` instead).
- [`ts_bindings.cpp`](ts_bindings.cpp): the module definition. Exposes:
  - Enums: `Player`, `Phase`, `DecisionType`, `Resolution`, `TimingBranch`, `OpMode`, `CardLocation`, `WarEra`, `Region`, `RegionalStatus`, `RollType`.
  - Structs: `MicroAction`, `CountryState`, `DecisionContext`, `GameState` (`raw_bytes()` for exact equality tests).
  - Engine API: `init_game()`, `step()`, `step_flat()`, `try_step()`, `try_step_flat()`,
    `auto_advance_step()`, `is_terminal()`, `get_terminal_utility()`, `get_legal_action_mask()`,
    `get_flat_action_mask()`, `get_legal_action_indices()`, and the held-scoring predicates.
  - RL: `extract_observation()`, `decode_flat_action()`, `encode_micro_action()`, `ActionMask`,
    `VectorizedBatchRunner` (contiguous multi-game simulation).
  - **The merged-influence view** (P23 / E4.1, [`engine/AGENTS.md`](../engine/AGENTS.md) §4):
    `merged_influence=False` on `get_flat_action_mask`, `ActionMask.generate_flat_mask`,
    `Engine.step_flat` / `try_step_flat`, plus `ActionMask.is_merged_influence_action(state, idx)`.
    The runner holds the view **per env and side** (`set_merged_influence(us, ussr)`,
    `set_merged_influence_env(i, us, ussr)`) because a tournament can seat an E4 and an E4.1 agent in
    one game; its cached mask is built for whoever decides next. Off everywhere by default (exactly
    E4); `TsVectorizedEnv` keeps the flags across `reset_all(base_seed)`.
  - **Observation feature blocks** ([`engine/AGENTS.md`](../engine/AGENTS.md) §6a):
    `extract_observation_features(state, perspective, features)` — the base plus the requested
    `OBS_FEATURE_*` blocks; `features` has no default and unknown bits raise. `obs_size_for(features)`
    is the width, `OBS_FEATURES_ALL` every known bit. The runner holds a set per env and side
    (`set_obs_features(us, ussr)`, `set_obs_features_env(i, us, ussr)`), writes each row in its
    decider's set and zero-pads to the widest (`obs_width`); `TsVectorizedEnv.set_obs_features`
    keeps the sets across `reset_all`, and `ts_env.model_obs_features(model)` reads a model's set
    from its weights.
  - **`BatchedSearch`** ([`batched_search.hpp`](batched_search.hpp)): the C++ tree behind
    `ai/search/batched_mcts.py` (`backend="cpp"`), Python-only. Up to `capacity` PUCT searches in
    lockstep, split into `num_groups` evaluation groups: `reset(roots, simulations, rng_state,
    num_groups)` (one budget or one per tree, so the Gumbel root's candidates share rounds), then per
    group `select_leaves(g)` → the network → `expand_and_backup(g, probs, values)` until
    `group_has_budget(g)` is false; the driver evaluates one group on the GPU while the tree selects
    the other's. `set_buffers(obs, masks)` writes leaves into caller-owned (page-locked) arrays.
    `root(i)` returns `(terminal, mover, value_us, actions, priors, visits, value_sums)`;
    `set_root_priors(i, priors)` takes root noise. **It is the Python tree bit for bit, random draws
    included**: chance seeds come from a C++ copy of CPython's MT19937 in the Python tree's order
    (`mt_state()` hands the state back), descents and backups run across `gomp_parallel_for` with
    only the seeds handed out serially, so nothing depends on the thread count. The grouping is part
    of the search's semantics (a row's network output depends on its batch), so
    `batched_mcts.search_groups` is the one rule both trees follow. Also takes the model's
    `obs_features`, `merged_influence` and `fpu_reduction` (0 = the original rule).
    `tests/training/test_batched_search_cpp.py` holds the two trees to identical visits and states.
  - Hand-knowledge helpers over `CardLocation`: `in_hand_of()`, `known_to_opponent()`, `hand_of()`,
    `hand_holder()`, `revealed()`, `reveal_hand()` / `reveal_both_hands()`. Use them instead of
    comparing a location to a hand constant ([`engine/AGENTS.md`](../engine/AGENTS.md) §7).
  - Metadata helpers: `MapData`, `CardData`, `StateMachine`, `Operations`, `Scoring`,
    `CardHandlers`; the `EffectBits` submodule (named 64-bit continuous-effect flags).
  - Serialization: `state_to_dict()` / `GameState.to_dict()` (the display state),
    `GameState.to_display_json()`, `GameState.to_save_json()` (byte for byte
    `json.dumps(to_save_dict(), sort_keys=True, separators=(",", ":"))`), `state_from_save_json()`;
    `game_ending_reason(state)` (`tools/lib/tournament_evaluator.classify_game_ending_reason`
    delegates to it) and `selftest_digest(games)`, the whole-game digest the WebAssembly build must
    reproduce.
  - **Save format** `ts_save_v2`: `GameState.to_save_dict()` / `state_from_save_dict()`, named
    fields (unknown keys ignored, missing ones defaulted) so a save opens in another build. It
    restores the board, tracks, card locations, RNG, headline owners, die-roll record **and the
    decision state machine** (`ctx_stack`, `ctx_stack_depth`): without the stack (v1) a state saved
    mid-decision restored at a different decision — 91 of 97 positions on one seed — with every
    visible field matching. `action_history` and `turn_aggregates` are deliberately not saved: no
    rule or observation reads them. If one ever does, add it here.
    Both loaders reject a supplied `ctx_stack_depth` unless it is an integer inside the six-entry
    stack. Missing depth still defaults; an invalid Wasm load leaves the live position intact.
- [`state_json.hpp`](state_json.hpp) / [`state_json.cpp`](state_json.cpp): **the display state, the
  save and its loader, and the ending reason, written once** as a dependency-free JSON value tree,
  converted to Python objects here and printed by the WebAssembly build — two hand-kept copies
  would drift.
- [`wasm/ts_engine_wasm.cpp`](wasm/ts_engine_wasm.cpp): the engine for the **browser workbench**, a
  flat C API compiled by Emscripten (`tools/scripts/build_web.sh`, into `web/ui/public/engine/`):
  new game, raw-bytes undo snapshots, display/save JSON, loading a save (refused unless it
  round-trips), stepping with the workbench's forced-die drain, masks in both action views, decode,
  observations (the base layout, and a model's view under its obs_features bits:
  `ts_observation_features`, `ts_obs_size_for`, which refuse an unknown bit), names, debug setters, the action layout, the engine fingerprint baked in at build,
  and the rollout root's playouts (`ts_rollout_*`, `ts_determinize`). For the page's action log, a
  click also keeps what it did before the roll-less chance steps after it (`ts_settled_display_json`:
  the state after its action or its last die roll -- the turn's cleanup, resolved in the same click,
  clears the roll record) and every die it rolled, in order (`ts_click_rolls_json`), and what each
  Event it resolved did to VP and DEFCON, over the same steps (`ts_click_event_tracks_json`, from
  `GameState::last_event_tracks`; `engine/AGENTS.md` §6c). Display only: the game is stepped
  exactly as before.
  The display state writes that record as `event_tracks` (with card names) and the save as
  `last_event_tracks` (`state_json.cpp`), next to `die_roll` / `last_roll`.
- [`selftest.hpp`](selftest.hpp): one whole-game digest (full save, both observations, the mask,
  every position) compiled into both builds; `tests/web/test_wasm_engine.py` compares them.
- [`rollout_batch.hpp`](rollout_batch.hpp) / [`rollout_batch.cpp`](rollout_batch.cpp): **the rollout
  root's playouts, written once for both builds.** `ts::rollout::determinize` is the C++ port of
  `ai/search/dmcts.determinize` (the opponent's unknown hand and the draw deck reshuffled from a
  SplitMix64 state, every count and seen card kept, an unrevealed opposing headline forgotten).
  `RolloutBatch` holds the worlds of one or more decisions — each candidate move in each sampled
  world, one dice stream per world (`shared_dice`) — and plays them greedily with the DEFCON-2
  suicide guard, marking terminal rows with their result and stopped rows as leaves. `Config::stop`
  chooses where a playout stops: `kBoundaries` (the default and the measured root: a horizon of
  action-round boundaries), `kTurnEnd` (the first position in another turn), `kLater` (the later
  of the two), `kEarlier` (the earlier: the turn's end when it comes within the horizon), `kPredeal`
  (the turn-end park after the last action, before the deal -- `at_turn_cleanup`, nobody's decision,
  read from the mover's side) or `kEarlierPredeal` (the earlier of the horizon and that park; P33).
  Python: `RolloutBatch(..., stop="boundaries"|"turn"|"later"|"earlier")`; the page passes
  `ts_rollout_reset`'s seventh argument (0 / 1 / 2 / 3; 4 / 5 for the pre-deal stops, Python
  `"predeal"` / `"earlier_predeal"`). The batch runner records the same park from both sides when
  `set_capture_predeal(True)` (read with `get_predeal()` after each step) -- the rows P33's pre-deal
  critic trains on; the decision stream is unchanged. `Config::leaf` chooses whose side a leaf's
  critic is read from: `kMover` (the default and the measured root: the root mover's at every leaf) or
  `kDecider` (the leaf's decider's, the side the critic is trained on, its value negated into the
  mover's frame where that is the opponent) or `kOwn` (past the stop, play on to the root mover's next
  decision and read it from the mover's side: on-distribution and with the mover's own hand; P33).
  Python `leaf="mover"|"decider"|"own"`, the page's eighth argument (0 / 1 / 2). `Config::aligned`
  (Python `aligned=True`, the page's tenth argument, the spec's `:aligned`, the player id's
  `aligned=true`; off by default, so every measured player keeps its count) counts every candidate's
  horizon from the end of the mover's own action round: without it a candidate that leaves choices
  inside the round (Ops points, an event's choices) counts the round's end as its first boundary and
  one that ends the round inside its own settling (space, a decision-free event) does not, so the two
  are read an action round apart (the owner's Decolonization replay, 2026-10-19). It also counts the
  whole headline as one boundary: the headline's events set the phasing side to their owners, so one
  that stops for a choice added boundaries and one that resolved at once did not -- at a headline
  choice the candidate itself decided how far its playout read. `Config::box4` (Python `box4=True`, the
  page's eleventh argument, `:box4`, `box4=true`; off by default) keeps the opponent's headline in every
  world at the headline choice of the side holding Man in Space alone (Space Race box 4: the opponent
  chose first and revealed it; `revealed_headline`, `determinize(..., keep_revealed)`), the rest of the
  hand redealt around it; off, the world forgets it and asks it again. The belief tracker's `reveal`
  (`BeliefTracker(..., reveal=True)`, `ts_belief_reset`'s fifth argument) draws every hand holding it.
  No observation slot shows the revealed card to the network. The network stays outside: the caller loops
  `fill_live` → logits → `step` until `all_done`, then `fill_leaves` → critic → `set_leaf_values`,
  then reads `values`. The root's choice is here too: `top_candidates` (the k most probable legal
  moves), `parse_rule` (argmax / z<x> / kl<t>) and `decide` (each candidate's mean value, its paired
  lead over the network's own move with its standard error, the pick). `step`, `fill_live` and
  `fill_leaves` run across the OpenMP team (`gomp_parallel_for`) from 32 rows up, rows listed
  serially first, so results are bit-identical at any thread count
  (`test_a_batch_plays_the_same_rows_on_one_thread_and_on_many`); the WebAssembly build stays
  serial. Python drives it from `ai/search/rollout_root.py` (`ts.RolloutBatch`, `ts.determinize`,
  `ts.rollout_candidates`, `ts.rollout_decide`, `ts.rollout_parse_rule`), the page from
  `web/ui/src/analysis/rollout.ts` — the same C++, so the two searchers differ only in the network's
  arithmetic. Held bit-identical across builds by `tests/web/test_wasm_engine.py`, covered by
  `tests/bindings/test_rollout_batch.py`, and the page's loop is checked against Python's by
  `tests/web/test_e2e_workbench.py`.
- [`playout_labels.hpp`](playout_labels.hpp) / [`playout_labels.cpp`](playout_labels.cpp): **P34's
  playout labeller** (native only; `tools/value_labels.py` drives it). `PlayoutLabels` takes queued
  positions (raw `GameState` bytes, `state_size()` each, with `ts.state_from_raw_bytes` the inverse
  for the same build), deals each into `worlds` worlds from the mover's view (`rollout::determinize`,
  fresh dice per world, from a stream seeded by (seed, id) so a position's labels do not depend on
  the batch around it), and plays every world greedily to the game's end under
  `paired_playouts.play_safe`'s rule (in the position's own action round, at DEFCON 2 or below, no
  option that loses on the spot while another exists). Per world, from the mover's side: the score,
  the VP track at the end, VP and DEFCON as the position's turn ends, and each region's swing at its
  next scoring through `ts::ScoringTap` (engine, `scoring.hpp`: a thread-local, inert unless set).
  `take_positions()` returns the sums per position. Queued with candidates (the sibling slice) each
  candidate plays in the same worlds on the same dice and the row's leaf -- the mover's next
  decision, or with `leaf_horizon` the rollout root's stop=earlier, leaf=own -- comes back raw from
  `take_siblings()`. Batching is continuous (a finished world frees its row, the queue refills it),
  the loops hand out row blocks from a shared counter (unequal cores), and `fill_live_codes` writes
  each observation float as its one-byte code in `codebook()` -- exact, a quarter of the upload --
  for the GPU to decode (the table is `ObsCodebook`, also bound alone: `encode_raw(states)` codes
  stored states' observations for the value fit). `world(base, id, w)` reproduces a world, which is how
  `tests/bindings/test_playout_labels.py` holds every world equal to a step-by-step Python
  `play_safe` playout of it.
- [`keyed_chance.hpp`](keyed_chance.hpp): **keyed chance**, both batches' `keyed=True` (off by
  default; the rollout player's `chance=keyed`). Before every engine step `rng_state` is set to
  hash(world key, situation) -- turn, action round, phasing side, phase, decision, staged roll with
  its target and roller, steps left, resolving card -- so a world's candidates draw the same dice and
  event cards wherever they reach the same situation; at the turn's end the next deal is forced
  (the engine's forced-deal slots) from one card priority order per world, with one card in reserve
  per hand for the Space Walk discard. A single continuation plays the same game distribution; only
  the coupling between continuations changes. Player fields are narrowed to unsigned bytes before
  packing: USSR is -1 and must not overwrite neighboring fields. Keyed measurements and labels
  from before this correction must be retaken; ordinary engine dice are unchanged.
  `ts.keyed_key(state, key)` is the keying step, for references and tests.
- **The oracle's worlds** (`RolloutBatch(oracle=True)`, `ts.determinize(..., keep_hand=True)`; the
  rollout agent string's `:oracle`): every world keeps the opponent's true hand and reshuffles only
  the draw deck, so the deck order, the dice and an unrevealed headline stay unknown. Not a
  deployable player -- the ceiling on what any belief over the opponent's hand could give the root
  (P35 item 3). Python-only; the browser build constructs the default (honest) worlds.
- **A belief's worlds** (P35 3b, `ai/search/belief.py`): `ts.place_hand(state, me, held)` splits
  `me`'s hidden pool as a 110-byte mask says (opponent's unknown hand / deck), `RolloutBatch.add(...,
  hands=)` deals each world's opposing hand from a worlds x 110 array instead of at random (with
  determinize's forgetting of forced deals and an unrevealed headline), and
  `ts.hypothesis_rows(state, me, held, features)` builds a past opposing decision's observation and
  mask under n hypothesised hands in one parallel call -- the policy-likelihood belief's inner loop,
  held equal to the Python rebuild by `tests/training/test_belief.py`.
- [`belief.hpp`](belief.hpp): **the belief tracker** (`ts.BeliefTracker`, compiled into
  `ts_rollout_batch` so the browser build has it too). Per game it keeps the turn's decisions; at a
  root decision it records the opponent's decisions that fit the hand, draws `particles` hands keyed by
  the seed and the position (a game's worlds do not depend on the batch -- tested), lists the rows
  (`rows_patch` for a device, `hypothesis_rows` per decision for a full build), takes the caller's
  log-probabilities at every temperature of `kTemperatures`, weighs the hands and the temperatures
  (model averaging by each side's evidence of its opponent across the game) and resamples the root's
  worlds. Belief state is per (game, side). The rollout spec's `:belief<N>` and the player id's
  `belief=<N>` name it.
- [`gomp_parallel.hpp`](gomp_parallel.hpp): `gomp_parallel_for`, the one OpenMP pool (§3).
- [`settle.py`](settle.py): `SettleMode`, the one rule for how far past decisions nobody controls
  the engine is advanced. Settlers that stop at different depths put two components on different
  decisions while both believe they are synchronised; a settle mode is a property of the
  measurement, so state it when it matters.
- [`ts_engine/`](ts_engine/__init__.pyi): **generated** type stub package (`__init__.pyi` plus one
  file per submodule), regenerated by the build from the module just built; never hand-edit it.
  What introspection cannot recover goes in [`ts_engine.pyi.pattern`](ts_engine.pyi.pattern).
- [`action_encoder.py`](action_encoder.py): the codec between the flat action space (220 slots) and
  `ts::MicroAction`.
- [`ts_env.py`](ts_env.py): `TsVectorizedEnv` and `TsSingleEnv`, the batch-runner environments for
  PyTorch RL loops; `check_obs_width(model)` and `model_obs_features(model)`.
  - `TsVectorizedEnv.step` returns an `info` dict whose `completed_episodes` entries carry `turn`
    (reads 11 for a completed game, since `finish_end_turn` increments before testing its bound),
    `ply` (game length in [`ai/game_length.py`](../ai/game_length.py)'s numeration, 1 = the USSR's
    turn-1 headline, 154 = all ten turns; prefer it to `turn`) and `ending_reason` (one of
    `ENDING_REASON_KEYS`: `20vp`, `europe_control`, `final_scoring`, `defcon1_self`,
    `defcon1_provoked`, `held_scoring`, `wargames`). The same values are per-env in
    `info["terminal_turns"]` / `["terminal_plies"]` / `["ending_reasons"]`.
  - **The runner's cached observation and mask**: `step_flat_all` rebuilds each env it steps,
    `reset_game` the env it resets, and `set_state` nothing — whoever calls it must refresh.
    `TsVectorizedEnv` refreshes only after `_apply_start_position` actually injects a start position
    (an observation costs ~7.9 µs against ~0.8 µs for a step; `tests/training/test_env_refresh.py`).
  - **Each env's state lives in a `ts::CachedState`** ([`engine/AGENTS.md`](../engine/AGENTS.md)
    §6b), so **`get_state(i)` returns a copy**: writing to it does not reach the runner; put a
    position back with `set_state(i, state)` and refresh. `BatchedSearch` nodes hold `CachedState`
    the same way.
  - A `start_provider` returns a mid-game pool position (stored pre-deal one turn before its target,
    so `start_turn` is `state.turn + 1`) or, for `--forced-opening`, a fresh deal whose setup a script
    has played (`tools/lib/openings.play_scripted_setup`, `start_turn` 1;
    `tests/training/test_forced_opening.py`).

**`step()` raises; `try_step()` returns a bool.** `step` validates against the same mask a caller
samples from, so a refusal means the action was chosen against a different state — a caller bug.
`step()`/`step_flat()` raise `RuntimeError` naming what the engine was asking for;
`try_step()`/`try_step_flat()` keep the bool for callers that genuinely probe legality (fuzzers,
mask-vs-step sweeps). Prefer `tools.lib.game_step.step_checked`. **The two mask spaces are not
interchangeable**: `get_legal_action_indices` is the 128-wide per-decision mask, while `step_flat`
reads the flat space of `get_flat_action_mask` (feeding one to the other once froze four tests on a
refused state).

---

## 2. Mandatory Documentation Maintenance Rule for Agents

> [!IMPORTANT]
> **Keep Python Bindings Documentation Synchronized**:
> Whenever adding or changing exported types, updating nanobind signatures, or altering serializer dictionaries, you **MUST** update this file and root [`AGENTS.md`](../AGENTS.md).

---

## 3. How to Compile & Test

**Threads: one OpenMP pool per process, torch's.** The batch runner's, `BatchedSearch`'s and
`RolloutBatch`'s parallel loops go through `gomp_parallel_for` ([`gomp_parallel.hpp`](gomp_parallel.hpp)),
which calls libgomp's `GOMP_parallel` directly, and the extension links GCC's `libgomp.so.1` by
exact file; torch ships libgomp and the loader shares it by soname, so there is one runtime, one
pool and one `OMP_WAIT_POLICY` (`tools/train.py` and `tools/tournament.py` set PASSIVE). Do **not**
go back to `#pragma omp`: under clang it links LLVM's libomp, a second pool of spinning workers
(measured 5-7% slower on a rollout loop), and clang's `-fopenmp=libgomp` compiles the loops
*serially* without a warning. `tests/bindings/test_build_toolchain.py` catches a non-clang engine
(`ts_engine.BUILD_COMPILER`), a second pool, and loops that do not spread across threads.

```bash
cmake -B build/release -S . -DPython_EXECUTABLE=$(pwd)/.venv/bin/python3   # clang, found by CMake
cmake --build build/release -j
PYTHONPATH=.:build/release .venv/bin/python -m pytest -q tests/bindings    # python -m pytest, never .venv/bin/pytest
```
