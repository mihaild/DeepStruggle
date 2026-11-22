# Search, Belief, Bindings & Browser Analysis (`bindings/`, `ai/search/`, `web/ui/`)

Audit at commit `fb9b817` (2026-11-22).

---

## 1. Belief Tracking (`bindings/belief.{hpp,cpp}`, `ai/search/belief.py`, `web/ui/src/analysis/rollout.ts`)

### S-1. [Critical] Float32 precision destruction on `-1e9` permanently freezes BMA temperature weights
- **Locations:** `ai/search/belief.py:194-195`, `web/ui/src/analysis/rollout.ts:98`, `bindings/belief.hpp:53, 96-97`, `bindings/belief.cpp:67, 178-187, 235`, `bindings/ts_bindings.cpp:1215-1219, 1290`, `tools/lib/batch_tournament.py:832-833`
- **Cause:** When an observed opponent action `a` is illegal in all `N` sampled particles, every particle gets `-1e9`, producing `ev ≈ -1e9`. `j.evidence[t]` and `sd.done[t]` are stored in 32-bit `float` (where 1 ULP at `1e9` is `64.0`).
- **Impact:**
  - All real inter-temperature log-likelihood differences (0.1–20 nats) round to 0.
  - At the next turn change, `sd.done[t] += sd.current[t]` pins `sd.done[t]` at `-1e9f`, after which every subsequent turn's evidence (`-5` to `-30`) is swallowed (`-1e9f + (-15.0f) == -1e9f`) for the **rest of the game**, freezing BMA temperature weights (usually to uniform `1/6`).
  - **Triggers:** Guaranteed against any `MERGED` (E4.1) or `MACRO` (P42) opponent (`hypothesis_masks` always emits E4 masks, so macro slots `220..303` and merged slots are `0` in every particle); also triggered in normal E4 play on rare hand constraints (Quagmire/Bear Trap scoring-card plays when all 2,048 particles contain a 2+ Op card, or low-Op Missile Envy gifts).
- **Fix direction:** Store `evidence`, `done`, and `current` in `double`, and skip (do not add to `evidence`) any decision where 0 particles assign legal/finite likelihood (or apply an $\epsilon$ smoothing floor).

### S-2. [High] BMA temperature inference flaw explains human overconfidence in [P35_belief_vs_people.md](../log/P35_belief_vs_people.md)
- **Locations:** `bindings/belief.hpp:35`, `bindings/belief.cpp:166-199`
- **Cause:** With `all_decisions=True`, BMA selects one global temperature $T \in \{1.0, 1.5, 2.0, 3.0, 5.0, 8.0\}$ (with no $T = \infty$ uniform-prior model) from marginal action likelihood summed across *every* decision (`POINT_NODE`, `SELECT_OP_MODE`, `CHOOSE_BRANCH`, `SELECT_CARD`).
- **Impact:** Strong humans agree fairly closely with the network on board placements (`POINT_NODE`), so low temperatures (`T = 1.5–2.0`) dominate the marginal likelihood by many nats and are then overconfidently applied to `SELECT_CARD` / `SELECT_PLAY_MODE`, where human card timing diverges from self-play — yielding `-3.3%` LL gain and `-0.53` joint bits vs uniform prior on human site games.

### S-3. [High] `left_since` count-equality false positive & mid-turn evidence overwrite
- **Locations:** `bindings/belief.cpp:44-59, 64-74, 144-150, 200-201`, `ai/search/belief.py:85-97`
- **Impact:**
  1. `left_since` checks `unknown_then == n_left + hand_now`, which simplifies to $c_1 = e_2 + y$ where $c_1$ = cards moving `opp_unknown → my_hand` (Grain Sales / Missile Envy), $e_2$ = cards moving `deck → discard/removed` (Our Man in Tehran, or card drawn by `me` via Ask Not and played), and $y$ = cards moving `deck → opp_unknown` (`opp`'s Ask Not). Whenever $c_1 = e_2 + y > 0$, `left_since` falsely passes with a **corrupted `left` mask** (putting deck discards into `opp`'s hypothesized hand at `d` and omitting cards `opp` actually held).
  2. If a mid-turn reshuffle / Ask Not / SALT invalidates earlier decisions of turn $T$, the next decision in turn $T$ that passes `left_since` executes `g.current = j.evidence`, overwriting and discarding earlier turn-$T$ evidence instead of folding it into `g.done`.

### S-4. [Medium] Python `ai/search/belief.py:Now.of` ignores Space Race Box 4 (`revealed_headline`)
- **Location:** `ai/search/belief.py:78-82` (compare `bindings/belief.cpp:92, 117`)
- **Impact:** Sets `headline_hidden = True` unconditionally during `HEADLINE` stage 0 without checking `!revealed`, diverging from C++ `Tracker::prepare` and causing `joint_mismatch` increments in `ai/eval/belief_site_probe.py:182`.

---

## 2. Rollout Root, Keyed Chance & Searchers

### S-5. [High] Leaderboard players & browser `DEFAULT_ROLLOUT` still use buggy `chance=keyed` instead of `keyed2`
- **Locations:** `bindings/keyed_chance.hpp:18-24, 47-61`, `leaderboard/players.json:90-610`, `web/ui/src/analysis/rollout_spec.ts:46-48`
- **Cause:** `keyed` widens signed `Player::USSR` (`-1`) to `uint64_t` (`0xFFFFFFFFFFFFFFFF`), wiping out `turn`, `action_round`, `decision_type`, and `pending_roll` via bitwise OR whenever USSR is phasing/deciding/rolling, causing USSR draws to reuse identical dice across turns/rolls in the same world. `keyed2` fixes this, yet zero leaderboard players and neither default uses it.

### S-6. [High] `Rule::Z` treats 1 surviving world as `se == 0.0f` and switches unconditionally (`z = ∞`)
- **Location:** `bindings/rollout_batch.cpp:329-331, 582-586`
- **Cause:** `paired_lead` returns `lead_se = 0.0f` when `ok.size() <= 1`, colliding with "zero variance over ≥2 worlds". `decide()` tests `m > 0.0f && (se == 0.0f || m / se >= rule.param)` without checking `ok.size() >= 2`, turning `z2` into raw `argmax` on a single noisy world (e.g., when The Cambridge Five drops all worlds except 1).

### S-7. [High] DEFCON-2 suicide guard (`loses_now` / `guarded_decision`) has keyed-chance desync and misses Ortega / Che / CMC > DEFCON 2
- **Locations:** `bindings/rollout_batch.cpp:28-57, 464-466`, `engine/src/events/late_war.cpp:144-153`, `engine/src/events/mid_war.cpp:607-616`
- **Impact:**
  1. **Keyed-chance desync:** Line 464 keys `r.state.rng_state` to the pre-step decision, then `loses_now` runs `drain_chance(probe)` *without* re-keying `ROLL_DIE`. The real playout (`settle_and_mark` → `keyed::advance`) *does* re-key `ROLL_DIE`, so war/space/Olympic/coup suicide probes roll a different die than the playout executes.
  2. **Missed free-coup events:** `trigger_ortega_elected` and `trigger_che` open a `POINT_NODE` without setting `ctx.op_mode = OpMode::COUP` (leaving `INFLUENCE`), so `guarded_decision` returns `false` on USSR DEFCON-2 Cuba coups via Ortega and Che coups under CMC.
  3. **CMC at DEFCON > 2:** `if (s.defcon > 2) return false;` skips guarding coups under active Cuban Missile Crisis if DEFCON was raised above 2 during the turn.

### S-8. [High] `Stop::kPredeal` / `kEarlierPredeal` parks before deterministic Turn-10 Final Scoring, Held Scoring losses, and Mil-Ops 20-VP wins
- **Locations:** `bindings/rollout_batch.cpp:173-186`, `engine/src/state_machine.cpp:696-760`
- **Cause:** `begin_turn_cleanup` stages `RollType::TURN_CLEANUP` *before* Mil-Ops VP, Eagle/Bear discard, Held Scoring instant loss, and Turn-10 Final Scoring (all deterministic, with no deal on Turn 10). `at_turn_cleanup` parks with `terminal = false` and sends these decided game endings to the critic instead of resolving exact `±1.0` terminal utility.

### S-9. [High] Mixed-rule root-mask mismatch throws in browser workbench & Python `RolloutRoot`; wrapper/worker bugs
- **Locations:** `web/ui/src/analysis/rollout.ts:124-156`, `ai/search/rollout_root.py:274-279, 303-324, 396-407, 461, 591-656`, `bindings/rollout_batch.cpp:207-216`, `ai/search/gumbel_root.py:169-176, 228, 274-281`, `ai/search/batched_mcts.py:520-533, 920-934`, `ai/training/search_workers.py:32-46`, `tools/lib/batch_tournament.py:143, 223`
- **Impacts:**
  1. **Browser `rolloutRoot` (`rollout.ts:124-156`) & Python `RolloutRoot.choose` / `ContinuationValues` / `RolloutTeacher`:** Build root masks/candidates from caller `seat_rules`, then `RolloutBatch::add` overwrites `seat_rules` via `seat_both(base, cfg_.features)` and throws `"a candidate is not legal at the position"` whenever they differ (e.g., analyzing a human seat with `OWN_EVENT_GATED` or a `seat_rules=0` saved state using an E8.1 model).
  2. **`EntryRolloutAgent` & `SparseRolloutAgent` (`rollout_root.py:591-656`):** Do not expose `.model` or `.obs_features` (`agent_view` returns `features=0`, seating them with `seat_rules=0`), omit `seat_decider` on non-searched moves (risking `IllegalActionError`), and do not forward `observe`/`new_games`/`select_actions_games` (crashing on `:belief`).
  3. **`CANONICAL_LEARNED` (P43 V3) bypasses searcher refusal (`batch_tournament.py:223`):** `getattr(agent, "obs_features", 0)` is `0` on `RolloutRootAgent`/`BatchedMCTSAgent`/wrappers, so `refuse_learned_order` never fires and playouts fail on refused placements.
  4. **`WorkerRolloutConfig` (`search_workers.py:32-46`):** Silently drops `keyed`, `keyed2`, `aligned`, `box4`, `oracle`, and `cuda_graphs` when dispatching to `SearchWorkers` / `SearchedGames`.
  5. **`dmcts.determinize` (`ai/search/dmcts.py:71-104`):** Fails to clear `forced_deal_us`/`forced_deal_ussr` (leaking forced deals) and ignores Space Race Box 4 (`revealed_headline`) in `GumbelRoot` and `BatchedMCTS`.
  6. **Additional searcher/eval bugs:** `BatchedMCTS._search` (`backend="python"`, `batched_mcts.py:520-533`) omits `seat_tree`; `BatchedMCTSAgent.select_action` (`920-934`) omits the Cambridge Five fallback (crashes in `play_match.py`); `RolloutRootAgent`/`BatchedMCTSAgent` mutate caller `GameState`s in place via `seat_decider` (corrupting `disagreement_bank.py:163`); `RolloutRootAgent.concurrent_decisions = True` races with `cuda_graphs = True` in `_choose_all`'s thread pool and `_GraphedForward` skips `_ROWS` chunking; `GumbelRoot.choose([])` / `RolloutRoot.choose([])` / `ContinuationValues.values([])` crash on empty batches; `GumbelRoot._halve` produces `NaN` on empty masks and ignores `advance_root`; `ai/eval/loading.py` & `ai/eval/playout.py` drop `macro_influence`; `tools/scripts/disagreement_bank.py:98, 134`, `tools/lib/batch_tournament.py:143` (`apply_pricing`), and `ai/eval/network_health.py:165` build masks without per-model `seat_decider`.

---

## 3. C++ Binding Memory Safety & `BIND-1` Status

### S-10. [High] `BatchedSearch::set_buffers` stores dangling raw pointers (UAF)
- **Locations:** `bindings/batched_search.hpp:166-171`, `bindings/ts_bindings.cpp:2243-2254, 2276-2285`
- **Impact:** Missing `nb::keep_alive<1, 2>(), nb::keep_alive<1, 3>()` lets caller-owned NumPy arrays be freed while `obs_`/`masks_` still point to them; calling `set_buffers` also frees `own_obs_`/`own_masks_`, invalidating any prior `observations()`/`masks()` NumPy views. (Similarly, `VectorizedBatchRunner`'s `get_observations`/`get_action_masks`/`get_step_track_vp`/`get_predeal` views can be invalidated if `set_obs_features*`/`set_capture_*` reallocates underlying `std::vector`s.)

### S-11. [High] `BeliefTracker` null-pointer dereference, `particles == 0` NaN, stale hands in `worlds()`, and missing checks
- **Locations:** `bindings/ts_bindings.cpp:1206-1207, 1220-1223, 1242-1249, 1316-1323`, `bindings/belief.cpp:86, 147, 152-162, 184, 214`
- **Impact:**
  1. `prepare()` dereferences `*states[p]` without a `nullptr` check (nanobind converts Python `None` to `nullptr`).
  2. `particles == 0` is accepted, producing `1.0f / 0.0f` (`inf`) and `log(0/0)` (`NaN`).
  3. `prepare()` invalidates `sd.cached_record = SIZE_MAX` without clearing `sd.hands`/`sd.mixture`, while `worlds()` checks only `!g.mixture.empty()`. If `set_log_likelihoods()` is skipped or returns early on `total != blocks`, `worlds()` silently resamples stale hands from an earlier turn/position.
  4. Python `rows_patch` skips unknown-feature-bit validation, and `set_log_likelihoods` omits `require_no_nan` (present in WASM `ts_engine_wasm.cpp:897`).

### S-12. [Medium] Unchecked `GameState::ctx_stack_depth` write & `BIND-1` audit note
- **Locations:** `bindings/ts_bindings.cpp:527, 566`, `engine/include/ts/game_state.hpp:414-419`, [BUGS.md](../../BUGS.md)
- **Impact:** `ctx_stack_depth` is writable via `.def_rw` with no `0..5` bounds check, while `s.ctx()` unconditionally indexes `ctx_stack[ctx_stack_depth]` and returns `reference_internal` (OOB write/read and bogus pointer registration in nanobind's `inst_c2p`).
- **`BIND-1` status:** Auditing all `nb::class_` and `rv_policy` bindings in `ts_bindings.cpp` confirms `GameState::ctx()` was the only binding returning an internal C++ reference. Until commit `db5f79e` (2026-10-06, the day before `BIND-1` was logged), it used bare `nb::rv_policy::reference` (no `keep_alive`), matching the `inst_new_int(): unexpected collision!` crash when a temporary `GameState` was freed while a Python `DecisionContext` lived. On current `HEAD`, `ctx()` uses `reference_internal` and `get_state()`/`row_state()` return copies; no other `nb::class_` method returns an unowned internal pointer.
