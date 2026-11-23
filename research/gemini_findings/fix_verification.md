# Fix Verification Against `main` (`0cd1d6a..7b8671b`, 2026-11-22)

Read-only verification of all fix commits landed on `main` after [audit_2026-11-22_verification.md](../log/audit_2026-11-22_verification.md) (`0cd1d6a`) through `HEAD` (`7b8671b`) against every finding in this directory ([engine.md](engine.md), [training.md](training.md), [models.md](models.md), [search_and_belief.md](search_and_belief.md), [research_record.md](research_record.md)).

> **Note:** `build/release/ts_engine*.so` in the working directory was still pre-E9 (Nov 19) at the time of inspection, so every verdict below was verified directly against committed source at `7b8671b`. Rebuild via [check_engine_fresh.sh](../../tools/scripts/check_engine_fresh.sh) before running tests or E9 matches.

---

## Summary Scorecard

| Area | Verified fixed | Regressions / incomplete fixes | Unfixed (confirmed in verification log) | Intentionally deferred / retired |
| :--- | :--- | :--- | :--- | :--- |
| **Engine (`E-1..19`, +3 minor)** | `E-1..5`, `E-8..19`, Minor 1 & 3 | **`E-6` regression**, **`E-7` regression**, `ec4e80c` schema drift | — | Minor 2 (`SUMMIT` forced dice) |
| **Training (`T-1..18`)** | `T-3`, `T-4(1)`, `T-6..8`, `T-10`, `T-11` (removed), `T-12`, `T-16`, `T-18(3,4,7)` | **`T-14` regression**, `T-1`, `T-4(2)`, `T-9` (RNG), `T-13` (Final Scoring) | **`T-5`**, **`T-15`** (worsened by `T-14`) | `T-2` (documented), `T-4(3)`, `T-17` (`TODO(pool)`), `T-18(1,2,5,6)` |
| **Models (`M-1..7`)** | `M-1` (sphere `w2`), `M-3.1`, `M-3.3` | `M-4` (missed consumers), `M-6` (dead hook / SWA), `M-7` (dropout, deleted tests, `plasticity_probe`) | — | `M-1` (non-sphere), `M-2`, `M-3.2`, `M-5` |
| **Search & Belief (`S-1..12`)** | `S-4`, `S-6`, `S-7`, `S-8`, `S-9.1..5`, `S-10`, `S-12`, `BIND-1` | `S-3` (partial-fold posterior & Tehran edge), `S-11` (WASM check / `±inf`) | **`S-1`**, `S-9.6` (remainder) | `S-2` (P57 plan), `S-5` (`keyed2` staged on E9) |
| **Research (`R-1..4`)** | `R-1`, `R-3` | `R-2` / `R-4` (minor doc leftovers) | — | — |

---

## 1. New Regressions & Incomplete Fixes in the Landed Commits

### High — Regressions Introduced by Fix Commits

#### V-1. `E-6` regression (`0d84934`): `card_in_play()` illegally hides `MISSILE_ENVY` in the opponent's hand during Action Round & Grain Sales Event plays
- **Locations:** `engine/include/ts/game_state.hpp:429-434`, `engine/src/state_machine.cpp:1323, 1428-1441`, `engine/src/events/mid_war.cpp:165, 176, 417, 479`, `engine/src/events/early_war.cpp:46, 114`, `engine/src/events/late_war.cpp:171, 200`, `engine/src/card_dispatcher.cpp:549, 1152, 1163, 1498, 2098`
- **Cause:** `GameState::card_in_play` intentionally checks `pending_op_card` (not `resolving_card`) across lower frames so that `MISSILE_ENVY` handed into the opponent's hand remains findable by the triggered Event (citing Turn 4 Headline of game 206). However, `SELECT_CARD` unconditionally sets `state.ctx().pending_op_card = card` (`state_machine.cpp:1323`), and `SELECT_PLAY_MODE` (`Resolution::EVENT && !opponent_card`, `state_machine.cpp:1428-1441`) **never clears `pending_op_card`** (likewise `trigger_grain_sales` at `mid_war.cpp:479` sets `pending_op_card = chosen_card` in both Action Round and Headline).
- **Impact:** Whenever `MISSILE_ENVY` is played as an Event in an Action Round (or via Grain Sales in AR/Headline), hands itself to the opponent's hand, pushes a child frame, and triggers a hand-scanning Event (direct or tie path), `ctx_stack[0].pending_op_card == MISSILE_ENVY` causes `card_in_play(MISSILE_ENVY)` to return `true`:
  - `FIVE_YEAR_PLAN` (`early_war.cpp:46`) skips `MISSILE_ENVY` in USSR's hand (even if sole card, diverging from Headline behavior and contradicting game 206);
  - `GRAIN_SALES` (`mid_war.cpp:417`) skips `MISSILE_ENVY` in USSR's hand;
  - `TERRORISM` (`late_war.cpp:171`) skips `MISSILE_ENVY` in the opponent's hand;
  - `BLOCKADE` and `LATIN_AMERICAN_DEBT_CRISIS` (`early_war.cpp:114`, `late_war.cpp:200`, `card_dispatcher.cpp:549, 1498, 2098`) exclude `MISSILE_ENVY` from legal 3-Op discards under `CONTAINMENT_ACTIVE`.
- **Fix direction:** Clear `state.ctx().pending_op_card = 0;` (and set `state.ctx().resolving_card = card;`) at `state_machine.cpp:1428` before `trigger_event`. *(Engine change — requires owner approval per [AGENTS.md](../../AGENTS.md) Invariant 11.)*

#### V-2. `E-7` regression (`0d84934`): tautological `spent_own` in Headline `advance_after_ops` discards `MISSILE_ENVY` from USSR's hand
- **Locations:** `engine/src/state_machine.cpp:359-365, 500-532, 1428-1441`
- **Cause:** At `state_machine.cpp:516-532` (Headline `advance_after_ops`), after line 517 confirms `is_in_any_hand(state.card_locations[op_card])`, line 526 computes `const Player spender = ops_spender(state, op_card);`. Because `ops_spender` (`state_machine.cpp:359-365`) returns the owner of whichever hand currently holds `op_card`, `in_hand_of(state.card_locations[op_card], spender)` is **tautologically `true`**, making `CardHandlers::Handover::Respect` on line 531 unreachable dead code whenever `op_card == MISSILE_ENVY`.
- **Impact:** In `Phase::HEADLINE`, when headlined `Grain Sales` draws `MISSILE_ENVY` to US (`ctx_stack[0].pending_op_card = MISSILE_ENVY`), US plays `MISSILE_ENVY` as `Resolution::EVENT` (leaving `pending_op_card` set), `MISSILE_ENVY` hands itself to `HAND_USSR_KNOWN` (`forced_card_player = USSR`) and triggers an Ops-granting US/Neutral Event (`ABM_TREATY`, `CIA_CREATED`, `JUNTA`, `KAL_007`, `TEAR_DOWN_THIS_WALL`, `OLYMPIC_GAMES` boycott, or `GRAIN_SALES` on empty hand) on `ctx_stack[1]`: when those Ops finish, `advance_after_ops` pops `ctx_stack[1]` (lines 500–506) and falls through to lines 515–532 on `ctx_stack[0]`. `spent_own` evaluates `true` and `relocate_played_card(..., Handover::Ignore)` **discards `MISSILE_ENVY` from USSR's hand to `DISCARD_PILE`**, after which `StateMachine::step:952` wipes `forced_card_player = NONE`. (Pre-`0d84934` kept `MISSILE_ENVY` in `HAND_USSR_KNOWN` via `Handover::Respect`.)
- **Fix direction:** Clear `pending_op_card = 0` at `state_machine.cpp:1428`; in Headline `advance_after_ops`, skip lines 515–532 if a pushed frame was popped (mirroring the Action Round branch at lines 542–556), and do not compute `spent_own` via `ops_spender`. *(Engine change — requires owner approval.)*

#### V-3. `T-14` regression (`f7218d8`): `--resume-fresh-optimizer` skips reordering yet still transforms & loads the old optimizer state
- **Locations:** `ai/training/generic_trainer.py:1751-1857, 1605-1611`, `ai/models/grow.py:167-169`
- **Cause:** `generic_trainer.py:1751` guards `optimizer_state_in_model_order(...)` with `if not fresh_optimizer:`, **but** lines `1772` (`widen_optimizer_state`), `1791` (`grow_tokens_optimizer_state`), `1821` (`grow_macro_optimizer_state`), `1834` (`_optimizer_state_with_added` + `_load_optimizer_state`), and `1847` (`_load_optimizer_state`) still run unconditionally *before* `if fresh_optimizer: trainer.optimizer.state.clear()` at lines `1853-1856`.
- **Impact:** Resuming a pre-`f7218d8` `--no-decay-norms-biases` state (which stores 2 non-model-ordered parameter groups) with `--resume-fresh-optimizer` (with or without grow flags) skips reordering, then feeds mismatched indices/shapes into `regroup_optimizer_state` / `load_state_dict` (or into a grow transform) and crashes before line `1856` clears the optimizer state.
- **Interaction with unfixed `T-15`:** Because post-`f7218d8` `--no-decay-norms-biases` creates dozens of contiguous `param_groups` and `optimizer_state_in_model_order` returns already-ordered states untouched (`nash_pg.py:423-424`), resuming *any* post-`f7218d8` `--no-decay-norms-biases` checkpoint with `--resume-grow-tokens` now crashes in `grow_tokens_optimizer_state` (`ai/models/grow.py:167-169`).
- **Fix direction:** Skip lines `1772`, `1791`, `1821`, `1834`, and `1847` (all optimizer transforms and loads) whenever `fresh_optimizer=True`; and fix `T-15` in `grow_tokens_optimizer_state`.

---

### Medium — Incomplete Fixes & Edge-Case Bugs

#### V-4. `T-1` incomplete (`0e9221e`): back-to-back `ROLL_DIE` nodes miss `BEFORE_ROLL` and `add_roll_faces`
- **Locations:** `bindings/ts_bindings.cpp:1934-1949`, `ai/training/afterstate_critic.py:103-113`
- **Cause:** `node_tensors()` (`afterstate_critic.py:396-454`) and `ts.gae_per_player` (`ts_bindings.cpp:1697-1706`) properly handle a forced step before roll 1 and multiple emitted `BEFORE_ROLL` rows. However, in both C++ and Python `walk_afterstates`, after roll 1 resolves, `keep_afterstate(..., kAfterRoll, forced)` pushes `kAfterRoll` and sets `fresh = true`. If the engine is immediately at a second `ROLL_DIE` chance node on the next iteration — e.g., Ops-first Coup followed by owed `Korean War`, `Arab-Israeli War`, or `Summit` (`engine/src/events/mid_war.cpp:92-98`, `engine/src/state_machine.cpp:567`), or two immediate-roll Headlines (`state_machine.cpp:1711`) — then `if (!fresh)` is `false` and `kinds.back() == kAfterDecision` is `false` (`kinds.back()` is `kAfterRoll`).
- **Impact:** Roll 2 receives no `BEFORE_ROLL` row and no `add_roll_faces` call (emitting `[BEFORE_ROLL, AFTER_ROLL, AFTER_ROLL]`), leaving roll 2's luck inside the advantage.
- **Fix direction:** When `at_chance(st)` and `fresh` is `true` with `kinds.back() == kAfterRoll`, also emit a `kBeforeRoll` row (or convert/tag appropriately) and call `add_roll_faces`.

#### V-5. `T-4(2)` incomplete + reward/discount mismatch + perf waste (`0e9221e`)
- **Locations:** `ai/training/nash_pg.py:2458-2464`, `ai/training/afterstate_critic.py:457-487`, `bindings/ts_bindings.cpp:1675-1708`
- **Issues:**
  1. **Incomplete coverage:** `nash_pg.py:2459` populates `_last_side` only when `self._as_rows` is non-empty. Bare `--per-player-gae` (no `--afterstate-value-coef`) and uncaptured last decisions when `--afterstate-frac < 1.0` still bootstrap from `-last_v`.
  2. **Reward double-counting / over-discounting:** `last_moment_values()` supplies `V_own(s_{lastdec+after})` as `cV[q]` at `t = T` in `ts.gae_per_player`. Harmless for flagship `gamma = 1.0` with terminal-only rewards (`cR = 0`), but with intermediate rewards (`--vp-potential`, `shaped`, `useful_actions`, dense `RegionTargetedReward`) `cR[q]` adds all rewards in `lastdec+1..T-1` *on top of* `V_own(s_{lastdec+after})` (double-counting them), and with `gamma < 1.0` it discounts `V_own(s_{lastdec+after})` by `gamma^(T - lastdec)` instead of `gamma^1`.
  3. **Wasted inference:** Under `--per-player-gae --afterstate-value-coef > 0` without `--afterstate-advantage`, `nash_pg.py:2461-2462` runs `read_all_node_values` over the *entire* rollout's `_as_rows` solely to read at most one row per env in `last_moment_values`.

#### V-6. `M-7` bugs & deleted test coverage (`ba8828e`)
- **Locations:** `ai/training/nash_pg.py:3065, 3174-3182`, `tests/training/test_update_bf16.py:1-39`, `ai/models/coldwar_net_v2.py:50`, `ai/eval/plasticity_probe.py:93, 144`, `ai/models/ladder_net.py:772-777`
- **Issues:**
  1. **Dropout mismatch:** `nash_pg.py:3065` calls `self.active_net.train()` *before* `old_lp_amp_all` is computed at `3174-3182` (unlike `reference_net`, which stays in `.eval()`). For any network with dropout (`res_dropout` defaults to `0.05` on `ColdWarNetV2`/`LadderNet`; `ColdWarNetV2.cross_attn` has `dropout=0.05`), `old_lp_amp_all` draws an independent random dropout mask different from epoch-0 minibatches (and consumes torch RNG on CPU too since line `3175` checks `self.update_bf16` without checking CUDA). Also under `--compile-update --update-bf16` with `use_risk` or `categorical_value`, `_act_fwd` compiles `active_net.forward` solely for this pre-pass.
  2. **Deleted tests:** `ba8828e` overwrote `tests/training/test_update_bf16.py`, deleting both `test_fp32_heads_change_nothing_without_autocast` and `test_fp32_heads_need_bf16`; no `fp32_heads` tests remain in `tests/`.
  3. **`plasticity_probe` still crashes on macro models:** `coldwar_net_v2.py:50` claims `pad_mask` covers `plasticity_probe`, but `ai/eval/plasticity_probe.py:93` still executes `model(obs, masks)[0].float().masked_fill(~masks, -1e9)` and line `144` multiplies `y[b]` (220) by `F.log_softmax(lg, dim=1)` (304). (Same redundant narrow re-masking in `ai/eval/departure_probe.py:84` and `ai/training/policy_fit.py:238`.)
  4. **`forward_prior_gain` under `--bf16-fp32-heads`:** `LadderNet.forward_prior_gain` (`ladder_net.py:772-777`) still omits `self._heads_in(...)`, leaving `prior_gain_head` in bf16 (same for `forward_aux`, `forward_card_aux`, `forward_opp_legal`, and `coldwar_net_v2.py:741` `defcon_risk`).

#### V-7. `M-4` & `M-6` missed consumers (`2198e12`, `0119569`)
- **Locations:** `ai/training/policy_fit.py:99-100, 162`, `ai/training/q_fit.py:68, 72, 236`, `ai/eval/plasticity_probe.py:46-47, 111, 125-127`, `tools/scripts/average_weights.py:48-53`
- **Issues:**
  1. `policy_fit.py:99-100` still defines its own `POLICY_MODULES` omitting `pe_macro` and `head_norms` (`q_fit.py:68, 236` imports it and leaves them frozen under `head="policy", train_trunk=False`).
  2. `policy_fit.py:162` and `q_fit.py:72` hardcode `("val_win_head", "val_vp_head", "val_logits_head")` instead of `head_registry.VALUE_HEADS` — **`"val_logits_head"` does not exist**, so `value_dist_head` is never frozen in either trainer.
  3. `plasticity_probe.py:46-47` uses `POLICY_HEAD_MODULES` (omitting `head_norms`, leaving it un-copied in `with_head` and frozen in `fit`) and omits `"sphere"` from `TRUNK`, so `sphere.*` always has `requires_grad=False` and `register_sphere_projection` at line `127` is a dead hook.
  4. `tools/scripts/average_weights.py:48-53` still performs Euclidean averaging on `sphere.blocks.*.w1/w2.weight` without re-projecting rows onto the unit sphere.

#### V-8. `T-13` edge case (`f95503c`): Final Scoring Europe Control omits `±40 VP` in `RegionTargetedReward`
- **Locations:** `ai/rewards/region_reward.py:48-49, 148-162`, `engine/src/scoring.cpp:313-327`
- **Details:** In-game Europe Control (`Scoring::score_region` via `CardHandlers::recorded`) correctly books `±40 VP` into `last_event_tracks`, fixing `T-13`'s score-dependence inversion. However, **Final Scoring** Europe Control (`Scoring::execute_final_scoring`) does not go through `recorded()` (`last_event_tracks` is empty) and sets `EUROPE_CONTROL_WIN`, so `game_ending_reason` returns `"Europe Control"` and `reached_final_scoring(st)` (`region_reward.py:49`) returns `False`. Final Scoring Europe Control therefore adds `0` instead of `±40` to `scored_vp` (paying `control_win = +1.0` instead of `+1.8` on win, `-1.0` instead of `-1.16` on loss, and omitting `±40` from `region/vp_per_game`).
- *Minor web display note:* `web/ui/src/game/describe.ts:287-303` (`describeEventTracks`) now logs `"Europe Scoring: US +40 VP"` alongside `"VP: x -> +20"` on in-game Europe Control because `tracks` sums to `±40 != netVp`.

#### V-9. `S-3` & `S-11` residual edge cases (`9d0857d`, `0567fca`)
- **Locations:** `bindings/belief.cpp:44-67, 145-165`, `ai/search/belief.py:89-104`, `bindings/wasm/ts_engine_wasm.cpp:873-897`
- **Issues:**
  1. **Partial-fold posterior loss (`belief.cpp:145-165`):** Because one `job.decisions` list drives both BMA evidence and hand weights, when an earlier decision drops while a newer decision stays valid (e.g., SALT Negotiations puts into `me`'s hand a card `opp` played between `d=0` and `d=1`), folding `current_decisions = [0, 1]` into `done` marks `1` as `spent` and removes `1` from `job.decisions` — preventing BMA double-counting, but also dropping `1` from the current position's hand posterior.
  2. **Ultra-rare `opp_unknown -> DRAW_DECK` false positive (`belief.cpp:44-67`, `belief.py:89-104`):** With `mine && d.locs[c] == ou` rejected, the count check simplifies to `(opp_unknown -> deck) == (deck -> elsewhere) + (deck -> opp_unknown)`. `OUR_MAN_IN_TEHRAN` (`engine/src/events/mid_war.cpp:646-650`) reshuffles discard into deck while leaving `HAND_US_UNKNOWN` intact; when `me == USSR` and `DISCARD_PILE` was empty at `d`, if US played a card to discard and later played Tehran reshuffling it into `DRAW_DECK` while discarding 1 deck card, `left_since` still falsely passes. Adding `if (now[c] == kDeck && d.locs[c] == ou) return false;` closes it.
  3. **Minor `S-11` gaps:** WASM `ts_belief_rows` (`ts_engine_wasm.cpp:873-893`) still omits `if (features & ~ts::obs_features::ALL)`; `require_no_nan` (`ts_bindings.cpp:1357`, `ts_engine_wasm.cpp:897`) permits `±inf`, where direct API calls passing `+inf` or all-`-inf` produce `NaN` in `logw - m` (`belief.cpp:220`).

#### V-10. Minor schema, test, RNG & documentation leftovers
- **Schema drift (`ec4e80c`):** `bindings/state_json.cpp:487` exports `headline_first_owner`, but `web/server/replay_types.py:92-97` (`GameStateDict`) and `tests/bindings/replay_schema.golden.json:62-66` were not updated to include `headline_first_owner: str`.
- **`T-9` RNG leak (`5397baf`, `3746b8a`):** `StartPositionPool.harvest()` (`ai/training/start_pool.py:224-225`, called at `generic_trainer.py:3715`) still consumes global torch RNG without `@preserves_training_rng`.
- **`S-7` test gap (`6b3a859`):** Added a unit test for `S-6` only, none for `S-7`.
- **`M-3.3` legacy compat (`c305f73`):** Any pre-`c305f73` checkpoint built off defaults (`num_attn_heads != 4` or non-conventional `card_lookup_dim`) lacks `num_attn_heads_buf`/`cl_heads_buf` and will fail `strict=True` / `_load_allowing_added_heads` on resume.
- **`R-2` / `R-4` doc gaps (`0ffad01`):** [research/log/README.md](../log/README.md) does not yet index the 4 new `P51`–`P55` logs (or `audit_2026-11-22_verification.md`); `P53_compile_bf16.md:26` and `P55_critic_everywhere.md:17` cite non-ancestor commit `a18694c` instead of `0e9221e`; [research/runs.md](../runs.md) notes `T-6` on P48b (`L137`) but not `T-10`/`T-13` on P49b (`R141–R163`, `L139, L141`) or `M-1` (`w2 x0.25`) on `E8-A30-R95-S44` (`L157`). Also `AGENTS.md:107` Mermaid diagram still labels the pool node as 30% (invariant 14).

---

## 2. Verified Fixed

### Engine (`0d84934`, `f95503c`, `ec4e80c`)
- **E-1:** `card_dispatcher.cpp:714, 739-740` saves and reads `roll_actor` for `OLYMPIC_GAMES`.
- **E-2:** `card_dispatcher.cpp:1285-1298` executes `STAR_WARS` USSR cards as `Player::USSR`.
- **E-3:** `state_machine.cpp:1364-1368, 1413-1417` gates `settle_we_will_bury_you` on `phasing_player == US && ctx_stack_depth == 0` and checks `EVENT && UN_INTERVENTION`.
- **E-4:** `state_machine.cpp:852-856` removes `|| state.forced_card_id == card_ids::MISSILE_ENVY`.
- **E-5:** `early_war.cpp:63-70` runs `trigger_event` before `relocate_played_card` in `FIVE_YEAR_PLAN`.
- **E-8:** `state_machine.cpp:1208` resets `allow_early_stop = 0` at play-mode commit.
- **E-9:** `state_machine.cpp:392-393` clears `start_influence_nodes` and resets `op_mode = OpMode::INFLUENCE` in `fire_owed_event`.
- **E-10:** `card_dispatcher.cpp:337-344, 1289-1294` and `state_machine.cpp:284-287, 1169-1172` charge Flower Power for US-played USSR war events in Headline & Star Wars.
- **E-11:** `mid_war.cpp:173-174` and `card_dispatcher.cpp:1157-1159` route headlined `MISSILE_ENVY` taking `UN_INTERVENTION` to Ops.
- **E-12:** `action_mask.cpp:129, 136, 141, 207` offers optional AR8 pass in trapped and Missile-Envy blocks.
- **E-13:** `state_machine.cpp:1509` routes Headline `SELECT_OP_MODE` decline through `advance_after_ops`.
- **E-14:** `observation.cpp:245-259` hides `OUR_MAN_IN_TEHRAN` `PEEKED_TEMP` cards from `Player::USSR`.
- **E-15:** `observation.cpp:506-510` resolves `Player::NONE` before all optional feature blocks.
- **E-16:** `observation.cpp:347-351` marks the first headliner's own committed card with `ACTIVE_NOW` (`headline_stage == 0`, zero opponent leak); `observation.golden.npz` updated.
- **E-17:** `engine.cpp:196-213` validates `forced_event_target` against canonical masks; `state_machine.cpp:959-963` clears `canon_prio` on pop.
- **E-18:** `action_mask.cpp:826-845` guards `apply_macro_expansion` on mode, positive influence gain, and unchanged context frame.
- **E-19:** `constants.hpp:400`, `state_machine.cpp:781`, `engine.cpp:280` add unobserved `HELD_SCORING_LOSS` (bit 50).
- **T-10:** `constants.hpp:403`, `card_dispatcher.cpp:1262`, `state_json.cpp:1175-1188` add `WARGAMES_END` (bit 51) and classify `ending_reason` strictly from engine flags/turn, wired through `ts_env.py`, `reward_calculator.py`, and web i18n (`en`/`zh-CN`).
- **Minor 1 & 3:** `card_dispatcher.cpp:1160-1175` pushes a fresh frame on `MISSILE_ENVY` tie-path events; `state_machine.cpp:1232-1233` uses `std::memcpy` for `before_ctx`.
- **`ec4e80c`:** Web UI reads `headline_first_owner` (with legacy fallback), E9 digest updated in `leaderboard/epochs.json`.

### Training (`b665fef`, `2847fab`, `b2aec22`, `054f8f0`, `4be3cb0`, `93f0eed`, `5397baf`, `3746b8a`, `1b7dcb2`, `7b8671b`)
- **T-3 (`93f0eed`):** `rollout_buffer.py:543-554` uses `torch.zeros_like` and falls back to batch-wide `(mean_adv, std_adv)` when a seat has `<= 1` stats row.
- **T-4(1) (`b2aec22`, `93f0eed`):** Blunder window removed; `slice_turn_boundaries` wired into `_gae_per_player_both` (`rollout_buffer.py:321, 631-635`).
- **T-6 (`b665fef`):** `restart_pool.py:154, 195-196, 284-285` snapshots `finished_v` in `provide()` before clearing `v_now` and consumes it in `complete()`.
- **T-7 (`5397baf`):** `generic_trainer.py:669` always calls `_episode_group_stats(whole, "")`.
- **T-8 (`5397baf`):** `restart_pool.py:72, 76-86, 133` refuses turns `< 3` (`TODO(pool)`).
- **T-11 & `7b8671b` (`b2aec22`, `054f8f0`, `4be3cb0`):** Blunder machinery removed cleanly; `--decisiveness-turns` and `--window-provoked-defcon` retired with `parser.error` on non-defaults (`train.py:1457`).
- **T-12 (`3746b8a`):** `generic_trainer.py:2252-2258` refuses `--vp-potential` with region/ending targets.
- **T-16 (`1b7dcb2`):** `online_distill.py:87` removes the extra `/ 20.0` on `v_vp`.
- **T-18(3, 4, 7) (`2847fab`):** `--mode warmup --arch mlp` builds `ColdWarNetMLP` (`train.py:1754-1755`); per-component resume reseeding works (`generic_trainer.py:1451-1502, 1889-1913`); `_load_optimizer_state` strips stale saved group keys like `lr_mult` (`1626-1628`).

### Models (`b716950`, `2198e12`, `c305f73`, `0119569`, `2ca9119`, `760a06f`)
- **M-1 (`b716950`):** `mup.py:58-69` preserves sphere/block kwargs in `base_shapes_for`, fixing the `0.25x` LR penalty on `SphereBlock.w2`.
- **M-3.1 (`760a06f`):** `ladder_net.py:1301-1312` recovers `entity` + `identity_dim` + per-entity heads cleanly.
- **M-3.3 (`c305f73`):** `ladder_net.py:84-90, 339-340, 440-441, 1278-1283` persists non-default `num_attn_heads_buf` and `cl_heads_buf` without changing default checkpoints.
- **M-4 (`2198e12`):** `ai/models/head_registry.py` created and wired into `mup.py`, `head_l2.py`, `coldwar_net_v2.py`, `ladder_net.py`, `reset_policy_head`, and `grow.py`.
- **M-6 (`0119569`, `2ca9119`):** `project_sphere` / `register_sphere_projection` hooked into `make_optimizer`, `_HumanInjector`, `run_search_distillation`, `run_behavioral_cloning_warmup`, `BehavioralCloningTrainer`, `value_fit`, `policy_fit`, `q_fit`, `card_board_probe`, and `WeightEMA.update`.

### Search, Bindings & Research (`6b3a859`, `9d0857d`, `316a3a2`, `2242a36`, `1bc664d`, `7afc0a7`, `0ffad01`, `1e3ee6a`, `7e8f8be`, `0d1f926`)
- **S-4 (`9d0857d`):** `Now.of(reveal=)` (`ai/search/belief.py:78-86`) mirrors C++ `revealed_headline`; `belief_site_probe.py:249, 266` passes `self.reveal`.
- **S-6 (`6b3a859`):** `rollout_batch.cpp:27, 615` sets `kMinZWorlds = 4` before switching off candidate 0.
- **S-7 (`6b3a859`):** `rollout_batch.cpp:32-70, 456-496` guards CMC at any DEFCON and `CHE`/`ORTEGA` target choices, and keys `drain_chance` per roll to match `keyed::advance`.
- **S-8 (`316a3a2`):** `rollout_batch.cpp:195, 201-215` plays Turn 10 through Final Scoring and probes deterministic Mil-Ops/Held-Scoring terminals on Turns 1–9 before parking.
- **S-9.1..5 (`2242a36`):** Root mask intersection in `RolloutRoot._forward` and `web/ui/src/analysis/rollout.ts`; `Entry`/`Sparse` wrappers delegate `model`/`obs_features`/`observe`/`new_games`/batch methods; `batch_tournament.py` passes `_agent_features` to `refuse_learned_order`; `WorkerRolloutConfig` forwards full `RolloutConfig`; `dmcts.determinize` clears `forced_deal_*` and supports `keep_revealed`/`box4`; `BatchedMCTS` Python backend adds `seat_tree` and `select_action` delegates to `select_actions_batch`.
- **S-10 (`1bc664d`):** `BatchedSearch.set_buffers` adds `keep_alive` + `own_viewed_` guard; `VectorizedBatchRunner` views use `SharedBuffer<T>` (`std::shared_ptr` + capsule).
- **S-12 & `BIND-1` (`1bc664d`, `7afc0a7`):** `GameState.ctx_stack_depth` is `.def_ro`; `BIND-1` closed in `BUGS.md` with ASan-verified lifetime test (`tests/bindings/test_binding_lifetimes.py`).
- **R-1, R-3, E9 leaderboard (`0ffad01`, `1e3ee6a`, `7e8f8be`, `0d1f926`):** Invariant 14 rewritten in `AGENTS.md` / `CLAUDE.md` and `watch_run.py` suppresses `nopool` when `opponent_frac == 0`; `P55_critic_everywhere.md` records matched-step `R169 vs B6` reads and the S45 single-seed caveat; E9 anchor set to B6 soup (1500) and B6 `keyed2` registered.

---

## 3. Still Unfixed vs. Intentionally Deferred

### Confirmed in `audit_2026-11-22_verification.md`, Still Unfixed
- **`S-1` (belief float32 `-1e9` evidence corruption):** Completely unfixed (`bindings/belief.hpp:53, 97-98`, `bindings/belief.cpp:75, 147, 223, 274`, `ai/search/belief.py:204`, `web/ui/src/analysis/rollout.ts:98`, `tools/lib/batch_tournament.py:832-833`). [P57_belief_against_people.md](../plans/P57_belief_against_people.md) (`L38-40`) notes the owner ruled zero-probability decisions impossible in a game and only plans to count them, leaving two real triggers unguarded: (1) finite-sample exhaustion across 2,048 particles on rare constraints (Quagmire/Bear Trap scoring-card plays when 0/2,048 sampled hands lack a 2+ Op card, or 1-Op Missile Envy gifts), and (2) `MERGED`/`MACRO` opponents fed into E4-only `hypothesis_masks` in `batch_tournament.py`.
- **`T-5` (critic-prior refusals & interactions):** 0 of 6 sub-items addressed — still no refusals for `--critic-prior-window > 0` combined with `--afterstate-*`, `--predeal-value-coef`, `--value-learner-only`, or `--same-perspective-bootstrap`, and `--opponent-frac 0 --opponent-per-game --critic-prior-window > 0 --critic-prior-gain learned` still hits `AssertionError` (`nash_pg.py:1500`).
- **`T-15` (multi-group `grow_tokens_optimizer_state` & combined grow alignment):** Untouched (`ai/models/grow.py:167-169`, `ai/training/generic_trainer.py:1768`), and now triggered by post-`f7218d8` `--no-decay-norms-biases` checkpoints as well as `--mup` and `--policy-head-lr-mult`.
- **`S-9.6` (unfixed remainder):**
  - In-place `seat_decider` mutation in `RolloutRootAgent` (`rollout_root.py:528-538`) and `BatchedMCTSAgent` (`batched_mcts.py:879-880`), corrupting `position_token(st)` in `disagreement_bank.py:141, 163`.
  - `RolloutRootAgent.concurrent_decisions = True` thread race with CUDA graphs and unchunked `_GraphedForward.__call__` (`rollout_root.py:176-184, 258-261, 498`).
  - Empty-batch `np.stack([])` crashes in `RolloutRoot._forward` (`rollout_root.py:285`) and `GumbelRoot._network` (`gumbel_root.py:171`).
  - `GumbelRoot._halve` empty-mask `NaN`, ignored `advance_root`, and unseated `_network` (`gumbel_root.py:169-176, 232-235, 273-278`).
  - `macro_influence` ignored in `ai/eval/loading.py:17-41` and `ai/eval/playout.py:39-61`; unseated masks in `disagreement_bank.py:98, 134` and `network_health.py:165`.

### Intentionally Deferred / Retired / Owner Decisions
- **`T-2` (`0e9221e`):** Total-weight semantics documented rather than rescaled; eager uncompiled `_value_only` retained.
- **`T-4(3)` (`93f0eed`):** Duplicate `block_mode()` removed; `_gae_backward` retained for `returns_vp`/`defcon_risk_target`; wasted `next_own_t` under `--same-perspective-bootstrap` + `--per-player-gae` left.
- **`T-11` (`b2aec22`, `054f8f0`, `4be3cb0`):** Retired (blunder machinery removed).
- **`T-17` (`f942fe8`):** Pool marked unmaintained (`TODO(pool)`); future-snapshot directory rebuild refused, fast-game exposure drop and step-0 member drop warned/counted only, other pool defects (`seat_balance` mask mutation before episode filtering, league-only `env_member = -1` bypass, untrained fallback `deepcopy(model)`, minor state bugs) left untouched.
- **`T-18(1, 2, 5, 6)` (`2847fab`):** Deferred as fail-loud (`cuda:1`, `CUBLASLT_WORKSPACE_SIZE`, `--resume-add-rules` missing buffer, `--ema-weights` + `ADDABLE_HEADS`).
- **`M-1` (non-sphere), `M-2`, `M-3.2`, `M-4` (secondary items), `M-5`, `M-6` (regularization conflict), `M-7` (bf16 logit cap / double `value_dist_head`):** Deferred (unused configs / fail-loud / low priority).
- **`S-2` (`9587dbc`):** Deferred to [P57_belief_against_people.md](../plans/P57_belief_against_people.md) (not yet run).
- **`S-5` (`0d1f926`):** B6 `keyed2` registered on E9 (`leaderboard/players.json:613-629`); `leaderboard/matches/E9.jsonl` is empty and defaults remain `chance=keyed`.
- **Engine Minor 2:** `SUMMIT` forced-dice convention intentionally kept (`engine/AGENTS.md:771-772`) for `ts_replayer_convert.py`.
