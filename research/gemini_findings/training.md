# Training Loop, Buffers, Pools & Rewards (`ai/training/`, `ai/rewards/`, `bindings/`)

Audit at commit `fb9b817` (2026-11-22).

---

## 1. Active P55 / P56 & Rollout Buffer Bugs (`R169`, `R170`)

### T-1. [High] P56 `--afterstate-advantage` misses die rolls preceded by a forced auto-advanced step
- **Locations:** `ai/training/afterstate_critic.py:403-413`, `bindings/ts_bindings.cpp:1882-1910`
- **Cause:** `node_tensors()` checks only the first emitted afterstate row per env:
  ```python
  is_roll = (kinds[first] == BEFORE_ROLL) & np.isfinite(avg[first])
  ```
  Whenever a decision auto-advances through a single-choice step before rolling (e.g., single legal coup/realignment target, single play mode, 2nd headline choice), `walk_afterstates` emits `[AFTER_DECISION, BEFORE_ROLL, AFTER_ROLL]`. `kinds[first]` is `AFTER_DECISION`.
- **Impact:** `node_tensors()` ignores row 1's `BEFORE_ROLL`, sets `U = V(AFTER_DECISION)` instead of the exact die average `avg`, and sets `U_post = U`, leaving die-roll luck inside the advantage (while `build_targets` still trains row 1 with `avg`).
- **Fix direction:** Scan the env's emitted slice `[first .. next_first)` for any `BEFORE_ROLL` row with finite `avg` (and pair with its matching `AFTER_ROLL`), or mark whether the step crossed a roll.

### T-2. [High] P55 / P56 extra critics (`R169`, `R170`) over-weight pre-deal rows ~150–250× and run uncompiled
- **Location:** `ai/training/nash_pg.py:3444-3463`
- **Cause:** `_pd_loss` computes `.mean()` over only ~60–120 pre-deal rows per minibatch (vs 16,384 main decision rows) and adds `vf_coef * predeal_value_coef * _pd_loss` with `predeal_value_coef = 1.0`. Both extra critics also call eager `_value_only(self.active_net, ...)` instead of a compiled `forward_value` via `_update_method`.
- **Impact:** Each pre-deal row gets ~150–250× the per-row gradient weight of a normal decision row (injecting high-variance critic gradients into the shared trunk) and total critic weight is tripled (`3 * vf_coef`) relative to policy loss. Eager forwards also leave throughput on the table under `--compile-update max-autotune`.
- **Fix direction:** Weight auxiliary critic losses proportionally to their row count relative to the main minibatch (or lower `predeal_value_coef` accordingly) and compile `forward_value`.

### T-3. [Critical] `RolloutBuffer.normalise_advantages()` leaves uninitialized GPU memory (`torch.empty_like`) under `--per-seat-adv-norm`
- **Locations:** `ai/training/rollout_buffer.py:596-613`, `ai/training/nash_pg.py:3230-3232, 3273`
- **Cause:**
  ```python
  normed = torch.empty_like(flat)
  for code in (1, -1):
      sel = fp == code
      st_sel = sel if own is None else (sel & own)
      if int(st_sel.sum()) > 1:
          s = flat[st_sel]
          normed[sel] = (flat[sel] - s.mean()) / (s.std() + 1e-8)
  ```
- **Impact:** Under `--per-seat-adv-norm --adv-norm-learner-only` with a locked seat (`--opponent-lock-side us/ussr --opponent-frac 1.0`), the opponent seat has `st_sel.sum() == 0`, leaving ~50% of `normed` uninitialized. `nash_pg.py` computes `surrogate` with `b_adv` *before* multiplying by `_keep_f = 0`, so any `NaN`/`Inf` in uninitialized memory yields `NaN * 0 = NaN` and corrupts `ppo_loss` and weights (and always corrupts `diagnostics()`, `ratio_negadv_max`, and `priority_indices`). Likewise if a seat has `st_sel.sum() == 1`, that learner row trains on garbage.
- **Fix direction:** Initialize `normed = torch.zeros_like(flat)` and handle `st_sel.sum() == 1` explicitly.

### T-4. [Medium] `--per-player-gae` leaks blunder windows across turns, ignores `--slice-turn-boundaries on`, and wastes compute
- **Locations:** `ai/training/rollout_buffer.py:319-346`, `bindings/ts_bindings.cpp:1670-1703`, `ai/training/nash_pg.py:2270`
- **Impact:**
  1. `_gae_per_player_both` ignores `window_mask` and `slice_turn_boundaries`: with `--per-player-gae --blunder-window`, `keep = ~window_mask` keeps only the blunder turn's override while earlier turns receive `pp_adv`/`pp_ret` that propagated the `-1` blunder reward across the turn boundary. With `--slice-turn-boundaries on`, turn boundaries stay silently unsliced.
  2. At the rollout boundary `t = T`, `ts.gae_per_player` (`ts_bindings.cpp:1675`) sets `cV[0] = LV[e]; cV[1] = -LV[e]`, so the non-moving side at `t = T` still bootstraps from the negated opponent view (~1/128 of steps).
  3. `compute_gae` runs the full interleaved `_gae_backward` pass even when `per_player_gae=True` (and `nash_pg.py:2270` still pays an extra forward for `next_values_own` under `--same-perspective-bootstrap`, which `_gae_per_player_both` discards). Duplicate `mode = self.block_mode()` at lines 319 and 323.

### T-5. [Medium-High] `--critic-prior-*` conflicts with `--afterstate-advantage`, extra critics, `--value-selfplay-only`/`--value-learner-only`, `--same-perspective-bootstrap`, and `--opponent-frac 0`
- **Locations:** `ai/training/nash_pg.py:1441, 2270-2276, 2394-2401, 2929-2954, 3118-3123, 3444-3463`, `ai/training/afterstate_critic.py:311-325`, `ai/training/opponent_pool.py:634`
- **Impact:**
  1. `build_afterstate_targets`, `_build_predeal_targets`, and `read_all_node_values` evaluate/train raw unshifted `_value_only(active_net, ...)` (no `prior_apply`) against prior-shifted `buffer.returns_win` and pool terminal results (`ends` ignores `learner`, `value_selfplay_only`, and `value_learner_only`). On pool die rolls where `after_ok == False`, `an = (ut_unshifted - v_t_shifted) + ...` injects a `-shift` bias into advantages.
  2. `next_own_t` (`nash_pg.py:2270-2276`) is never shifted by `prior_apply`, injecting `-shift` TD error on pool rows under `--same-perspective-bootstrap`.
  3. `next_decision_targets` (`afterstate_critic.py:322`) assumes zero-sum terminal rewards (`rewards[dn, e] * players[dn, e] * side`), silently wrong for non-zero-sum reward calculators.
  4. `nash_pg.py:2400` uses end-of-rollout `self._selfplay_mask` instead of per-row self-play status if `seat_balance` flipped envs mid-rollout.
  5. Combining `--opponent-frac 0.0` with `--opponent-per-game --critic-prior-window > 0 --critic-prior-gain learned` crashes with `AssertionError` at `nash_pg.py:1441`.
  6. `prior_rate_of` (`opponent_pool.py:634`) initializes brand-new snapshots to `0.5` rather than `_prior_self(side)`, giving a fresh self-snapshot a nonzero initial `beta` whenever self-play is not 50/50.

---

## 2. Restart Pool & Start Pool (P48b / `R138`)

### T-6. [Critical] `RestartPool.provide()` wipes `v_now` before `complete()` reads it
- **Locations:** `ai/training/restart_pool.py:174-195, 260-273`, `bindings/ts_env.py:663`, `ai/training/nash_pg.py:2215-2224`
- **Cause:** Inside `TsVectorizedEnv.step()`, done envs auto-reset and call `start_provider(i)` (`restart_pool.provide(i)`) *before* `step()` returns to `nash_pg.py:2223` (`restart_pool.complete()`). Line 194 of `provide()` executes `self.v_now[env_idx] = None` for the *new* continuation before `complete()` reads `v = self.v_now[env_idx]` (line 264) for the *finished* continuation.
- **Impact:** On ~100% of completions (whenever any bucket is stocked), `complete()` gets `v is None` and falls back to the stale admission value `pos.p_admit`, invalidating `pos.v_values`, disagreement `z`, `put_back_probability`, `put_back_by_disagreement`, and `restart/critic_excess_turn{t}` in P48b (`R138`).
- **Fix direction:** Save `self.finished_v[env_idx] = self.v_now[env_idx]` alongside `self.finished[env_idx] = self.playing[env_idx]` at lines 174–177 and read `self.finished_v[env_idx]` in `complete()`.

### T-7. [Critical] `summarize_completed_episodes` crashes with `KeyError` on restart-only iterations
- **Location:** `ai/training/generic_trainer.py:659-665, 3864, 3875`
- **Cause:** When an iteration completes ≥1 restarted game and 0 whole games (`len(restarted) >= 1, len(whole) == 0`), line 662 sets `stats = {}` (omitting `"episodes_completed"`, `"mean_turn"`, `"mean_ply"`).
- **Impact:** Immediate `KeyError: 'episodes_completed'` at line 3864 (and `KeyError: 'mean_turn'` at line 3875 on `it % 10 == 0`).
- **Fix direction:** Always populate the empty-group defaults via `_episode_group_stats([], "")`.

### T-8. [High] Turn-2 restarts permanently leak `self.out[2]` and corrupt whole-game stats
- **Locations:** `ai/training/restart_pool.py:70, 192, 268`, `bindings/ts_env.py:447`, `ai/training/nash_pg.py:2199, 2221`
- **Cause:** `parse_turn_weights` permits `t == 2`, whose pre-deal state has `state.turn == 1`. `ts_env.py:447` sets `env_start_turn = 1 if int(state.turn) <= 1 else int(state.turn) + 1`, marking Turn-2 restarts as `start_turn == 1`.
- **Impact:** `nash_pg.py:2199` treats Turn-2 restarts as whole games, and line 2221 (`if start_turn > 1:`) never calls `restart_pool.complete()`, so `self.out[2]` increments forever until bucket 2 wedges full.

### T-9. [Medium] `--forced-opening` silently ignores `--restart-pool-frac`; `StartPositionPool.harvest()` ignores `obs_features`/action views
- **Locations:** `ai/training/generic_trainer.py:2734-2756`, `ai/training/start_pool.py:191-218`
- **Impact:**
  - `if forced_opening:` rejects `start_pool_frac > 0` but silently skips `elif restart_pool_frac > 0` while recording `restart_pool_frac` in `metadata.json`.
  - `StartPositionPool.harvest()` creates a bare `TsVectorizedEnv` without calling `set_obs_features` or `set_action_views` (crashing on `ops_budget` models, silently misreading E8/E8.1/merged/macro models), and draws from global torch RNG without `@preserves_training_rng`.

---

## 3. Rewards & Game-Ending Classification

### T-10. [High] Final Scoring clamped to ±20 VP is misclassified as `"20 VP"` (breaking `RegionTargetedReward` and `EndingTargetedReward`)
- **Locations:** `bindings/state_json.cpp:1179-1185`, `ai/rewards/region_reward.py:48-49, 148`, `ai/rewards/reward_calculator.py:63-70, 105`, `engine/src/scoring.cpp:377-380`
- **Cause:** `ending_reason()` checks `if (abs_vp >= 20) return "20 VP";` *before* `if (state.turn >= 10) return "final scoring";`. Because Final Scoring (`state.turn == 11`) clamps `victory_points` to `[-20, 20]`, any ±20 VP Final Scoring game returns `"20 VP"`.
- **Impact:**
  - `RegionTargetedReward.reached_final_scoring()` returns `False` and **drops `final_share_us()`** on the largest Final Scoring blowouts.
  - `EndingTargetedReward(targets=["final_scoring"])` pays `0.0` instead of `+1.0` on ±20 VP Final Scoring wins.
  - `EndingTargetedReward(targets=["20vp"])` wrongly pays `+1.0` on clamped Final Scoring wins AND on opponent Held Scoring Card losses (`state_machine.cpp:763, 765` also sets `victory_points = ±20`).
- **Fix direction:** Check `state.turn > 10` (Final Scoring) before `abs_vp >= 20`, and distinguish Held Scoring losses explicitly.

### T-11. [High] `BlunderAwareRewardCalculator` with default `--no-blunder-window` gives `0.0` to both sides when USSR loses to a Held Scoring Card
- **Locations:** `ai/rewards/reward_calculator.py:224-230, 333`, `ai/training/generic_trainer.py:2696`
- **Cause:** `blunder_window` defaults to `False` while `held_scoring_blunder` defaults to `True`. When USSR holds a scoring card at turn end and US made the final AR6/AR7 decision (`p_act == +1`), `is_held_scoring_loss(st, US)` is `False`, so `rewards[i] = 0.0`. Without `blunder_window`, `RolloutBuffer` never applies its `-1.0` blunder override.
- **Impact:** USSR receives `0.0` penalty for losing the game by holding a scoring card (while US holding one gives `-1.0` to US and `+1.0` to USSR). Also `UsefulActionsReward.__init__` does not accept `held_scoring_blunder`, silently ignoring `--no-held-scoring-blunder` under `useful_actions` and `curriculum` Stage 1.

### T-12. [High] `VPPotentialShaping` (`--vp-potential`) breaks `RegionTargetedReward` and both-seat `EndingTargetedReward`
- **Locations:** `ai/training/generic_trainer.py:2722-2727, 2796, 3283`, `ai/training/nash_pg.py:2478`, `ai/rewards/region_reward.py:137`
- **Cause:** Wrapping `reward_calc = VPPotentialShaping(reward_calc, vp_potential)` makes every downstream `isinstance(reward_calc, RegionTargetedReward)` / `(RegionTargetedReward, EndingTargetedReward)` check return `False`.
- **Impact:** Skips `attach(env.runner)` (silently dropping all in-game scoring card VP with no error for locked-seat runs), skips `attach_sides()` and guard checks (crashing both-seat runs on step 1), and skips `metrics()` (leaking `self.finished` lists unboundedly).

### T-13. [Medium] In-game Europe Control win feeds artificial `±20 - prev_vp` delta into `RegionTargetedReward.scored_vp`
- **Locations:** `ai/rewards/region_reward.py:137-162`, `engine/src/scoring.cpp:227-241`, `engine/src/card_dispatcher.cpp:350-362`
- **Impact:** Europe Control teleports `victory_points` to `±20`, recording `vp_delta = ±20 - prev_vp` (up to ±39) in `last_event_tracks`. Adding that into `scored_vp` makes a Europe Control win from 10 VP *behind* (`vp_delta = +30`, hitting `cap=2.0`) pay `+1.80`, whereas winning Europe Control from 19 VP *ahead* (`vp_delta = +1`) pays only `control_win = +1.00`, inverting payoff vs board dominance and distorting `region/vp_per_game`.

---

## 4. Opponent Pool, Distillation, Optimizer & Resume Logic

### T-14. [Critical] `--no-decay-norms-biases` breaks `named_parameters()` order, scrambling Adam moments on resume/growth
- **Locations:** `ai/training/nash_pg.py:345-359, 370-383`, `ai/training/generic_trainer.py:1522-1561`, `ai/models/grow.py:80-97, 141-166, 204-234`
- **Cause:** When `decay_norms_biases=False` and `lr_mults=None`, `make_optimizer` puts all `ndim >= 2` tensors in group 0 and all `ndim < 2` tensors in group 1, destroying `named_parameters()` order (unlike the `lr_mults` branch at lines 351–359). Because PyTorch's `Optimizer.state_dict()` always assigns sequential IDs `0..N-1` in `param_groups` order, the guard `flat_saved != list(range(len(flat_saved)))` in `regroup_optimizer_state`, `_optimizer_state_with_added`, and `grow_macro_optimizer_state` is **vacuous**.
- **Impact:** Resuming across `--no-decay-norms-biases` (without `lr_mults`) and `--mup` / `--policy-head-lr-mult` / default decay / `ADDABLE_HEADS` / `--resume-widen` / `--resume-grow-tokens` / `--resume-grow-sphere` / `--macro-influence` maps saved Adam moments onto the wrong parameters (crashing on shape mismatch at `opt.step()`, or silently swapping moments between equal-shaped tensors).
- **Fix direction:** Build contiguous runs in `named_parameters()` order in the `decay_norms_biases=False, lr_mults=None` branch too.

### T-15. [High] `grow_tokens_optimizer_state` corrupts multi-group optimizer states; `--resume-widen` + `--resume-grow-tokens` misaligns indices
- **Locations:** `ai/models/grow.py:164-166`, `ai/training/generic_trainer.py:1669-1692`
- **Impact:**
  1. `grow_tokens_optimizer_state` sets `groups[0]["params"] = list(range(len(names)))` while leaving `groups[1:]` intact, creating duplicate param IDs and crashing on any multi-group (`--mup`, `--policy-head-lr-mult`) checkpoint.
  2. `generic_trainer.py:1669` builds `names` for `widen_optimizer_state` excluding `sphere.*` but not `tok_*`; because `tok_*` precedes `fusion_in` in `LadderNet`, combining `--resume-widen` and `--resume-grow-tokens` shifts `fusion_in`'s index and fails to widen its moments.

### T-16. [High] `online_distill.py` double-divides `v_vp` by 20 (400× too small VP value loss)
- **Locations:** `ai/training/online_distill.py:85`, `ai/models/coldwar_net_v2.py:481-487`
- **Cause:** `F.mse_loss(s_vp.float().view(-1) / 20.0, t_vp / 20.0)` divides `v_vp` by `20.0` even though `v_vp` is already normalized to `[-1, 1]`.
- **Impact:** Scales the VP value head's distillation loss by `1/400` (`0.0025`), leaving `val_vp_head` nearly untrained during online distillation.

### T-17. [High] `OpponentPool` accounting, seat-locking, `seat_balance`, and resume bugs
- **Locations:** `ai/training/opponent_pool.py:276, 322-324, 513-523, 541-557, 579-599, 713-758, 798, 838-861, 913`, `ai/training/nash_pg.py:2344-2364`, `ai/training/generic_trainer.py:3138-3201, 3282`
- **Impacts:**
  1. **Dropped fast-game exposure (`opponent_pool.py:721-758`):** Under `seat_balance=False`, exposure is seeded only in `start_iteration()`, not on episode reset. Any mixed game that starts and finishes within one 128-step rollout (e.g., Turn 1–2 DEFCON suicide) has `total == 0.0` and is never added to `games`/`wins`/`side_games`/`side_wins`, biasing PFSP and `opp_league_win_{us,ussr}`.
  2. **`seat_balance=True` misfilters `completed_episodes` (`nash_pg.py:2344-2364`):** Overwrites `self._selfplay_mask[_i]` for the *next* episode before filtering the *just-finished* episode, leaking mixed games into self-play stats and dropping self-play games.
  3. **League-only seat-lock bypass (`opponent_pool.py:513-557, 713-720`):** When `own == []` and a member is evicted mid-game or no member fits `-side`, `env_member = -1` falls back to `self.current` (picked ignoring `self.seats`), seating a locked exploiter in its untrained seat while `member == -1` skips `league_wrong_seat_games`.
  4. **Self-pool resume bugs (`generic_trainer.py:3138-3201`):**
     - Regex `(snapshot|pool)_(\d+)steps\.pt$` never matches `snapshot_0s.pt`, dropping the step-0 anchor on every resume.
     - Fallback `else:` (line 3198) runs `_seed_nets = [deepcopy(model)]` *before* `load_resume_state()` (line 3479), seeding the pool with random untrained weights if no snapshots are on disk.
     - Branching from `<run_dir>:<steps>` with `--reset-opponent-pool` never filters `n <= resume_step`, loading *future* snapshots from the parent run.
  5. **Minor pool bugs:** `per_game_own` writes own-pool IDs into `env_member`, corrupting `opp_league_frac_envs` (`913`) and polluting `league_seat_games` (`717`); `_remove()` leaks `side_wins`/`side_games`/`league_seat_games`; `state_dict()` omits `game_rng`/`league_rng`; `load_state_dict()` allocates fallback IDs before updating `self._next_id` (`838-845`, ID collision risk); `fixed_env_sides` silently overrides `lock_learner_side` (`276`); initial `_league_scan()` runs after `__init__` (`generic_trainer.py:3282`), giving 0 league games in the first batch.

### T-18. [Medium] Multi-GPU device, CLI, and resume-override bugs
- **Locations:** `ai/training/nash_pg.py:556`, `ai/training/graphed_forward.py:116-119`, `ai/training/train.py:1477, 1773-1776`, `ai/training/generic_trainer.py:1533-1540, 1647, 1790-1824, 3554`
- **Impacts:**
  1. `nash_pg.py:556` collapses `torch.device("cuda:1")` to `"cuda"` (`cuda:0`), causing cross-device crashes against `generic_trainer.py`'s `dev = cuda:1`; `GraphedForward.replay()` omits `with torch.cuda.device(...)`.
  2. `train.py:1477` only sets `CUBLASLT_WORKSPACE_SIZE=0` when `args.ladder_token_layers > 0`, missing a non-token learner facing a token-path league/fixed opponent under CUDA graphs.
  3. `train.py:1773-1776` falls through to `create_coldwar_net` (V1) for `--mode warmup --arch mlp`.
  4. `load_resume_state` only checks `seed != recorded_seed`, silently ignoring `--seed-sampling`/`--seed-env`/`--seed-pool` overrides when `--seed` is unchanged (and never reseeds Python's `random`).
  5. `--resume-add-rules` (`generic_trainer.py:1647`) skips state dicts lacking `"obs_features"` (base `0` checkpoints), failing `check_state_obs_features` at line 1656.
  6. Resuming `--ema-weights` while adding any `ADDABLE_HEADS` crashes in `WeightEMA` (`strict=True` at `generic_trainer.py:3554` / `schedule.py:89`).
  7. `_load_optimizer_state` (`generic_trainer.py:1533-1540`) does not strip stale `"lr_mult"` keys from saved groups when the new launch has none.
