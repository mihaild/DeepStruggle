# Model Architectures, Scaling & Precision (`ai/models/`, `ai/training/head_l2.py`)

Audit at commit `fb9b817` (2026-11-22).

---

## High-Severity Scaling & Correctness Bugs

### M-1. `--mup` is a complete no-op on `TokenTransformer` (`token_main` / `token_layers`), `entity` mode, and widened `per_entity_heads`
- **Locations:** `ai/models/mup.py:38-66`, `ai/models/token_transformer.py:67`, `ai/training/nash_pg.py:351-359`
- **Cause:**
  1. `base_shapes_for` only overrides `entity_proj_dim`, `hidden_dim`, and `num_res_blocks`. It leaves `token_dim`, `entity_dim`, `per_entity_heads`, `region_attn_dim`, and `card_lookup_dim` at the **wide model's own width**.
  2. `mup_lr_mults` excludes only `nn.Embedding.weight`, not bare 2D `nn.Parameter` embeddings (`tok_country_id`, `tok_card_id`, `cl_identity`) — so once `token_dim` is varied in `base_shapes_for`, those embeddings will be misclassified as linear weight matrices (`fan_in = shape[1]`).
  3. `TokenBlock.forward` (`token_transformer.py:67`) uses standard $1/\sqrt{d_h}$ attention scaling instead of µP's $1/d_h$ scaling, and neither `TokenBlock` nor `ResBlock` applies residual depth scaling ($1/\sqrt{2L}$ / $1/\sqrt{L}$).
- **Impact:** When widening a `TokenTransformer` (`token_dim` 128 → 256/512), every `tok_main.blocks.*`, `fusion_in.0.weight`, and `pe_*.0.weight` gets `lr_mult = 1.0` (zero µP scaling).
- **Also in `mup.py` & `nash_pg.py`:**
  - **Unintended `0.25×` LR penalty on `SphereBlock.w2`:** `base_shapes_for` pops `sphere_blocks`, so `mup_lr_mults` falls back to `base_f = base_width` (480). For `sphere.blocks.*.w2.weight` (`fan_in = 4 * hidden_dim`), even at `hidden_dim == 480`, `lr_mult = 480 / 1920 = 0.25`.
  - **Coupled AdamW weight decay (`nash_pg.py:351-359`):** PyTorch `AdamW` decays by `lr * weight_decay`. Scaling `lr = lr * mult` without dividing `weight_decay` by `mult` proportionally reduces effective weight decay on widened layers under `--mup` and on slowed policy heads under `--policy-head-lr-mult < 1`.

### M-2. Guaranteed forward-pass crashes on valid `LadderNet` constructor combinations
- **Locations:** `ai/models/ladder_net.py:263, 654-658, 671-675, 993, 997-999, 1094, 1100-1109`
- **Bugs:**
  1. **`token_main=True` + `identity_dim > 0` + (`branch_head` or `play_mode_head`):** `__init__` sizes `branch_head_net[0]` and `play_mode_head_net[0]` to `hidden_dim + 110 + token_dim`, whereas `_encode` sets `tokens[2]` to `cat([card_identity, tok_cards])` of width `identity_dim + token_dim` and `_policy_logits:999` appends all of `tokens[2]`, crashing with a matmul shape mismatch.
  2. **`input_mode="grouped"` + `per_entity_heads=0` + (`branch_head` or `play_mode_head`):** `__init__` allows this combination (only rejecting `input_mode == "flat"`), but `_encode` only builds `tokens` when `raw_tokens` (`per_entity_heads > 0`) is True, triggering `AssertionError` at `_policy_logits:993` (`assert tokens is not None`).

### M-3. Checkpoint auto-detection crashes & silent corruption (`ladder_config_from_state_dict`, `check_checkpoint_layout`)
- **Locations:** `ai/models/ladder_net.py:359-360, 506-515, 1228-1237, 1261-1262, 1281-1289, 1312`, `ai/models/coldwar_net_v2.py:1020-1044`
- **Bugs:**
  1. **`input_mode="entity"` + `per_entity_heads > 0` + `identity_dim > 0` crashes on reload:** `ladder_config_from_state_dict` (`1282-1289`) subtracts `entity_dim + ctx_w` from `pe_country.0.weight.shape[1]` and compares the remainder to `BOARD_FEATURES`, forgetting that `entity` mode adds `identity_dim` to `board_in` (`359`), so `head_static` is misdetected and `load_state_dict(strict=True)` crashes.
  2. **`OBS_FEATURE_OPS_BUDGET` (`+3` globals) breaks reload for `token_main` and `flat`:** `coldwar_net_v2.py:1038-1043` (`token_main`) checks `widths == (26, 14, 100)`, rejecting `(26, 14, 103)`. `coldwar_net_v2.py:1020-1028` (`flat`) rejects `3827`/`2463`, and `ladder_net.py:1230-1237` uses `static_input_mask()` (3824) instead of `3824 + _obs_extra(obs_features)`, misdetecting `drop_static=True` as `False`.
  3. **Silent architecture corruption for non-default `card_lookup_dim` or `num_attn_heads`:** `ladder_net.py:1261-1262` hardcodes `card_lookup_dim = 32 if hk % 32 == 0 else hk` and line 1312 hardcodes `num_attn_heads = 4`. Any checkpoint with e.g. `card_lookup_heads=4, card_lookup_dim=16` (`hk=64`) silently loads under `(heads=2, dim=32)` with wrong head splitting and $1/\sqrt{d_k}$ scaling.

---

## Medium-Severity Wiring, Compilation & Precision Bugs

### M-4. Incomplete head registries (`pe_macro`, `pe_card`, `region_attn`, `value_dist_head`, `head_norms`)
- **Locations:** `ai/models/ladder_net.py:701, 1051`, `ai/models/mup.py:25-30`, `ai/models/coldwar_net_v2.py:917-918`, `ai/training/head_l2.py:22`, `ai/training/generic_trainer.py:1598-1616`, `ai/models/grow.py:104`
- **Impacts:**
  - `pe_macro` (P42) is omitted from `--ladder-head-norm` (`ladder_net.py:701, 1051` hardcodes `norm=None`), `mup.POLICY_HEAD_PREFIXES`, `POLICY_HEAD_MODULES`, `HEAD_L2_MODULES`, `reset_policy_head`, and `grow.ENTITY_HEADS` (crashing `--resume-grow-tokens` on a macro network). Also `head_center=True` removes state-dependent mean offsets between `pe_country` and `pe_macro` even though both are legal simultaneously.
  - `pe_card` and `region_attn` are omitted from `POLICY_HEAD_MODULES`, `HEAD_L2_MODULES`, and `reset_policy_head` (`region_attn` also missing from `mup.POLICY_HEAD_PREFIXES`).
  - `reset_policy_head` skips `head_norms`, leaving stale trained LayerNorm affines attached to freshly re-initialized policy heads.
  - `value_dist_head` is omitted from `HEAD_L2_MODULES` and `mup.OUTPUT_LAYERS`.
  - `grow_tokens_state_dict` (`grow.py:125-138`) and `grow_macro_state_dict` (`188-201`) fail if combined with `--resume-grow-sphere` or any `ADDABLE_HEADS` in the same resume, and `widen_state_dict`/`widen_optimizer_state` omit `cl_query.weight` (`card_lookup=True`).

### M-5. `TokenTransformer` Dynamo cache exhaustion, unmasked empty slots, and `RMSNorm` dtype fragility
- **Locations:** `ai/models/token_transformer.py:24-35, 62-73, 85-105`, `ai/models/ladder_net.py:826-863`, `ai/training/generic_trainer.py:3911`
- **Impacts:**
  1. Module-level `_run_compiled = torch.compile(_run)` (`token_transformer.py:105`) guards on `id(net)`, batch size, and grad mode. With `deepcopy` own-pool snapshots (up to 12 distinct `id`s) plus variable rollout/eval/chunk/afterstate batch sizes, it exceeds `torch._dynamo.config.cache_size_limit` (default 8) and silently falls back to eager. (Restored pool snapshots from disk also lose `compiled=True`.)
  2. `_asymmetric_pass` (`ladder_net.py:859-862`) passes up to `n - #held` duplicate `tok_empty_slot` tokens into `F.scaled_dot_product_attention` with `attn_mask=None`, letting every active token attend to all duplicate empty tokens and shifting attention denominators as hand size changes across a turn.
  3. `RMSNorm` (`token_transformer.py:34`) passes `fp32` `self.gain` to `F.rms_norm` without casting to `x.dtype` (masked inside `LadderNet._token_pass` only because adding `fp32` `tok_*_id` promotes `c`/`k` back to `fp32` before entering `TokenTransformer`; calling `TokenTransformer` directly on `bf16`/`fp16` raises a `RuntimeError`).

### M-6. `SphereTrunk` unit-norm invariant broken by non-RL trainers & `WeightEMA`
- **Locations:** `ai/models/hyperspherical.py:38-49, 97-101`, `ai/training/schedule.py:91-104`, `ai/training/generic_trainer.py:716, 813, 908`, `ai/training/{behavioral_cloning,value_fit,policy_fit,q_fit}.py`
- **Impacts:**
  - `project_weights()` is hooked only in `nash_pg.make_optimizer`. `_HumanInjector`, `run_search_distillation`, `run_behavioral_cloning_warmup`, `behavioral_cloning.py`, `value_fit.py`, `policy_fit.py`, and `q_fit.py` create raw `AdamW` without projecting weights back to the sphere.
  - `WeightEMA.update` (`schedule.py:98-103`) does Euclidean `lerp_` on `w1`/`w2` without re-projecting, shrinking faster-moving rows below unit norm.
  - Regularization conflict: `w1`/`w2` (`ndim == 2`) get weight decay right before `project()` rescales them back to unit norm (effectively inflating step size by $1/(1 - \eta\lambda)$); `scaler` (`ndim == 1`) is positive-homogeneous of degree 0, so with `--no-decay-norms-biases` orthogonal Adam steps monotonically inflate $\|s\|_2$ and progressively damp angular updates; with default `decay_norms_biases=True`, `alpha` decays toward 0.

### M-7. `--update-bf16` / `--bf16-fp32-heads` / `mask_logits` edge cases
- **Locations:** `ai/models/coldwar_net_v2.py:21-45, 458-519, 730`, `ai/models/ladder_net.py:738-769, 984`, `ai/training/nash_pg.py:3003-3007, 3176, 3466`
- **Impacts:**
  - `collect_rollouts` never runs under `self._amp()`, so on non-token trunks (`A8`), rollouts run fp32/TF32 while updates run bf16 even with `--bf16-fp32-heads`, injecting baseline `log_ratio`/`target_kl` noise before any gradient step.
  - `forward_prior_gain`, `forward_aux`, `forward_card_aux`, `forward_opp_legal`, and `defcon_risk` do not call `_heads_in`. At `nash_pg.py:3176`, `forward_prior_gain` runs *outside* `self._amp()` (fp32 trunk) on the same minibatch where `cur_v_win` ran *inside* `self._amp()` (bf16 trunk); `nash_pg.py:3466` (`searched_games`) calls full `active_net` outside `_amp()`.
  - Without `--bf16-fp32-heads`, `_cap_logits:984` casts capped fp32 logits back to bf16 and `val_win_head`'s `Tanh` runs in bf16 (rounding `|x| >= 3.1` to exact `±1.0`).
  - `ColdWarNetV2.forward_value` (`499`) evaluates `value_dist_head` + `softmax` unnecessarily under `categorical_value=True`, and `forward_with_value_logits` (`514-518`) evaluates `value_dist_head` twice.
  - `pad_to_mask` / `mask_logits` (`coldwar_net_v2.py:21-45`) only pads narrower logits (`220`) up to wider masks (`304`), crashing if a macro model (`304`) is handed a 220-slot mask (hits `_HumanInjector`, `ai/eval/agreement.py:55`, `ai/eval/plasticity_probe.py:47`).
