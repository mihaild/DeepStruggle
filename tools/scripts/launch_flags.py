#!/usr/bin/env python3
"""Reconstruct a run's non-default `tools/train.py` flags from its `metadata.json`.

Three arms of E4 were launched with flags the lineage does not use -- no opponent pool, the bare
architecture defaults, and a warm start -- because the command was copied from the template in
`CLAUDE.md` rather than derived from a run that worked. Each time the fix was to add the missing
flag to the template, and each time the *next* flag drifted instead. Prose cannot fix that: the
flags you forget are by definition the ones you are not thinking about.

So: do not copy a template. Print what a healthy run actually used, and diff.

    tools/scripts/launch_flags.py <run-dir>                 # what that run used
    tools/scripts/launch_flags.py <run-dir> --diff <other>  # what differs between two runs

Defaults come from `build_parser()` in `ai/training/train.py`, so a flag added to the CLI tomorrow
is handled without touching this file.

Two readings of "non-default", kept apart:

* `non_default` -- against each setting's default *in the run's own epoch* (`_EPOCH_DEFAULTS`,
  `_CODE_BASELINE`): what the A / R codes name (tools/scripts/run_codes.py) and what `--diff` compares.
* `launch_settings` / `command_flags` -- against *today's* `train.py` defaults, written with the
  parser's own option strings: the command that relaunches the run as recorded. It is parsed back with
  `build_parser()` (and the ladder's required axes checked) before it is printed; a setting today's CLI
  cannot express is an error, never a flag train.py would reject or a silently dropped one (2026-11-22:
  the printed command had `--no-ladder-card-self-attention`, which no parser defines, and lacked the
  required `--ladder-head-context` / `--ladder-head-static`, whose defaults were read off their
  `--no-` twins' store_false).
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import sys
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from ai.training.run_name import LEGACY_RUN_NAME_RE, is_run_name, parse_run_name  # noqa: E402
from ai.training.generic_trainer import RULE_FEATURES, VIEW_RULE_FEATURES  # noqa: E402
from ai.training.train import _ladder_config, build_parser  # noqa: E402

#: metadata keys that describe the outcome rather than the request, so they are never flags.
_NOT_FLAGS = {
    "run_id", "run_name", "base_commit", "commit_message", "git_dirty", "git_commit",
    "git_message", "timestamp", "start_time", "description", "resumed_from", "obs_layout",
}


#: Settings derived from another one's recorded value rather than recorded themselves.
_DERIVED = {"obs_rules": "obs_features", "obs_view": "obs_features"}


def _split_rules(flags: dict) -> dict:
    """Move the rule bits of `obs_features` out of it: `obs_rules`, the engine's minor version (E8.1), and
    `obs_view`, the view bits (canonical_first / canonical_last), which the V code names. Neither is an observation the A code
    names, so earlier runs keep their codes."""
    if "obs_features" in flags and flags["obs_features"] is not None:
        names = list(flags["obs_features"])
        flags["obs_features"] = [n for n in names if n not in RULE_FEATURES]
        flags["obs_rules"] = [n for n in names if n in RULE_FEATURES and n not in VIEW_RULE_FEATURES]
        flags["obs_view"] = [n for n in names if n in VIEW_RULE_FEATURES]
    return flags


#: Settings whose CLI default changed without an engine epoch, compared against the value they had before the change for
#: every run: a code is a set of flags, so a run that takes today's default still names it, and every run launched before
#: keeps its code. Pure win/loss became the default on 2026-11-06 (owner): a run that takes it is R119, not R102.
#: Both were removed on 2026-11-22 (blunder_aware is no longer a --reward-scheme choice; --no-blunder-window and
#: --no-held-scoring-blunder are accepted with no effect, and the trainer records them at today's values), so the
#: baseline stays what every earlier run is read against.
#: --side-independent-update (owner, 2026-11-26) has no CLI default of its own (None): the trainer takes it on for a new
#: --ladder-side-nets run and off for any other, records it only when on, and every run before it ran the pooled update
#: -- so a run's code names it only when it was on, and a relaunch names it only where the run differs from what an
#: unnamed launch would do (`launch_settings`).
#: uncapped_deficit, honest_context and relative_effects joined the default --obs-features on 2026-11-26 (owner) in E9,
#: with no new epoch: every run before read the base slots, so the observation half of the default (rule bits apart) is
#: read as E8's headline_reveal and a run reading any of the three names it in its A code (_EPOCH_DEFAULTS still gives
#: [] before E8).
_CODE_BASELINE = {"reward_scheme": "blunder_aware", "no_blunder_window": False, "side_independent_update": False,
                  "obs_features": ["headline_reveal"]}


def _defaults() -> dict:
    parser = build_parser()
    out = _split_rules({a.dest: a.default for a in parser._actions if a.dest != "help"})
    out.update(_CODE_BASELINE)
    return out


def _metadata(run_dir: str) -> dict:
    path = run_dir if run_dir.endswith(".json") else os.path.join(run_dir, "metadata.json")
    with open(path, encoding="utf-8") as f:
        return json.load(f)


#: metadata keys recorded under a different name than the CLI flag's dest, and how to convert.
_RENAMED = {
    "ent_coef": ("entropy_coef", lambda v: v),
    "cuda_graphs": ("no_cuda_graphs", lambda v: not v),
    "aux_own_coef": ("aux_ownership", lambda v: v),
    "aux_vp_coef": ("aux_vp_margin", lambda v: v),
    "aux_card_coef": ("aux_card_events", lambda v: v),
    "token_chunk": ("ladder_token_chunk", lambda v: v),
    "token_compile": ("ladder_token_compile", lambda v: v),
}
#: keys of the `ladder_config` dict and the `--ladder-*` flag each one came from.
_LADDER = {
    "input_mode": "ladder_input_mode", "aggregation": "ladder_aggregation",
    "entity_dim": "ladder_entity_dim", "entity_proj_dim": "ladder_entity_proj_dim",
    "card_self_attention": "ladder_card_self_attention", "cross_attention": "ladder_cross_attention",
    "head_context": "ladder_head_context", "head_static": "ladder_head_static",
    "head_entities": "ladder_head_entities", "hidden_dim": "ladder_hidden_dim",
    "head_center": "ladder_head_center",
    "num_res_blocks": "ladder_res_blocks", "card_lookup": "ladder_card_lookup",
    "card_lookup_heads": "ladder_card_lookup_heads", "card_lookup_dim": "ladder_card_lookup_dim",
    "card_lookup_identity_dim": "ladder_card_lookup_identity_dim",
    "token_layers": "ladder_token_layers", "token_dim": "ladder_token_dim",
    "token_main": "ladder_token_main", "token_heads": "ladder_token_heads", "token_mlp_mult": "ladder_token_mlp_mult",
    "token_card_memory": "ladder_token_card_memory",
    "branch_head": "ladder_branch_head", "play_mode_head": "ladder_play_mode_head",
    "logit_cap": "ladder_logit_cap", "logit_cap_grad": "ladder_logit_cap_grad",
    "res_dropout": "ladder_res_dropout", "trunk_norm": "ladder_trunk_norm",
    "res_post_norm": "ladder_res_post_norm",
    "sphere_blocks": "ladder_sphere_blocks", "sphere_expansion": "ladder_sphere_expansion",
    "sphere_out_scale": "ladder_sphere_out_scale", "head_norm": "ladder_head_norm",
    "sphere_carry_norm": "ladder_sphere_carry_norm",
    "logit_cap_leak": "ladder_logit_cap_leak", "logit_cap_leak_up": "ladder_logit_cap_leak_up",
    "region_attn_dim": "ladder_region_attn_dim", "region_attn_heads": "ladder_region_attn_heads",
    "side_heads": "ladder_side_heads", "side_nets": "ladder_side_nets",
}
#: `ladder_config` keys that are not a `--ladder-*` flag, so they are not mapped: the network's
#: fixed attention width, markers derived from other flags (the aux coefficients, --obs-features,
#: each recorded at the top level), and copies of top-level flags of the same name. Any key in
#: neither this set nor `_LADDER` is a setting the diff cannot see, and fails the command.
_LADDER_NOT_FLAGS = frozenset({
    "num_attn_heads", "aux_heads", "card_aux", "opp_legal_aux", "obs_features", "prior_gain",
    "per_entity_heads", "identity_dim", "drop_static", "categorical_value", "macro_influence",
})


#: CLI dests that are not training settings of an RL run, so their absence from metadata is not a
#: blind spot: where the run's files go, the device, logging, the post-run tournament, the other
#: modes' inputs (warmup / distill), naming, and curriculum_switch_fraction, which is recorded as
#: the curriculum_switch_steps it resolves to.
UNCHECKED_BY_DESIGN = frozenset({
    "resume", "output_dir", "device", "tensorboard", "mode", "post_tournament",
    "post_tournament_models", "post_tournament_games", "bc_epochs", "distill_dataset",
    "distill_epochs", "distill_lr", "warmup_dataset", "run_name", "description",
    "curriculum_switch_fraction", "new_directory",
    # --mode value-fit's inputs: ai/training/value_fit.py records them in its own metadata.json
    "value_dataset", "value_reinit", "value_arch", "value_epochs", "value_lr", "value_batch",
    "value_sibling_groups", "value_sibling_coef", "value_vp_coef", "value_aux_coef", "value_eval_every",
    "value_patience", "value_siblings", "value_sibling_loss", "value_sibling_decisions", "value_leaf_coef",
    "value_leaf_weights",
    # --mode policy-fit's inputs: ai/training/policy_fit.py records them in its own metadata.json
    "policy_dataset", "policy_worlds", "policy_reinit", "policy_arch", "policy_epochs", "policy_lr",
    "policy_batch", "policy_eval_every", "policy_patience", "policy_train_fraction", "policy_head_only",
    "policy_kl_anchor", "policy_target", "policy_reinit_heads",
    # --mode online-distill's inputs (P39 arm 2): ai/training/online_distill.py records them in its own metadata.json
    "distill_teacher", "distill_positions", "distill_play_temperature", "distill_value_coef", "distill_rollout_len",
    "distill_epochs_per_rollout", "distill_holdout", "distill_eval_every", "distill_patience", "distill_lr_final_frac",
    # --mode q-fit's inputs: ai/training/q_fit.py records them in its own metadata.json
    "q_dataset", "q_target", "q_head", "q_train_trunk", "q_weighting", "q_se_floor", "q_baseline", "q_judge",
    "q_epochs", "q_lr", "q_batch", "q_eval_every", "q_patience", "q_train_fraction",
})


#: Settings added after runs that lack them, and the value every such run had: all training before
#: `tf32` was recorded ran in fp32, and before `pool_every_steps` existed the pool grew at each
#: snapshot. Filled in rather than reported as unrecorded, so a diff against an older run shows
#: these as the real differences they are.
_BEFORE_RECORDED = {
    "tf32": lambda meta: False,
    "pool_every_steps": lambda meta: meta.get("snapshot_every_steps"),
    # before --block-lambda existed every run used the ordinary lambda everywhere
    "block_lambda": lambda meta: "setup" if meta.get("setup_block_lambda") else "off",
    # P24: no league before it existed
    "league_pool_size": lambda meta: 4,
    "adv_norm_learner_only": lambda meta: False,
    # recorded only when on (2026-11-26): a run without it updated its two side networks together
    "side_independent_update": lambda meta: False,
    # no setup entropy floor before it existed (2026-09-27)
    "setup_entropy_floor": lambda meta: 0.0,
    "setup_entropy_lr": lambda meta: 0.01,
    "setup_entropy_max_coef": lambda meta: 1.0,
    "setup_mc_credit": lambda meta: False,
    "setup_mc_coef": lambda meta: 1.0,
    "setup_mc_min_batch": lambda meta: 512,
    # P28 step 2 (2026-09-29): a constant rate and no weight average before these existed.
    "lr_schedule": lambda meta: "constant",
    "lr_schedule_every": lambda meta: 60_000_000,
    "lr_schedule_values": lambda meta: [1e-4, 3e-5],
    "lr_schedule_span": lambda meta: 200_000_000,
    "lr_min": lambda meta: 3e-5,
    # P39 arm 1 (2026-10-26): no learning-rate warmup before it existed.
    "lr_warmup_steps": lambda meta: 0,
    # P39 arm 3 (2026-10-27): PPO's clip and epochs were fixed at 0.2 and 4 before they were flags.
    "clip_eps": lambda meta: 0.2,
    "ppo_epochs": lambda meta: 4,
    # P39 arms 4 and 5 (2026-10-28): one AdamW group at (0.9, 0.999) / 1e-8, no muP, before they were flags.
    "adam_beta2": lambda meta: 0.999,
    "adam_eps": lambda meta: 1e-8,
    "policy_head_lr_mult": lambda meta: 1.0,
    "mup": lambda meta: False,
    # P39 arm 6 (2026-10-28): no sphere grown in place before it existed.
    "resume_grow_sphere": lambda meta: False,
    "mup_base_proj_dim": lambda meta: 256,
    "mup_base_hidden_dim": lambda meta: 480,
    "ema_weights": lambda meta: 0.0,
    # P29 bet 2 (2026-09-30): no aux targets before they existed.
    "aux_ownership": lambda meta: 0.0,
    "aux_vp_margin": lambda meta: 0.0,
    "aux_sample_frac": lambda meta: 0.1,
    "aux_min_batch": lambda meta: 4096,
    # The view spec (2026-10-01): no appended observation blocks before they existed.
    "obs_features": lambda meta: [],
    # P30 (2026-09-30): no card-event target before it existed.
    "aux_card_events": lambda meta: 0.0,
    "aux_card_sample_frac": lambda meta: 0.0005,
    "aux_card_min_batch": lambda meta: 2048,
    "aux_card_buffer": lambda meta: 65536,
    "aux_card_steps": lambda meta: 4,
    "aux_card_batch": lambda meta: 512,
    # P36 (2026-10-20): no gated active search before it existed.
    "active_search_frac": lambda meta: 1.0 / 64.0,
    "active_search_ce_coef": lambda meta: 0.1,
    "active_search_gate_every_steps": lambda meta: 40_000_000,
    "active_search_gate_games": lambda meta: 200,
    "active_search_gate_threshold": lambda meta: 0.55,
    "active_search_gate_only": lambda meta: False,
    "active_search_refresh_steps": lambda meta: 10_000_000,
    "active_search_refresh_steps_late": lambda meta: 50_000_000,
    "active_search_late_from": lambda meta: 400_000_000,
    # P41 J1 (2026-10-30): no zero-avoiding term before it existed.
    "zero_avoid_coef": lambda meta: 0.0,
    # P40 Arm S (2026-10-29): no searched opponent before it existed.
    "searched_opponent_share": lambda meta: 0.125,
    "searched_opponent_scope": lambda meta: "entry",
    "searched_opponent_refresh_steps": lambda meta: 50_000_000,
    "searched_opponent_batch": lambda meta: 32,
    "searched_opponent_max_wait": lambda meta: 8,
    # no forced opening before it existed (2026-09-28); None cannot be filled in, so it is recorded
    # below as "unrecorded" for older runs and diffs as the default.
}


#: Defaults that changed at an engine epoch: {dest: (the first epoch with today's default, the default
#: every earlier run was launched under)}. A run is compared against its own epoch's default, so an
#: old run that took the default then is not read as departing from the default now -- and its A and
#: R codes (tools/scripts/run_codes.py, which reads non_default) stay what they were.
_EPOCH_DEFAULTS = {
    # E8 (owner, 2026-10-20): a new network reads Space Race box 4's revealed headline
    # (HEADLINE_REVEAL) unless --obs-features says otherwise; before it, no optional feature.
    "obs_features": (8, []),
}
_RUN_DIR_SUFFIX = re.compile(r"_\d{8}_\d{6}(_VOID)?$")


def run_engine(run_dir: str) -> Optional[int]:
    """The engine revision a run was launched on, from the name it was launched under (the last E of
    a branch name, which an engine change names), or None for a run with no engine in its name."""
    meta = _metadata(run_dir)
    d = os.path.dirname(run_dir) if run_dir.endswith(".json") else run_dir
    name = str(meta.get("run_name") or _RUN_DIR_SUFFIX.sub("", os.path.basename(d.rstrip("/"))))
    if is_run_name(name) and not LEGACY_RUN_NAME_RE.match(name):
        m = re.match(r"^(\d+)", parse_run_name(name).fields()["E"])
    else:
        m = re.match(r"^E(\d+)", name)
    return int(m.group(1)) if m else None


def defaults_for(run_dir: str) -> dict:
    """The CLI defaults as they stood for this run's engine epoch (`_EPOCH_DEFAULTS`). A run with no
    engine in its name predates the epochs that changed a default."""
    dflt, engine = _defaults(), run_engine(run_dir)
    for dest, (since, before) in _EPOCH_DEFAULTS.items():
        if engine is None or engine < since:
            dflt[dest] = before
    return dflt


def recorded(run_dir: str, split: bool = True) -> dict:
    """{dest: value} for every CLI setting the run's metadata records, under the CLI's own names. `split`: the rule and
    view bits of `obs_features` apart (`obs_rules`, `obs_view`, as the codes read them); False keeps the one list the
    run was launched with."""
    meta, dflt = _metadata(run_dir), _defaults()
    out = {}
    for k, known in _BEFORE_RECORDED.items():
        if k not in meta and known(meta) is not None:
            out[k] = known(meta)
    for k, v in meta.items():
        if k in _NOT_FLAGS:
            continue
        if k in _RENAMED:
            dest, conv = _RENAMED[k]
            out[dest] = conv(v)
        elif k == "ladder_config" and isinstance(v, dict):
            for ck, cv in v.items():
                if ck in _LADDER:
                    out[_LADDER[ck]] = cv
            # added 2026-09-25; every ladder config before it had uncentred heads
            out.setdefault("ladder_head_center", bool(v.get("head_center", False)))
            # P30 C1 (2026-10-01): recorded only when on; off, the width is inert at its default
            out.setdefault("ladder_token_layers", 0)
            out.setdefault("ladder_token_dim", 128)
            # the branch head (2026-10-06): recorded only when on
            out.setdefault("ladder_branch_head", False)
            out.setdefault("ladder_play_mode_head", False)
            # the logit cap (2026-10-09): recorded only when on
            out.setdefault("ladder_logit_cap", 0.0)
            out.setdefault("ladder_logit_cap_grad", "tanh")
            out.setdefault("ladder_logit_cap_leak", 0.0)
            out.setdefault("ladder_logit_cap_leak_up", 0.0)
            # Cell A's region attention (2026-10-16): recorded only when on
            out.setdefault("ladder_region_attn_dim", 0)
            out.setdefault("ladder_region_attn_heads", 4)
        elif k in dflt:
            out[k] = v
    # Before 2026-09-27 a run that took the default rollout bands recorded None, and the default
    # was the sharpening bands; since then the bands actually used are always recorded.
    if "rollout_temps" in meta and meta["rollout_temps"] is None:
        out["rollout_temps"] = [0.15, 0.50, 0.10, 0.35]
    return _split_rules(out) if split else out


def unmapped_ladder_keys(run_dir: str) -> List[str]:
    """`ladder_config` keys this file does not know, so `recorded` drops them unseen."""
    lc = _metadata(run_dir).get("ladder_config")
    if not isinstance(lc, dict):
        return []
    return sorted(k for k in lc if k not in _LADDER and k not in _LADDER_NOT_FLAGS)


def unrecorded(run_dir: str) -> list:
    """CLI dests this run's metadata says nothing about, so a diff cannot check them."""
    rec, dflt = recorded(run_dir), _defaults()
    return sorted(d for d in dflt if d not in rec)


def non_default(run_dir: str) -> dict:
    """{dest: value} for every recorded setting that differs from the CLI default of its epoch."""
    dflt = defaults_for(run_dir)
    return {k: v for k, v in recorded(run_dir).items() if k in dflt and v != dflt[k]}


def epoch_default_rows(a_dir: str, b_dir: str) -> list:
    """(dest, a's value, b's value) for each epoch-dependent setting the two runs differ in, by the
    value each actually used: an E7 run and an E8 run that each took the default read different
    observations, which a diff of non-default flags would hide, and an E8 run that named no feature
    reads what an E7 run at its default did, which it would show as a difference."""
    rows = []
    for dest in _EPOCH_DEFAULTS:
        va = recorded(a_dir).get(dest, defaults_for(a_dir)[dest])
        vb = recorded(b_dir).get(dest, defaults_for(b_dir)[dest])
        if va != vb:
            rows.append((dest, va, vb))
    return rows


def as_flags(settings: dict) -> list:
    parts = []
    for k in sorted(settings):
        v = settings[k]
        if k in _DERIVED and isinstance(v, (list, tuple)):
            # Recorded apart from the flag they come from (the rule and view bits of --obs-features).
            parts.append(f"--{_DERIVED[k].replace('_', '-')} ... " + " ".join(str(x) for x in v))
            continue
        flag = "--" + k.replace("_", "-")
        if isinstance(v, bool):
            parts.append(flag if v else f"--no-{k.replace('_', '-')}")
        elif isinstance(v, (list, tuple)):
            parts.append(f"{flag} " + " ".join(str(x) for x in v))
        else:
            parts.append(f"{flag} {v}")
    return parts


def parser_defaults() -> Dict[str, Any]:
    """Today's `train.py` defaults as argparse applies them (`parse_args([])`): the first action of a dest sets its
    default, so `--ladder-head-context`'s None is not overwritten by its `--no-` twin's store_false True."""
    return vars(build_parser().parse_args([]))


def _actions(parser: argparse.ArgumentParser) -> Dict[str, List[argparse.Action]]:
    by: Dict[str, List[argparse.Action]] = defaultdict(list)
    for a in parser._actions:
        if a.option_strings and a.dest != "help":
            by[a.dest].append(a)
    return by


def _stores(a: argparse.Action, v: Any) -> Optional[str]:
    """The option string of flag action `a` that sets its dest to the constant `v`, or None."""
    if isinstance(a, argparse.BooleanOptionalAction):
        pos = [s for s in a.option_strings if not s.startswith("--no-")]
        neg = [s for s in a.option_strings if s.startswith("--no-")]
        return (pos or [None])[0] if v is True else (neg or [None])[0] if v is False else None
    if a.nargs == 0 and getattr(a, "const", None) is not None and a.const == v and type(a.const) is type(v):
        return a.option_strings[0]   # store_true / store_false / store_const
    return None


def _equivalent(v: Any, default: Any, acts: List[argparse.Action]) -> bool:
    """`v` (recorded) is what today's default gives: equal; an empty list where the default is None (the trainer
    records `list(x or [])`); or False for a store_true flag defaulting to None with no way to say False, which the
    trainer reads as bool(None)."""
    if v == default:
        return True
    if default is None and isinstance(v, (list, tuple)) and len(v) == 0:
        return True
    return default is None and v is False and not any(_stores(a, False) for a in acts)


#: Flags retired with no effect (2026-11-22) but still accepted for older commands: whatever a run recorded for them,
#: today's train.py does the same thing, so they are not part of a relaunch.
_RETIRED_NO_EFFECT = frozenset({"no_blunder_window", "no_held_scoring_blunder"})


def launch_settings(run_dir: str) -> Dict[str, Any]:
    """{dest: value} today's `train.py` must be given, beyond its own defaults, to relaunch the run as its metadata
    records it (settings outside an RL run's training, `UNCHECKED_BY_DESIGN`, and no-effect retired flags excluded)."""
    dflt, acts = parser_defaults(), _actions(build_parser())
    rec = recorded(run_dir, split=False)
    # What an unnamed --side-independent-update resolves to for a new launch of this run's shape (generic_trainer.
    # resolve_side_independent_update): on for --ladder-side-nets, so an A39 run that ran without it relaunches with
    # --no-side-independent-update.
    dflt["side_independent_update"] = bool(rec.get("ladder_side_nets", False))
    return {k: v for k, v in rec.items()
            if k in dflt and k not in UNCHECKED_BY_DESIGN | _RETIRED_NO_EFFECT
            and not _equivalent(v, dflt[k], acts.get(k, []))}


def command_flags(settings: Dict[str, Any]) -> Tuple[List[str], List[str]]:
    """`settings` as `train.py` flags, one string per flag, with the parser's own option strings, and every setting
    today's CLI cannot express (a retired value, a bool with no flag for it), which no flag is printed for."""
    acts = _actions(build_parser())
    parts: List[str] = []
    bad: List[str] = []
    for k in sorted(settings):
        v, cands = settings[k], acts.get(k, [])
        flag = next((s for a in cands for s in [_stores(a, v)] if s), None)
        if flag is not None:
            parts.append(flag)
            continue
        store = next((a for a in cands if a.nargs != 0), None)
        if store is None or v is None or isinstance(v, bool) and store.type is not None:
            bad.append(f"{k}={v!r}")
            continue
        opt = store.option_strings[0]
        if isinstance(v, (list, tuple)):
            if not v and store.nargs not in ("*", "?"):
                bad.append(f"{k}={v!r}")
                continue
            parts.append(" ".join([opt] + [shlex.quote(str(x)) for x in v]))
        elif store.choices is not None and v not in store.choices:
            bad.append(f"{k}={v!r} (not one of {sorted(map(str, store.choices))})")
        else:
            parts.append(f"{opt} {shlex.quote(str(v))}")
    return parts, bad


def check_command(flags: List[str], settings: Dict[str, Any]) -> List[str]:
    """Parse `flags` back with `build_parser()` and list what does not round-trip to `settings` (and the ladder's
    required structural axes, which train.py checks after parsing). Empty: the command relaunches the run."""
    argv = [tok for part in flags for tok in shlex.split(part)]
    try:
        ns = build_parser().parse_args(argv)
    except SystemExit:
        return [f"train.py refuses: {' '.join(argv)}"]
    acts, out = _actions(build_parser()), []
    for k, v in settings.items():
        got = getattr(ns, k)
        same = got == v or (isinstance(v, (list, tuple)) and list(got or []) == list(v))
        if not same and not _equivalent(v, got, acts.get(k, [])):
            out.append(f"{k}: recorded {v!r}, the command gives {got!r}")
    if ns.arch == "ladder":
        try:
            _ladder_config(ns)
        except SystemExit as e:
            out.append(str(e))
    return out


def _flag_name(dest: str) -> str:
    """How `--diff` names a setting: its train.py option (the positive one of a pair), or, for the rule and view bits
    recorded apart from `--obs-features`, that flag and which bits."""
    if dest in _DERIVED:
        return f"--{_DERIVED[dest].replace('_', '-')} ({ {'obs_rules': 'rule', 'obs_view': 'view'}.get(dest, dest)} bits)"
    opts = [s for a in _actions(build_parser()).get(dest, []) for s in a.option_strings]
    pos = [s for s in opts if not s.startswith("--no-")]
    return (pos or opts or ["--" + dest.replace("_", "-")])[0]


def _blind_spots(runs: List[Tuple[str, str]]) -> bool:
    """Print what the runs' metadata does not let a diff check; True if a ladder key is unmapped.

    A flag a run's metadata does not record cannot be diffed; say which, so "agree" is not read as
    covering them. Older runs predate some keys (lr, batch and buffer size, gamma: 2026-09-24). An
    unmapped `ladder_config` key is an architecture setting this file would drop, which is an error.
    """
    unmapped_any = False
    for name, run in runs:
        miss = [d for d in unrecorded(run) if d not in UNCHECKED_BY_DESIGN]
        if miss:
            print(f"# not recorded by {name}, so not checked: " + ", ".join(miss))
        unmapped = unmapped_ladder_keys(run)
        if unmapped:
            unmapped_any = True
            print(f"launch_flags: ERROR: ladder_config keys of {name} with no mapping here (add each to "
                  f"_LADDER or _LADDER_NOT_FLAGS): " + ", ".join(unmapped), file=sys.stderr)
    return unmapped_any


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_dir", help="a run directory, or its metadata.json")
    ap.add_argument("--diff", metavar="OTHER",
                    help="a second run; print only what differs between the two")
    args = ap.parse_args()

    a = non_default(args.run_dir)
    if not args.diff:
        name = os.path.basename(args.run_dir.rstrip("/"))
        settings = launch_settings(args.run_dir)
        flags, bad = command_flags(settings)
        problems = check_command(flags, {k: v for k, v in settings.items() if all(not b.startswith(k + "=") for b in bad)})
        print(f"# flags today's train.py needs, beyond its defaults, to relaunch {name}")
        for p in flags:
            print(f"  {p}")
        failed = False
        if bad:
            failed = True
            print(f"launch_flags: ERROR: {name}: today's train.py cannot express these recorded settings, so the "
                  f"command above does not relaunch the run: " + ", ".join(bad), file=sys.stderr)
        if problems:
            failed = True
            print(f"launch_flags: ERROR: {name}: the command does not round-trip: " + "; ".join(problems),
                  file=sys.stderr)
        return 2 if _blind_spots([(name, args.run_dir)]) or failed else 0

    b = non_default(args.diff)
    keys = sorted(set(a) | set(b))
    na = os.path.basename(args.run_dir.rstrip("/"))[:28]
    nb = os.path.basename(args.diff.rstrip("/"))[:28]
    rows = [(k, a.get(k, "<default>"), b.get(k, "<default>")) for k in keys
            if k not in _EPOCH_DEFAULTS and a.get(k, "<default>") != b.get(k, "<default>")]
    rows += epoch_default_rows(args.run_dir, args.diff)
    if not rows:
        print(f"# {na} and {nb} agree on every non-default flag they record")
    else:
        print(f"{'flag':28s}{na:>30s}{nb:>30s}")
        for k, va, vb in rows:
            print(f"{_flag_name(k):28s}{str(va):>30s}{str(vb):>30s}")
    if _blind_spots([(na, args.run_dir), (nb, args.diff)]):
        return 2
    return 1 if rows else 0


if __name__ == "__main__":
    raise SystemExit(main())
