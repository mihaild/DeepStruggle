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
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from typing import List, Optional, Tuple

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from ai.training.run_name import LEGACY_RUN_NAME_RE, is_run_name, parse_run_name  # noqa: E402
from ai.training.train import build_parser  # noqa: E402

#: metadata keys that describe the outcome rather than the request, so they are never flags.
_NOT_FLAGS = {
    "run_id", "run_name", "base_commit", "commit_message", "git_dirty", "git_commit",
    "git_message", "timestamp", "start_time", "description", "resumed_from", "obs_layout",
}


def _defaults() -> dict:
    parser = build_parser()
    return {a.dest: a.default for a in parser._actions if a.dest != "help"}


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
    "branch_head": "ladder_branch_head", "play_mode_head": "ladder_play_mode_head",
    "logit_cap": "ladder_logit_cap", "logit_cap_grad": "ladder_logit_cap_grad",
    "res_dropout": "ladder_res_dropout", "trunk_norm": "ladder_trunk_norm",
    "res_post_norm": "ladder_res_post_norm",
    "sphere_blocks": "ladder_sphere_blocks", "sphere_expansion": "ladder_sphere_expansion",
    "sphere_out_scale": "ladder_sphere_out_scale", "head_norm": "ladder_head_norm",
    "sphere_carry_norm": "ladder_sphere_carry_norm",
    "logit_cap_leak": "ladder_logit_cap_leak", "logit_cap_leak_up": "ladder_logit_cap_leak_up",
    "region_attn_dim": "ladder_region_attn_dim", "region_attn_heads": "ladder_region_attn_heads",
}
#: `ladder_config` keys that are not a `--ladder-*` flag, so they are not mapped: the network's
#: fixed attention width, markers derived from other flags (the aux coefficients, --obs-features,
#: each recorded at the top level), and copies of top-level flags of the same name. Any key in
#: neither this set nor `_LADDER` is a setting the diff cannot see, and fails the command.
_LADDER_NOT_FLAGS = frozenset({
    "num_attn_heads", "aux_heads", "card_aux", "opp_legal_aux", "obs_features",
    "per_entity_heads", "identity_dim", "drop_static", "categorical_value",
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
    "distill_epochs_per_rollout", "distill_holdout", "distill_eval_every", "distill_patience",
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


def recorded(run_dir: str) -> dict:
    """{dest: value} for every CLI setting the run's metadata records, under the CLI's own names."""
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
    return out


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
        flag = "--" + k.replace("_", "-")
        if isinstance(v, bool):
            parts.append(flag if v else f"--no-{k.replace('_', '-')}")
        elif isinstance(v, (list, tuple)):
            parts.append(f"{flag} " + " ".join(str(x) for x in v))
        else:
            parts.append(f"{flag} {v}")
    return parts


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
        print(f"# non-default flags used by {name}")
        for p in as_flags(a):
            print(f"  {p}")
        return 2 if _blind_spots([(name, args.run_dir)]) else 0

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
            print(f"{'--' + k.replace('_', '-'):28s}{str(va):>30s}{str(vb):>30s}")
    if _blind_spots([(na, args.run_dir), (nb, args.diff)]):
        return 2
    return 1 if rows else 0


if __name__ == "__main__":
    raise SystemExit(main())
