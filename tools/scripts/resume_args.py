#!/usr/bin/env python3
"""Reconstruct a run's `tools/train.py` flags from the metadata.json it wrote.

Resuming a run means repeating its whole argv. `--resume <dir>` restores the weights, the
optimizer and the step count, but every architecture and hyperparameter flag has to be supplied
again, and typing them by hand splits into two failure modes that are nothing alike:

* an **architecture** mismatch fails loudly. Resuming E3-33-30 without its `--identity-dim 16
  --graph-layers 0 --per-entity-heads 64 --self-transform` raised on `load_state_dict` with
  missing gconv keys and a card_fc shape of 30 against 14. Cheap: you lose a launch.
* a **hyperparameter** mismatch does not fail at all. Dropping `--rollout-temps 0.8 1.2 0.7 1.1`
  would have resumed the temperature arm at the default sharpening bands, and the run would have
  continued, logged, snapshotted and been compared against its matched control as though it were
  still the same experiment. That is the silent-wrong-measurement class this project keeps
  paying for, and it is the reason this script exists.

So derive the flags from the record the run itself wrote rather than from memory.

**Unknown keys are reported, not ignored.** A metadata key this script cannot map is printed to
stderr as UNMAPPED. When a new flag is added to train.py and not added here, that warning is the
only thing standing between you and a resume that quietly drops it -- which is the same failure
this script was written to prevent, one level up.

    tools/scripts/resume_args.py <run-dir>              # flags, one per line
    tools/scripts/resume_args.py <run-dir> --oneline    # a single shell-ready line

Compose it into a launch:

    ARGS=$(tools/scripts/resume_args.py "$RUN" --oneline) || exit 1
    tools/train.py --resume "$RUN" $ARGS
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import sys
from typing import Any, Dict, List

# metadata key -> flag, for plain value flags.
VALUE_FLAGS: Dict[str, str] = {
    "arch": "--arch",
    "seed": "--seed",
    "num_envs": "--num-envs",
    "train_steps": "--train-steps",
    "reward_scheme": "--reward-scheme",
    "ent_coef": "--entropy-coef",
    "eta": "--eta",
    "vf_coef": "--vf-coef",
    "value_dist_coef": "--value-dist-coef",
    "ref_update_freq": "--ref-update-freq",
    "snapshot_every_steps": "--snapshot-every-steps",
    "resume_every_steps": "--resume-every-steps",
    "opponent_frac": "--opponent-frac",
    "opponent_pool_size": "--opponent-pool-size",
    "opponent_pfsp_weighting": "--opponent-pfsp-weighting",
    "opponent_pfsp_uniform_mix": "--opponent-pfsp-uniform-mix",
    "opponent_lock_side": "--opponent-lock-side",
    "curriculum_switch_steps": "--curriculum-switch-steps",
    "adv_filter_quantile": "--adv-filter-quantile",
    "decisiveness_turns": "--decisiveness-turns",
    "setup_explore_frac": "--setup-explore-frac",
    "search_ce_coef": "--search-ce-coef",
    "search_sims": "--search-sims",
    "search_subsample": "--search-subsample",
    "search_node_filter": "--search-node-filter",
    "identity_dim": "--identity-dim",
    "attn_readout": "--attn-readout",
    "per_entity_heads": "--per-entity-heads",
    "graph_layers": "--graph-layers",
    "warmup_checkpoint": "--warmup-checkpoint",
    "run_name": "--run-name",
}

# metadata key -> flag, for store_true flags: emitted only when the value is true.
BOOL_FLAGS: Dict[str, str] = {
    "opponent_self_pool": "--opponent-self-pool",
    "opponent_pfsp": "--opponent-pfsp",
    "self_transform": "--self-transform",
    "drop_static": "--drop-static",
    "categorical_value": "--categorical-value",
    "per_player_gae": "--per-player-gae",
    "same_perspective_bootstrap": "--same-perspective-bootstrap",
    "window_provoked_defcon": "--window-provoked-defcon",
    "resume_every_snapshot": "--resume-every-snapshot",
    "reset_opponent_pool": "--reset-opponent-pool",
}

# metadata key -> flag, for list-valued flags.
LIST_FLAGS: Dict[str, str] = {
    "rollout_temps": "--rollout-temps",
    "eval_opponents": "--eval-opponents",
    "opponent_checkpoints": "--opponent-checkpoints",
}

# Recorded for provenance, not settings to replay.
IGNORED = {
    "run_id", "base_commit", "commit_message", "git_dirty", "description",
    "obs_layout", "resumed_from", "training_mode",
}


def build(meta: Dict[str, Any]) -> tuple[List[str], List[str]]:
    args: List[str] = []
    unmapped: List[str] = []

    for key in sorted(meta):
        if key in IGNORED:
            continue
        val = meta[key]
        if key in VALUE_FLAGS:
            # None means "not set"; the flag's own default applies.
            if val is not None:
                args += [VALUE_FLAGS[key], str(val)]
        elif key in BOOL_FLAGS:
            if val:
                args.append(BOOL_FLAGS[key])
        elif key in LIST_FLAGS:
            if val:
                args.append(LIST_FLAGS[key])
                args += [str(v) for v in val]
        else:
            unmapped.append(key)
    return args, unmapped


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter,
                                 allow_abbrev=False)
    ap.add_argument("run_dir")
    ap.add_argument("--oneline", action="store_true",
                    help="emit one shell-quoted line instead of one flag per line")
    a = ap.parse_args()

    path = os.path.join(a.run_dir, "metadata.json")
    if not os.path.exists(path):
        print("no metadata.json in %s" % a.run_dir, file=sys.stderr)
        return 1
    with open(path, encoding="utf-8") as fh:
        meta = json.load(fh)

    args, unmapped = build(meta)
    if unmapped:
        # stderr, so it is visible in a terminal but does not corrupt $(...) capture.
        print("UNMAPPED metadata keys (add them to resume_args.py before trusting this "
              "resume): %s" % ", ".join(unmapped), file=sys.stderr)

    if a.oneline:
        print(" ".join(shlex.quote(x) for x in args))
    else:
        for x in args:
            print(x)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
