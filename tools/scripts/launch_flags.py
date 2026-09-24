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
import sys

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

#: metadata keys that describe the outcome rather than the request, so they are never flags.
_NOT_FLAGS = {
    "run_id", "run_name", "base_commit", "commit_message", "git_dirty", "git_commit",
    "git_message", "timestamp", "start_time", "description", "resumed_from", "obs_layout",
}


def _defaults() -> dict:
    from ai.training.train import build_parser

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
}
#: keys of the `ladder_config` dict and the `--ladder-*` flag each one came from.
_LADDER = {
    "input_mode": "ladder_input_mode", "aggregation": "ladder_aggregation",
    "entity_dim": "ladder_entity_dim", "entity_proj_dim": "ladder_entity_proj_dim",
    "card_self_attention": "ladder_card_self_attention", "cross_attention": "ladder_cross_attention",
    "head_context": "ladder_head_context", "head_static": "ladder_head_static",
    "head_entities": "ladder_head_entities", "hidden_dim": "ladder_hidden_dim",
    "num_res_blocks": "ladder_res_blocks", "card_lookup": "ladder_card_lookup",
    "card_lookup_heads": "ladder_card_lookup_heads", "card_lookup_dim": "ladder_card_lookup_dim",
    "card_lookup_identity_dim": "ladder_card_lookup_identity_dim",
}


#: CLI dests that are not training settings of an RL run, so their absence from metadata is not a
#: blind spot: where the run's files go, the device, logging, the post-run tournament, the other
#: modes' inputs (warmup / distill), naming, and curriculum_switch_fraction, which is recorded as
#: the curriculum_switch_steps it resolves to.
UNCHECKED_BY_DESIGN = frozenset({
    "resume", "output_dir", "device", "tensorboard", "mode", "post_tournament",
    "post_tournament_models", "post_tournament_games", "bc_epochs", "distill_dataset",
    "distill_epochs", "distill_lr", "warmup_dataset", "run_name", "description",
    "curriculum_switch_fraction",
})


#: Settings added after runs that lack them, and the value every such run had: all training before
#: `tf32` was recorded ran in fp32, and before `pool_every_steps` existed the pool grew at each
#: snapshot. Filled in rather than reported as unrecorded, so a diff against an older run shows
#: these as the real differences they are.
_BEFORE_RECORDED = {
    "tf32": lambda meta: False,
    "pool_every_steps": lambda meta: meta.get("snapshot_every_steps"),
}


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
        elif k in dflt:
            out[k] = v
    return out


def unrecorded(run_dir: str) -> list:
    """CLI dests this run's metadata says nothing about, so a diff cannot check them."""
    rec, dflt = recorded(run_dir), _defaults()
    return sorted(d for d in dflt if d not in rec)


def non_default(run_dir: str) -> dict:
    """{dest: value} for every recorded setting that differs from the CLI default."""
    dflt = _defaults()
    return {k: v for k, v in recorded(run_dir).items() if k in dflt and v != dflt[k]}


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


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_dir", help="a run directory, or its metadata.json")
    ap.add_argument("--diff", metavar="OTHER",
                    help="a second run; print only what differs between the two")
    args = ap.parse_args()

    a = non_default(args.run_dir)
    if not args.diff:
        print(f"# non-default flags used by {os.path.basename(args.run_dir.rstrip('/'))}")
        for p in as_flags(a):
            print(f"  {p}")
        return 0

    b = non_default(args.diff)
    keys = sorted(set(a) | set(b))
    na = os.path.basename(args.run_dir.rstrip("/"))[:28]
    nb = os.path.basename(args.diff.rstrip("/"))[:28]
    rows = [(k, a.get(k, "<default>"), b.get(k, "<default>")) for k in keys
            if a.get(k, "<default>") != b.get(k, "<default>")]
    if not rows:
        print(f"# {na} and {nb} agree on every non-default flag they record")
    else:
        print(f"{'flag':28s}{na:>30s}{nb:>30s}")
        for k, va, vb in rows:
            print(f"{'--' + k.replace('_', '-'):28s}{str(va):>30s}{str(vb):>30s}")
    # A flag a run's metadata does not record cannot be diffed; say which, so "agree" is not read
    # as covering them. Older runs predate some keys (lr, batch and buffer size, gamma: 2026-09-24).
    _ignore = UNCHECKED_BY_DESIGN
    for name, run in ((na, args.run_dir), (nb, args.diff)):
        miss = [d for d in unrecorded(run) if d not in _ignore]
        if miss:
            print(f"# not recorded by {name}, so not checked: " + ", ".join(miss))
    return 1 if rows else 0


if __name__ == "__main__":
    raise SystemExit(main())
