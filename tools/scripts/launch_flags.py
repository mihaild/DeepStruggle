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


def non_default(run_dir: str) -> dict:
    """{dest: value} for every metadata key that differs from the CLI default."""
    meta, dflt = _metadata(run_dir), _defaults()
    out = {}
    for k, v in meta.items():
        if k in _NOT_FLAGS or k not in dflt:
            continue
        if v != dflt[k]:
            out[k] = v
    return out


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
        print(f"# {na} and {nb} agree on every non-default flag")
        return 0
    print(f"{'flag':28s}{na:>30s}{nb:>30s}")
    for k, va, vb in rows:
        print(f"{'--' + k.replace('_', '-'):28s}{str(va):>30s}{str(vb):>30s}")
    return 1 if rows else 0


if __name__ == "__main__":
    raise SystemExit(main())
