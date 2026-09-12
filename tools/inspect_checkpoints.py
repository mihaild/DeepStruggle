#!/usr/bin/env python3
"""Checkpoint Inspector: Summarizes trained models, architectures, and sizes."""

import os
import sys

_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _root not in sys.path:
    sys.path.insert(0, _root)

from tools.lib.checkpoint_utils import discover_checkpoints


def main():
    chkpt_root = os.path.join(_root, "data", "checkpoints") if os.path.exists(os.path.join(_root, "data", "checkpoints")) else os.path.join(_root, "checkpoints")
    checkpoints = discover_checkpoints(chkpt_root)

    print("=" * 95)
    print("TWILIGHT STRUGGLE AI: CHECKPOINT REGISTRY")
    print("=" * 95)
    print(f"{'Filename':35s} | {'Size':8s} | {'Modified':19s} | {'Architecture':30s}")
    print("-" * 95)

    for c in checkpoints:
        rel_path = os.path.relpath(c["path"], _root)
        print(f"{rel_path:35s} | {c['size_mb']:6.1f} MB | {c['mtime']:19s} | {c['arch']:30s}")

    print("=" * 95)
    print(f"Total Checkpoints Found: {len(checkpoints)}\n")


if __name__ == "__main__":
    main()
