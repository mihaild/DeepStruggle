#!/usr/bin/env python3
"""Checkpoint Inspector: Summarizes trained models, architectures, and tournament rankings."""

import os
import sys
import time
import json
import torch

_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _root not in sys.path:
    sys.path.insert(0, _root)


def inspect_checkpoint(path: str) -> dict:
    stat = os.stat(path)
    size_mb = stat.st_size / (1024 * 1024)
    mtime = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(stat.st_mtime))

    arch = "Unknown"
    try:
        sd = torch.load(path, map_location="cpu", weights_only=True)
        if any("cross_attn" in k or "cross_card_proj" in k for k in sd.keys()):
            arch = "ColdWarNetV2 (Cross-Attention)"
        elif any("fusion_in" in k for k in sd.keys()):
            arch = "ColdWarNet (V1)"
    except Exception:
        pass

    return {
        "path": path,
        "filename": os.path.basename(path),
        "dir": os.path.dirname(path),
        "size_mb": size_mb,
        "mtime": mtime,
        "arch": arch,
    }


def main():
    chkpt_root = os.path.join(_root, "checkpoints")
    all_checkpoints = []

    for root, _, files in os.walk(chkpt_root):
        for f in files:
            if f.endswith(".pt"):
                full_path = os.path.join(root, f)
                all_checkpoints.append(inspect_checkpoint(full_path))

    all_checkpoints.sort(key=lambda x: x["path"])

    print("=" * 95)
    print("TWILIGHT STRUGGLE AI: CHECKPOINT REGISTRY")
    print("=" * 95)
    print(f"{'Filename':35s} | {'Size':8s} | {'Modified':19s} | {'Architecture':30s}")
    print("-" * 95)

    for c in all_checkpoints:
        rel_path = os.path.relpath(c["path"], _root)
        print(f"{rel_path:35s} | {c['size_mb']:6.1f} MB | {c['mtime']:19s} | {c['arch']:30s}")

    print("=" * 95)
    print(f"Total Checkpoints Found: {len(all_checkpoints)}\n")


if __name__ == "__main__":
    main()
