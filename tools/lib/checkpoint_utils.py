"""Checkpoint inspection and discovery utilities for Twilight Struggle neural models."""

import os
import time
from typing import Dict, Any, List, Optional
import torch


def inspect_checkpoint(path: str) -> Dict[str, Any]:
    """Inspects a PyTorch checkpoint file and detects its network architecture."""
    stat = os.stat(path)
    size_mb = stat.st_size / (1024 * 1024)
    mtime = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(stat.st_mtime))

    arch = "Unknown"
    try:
        sd = torch.load(path, map_location="cpu", weights_only=True)
        if any("belief_head" in k or "card_transformer" in k for k in sd.keys()):
            arch = "retired (V4: card transformer + belief head)"
        elif any("node_pointer_proj" in k or "cross_b2c" in k for k in sd.keys()):
            arch = "retired (V3: dual pointer co-attention)"
        elif any("cross_attn" in k or "cross_card_proj" in k for k in sd.keys()):
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


def discover_checkpoints(directory: str) -> List[Dict[str, Any]]:
    """Recursively discovers and inspects all .pt checkpoints under a directory."""
    checkpoints: List[Dict[str, Any]] = []
    if not os.path.exists(directory):
        return checkpoints

    for root, _, files in os.walk(directory):
        for f in files:
            if f.endswith(".pt"):
                full_path = os.path.join(root, f)
                checkpoints.append(inspect_checkpoint(full_path))

    checkpoints.sort(key=lambda x: x["path"])
    return checkpoints
