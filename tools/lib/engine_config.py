"""The engine configuration a run was trained under.

Width is not a version number. Two observation layouts of the same width can differ in content,
and nothing in the stack can tell them apart -- not the width assertions, not `layout_of`, not a
forward pass. So a run records which engine features it trained with, and a missing entry is
false: a checkpoint from before a flag existed keeps the behaviour it learned, without anyone
having to remember what that was.

Recorded in the run's metadata.json under "engine_config". Read back by name, never inferred.
"""

from __future__ import annotations

import json
import os
from typing import Dict

import ts_engine as ts

#: Flag name -> bit. The name is what appears in metadata.json; the bit is what the engine takes.
#:
#: `staged_cards` is retired and does nothing. It made a card a decision was *about* visible to
#: the player deciding, which Grain Sales needed because it showed a card without moving it; the
#: card now goes to PEEKED_TEMP and reaches the observation through the ordinary location chain,
#: so there is nothing left to switch on. It stays listed, and the bit stays reserved, because
#: runs that recorded it by name are still on disk and `to_mask` refuses names it does not know
#: -- dropping it would turn reading their metadata into an error.
FLAGS: Dict[str, int] = {
    "staged_cards": int(ts.OBS_FLAG_STAGED_CARDS),
}


def to_mask(config: Dict[str, bool]) -> int:
    """Bitmask for a config. Anything absent or false contributes nothing."""
    mask = 0
    for name, on in config.items():
        if not on:
            continue
        if name not in FLAGS:
            raise ValueError(f"unknown engine flag {name!r}; known: {sorted(FLAGS)}")
        mask |= FLAGS[name]
    return mask


def from_names(names: list[str] | None) -> Dict[str, bool]:
    """Config from a list of flag names, as the CLI supplies them."""
    chosen = set(names or [])
    unknown = chosen - set(FLAGS)
    if unknown:
        raise ValueError(f"unknown engine flag(s) {sorted(unknown)}; known: {sorted(FLAGS)}")
    return {name: (name in chosen) for name in sorted(FLAGS)}


def for_checkpoint(checkpoint_path: str) -> Dict[str, bool]:
    """The config a checkpoint was trained under, from its run directory's metadata.json.

    A snapshot is a bare state dict and carries nothing itself, so this reads the metadata beside
    it. No metadata, or no engine_config in it, means every flag is false -- which is what every
    run before this recorded, by not recording anything.
    """
    meta = os.path.join(os.path.dirname(os.path.abspath(checkpoint_path)), "metadata.json")
    if not os.path.exists(meta):
        return {name: False for name in FLAGS}
    try:
        with open(meta, encoding="utf-8") as f:
            blob = json.load(f)
    except Exception:
        return {name: False for name in FLAGS}
    recorded = blob.get("engine_config") or {}
    return {name: bool(recorded.get(name, False)) for name in FLAGS}


def mask_for_checkpoint(checkpoint_path: str) -> int:
    return to_mask(for_checkpoint(checkpoint_path))
