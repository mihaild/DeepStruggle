"""What the local workbench server offers the page: this machine's checkpoints and replays.

The workbench runs in the browser (web/ui: the engine as WebAssembly, models with
onnxruntime-web), so the server holds no game. It only lets the page pick from local files it
could not otherwise reach:

* checkpoints, listed from the checkpoints tree and exported to ONNX on first request
  (tools/export_onnx.py, verified against torch), cached next to the shared data tree;
* replays, from the replay directory.

Only files under those two trees can be named, by a path relative to them, so a URL can carry
the choice and cannot point the server at an arbitrary file.
"""
from __future__ import annotations

import os
import re
import threading
from typing import List, Optional

from tools.lib.data_root import checkpoints_dir, data_path
from tools.lib.engine_fingerprint import fingerprint
from web.server.replay_types import AnalysisModelListDict, AnalysisRunDict

_SNAPSHOT_STEPS = re.compile(r"snapshot_(\d+)steps\.pt$")


class LocalFileError(ValueError):
    """A request for something the local server does not offer."""


def models_root() -> str:
    """The checkpoints tree. `$TS_CHECKPOINTS_DIR` overrides the shared one, per call."""
    return os.environ.get("TS_CHECKPOINTS_DIR") or checkpoints_dir()


def onnx_cache_root() -> str:
    """Where exports are kept. `$TS_ONNX_CACHE_DIR` overrides it, per call."""
    return os.environ.get("TS_ONNX_CACHE_DIR") or data_path("onnx_cache")


def _is_weights_file(name: str) -> bool:
    # `resume_*.pt` hold optimizer and scheduler state for continuing a run, not a network.
    return name.endswith(".pt") and not name.startswith("resume_")


def _snapshot_sort_key(name: str) -> tuple[int, int, str]:
    m = _SNAPSHOT_STEPS.search(name)
    if m:
        return (1, int(m.group(1)), name)
    if name == "snapshot_final.pt":
        return (2, 0, name)
    return (0, 0, name)


def list_models(root: Optional[str] = None) -> AnalysisModelListDict:
    """Every network under the checkpoints tree, grouped by run directory, newest run first."""
    root = root or models_root()
    runs: List[AnalysisRunDict] = []
    loose: List[str] = []
    if os.path.isdir(root):
        dated: List[tuple[float, AnalysisRunDict]] = []
        for entry in os.scandir(root):
            if entry.is_file() and _is_weights_file(entry.name):
                loose.append(entry.name)
            elif entry.is_dir():
                try:
                    snaps = [f.name for f in os.scandir(entry.path) if f.is_file() and _is_weights_file(f.name)]
                except OSError:
                    continue
                if snaps:
                    snaps.sort(key=_snapshot_sort_key)
                    dated.append((entry.stat().st_mtime, {"run": entry.name, "snapshots": snaps}))
        dated.sort(key=lambda t: t[0], reverse=True)
        runs = [r for _, r in dated]
    loose.sort()
    return {"root": root, "runs": runs, "loose": loose}


def resolve_model_path(rel: str, root: Optional[str] = None) -> str:
    """The absolute path of a checkpoint named relative to the checkpoints tree; anything that
    escapes it, or is not an existing network file, is refused."""
    root = os.path.realpath(root or models_root())
    path = os.path.realpath(os.path.join(root, rel))
    if (not path.startswith(root + os.sep) or not _is_weights_file(os.path.basename(path))
            or not os.path.isfile(path)):
        raise LocalFileError(f"no checkpoint {rel!r} under {root}")
    return path


# One export at a time, process-wide: torch.onnx's exporter (torch.export underneath) keeps
# global state, and two exports in parallel threads fail each other -- observed as two refused
# exports when the page asked for two checkpoints at once. A request for a checkpoint another
# request is already exporting then finds it in the cache.
_export_lock = threading.Lock()


def onnx_for(rel: str) -> str:
    """The ONNX export of checkpoint `rel`, exported and verified on first request.

    Keyed by the checkpoint's size and mtime and by the engine fingerprint: a retrained snapshot
    or a rebuilt engine gets a fresh export (whose metadata names the engine it was made next
    to), never a stale one.
    """
    from tools.export_onnx import export

    path = resolve_model_path(rel)
    st = os.stat(path)
    key = f"{st.st_size}-{st.st_mtime_ns}-{fingerprint()[:12]}.onnx"
    out = os.path.join(onnx_cache_root(), os.path.relpath(path, os.path.realpath(models_root())), key)
    if os.path.exists(out):
        return out
    with _export_lock:
        if not os.path.exists(out):
            export(path, out)
    return out
