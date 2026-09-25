#!/usr/bin/env python3
"""Export a checkpoint to ONNX for the browser workbench, and prove the export is the checkpoint.

    PYTHONPATH=.:build/release .venv/bin/python tools/export_onnx.py \
        --checkpoint data/checkpoints/<run>/snapshot_final.pt --out model.onnx

The page (web/ui) runs the file with onnxruntime-web, from a file dropped onto it, from a
Hugging Face repo, or from the local server, which calls `export()` on demand. Everything the page
must know before it runs a model travels inside the file, as ONNX metadata_props:

    ts.format             "ts-onnx-v1"
    ts.obs_size           observation width the network reads -- the page refuses a mismatch
    ts.action_size        flat action space width
    ts.merged_influence   "true" for an E4.1 (P23) network, which decides in the merged view
    ts.label              run-attributed name, e.g. "E4-08-03-160M.11@240M"
    ts.checkpoint         the checkpoint's run/file name, and ts.checkpoint_sha256 its digest
    ts.engine_fingerprint the engine sources this export was made next to

Two properties are checked on real positions and the export is refused if either fails:

* **The value heads ignore the mask.** Python asks the critic with no mask; the ONNX graph always
  takes one, so the page passes all ones. That is only the same question if no value head reads
  the mask.
* **ONNX Runtime agrees with torch**: the same favourite move on every position, probabilities
  within 1e-3 and values within 1e-4. An export that runs but computes something else is the
  failure this exists to catch.
"""
from __future__ import annotations

import argparse
import hashlib
import os
import random
import sys
import tempfile
from typing import Dict, List, Tuple

import numpy as np
import torch

import ts_engine as ts
from bindings.action_encoder import ActionEncoder
from tools.lib.engine_fingerprint import fingerprint
from tools.lib.game_step import drain_chance
from tools.lib.player_agent import NeuralAgent

FORMAT = "ts-onnx-v1"
INPUTS = ("obs", "mask")
OUTPUTS = ("logits", "v_win", "v_vp")


class ExportRefused(RuntimeError):
    """The export would not be the checkpoint."""


def _positions(n: int, merged: bool, seed: int = 20260925) -> Tuple[np.ndarray, np.ndarray]:
    """Real decision nodes from random games, with masks in the network's own action view."""
    rng = random.Random(seed)
    obs: List[np.ndarray] = []
    masks: List[np.ndarray] = []
    game = 1
    while len(obs) < n:
        s = ts.GameState()
        ts.Engine.init_game(s, seed + game)
        drain_chance(s)
        while not ts.Engine.is_terminal(s) and len(obs) < n:
            p = s.ctx().decision_player
            if p != ts.Player.NONE and rng.random() < 0.25:
                obs.append(np.asarray(ts.extract_observation(s, p), dtype=np.float32))
                masks.append(np.asarray(ActionEncoder.get_legal_mask(s, merged), dtype=np.uint8))
            ts.Engine.step_flat(s, rng.choice(ActionEncoder.get_legal_indices(s)))
            drain_chance(s)
        game += 1
    return np.stack(obs), np.stack(masks)


def _sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def export(checkpoint: str, out: str, *, positions: int = 128) -> Dict[str, str]:
    """Write `checkpoint` as ONNX to `out`, verified. Returns the metadata written."""
    import onnx
    import onnxruntime as ort

    agent = NeuralAgent.from_checkpoint(checkpoint, device="cpu")
    model = agent.model.eval()
    obs, masks = _positions(positions, agent.merged_influence)
    obs_t, masks_t = torch.from_numpy(obs), torch.from_numpy(masks)

    with torch.no_grad():
        t_logits, t_win, t_vp = model(obs_t, masks_t)
        _, n_win, n_vp = model(obs_t, None)
        _, o_win, o_vp = model(obs_t, torch.ones_like(masks_t))
    for name, a, b in (("v_win", n_win, o_win), ("v_vp", n_vp, o_vp), ("v_win", n_win, t_win)):
        gap = float((a - b).abs().max())
        if gap > 1e-6:
            raise ExportRefused(f"{name} depends on the mask (max gap {gap:.2e}); the page's critic "
                                f"rows would not ask the question Python asks")

    meta = {
        "ts.format": FORMAT,
        "ts.obs_size": str(int(ts.OBS_SIZE)),
        "ts.action_size": str(int(ActionEncoder.FLAT_ACTION_SIZE)),
        "ts.merged_influence": "true" if agent.merged_influence else "false",
        "ts.label": agent.name,
        "ts.checkpoint": os.path.join(os.path.basename(os.path.dirname(os.path.abspath(checkpoint))),
                                      os.path.basename(checkpoint)),
        "ts.checkpoint_sha256": _sha256(checkpoint),
        "ts.engine_fingerprint": fingerprint(),
    }

    batch = torch.export.Dim("batch", min=1, max=4096)
    program = torch.onnx.export(
        model, (obs_t[:2], masks_t[:2]), dynamo=True,
        input_names=list(INPUTS), output_names=list(OUTPUTS),
        dynamic_shapes=({0: batch}, {0: batch}))
    if program is None:
        raise ExportRefused("torch.onnx.export produced no program")
    out_dir = os.path.dirname(os.path.abspath(out))
    os.makedirs(out_dir, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=out_dir) as tmp:
        raw = os.path.join(tmp, "raw.onnx")
        program.save(raw)
        proto = onnx.load(raw)   # weights inline: the page loads a single file
        del proto.metadata_props[:]
        for k, v in meta.items():
            entry = proto.metadata_props.add()
            entry.key, entry.value = k, v
        final = os.path.join(tmp, "final.onnx")
        onnx.save_model(proto, final, save_as_external_data=False)

        sess = ort.InferenceSession(final, providers=["CPUExecutionProvider"])
        r_logits, r_win, r_vp = sess.run(None, {"obs": obs, "mask": masks})
        legal = masks > 0
        tp = torch.softmax(t_logits, dim=-1).numpy()
        rp = torch.softmax(torch.from_numpy(np.asarray(r_logits)), dim=-1).numpy()
        if not np.array_equal(tp.argmax(1), rp.argmax(1)):
            raise ExportRefused("ONNX Runtime picks a different favourite move than torch")
        dp = float(np.abs(tp - rp)[legal].max())
        dv = max(float(np.abs(t_win.numpy().ravel() - np.asarray(r_win).ravel()).max()),
                 float(np.abs(t_vp.numpy().ravel() - np.asarray(r_vp).ravel()).max()))
        if dp > 1e-3 or dv > 1e-4:
            raise ExportRefused(f"ONNX Runtime disagrees with torch (max |dp| {dp:.1e}, |dv| {dv:.1e})")
        os.replace(final, out)
    return meta


def main(argv: List[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--checkpoint", required=True, help="a snapshot_*.pt")
    ap.add_argument("--out", required=True, help="the .onnx to write")
    ap.add_argument("--positions", type=int, default=128, help="real positions the export is verified on")
    args = ap.parse_args(argv)
    try:
        meta = export(args.checkpoint, args.out, positions=args.positions)
    except ExportRefused as e:
        print(f"export_onnx: refused: {e}", file=sys.stderr)
        return 1
    print(f"export_onnx: {meta['ts.label']} -> {args.out} ({os.path.getsize(args.out) / 1e6:.1f} MB, "
          f"{'E4.1' if meta['ts.merged_influence'] == 'true' else 'E4'} view)")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
