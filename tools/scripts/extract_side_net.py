#!/usr/bin/env python3
"""One side's network out of a --ladder-side-nets (A39) checkpoint or resume state (owner, 2026-11-26).

    PYTHONPATH=. .venv/bin/python tools/scripts/extract_side_net.py <a39 .pt> --side ussr --output <out .pt>

A plain checkpoint (snapshot, SWA) gives the side's state dict, a plain LadderNet's (A8 for T1's flags). A resume state
gives a resume state for that network alone -- weights, pi_ref, Adam moments mapped by name, counters, RNG -- marked
`side_only`, so a launch resuming it must be seat-locked to that side against a fixed opponent with
--update-learner-rows-only (generic_trainer.check_side_only). Write a resume state beside its source, named
`side-<side>_resume_<steps>steps.pt`, so the branch's lineage and step are read from it (tools/scripts/run_codes.py).
Put the network back beside the base's other side with recombine_side_nets.py (tools/lib/side_nets.py)."""

from __future__ import annotations

import argparse
import os

import torch

from tools.lib.side_nets import SIDES, extract


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("source", help="an A39 checkpoint (state dict) or resume state")
    ap.add_argument("--side", required=True, choices=SIDES)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()
    if os.path.exists(args.output):
        raise SystemExit(f"{args.output} exists; refusing to overwrite it")
    blob = torch.load(args.source, map_location="cpu", weights_only=False)
    out = extract(blob, args.side, source=os.path.abspath(args.source))
    torch.save(out, args.output)
    kind = "resume state" if "model_state_dict" in out else "checkpoint"
    n = len(out["model_state_dict"] if "model_state_dict" in out else out)
    print(f"wrote {args.output}: the {args.side} network alone ({kind}, {n} entries)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
