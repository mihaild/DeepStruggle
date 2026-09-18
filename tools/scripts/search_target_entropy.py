#!/usr/bin/env python3
"""Does the searcher's visit distribution flatten as an arm declines?

Every X4b arm's decline carries the same signature: the policy's own entropy **rises** while its
strength falls. That is backwards for a policy that is merely over-sharpening on its CE targets,
and it admits two readings that no logged series separates:

* **the CE term is teaching it.** If the searcher's visit distribution flattens -- because the
  network guiding the search has weakened -- then the cross-entropy term is actively pulling the
  policy toward a flatter target, which weakens it further. A feedback loop, and the CE term is
  the mechanism.
* **the policy flattens for some other reason** and the search targets are innocent bystanders.
  Then the CE term is failing to prevent the decline rather than causing it, and removing it would
  not help.

`search_target_entropy` now logs this during training, but only from the next launch, and the arm
that is actually declining (`E3-35-28`) is running the old code. This recovers the same number
offline.

**The states are held fixed across snapshots.** Each snapshot's network guides a search over the
*same* positions, so what is measured is the network's effect on search sharpness and not a drift
in which positions the arm reaches -- those move together during a decline, and letting both vary
would make the result uninterpretable.

    tools/scripts/search_target_entropy.py --run <run-dir> --every 4 --states 60 --sims 64
"""

from __future__ import annotations

import argparse
import glob
import math
import os
import re
import statistics
import sys
from typing import List, Tuple

import numpy as np
import torch

import ts_engine as ts

from ai.search.batched_mcts import BatchedMCTS, BatchedMCTSConfig
from bindings.action_encoder import ActionEncoder
from tools.lib.player_agent import load_agent


def snapshots(run_dir: str) -> List[Tuple[int, str]]:
    out = []
    for path in glob.glob(os.path.join(run_dir, "snapshot_*steps.pt")):
        m = re.search(r"snapshot_(\d+)steps\.pt$", os.path.basename(path))
        if m:
            out.append((int(m.group(1)), path))
    return sorted(out)


def collect_states(ckpt: str, want: int, device: str) -> List[ts.GameState]:
    """Positions to ask every snapshot about, generated once from a single checkpoint.

    Using one checkpoint's self-play for all of them is the point: the comparison is between
    networks on identical inputs.
    """
    agent = load_agent(ckpt, device=device)
    states: List[ts.GameState] = []
    seed = 20260918
    while len(states) < want:
        st = ts.GameState()
        ts.Engine.init_game(st, seed)
        seed += 1
        guard = 0
        while not ts.Engine.is_terminal(st) and guard < 600 and len(states) < want:
            guard += 1
            ctx = st.ctx()
            pl = ctx.decision_player
            if pl == ts.Player.NONE:
                # chance node: drain it. try_step is the boolean half of the try_step/step split
                # -- step returns None and RAISES, so its result must never be truth-tested.
                if not ts.Engine.try_step(st, ts.MicroAction(ctx.decision_type, 0, 0, 0)):
                    break
                continue
            mask = np.asarray(ActionEncoder.get_legal_mask(st))
            n_legal = int(mask.sum())
            if n_legal == 0:
                break
            if n_legal > 1:
                # Only positions with a real choice. A forced move has a degenerate visit
                # distribution by construction, and including those would drag every snapshot's
                # mean toward zero entropy for a reason that has nothing to do with the arm.
                states.append(st.clone())
            idx = agent.select_action(st, pl, temperature=0.0)
            if not ts.Engine.try_step(st, ts.decode_flat_action(st, int(idx))):
                break
    return states[:want]


def target_entropy(net_ckpt: str, states: List[ts.GameState], sims: int,
                   device: str) -> dict:
    """Per seat: target entropy, top-1 share, and how far the policy sits from its own target.

    The seat split is the point. Two independent search-CE arms decline on the **USSR seat only**
    while their US seat holds flat, so a pooled number over both seats averages a failing half
    with a healthy one and shows a mild slide that belongs to neither.

    `ce` is the cross-entropy of the search target against the policy's own distribution at the
    same state -- the quantity the CE term minimises. It says how far each seat's policy has
    drifted from the target it is being trained toward, which entropy alone cannot: a target can
    stay perfectly sharp while the policy walks away from it.
    """
    agent = load_agent(net_ckpt, device=device)
    net = getattr(agent, "model", None) or getattr(agent, "net", None)
    if net is None:
        raise SystemExit(f"could not reach the network inside {net_ckpt}")
    searcher = BatchedMCTS(
        net, device=device,
        config=BatchedMCTSConfig(
            simulations=sims, temperature=0.0, auto_advance=True,
            advance_root=False, determinize=True, node_filter="all", subsample=1.0))
    res = searcher.run([s.clone() for s in states])

    out: dict = {}
    for seat in ("US", "USSR", "all"):
        out[seat] = {"ent": [], "top": [], "ce": []}

    for st, (acts, visits) in zip(states, res):
        if not acts:
            continue
        v = np.asarray(visits, dtype=np.float64)
        tot = float(v.sum())
        if tot <= 0.0:
            continue
        p = v / tot
        nz = p > 0.0
        ent = float(-(p[nz] * np.log(p[nz])).sum())
        top = float(p.max())

        # the policy's own distribution at this state, for the CE the training term minimises
        pl = st.ctx().decision_player
        seat = "US" if pl == ts.Player.US else "USSR"
        obs = torch.from_numpy(
            np.asarray(ts.extract_observation(st, pl), dtype=np.float32)).unsqueeze(0).to(device)
        mask = torch.from_numpy(
            np.asarray(ActionEncoder.get_legal_mask(st), dtype=np.uint8)).unsqueeze(0).to(device)
        with torch.no_grad():
            logits, _, _ = net(obs, mask)
            log_p = torch.log_softmax(logits, dim=-1)[0].cpu().numpy()
        ce = float(-sum(p[i] * log_p[int(a)] for i, a in enumerate(acts) if p[i] > 0.0))

        for key in (seat, "all"):
            out[key]["ent"].append(ent)
            out[key]["top"].append(top)
            out[key]["ce"].append(ce)

    summary: dict = {}
    for seat, d in out.items():
        if d["ent"]:
            summary[seat] = (statistics.mean(d["ent"]), statistics.mean(d["top"]),
                             statistics.mean(d["ce"]), len(d["ent"]))
        else:
            summary[seat] = (float("nan"), float("nan"), float("nan"), 0)
    return summary


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter,
                                 allow_abbrev=False)
    ap.add_argument("--run", required=True)
    ap.add_argument("--every", type=int, default=4)
    ap.add_argument("--states", type=int, default=60)
    ap.add_argument("--sims", type=int, default=64)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--state-source", default=None,
                    help="checkpoint whose self-play supplies the fixed positions "
                         "(default: the run's earliest snapshot)")
    a = ap.parse_args()

    snaps = snapshots(a.run)[:: a.every]
    if not snaps:
        print("no snapshots in", a.run, file=sys.stderr)
        return 1

    source = a.state_source or snapshots(a.run)[0][1]
    print("states from: %s" % os.path.basename(source), flush=True)
    states = collect_states(source, a.states, a.device)
    print("collected %d searchable positions, %d simulations each\n" % (len(states), a.sims),
          flush=True)

    hdr = "%-12s" % "steps"
    for seat in ("US", "USSR", "all"):
        hdr += " | %s H   top1    CE" % seat.ljust(4)
    print(hdr)
    print("-" * len(hdr))
    for steps, path in snaps:
        s = target_entropy(path, states, a.sims, a.device)
        row = "%-12d" % steps
        for seat in ("US", "USSR", "all"):
            h, top, ce, n = s[seat]
            row += " | %5.3f %5.3f %6.3f" % (h, top, ce)
        print(row, flush=True)
    counts = target_entropy(snaps[0][1], states, a.sims, a.device)
    print("\npositions per seat: US %d, USSR %d" % (counts["US"][3], counts["USSR"][3]))
    print("max possible entropy for a 212-way choice: %.3f" % math.log(212))
    print("CE is the search target against the policy's own distribution -- the quantity the "
          "training CE term minimises.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
