#!/usr/bin/env python3
"""Does the searcher shift play-mode choices between EVENT and OPS, and which way?

The hypothesis this exists to test: a searcher that is bad at influence *placement* will
underrate the whole "play for operations" branch, because the value it backs up from that branch
is the value of the placements it chose inside it. If it cannot place well, ops look worse than
they are and it drifts toward EVENT (or toward SPACE/PASS). Distilling that produces a policy with
a systematically biased play-mode preference, and no amount of it builds a strong player.

The test is direct: at `SELECT_PLAY_MODE` nodes, compare the raw policy's distribution with the
searcher's visit distribution over the same legal set. Flat actions are fixed --

    110 EVENT   111 OPS   112 SPACE   113 PASS

and `CHOOSE_TIMING_BRANCH` is reported alongside (114 OPS_FIRST, 115 EVENT_FIRST) because it is
the same question asked about ordering.

The headline number is conditional on **both EVENT and OPS being legal**, since a node where only
one is available says nothing about preference.

    tools/scripts/play_mode_preference.py --checkpoint <ckpt.pt> --dataset <targets.jsonl.gz>
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import List

import numpy as np
import torch

from ai.training.warmup_dataset_loader import WarmupDataset
from tools.lib.player_agent import NeuralAgent

EVENT, OPS, SPACE, PASS = 110, 111, 112, 113
OPS_FIRST, EVENT_FIRST = 114, 115
SELECT_PLAY_MODE, CHOOSE_TIMING_BRANCH = 2, 3


def _pct(x: float) -> str:
    return f"{x:6.2%}"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter,
                                 allow_abbrev=False)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--max-games", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=512)
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()

    dev = torch.device(a.device if torch.cuda.is_available() else "cpu")
    model = NeuralAgent.from_checkpoint(a.checkpoint, device=dev).model
    model.eval()

    meta = a.dataset + ".meta.json"
    if os.path.exists(meta):
        with open(meta, encoding="utf-8") as f:
            print("dataset:", json.dumps(json.load(f).get("searcher"), sort_keys=True))

    # accumulators, conditional on both EVENT and OPS legal
    pol_mass = np.zeros(4)
    sea_mass = np.zeros(4)
    n_both = 0
    flips = np.zeros((2, 2))          # policy argmax (event/ops) x search argmax (event/ops)
    pol_ev_list: List[float] = []
    sea_ev_list: List[float] = []

    # timing branch, reported separately
    t_pol = np.zeros(2)
    t_sea = np.zeros(2)
    n_t = 0

    with torch.no_grad():
        for b_obs, b_mask, b_pi, b_dt in WarmupDataset(a.dataset).stream_policy_batches(
                batch_size=a.batch_size, max_games=a.max_games, device=dev):
            logits, _, _ = model(b_obs, b_mask)
            probs = torch.softmax(logits, dim=-1)
            probs = probs * b_mask.float()
            probs = probs / probs.sum(dim=-1, keepdim=True).clamp(min=1e-9)

            dt = b_dt.cpu().numpy()
            m = b_mask.cpu().numpy()
            p = probs.cpu().numpy()
            s = b_pi.cpu().numpy()

            sel = (dt == SELECT_PLAY_MODE) & (m[:, EVENT] > 0) & (m[:, OPS] > 0)
            if sel.any():
                idx = [EVENT, OPS, SPACE, PASS]
                pol_mass += p[sel][:, idx].sum(axis=0)
                sea_mass += s[sel][:, idx].sum(axis=0)
                n_both += int(sel.sum())
                # share of the EVENT/OPS pair only, which is the cleanest contrast
                pe = p[sel][:, EVENT] / np.clip(p[sel][:, EVENT] + p[sel][:, OPS], 1e-9, None)
                se = s[sel][:, EVENT] / np.clip(s[sel][:, EVENT] + s[sel][:, OPS], 1e-9, None)
                pol_ev_list.extend(pe.tolist())
                sea_ev_list.extend(se.tolist())
                pa = (p[sel][:, EVENT] < p[sel][:, OPS]).astype(int)   # 0 event, 1 ops
                sa = (s[sel][:, EVENT] < s[sel][:, OPS]).astype(int)
                for i, j in zip(pa, sa):
                    flips[i, j] += 1

            selt = (dt == CHOOSE_TIMING_BRANCH) & (m[:, OPS_FIRST] > 0) & (m[:, EVENT_FIRST] > 0)
            if selt.any():
                t_pol += p[selt][:, [OPS_FIRST, EVENT_FIRST]].sum(axis=0)
                t_sea += s[selt][:, [OPS_FIRST, EVENT_FIRST]].sum(axis=0)
                n_t += int(selt.sum())

    if n_both == 0:
        print("no SELECT_PLAY_MODE node had both EVENT and OPS legal -- nothing to compare")
        return 1

    pol = pol_mass / n_both
    sea = sea_mass / n_both
    pe = np.asarray(pol_ev_list)
    se = np.asarray(sea_ev_list)

    print(f"\n{n_both:,} SELECT_PLAY_MODE decisions with BOTH event and ops legal\n")
    print(f"  {'':<8} {'policy':>9} {'search':>9} {'shift':>9}")
    for k, name in enumerate(["EVENT", "OPS", "SPACE", "PASS"]):
        print(f"  {name:<8} {_pct(pol[k]):>9} {_pct(sea[k]):>9} {sea[k]-pol[k]:+9.2%}")

    print(f"\n  share of the EVENT/OPS pair going to EVENT:")
    print(f"    policy {pe.mean():6.2%}   search {se.mean():6.2%}   "
          f"shift {se.mean()-pe.mean():+.2%}")

    tot = flips.sum()
    print(f"\n  argmax between EVENT and OPS ({int(tot):,} decisions):")
    print(f"    both say EVENT    {_pct(flips[0,0]/tot)}")
    print(f"    both say OPS      {_pct(flips[1,1]/tot)}")
    print(f"    policy OPS  -> search EVENT  {_pct(flips[1,0]/tot)}  <- search pulling to event")
    print(f"    policy EVENT -> search OPS   {_pct(flips[0,1]/tot)}  <- search pulling to ops")
    net = (flips[1, 0] - flips[0, 1]) / tot
    print(f"    net flow to EVENT {net:+.2%}")

    if n_t:
        tp = t_pol / n_t
        ts = t_sea / n_t
        print(f"\n  {n_t:,} CHOOSE_TIMING_BRANCH decisions with both orders legal")
        print(f"    OPS_FIRST    policy {_pct(tp[0])}  search {_pct(ts[0])}  {ts[0]-tp[0]:+.2%}")
        print(f"    EVENT_FIRST  policy {_pct(tp[1])}  search {_pct(ts[1])}  {ts[1]-tp[1]:+.2%}")

    print("\nReading: a large positive shift toward EVENT is the signature of a searcher that")
    print("cannot place influence well -- it backs up a poor value for the ops branch because the")
    print("placements inside that branch are its own. A shift near zero rules that out.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
