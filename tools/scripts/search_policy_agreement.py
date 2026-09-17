#!/usr/bin/env python3
"""How far apart are the searcher and the policy that produced its prior?

P15-X4a's fallback question, worth asking *before* distilling rather than only after a null:
if the policy already agrees with the searcher everywhere, cross-entropy toward the searcher
cannot move it, and the search edge must live in something a single policy cannot express — the
averaging over sampled worlds acting as a state-dependent mixture, say. The plan says that is a
finding in its own right.

Reports, over a search-target dataset:

* **top-1 agreement** — how often the policy's argmax is the searcher's argmax. The headroom for
  a greedy player.
* **KL(search || policy)** — the CE term's actual gradient signal, which is non-zero even where
  the argmax agrees.
* **the searcher's own concentration** — a flat visit distribution is a weak teacher regardless
  of agreement, so it bounds what any distillation could extract.

    tools/scripts/search_policy_agreement.py --checkpoint <ckpt.pt> --dataset <targets.jsonl.gz>
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

    meta_path = a.dataset + ".meta.json"
    if os.path.exists(meta_path):
        with open(meta_path, encoding="utf-8") as f:
            print("dataset:", json.dumps(json.load(f).get("searcher"), sort_keys=True))

    n = 0
    top1 = 0
    kl_sum = 0.0
    ce_sum = 0.0
    ent_sum = 0.0
    legal_sum = 0
    top_mass: List[float] = []
    ranks: List[int] = []

    with torch.no_grad():
        for b_obs, b_mask, b_pi in WarmupDataset(a.dataset).stream_policy_batches(
                batch_size=a.batch_size, max_games=a.max_games, device=dev):
            logits, _, _ = model(b_obs, b_mask)
            logp = torch.log_softmax(logits, dim=-1)
            p_search = b_pi
            nz = p_search > 0

            ce = -(p_search * logp).sum(dim=-1)
            ent = -(p_search[nz] * torch.log(p_search[nz])).sum() / max(1, b_obs.shape[0])
            kl = ce.mean() - ent

            bs = b_obs.shape[0]
            n += bs
            top1 += int((logits.argmax(-1) == p_search.argmax(-1)).sum().item())
            ce_sum += float(ce.sum().item())
            kl_sum += float(kl.item()) * bs
            ent_sum += float(ent.item()) * bs
            legal_sum += int(b_mask.sum().item())
            top_mass.extend(p_search.max(dim=-1).values.cpu().numpy().tolist())

            # where the policy ranks the searcher's choice
            best = p_search.argmax(-1, keepdim=True)
            order = logits.argsort(dim=-1, descending=True)
            ranks.extend((order == best).float().argmax(dim=-1).cpu().numpy().tolist())

    if n == 0:
        print("no search targets in the dataset")
        return 1

    tm = np.asarray(top_mass)
    rk = np.asarray(ranks)
    print(f"\n{n:,} searched positions, mean {legal_sum / n:.1f} legal actions each\n")
    print(f"  top-1 agreement, policy vs searcher : {top1 / n:6.1%}")
    print(f"  searcher's choice in the policy's top 3 : {(rk < 3).mean():6.1%}")
    print(f"  median rank the policy gives it     : {int(np.median(rk))}")
    print()
    print(f"  cross-entropy CE(search || policy)  : {ce_sum / n:6.4f}")
    print(f"  entropy of the search target        : {ent_sum / n:6.4f}")
    print(f"  KL(search || policy)                : {kl_sum / n:6.4f}   <- the CE term's signal")
    print()
    print(f"  searcher's top-action visit share   : mean {tm.mean():.3f}, "
          f"median {np.median(tm):.3f}")
    print(f"  targets that are near-deterministic (>0.9) : {(tm > 0.9).mean():6.1%}")
    print(f"  targets that are near-flat (<0.3)          : {(tm < 0.3).mean():6.1%}")
    print()
    print("Reading: high agreement with low KL means little for CE to move, and the search edge")
    print("is not expressible as a re-weighting of this policy. Low agreement means the headroom")
    print("is real and the question is whether SFT can take it.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
