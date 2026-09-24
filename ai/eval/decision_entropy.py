"""Per seat and per decision type: how many options the policy has, and how spread its choice is.

Written for P23. E4.1's merged-influence view collapsed on the US seat in 4 of 4 runs, with the US
policy's entropy stuck at 2.5-2.7 while E4's settled at 1.2-1.7, and the entropy coefficient did
not move it. The owner's hypothesis is that the US has many more places to put influence than the
USSR, so the US seat never learns to rank them. This probe measures exactly that: for every
decision a checkpoint makes in self-play, the number of legal actions, the policy's entropy over
them, that entropy as a fraction of the maximum (log of the legal count), and the probability of
its top choice, split by seat, by phase (the opening -- setup placement and headline -- or play)
and by decision type.

The checkpoint plays itself in its own action view (E4 or E4.1, read from its run directory),
sampling at temperature 1, so the positions are the ones its policy reaches. Runs on the CPU by
default, so it can run beside training without touching the GPU.

    python -m ai.eval.decision_entropy <snapshot.pt> [<snapshot.pt> ...] --envs 64 --steps 400
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import ts_engine as ts

from bindings.ts_env import TsVectorizedEnv

#: (seat, phase, decision type): seat "US"/"USSR"; phase "opening" (turn 1, action round 0: the
#: setup placement and the headline) or "play".
Key = Tuple[str, str, str]


@dataclass
class Stats:
    n: int = 0
    forced: int = 0            # decisions with a single legal action (no choice to measure)
    legal: float = 0.0         # sums over non-forced decisions
    entropy: float = 0.0
    entropy_frac: float = 0.0  # entropy / log(legal)
    top_p: float = 0.0

    def add(self, legal: int, entropy: float, top_p: float) -> None:
        self.n += 1
        if legal <= 1:
            self.forced += 1
            return
        self.legal += legal
        self.entropy += entropy
        self.entropy_frac += entropy / math.log(legal)
        self.top_p += top_p

    def row(self) -> Dict[str, float]:
        m = max(1, self.n - self.forced)
        return {"n": float(self.n), "forced": float(self.forced), "legal": self.legal / m,
                "entropy": self.entropy / m, "entropy_frac": self.entropy_frac / m,
                "top_p": self.top_p / m}


def probe(model: nn.Module, merged: bool, envs: int = 64, steps: int = 400, seed: int = 0,
          device: str = "cpu") -> Dict[Key, Stats]:
    """Self-play `steps` batched decisions over `envs` games in the given view; aggregate by key."""
    dev = torch.device(device)
    model = model.to(dev).eval()
    env = TsVectorizedEnv(num_envs=envs, base_seed=seed)
    env.set_merged_influence(merged, merged)
    obs, masks, info = env.reset_all()
    gen = torch.Generator(device=dev).manual_seed(seed)
    out: Dict[Key, Stats] = {}
    for _ in range(steps):
        players = np.asarray(info["decision_players"])
        keys: List[Optional[Key]] = []
        for i in range(envs):
            st = env.runner.get_state(i)
            ctx = st.ctx()
            p = int(players[i])
            if p not in (int(ts.Player.US), int(ts.Player.USSR)):
                keys.append(None)
                continue
            seat = "US" if p == int(ts.Player.US) else "USSR"
            phase = "opening" if (int(st.turn) == 1 and int(st.action_round) == 0) else "play"
            keys.append((seat, phase, ts.DecisionType(int(ctx.decision_type)).name))
        obs_t = torch.as_tensor(np.asarray(obs), dtype=torch.float32, device=dev)
        mask_t = torch.as_tensor(np.asarray(masks), device=dev)
        with torch.no_grad():
            logits = model(obs_t, mask_t)[0].float()
        legal = mask_t.bool()
        logits = logits.masked_fill(~legal, float("-inf"))
        logp = F.log_softmax(logits, dim=-1)
        prob = logp.exp()
        ent = -(prob * torch.where(legal, logp, torch.zeros_like(logp))).sum(-1)
        top = prob.max(-1).values
        n_legal = legal.sum(-1)
        actions = torch.multinomial(prob, 1, generator=gen).squeeze(-1)
        for i, k in enumerate(keys):
            if k is not None:
                out.setdefault(k, Stats()).add(int(n_legal[i]), float(ent[i]), float(top[i]))
        obs, masks, _r, _d, info = env.step(actions.cpu().numpy())
    return out


def table(name: str, stats: Dict[Key, Stats]) -> str:
    lines = [f"## {name}", "",
             "| seat | phase | decision | n | forced | legal | entropy | entropy / log(legal) | top p |",
             "|:---|:---|:---|---:|---:|---:|---:|---:|---:|"]
    for k in sorted(stats, key=lambda k: (k[1] != "opening", k[2], k[0])):
        r = stats[k].row()
        lines.append(f"| {k[0]} | {k[1]} | {k[2]} | {int(r['n'])} | {int(r['forced'])} | "
                     f"{r['legal']:.1f} | {r['entropy']:.2f} | {r['entropy_frac']:.2f} | {r['top_p']:.2f} |")
    return "\n".join(lines)


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("checkpoints", nargs="+")
    ap.add_argument("--envs", type=int, default=64)
    ap.add_argument("--steps", type=int, default=400)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--json", default=None, help="also write every table's rows here")
    a = ap.parse_args(argv)

    from tools.lib.player_agent import NeuralAgent
    dump: Dict[str, List[Dict[str, object]]] = {}
    for path in a.checkpoints:
        agent = NeuralAgent.from_checkpoint(path, device=a.device)
        merged = bool(agent.merged_influence)
        stats = probe(agent.model, merged, envs=a.envs, steps=a.steps, seed=a.seed, device=a.device)
        name = f"{agent.name} ({'E4.1' if merged else 'E4'} view)"
        print(table(name, stats), flush=True)
        print(flush=True)
        dump[name] = [{"seat": k[0], "phase": k[1], "decision": k[2], **v.row()} for k, v in stats.items()]
    if a.json:
        with open(a.json, "w", encoding="utf-8") as f:
            json.dump(dump, f, indent=1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
