"""Do the two sides' policy gradients actually conflict?

Gradient surgery (PCGrad and relatives) projects away the component of one task's gradient that
opposes another's, and is motivated only when the task gradients genuinely conflict -- negative
cosine similarity. Treating "win as US" and "win as USSR" as two tasks sharing one network makes
that a measurable question rather than an assumption, and the answer decides whether the method
applies at all:

  cos < 0   the two sides pull against each other; PCGrad has something to remove
  cos ~ 0   the updates are largely independent; surgery would be a no-op at 2x the backward cost
  cos > 0   the sides reinforce each other; surgery would be actively wrong

The surrogate is the plain policy-gradient direction `-(logp * A)`, which is what PPO's gradient
reduces to on the first epoch, when the importance ratio is still exactly 1.
"""
from __future__ import annotations

from typing import Any, Dict, List

import numpy as np


def _flat_grad(params: List[Any]) -> Any:
    import torch

    return torch.cat([(p.grad.detach().reshape(-1) if p.grad is not None
                       else torch.zeros(p.numel(), device=p.device))
                      for p in params])


def measure(trainer: Any, repeats: int = 4, chunk: int = 1024) -> Dict[str, Any]:
    """Cosine between the US-only and USSR-only policy gradients, over several rollouts.

    The per-side gradient is accumulated over minibatches of `chunk` transitions rather than
    taken in one pass: the full side of a rollout does not fit in memory alongside a running
    training job, and the accumulated sum is the same gradient either way (each chunk is
    weighted by its share of the side's transitions).
    """
    import torch
    import torch.nn.functional as F

    net = trainer.active_net
    params = [p for p in net.parameters() if p.requires_grad]
    out: List[Dict[str, float]] = []

    for _ in range(repeats):
        trainer.collect_rollouts()
        b = trainer.buffer
        obs = b.obs.reshape(-1, b.obs.shape[-1])
        masks = b.masks.reshape(-1, b.masks.shape[-1])
        actions = b.actions.reshape(-1)
        adv = b.advantages.reshape(-1)
        players = b.players.reshape(-1)

        grads = {}
        for tag, code in (("us", 1), ("ussr", -1)):
            sel = (players == code).nonzero(as_tuple=True)[0]
            if sel.numel() == 0:
                continue
            net.zero_grad(set_to_none=True)
            total = int(sel.numel())
            for start in range(0, total, chunk):
                idx = sel[start:start + chunk]
                logits, _, _ = net(obs[idx], masks[idx])
                logp = F.log_softmax(logits, dim=-1).gather(
                    1, actions[idx].unsqueeze(1)).squeeze(1)
                # Weighted by this chunk's share so the accumulation equals the full-batch mean.
                loss = -(logp * adv[idx]).sum() / total
                loss.backward()
                del logits, logp, loss
            grads[tag] = _flat_grad(params).clone()

        if len(grads) == 2:
            g_us, g_su = grads["us"], grads["ussr"]
            cos = float(F.cosine_similarity(g_us.unsqueeze(0), g_su.unsqueeze(0)).item())
            out.append({
                "cosine": cos,
                "norm_us": float(g_us.norm()),
                "norm_ussr": float(g_su.norm()),
                "norm_ratio": float(g_su.norm() / (g_us.norm() + 1e-12)),
            })
        net.zero_grad(set_to_none=True)

    cosines = np.array([r["cosine"] for r in out])
    return {
        "per_rollout": out,
        "cosine_mean": float(cosines.mean()) if len(cosines) else float("nan"),
        "cosine_std": float(cosines.std()) if len(cosines) > 1 else 0.0,
        "frac_conflicting": float((cosines < 0).mean()) if len(cosines) else float("nan"),
    }
