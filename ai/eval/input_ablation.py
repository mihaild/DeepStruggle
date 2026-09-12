"""Does the network actually use each slice of its observation?

`coldwar_net` spends a Conv1d, a 512->128 projection and 128 of the 768 fusion inputs on a 16-step
action history that has never been shown to matter. The cheapest test is to zero a slice at
inference and see whether the policy moves: a branch the network relies on cannot be blanked
without changing what it does.

This is a necessary-condition test, not a sufficient one. A slice that changes nothing when zeroed
is certainly not being used; a slice that does change something might still be carrying information
that is redundant with the rest of the observation, so a real removal has to be confirmed by
retraining. It is meant to say whether that retraining is worth running.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np

import ts_engine as ts

#: Slices of the observation, matching ColdWarNet's own offsets.
SLICES: Dict[str, Tuple[int, int]] = {
    "board": (0, 2352),
    "cards": (2352, 2352 + 1320),
    "global": (3672, 3672 + 76),
    "history": (3748, 3748 + 512),
    "turn_aggregates": (4260, 4292),
}

#: Within each 32-wide history step the encoder writes only slots 0..8; the rest are memset to
#: zero and never touched (`engine/src/observation.cpp`). 16 x 23 = 368 floats of the 4293 are
#: structurally constant.
HISTORY_DEAD_SLOTS = 16 * 23


@dataclass
class AblationResult:
    name: str
    positions: int = 0
    argmax_changed: int = 0
    kl: List[float] = field(default_factory=list)
    value_delta: List[float] = field(default_factory=list)

    @property
    def changed_pct(self) -> float:
        return 100.0 * self.argmax_changed / max(1, self.positions)

    def mean_kl(self) -> float:
        return float(np.mean(self.kl)) if self.kl else float("nan")

    def mean_abs_value_delta(self) -> float:
        return float(np.mean(np.abs(self.value_delta))) if self.value_delta else float("nan")


def collect_observations(model: Any, device: Any, num_envs: int = 256, base_seed: int = 5150,
                         max_iters: int = 4000, want: int = 6000,
                         temperature: float = 0.1) -> Tuple[np.ndarray, np.ndarray]:
    """Observations and masks from self-play under the model's own policy."""
    import torch

    runner = ts.VectorizedBatchRunner(num_envs, base_seed)
    obs_out: List[np.ndarray] = []
    mask_out: List[np.ndarray] = []
    model.eval()

    for _ in range(max_iters):
        if sum(len(o) for o in obs_out) >= want:
            break
        terminals = runner.get_terminals()
        if all(terminals):
            break
        obs = np.asarray(runner.get_observations(), dtype=np.float32)
        masks = np.asarray(runner.get_action_masks())
        keep = ~np.asarray(terminals, dtype=bool)
        if keep.any():
            obs_out.append(obs[keep].copy())
            mask_out.append(masks[keep].copy())
        with torch.no_grad():
            logits = model(torch.from_numpy(obs).to(device),
                           torch.from_numpy(masks).to(device))[0]
            picks = torch.multinomial(
                torch.softmax(logits / temperature, dim=-1), 1).squeeze(-1).cpu().numpy()
        runner.step_flat_all([int(a) for a in picks], auto_advance=True)

    return (np.concatenate(obs_out)[:want], np.concatenate(mask_out)[:want])


def ablate(model: Any, obs: np.ndarray, mask: np.ndarray, device: Any,
           lo: int, hi: int, name: str, batch: int = 512) -> AblationResult:
    """Zero `obs[:, lo:hi]` and measure how far the policy and value move."""
    import torch

    out = AblationResult(name=name)
    for start in range(0, len(obs), batch):
        o = obs[start:start + batch]
        m = mask[start:start + batch]
        o_zero = o.copy()
        o_zero[:, lo:hi] = 0.0

        with torch.no_grad():
            mt = torch.from_numpy(m).to(device)
            base_logits, base_v, _ = model(torch.from_numpy(o).to(device), mt)
            abl_logits, abl_v, _ = model(torch.from_numpy(o_zero).to(device), mt)
            base_lp = torch.log_softmax(base_logits, dim=-1)
            abl_lp = torch.log_softmax(abl_logits, dim=-1)
            base_p = base_lp.exp()
            # KL(base || ablated), summed over legal actions only -- the illegal ones carry
            # -inf logits in both and would poison the sum.
            legal = mt.bool()
            kl = torch.where(legal, base_p * (base_lp - abl_lp),
                             torch.zeros_like(base_p)).sum(dim=-1)
            changed = (base_logits.argmax(dim=-1) != abl_logits.argmax(dim=-1))
            dv = (base_v.squeeze(-1) - abl_v.squeeze(-1))

        out.positions += len(o)
        out.argmax_changed += int(changed.sum().item())
        out.kl.extend([float(x) for x in kl.cpu().numpy()])
        out.value_delta.extend([float(x) for x in dv.cpu().numpy()])
    return out


def constant_columns(obs: np.ndarray, lo: int, hi: int, tol: float = 1e-9) -> int:
    """How many columns in a slice never vary across the sample -- pure dead weight."""
    window = obs[:, lo:hi]
    return int((window.max(axis=0) - window.min(axis=0) <= tol).sum())
