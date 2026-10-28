"""SimbaV2's hyperspherical trunk (Lee et al., ICML 2025), P38 rung 2 (research/plans/P38_stability_ladder.md).

The two failures the record measured in deeper trunks are removed by construction: the residual stream cannot grow,
because it lives on the unit sphere and is renormalised after every block (A22's stream grew to RMS 37, A25's
features' effective rank fell to ~22 of 768), and the blocks' weights cannot grow, because every row is projected
back to unit norm after each optimiser step (the measured head-weight growth, P37a, is the same disease in the heads).

    h_0      = l2(stem)                                   the fusion's LayerNorm + GELU output, on the sphere
    h~       = l2(W2 ReLU((W1 h) * s))                    W1: d -> e*d, W2: e*d -> d, unit-norm rows, no bias
    h        = l2(h + alpha * (h~ - h))                   alpha per dimension, init 1 / (L + 1)
    out      = c * sqrt(d) * h_L                          the heads read per-element RMS c (default 1)
    out      = |stem| * h_L                               with carry_norm (P39 arm 6): the stem's own norm

`s` (per hidden unit, init 1) only reweights the hidden units: ReLU is scale-equivariant and the branch is
renormalised, so a uniform scale changes nothing. A small alpha at initialisation keeps a fresh deep trunk close to
its stem -- what depth-muP's 1/sqrt(L) residual scaling also buys. Not taken from SimbaV2: RSNorm and the input
shift (our observation is 0/1 and fractions by design; the fusion is the stem), the categorical critic and reward
scaling (continuous-control return scales).
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class SphereBlock(nn.Module):
    """One SimbaV2 block on unit-norm rows of width `dim`."""

    def __init__(self, dim: int, expansion: int, num_blocks: int) -> None:
        super().__init__()
        self.w1 = nn.Linear(dim, expansion * dim, bias=False)
        self.w2 = nn.Linear(expansion * dim, dim, bias=False)
        self.scaler = nn.Parameter(torch.ones(expansion * dim))
        self.alpha = nn.Parameter(torch.full((dim,), 1.0 / (num_blocks + 1)))
        self.project()

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        branch = F.normalize(self.w2(F.relu(self.w1(h) * self.scaler)), dim=-1)
        return F.normalize(h + self.alpha * (branch - h), dim=-1)

    @torch.no_grad()
    def project(self) -> None:
        """Every weight row back to unit L2 norm (rows are output units, so each unit computes a cosine)."""
        for lin in (self.w1, self.w2):
            lin.weight.copy_(F.normalize(lin.weight, dim=1))


class SphereTrunk(nn.Module):
    """`num_blocks` SphereBlocks between the stem and the heads (see the module docstring)."""

    def __init__(self, dim: int, num_blocks: int, expansion: int = 4, out_scale: float = 1.0,
                 carry_norm: bool = False) -> None:
        super().__init__()
        if num_blocks <= 0 or expansion <= 0 or out_scale <= 0.0:
            raise ValueError(f"a sphere trunk needs blocks, an expansion and a positive output scale, got "
                             f"{num_blocks} / {expansion} / {out_scale}")
        self.dim = int(dim)
        self.expansion = int(expansion)
        self.blocks = nn.ModuleList([SphereBlock(dim, expansion, num_blocks) for _ in range(num_blocks)])
        # The heads' input RMS. 1 by default; A8's heads read ~0.55 (60M) to ~0.37 (400M), and at 1 the sphere trunk's
        # per-update KL sat at ~3x the control's whatever the learning rate (P38 rung 2b). A buffer only off 1, so the
        # first sphere checkpoints load unchanged and the value is recovered from the weights.
        self.out_scale = float(out_scale)
        if self.out_scale != 1.0:
            self.register_buffer("out_scale_buf", torch.tensor(self.out_scale, dtype=torch.float32))
        # --ladder-sphere-carry-norm (P39 arm 6): the output keeps the stem's own norm instead of a fixed RMS, so the
        # blocks only rotate the trunk's output -- and with every alpha at 0 the trunk is the identity, which is what
        # lets it be inserted into a trained network in place (`identity_init`). A buffer only when on.
        self.carry_norm = bool(carry_norm)
        if self.carry_norm:
            if self.out_scale != 1.0:
                raise ValueError("carry_norm keeps the stem's norm; it takes no output scale")
            self.register_buffer("carry_norm_buf", torch.tensor(1.0, dtype=torch.float32))

    def forward(self, stem: torch.Tensor) -> torch.Tensor:
        h = F.normalize(stem, dim=-1)
        for block in self.blocks:
            h = block(h)
        if self.carry_norm:
            return h * stem.norm(dim=-1, keepdim=True)
        return h * (self.out_scale * math.sqrt(self.dim))

    @torch.no_grad()
    def identity_init(self) -> None:
        """Every block's alpha to 0: with carry_norm the trunk then returns its input exactly (grown in place); the
        alphas take gradient at once and the blocks' weights follow as the alphas leave 0."""
        if not self.carry_norm:
            raise ValueError("only a carry_norm sphere trunk is the identity at alpha 0")
        for block in self.blocks:
            assert isinstance(block, SphereBlock)
            block.alpha.zero_()

    @torch.no_grad()
    def project(self) -> None:
        for block in self.blocks:
            assert isinstance(block, SphereBlock)
            block.project()
