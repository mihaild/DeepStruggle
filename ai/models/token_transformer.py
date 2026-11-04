"""The token transformer as a LadderNet's main path (P41 J5, owner 2026-10-31; Jaxcalibur's shape).

195 tokens -- the global one, then 84 countries, then 110 cards -- through `layers` pre-RMSNorm blocks: full self-attention
shared by all of them, then an MLP with its own weights per token type (a mixture of experts by position: only a token's
own runs, so a type's parameters cost no compute on the others). Nothing here knows the game; `LadderNet` builds the tokens
and reads them.

With a memory (`memory=True`, the asymmetric layout; owner 2026-11-04) the active tokens are fewer -- the global token, the
countries and a few hand slots -- and the 110 card tokens become a read-only memory: normalised once per forward, projected
to keys and values by each block, attended by the active tokens beside their own, never updated (no FFN, no self-attention
among them). The forward then returns the active tokens followed by the normalised memory.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

#: Where each token type sits: the global token, then the countries, then the cards.
COUNTRY_START, CARD_START = 1, 85


class RMSNorm(nn.Module):
    """x / RMS(x) over the features, times a learned per-feature gain; no centring."""

    def __init__(self, dim: int, eps: float = 1e-8) -> None:
        super().__init__()
        self.gain = nn.Parameter(torch.ones(dim))
        self.eps = float(eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # The fused kernel: the same function as gain * x / sqrt(mean(x^2) + eps), one kernel instead of five.
        return F.rms_norm(x, (x.shape[-1],), self.gain, self.eps)


class TokenBlock(nn.Module):
    def __init__(self, dim: int, heads: int, mlp_mult: int, memory: bool = False) -> None:
        super().__init__()
        if dim % heads:
            raise ValueError(f"token width {dim} is not divisible by {heads} attention heads")
        self.heads = int(heads)
        self.norm_attn = RMSNorm(dim)
        self.qkv = nn.Linear(dim, 3 * dim)
        self.out = nn.Linear(dim, dim)
        self.norm_mlp = RMSNorm(dim)
        width = int(mlp_mult) * dim

        def mlp() -> nn.Sequential:
            return nn.Sequential(nn.Linear(dim, width), nn.GELU(), nn.Linear(width, dim))

        self.mlp_global, self.mlp_country, self.mlp_card = mlp(), mlp(), mlp()
        #: The memory's keys and values at this block, from the memory normalised once per forward.
        self.mem_kv = nn.Linear(dim, 2 * dim) if memory else None

    def forward(self, x: torch.Tensor, mem: torch.Tensor | None = None) -> torch.Tensor:
        b, n, d = x.shape

        def split(z: torch.Tensor) -> torch.Tensor:
            return z.view(b, z.shape[1], self.heads, d // self.heads).transpose(1, 2)

        q, k, v = self.qkv(self.norm_attn(x)).chunk(3, dim=-1)
        if mem is not None:
            assert self.mem_kv is not None
            km, vm = self.mem_kv(mem).chunk(2, dim=-1)
            k, v = torch.cat([k, km], dim=1), torch.cat([v, vm], dim=1)
        attn = F.scaled_dot_product_attention(split(q), split(k), split(v))
        x = x + self.out(attn.transpose(1, 2).reshape(b, n, d))
        y = self.norm_mlp(x)
        return x + torch.cat([self.mlp_global(y[:, :COUNTRY_START]),
                              self.mlp_country(y[:, COUNTRY_START:CARD_START]),
                              self.mlp_card(y[:, CARD_START:])], dim=1)


class TokenTransformer(nn.Module):
    def __init__(self, dim: int, layers: int, heads: int, mlp_mult: int, memory: bool = False) -> None:
        super().__init__()
        if layers < 1:
            raise ValueError(f"a token transformer needs a layer, got {layers}")
        self.blocks = nn.ModuleList(TokenBlock(dim, heads, mlp_mult, memory) for _ in range(int(layers)))
        self.norm = RMSNorm(dim)
        self.norm_mem = RMSNorm(dim) if memory else None
        # In the weights so the configuration is recovered from them: heads leave no shape of their own.
        self.register_buffer("heads_buf", torch.tensor(int(heads), dtype=torch.int64))
        #: Run the blocks through torch.compile (--ladder-token-compile): a plain attribute, not state, so the weights
        #: and their names are the same either way; copies (the reference, pool snapshots) share the one compiled code.
        self.compiled = False

    def forward(self, x: torch.Tensor, mem: torch.Tensor | None = None) -> torch.Tensor:
        if (mem is None) != (self.norm_mem is None):
            raise ValueError("a token transformer built with a memory needs one at every call, and one built without "
                             "takes none")
        return _run_compiled(self, x, mem) if self.compiled else _run(self, x, mem)


def _run(net: TokenTransformer, x: torch.Tensor, mem: torch.Tensor | None) -> torch.Tensor:
    m = None if net.norm_mem is None or mem is None else net.norm_mem(mem)
    for block in net.blocks:
        x = block(x, m)
    return net.norm(x) if m is None else torch.cat([net.norm(x), m], dim=1)


#: One compiled function for every TokenTransformer: dynamo reads a module's parameters as inputs, so the learner,
#: its reference and its pool snapshots share the code instead of each compiling its own.
_run_compiled = torch.compile(_run)
