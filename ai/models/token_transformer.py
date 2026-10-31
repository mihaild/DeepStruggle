"""The token transformer as a LadderNet's main path (P41 J5, owner 2026-10-31; Jaxcalibur's shape).

195 tokens -- the global one, then 84 countries, then 110 cards -- through `layers` pre-RMSNorm blocks: full self-attention
shared by all of them, then an MLP with its own weights per token type (a mixture of experts by position: only a token's
own runs, so a type's parameters cost no compute on the others). Nothing here knows the game; `LadderNet` builds the tokens
and reads them.
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
        return self.gain * x / torch.sqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)


class TokenBlock(nn.Module):
    def __init__(self, dim: int, heads: int, mlp_mult: int) -> None:
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, n, d = x.shape

        def split(z: torch.Tensor) -> torch.Tensor:
            return z.view(b, n, self.heads, d // self.heads).transpose(1, 2)

        q, k, v = self.qkv(self.norm_attn(x)).chunk(3, dim=-1)
        attn = F.scaled_dot_product_attention(split(q), split(k), split(v))
        x = x + self.out(attn.transpose(1, 2).reshape(b, n, d))
        y = self.norm_mlp(x)
        return x + torch.cat([self.mlp_global(y[:, :COUNTRY_START]),
                              self.mlp_country(y[:, COUNTRY_START:CARD_START]),
                              self.mlp_card(y[:, CARD_START:])], dim=1)


class TokenTransformer(nn.Module):
    def __init__(self, dim: int, layers: int, heads: int, mlp_mult: int) -> None:
        super().__init__()
        if layers < 1:
            raise ValueError(f"a token transformer needs a layer, got {layers}")
        self.blocks = nn.ModuleList(TokenBlock(dim, heads, mlp_mult) for _ in range(int(layers)))
        self.norm = RMSNorm(dim)
        # In the weights so the configuration is recovered from them: heads leave no shape of their own.
        self.register_buffer("heads_buf", torch.tensor(int(heads), dtype=torch.int64))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for block in self.blocks:
            x = block(x)
        return self.norm(x)
