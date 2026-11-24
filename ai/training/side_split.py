"""A side-split network's forward at one copy's cost per row, where shapes must be static (--ladder-side-nets A39,
--ladder-side-heads A38).

A side-split network routes each row through its own side's copy (I_AM_US). Eagerly it splits the batch by side, but
inside a fixed-shape region -- the update's torch.compile, the rollout's CUDA graphs -- a split is a data-dependent
shape, so `LadderNet._routed` / `_per_side` run every row through BOTH copies there and keep each row's own side's
output (`static_routing`): the same function at twice the copied part's cost (S2, A39, trained at 86k steps/s alone
against A8's 118k).

Here the split happens outside the fixed-shape region instead: each side's rows are gathered, padded with copies of
one of them up to a bucket size (so the region sees a handful of shapes), run through that side's copy alone
(`LadderNet.side_forward`) and scattered back, the padding dropped. Every row's output is its own side's copy's, as
before; the padded rows' outputs are discarded, so they enter no loss and no gradient. Nothing in the forward mixes
rows (no batch statistics; dropout aside, whose masks the compiled path never matched anyway), so outputs and
gradients are today's up to floating-point summation order.

`SplitForward` is the update's compiled call (`NashPG._update_net` / `_update_method`). The per-side part is compiled
once per side with its batch dimension marked dynamic: max-autotune-no-cudagraphs on a 16,384-row A39 minibatch ran
forward+backward at 16.1 ms dynamic, 15.7 ms with a static compile per bucket (which compiled for 826 s against 436 s
and piles bucket after bucket onto dynamo's recompile limit), and the both-copies path at 26.7 ms (2026-11-24, GPU
shared with two trainers).

The rollout keeps its whole-network CUDA graph (every row through both copies). A per-side graph pair was measured
and is slower there: the rollout forward is launch-bound, so a 256-row graph costs about what a 512-row one does and
two of them, plus the gather and scatter, cost more (0.83 ms a step against 0.65 at 512 envs, A8 0.46).
"""

from __future__ import annotations

import types
from typing import Any, Callable, List, Mapping, Tuple

import torch

from ai.models.head_registry import is_side_split
from ai.models.ladder_net import SIDE_ROW_FNS, LadderNet

#: Rows a side's part of an update minibatch is padded to a multiple of. A 16,384-row minibatch splits about evenly
#: (binomial: +-64 rows a side), so a side's part falls in two or three buckets, and the padding costs ~1.5%.
UPDATE_BUCKET = 256


def bucket_size(n: int, step: int) -> int:
    """`n` rounded up to a multiple of `step` (at least `step`)."""
    if step < 1:
        raise ValueError(f"bucket step must be positive, got {step}")
    return max(step, -(-int(n) // step) * step)


def padded_rows(idx: torch.Tensor, size: int) -> torch.Tensor:
    """`idx` padded to `size` entries with copies of its first (a valid row, whose output is discarded)."""
    n = int(idx.numel())
    if size < n or n == 0:
        raise ValueError(f"cannot pad {n} rows to {size}")
    return idx if size == n else torch.cat([idx, idx[:1].expand(size - n)])


def _head(x: Any, n: int) -> Any:
    """The first `n` rows of a tensor, or of each tensor of a tuple (None stays None)."""
    if isinstance(x, tuple):
        return tuple(_head(t, n) for t in x)
    return None if x is None else x[:n]


def _merge(parts: List[Tuple[torch.Tensor, Any]]) -> Any:
    """The parts' rows (index tensor, outputs) put back in batch order."""
    if len(parts) == 1:
        return parts[0][1]          # one side: its rows are the batch, in order
    (i0, a), (i1, b) = parts
    order = torch.argsort(torch.cat([i0, i1]))
    return _cat_select(a, b, order)


def _cat_select(a: Any, b: Any, order: torch.Tensor) -> Any:
    if isinstance(a, tuple):
        return tuple(_cat_select(x, y, order) for x, y in zip(a, b))
    if a is None:
        return None
    return torch.cat([a, b], dim=0).index_select(0, order)


class SplitForward:
    """`net.<method>(obs, mask)` for a side-split LadderNet, each side's rows through that side's copy only, padded to a
    multiple of `bucket` rows, the per-side call compiled with torch.compile(**compile) (None: eager).

    The per-side call's batch dimension is marked dynamic, so torch.compile builds one graph per side (and grad mode);
    the buckets keep the shapes it runs at few.
    """

    def __init__(self, net: LadderNet, method: str, compile: Mapping[str, Any] | None,
                 bucket: int = UPDATE_BUCKET) -> None:
        if not is_side_split(net):
            raise ValueError("SplitForward is for side-split networks (--ladder-side-heads / --ladder-side-nets)")
        if method not in SIDE_ROW_FNS:
            raise ValueError(f"no per-side implementation of {method!r}; known: {sorted(SIDE_ROW_FNS)}")
        if bucket < 2:
            raise ValueError(f"bucket must be at least 2 rows (torch.compile specialises a batch of 1), got {bucket}")
        self.net = net
        self.method = method
        self.bucket = int(bucket)
        self.compiled = compile is not None
        fn = SIDE_ROW_FNS[method]
        self._parts: List[Callable[..., Any]] = []
        for s in (0, 1):
            part = _side_part(net.side_net(s), fn, f"{method}_side{s}")
            self._parts.append(torch.compile(part, **compile) if compile is not None else part)

    def __call__(self, obs: torch.Tensor, mask: torch.Tensor | None = None) -> Any:
        us = self.net.is_us(obs)
        sides = (torch.nonzero(us).squeeze(1), torch.nonzero(~us).squeeze(1))
        parts: List[Tuple[torch.Tensor, Any]] = []
        for s, idx in enumerate(sides):
            n = int(idx.numel())
            if n == 0:
                continue
            rows = padded_rows(idx, bucket_size(n, self.bucket))
            o = obs.index_select(0, rows)
            m = None if mask is None else mask.index_select(0, rows)
            if self.compiled:
                torch._dynamo.mark_dynamic(o, 0)
                if m is not None:
                    torch._dynamo.mark_dynamic(m, 0)
            with self.net.fixed_side(s):
                out = self._parts[s](o, m)
            parts.append((idx, _head(out, n)))
        return _merge(parts)


def _side_part(net: LadderNet, fn: Callable[..., Any], name: str) -> Callable[..., Any]:
    """`fn(net, obs, mask)` as a function of (obs, mask): the per-side call a SplitForward compiles. Each gets a code
    object of its own (`name`): torch.compile keeps its cache per code object, and one shared by every side, method
    and network would pile their entries onto one cache and reach dynamo's recompile limit, past which it runs eager."""
    def part(obs: torch.Tensor, mask: torch.Tensor | None) -> Any:
        return fn(net, obs, mask)
    return types.FunctionType(part.__code__.replace(co_name=name, co_qualname=name), part.__globals__, name,
                              part.__defaults__, part.__closure__)
