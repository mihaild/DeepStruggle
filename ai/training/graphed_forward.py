"""CUDA-graph replay of a policy network's rollout forward, at the rollout's fixed batch shape.

The rollout forward is launch-bound on the host: M2d issues ~317 kernels per call, and its forward
over 512 positions costs about the same as over 16 (research/log/training_throughput_cpu.md). A CUDA
graph records those kernels once and replays them with a single launch. It replays the SAME
kernels on the same inputs, so its outputs are bitwise identical to eager -- unlike
`torch.compile`, which fuses and reorders arithmetic and is held back by P11's adoption gates for
that reason.

**Never replay two of these graphs concurrently on different streams.** A graph bakes in the cuBLAS
workspace of the stream it was captured on, and `torch.cuda.Stream()` comes from a pool of 32 per
device, reused round-robin. Once the pool wraps, two graphs can hold the same workspace, and
concurrent replays then race on it: wrong logits when both captures used the default capture
stream (fixed by the per-graph capture stream below), and a GPU deadlock in split-K kernels once
the pool wrapped after ~16 captures (E4-42, E4-44; see nash_pg._graphed_forward). Replayed one
after the other they are safe whatever they share.

A graph reads the network's parameters by address. The optimiser updates them in place, so a
replay always sees the current weights, and `load_state_dict` copies into the existing tensors.
Anything that REPLACES a parameter or buffer tensor (a `.to()` onto another device, assigning a new
`nn.Parameter`) would leave the graph reading freed memory; `stale()` detects that and the caller
re-captures.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn


class GraphedForward:
    """`net(obs, mask)` under `torch.no_grad()`, captured once for a fixed batch and replayed.

    Use as: `load(obs, mask)`, then `replay()` on whichever stream should run it, then read
    `outputs()` (clones, so they survive the next replay). Capture and replay need eval mode: a
    training-mode dropout would be baked into the graph with a fixed random mask.
    """

    def __init__(self, net: nn.Module, batch: int, obs_dim: int, action_dim: int,
                 device: torch.device, warmup: int = 3) -> None:
        if device.type != "cuda":
            raise ValueError("GraphedForward needs a CUDA device")
        if net.training:
            raise ValueError("capture the network in eval mode")
        self.net = net
        self.batch = int(batch)
        self.static_obs = torch.zeros((batch, obs_dim), dtype=torch.float32, device=device)
        self.static_mask = torch.ones((batch, action_dim), dtype=torch.uint8, device=device)
        self._ptrs = self._current_ptrs()
        side = torch.cuda.Stream(device=device)
        side.wait_stream(torch.cuda.current_stream(device))
        with torch.cuda.stream(side), torch.no_grad():
            for _ in range(warmup):          # allocator and library workspaces, off the graph
                net(self.static_obs, self.static_mask)
        torch.cuda.current_stream(device).wait_stream(side)
        self.graph = torch.cuda.CUDAGraph()
        # A capture stream of its own, so a capture does not run on the caller's stream. This is
        # NOT a guarantee of a private cuBLAS workspace: streams come from a reused pool of 32, so
        # graphs must still never be replayed concurrently (see the module docstring).
        self._capture_stream = torch.cuda.Stream(device=device)
        with torch.cuda.graph(self.graph, stream=self._capture_stream), torch.no_grad():
            self.static_out: Tuple[torch.Tensor, ...] = tuple(
                net(self.static_obs, self.static_mask))

    def _current_ptrs(self) -> List[int]:
        return ([p.data_ptr() for p in self.net.parameters()]
                + [b.data_ptr() for b in self.net.buffers()])

    def stale(self) -> bool:
        """True when a parameter or buffer tensor was replaced since capture."""
        return self._current_ptrs() != self._ptrs

    def load(self, obs: torch.Tensor, mask: torch.Tensor) -> None:
        if obs.shape[0] != self.batch:
            raise ValueError(f"captured for batch {self.batch}, got {obs.shape[0]}")
        self.static_obs.copy_(obs)
        self.static_mask.copy_(mask)

    def replay(self) -> None:
        if self.net.training:
            raise RuntimeError("replaying a graph captured in eval mode while the network trains")
        self.graph.replay()

    def outputs(self) -> Tuple[torch.Tensor, ...]:
        return tuple(t.clone() for t in self.static_out)


class GraphCache:
    """One `GraphedForward` per network, captured on first use and re-captured when stale.

    Keyed by `id(net)`. `retain(nets)` drops the graphs of networks no longer in use -- the
    opponent pool evicts members -- so their private memory pools are released.
    """

    def __init__(self, batch: int, obs_dim: int, action_dim: int, device: torch.device) -> None:
        self.batch, self.obs_dim, self.action_dim, self.device = batch, obs_dim, action_dim, device
        self._graphs: Dict[int, GraphedForward] = {}

    def get(self, net: nn.Module) -> GraphedForward:
        g: Optional[GraphedForward] = self._graphs.get(id(net))
        if g is None or g.net is not net or g.stale():
            g = GraphedForward(net, self.batch, self.obs_dim, self.action_dim, self.device)
            self._graphs[id(net)] = g
        return g

    def retain(self, nets: List[nn.Module]) -> None:
        keep = {id(n) for n in nets}
        for k in [k for k in self._graphs if k not in keep]:
            del self._graphs[k]
