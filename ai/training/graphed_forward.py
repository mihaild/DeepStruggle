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
stream (fixed then by a capture stream per graph, since replaced by the one-stream design
below), and a GPU deadlock in split-K kernels once the pool wrapped after ~16 captures (E4-42,
E4-44; see nash_pg._graphed_forward). Replayed one after the other they are safe whatever they
share.

**All of a cache's graphs are captured on ONE stream, and each capture warms up on that stream
first.** cuBLAS keeps a workspace per (handle, stream), allocated on the stream's first cuBLAS call.
Warming up on a side stream left that first call inside the capture, so the workspace came out of
the capturing graph's private memory pool -- and when the pool wrapped, a later graph captured on
the same stream baked in the same address. Dropping the first graph (the opponent pool evicting its
member) freed that memory while the later graph kept writing to it: `CUDA error: unspecified launch
failure` 16-30 iterations after a snapshot (E4-48-05-160M.11, E4-49-11, the rounding demo), about
one capture in 40 in a stress harness, and none with the workspace disabled or with drops disabled
(research/log/training_throughput_cpu.md). Warming up on the capture stream allocates its workspace
eagerly, from the ordinary allocator, before any capture; every graph then shares that one
workspace, which is safe because replays are serial, and dropping a graph frees nothing another
graph uses.

A graph reads the network's parameters by address. The optimiser updates them in place, so a
replay always sees the current weights, and `load_state_dict` copies into the existing tensors.
Anything that REPLACES a parameter or buffer tensor (a `.to()` onto another device, assigning a new
`nn.Parameter`) would leave the graph reading freed memory; `stale()` detects that and the caller
re-captures. `stale()` walks every parameter and buffer, which on M2d cost ~50 ms per 128-step
rollout when `GraphCache.get` ran it on every step (research/log/P26_quick_screen.md). Nothing
replaces a tensor in the middle of a rollout -- snapshots, pool admissions and resumes all happen
between iterations -- so `GraphCache` checks each network once per `begin_rollout()`.
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
                 device: torch.device, warmup: int = 3,
                 stream: Optional[torch.cuda.Stream] = None) -> None:
        if device.type != "cuda":
            raise ValueError("GraphedForward needs a CUDA device")
        if net.training:
            raise ValueError("capture the network in eval mode")
        self.net = net
        self.batch = int(batch)
        self.static_obs = torch.zeros((batch, obs_dim), dtype=torch.float32, device=device)
        self.static_mask = torch.ones((batch, action_dim), dtype=torch.uint8, device=device)
        self._ptrs = self._current_ptrs()
        # Warm up ON the capture stream, never on a side stream: this first eager run is what
        # allocates the stream's cuBLAS workspace, and it must happen outside the graph's private
        # pool (see the module docstring). A GraphCache passes one stream for all its graphs.
        self._capture_stream = stream if stream is not None else torch.cuda.Stream(device=device)
        self._capture_stream.wait_stream(torch.cuda.current_stream(device))
        with torch.cuda.stream(self._capture_stream), torch.no_grad():
            for _ in range(warmup):          # allocator and library workspaces, off the graph
                net(self.static_obs, self.static_mask)
        torch.cuda.current_stream(device).wait_stream(self._capture_stream)
        self.graph = torch.cuda.CUDAGraph()
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
        #: Networks whose graph was checked against `stale()` since the last `begin_rollout()`.
        self._checked: set[int] = set()
        #: The one capture stream every graph of this cache is captured on (module docstring).
        self._stream = torch.cuda.Stream(device=device)

    def begin_rollout(self) -> None:
        """Call before each rollout: every network's graph is checked for staleness again at its
        next `get()`, and not again until the next `begin_rollout()`."""
        self._checked.clear()

    def get(self, net: nn.Module) -> GraphedForward:
        g: Optional[GraphedForward] = self._graphs.get(id(net))
        if g is None or g.net is not net or (id(net) not in self._checked and g.stale()):
            g = GraphedForward(net, self.batch, self.obs_dim, self.action_dim, self.device,
                               stream=self._stream)
            self._graphs[id(net)] = g
        self._checked.add(id(net))
        return g

    def retain(self, nets: List[nn.Module]) -> None:
        keep = {id(n) for n in nets}
        for k in [k for k in self._graphs if k not in keep]:
            del self._graphs[k]
            self._checked.discard(k)
