"""Guards for the per-side gradient-conflict and value-residual probes."""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from ai.eval.side_gradient_conflict import _flat_grad
from ai.eval.value_residual import HANDS, SCORING_CARDS, _hand_stats


def test_flat_grad_concatenates_in_parameter_order() -> None:
    net = torch.nn.Sequential(torch.nn.Linear(3, 4), torch.nn.Linear(4, 2))
    params = [p for p in net.parameters() if p.requires_grad]
    net(torch.ones(1, 3)).sum().backward()
    flat = _flat_grad(params)
    assert flat.numel() == sum(p.numel() for p in params)
    off = 0
    for p in params:
        grad = p.grad
        assert grad is not None, "every parameter should have received a gradient here"
        assert torch.equal(flat[off:off + p.numel()], grad.reshape(-1))
        off += p.numel()


def test_flat_grad_treats_missing_grad_as_zero() -> None:
    """A parameter that got no gradient must contribute zeros, not be skipped."""
    net = torch.nn.Sequential(torch.nn.Linear(3, 4), torch.nn.Linear(4, 2))
    params = [p for p in net.parameters() if p.requires_grad]
    net.zero_grad(set_to_none=True)
    flat = _flat_grad(params)
    assert flat.numel() == sum(p.numel() for p in params)
    assert torch.count_nonzero(flat) == 0


def test_chunked_accumulation_equals_full_batch_gradient() -> None:
    """The chunking in measure() must not change the gradient it reports."""
    torch.manual_seed(7)
    net = torch.nn.Linear(6, 5)
    x = torch.randn(64, 6)
    a = torch.randn(64)
    acts = torch.randint(0, 5, (64,))

    def logp_of(rows):
        return F.log_softmax(net(x[rows]), dim=-1).gather(
            1, acts[rows].unsqueeze(1)).squeeze(1)

    params = [p for p in net.parameters()]
    net.zero_grad(set_to_none=True)
    (-(logp_of(torch.arange(64)) * a)).mean().backward()
    full = _flat_grad(params).clone()

    net.zero_grad(set_to_none=True)
    for start in range(0, 64, 16):
        rows = torch.arange(start, min(start + 16, 64))
        (-(logp_of(rows) * a[rows]).sum() / 64).backward()
    chunked = _flat_grad(params).clone()

    assert torch.allclose(full, chunked, atol=1e-6), "chunked accumulation drifted"


def test_cosine_sign_convention() -> None:
    """Negative cosine must mean opposing directions -- the whole reading depends on it."""
    g = torch.randn(50)
    assert F.cosine_similarity(g.unsqueeze(0), (-g).unsqueeze(0)).item() < -0.99
    assert F.cosine_similarity(g.unsqueeze(0), g.unsqueeze(0)).item() > 0.99


def test_hand_stats_counts_both_hand_locations() -> None:
    import ts_engine as ts

    state = ts.GameState()
    ts.Engine.init_game(state, 21)
    hs = _hand_stats(state)
    # Counted independently, straight off the engine, including the KNOWN half.
    for side in ("US", "USSR"):
        expected = sum(1 for c in range(1, 111)
                       if state.get_card_location(c) in HANDS[side])
        assert hs["sizes"][side] == expected
        assert hs["scoring"][side] == sum(
            1 for c in SCORING_CARDS if state.get_card_location(c) in HANDS[side])
    assert len(SCORING_CARDS) == 7
    assert np.isfinite(hs["sizes"]["US"] + hs["sizes"]["USSR"])
