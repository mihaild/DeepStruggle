"""The legal-action mask fills with masked_fill rather than torch.where against a scalar tensor.

The two produce the same values; the change exists because building the scalar on the device was a
blocking host-to-device copy on every forward pass. This pins that the values did not move.
"""

from __future__ import annotations

import torch


def test_masked_fill_equals_the_torch_where_it_replaced() -> None:
    gen = torch.Generator().manual_seed(0)
    logits = torch.randn(64, 220, generator=gen)
    mask = torch.rand(64, 220, generator=gen) > 0.7
    old = torch.where(mask, logits, torch.tensor(-1e9, dtype=logits.dtype))
    new = logits.masked_fill(~mask, -1e9)
    assert torch.equal(old, new)
