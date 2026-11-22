"""tools/scripts/average_weights.py: a uniform average of snapshots, refusing to mix architectures."""

from __future__ import annotations

import pytest
import torch

from ai.models.ladder_net import create_ladder_net
from tests.training.test_token_main import A8
from tools.scripts.average_weights import average


def _save(tmp_path, name: str, w: float, steps: int, shape=(2, 2)) -> str:
    p = str(tmp_path / name)
    torch.save({"lin.weight": torch.full(shape, w), "bn.num_batches_tracked": torch.tensor(steps)}, p)
    return p


def test_floats_are_averaged_and_integer_buffers_come_from_the_last(tmp_path) -> None:
    paths = [_save(tmp_path, "a.pt", 1.0, 10), _save(tmp_path, "b.pt", 3.0, 20)]
    out = average(paths)
    assert torch.allclose(out["lin.weight"], torch.full((2, 2), 2.0))
    assert int(out["bn.num_batches_tracked"]) == 20


def test_different_shapes_refuse_to_mix(tmp_path) -> None:
    paths = [_save(tmp_path, "a.pt", 1.0, 10), _save(tmp_path, "b.pt", 3.0, 20, shape=(3, 2))]
    with pytest.raises(ValueError):
        average(paths)


def test_an_averaged_sphere_trunk_is_projected_back_to_unit_rows(tmp_path) -> None:
    """Fix verification V-7.4: two sphere networks' unit rows average to shorter rows; the SWA / soup is projected back,
    as the live weights are after every step, and the rest of the average is untouched."""
    paths = []
    for seed in (0, 1):
        torch.manual_seed(seed)
        net = create_ladder_net("cpu", **{**A8, "entity_proj_dim": 64, "sphere_blocks": 2})
        p = str(tmp_path / f"s{seed}.pt")
        torch.save(net.state_dict(), p)
        paths.append(p)
    states = [torch.load(p, weights_only=True) for p in paths]
    out = average(paths)
    keys = [k for k in out if k.startswith("sphere.blocks.") and k.endswith((".w1.weight", ".w2.weight"))]
    assert len(keys) == 4
    for k in keys:
        assert torch.allclose(out[k].norm(dim=1), torch.ones(out[k].shape[0]), atol=1e-5), k
        raw = (states[0][k] + states[1][k]) / 2
        assert float(raw.norm(dim=1).min()) < 0.999, "the plain mean should be off the sphere"
        assert torch.allclose(out[k], torch.nn.functional.normalize(raw, dim=1), atol=1e-6)
    other = next(k for k in out if k.startswith("fusion_in") and out[k].is_floating_point())
    assert torch.allclose(out[other], (states[0][other] + states[1][other]) / 2)
