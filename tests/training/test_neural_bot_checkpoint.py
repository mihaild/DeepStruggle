"""NeuralBot loads a checkpoint through the architecture-detecting loader.

It used to pick V1 or V2 by weight name itself, so a LadderNet checkpoint -- every E7 model --
was rebuilt as a V1 ColdWarNet and refused by load_state_dict, and `tools/play_match.py` could not
play a match between two current checkpoints. Built from a freshly initialised network saved to a
temp file, since no checkpoint is committed.
"""

from __future__ import annotations

from typing import Any, Dict

import pytest
import torch

from ai.models.ladder_net import create_ladder_net

#: A small shallow-trunk rung (the E7 recipe's structure at toy width).
CFG: Dict[str, Any] = dict(
    input_mode="grouped", aggregation="flatten", entity_dim=16, entity_proj_dim=256,
    card_self_attention=False, cross_attention=False, per_entity_heads=64, head_context=True,
    head_static=True, head_entities="country", head_center=True, identity_dim=0, drop_static=True,
    hidden_dim=64, num_res_blocks=0, num_attn_heads=4, card_lookup=False, card_lookup_heads=0,
    card_lookup_dim=0, card_lookup_identity_dim=0, categorical_value=False)


def test_neural_bot_loads_a_ladder_checkpoint(tmp_path: Any) -> None:
    from bot.neural_bot import NeuralBot

    net = create_ladder_net("cpu", **CFG)
    path = tmp_path / "snapshot_1000steps.pt"
    torch.save(net.state_dict(), path)
    bot = NeuralBot("US", model_path=str(path), device="cpu")
    assert type(bot.model).__name__ == "LadderNet"
    for name, t in net.state_dict().items():
        assert torch.equal(t, bot.model.state_dict()[name]), name


def test_neural_bot_refuses_a_feature_checkpoint(tmp_path: Any) -> None:
    import ts_engine as ts

    from bot.neural_bot import NeuralBot

    net = create_ladder_net("cpu", **{**CFG, "obs_features": int(ts.OBS_FEATURE_OPS_BUDGET)})
    path = tmp_path / "snapshot_1000steps.pt"
    torch.save(net.state_dict(), path)
    with pytest.raises(ValueError, match="feature blocks"):
        NeuralBot("US", model_path=str(path), device="cpu")
