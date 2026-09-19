"""The P21 ladder backbone: every rung builds, and every illegal combination is refused.

`research/plans/P21_architecture_ladder.md` needs one backbone whose structure is configuration,
so each mechanism gets a matched control. The risk that comes with that is the one this
repository has already paid for five times: a defaulted variant argument makes a model silently
become a different architecture from the one the metadata claims, and nothing fails — a
mismatched model returns a number instead of raising.

So these tests pin two things equally: that the rungs work, and that the *refusals* work.
"""

from __future__ import annotations

from typing import cast

import pytest
import torch

from ai.models.coldwar_net_v2 import ColdWarNetV2, create_like
from ai.models.ladder_net import AGGREGATIONS, INPUT_MODES, LadderNet
from bindings.action_encoder import ActionEncoder

BASE = dict(hidden_dim=128, num_res_blocks=1, entity_proj_dim=64,
            num_attn_heads=4, categorical_value=False)
D = 8
A = ActionEncoder.FLAT_ACTION_SIZE


def rung(**over):
    cfg = dict(input_mode="entity", aggregation="flatten", entity_dim=D,
               card_self_attention=False, cross_attention=False,
               per_entity_heads=0, identity_dim=0, drop_static=False)
    cfg.update(over)
    return LadderNet(**BASE, **cfg)


RUNGS = {
    "M0": dict(input_mode="flat", drop_static=True),
    "M1": dict(input_mode="grouped", drop_static=True),
    "M2": dict(),
    "M3": dict(card_self_attention=True),
    "M4": dict(card_self_attention=True, cross_attention=True),
    "M5": dict(card_self_attention=True, cross_attention=True, aggregation="pool"),
    "M4-pe": dict(card_self_attention=True, cross_attention=True, per_entity_heads=16),
    "M5-pe": dict(card_self_attention=True, cross_attention=True, aggregation="pool",
                  per_entity_heads=16),
    "M5-id": dict(card_self_attention=True, cross_attention=True, aggregation="pool",
                  identity_dim=8),
}


@pytest.mark.parametrize("name", sorted(RUNGS))
def test_every_rung_runs_a_forward_pass(name) -> None:
    model = rung(**RUNGS[name]).eval()
    obs = torch.randn(3, ColdWarNetV2.TOTAL_OBS_SIZE)
    mask = torch.ones(3, A, dtype=torch.uint8)
    with torch.no_grad():
        logits, v_win, v_vp = model.forward(obs, mask)
    assert logits.shape == (3, A)
    assert v_win.shape[0] == 3 and v_vp.shape[0] == 3
    assert torch.isfinite(logits).all()


@pytest.mark.parametrize("name", sorted(RUNGS))
def test_create_like_rebuilds_a_loadable_twin(name) -> None:
    """A frozen evaluation copy must have the same state dict.

    Two runs have died at their first snapshot because a copy was built by listing arguments and
    one was left at a factory default. `create_like` rebuilds from `ladder_config()` instead.
    """
    model = rung(**RUNGS[name])
    twin = create_like(model)
    assert isinstance(twin, LadderNet)
    twin.load_state_dict(model.state_dict(), strict=True)
    assert twin.ladder_config() == model.ladder_config()


def test_the_mask_is_applied() -> None:
    model = rung().eval()
    obs = torch.randn(2, ColdWarNetV2.TOTAL_OBS_SIZE)
    mask = torch.zeros(2, A, dtype=torch.uint8)
    mask[:, 5] = 1
    with torch.no_grad():
        logits, _, _ = model.forward(obs, mask)
    assert logits[:, 5].gt(-1e8).all()
    assert logits[:, 6].lt(-1e8).all()


def test_a_wrong_width_observation_raises_rather_than_being_misread() -> None:
    model = rung().eval()
    with pytest.raises(ValueError, match="floats wide"):
        model.forward(torch.randn(2, ColdWarNetV2.TOTAL_OBS_SIZE - 1),
                      torch.ones(2, A, dtype=torch.uint8))


# --------------------------------------------------------------- the refusals

def test_per_entity_heads_refused_without_tokens() -> None:
    with pytest.raises(ValueError, match="no entity tokens"):
        rung(input_mode="flat", per_entity_heads=16)


def test_attention_refused_without_tokens() -> None:
    with pytest.raises(ValueError, match="no entity tokens"):
        rung(input_mode="grouped", card_self_attention=True)


def test_identity_refused_on_a_positional_reader() -> None:
    """Position already identifies the entity, so identity there is pure redundancy."""
    with pytest.raises(ValueError, match="already identifies"):
        rung(input_mode="flat", identity_dim=8)


def test_drop_static_refused_with_a_shared_encoder() -> None:
    """The static slots are how a position-blind shared encoder knows what it is looking at."""
    with pytest.raises(ValueError, match="positional readers"):
        rung(input_mode="entity", drop_static=True)


@pytest.mark.parametrize("bad", ["meanmax", "sum", ""])
def test_unknown_aggregation_refused(bad) -> None:
    with pytest.raises(ValueError, match="aggregation must be one of"):
        rung(aggregation=bad)


@pytest.mark.parametrize("bad", ["tokens", "mlp", ""])
def test_unknown_input_mode_refused(bad) -> None:
    with pytest.raises(ValueError, match="input_mode must be one of"):
        rung(input_mode=bad)


def test_the_axis_vocabularies_are_what_the_plan_names() -> None:
    assert set(INPUT_MODES) == {"flat", "grouped", "entity"}
    assert set(AGGREGATIONS) == {"flatten", "pool"}


# ------------------------------------------------------- position vs. pooling

def test_flatten_keeps_position_and_pool_discards_it() -> None:
    """The property the whole ladder turns on, asserted directly.

    Permuting the 84 country rows must change a flattened model's output and must NOT change a
    pooled one's, because mean and max are symmetric.
    """
    obs = torch.randn(1, ColdWarNetV2.TOTAL_OBS_SIZE)
    board = obs[:, :ColdWarNetV2.BOARD_SIZE].view(1, 84, ColdWarNetV2.BOARD_FEATURES)
    perm = torch.randperm(84)
    shuffled = obs.clone()
    shuffled[:, :ColdWarNetV2.BOARD_SIZE] = board[:, perm].reshape(1, -1)

    for aggregation, should_change in (("flatten", True), ("pool", False)):
        model = rung(aggregation=aggregation).eval()
        with torch.no_grad():
            # extract_features returns the trunk alone unless attention weights are asked for
            a = cast(torch.Tensor, model.extract_features(obs))
            b = cast(torch.Tensor, model.extract_features(shuffled))
        changed = not torch.allclose(a, b, atol=1e-5)
        assert changed is should_change, (
            f"{aggregation}: permuting countries {'did not change' if should_change else 'changed'}"
            f" the trunk, which contradicts what this aggregation is supposed to do")


# ------------------------------------------------------------- the CLI contract

def _args(argv):
    from ai.training.train import build_parser
    return build_parser().parse_args(argv)


FULL_CLI = ["--arch", "ladder", "--ladder-input-mode", "entity",
            "--ladder-aggregation", "flatten", "--ladder-entity-dim", "16",
            "--ladder-entity-proj-dim", "256", "--ladder-hidden-dim", "384",
            "--ladder-res-blocks", "4", "--ladder-card-self-attention"]


def test_cli_builds_the_configuration_it_was_given() -> None:
    from ai.training.train import _ladder_config
    cfg = _ladder_config(_args(FULL_CLI))
    assert cfg is not None
    assert cfg["input_mode"] == "entity"
    assert cfg["aggregation"] == "flatten"
    assert cfg["entity_dim"] == 16
    assert cfg["card_self_attention"] is True
    assert cfg["cross_attention"] is False


def test_cli_refuses_an_underspecified_ladder() -> None:
    """No silent defaults: --arch ladder must name every axis, and the error lists them."""
    from ai.training.train import _ladder_config
    with pytest.raises(SystemExit) as e:
        _ladder_config(_args(["--arch", "ladder"]))
    msg = str(e.value)
    for flag in ("--ladder-input-mode", "--ladder-aggregation", "--ladder-entity-dim",
                 "--ladder-hidden-dim", "--ladder-res-blocks"):
        assert flag in msg


def test_cli_refuses_ladder_flags_under_another_architecture() -> None:
    """Silently ignoring them would train a different architecture from the one requested."""
    from ai.training.train import _ladder_config
    with pytest.raises(SystemExit, match="only apply to --arch ladder"):
        _ladder_config(_args(["--arch", "v2", "--ladder-aggregation", "pool"]))


def test_a_non_ladder_run_is_unaffected() -> None:
    from ai.training.train import _ladder_config
    assert _ladder_config(_args(["--arch", "v2"])) is None


def test_the_cli_configuration_builds_a_model_that_round_trips() -> None:
    from ai.models.ladder_net import create_ladder_net
    from ai.training.train import _ladder_config
    cfg = _ladder_config(_args(FULL_CLI))
    assert cfg is not None
    model = create_ladder_net(torch.device("cpu"), categorical_value=False, **cfg)
    assert model.ladder_config()["input_mode"] == "entity"
    create_like(model).load_state_dict(model.state_dict(), strict=True)
