"""The P21 ladder backbone: every rung builds, and every illegal combination is refused.

`research/plans/P21_architecture_ladder.md` needs one backbone whose structure is configuration,
so each mechanism gets a matched control. The risk that comes with that is the one this
repository has already paid for five times: a defaulted variant argument makes a model silently
become a different architecture from the one the metadata claims, and nothing fails — a
mismatched model returns a number instead of raising.

So these tests pin two things equally: that the rungs work, and that the *refusals* work.
"""

from __future__ import annotations

from typing import Any, Dict, cast

import pytest
import torch

from ai.models.coldwar_net_v2 import ColdWarNetV2, create_like
from ai.models.ladder_net import AGGREGATIONS, INPUT_MODES, LadderNet
from bindings.action_encoder import ActionEncoder

BASE: Dict[str, Any] = dict(hidden_dim=128, num_res_blocks=1, entity_proj_dim=64,
                            num_attn_heads=4, categorical_value=False)
D = 8
A = ActionEncoder.FLAT_ACTION_SIZE


def rung(**over):
    """A rung at the test defaults. Any axis, including a BASE one, may be overridden."""
    cfg: Dict[str, Any] = dict(BASE)
    cfg.update(input_mode="entity", aggregation="flatten", entity_dim=D,
               card_self_attention=False, cross_attention=False,
               per_entity_heads=0, head_context=True, head_static=True,
               head_entities="both", identity_dim=0, drop_static=False)
    cfg.update(over)
    return LadderNet(**cfg)


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

def test_per_entity_heads_refused_on_a_flat_model() -> None:
    """`flat` never reshapes into entities, so there is nothing per-entity to read."""
    with pytest.raises(ValueError, match="never reshapes"):
        rung(input_mode="flat", per_entity_heads=16)


def test_per_entity_heads_work_on_raw_grouped_features() -> None:
    """The lookup mechanism without a shared encoder in front of it.

    A shared `Linear(26 -> d)` is a LOSSY restriction of the grouped projection, not an addition
    to it: each country's 26 hand-crafted slots are forced through one rank-d map before anything
    downstream sees them. `grouped` keeps every slot at a fixed offset, so `pe_country` can read
    country i's own raw features directly and the lookup is isolated from that compression.
    """
    model = rung(input_mode="grouped", drop_static=True, per_entity_heads=16).eval()
    assert model.raw_tokens is True
    obs = torch.randn(2, ColdWarNetV2.TOTAL_OBS_SIZE)
    with torch.no_grad():
        logits, _, _ = model.forward(obs, torch.ones(2, A, dtype=torch.uint8))
    assert logits.shape == (2, A) and torch.isfinite(logits).all()


def test_attention_refused_without_a_token_space() -> None:
    with pytest.raises(ValueError, match="no attention tokens"):
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


# ------------------------------------------------- the frozen-copy path (regression)

def test_a_ladder_model_gets_a_model_derived_frozen_copy() -> None:
    """The snapshot path must rebuild a LadderNet, not a default-width network.

    `generic_trainer.evaluate_and_log_snapshot` chose the frozen opponent copy with an ALLOW-LIST,
    `arch in ("v2", "mlp")`. Adding `ladder` to the CLI without adding it there sent a
    3.18M-parameter ladder model down the v1 branch, which built a 512-wide trunk for a 480-wide
    checkpoint and died at the first snapshot — hours into the run, not at startup. The guard is
    now inverted so only the legacy v1 backbone is special-cased.

    This test pins the property that actually matters: a copy made the way the trainer makes one
    loads the original's weights.
    """
    model = rung(input_mode="flat", drop_static=True, hidden_dim=480, entity_proj_dim=320)
    frozen = create_like(model)
    assert isinstance(frozen, LadderNet)
    frozen.load_state_dict(model.state_dict())          # strict by default
    assert frozen.ladder_config() == model.ladder_config()


def test_a_non_default_trunk_width_survives_the_copy() -> None:
    """The specific shape that failed: 480 rebuilt as 512."""
    model = rung(hidden_dim=480, entity_proj_dim=320)
    assert model.ladder_config()["hidden_dim"] == 480
    twin = create_like(model)
    assert isinstance(twin, LadderNet)
    assert twin.ladder_config()["hidden_dim"] == 480


# --------------------------------- checkpoint round trip through the real loader

@pytest.mark.parametrize("name", sorted(RUNGS))
def test_every_rung_survives_a_checkpoint_round_trip(name, tmp_path) -> None:
    """Save a rung, load it the way the tournament does, and get the same architecture back.

    Checkpoints here are bare state dicts and the architecture is detected by weight name. That
    detection did not know about LadderNet, so the first ladder checkpoint was rebuilt as a
    ColdWarNetV2 and the tournament died on shape mismatches. Two traps made it worse than a
    missing branch:

      * `lad_cross_attn.*` contains the substring `cross_attn`, so a cross-attention rung matched
        the v2 test and would have been rebuilt as v2 — loading most tensors and silently
        dropping the rest, which rates a different network than the one that trained.
      * `check_checkpoint_layout` reads the layout off `card_fc.0.weight`, which no ladder rung
        has, so every rung was refused until it learned to read `lad_*` widths instead.
    """
    from ai.models.ladder_net import ladder_config_from_state_dict
    from tools.lib.player_agent import NeuralAgent

    model = rung(**RUNGS[name], hidden_dim=192, entity_proj_dim=96)
    sd = model.state_dict()

    recovered = ladder_config_from_state_dict(dict(sd))
    assert recovered == model.ladder_config(), f"{name}: config not recoverable from weights"

    path = tmp_path / f"{name}.pt"
    torch.save(sd, path)
    agent = NeuralAgent.from_checkpoint(str(path), device="cpu")
    assert isinstance(agent.model, LadderNet), f"{name}: loaded as {type(agent.model).__name__}"
    assert agent.model.ladder_config() == model.ladder_config()


def test_a_non_ladder_checkpoint_is_not_claimed_by_the_ladder_detector() -> None:
    """The detector must return None for v2, or it would hijack every existing checkpoint."""
    from ai.models.coldwar_net_v2 import create_coldwar_net_v2
    from ai.models.ladder_net import ladder_config_from_state_dict
    v2 = create_coldwar_net_v2(torch.device("cpu"), categorical_value=False,
                               identity_dim=16, per_entity_heads=64, graph_layers=0,
                               self_transform=True)
    assert ladder_config_from_state_dict(dict(v2.state_dict())) is None


# ------------------------------- M2.5: identity for the head, and only the head

def test_head_identity_reaches_the_head_and_not_the_trunk() -> None:
    """M2.5 puts a learned per-entity vector into the per-entity head only.

    Identity is needed exactly where a function is SHARED across entities. With a positional
    trunk that is the per-entity head and nothing else: `pe_card` cannot tell 95 of 110 cards
    apart (The Voice of America, Colonial Rear Guards and Grain Sales to Soviets are one
    indistinguishable triple), while the trunk reads every card at its own offset and can.

    So the head's input widens by exactly `identity_dim` and the trunk's does not move at all.
    """
    common = dict(input_mode="grouped", drop_static=True, per_entity_heads=16)
    m2 = rung(**common, identity_dim=0)
    m25 = rung(**common, identity_dim=8)

    def in_w(model, name) -> int:
        layer = getattr(model, name)[0]
        assert isinstance(layer, torch.nn.Linear)
        return int(layer.in_features)

    assert in_w(m25, "pe_country") == in_w(m2, "pe_country") + 8
    assert in_w(m25, "pe_card") == in_w(m2, "pe_card") + 8
    assert in_w(m25, "fusion_in") == in_w(m2, "fusion_in"), \
        "identity leaked into the trunk, which reads raw slots at fixed offsets and needs none"
    assert m25.country_identity is not None and m25.card_identity is not None


def test_head_identity_is_nearly_free_in_parameters() -> None:
    """Any Elo difference at this rung cannot be a capacity effect."""
    common = dict(input_mode="grouped", drop_static=True, per_entity_heads=16)
    n2 = sum(int(p.numel()) for p in rung(**common, identity_dim=0).parameters())
    n25 = sum(int(p.numel()) for p in rung(**common, identity_dim=8).parameters())
    assert 0 < (n25 - n2) / n2 < 0.01, f"identity added {(n25 - n2) / n2:.2%} of parameters"


def test_identity_still_refused_where_nothing_shares_weights() -> None:
    """Without a per-entity head there is no shared function, so identity is pure redundancy."""
    with pytest.raises(ValueError, match="only useful"):
        rung(input_mode="grouped", identity_dim=8, per_entity_heads=0)


# --------------------------------------------------- the M2 family (head decomposition)

M2_FAMILY = {
    "M2":     dict(head_context=True,  head_static=True,  head_entities="both",    identity_dim=0),
    "M2a":    dict(head_context=False, head_static=True,  head_entities="both",    identity_dim=0),
    "M2b":    dict(head_context=True,  head_static=False, head_entities="both",    identity_dim=0),
    "M2c":    dict(head_context=False, head_static=False, head_entities="both",    identity_dim=0),
    "M2d":    dict(head_context=True,  head_static=True,  head_entities="country", identity_dim=0),
    "M2e":    dict(head_context=True,  head_static=True,  head_entities="card",    identity_dim=0),
    "M2.5":   dict(head_context=True,  head_static=True,  head_entities="both",    identity_dim=8),
    "M2.5b":  dict(head_context=True,  head_static=False, head_entities="both",    identity_dim=8),
}


@pytest.mark.parametrize("name", sorted(M2_FAMILY))
def test_m2_family_builds_runs_and_round_trips(name, tmp_path) -> None:
    """Each arm decomposes the correction, which is a function of item features and context.

    Both degenerate cases say where the content must live: constants alone are a fixed per-TYPE
    bias, and context alone is identical for all 84 countries -- a global shift the dense head
    already supplies. So the mechanism is dynamic per-item features, optionally modulated by
    context and optionally offset by per-item constants, and each "optionally" is an arm.
    """
    from ai.models.ladder_net import ladder_config_from_state_dict
    from tools.lib.player_agent import NeuralAgent

    model = rung(input_mode="grouped", drop_static=True, per_entity_heads=16,
                 **M2_FAMILY[name]).eval()
    obs = torch.randn(2, ColdWarNetV2.TOTAL_OBS_SIZE)
    with torch.no_grad():
        logits, _, _ = model.forward(obs, torch.ones(2, A, dtype=torch.uint8))
    assert logits.shape == (2, A) and torch.isfinite(logits).all()

    assert ladder_config_from_state_dict(dict(model.state_dict())) == model.ladder_config()
    path = tmp_path / f"{name}.pt"
    torch.save(model.state_dict(), path)
    agent = NeuralAgent.from_checkpoint(str(path), device="cpu")
    assert isinstance(agent.model, LadderNet)
    assert agent.model.ladder_config() == model.ladder_config()


def test_head_widths_match_what_each_arm_claims_to_read() -> None:
    """The point of the family is that each arm reads exactly one thing less than M2."""
    def w(**over) -> int:
        m = rung(input_mode="grouped", drop_static=True, per_entity_heads=16, **over)
        assert m.pe_country is not None
        layer = m.pe_country[0]
        assert isinstance(layer, torch.nn.Linear)
        return int(layer.in_features)

    full = w(**M2_FAMILY["M2"])
    assert full == ColdWarNetV2.BOARD_FEATURES + 16              # 26 raw + 16 context
    assert w(**M2_FAMILY["M2a"]) == ColdWarNetV2.BOARD_FEATURES  # context removed
    assert w(**M2_FAMILY["M2b"]) == full - 11                    # 11 constant slots removed
    assert w(**M2_FAMILY["M2c"]) == ColdWarNetV2.BOARD_FEATURES - 11
    assert w(**M2_FAMILY["M2.5"]) == full + 8                    # identity added


def test_switching_a_head_off_leaves_the_other_intact() -> None:
    country_only = rung(input_mode="grouped", drop_static=True, per_entity_heads=16,
                        **M2_FAMILY["M2d"])
    card_only = rung(input_mode="grouped", drop_static=True, per_entity_heads=16,
                     **M2_FAMILY["M2e"])
    assert country_only.pe_country is not None and country_only.pe_card is None
    assert card_only.pe_card is not None and card_only.pe_country is None


def test_head_axes_refused_without_heads() -> None:
    """They are inert without a head, so a non-canonical value means a misunderstanding."""
    with pytest.raises(ValueError, match="only mean anything"):
        rung(input_mode="grouped", per_entity_heads=0, head_context=False)
