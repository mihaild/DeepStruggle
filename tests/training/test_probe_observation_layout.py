"""A model must never be handed an observation of a width it does not read.

Four times in this repository a probe fed a network the wrong observation and nobody noticed,
because it does not raise: the width was checked at extraction and not at the network, so the
model read the wrong floats and played near-randomly while returning ordinary-looking numbers.
`ai/eval/position_diagnostics.py` reported a mean final turn of 1-2 against an actual 6.8 for the
whole of arm H2, and 0.0 empty battlegrounds at turn 8 -- not because battlegrounds were
contested but because no game survived to turn 8.

Two things now stand between that and a repeat, and this file checks both:

* there is one observation layout, so there is no `layout` argument to default wrongly;
* the network itself refuses a vector of the wrong width, which is the check that was missing.
"""

import pytest
import torch
import ts_engine

from ai.models.coldwar_net_v2 import check_checkpoint_layout, create_coldwar_net_v2
from bindings.ts_env import check_obs_width


class _FakeModel:
    def __init__(self, width: int) -> None:
        self.TOTAL_OBS_SIZE = width


def test_the_engine_emits_one_width_under_both_names() -> None:
    assert int(ts_engine.OBS_SIZE) == int(ts_engine.OBS_SIZE_V23) == 3824


def test_the_retired_layouts_are_gone_from_the_module() -> None:
    """Not merely unused: absent, so nothing can ask for one and get a plausible answer."""
    for name in ("OBS_SIZE_LEGACY", "OBS_SIZE_V21", "OBS_SIZE_V22"):
        assert not hasattr(ts_engine, name), f"{name} is still exported"


def test_a_real_model_passes_the_width_check() -> None:
    assert check_obs_width(create_coldwar_net_v2("cpu")) == int(ts_engine.OBS_SIZE)


@pytest.mark.parametrize("width", [4293, 3891, 3825, 0])
def test_a_checkpoint_from_a_retired_layout_is_refused(width: int) -> None:
    """legacy, v2.1 and v2.2 respectively, plus a model that reports no width at all."""
    with pytest.raises(ValueError, match="the engine emits"):
        check_obs_width(_FakeModel(width))


def test_the_network_refuses_a_vector_of_the_wrong_width() -> None:
    """The half that was missing. Slicing is why a wide vector got through."""
    model = create_coldwar_net_v2("cpu")
    with pytest.raises(ValueError, match="floats wide"):
        model(torch.zeros((2, 4293), dtype=torch.float32),
              torch.ones((2, 212), dtype=torch.uint8))


def test_a_retired_checkpoints_weights_are_refused() -> None:
    """A checkpoint is a bare state dict and names no layout, so it is read off the weights."""
    good = create_coldwar_net_v2("cpu").state_dict()
    check_checkpoint_layout(good)                      # the current layout passes

    stale = dict(good)
    stale["global_proj.0.weight"] = torch.zeros((128, 76))   # legacy/v2.1 global block
    with pytest.raises(ValueError, match="retired layouts"):
        check_checkpoint_layout(stale)


@pytest.mark.parametrize("probe_module,probe_name", [
    ("ai.eval.position_diagnostics", "profile_self_play_batched"),
    ("ai.eval.decisive_probe", "measure_decisive_batched"),
])
def test_the_probes_check_the_model_before_running_it(probe_module, probe_name):
    """A probe handed a model of the wrong width must fail, not report numbers."""
    import importlib

    probe = getattr(importlib.import_module(probe_module), probe_name)
    with pytest.raises(Exception):
        probe(_FakeModel(4293), num_envs=4, max_iters=40)


def test_the_probe_scalars_and_the_trainer_tag_table_agree() -> None:
    """Every `diag/` key the probe produces is charted, and every charted one is produced.

    This is a drift guard with a real precedent. `diag/frac_reaching_turn9` was dropped from the
    probe while the snapshot-evaluation print still read it; the KeyError was caught by the broad
    `except Exception` around that block, so position_metrics came back **empty** and the whole
    strategy/battleground group would have gone missing from the run, announced only by one line
    in a long log. Checking both directions catches the drift whichever side moves.
    """
    from ai.eval.position_diagnostics import profile_self_play_batched
    from ai.training.generic_trainer import MULTILINE_CHARTS, TB_TAGS

    produced = set(profile_self_play_batched(
        create_coldwar_net_v2("cpu"), num_envs=4, max_iters=80)["scalars"])

    # A key is charted either as a series of its own or as one line of a combined chart -- the
    # four battleground series share `strategy/battlegrounds` rather than having four charts.
    on_a_combined_chart = {
        source for lines in MULTILINE_CHARTS.values() for source in lines.values()
        if isinstance(source, str)
    }
    charted = {k for k in TB_TAGS if k.startswith("diag/")} | {
        k for k in on_a_combined_chart if k.startswith("diag/")}

    assert produced == charted, (
        f"probe scalars and TB tags have drifted apart; "
        f"produced but not charted: {sorted(produced - charted)}; "
        f"charted but not produced: {sorted(charted - produced)}")
