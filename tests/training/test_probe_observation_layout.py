"""Probes must take their observation layout from the model, never from the env default.

This is the fourth instance of the same bug class in this repository, and the reason it keeps
recurring is that it does not raise: `TsVectorizedEnv` defaults to `legacy`, the network's width
assertion passes because it is checked somewhere else, and the model simply reads the wrong
floats and plays near-randomly. `ai/eval/position_diagnostics.py` reported a mean final turn of
1-2 against an actual 6.8 for the whole of arm H2, and 0.0 empty battlegrounds at turn 8 -- not
because battlegrounds were contested but because no game survived to turn 8.
"""

import pytest
import ts_engine

from ai.models.coldwar_net_v2 import create_for_layout
from bindings.ts_env import LAYOUT_BY_OBS_SIZE, layout_for_model


class _FakeModel:
    def __init__(self, width: int) -> None:
        self.TOTAL_OBS_SIZE = width


def test_every_known_width_maps_to_its_layout() -> None:
    assert LAYOUT_BY_OBS_SIZE[int(ts_engine.OBS_SIZE_LEGACY)] == "legacy"
    assert LAYOUT_BY_OBS_SIZE[int(ts_engine.OBS_SIZE_V21)] == "v2.1"
    assert LAYOUT_BY_OBS_SIZE[int(ts_engine.OBS_SIZE_V23)] == "v2.3"


@pytest.mark.parametrize("layout", ["legacy", "v2.1", "v2.3"])
def test_a_real_model_reports_the_layout_it_was_built_for(layout: str) -> None:
    model = create_for_layout(layout)
    assert layout_for_model(model) == layout


def test_an_unknown_width_raises_rather_than_guessing() -> None:
    # v2.2 is retired and its width no longer maps to anything. Guessing here is precisely the
    # failure this helper exists to prevent, so it must raise.
    with pytest.raises(ValueError, match="cannot determine the observation layout"):
        layout_for_model(_FakeModel(3825))
    with pytest.raises(ValueError, match="cannot determine the observation layout"):
        layout_for_model(_FakeModel(0))


@pytest.mark.parametrize("probe_module,probe_name", [
    ("ai.eval.position_diagnostics", "profile_self_play_batched"),
    ("ai.eval.decisive_probe", "measure_decisive_batched"),
])
def test_the_probes_build_their_env_with_the_models_layout(monkeypatch, probe_module, probe_name):
    """Captured at the constructor: whatever else the probe does, the env must not be legacy."""
    import importlib

    import bindings.ts_env as ts_env

    captured: dict = {}
    real = ts_env.TsVectorizedEnv

    class Spy(real):  # type: ignore[misc,valid-type]
        def __init__(self, *args, **kwargs):
            captured.update(kwargs)
            super().__init__(*args, **kwargs)

    monkeypatch.setattr(ts_env, "TsVectorizedEnv", Spy)

    model = create_for_layout("v2.3")
    probe = getattr(importlib.import_module(probe_module), probe_name)
    probe(model, num_envs=4, obs_flags=0, max_iters=40)

    assert captured.get("layout") == "v2.3", (
        f"{probe_name} built its env with layout {captured.get('layout')!r}; a v2.3 model fed "
        f"legacy observations plays at random and every number the probe returns is noise")
    assert captured.get("obs_flags") == 0


def test_the_probe_scalars_and_the_trainer_tag_table_agree() -> None:
    """Every `diag/` key the probe produces is charted, and every charted one is produced.

    This is a drift guard with a real precedent. `diag/frac_reaching_turn9` was dropped from the
    probe while the snapshot-evaluation print still read it; the KeyError was caught by the broad
    `except Exception` around that block, so position_metrics came back **empty** and the whole
    strategy/ battleground group would have gone missing from the run, announced only by one line
    in a long log. Checking both directions catches the drift whichever side moves.
    """
    from ai.eval.position_diagnostics import profile_self_play_batched
    from ai.models.coldwar_net_v2 import create_for_layout
    from ai.training.generic_trainer import TB_TAGS

    produced = set(profile_self_play_batched(
        create_for_layout("v2.3"), num_envs=4, max_iters=80)["scalars"])
    charted = {k for k in TB_TAGS if k.startswith("diag/")}

    assert produced == charted, (
        f"probe scalars and TB tags have drifted apart; "
        f"produced but not charted: {sorted(produced - charted)}; "
        f"charted but not produced: {sorted(charted - produced)}")
