"""A training default is stated once: tools/train.py, train_pipeline and NashPGTrainer agree.

The same knob has a default in three places -- the CLI's argparse default, `train_pipeline`'s
keyword default, and `NashPGTrainer.__init__`'s -- and a run takes whichever its entry point
reaches. They drifted once at a cost: `ref_update_freq` read 200,000 in one and 5,000,000 in
another, and E4's arms trained under the wrong one until the compute was spent
(research/findings/training/e4_architecture_discontinuity.md). A rule each new field has to
remember is not a rule, so this compares every same-named default.

What the CLI hands the pipeline is captured from `main()` itself (with `train_pipeline` replaced
by a recorder), so the CLI's own conversions -- `--slice-turn-boundaries auto` to None, feature
names to bits, `--block-lambda` resolved -- are applied rather than re-implemented here.

Two spellings of one value are not drift, and each such equivalence is listed with the line of
product code that makes it so (`_SAME`). Anything else that differs fails, unless it is in
`KNOWN_DRIFT`, which names the disagreements this test found when it was written and leaves them
to the owner: an entry must still differ (fixing one fails the test until the entry goes), so the
list only shrinks.
"""

from __future__ import annotations

import inspect
import sys
from typing import Any, Callable, Dict, Tuple

import pytest

import ai.training.train as train_mod
from ai.training.generic_trainer import train_pipeline
from ai.training.nash_pg import DEFAULT_ROLLOUT_TEMPS, NashPGTrainer


def _seq(v: Any) -> Any:
    return tuple(v) if isinstance(v, (list, tuple)) else v


#: name -> canonicaliser applied to every side's value before comparing. Each is a substitution
#: the product makes itself.
_SAME: Dict[str, Callable[[Any], Any]] = {
    # resolve_device(None) is "cuda if available, else cpu" -- what "cuda" resolves to as well.
    "device": lambda v: "cuda" if v is None else v,
    # generic_trainer: `opp_specs = eval_opponents or ["random", "heuristic"]`
    "eval_opponents": lambda v: ("random", "heuristic") if v is None else _seq(v),
    # generic_trainer: `additional_models=post_tournament_models or ["random", "heuristic"]` --
    # a set of opponents, whatever order the CLI lists them in.
    "post_tournament_models": lambda v: frozenset(("random", "heuristic") if v is None else v),
    # nash_pg: `list(rollout_temps) if rollout_temps else list(DEFAULT_ROLLOUT_TEMPS)`
    "rollout_temps": lambda v: tuple(DEFAULT_ROLLOUT_TEMPS) if v is None else _seq(v),
    # generic_trainer passes `False if slice_turn_boundaries is None else ...` to the trainer.
    "slice_turn_boundaries": lambda v: False if v is None else v,
    # generic_trainer: `teacher_seat={"us": 1, "ussr": -1, "both": 0}[teacher_seat]`
    "teacher_seat": lambda v: {"us": 1, "ussr": -1, "both": 0}.get(v, v),
    # generic_trainer: `tuple(x or ())` for each of these.
    "setup_script_openings": lambda v: () if v is None else _seq(v),
    "seed_scenarios": lambda v: () if v is None else _seq(v),
    "force_applicable_events": lambda v: () if v is None else _seq(v),
    # generic_trainer.resolve_side_independent_update: None is off for a new run without --ladder-side-nets (the
    # defaults' network) and for a resumed state that did not record it; on only for a new A39 run.
    "side_independent_update": lambda v: False if v is None else v,
}

#: (where, name) -> why it is left as it is. Found by this test on 2026-10-28 and reported, not
#: aligned: which side is right is the owner's call.
#: Empty: the three it found when written (forced_openings_us/_ussr, the trainer's num_envs) were
#: aligned to the CLI's defaults (owner, 2026-10-29).
KNOWN_DRIFT: Dict[Tuple[str, str], str] = {}


def _canon(name: str, value: Any) -> Any:
    return _SAME.get(name, _seq)(value)


def _defaults(fn: Callable[..., Any]) -> Dict[str, Any]:
    return {k: p.default for k, p in inspect.signature(fn).parameters.items()
            if p.default is not inspect.Parameter.empty}


@pytest.fixture(scope="module")
def cli_kwargs() -> Dict[str, Any]:
    """What `tools/train.py` with no arguments hands train_pipeline."""
    captured: Dict[str, Any] = {}

    def record(**kwargs: Any) -> None:
        captured.update(kwargs)

    mp = pytest.MonkeyPatch()
    mp.setattr(train_mod, "train_pipeline", record)
    mp.setattr(sys, "argv", ["train.py"])
    try:
        train_mod.main()
    finally:
        mp.undo()
    assert captured, "main() with no arguments did not reach train_pipeline"
    return captured


def _drift(a: Dict[str, Any], b: Dict[str, Any], where: str) -> Dict[str, Tuple[Any, Any]]:
    return {k: (a[k], b[k]) for k in sorted(set(a) & set(b))
            if _canon(k, a[k]) != _canon(k, b[k]) and (where, k) not in KNOWN_DRIFT}


def test_every_cli_value_is_a_train_pipeline_parameter(cli_kwargs: Dict[str, Any]) -> None:
    missing = sorted(set(cli_kwargs) - set(_defaults(train_pipeline)))
    assert not missing, f"the CLI passes parameters train_pipeline gives no default: {missing}"


def test_the_cli_defaults_are_train_pipelines(cli_kwargs: Dict[str, Any]) -> None:
    bad = _drift(cli_kwargs, _defaults(train_pipeline), "cli->pipeline")
    assert not bad, "tools/train.py and train_pipeline disagree on (cli, pipeline): " + repr(bad)


def test_train_pipelines_defaults_are_the_trainers() -> None:
    bad = _drift(_defaults(train_pipeline), _defaults(NashPGTrainer.__init__), "pipeline->trainer")
    assert not bad, "train_pipeline and NashPGTrainer disagree on (pipeline, trainer): " + repr(bad)


def test_the_cli_defaults_are_the_trainers_where_the_pipeline_has_none(cli_kwargs: Dict[str, Any]) -> None:
    """A trainer parameter the CLI sets under the same name, with no pipeline parameter between."""
    trainer = _defaults(NashPGTrainer.__init__)
    direct = {k: v for k, v in cli_kwargs.items() if k in trainer}
    bad = _drift(direct, trainer, "cli->trainer")
    # Anything here that the pipeline also has was compared above, through the pipeline.
    bad = {k: v for k, v in bad.items() if (("pipeline->trainer", k) not in KNOWN_DRIFT
                                             and ("cli->pipeline", k) not in KNOWN_DRIFT)}
    assert not bad, "tools/train.py and NashPGTrainer disagree on (cli, trainer): " + repr(bad)


def test_every_known_drift_is_still_drift(cli_kwargs: Dict[str, Any]) -> None:
    """Fixing one of them must remove its entry, so the list cannot hide a new one later."""
    sides = {"cli->pipeline": (cli_kwargs, _defaults(train_pipeline)),
             "pipeline->trainer": (_defaults(train_pipeline), _defaults(NashPGTrainer.__init__))}
    for (where, name), why in KNOWN_DRIFT.items():
        a, b = sides[where]
        assert name in a and name in b, f"{where} {name}: no longer a parameter on both sides"
        assert _canon(name, a[name]) != _canon(name, b[name]), (
            f"{where} {name} agrees now ({a[name]!r}); remove it from KNOWN_DRIFT ({why})")
