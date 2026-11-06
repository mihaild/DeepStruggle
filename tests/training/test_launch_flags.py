"""tools/scripts/launch_flags.py: the drift check CLAUDE.md invariant 15 rests on.

It diffs runs by their metadata.json, so a CLI setting the metadata records under another name, or
not at all, is a blind spot. The entropy coefficient was one (recorded as `ent_coef`): E4.1-02-36
and E4-54-36, launched with --entropy-coef 0.0076, diffed as identical to their 0.01 baselines.
"""

from __future__ import annotations

import importlib.util
import json
import os
import tempfile
from types import ModuleType

from ai.training.train import build_parser

_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "tools", "scripts", "launch_flags.py")


def _lf() -> ModuleType:
    spec = importlib.util.spec_from_file_location("launch_flags", _PATH)
    assert spec is not None and spec.loader is not None
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def _run(meta: dict) -> str:
    d = tempfile.mkdtemp()
    with open(os.path.join(d, "metadata.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f)
    return d


def test_renamed_keys_map_to_real_cli_flags() -> None:
    lf = _lf()
    dests = set(lf._defaults())
    for key, (dest, _conv) in lf._RENAMED.items():
        assert dest in dests, f"{key} maps to {dest}, which is not a train.py flag"
    for key, dest in lf._LADDER.items():
        assert dest in dests, f"ladder_config[{key}] maps to {dest}, which is not a train.py flag"
    assert lf.UNCHECKED_BY_DESIGN <= dests, "UNCHECKED_BY_DESIGN names a flag train.py no longer has"


def test_the_entropy_coefficient_graphs_and_ladder_config_are_diffed() -> None:
    lf = _lf()
    a = _run({"ent_coef": 0.01, "cuda_graphs": True,
              "ladder_config": {"hidden_dim": 480, "num_res_blocks": 4}})
    b = _run({"ent_coef": 0.0076, "cuda_graphs": False,
              "ladder_config": {"hidden_dim": 512, "num_res_blocks": 4}})
    ra, rb = lf.recorded(a), lf.recorded(b)
    assert (ra["entropy_coef"], rb["entropy_coef"]) == (0.01, 0.0076)
    assert (ra["no_cuda_graphs"], rb["no_cuda_graphs"]) == (False, True)
    assert (ra["ladder_hidden_dim"], rb["ladder_hidden_dim"]) == (480, 512)
    assert lf.non_default(b)["entropy_coef"] == 0.0076


def test_settings_a_run_does_not_record_are_reported_not_passed() -> None:
    lf = _lf()
    old = _run({"ent_coef": 0.01})
    miss = set(lf.unrecorded(old))
    assert {"lr", "batch_size", "buffer_size", "gamma"} <= miss


def test_every_training_setting_is_recorded_by_the_trainer() -> None:
    """Each train.py flag must appear as a metadata key the trainer writes (under its own name, or
    via _RENAMED / ladder_config), or be listed as not a training setting. A flag added to the CLI
    and not to metadata fails here instead of becoming the next invisible drift."""
    lf = _lf()
    src = open(os.path.join(os.path.dirname(__file__), "..", "..", "ai", "training",
                            "generic_trainer.py"), encoding="utf-8").read()
    written = {k for k in lf._defaults() if f'"{k}":' in src}
    written |= {dest for key, (dest, _c) in lf._RENAMED.items() if f'"{key}":' in src}
    if '"ladder_config":' in src:
        written |= set(lf._LADDER.values())
    written |= {d for d, source in lf._DERIVED.items() if source in written}
    missing = sorted(set(lf._defaults()) - written - set(lf.UNCHECKED_BY_DESIGN))
    assert not missing, f"train.py flags the trainer never records in metadata.json: {missing}"


def test_runs_before_tf32_and_the_pool_interval_were_fp32_and_pooled_per_snapshot(tmp_path) -> None:
    import json
    from tools.scripts.launch_flags import recorded, unrecorded
    (tmp_path / "metadata.json").write_text(json.dumps({"snapshot_every_steps": 5_000_000}))
    rec = recorded(str(tmp_path))
    assert rec["tf32"] is False and rec["pool_every_steps"] == 5_000_000
    assert "tf32" not in unrecorded(str(tmp_path))


def test_a_run_recording_no_rollout_temps_used_the_legacy_bands() -> None:
    """Before 2026-09-27 the default bands were 0.15 0.50 0.10 0.35 and were recorded as None;
    the default is now 1.0 x 4, so a diff must not read the old None as the new default."""
    lf = _lf()
    old = _run({"rollout_temps": None})
    new = _run({"rollout_temps": [1.0, 1.0, 1.0, 1.0]})
    assert lf.recorded(old)["rollout_temps"] == [0.15, 0.50, 0.10, 0.35]
    assert "rollout_temps" in lf.non_default(old)
    assert "rollout_temps" not in lf.non_default(new)


def test_an_epoch_default_is_read_as_the_runs_own_epoch_had_it() -> None:
    """E8 made headline_reveal the default view (owner, 2026-10-20). An E7 run that took the default
    then read no optional feature, so it is still at its default -- its A code does not move -- while
    an E8 run that named none departs from E8's."""
    lf = _lf()
    assert lf._defaults()["obs_features"] == ["headline_reveal"]   # its rule bits are obs_rules
    e7 = _run({"run_name": "E7-A8-R1-S44", "obs_features": []})
    e7_branch = _run({"run_name": "E7-A8-R1-S44@6400M+R34", "obs_features": []})
    legacy = _run({"run_name": "E7-20-44", "obs_features": []})
    unnamed = _run({"obs_features": []})
    e8 = _run({"run_name": "E8-A8-R1-S44", "obs_features": ["headline_reveal"]})
    e8_none = _run({"run_name": "E8-A8-R1-S44", "obs_features": []})
    e8_cont = _run({"run_name": "E7-A8-R1-S44@6800M+E8", "obs_features": []})
    assert [lf.run_engine(d) for d in (e7, e7_branch, legacy, unnamed, e8, e8_cont)] == [7, 7, 7, None, 8, 8]
    for d in (e7, e7_branch, legacy, unnamed, e8):
        assert "obs_features" not in lf.non_default(d)
    assert lf.non_default(e8_none)["obs_features"] == []
    assert lf.non_default(e8_cont)["obs_features"] == []


def test_the_diff_compares_the_view_each_run_actually_read() -> None:
    """Each at its own default, an E7 and an E8 run read different observations: the diff says so.
    An E8 run that named no feature reads what the E7 run did: the diff does not invent a row."""
    lf = _lf()
    e7 = _run({"run_name": "E7-A8-R1-S44", "obs_features": []})
    e8 = _run({"run_name": "E8-A8-R1-S44", "obs_features": ["headline_reveal"]})
    e8_none = _run({"run_name": "E8-A8-R1-S44", "obs_features": []})
    assert lf.epoch_default_rows(e7, e8) == [("obs_features", [], ["headline_reveal"])]
    assert lf.epoch_default_rows(e7, e8_none) == []


def test_a_rule_bit_is_the_minor_version_not_a_code() -> None:
    """opp_event_gated (ENG-9, the default since 2026-11-06) is the rule the network plays: reported
    as `obs_rules`, apart from obs_features. An E8 run from before it keeps the code it had -- its
    obs_features are E8's default -- and only obs_rules says it plays the pre-fix rule."""
    lf = _lf()
    assert lf._defaults()["obs_rules"] == ["opp_event_gated"]
    e8 = _run({"run_name": "E8-A8-R1-S44", "obs_features": ["headline_reveal"]})
    e81 = _run({"run_name": "E8.1-A8-R1-S44", "obs_features": ["headline_reveal", "opp_event_gated"]})
    assert "obs_features" not in lf.non_default(e8) and "obs_features" not in lf.non_default(e81)
    assert lf.non_default(e8)["obs_rules"] == []
    assert "obs_rules" not in lf.non_default(e81)
    assert lf.recorded(e81)["obs_features"] == ["headline_reveal"]


def test_obs_features_default_and_opt_out_on_the_cli() -> None:
    """A new network reads headline_reveal (since E8) and plays opp_event_gated (E8.1, since
    2026-11-06); the list replaces the default, and the flag with no names is none -- E7's view,
    which continuing an E7 run needs, as continuing an E8 run needs headline_reveal alone."""
    p = build_parser()
    assert p.parse_args([]).obs_features == ["headline_reveal", "opp_event_gated"]
    assert p.parse_args(["--obs-features"]).obs_features == []
    assert p.parse_args(["--obs-features", "ops_budget"]).obs_features == ["ops_budget"]
