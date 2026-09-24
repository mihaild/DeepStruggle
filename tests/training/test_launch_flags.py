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
    missing = sorted(set(lf._defaults()) - written - set(lf.UNCHECKED_BY_DESIGN))
    assert not missing, f"train.py flags the trainer never records in metadata.json: {missing}"


def test_runs_before_tf32_and_the_pool_interval_were_fp32_and_pooled_per_snapshot(tmp_path) -> None:
    import json
    from tools.scripts.launch_flags import recorded, unrecorded
    (tmp_path / "metadata.json").write_text(json.dumps({"snapshot_every_steps": 5_000_000}))
    rec = recorded(str(tmp_path))
    assert rec["tf32"] is False and rec["pool_every_steps"] == 5_000_000
    assert "tf32" not in unrecorded(str(tmp_path))
