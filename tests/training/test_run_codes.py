"""Architecture and recipe codes are derived from what runs recorded, and names from their lineage.

The codes replace attempt numbers, which fixed "observation, architecture, intervention and
recipe" in prose -- the place lineage-critical flags drifted from four times. Here a code is its
flags, so the tests build runs on disk and check the names that come out.
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, Optional

import pytest

from ai.training.run_name import parse_label, parse_run_name
from tools.scripts import run_codes as rc
from tools.value_labels import labelling_signature

TS = iter(f"2026100{7 + h // 24}_{h % 24:02d}0000" for h in range(72))


def _run(root: str, name: str, parent: Optional[str] = None, resume: str = "resume_state.pt",
         seed: int = 44, steps: int = 200_000_000, **flags: Any) -> str:
    d = os.path.join(root, f"{name}_{next(TS)}")
    os.makedirs(d)
    meta: Dict[str, Any] = {"train_steps": steps, "seed_env": seed, "seed_sampling": seed,
                            "seed_pool": seed, "seed_init": 44, "opponent_frac": 0.3,
                            "opponent_self_pool": True}
    meta.update(flags)
    if parent is not None:
        meta["resumed_from"] = parent if resume == "<dir>" else os.path.join(parent, resume)
    with open(os.path.join(d, "metadata.json"), "w") as f:
        json.dump(meta, f)
    return d


@pytest.fixture
def lineage(tmp_path: Any) -> Dict[str, str]:
    root = str(tmp_path)
    a = _run(root, "E9-01-44", steps=100_000_000)
    b = _run(root, "E9-02-44", parent=a, steps=300_000_000)                      # continuation
    c = _run(root, "E9-03-44", parent=b, resume="resume_150000000steps.pt", seed=45)
    d = _run(root, "E9-04-44", parent=b, resume="resume_150000000steps.pt", opponent_frac=0.5)
    e = _run(root, "E9-05-44", parent=b, resume="resume_200000000steps.pt", ladder_res_blocks=2)
    f = _run(root, "E9-06-44")                                                   # same as a
    return {k: os.path.basename(v) for k, v in dict(a=a, b=b, c=c, d=d, e=e, f=f).items()} | \
        {"root": root}


def _names(lineage: Dict[str, str]) -> Dict[str, str]:
    runs = rc.load_runs(lineage["root"])
    names, missing = rc.assign(runs, {"architectures": {}, "recipes": {}, "overrides": {}}, True)
    assert not missing
    return {k: names[v].text() for k, v in lineage.items() if k != "root"}


def test_names_follow_the_lineage(lineage: Dict[str, str]) -> None:
    n = _names(lineage)
    assert n["a"] == "E9-A1-R1-S44"
    assert n["b"] == n["a"]                       # a continuation that changes nothing keeps it
    assert n["c"] == "E9-A1-R1-S44@150M+S45"      # branch points are absolute lineage steps
    assert n["d"] == "E9-A1-R1-S44@150M+R2"       # a recipe flag is R
    assert n["e"] == "E9-A1-R1-S44@200M+A2"       # an architecture flag is A, not R
    assert n["f"] == "E9-A1-R1-S44-2"             # a second from-scratch run is a replicate


def test_without_update_an_unknown_recipe_is_reported(lineage: Dict[str, str]) -> None:
    runs = rc.load_runs(lineage["root"])
    _, missing = rc.assign(runs, {"architectures": {}, "recipes": {}, "overrides": {}}, False)
    assert missing and all("no " in m for m in missing)


def test_runs_recorded_on_another_machine_are_kept_and_counted(lineage: Dict[str, str],
                                                               monkeypatch: pytest.MonkeyPatch) -> None:
    """A run made elsewhere (CI's fits) is in run_codes.json but not under this root: an update here keeps its
    entry and lists it against its codes, where re-deriving from local directories alone would drop it."""
    runs = rc.load_runs(lineage["root"])
    codes: Dict[str, Any] = {"architectures": {}, "recipes": {}, "overrides": {}}
    names, _ = rc.assign(runs, codes, True)
    ci = "E9-A1-R1-S44@100M+S46_20261009_000000"
    codes["runs"] = {lineage["a"]: names[lineage["a"]].text(), ci: "E9-A1-R1-S44@100M+S46"}
    assert rc.recorded_elsewhere(codes, runs) == {ci: "E9-A1-R1-S44@100M+S46"}
    md = os.path.join(lineage["root"], "codes.md")
    monkeypatch.setattr(rc, "CODES_MD", md)
    rc.write_codes_md(codes, names, runs, rc.recorded_elsewhere(codes, runs))
    row = next(l for l in open(md, encoding="utf-8") if l.startswith("| **R1** |"))
    assert "E9-A1-R1-S44@100M+S46" in row


def test_an_inert_setting_does_not_split_a_recipe(tmp_path: Any) -> None:
    """A card-event batch size with the card-event target off trains exactly as without it."""
    root = str(tmp_path)
    _run(root, "E9-01-44")
    _run(root, "E9-02-44", seed=45, aux_card_sample_frac=0.005)
    runs = rc.load_runs(root)
    names, _ = rc.assign(runs, {"architectures": {}, "recipes": {}, "overrides": {}}, True)
    assert sorted(p.text() for p in names.values()) == ["E9-A1-R1-S44", "E9-A1-R1-S45"]


def test_a_view_is_its_own_code_and_moves_nothing_else(tmp_path: Any) -> None:
    """The view (V, owner 2026-11-07): a canonical setup order or an action view gets a V code, leaving A and R as
    they were; no view flag is V0 and is not written. The legacy merged-influence run keeps its engine minor
    version instead."""
    root = str(tmp_path)
    a = _run(root, "E8-A1-R1-S44", steps=400_000_000, obs_features=["headline_reveal"])
    _run(root, "E8-A1-R1-S44@300M+V1", parent=a, resume="resume_300000000steps.pt",
         obs_features=["headline_reveal", "canonical_setup"])
    _run(root, "E8-A1-R1-V2-S45", seed=45, obs_features=["headline_reveal"], macro_influence=True)
    _run(root, "E6-09-44", merged_influence=True)
    runs = rc.load_runs(root)
    names, missing = rc.assign(runs, {"architectures": {}, "recipes": {}, "views": {}, "overrides": {}}, True)
    assert not missing
    got = sorted(p.text() for p in names.values())
    assert got == ["E6.1-A1-R1-S44", "E8-A1-R1-S44", "E8-A1-R1-S44@300M+V1", "E8-A1-R1-V2-S45"], got


@pytest.fixture(scope="module")
def codes() -> Dict[str, Any]:
    with open(rc.CODES_JSON, encoding="utf-8") as f:
        return json.load(f)


class TestTheCommittedRegistry:
    """research/run_codes.json is what names mean; it must be complete and self-consistent."""

    def test_every_code_is_described_and_distinct(self, codes: Dict[str, Any]) -> None:
        for table in ("architectures", "recipes", "views"):
            keys = [json.dumps(e["flags"], sort_keys=True) for e in codes[table].values()]
            assert len(set(keys)) == len(keys), f"two {table} share their flags"
            assert all(e["description"] for e in codes[table].values())

    def test_every_mapped_name_parses_and_uses_known_codes(self, codes: Dict[str, Any]) -> None:
        for d, name in codes["runs"].items():
            f = parse_run_name(name).fields()
            assert "A" + f["A"] in codes["architectures"], d
            assert "R" + f["R"] in codes["recipes"], d
            assert f["V"] == "0" or "V" + f["V"] in codes["views"], d

    def test_the_best_soup_is_expressible(self, codes: Dict[str, Any]) -> None:
        """The 4,800M soup's three ingredients are exactly the three mapped runs."""
        lab = parse_label("E7-A4-R1-S44@4390M+(S44,45,46)@4800M")
        mapped = set(codes["runs"].values())
        assert all(r.text() in mapped for r in lab.ingredients)


def test_link_directories_alias_runs_without_becoming_runs(lineage: Dict[str, str]) -> None:
    """`--link` gives each old-scheme run a new-name directory of symlinks. It is an alias: the
    derivation must not count it as a run, and a run resumed through it hangs off the real one."""
    root = lineage["root"]
    runs = rc.load_runs(root)
    names, _ = rc.assign(runs, {"architectures": {}, "recipes": {}, "overrides": {}}, True)
    made = rc.link_runs(runs, names, root)
    assert len(made) == 6 and "E9-A1-R1-S44@150M+S45_" + lineage["c"].split("_", 1)[1] in made
    link = os.path.join(root, made[0])
    assert os.path.islink(os.path.join(link, "metadata.json"))
    assert rc.link_runs(runs, names, root) == []                  # idempotent
    # a continuation launched under the new name, resumed through the link directory
    c_link = os.path.join(root, "E9-A1-R1-S44@150M+S45_" + lineage["c"].split("_", 1)[1])
    _run(root, "E9-A1-R1-S44@150M+S45", parent=c_link, resume="<dir>", seed=45, steps=400_000_000)
    runs2 = rc.load_runs(root)
    assert len(runs2) == len(runs) + 1
    names2, _ = rc.assign(runs2, {"architectures": {}, "recipes": {}, "overrides": {}}, True)
    newest = max(runs2, key=lambda r: r.ts)
    assert names2[newest.dir].text() == "E9-A1-R1-S44@150M+S45"   # a continuation keeps the name


def test_a_labelling_recipe_gets_one_d_code_and_keeps_it(tmp_path: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    """tools/value_labels.py's settings -> a D code: the same settings find the same code, new ones
    the next, and nothing is assigned without `update`."""
    codes = tmp_path / "run_codes.json"
    codes.write_text(json.dumps({"architectures": {}, "recipes": {}, "overrides": {}, "runs": {}}))
    monkeypatch.setattr(rc, "CODES_JSON", str(codes))
    sig = {"temperature": 0.25, "keep": 8, "worlds": 32, "stop": "game"}
    assert rc.labelling_code(sig, update=False) is None
    assert rc.labelling_code(sig, update=True) == "D1"
    assert rc.labelling_code(dict(reversed(list(sig.items()))), update=False) == "D1"
    assert rc.labelling_code({**sig, "worlds": 64}, update=True) == "D2"
    assert set(json.loads(codes.read_text())["labellings"]) == {"D1", "D2"}


def test_a_labellings_chance_is_named_only_when_keyed() -> None:
    """keyed2 is its own recipe, and a labelling without keyed chance keeps the code it had."""
    pmeta = {"temperature": 0.25, "keep": 8, "candidates": 4, "holdout": 0.05}
    lmeta: Dict[str, Any] = {"worlds": 32, "safe": True, "siblings": 0, "sibling_worlds": 0, "sibling_k": 4,
                             "leaf_horizon": 0}
    assert "chance" not in labelling_signature(pmeta, lmeta)
    assert labelling_signature(pmeta, {**lmeta, "keyed": True})["chance"] == "keyed"
    assert labelling_signature(pmeta, {**lmeta, "keyed": False, "keyed2": True})["chance"] == "keyed2"


def test_a_value_fit_is_its_networks_shape_read_for_its_value(tmp_path: Any) -> None:
    """A value fit's A code is its network's flags plus value_only, so it never shares a code with
    the RL network it was fine-tuned from; its value-fit flags make its R code."""
    root = str(tmp_path)
    rl = _run(root, "E9-01-44")
    vf = _run(root, "E9-02-44", parent=rl, mode="value-fit", value_dataset="/d/p34", value_lr=2e-4)
    runs = {os.path.basename(r.dir): r for r in rc.load_runs(root)}
    a_rl = rc.arch_sig(runs[os.path.basename(rl)])
    a_vf = rc.arch_sig(runs[os.path.basename(vf)])
    assert a_vf == {**a_rl, "value_only": True}
    assert rc.recipe_sig(runs[os.path.basename(vf)], {}).get("value_dataset") == "/d/p34"


def test_a_run_started_from_a_swa_branches_where_its_window_ends(tmp_path: Any) -> None:
    """A value fit fine-tuned from `swa_11920-12000M.pt` is named as a branch at 12,000M."""
    root = str(tmp_path)
    rl = _run(root, "E9-01-44", steps=12_800_000_000)
    _run(root, "E9-02-44", parent=rl, resume="swa_11920-12000M.pt", mode="value-fit", value_lr=2e-4)
    runs = rc.load_runs(root)
    child = next(r for r in runs if r.parent is not None)
    assert child.branch_step == 12_000_000_000


def test_a_value_fits_name_carries_its_seed_and_its_branch(tmp_path: Any) -> None:
    """A value fit records one seed; fine-tuned from its parent's SWA it is named as a branch with
    its own A and R codes, from scratch as a root."""
    root = str(tmp_path)
    rl = _run(root, "E9-01-44", steps=12_800_000_000)
    fits = [_run(root, "E9-02-44", parent=rl, resume="swa_11920-12000M.pt", mode="value-fit", value_dataset="/d"),
            _run(root, "E9-03-44", mode="value-fit", value_dataset="/d", value_reinit=True)]
    for d in fits:                     # `seed` is the helper's own argument: written here
        path = os.path.join(d, "metadata.json")
        with open(path) as f:
            meta = json.load(f)
        meta["seed"] = 44
        with open(path, "w") as f:
            json.dump(meta, f)
    names, missing = rc.assign(rc.load_runs(root), {"architectures": {}, "recipes": {}, "overrides": {}}, update=True)
    got = sorted(p.text() for p in names.values())
    assert got == ["E9-A1-R1-S44", "E9-A1-R1-S44@12000M+A2-R2", "E9-A2-R3-S44"]

