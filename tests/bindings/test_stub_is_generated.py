"""`bindings/ts_engine/` is generated from the built module, and must stay that way.

A hand-maintained stub drifts from the bindings silently, and a stub that lies is worse than no
stub at all: it is exactly what type checking trusts. This regenerates the stubs from the module
that is actually imported and compares them to the committed files, so both directions of drift
fail here — a binding changed without rebuilding, and a stub edited by hand.

If this fails, do not edit the .pyi files. Rebuild:

    cmake --build build/release -j

and commit what changed. To change something introspection cannot recover (the shape of a value
returned as an untyped `nb::dict`), edit `bindings/ts_engine.pyi.pattern` instead.
"""

import difflib
import pathlib
import subprocess
import sys

import pytest
import ts_engine

REPO = pathlib.Path(__file__).resolve().parents[2]
STUB_DIR = REPO / "bindings" / "ts_engine"
PATTERN = REPO / "bindings" / "ts_engine.pyi.pattern"


def _generate(into: pathlib.Path) -> dict[str, str]:
    module_dir = pathlib.Path(ts_engine.__file__).resolve().parent
    subprocess.run(
        [sys.executable, "-m", "nanobind.stubgen",
         "-m", "ts_engine", "-i", str(module_dir),
         "-r", "-O", str(into), "-p", str(PATTERN), "-q"],
        check=True, capture_output=True, text=True, cwd=str(REPO),
    )
    generated = into / "ts_engine"
    return {f.name: f.read_text() for f in sorted(generated.glob("*.pyi"))}


def _committed() -> dict[str, str]:
    return {f.name: f.read_text() for f in sorted(STUB_DIR.glob("*.pyi"))}


def test_the_pattern_file_exists() -> None:
    # The stubs are generated *with* it; without it the CountryInfo TypedDict silently vanishes
    # and get_country_info goes back to returning an unchecked `dict`.
    assert PATTERN.is_file(), f"{PATTERN} is missing; the generated stubs would lose their types"


def test_the_stub_package_is_committed() -> None:
    assert (STUB_DIR / "__init__.pyi").is_file()
    # EffectBits is a nanobind submodule and needs its own stub: rendered flat it becomes
    # `from ts_engine import EffectBits`, which resolves to nothing.
    assert (STUB_DIR / "EffectBits.pyi").is_file()
    assert not list(STUB_DIR.glob("*.py")), (
        "a .py file here would let `bindings/ts_engine` shadow the real extension module")


def test_the_committed_stubs_match_the_built_module(tmp_path: pathlib.Path) -> None:
    pytest.importorskip("nanobind")
    generated, committed = _generate(tmp_path), _committed()

    assert set(generated) == set(committed), (
        f"stub files differ: generated {sorted(generated)}, committed {sorted(committed)}")

    for name in sorted(generated):
        if generated[name] == committed[name]:
            continue
        diff = "\n".join(difflib.unified_diff(
            committed[name].splitlines(), generated[name].splitlines(),
            fromfile=f"committed bindings/ts_engine/{name}",
            tofile="generated from the built module", lineterm="", n=2))
        pytest.fail(
            f"bindings/ts_engine/{name} is out of date with the bindings it describes.\n"
            "Rebuild (cmake --build build/release -j) and commit what changed; "
            "do not hand-edit it.\n\n" + diff[:4000])


def test_the_typed_dict_survived_generation() -> None:
    """The part introspection cannot produce, and so the part most likely to be lost."""
    text = (STUB_DIR / "__init__.pyi").read_text()
    assert "class CountryInfo(TypedDict):" in text
    assert "TypedDict" in text.split("\n\n")[1], "the TypedDict import was not injected"
    assert "def get_country_info(arg: int, /) -> CountryInfo:" in text
    # region is the enum, not an int -- the distinction the stub exists to record.
    assert "region: Region" in text


def test_country_state_is_read_only_in_the_stub() -> None:
    """def_ro in the bindings must show up as a property with no setter."""
    text = (STUB_DIR / "__init__.pyi").read_text()
    block = text.split("class CountryState:")[1].split("\nclass ")[0]
    assert "def us_influence" in block
    assert "setter" not in block, (
        "CountryState is bound def_ro; a setter in the stub means the binding regressed to "
        "def_rw, where assignment silently mutates a discarded copy")
