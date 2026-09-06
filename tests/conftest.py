from __future__ import annotations

import pytest


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--run-fuzz",
        action="store_true",
        default=False,
        help="Run differential fuzzing tests against external engines",
    )


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    if config.getoption("--run-fuzz"):
        return
    expr = getattr(config.option, "markexpr", "") or ""
    if "differential_fuzz" in expr:
        return
    for arg in config.args:
        if "test_differential_fuzzing" in arg:
            return

    skip_marker = pytest.mark.skip(
        reason="Differential fuzzers against external engines are not run by default. Use --run-fuzz, -m differential_fuzz, or specify tests/differential/test_differential_fuzzing.py"
    )
    for item in items:
        if "differential_fuzz" in item.keywords:
            item.add_marker(skip_marker)


def pytest_ignore_collect(collection_path, config: pytest.Config) -> bool | None:
    """Keep tests/differential out of collection unless it is asked for.

    Marking those tests skipped is not enough: the skip is applied after collection, and the
    modules fail at *import*, so a plain `pytest tests/` aborts with a collection error instead
    of running the suite. Ignoring the directory here rather than through an `--ignore` in
    pytest.ini keeps the escape hatches working -- `--run-fuzz`, `-m differential_fuzz`, or
    naming a path under tests/differential -- since those are all visible on `config`.
    """
    if "differential" not in collection_path.parts:
        return None
    if config.getoption("--run-fuzz"):
        return None
    if "differential_fuzz" in (getattr(config.option, "markexpr", "") or ""):
        return None
    if any("differential" in str(arg) for arg in config.args):
        return None
    return True
