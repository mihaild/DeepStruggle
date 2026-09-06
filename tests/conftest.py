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
