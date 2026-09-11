"""`import bindings` must not depend on every observation layout existing in the build.

Reported from a working checkout: the web server died on

    File "bindings/ts_env.py", line 59, in <module>
        int(ts.OBS_SIZE_V23): "v2.3",
    AttributeError: module 'ts_engine' has no attribute 'OBS_SIZE_V23'

The module-level width table reached straight for each constant, so an engine build predating
v2.3 raised from inside `import bindings` — and took down every consumer of the package,
including a web server that never uses v2.3. A stale build is a real problem and still gets a
loud error; it just has to come from whatever needs the missing layout, not from the import.
"""

import subprocess
import sys

import pytest
import ts_engine

from bindings.ts_env import (
    LAYOUT_BY_OBS_SIZE,
    layout_for_model,
    obs_size_for_layout,
)

# Runs in a subprocess: deleting an attribute off the extension module would leak into every
# later test in the same process.
_PROBE = """
import sys
import ts_engine
delattr(ts_engine, "OBS_SIZE_V23")

import bindings
from bindings.ts_env import LAYOUT_BY_OBS_SIZE, obs_size_for_layout
from tools.lib.player_agent import PlayerAgent          # the web server's import chain

assert "v2.3" not in LAYOUT_BY_OBS_SIZE.values(), LAYOUT_BY_OBS_SIZE
assert obs_size_for_layout("v2.1") == int(ts_engine.OBS_SIZE_V21)
try:
    obs_size_for_layout("v2.3")
except RuntimeError as exc:
    assert "rebuild" in str(exc).lower(), str(exc)
else:
    raise AssertionError("an unavailable layout must raise")
print("OK")
"""


def test_importing_bindings_survives_an_engine_without_v23() -> None:
    result = subprocess.run([sys.executable, "-c", _PROBE], capture_output=True, text=True)
    assert result.returncode == 0, (
        f"import chain broke against an older engine:\n{result.stderr[-2000:]}")
    assert "OK" in result.stdout


def test_the_table_describes_this_build() -> None:
    for name in ("OBS_SIZE_LEGACY", "OBS_SIZE_V21", "OBS_SIZE_V23"):
        width = getattr(ts_engine, name, None)
        if width is not None:
            assert int(width) in LAYOUT_BY_OBS_SIZE


def test_an_unknown_layout_name_is_rejected() -> None:
    with pytest.raises(ValueError, match="unknown observation layout"):
        obs_size_for_layout("v2.2")     # retired, and never coming back


def test_available_layouts_round_trip() -> None:
    for width, name in LAYOUT_BY_OBS_SIZE.items():
        assert obs_size_for_layout(name) == width

        class _Model:
            TOTAL_OBS_SIZE = width

        assert layout_for_model(_Model()) == name
