"""`import bindings` must not depend on an engine constant existing in the build.

Reported from a working checkout: the web server died on

    File "bindings/ts_env.py", line 59, in <module>
        int(ts.OBS_SIZE_V23): "v2.3",
    AttributeError: module 'ts_engine' has no attribute 'OBS_SIZE_V23'

The module-level width table reached straight for each constant, so an engine build predating
v2.3 raised from inside `import bindings` — and took down every consumer of the package,
including a web server that never uses v2.3. A stale build is a real problem and still gets a
loud error; it just has to come from whatever needs the engine, not from the import.

The layout table is gone — there is one layout — but the shape of the mistake outlives it, and
the single-layout refactor reintroduced it twice before this test caught it: `TsEnv` held
`OBSERVATION_SIZE = int(ts.OBS_SIZE)` as a class attribute and `RolloutBuffer` took
`obs_dim: int = int(ts.OBS_SIZE)` as a default argument. Both are evaluated while the module
loads. So the rule is not about layouts: **no engine constant is read at import time**, and
`bindings.ts_env.obs_size()` is where it is read instead.
"""

import subprocess
import sys

import pytest
import ts_engine

from bindings.ts_env import check_obs_width, obs_size

# Runs in a subprocess: deleting an attribute off the extension module would leak into every
# later test in the same process.
_PROBE = """
import sys
import ts_engine
delattr(ts_engine, "OBS_SIZE")

import bindings                                         # must not raise
from bindings.ts_env import obs_size
from ai.training.rollout_buffer import RolloutBuffer    # a default argument once read it
from tools.lib.player_agent import PlayerAgent          # the web server's import chain

try:
    obs_size()
except RuntimeError as exc:
    assert "rebuild" in str(exc).lower(), str(exc)
else:
    raise AssertionError("an engine without OBS_SIZE must raise when the width is needed")
print("OK")
"""


def test_importing_bindings_survives_an_engine_without_obs_size() -> None:
    result = subprocess.run([sys.executable, "-c", _PROBE], capture_output=True, text=True)
    assert result.returncode == 0, (
        f"import chain broke against an older engine:\n{result.stderr[-2000:]}")
    assert "OK" in result.stdout


def test_obs_size_describes_this_build() -> None:
    assert obs_size() == int(ts_engine.OBS_SIZE) == int(ts_engine.OBS_SIZE_V23)


def test_the_retired_layout_constants_are_absent() -> None:
    """Not merely unused: gone, so nothing can read one and get a plausible width."""
    for name in ("OBS_SIZE_LEGACY", "OBS_SIZE_V21", "OBS_SIZE_V22"):
        assert not hasattr(ts_engine, name), f"{name} is still exported"


def test_a_model_of_the_wrong_width_is_refused() -> None:
    class _Model:
        TOTAL_OBS_SIZE = 4293        # legacy, retired

    with pytest.raises(ValueError, match="the engine emits"):
        check_obs_width(_Model())
