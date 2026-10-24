"""Search isolation and cancellation use the page's TypeScript and the real Wasm engine."""
from pathlib import Path

from tests.web.js_runner import PUBLIC, run_ts


def test_rollout_runner_isolation_and_model_changes(tmp_path: Path) -> None:
    result = run_ts("rollout_runner.ts", str(tmp_path), Path(PUBLIC).as_uri() + "/")
    assert result == {"inline": True, "invalidSave": True, "lifecycle": True}
