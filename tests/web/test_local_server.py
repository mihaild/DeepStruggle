"""The local workbench server: this machine's checkpoints and replays, and nothing else.

The workbench runs in the browser; web/server/main.py only lists local files and turns a
checkpoint into the ONNX the page runs. Checkpoints and replays are never committed, so both
trees are written into temp directories here and the server is pointed at them.
"""
from __future__ import annotations

import json
import os
from typing import Iterator

import onnx
import pytest
import torch
from starlette.testclient import TestClient

from tools.lib.engine_fingerprint import fingerprint
from web.server.main import app

RUN = "E9-03-01_20260101_000000"
SNAP = "snapshot_1000000steps.pt"


@pytest.fixture(scope="module")
def trees(tmp_path_factory: pytest.TempPathFactory) -> Iterator[dict]:
    from ai.models.coldwar_net_v2 import create_coldwar_net_v2

    ckpt = tmp_path_factory.mktemp("checkpoints")
    run = ckpt / RUN
    run.mkdir()
    torch.manual_seed(3)
    torch.save(create_coldwar_net_v2(torch.device("cpu")).state_dict(), run / SNAP)
    (run / "resume_1000000steps.pt").write_bytes(b"training state, not a network")
    (run / "snapshot_broken.pt").write_bytes(b"not a checkpoint")
    (run / "metadata.json").write_text(json.dumps({"merged_influence": False}))
    replays = tmp_path_factory.mktemp("replays")
    (replays / "one.tslog.json").write_text(json.dumps({"metadata": {"game_id": "one", "total_steps": 0}, "steps": []}))
    cache = tmp_path_factory.mktemp("onnx_cache")
    env = {"TS_CHECKPOINTS_DIR": str(ckpt), "TS_REPLAYS_DIR": str(replays), "TS_ONNX_CACHE_DIR": str(cache)}
    old = {k: os.environ.get(k) for k in env}
    os.environ.update(env)
    try:
        yield {"checkpoints": str(ckpt), "replays": str(replays), "cache": str(cache)}
    finally:
        for k, v in old.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


@pytest.fixture(scope="module")
def client() -> TestClient:
    return TestClient(app)


def test_info_names_the_engine_the_sources_build(trees: dict, client: TestClient) -> None:
    info = client.get("/api/local/info").json()
    assert info["engine_fingerprint"] == fingerprint()
    assert info["checkpoints_root"] == trees["checkpoints"]


def test_models_lists_networks_only(trees: dict, client: TestClient) -> None:
    listing = client.get("/api/local/models").json()
    assert listing["runs"] == [{"run": RUN, "snapshots": ["snapshot_broken.pt", SNAP]}]


@pytest.mark.parametrize("path", ["../x.pt", "/etc/passwd", f"{RUN}/resume_1000000steps.pt", f"{RUN}/metadata.json"])
def test_onnx_cannot_name_anything_outside_the_checkpoints(trees: dict, client: TestClient, path: str) -> None:
    assert client.get("/api/local/models/onnx", params={"path": path}).status_code == 404


def test_a_checkpoint_is_served_as_a_verified_self_describing_onnx(trees: dict, client: TestClient, tmp_path) -> None:
    res = client.get("/api/local/models/onnx", params={"path": f"{RUN}/{SNAP}"})
    assert res.status_code == 200, res.text
    out = tmp_path / "m.onnx"
    out.write_bytes(res.content)
    meta = {p.key: p.value for p in onnx.load(str(out)).metadata_props}
    assert meta["ts.format"] == "ts-onnx-v1"
    assert meta["ts.merged_influence"] == "false"
    assert meta["ts.engine_fingerprint"] == fingerprint()
    assert meta["ts.checkpoint"] == f"{RUN}/{SNAP}"
    assert int(meta["ts.obs_size"]) == 3824

    # Cached: the second request is the same file, not a second export.
    cached = [os.path.join(d, f) for d, _, fs in os.walk(trees["cache"]) for f in fs]
    assert len(cached) == 1
    mtime = os.path.getmtime(cached[0])
    assert client.get("/api/local/models/onnx", params={"path": f"{RUN}/{SNAP}"}).content == res.content
    assert os.path.getmtime(cached[0]) == mtime


def test_a_checkpoint_that_does_not_load_is_a_422_not_a_crash(trees: dict, client: TestClient) -> None:
    res = client.get("/api/local/models/onnx", params={"path": f"{RUN}/snapshot_broken.pt"})
    assert res.status_code == 422
    assert "could not export" in res.json()["detail"]


def test_replays_are_listed_and_served_from_their_directory_only(trees: dict, client: TestClient) -> None:
    names = [r["filename"] for r in client.get("/api/local/replays").json()]
    assert names == ["one.tslog.json"]
    assert client.get("/api/local/replays/one.tslog.json").json()["metadata"]["game_id"] == "one"
    assert client.get("/api/local/replays/..%2Fsecret").status_code == 404
    assert client.get("/api/local/replays/nope.tslog.json").status_code == 404


def test_the_server_side_game_is_gone(client: TestClient) -> None:
    """The game runs in the page now; the old session endpoints must not linger half-working."""
    assert client.post("/api/games/new", json={}).status_code in (404, 405)
    assert client.get("/api/metadata/map").status_code == 404
