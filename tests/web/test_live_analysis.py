"""Live model analysis in the workbench: readout, "play favourite", and shareable positions.

Checkpoints are never committed, so the tests write random-weight ones into a temp tree and point
the server at it with `TS_CHECKPOINTS_DIR`. Random weights are enough: what is under test is that
the readout covers exactly the legal actions of the model's own view, that the favourite is
applied in that view, and that a position survives the trip through the address bar.
"""
from __future__ import annotations

import json
import os
import random
from typing import Any, Dict, Iterator

import numpy as np
import pytest
import torch
from starlette.testclient import TestClient

import ts_engine as ts
from ai.models.coldwar_net_v2 import create_coldwar_net_v2
from bindings.action_encoder import ActionEncoder
from tools.lib.game_step import drain_chance
from web.server.analysis import (AnalysisError, OPS_INFLUENCE_SLOT, decode_position,
                                 encode_position, list_models, resolve_model_path)
from web.server.main import app

E4_MODEL = "E9-01-01_20260101_000000/snapshot_1000000steps.pt"
MERGED_MODEL = "E9.1-01-01_20260101_000000/snapshot_2000000steps.pt"


@pytest.fixture(scope="module")
def checkpoints(tmp_path_factory: pytest.TempPathFactory) -> Iterator[str]:
    root = tmp_path_factory.mktemp("checkpoints")
    torch.manual_seed(0)
    for rel, merged in ((E4_MODEL, False), (MERGED_MODEL, True)):
        run = root / os.path.dirname(rel)
        run.mkdir()
        torch.save(create_coldwar_net_v2(torch.device("cpu")).state_dict(), run / os.path.basename(rel))
        # Training state, not a network: must never be offered.
        (run / "resume_1000000steps.pt").write_bytes(b"not a model")
        (run / "metadata.json").write_text(json.dumps(
            {"merged_influence": merged, "merged_influence_from_step": 0}))
    old = os.environ.get("TS_CHECKPOINTS_DIR")
    os.environ["TS_CHECKPOINTS_DIR"] = str(root)
    try:
        yield str(root)
    finally:
        if old is None:
            os.environ.pop("TS_CHECKPOINTS_DIR", None)
        else:
            os.environ["TS_CHECKPOINTS_DIR"] = old


@pytest.fixture(scope="module")
def client() -> TestClient:
    return TestClient(app)


def _random_position(seed: int, steps: int) -> ts.GameState:
    state = ts.GameState()
    ts.Engine.init_game(state, seed)
    drain_chance(state)
    rng = random.Random(seed)
    for _ in range(steps):
        if ts.Engine.is_terminal(state):
            break
        legal = ActionEncoder.get_legal_indices(state)
        ts.Engine.step_flat(state, rng.choice(legal))
        drain_chance(state)
    return state


def _op_choice_position() -> ts.GameState:
    """A position at an op-choice node where the merged view offers composed actions."""
    for seed in range(1, 200):
        state = _random_position(seed, 0)
        rng = random.Random(seed)
        for _ in range(400):
            if ts.Engine.is_terminal(state):
                break
            merged = ActionEncoder.get_legal_mask(state, True)
            if any(ts.ActionMask.is_merged_influence_action(state, int(i))
                   for i in np.nonzero(merged)[0] if int(i) != OPS_INFLUENCE_SLOT):
                return state
            ts.Engine.step_flat(state, rng.choice(ActionEncoder.get_legal_indices(state)))
            drain_chance(state)
    raise AssertionError("no op-choice position with composed actions found")


def _receive_state(ws: Any) -> Dict[str, Any]:
    while True:
        msg = ws.receive_json()
        if msg["type"] == "STATE_UPDATE":
            return msg


def _load(client: TestClient, game_id: str, state: ts.GameState) -> Dict[str, Any]:
    res = client.post(f"/api/games/{game_id}/position", json={"position": encode_position(state)})
    assert res.status_code == 200, res.text
    return res.json()


# -- position tokens --------------------------------------------------------------------------

def test_position_token_round_trips_mid_game() -> None:
    state = _random_position(11, 250)
    token = encode_position(state)
    assert len(token) < 2000, "a position must fit comfortably in an address bar"
    assert all(c.isalnum() or c in "-_" for c in token), "token must be URL-safe"
    back = decode_position(token)
    assert back.to_save_dict() == state.to_save_dict()
    # Same decision, same legal actions, same observation: it is the same position.
    np.testing.assert_array_equal(ActionEncoder.get_legal_mask(back), ActionEncoder.get_legal_mask(state))
    np.testing.assert_array_equal(ts.extract_observation(back, ts.Player.US),
                                  ts.extract_observation(state, ts.Player.US))


@pytest.mark.parametrize("token", ["", "not base64 !!", "eJwrSS0uAQAEXQHB"])  # last: zlib("test")
def test_malformed_position_token_is_refused(token: str) -> None:
    with pytest.raises(AnalysisError):
        decode_position(token)


def test_position_that_does_not_round_trip_is_refused() -> None:
    import base64
    import zlib
    save = _random_position(3, 40).to_save_dict()
    save["defcon"] = 300  # would be truncated by the engine's uint8
    raw = json.dumps(save).encode()
    token = base64.urlsafe_b64encode(zlib.compress(raw)).decode().rstrip("=")
    with pytest.raises(AnalysisError):
        decode_position(token)


# -- model listing ----------------------------------------------------------------------------

def test_model_listing_offers_only_networks(checkpoints: str, client: TestClient) -> None:
    listing = client.get("/api/analysis/models").json()
    runs = {r["run"]: r["snapshots"] for r in listing["runs"]}
    assert runs == {os.path.dirname(E4_MODEL): [os.path.basename(E4_MODEL)],
                    os.path.dirname(MERGED_MODEL): [os.path.basename(MERGED_MODEL)]}
    assert listing == list_models()


@pytest.mark.parametrize("rel", ["../outside.pt", "/etc/passwd",
                                 f"{os.path.dirname(E4_MODEL)}/resume_1000000steps.pt",
                                 f"{os.path.dirname(E4_MODEL)}/metadata.json"])
def test_model_path_cannot_leave_the_checkpoints_tree(checkpoints: str, rel: str) -> None:
    with pytest.raises(AnalysisError):
        resolve_model_path(rel)


# -- the live readout -------------------------------------------------------------------------

def test_analysis_is_opt_in_and_covers_exactly_the_legal_actions(checkpoints: str, client: TestClient) -> None:
    game_id = "analysis-readout"
    client.post("/api/games/new", json={"game_id": game_id, "seed": 5})
    with client.websocket_connect(f"/ws/game/{game_id}") as ws:
        first = _receive_state(ws)
        assert "analysis" not in first, "a socket that did not ask gets no readout"
        assert first["state"]["position"]

        ws.send_json({"type": "SET_ANALYSIS_MODEL", "model": E4_MODEL})
        msg = _receive_state(ws)
        a = msg["analysis"]
        assert msg["analysis_model"] == E4_MODEL and a["model"] == E4_MODEL
        assert a["merged_influence"] is False

        state = decode_position(msg["state"]["position"])
        legal = set(ActionEncoder.get_legal_indices(state))
        assert {c["idx"] for c in a["choices"]} == legal
        assert abs(sum(c["p"] for c in a["choices"]) - 1.0) < 1e-3
        assert a["policy"]["argmax_idx"] == max(a["choices"], key=lambda c: c["p"])["idx"]
        # Every choice names the MicroAction a click would send.
        for c in a["choices"]:
            ma = ts.decode_flat_action(state, c["idx"])
            assert (c["decision_type"], c["primary_id"], c["flags"]) == (
                int(ma.decision_type), int(ma.primary_id), int(ma.flags))
        for k in ("v_win_us", "v_win_ussr", "v_vp_us", "v_vp_ussr"):
            assert k in a["critic"]

        # Turning it off stops the readout.
        ws.send_json({"type": "SET_ANALYSIS_MODEL", "model": None})
        assert "analysis" not in _receive_state(ws)


def test_unknown_model_reports_an_error(checkpoints: str, client: TestClient) -> None:
    with client.websocket_connect("/ws/game/analysis-bad-model") as ws:
        _receive_state(ws)
        ws.send_json({"type": "SET_ANALYSIS_MODEL", "model": "nope/snapshot_final.pt"})
        err = ws.receive_json()
        assert err["type"] == "ANALYSIS_ERROR" and "nope" in err["message"]
        assert "analysis" not in _receive_state(ws)


def test_play_favourite_plays_the_argmax(checkpoints: str, client: TestClient) -> None:
    game_id = "analysis-favourite"
    client.post("/api/games/new", json={"game_id": game_id, "seed": 21})
    with client.websocket_connect(f"/ws/game/{game_id}") as ws:
        _receive_state(ws)
        ws.send_json({"type": "SET_ANALYSIS_MODEL", "model": E4_MODEL})
        msg = _receive_state(ws)
        for _ in range(12):
            before = decode_position(msg["state"]["position"])
            fav = msg["analysis"]["policy"]["argmax_idx"]
            ws.send_json({"type": "PLAY_FLAT", "flat_idx": fav})
            msg = _receive_state(ws)

            expected = before.clone()
            ts.Engine.step_flat(expected, fav)
            drain_chance(expected)
            if ts.Engine.is_terminal(expected):
                break
            # Chance nodes draw from the state's own RNG, so the result is exact.
            assert decode_position(msg["state"]["position"]).to_save_dict() == expected.to_save_dict()


def test_merged_view_model_paints_and_plays_composed_actions(checkpoints: str, client: TestClient) -> None:
    game_id = "analysis-merged"
    origin = _op_choice_position()
    _load(client, game_id, origin)
    with client.websocket_connect(f"/ws/game/{game_id}") as ws:
        _receive_state(ws)
        ws.send_json({"type": "SET_ANALYSIS_MODEL", "model": MERGED_MODEL})
        msg = _receive_state(ws)
        a = msg["analysis"]
        assert a["merged_influence"] is True
        assert {c["idx"] for c in a["choices"]} == set(ActionEncoder.get_legal_indices(origin, True))
        composed = [c for c in a["choices"] if c.get("composed") and "country_id" in c]
        assert composed, "an op-choice node in the merged view offers composed placements"
        commit = ts.decode_flat_action(origin, OPS_INFLUENCE_SLOT)
        for c in composed:
            assert c["country_id"] == c["idx"] - ActionEncoder.NODE_OFFSET
            # Painted on the influence button: the commit half is what that button sends.
            assert (c["decision_type"], c["primary_id"]) == (int(commit.decision_type), int(commit.primary_id))

        pick = composed[0]["idx"]
        steps_before = msg["state"]["step_index"]
        ws.send_json({"type": "PLAY_FLAT", "flat_idx": pick})
        after = _receive_state(ws)

        expected = origin.clone()
        ts.Engine.step_flat(expected, pick, False, True)  # the engine's own composition
        drain_chance(expected)
        assert decode_position(after["state"]["position"]).to_save_dict() == expected.to_save_dict()
        assert after["state"]["step_index"] == steps_before + 2, "logged as its two E4 steps"

        # Both halves undo like clicks.
        ws.send_json({"type": "UNDO_ACTION"})
        _receive_state(ws)
        ws.send_json({"type": "UNDO_ACTION"})
        back = _receive_state(ws)
        assert decode_position(back["state"]["position"]).to_save_dict() == origin.to_save_dict()


def test_illegal_flat_action_changes_nothing(checkpoints: str, client: TestClient) -> None:
    game_id = "analysis-illegal"
    client.post("/api/games/new", json={"game_id": game_id, "seed": 8})
    with client.websocket_connect(f"/ws/game/{game_id}") as ws:
        msg = _receive_state(ws)
        legal = set(ActionEncoder.get_legal_indices(decode_position(msg["state"]["position"])))
        illegal = next(i for i in range(ActionEncoder.FLAT_ACTION_SIZE) if i not in legal)
        ws.send_json({"type": "PLAY_FLAT", "flat_idx": illegal})
        ws.send_json({"type": "PING"})
        assert ws.receive_json()["type"] == "PONG", "a refused action broadcasts nothing"
    assert client.get(f"/api/games/{game_id}").json()["step_index"] == 0


# -- shared links -----------------------------------------------------------------------------

def test_shared_position_loads_and_a_reload_keeps_history(client: TestClient) -> None:
    game_id = "analysis-link"
    client.post("/api/games/new", json={"game_id": game_id, "seed": 1})
    target = _random_position(77, 120)

    loaded = _load(client, game_id, target)
    assert loaded["changed"] is True
    assert decode_position(loaded["state"]["position"]).to_save_dict() == target.to_save_dict()
    assert loaded["state"]["can_undo"] is False

    # Play one move, then "reload" with the new URL: the game keeps its undo history.
    with client.websocket_connect(f"/ws/game/{game_id}") as ws:
        msg = _receive_state(ws)
        legal = ActionEncoder.get_legal_indices(decode_position(msg["state"]["position"]))
        ws.send_json({"type": "PLAY_FLAT", "flat_idx": legal[0]})
        moved = _receive_state(ws)["state"]
    again = client.post(f"/api/games/{game_id}/position", json={"position": moved["position"]}).json()
    assert again["changed"] is False
    assert again["state"]["can_undo"] is True


def test_bad_shared_position_is_a_400(client: TestClient) -> None:
    res = client.post("/api/games/analysis-bad-link/position", json={"position": "garbage"})
    assert res.status_code == 400
