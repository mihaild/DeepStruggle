"""The contract the workbench uses to put a probability on the right card, button or country.

A replay's policy trace stores flat action indices. To paint one onto a card or a map node the
frontend has to turn `(decision_type, primary_id, flags)` back into that index, and it does so
from `/api/metadata/action_space` rather than from a second copy of the offsets. If that mapping
is ever wrong, nothing fails: every probability simply lands on the wrong thing while still
looking plausible. So it is checked here against a real generated game, where the answer is
known -- the step's own `flat_action_idx`.
"""
from __future__ import annotations

import json
from typing import Any, Dict, Optional

import pytest
from starlette.testclient import TestClient

from web.server.main import app

client = TestClient(app)

FLAG_CONFIRM_DONE = 0x80


@pytest.fixture(scope="module")
def action_space() -> Dict[str, Any]:
    res = client.get("/api/metadata/action_space")
    assert res.status_code == 200
    return res.json()


@pytest.fixture(scope="module")
def traced_replay(tmp_path_factory) -> Dict[str, Any]:
    """A real traced game -- data/replays is git-ignored and may hold nothing."""
    import torch

    from ai.models.coldwar_net_v2 import create_coldwar_net_v2
    from tools.lib.self_play import generate_self_play_replay

    torch.manual_seed(5)
    model = create_coldwar_net_v2("cpu")
    model.eval()
    out = str(tmp_path_factory.mktemp("traced") / "mapping.tslog.json")
    doc, _ = generate_self_play_replay(model=model, seed=99, temperature=0.3, output_path=out,
                                       device="cpu", verbose=False, max_steps=150)
    return json.loads(json.dumps(doc))


def flat_index(space: Dict[str, Any], decision_type: int, primary_id: int,
               flags: int) -> Optional[int]:
    """The frontend's mapping, in Python. Kept in step with `trace_view.ts::flatIndex`."""
    off = space["offsets"]
    dt = space["decision_types"]
    if flags & FLAG_CONFIRM_DONE:
        return int(space["confirm_done_index"])
    if decision_type == dt["SELECT_CARD"]:
        return off["card"] + primary_id - 1 if 1 <= primary_id <= 110 else None
    if decision_type == dt["SELECT_PLAY_MODE"]:
        return off["play_mode"] + primary_id
    if decision_type == dt["CHOOSE_TIMING_BRANCH"]:
        return off["timing"] + primary_id
    if decision_type == dt["SELECT_OP_MODE"]:
        return off["op_mode"] + primary_id
    if decision_type == dt["POINT_NODE"]:
        return off["node"] + primary_id
    if decision_type == dt["CHOOSE_BRANCH"]:
        return off["branch"] + primary_id
    return None


def test_the_endpoint_agrees_with_the_encoder(action_space: Dict[str, Any]) -> None:
    from bindings.action_encoder import ActionEncoder

    assert action_space["size"] == ActionEncoder.FLAT_ACTION_SIZE
    assert action_space["confirm_done_index"] == ActionEncoder.CONFIRM_DONE_INDEX
    assert action_space["offsets"] == {
        "card": ActionEncoder.CARD_OFFSET,
        "play_mode": ActionEncoder.PLAY_MODE_OFFSET,
        "op_mode": ActionEncoder.OP_MODE_OFFSET,
        "roll_die": ActionEncoder.ROLL_DIE_INDEX,
        "node": ActionEncoder.NODE_OFFSET,
        "branch": ActionEncoder.BRANCH_OFFSET,
    }


def test_every_recorded_action_maps_back_to_its_own_flat_index(
        action_space: Dict[str, Any], traced_replay: Dict[str, Any]) -> None:
    """The board shows the state after step N, so step N+1 is the decision open at it.

    That pairing is what the workbench relies on: the decision type comes from the displayed
    snapshot and the probabilities from the next step's policy. If the two ever described
    different nodes, this mapping would stop reproducing the recorded index.
    """
    steps = traced_replay["steps"]
    assert len(steps) > 20, "too short a game to be worth checking"

    checked = 0
    for shown, following in zip(steps, steps[1:]):
        ctx = (shown["state_snapshot"] or {}).get("decision_context") or {}
        decision_type = ctx.get("decision_type")
        if decision_type is None:
            continue
        action = following["action"]
        got = flat_index(action_space, int(decision_type), int(action["primary_id"]),
                         int(action["flags"]))
        assert got is not None, (
            f"step {following['step_index']} ({following['description']}) did not map to any "
            f"flat index from decision type {decision_type}")
        assert got == action["flat_action_idx"], (
            f"step {following['step_index']} ({following['description']}) mapped to {got}, but "
            f"the replay recorded {action['flat_action_idx']} -- the workbench would put this "
            f"probability on the wrong choice")
        checked += 1

    assert checked > 20, f"only {checked} steps were checked"


def test_the_trace_lists_every_legal_action(traced_replay: Dict[str, Any]) -> None:
    """The board labels one option at a time, so a truncated distribution leaves holes."""
    chosen = [s for s in traced_replay["steps"] if s["policy"]["source"] == "policy"]
    assert chosen, "no step had a real choice"
    for step in chosen:
        pol = step["policy"]
        assert len(pol["top"]) == pol["n_legal"], (
            f"step {step['step_index']} listed {len(pol['top'])} of {pol['n_legal']} legal "
            f"actions")
        assert pol["p_tail"] == 0.0
