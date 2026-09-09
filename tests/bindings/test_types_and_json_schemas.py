"""Tests for TypedDict annotations, pyrefly static type checking, and JSON replay schema conformance."""

import json
import os
import subprocess
import sys
import pytest
from typing import Dict, Any

import ts_engine as ts
from tools.lib.self_play import generate_self_play_replay
from ai.models.coldwar_net import create_coldwar_net
from web.server.replay import ReplayLogger, ReplayManager, replays_dir
from web.server.replay_types import (
    GameStateDict,
    ReplayLogDict,
    ReplayStepDict,
    ReplayMetadataDict,
    ReplaySummaryDict,
    AuditGameReportDict,
)


def test_pyrefly_type_checking():
    """Runs pyrefly check on core typed modules and asserts zero type errors."""
    files_to_check = [
        "web/server/replay_types.py",
        "web/server/replay.py",
        "tools/eval/self_play.py",
        "tools/generate_replay.py",
    ]
    cmd = [sys.executable, "-m", "pyrefly", "check"] + files_to_check
    proc = subprocess.run(cmd, capture_output=True, text=True)
    assert proc.returncode == 0, f"pyrefly type check failed:\n{proc.stdout}\n{proc.stderr}"


def test_gamestate_dict_schema():
    """Validates that ts.state_to_dict produces all keys required by GameStateDict."""
    state = ts.GameState()
    ts.Engine.init_game(state, 42)
    state_dict = ts.state_to_dict(state)

    required_keys = [
        "victory_points",
        "defcon",
        "mil_ops",
        "space",
        "turn",
        "action_round",
        "phasing_player",
        "current_phase",
        "current_phase_name",
        "countries",
        "hands",
        "discard_pile",
        "removed_pile",
        "draw_deck_count",
        "decision_context",
        "legal_actions",
        "is_terminal",
        "terminal_utility",
    ]
    for k in required_keys:
        assert k in state_dict, f"Missing key {k} in state_to_dict output"

    assert isinstance(state_dict["countries"], dict)
    assert len(state_dict["countries"]) == 84
    assert isinstance(state_dict["hands"], dict)
    assert "US" in state_dict["hands"]
    assert "USSR" in state_dict["hands"]


def test_replays_directory_schema_conformance(generated_replay_dir: str):
    """Validates the .tslog.json files the writer produces against the ReplayLogDict schema.

    Reads from the `generated_replay_dir` fixture rather than `data/replays/`, which is
    git-ignored and empty on a fresh checkout. This used to require a specific ambient file,
    `v2_selfplay_demo.tslog.json`, that no longer had to exist anywhere.
    """
    replays = ReplayManager.list_replays()
    assert len(replays) > 0, f"fixture directory {generated_replay_dir} produced no replays"

    # Check that list_replays returns proper summary types
    for r in replays:
        assert "filename" in r
        assert "game_id" in r
        assert "total_steps" in r
        assert isinstance(r["total_steps"], int)
        assert r["total_steps"] > 0

    loaded = ReplayManager.load_replay(replays[0]["filename"])
    assert loaded is not None, f"failed to load {replays[0]['filename']}"
    assert "steps" in loaded, "a replay should carry a standard steps array"
    assert len(loaded["steps"]) > 0
    assert "state_snapshot" in loaded["steps"][0]


def test_generate_self_play_replay_execution(tmp_path):
    """Validates that generate_self_play_replay produces a compliant ReplayLogDict."""
    out_file = str(tmp_path / "test_self_play.tslog.json")

    replay_dict, saved_path = generate_self_play_replay(
        model=create_coldwar_net("cpu"),
        seed=101,
        temperature=0.3,
        game_id="test_sim_game",
        output_path=out_file,
        max_steps=25,
        verbose=False,
    )

    assert os.path.exists(saved_path)
    assert saved_path == out_file

    assert "version" in replay_dict
    assert "metadata" in replay_dict
    assert "steps" in replay_dict
    assert len(replay_dict["steps"]) > 0

    step0 = replay_dict["steps"][0]
    assert "step_index" in step0
    assert "state_snapshot" in step0
    assert "action" in step0
    assert "description" in step0
    assert isinstance(step0["state_snapshot"], dict)

    # Verify reload
    with open(saved_path, "r", encoding="utf-8") as f:
        loaded = json.load(f)
    assert loaded["metadata"]["game_id"] == "test_sim_game"
    assert len(loaded["steps"]) == len(replay_dict["steps"])
