"""Guards for the policy/critic trace recorded on a replay.

The one that matters most is `test_the_trace_does_not_move_the_decision_stream`: recording what
the policy believed must not change what it played. Everything else here checks that the numbers
on a step describe *that* step -- the distribution the move was drawn from, and the critic's
reading of the position it produced.
"""
from __future__ import annotations

import json
import os

import pytest
import torch

from tools.lib.self_play import generate_self_play_replay
from web.server.replay_types import ReplayLogDict


@pytest.fixture(scope="module")
def model():
    from ai.models.coldwar_net_v2 import create_coldwar_net_v2
    torch.manual_seed(31)
    m = create_coldwar_net_v2("cpu")
    m.eval()
    return m


@pytest.fixture(scope="module")
def traced(model, tmp_path_factory) -> ReplayLogDict:
    out = str(tmp_path_factory.mktemp("traced") / "traced.tslog.json")
    torch.manual_seed(404)
    doc, _path = generate_self_play_replay(
        model=model, seed=515, temperature=0.3, output_path=out,
        device="cpu", verbose=False, max_steps=120)
    return doc


def test_every_step_carries_a_policy_block(traced: ReplayLogDict) -> None:
    assert traced["steps"], "replay has no steps"
    for step in traced["steps"]:
        pol = step.get("policy")
        assert pol is not None, f"step {step['step_index']} has no policy block"
        assert pol["source"] in ("policy", "forced", "scripted")
        # The block must describe the action that was actually recorded, not the one the model
        # would have preferred -- that is `argmax_idx`, and they are allowed to differ.
        assert pol["chosen_idx"] == step["action"]["flat_action_idx"]


def test_chosen_probabilities_are_consistent_with_the_listing(traced: ReplayLogDict) -> None:
    for step in traced["steps"]:
        pol = step["policy"]
        if pol["source"] != "policy":
            continue
        listed = {e["idx"]: e["p"] for e in pol["top"]}
        assert pol["chosen_idx"] in listed
        assert listed[pol["chosen_idx"]] == pytest.approx(pol["p_chosen"])
        assert pol["p_max"] == pytest.approx(max(listed.values()))
        assert pol["p_max"] >= pol["p_chosen"] - 1e-9
        assert sum(listed.values()) + pol["p_tail"] == pytest.approx(1.0, abs=2e-4)


def test_the_critic_reads_the_state_the_snapshot_shows(traced: ReplayLogDict) -> None:
    for step in traced["steps"]:
        cri = step.get("critic")
        assert cri is not None, f"step {step['step_index']} has no critic block"
        assert cri["at"] == "after"
        assert cri["win_residual"] == pytest.approx(cri["v_win_us"] + cri["v_win_ussr"], abs=1e-5)
        assert -1.0 <= cri["v_win_us"] <= 1.0


def test_the_trace_metadata_says_where_the_numbers_came_from(traced: ReplayLogDict) -> None:
    meta = traced["metadata"]["trace"]
    assert meta["mode"] == "inline"
    assert meta["arch"] == "ColdWarNetV2"
    assert meta["temperature"] == pytest.approx(0.3)
    assert len(meta["engine_fingerprint"]) == 64
    assert meta["top_k"] == 0, "0 means every legal action is listed, which is the default"


def test_the_trace_does_not_move_the_decision_stream(model, tmp_path) -> None:
    """Same seed, same game -- recording a belief must not change what is played."""
    def actions(trace: bool) -> list[int]:
        torch.manual_seed(808)
        doc, _ = generate_self_play_replay(
            model=model, seed=616, temperature=0.3, trace=trace,
            output_path=str(tmp_path / f"t{trace}.tslog.json"),
            device="cpu", verbose=False, max_steps=150)
        return [s["action"]["flat_action_idx"] for s in doc["steps"]]

    assert actions(True) == actions(False)


def test_untraced_replays_carry_nothing_extra(model, tmp_path) -> None:
    torch.manual_seed(909)
    doc, path = generate_self_play_replay(
        model=model, seed=717, temperature=0.3, trace=False,
        output_path=str(tmp_path / "untraced.tslog.json"),
        device="cpu", verbose=False, max_steps=60)
    assert "trace" not in doc["metadata"]
    assert all("policy" not in s and "critic" not in s for s in doc["steps"])
    with open(path, encoding="utf-8") as f:
        assert "policy" not in json.load(f)["steps"][0]


def test_critic_every_decision_skips_the_settled_steps(model, tmp_path) -> None:
    # The seed is load-bearing: this needs a game holding BOTH a forced step and a chosen one,
    # and forced steps are rare -- P17 merged the card-play chain and took roughly a fifth of
    # the decisions with it, mostly ones with little discretion, which left seed 818 with none
    # inside max_steps and the test asserting against an empty half. Seed 717 has four.
    torch.manual_seed(111)
    doc, _ = generate_self_play_replay(
        model=model, seed=717, temperature=0.3, trace_critic_every="decision",
        output_path=str(tmp_path / "dec.tslog.json"),
        device="cpu", verbose=False, max_steps=80)
    forced = [s for s in doc["steps"] if s["policy"]["source"] == "forced"]
    chosen = [s for s in doc["steps"] if s["policy"]["source"] == "policy"]
    assert forced and chosen, "this game exercised only one kind of step"
    assert all("critic" not in s for s in forced)
    assert all("critic" in s for s in chosen)


def test_an_unknown_critic_cadence_is_refused(model, tmp_path) -> None:
    with pytest.raises(ValueError, match="trace_critic_every"):
        generate_self_play_replay(model=model, seed=1, trace_critic_every="sometimes",
                                  output_path=str(tmp_path / "x.tslog.json"),
                                  device="cpu", verbose=False, max_steps=5)


# -- the post-hoc annotator ----------------------------------------------------------------

@pytest.fixture(scope="module")
def checkpoint(model, tmp_path_factory) -> str:
    path = str(tmp_path_factory.mktemp("ckpt") / "probe.pt")
    torch.save(model.state_dict(), path)
    return path


@pytest.fixture(scope="module")
def untraced_replay(model, tmp_path_factory) -> str:
    out = str(tmp_path_factory.mktemp("plain") / "plain.tslog.json")
    torch.manual_seed(222)
    generate_self_play_replay(model=model, seed=919, temperature=0.3, trace=False,
                              output_path=out, device="cpu", verbose=False, max_steps=100)
    return out


def test_annotating_a_replay_reads_the_recorded_action(untraced_replay: str, model) -> None:
    from tools.annotate_replay import annotate

    doc = annotate(untraced_replay, model)
    annotated = [s for s in doc["steps"] if (s.get("policy") or {}).get("source") == "annotated"]
    assert annotated, "no step had a real choice to annotate"
    for step in annotated:
        pol = step["policy"]
        # The probability reported is the one the model puts on the move the replay recorded,
        # which is the whole point -- not on the move the model would have made.
        assert pol["chosen_idx"] == step["action"]["flat_action_idx"]
        listed = {e["idx"]: e["p"] for e in pol["top"]}
        assert listed[pol["chosen_idx"]] == pytest.approx(pol["p_chosen"])
        assert step["critic"]["at"] == "after"


def test_a_drifted_reconstruction_is_refused_not_reported(untraced_replay: str, model,
                                                          tmp_path) -> None:
    """A checker that never fires is not a checker: a tampered replay must abort the run."""
    from tools.annotate_replay import ReconstructionDiverged, annotate

    with open(untraced_replay, encoding="utf-8") as f:
        doc = json.load(f)
    # Swap two actions in the middle: the replay now describes a game the engine will not
    # reproduce, so the numbers would belong to a different game.
    mid = len(doc["steps"]) // 2
    doc["steps"][mid]["action"]["flat_action_idx"] = (
        doc["steps"][mid]["action"]["flat_action_idx"] + 40) % 212
    bad = str(tmp_path / "tampered.tslog.json")
    with open(bad, "w", encoding="utf-8") as f:
        json.dump(doc, f)

    with pytest.raises(ReconstructionDiverged):
        annotate(bad, model)


def test_the_annotator_cli_writes_a_file(untraced_replay: str, checkpoint: str,
                                         tmp_path) -> None:
    from tools.annotate_replay import main

    out = str(tmp_path / "annotated.tslog.json")
    rc = main(["--replay", untraced_replay, "--model", checkpoint, "--out", out, "--limit", "30"])
    assert rc == 0 and os.path.exists(out)
    with open(out, encoding="utf-8") as f:
        doc = json.load(f)
    assert doc["metadata"]["trace"]["mode"] == "annotated"
    assert len(doc["metadata"]["trace"]["checkpoint_sha256_12"]) == 12
    assert any("policy" in s for s in doc["steps"])
