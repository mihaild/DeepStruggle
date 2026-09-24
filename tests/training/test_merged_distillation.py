"""P23: an E4 policy translated into the E4.1 view, recorded, replayed and distilled."""

from __future__ import annotations

import gzip
import json
import os
import tempfile

import numpy as np
import pytest
import torch

from tools.lib.merged_targets import CONFIRM, INFL, NODE, factorised_policy


def test_the_factorised_policy_is_the_two_e4_steps() -> None:
    p_e4 = np.zeros((1, 220))
    p_e4[0, INFL] = 0.4          # commit ops to influence
    p_e4[0, 113] = 0.6           # some other op mode
    p_post = np.zeros((1, 220))
    p_post[0, 120] = 0.75        # first point in node 120 after the commit
    p_post[0, 130] = 0.25
    mask = np.zeros((1, 220), dtype=np.uint8)
    mask[0, [113, 120, 130]] = 1  # the E4.1 view: bare commit gone, nodes offered
    out = factorised_policy(p_e4, p_post, mask)
    assert out[0, 113] == pytest.approx(0.6)
    assert out[0, 120] == pytest.approx(0.4 * 0.75)
    assert out[0, 130] == pytest.approx(0.4 * 0.25)
    assert out[0, INFL] == 0.0
    assert out.sum() == pytest.approx(1.0)


def test_the_bare_commit_keeps_the_confirm_mass_where_the_view_still_offers_it() -> None:
    p_e4 = np.zeros((1, 220))
    p_e4[0, INFL] = 1.0
    p_post = np.zeros((1, 220))
    p_post[0, CONFIRM] = 0.5
    p_post[0, 150] = 0.5
    mask = np.zeros((1, 220), dtype=np.uint8)
    mask[0, [INFL, 150]] = 1
    out = factorised_policy(p_e4, p_post, mask)
    assert out[0, INFL] == pytest.approx(0.5) and out[0, 150] == pytest.approx(0.5)
    assert out[0, NODE].sum() == pytest.approx(0.5)


def _generate(path: str, merged: bool) -> None:
    from ai.models.coldwar_net_v2 import create_coldwar_net_v2
    import tools.generate_policy_targets as gpt
    torch.manual_seed(0)
    d = os.path.dirname(path)
    ckpt = os.path.join(d, "snapshot_final.pt")
    torch.save(create_coldwar_net_v2().state_dict(), ckpt)
    argv = ["--us", ckpt, "--ussr", ckpt, "--total-games", "2", "--batch-size", "2",
            "--temperature", "1.0", "--device", "cpu", "--output-path", path]
    if merged:
        argv.append("--merged-view")
    import sys
    old = sys.argv
    sys.argv = ["generate_policy_targets.py"] + argv
    try:
        assert gpt.main() == 0
    finally:
        sys.argv = old


def test_a_merged_dataset_replays_in_the_merged_view_and_not_in_e4() -> None:
    from ai.training.warmup_dataset_loader import WarmupDataset
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "t.jsonl.gz")
        _generate(path, merged=True)
        with open(path + ".meta.json", encoding="utf-8") as f:
            assert json.load(f)["merged_influence"] is True
        with gzip.open(path, "rt", encoding="utf-8") as f:
            games = [json.loads(line) for line in f]
        recorded = sum(1 for g in games for a in g["actions"] if a.get("search_pi"))
        replayed = list(WarmupDataset(path).stream_policy_transitions(merged=True))
        assert len(replayed) == recorded, "the merged replay lost sync with the recorded games"
        for _obs, mask, target, _dt in replayed:
            assert target.sum() == pytest.approx(1.0, abs=1e-5)
            assert np.all(target[mask == 0] == 0)
        composed = sum(1 for _o, _m, t, dt in replayed if t[NODE].sum() > 0 and dt in (2, 4))
        assert composed > 0, "no composed op-choice target was recorded"
        e4 = list(WarmupDataset(path).stream_policy_transitions(merged=False))
        assert len(e4) < recorded, "replaying a merged dataset in E4 should desynchronise"


def test_distilling_a_merged_dataset_marks_the_checkpoint_as_e4_1() -> None:
    from ai.models.coldwar_net_v2 import create_coldwar_net_v2
    from ai.training.generic_trainer import run_search_distillation
    from tools.lib.action_view import checkpoint_merged_influence
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "t.jsonl.gz")
        _generate(path, merged=True)
        out_dir = os.path.join(d, "distilled")
        os.makedirs(out_dir)
        out = os.path.join(out_dir, "snapshot_final.pt")
        torch.manual_seed(1)
        run_search_distillation(create_coldwar_net_v2(), path, out, epochs=1, batch_size=64,
                                device="cpu")
        assert checkpoint_merged_influence(out) is True
