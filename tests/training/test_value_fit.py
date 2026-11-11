"""P34's value fit (ai/training/value_fit.py) end to end on a tiny playout-labelled dataset: the run
evaluates the initial network on the held-out games first, writes a checkpoint the agent loader
reads strictly (so a rollout root can take it as its leaf network), records its settings where
run_codes.py reads them, and learns a constant it can see in its labels."""
from __future__ import annotations

import json
import pathlib
from typing import Any, Dict

import numpy as np
import pytest
import torch

import ts_engine as ts
from ai.models.ladder_net import create_ladder_net
from ai.training.value_fit import ValueFitConfig, parse_arch, run_value_fit
from tests.training.test_ladder_tokens import SHALLOW
from tools.lib.engine_fingerprint import fingerprint
from tools.lib.player_agent import NeuralAgent


def _states(n: int, seed: int) -> np.ndarray:
    runner = ts.VectorizedBatchRunner(1, seed)
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n):
        for _ in range(int(rng.integers(5, 40))):
            if runner.get_terminals()[0]:
                runner.reset_game(0, int(rng.integers(1 << 30)))
            m = np.asarray(runner.get_action_masks())[0]
            runner.step_flat_all([int(rng.choice(np.flatnonzero(m)))], True)
        if runner.get_terminals()[0]:
            runner.reset_game(0, int(rng.integers(1 << 30)))
        out.append(np.frombuffer(runner.get_state(0).raw_bytes(), dtype=np.uint8))
    return np.stack(out)


def _split(n: int, seed: int, score: float) -> Dict[str, Any]:
    st = _states(n, seed)
    return {"state": st, "id": np.arange(n, dtype=np.int64) + seed * 1000, "score": np.full(n, score, np.float32),
            "score_se": np.zeros(n, np.float32), "final_vp": np.zeros(n, np.float32),
            "turn_vp": np.zeros(n, np.float32), "turn_defcon": np.full(n, 3.0, np.float32),
            "swing": np.full((n, 7), np.nan, np.float32), "swing_frac": np.zeros((n, 7), np.float32),
            "holdout": np.zeros(n, bool)}


def _init_net(tmp_path: pathlib.Path) -> pathlib.Path:
    """The fit's starting weights, seeded here: run_value_fit seeds only its own training, so an
    unseeded network would take whatever torch RNG state the tests before it on this worker left."""
    torch.manual_seed(0)
    init = tmp_path / "init.pt"
    torch.save(create_ladder_net("cpu", **SHALLOW).state_dict(), init)
    return init


def test_a_value_fit_runs_end_to_end(tmp_path: pathlib.Path) -> None:
    ds = tmp_path / "ds"
    ds.mkdir()
    np.savez(ds / "train.npz", **_split(48, 1, 0.8))
    np.savez(ds / "holdout.npz", **_split(16, 2, 0.8))
    (ds / "manifest.json").write_text(json.dumps({"dataset": "x~D1(seed=1,games=1)", "engine": fingerprint(),
                                                  "files": {}}))
    init = _init_net(tmp_path)
    out = tmp_path / "run"
    cfg = ValueFitConfig(dataset=str(ds), init=str(init), batch=16, lr=3e-3, warmup_steps=1, max_epochs=40,
                         eval_every=10, patience=100, device="cpu", bf16=False)
    run_dir = run_value_fit(cfg, run_name=None, description="test", output_dir=str(out),
                            cli={"mode": "value-fit", "value_dataset": str(ds), "warmup_checkpoint": str(init)})
    lines = [json.loads(x) for x in (pathlib.Path(run_dir) / "value_fit_metrics.jsonl").read_text().splitlines()]
    assert lines[0]["step"] == 0
    # The labels are a constant 0.8 the network can learn: the held-out error falls well below where
    # the random initial network starts.
    assert lines[-1]["hold_score_mse"] < 0.5 * lines[0]["hold_score_mse"]
    meta = json.loads((pathlib.Path(run_dir) / "metadata.json").read_text())
    assert meta["mode"] == "value-fit" and meta["value_dataset"] == str(ds)
    assert meta["resumed_from"] == str(init)
    agent = NeuralAgent.from_checkpoint(str(pathlib.Path(run_dir) / "value_fit_best.pt"), device="cpu")
    assert agent.model is not None


def test_value_arch_settings_are_typed_and_need_fresh_weights(tmp_path: pathlib.Path) -> None:
    assert parse_arch(["hidden_dim=768", "num_res_blocks=8", "head_center=true", "logit_cap=2.5"]) == {
        "hidden_dim": 768, "num_res_blocks": 8, "head_center": True, "logit_cap": 2.5}
    init = _init_net(tmp_path)
    cfg = ValueFitConfig(dataset=str(tmp_path), init=str(init), arch={"hidden_dim": 96}, device="cpu")
    with pytest.raises(SystemExit, match="value-reinit"):
        run_value_fit(cfg, run_name=None, description=None, output_dir=str(tmp_path / "run"))


def test_the_decision_sibling_loss_and_leaf_weights_run_and_report(tmp_path: pathlib.Path) -> None:
    """P34 revision 2: siblings from another directory, the per-decision mean loss beside each leaf's
    own, the main rows weighted to the root's leaf distribution; the run reports KL, the decision sign
    agreement and its split-half ceiling."""
    ds, sd2 = tmp_path / "ds", tmp_path / "d2"
    ds.mkdir()
    sd2.mkdir()
    np.savez(ds / "train.npz", **_split(32, 3, 0.6))
    np.savez(ds / "holdout.npz", **_split(12, 4, 0.6))
    (ds / "manifest.json").write_text(json.dumps({"dataset": "x~D1(seed=1,games=1)", "engine": fingerprint(), "files": {}}))
    k, w, n = 2, 4, 10
    leaves = _states(n * k * w, 5)
    cand, world, pid = np.meshgrid(np.arange(k), np.arange(w), np.arange(n), indexing="ij")
    cand, world, pid = cand.ravel(), world.ravel(), pid.ravel()
    rows = cand.size
    meta = np.stack([cand, world, np.zeros(rows, int), np.ones(rows, int), np.zeros(rows, int)], axis=1).astype(np.int32)
    score = np.where(cand == 0, 1.0, 0.0).astype(np.float32)     # candidate 0 always wins
    np.savez(sd2 / "siblings.npz", id=pid.astype(np.int64), meta=meta,
             cols=np.stack([score, np.zeros(rows), np.zeros(rows), np.full(rows, 3.0)], axis=1).astype(np.float32),
             swing=np.zeros((rows, 7), np.int32), leaf=leaves, holdout=pid >= 7)
    (sd2 / "manifest.json").write_text(json.dumps({"dataset": "x~D2(seed=1,games=1)", "files": {}}))
    init = _init_net(tmp_path)
    cfg = ValueFitConfig(dataset=str(ds), init=str(init), batch=16, lr=3e-3, warmup_steps=1, max_epochs=10,
                         eval_every=5, patience=100, device="cpu", bf16=False, siblings=str(sd2),
                         sibling_loss="decision", sibling_decisions=4, leaf_coef=1.0, leaf_weights=True)
    run_dir = run_value_fit(cfg, run_name=None, description="test", output_dir=str(tmp_path / "run"))
    lines = [json.loads(x) for x in (pathlib.Path(run_dir) / "value_fit_metrics.jsonl").read_text().splitlines()]
    last = lines[-1]
    for key in ("hold_kl", "hold_kl_leafw", "hold_dec_sign_agree", "hold_dec_sign_ceiling", "hold_dec_corr"):
        assert key in last and np.isfinite(last[key])
    assert last["hold_dec_sign_ceiling"] == 1.0            # every world agrees: candidate 0 wins
    # Only a rate here, not a learning claim: 3 held-out decisions put it on {0, 1/3, 2/3, 1}, the
    # leaves are random states that barely show which candidate they follow, and the initial network
    # can already score 1 -- "it did not fall" flipped with the initial weights.
    assert all(0.0 <= x["hold_dec_sign_agree"] <= 1.0 for x in lines)
    assert last["train_sib"] >= 0 and last["train_leaf"] > 0
    meta_out = json.loads((pathlib.Path(run_dir) / "metadata.json").read_text())
    assert meta_out["siblings_dataset"] == "x~D2(seed=1,games=1)"

