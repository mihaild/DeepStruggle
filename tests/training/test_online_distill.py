"""P39 arm 2: online distillation. The loss is zero with agreement 1 when the student is the teacher; a short run on the
CPU plays the teacher's games, evaluates a held-out set, and writes a best.pt that loads into the student's shape (what
--warmup-checkpoint reads) beside its metrics; a teacher on another observation view is refused."""

import copy
import json
import os
from typing import Any, Dict

import pytest
import torch

from ai.models.ladder_net import create_ladder_net
from ai.training.online_distill import DistillConfig, _losses, run_online_distillation

SHAPE: Dict[str, Any] = dict(input_mode="grouped", aggregation="flatten", entity_dim=16, entity_proj_dim=32,
                             card_self_attention=False, cross_attention=False, per_entity_heads=16, head_context=True,
                             head_static=True, head_entities="country", head_center=True, identity_dim=0,
                             drop_static=True, hidden_dim=64, num_res_blocks=0, num_attn_heads=4, card_lookup=False,
                             card_lookup_heads=0, card_lookup_dim=0, card_lookup_identity_dim=0,
                             categorical_value=False, branch_head=True, play_mode_head=True)


def _net(seed: int = 0, **over: Any) -> torch.nn.Module:
    torch.manual_seed(seed)
    return create_ladder_net("cpu", **{**SHAPE, **over})


def test_the_loss_is_zero_when_the_student_is_the_teacher() -> None:
    teacher = _net().eval()
    student = copy.deepcopy(teacher)
    obs, mask = torch.rand(16, 3824), torch.ones(16, 220, dtype=torch.bool)
    with torch.no_grad():
        lg, vw, vv = teacher(obs, mask)
        loss, parts = _losses(student, obs, mask, lg.float(), vw.view(-1), vv.view(-1), 0.5)
    assert float(loss) == pytest.approx(0.0, abs=1e-5) and float(parts["agree"]) == 1.0


def test_a_short_run_writes_a_loadable_best_student(tmp_path: Any) -> None:
    teacher_path = os.path.join(str(tmp_path), "teacher.pt")
    torch.save(_net(seed=1).state_dict(), teacher_path)
    student = _net(seed=2, entity_proj_dim=64, sphere_blocks=2, sphere_carry_norm=True)
    cfg = DistillConfig(teacher=teacher_path, positions=512, num_envs=4, rollout_len=16, batch=32, holdout=64,
                        eval_every=256, patience=10, device="cpu")
    run_dir = os.path.join(str(tmp_path), "run")
    best = run_online_distillation(student, cfg, run_dir, {"run_name": "test"})
    assert set(best) >= {"kl", "agree", "value_mse", "positions"} and 0.0 <= best["agree"] <= 1.0
    rows = [json.loads(l) for l in open(os.path.join(run_dir, "distill_metrics.jsonl"))]
    assert rows[0]["positions"] == 0 and rows[-1]["positions"] >= 512
    fresh = _net(seed=3, entity_proj_dim=64, sphere_blocks=2, sphere_carry_norm=True)
    fresh.load_state_dict(torch.load(os.path.join(run_dir, "best.pt"), weights_only=True))
    meta = json.load(open(os.path.join(run_dir, "metadata.json")))
    assert meta["mode"] == "online-distill" and meta["config"]["teacher"] == teacher_path


def test_a_teacher_on_another_view_is_refused(tmp_path: Any) -> None:
    teacher_path = os.path.join(str(tmp_path), "teacher.pt")
    torch.save(_net(seed=1).state_dict(), teacher_path)
    student = _net(seed=2, obs_features=1 << 2)
    cfg = DistillConfig(teacher=teacher_path, positions=64, num_envs=2, rollout_len=4, batch=8, holdout=8,
                        device="cpu")
    with pytest.raises(ValueError):
        run_online_distillation(student, cfg, os.path.join(str(tmp_path), "run"), {})
