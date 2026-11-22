"""Audit M-7: under --update-bf16 the PPO ratio's old log-probs are taken once, before the epochs, from the update's
own forward of the not-yet-updated network on the learner's rows -- not from the fp32 rollout. bf16 autocast runs only
on CUDA, so on the CPU the recomputed log-probs are the rollout's own: a stored learner log-prob shifted after the
rollout is ignored with the flag and changes the update without it.

P52 (--bf16-fp32-heads): the heads-in-fp32 switch changes nothing without autocast, every head goes through it, and
the flag refuses to run without --update-bf16. (The autocast path itself needs CUDA; its numerics were measured on the
GPU.)"""

from __future__ import annotations

import copy
from typing import Any, Dict, List, Tuple

import numpy as np
import pytest
import torch

from ai.models.coldwar_net_v2 import create_coldwar_net_v2
from ai.models.ladder_net import LadderNet, create_ladder_net
from ai.training import NashPGTrainer
from bindings.ts_env import TsVectorizedEnv
from tests.training.test_token_main import CFG


def _step(update_bf16: bool, shift: float) -> Dict[str, Any]:
    torch.manual_seed(0)
    dev = torch.device("cpu")
    t: Any = NashPGTrainer(active_net=create_ladder_net(dev, **CFG), env=TsVectorizedEnv(num_envs=4, base_seed=11),
                           num_envs=4, buffer_size=16, batch_size=32, lr=3e-4, eta=0.1, ref_update_freq=500,
                           cuda_graphs=False, device=dev, update_bf16=update_bf16)
    t.collect_rollouts()
    t.buffer.log_probs.add_(shift * (t.buffer.learner > 0.5).float())
    torch.manual_seed(1)
    m = t.train_step()
    return {"m": m, "w": copy.deepcopy(t.active_net.state_dict())}


def test_the_learners_old_log_probs_come_from_the_update_forward_under_the_flag() -> None:
    clean, shifted = _step(True, 0.0), _step(True, 0.3)
    for k, v in clean["w"].items():
        assert torch.allclose(v, shifted["w"][k], atol=1e-6), k
    plain_shifted = _step(False, 0.3)
    assert np.isfinite(plain_shifted["m"]["loss"])
    assert any(not torch.allclose(v, plain_shifted["w"][k], atol=1e-6) for k, v in clean["w"].items())


def _net(**kw: Any) -> LadderNet:
    torch.manual_seed(0)
    cfg: Dict[str, Any] = dict(input_mode="grouped", aggregation="flatten", entity_dim=16, entity_proj_dim=64,
                               card_self_attention=False, cross_attention=False, per_entity_heads=16, head_context=True,
                               head_static=True, head_entities="country", head_center=True, identity_dim=0,
                               drop_static=True, hidden_dim=64, num_res_blocks=0, num_attn_heads=4, card_lookup=False,
                               card_lookup_heads=0, card_lookup_dim=0, card_lookup_identity_dim=0,
                               categorical_value=False)
    cfg.update(kw)
    return create_ladder_net(torch.device("cpu"), **cfg).eval()


def _spy_step(update_bf16: bool) -> Tuple[List[bool], torch.Tensor]:
    """One update of a network with dropout: the active network's training flag at each of its forwards, and torch's
    CPU generator at its first train-mode forward (the first minibatch)."""
    dev = torch.device("cpu")
    net = _net(num_res_blocks=2, res_dropout=0.3)
    t: Any = NashPGTrainer(active_net=net, env=TsVectorizedEnv(num_envs=4, base_seed=11), num_envs=4, buffer_size=16,
                           batch_size=32, lr=3e-4, eta=0.1, ref_update_freq=500, cuda_graphs=False, device=dev,
                           update_bf16=update_bf16)
    t.collect_rollouts()
    modes: List[bool] = []
    first_train_rng: List[torch.Tensor] = []
    plain = net.forward

    def spy(*a: Any, **k: Any) -> Any:
        modes.append(bool(net.training))
        if net.training and not first_train_rng:
            first_train_rng.append(torch.get_rng_state())
        return plain(*a, **k)

    object.__setattr__(net, "forward", spy)
    torch.manual_seed(1)
    t.train_step()
    return modes, first_train_rng[0]


def test_the_bf16_pre_pass_runs_as_the_rollout_did_and_draws_nothing() -> None:
    """Fix verification V-6.1: the old log-probs are the behaviour policy's, so the pre-pass runs in eval mode, as the
    rollout did -- in train mode each dropout layer drew a mask no minibatch shares, and moved torch's global stream
    the rollout samples from."""
    modes, rng_on = _spy_step(True)
    n_pre = -(-16 * 4 // 32)                                    # the buffer in chunks of the minibatch size
    assert modes[:n_pre] == [False] * n_pre and all(modes[n_pre:])
    modes_off, rng_off = _spy_step(False)
    assert all(modes_off)
    assert torch.equal(rng_on, rng_off), "the pre-pass moved torch's generator"


def test_fp32_heads_change_nothing_without_autocast() -> None:
    """Restored (fix verification V-6.2; deleted by ba8828e)."""
    net = _net(aux_heads=True, card_aux=True, opp_legal_aux=True, prior_gain=True)
    env = TsVectorizedEnv(num_envs=8, base_seed=3)
    obs, masks, _ = env.reset_all()
    o, m = torch.from_numpy(obs).float(), torch.from_numpy(masks)

    def heads() -> List[torch.Tensor]:
        out: List[torch.Tensor] = []
        out += list(net(o, m)) + [net.forward_value(o)] + list(net.forward_with_risk(o, m))
        out += list(net.forward_aux(o)) + [net.forward_card_aux(o), net.forward_opp_legal(o), net.forward_prior_gain(o)]
        return out

    with torch.no_grad():
        plain = heads()
        setattr(net, "fp32_heads", True)
        fp32 = heads()
    assert len(plain) == len(fp32)
    for a, b in zip(plain, fp32):
        assert torch.equal(a, b)


def test_every_head_goes_through_the_fp32_switch() -> None:
    """Fix verification V-6.4: the auxiliary heads (prior gain, ownership / VP margin, card events, opponent legality)
    and ColdWarNetV2's defcon_risk read the trunk through _heads_in like the policy and value heads, so
    --bf16-fp32-heads runs all of them in fp32."""
    net = _net(aux_heads=True, card_aux=True, opp_legal_aux=True, prior_gain=True)
    v2 = create_coldwar_net_v2("cpu").eval()
    o = torch.zeros(2, 3824)
    for model, calls in ((net, ("forward_aux", "forward_card_aux", "forward_opp_legal", "forward_prior_gain")),
                         (v2, ("defcon_risk",))):
        seen: List[str] = []
        real = model._heads_in

        def spy(h: torch.Tensor, tokens: Any, _real: Any = real) -> Any:
            seen.append("x")
            return _real(h, tokens)

        object.__setattr__(model, "_heads_in", spy)
        with torch.no_grad():
            for name in calls:
                before = len(seen)
                getattr(model, name)(o)
                assert len(seen) == before + 1, f"{type(model).__name__}.{name} bypasses _heads_in"


def test_fp32_heads_need_bf16() -> None:
    """Restored (fix verification V-6.2; deleted by ba8828e)."""
    with pytest.raises(ValueError, match="needs --update-bf16"):
        NashPGTrainer(active_net=_net(), env=TsVectorizedEnv(num_envs=4, base_seed=7), num_envs=4, buffer_size=4,
                      lr=3e-4, eta=0.1, ref_update_freq=500, cuda_graphs=False, device=torch.device("cpu"),
                      bf16_fp32_heads=True)
