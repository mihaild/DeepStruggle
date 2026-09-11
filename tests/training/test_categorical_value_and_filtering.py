"""P1: the categorical value head, and advantage filtering.

Two independent changes, screened as a 2x2, so they are tested independently here too.

The categorical head replaces two scalar regressions with one distribution over final VP. What
must hold is that nothing downstream notices: `v_win` and `v_vp` are still produced, still have
the same shapes and ranges, and still mean the same thing — because the tournament code, the
probes and NashPG's advantage computation all read them.
"""

import pytest
import torch

import ai.models.coldwar_net_v2 as M
from ai.models.coldwar_net_v2 import VALUE_ATOMS, VP_LIMIT, create_for_layout, create_like


def _net(categorical: bool):
    net = create_for_layout("v2.3", "cpu", categorical_value=categorical)
    net.eval()
    return net


def _batch(net, n: int = 8):
    return torch.randn(n, net.TOTAL_OBS_SIZE), torch.ones(n, 212, dtype=torch.uint8)


# --- the head itself --------------------------------------------------------------------------

def test_the_support_is_the_engines_actual_vp_range() -> None:
    net = _net(True)
    assert VALUE_ATOMS == 2 * VP_LIMIT + 1 == 41
    assert float(net.value_support[0]) == -20.0
    assert float(net.value_support[-1]) == 20.0
    # An atom exactly on zero is what makes P(VP>0) - P(VP<0) a clean win probability.
    assert 0.0 in [float(x) for x in net.value_support]


@pytest.mark.parametrize("categorical", [False, True])
def test_both_heads_expose_the_same_interface(categorical: bool) -> None:
    net = _net(categorical)
    obs, mask = _batch(net)
    with torch.no_grad():
        logits, v_win, v_vp = net(obs, mask)
    assert logits.shape == (8, 212)
    assert v_win.shape == (8, 1) and v_vp.shape == (8, 1)
    assert bool((v_win >= -1.0).all() and (v_win <= 1.0).all()), "v_win must stay a utility"
    assert bool((v_vp >= -VP_LIMIT).all() and (v_vp <= VP_LIMIT).all())


def test_the_two_forward_paths_agree() -> None:
    net = _net(True)
    obs, mask = _batch(net)
    with torch.no_grad():
        _, v_win, v_vp = net(obs, mask)
        _, v_win2, v_vp2, value_logits = net.forward_with_value_logits(obs, mask)
    assert torch.allclose(v_win, v_win2) and torch.allclose(v_vp, v_vp2)
    assert value_logits is not None and value_logits.shape == (8, VALUE_ATOMS)


def test_a_scalar_model_has_no_value_logits() -> None:
    net = _net(False)
    obs, mask = _batch(net)
    with torch.no_grad():
        *_, value_logits = net.forward_with_value_logits(obs, mask)
    assert value_logits is None, "a scalar head must not pretend to have a distribution"


def test_the_scalars_are_derived_from_the_distribution() -> None:
    """v_vp = E[VP] and v_win = P(VP>0) - P(VP<0), computed by hand from the logits."""
    net = _net(True)
    obs, mask = _batch(net, 4)
    with torch.no_grad():
        _, v_win, v_vp, logits = net.forward_with_value_logits(obs, mask)
        assert logits is not None
        probs = torch.softmax(logits, dim=-1)
        support = net.value_support
        expect_vp = (probs * support).sum(dim=-1, keepdim=True)
        expect_win = ((probs * (support > 0)).sum(dim=-1, keepdim=True)
                      - (probs * (support < 0)).sum(dim=-1, keepdim=True))
    assert torch.allclose(v_vp, expect_vp, atol=1e-5)
    assert torch.allclose(v_win, expect_win, atol=1e-5)


# --- the two-hot target -----------------------------------------------------------------------

@pytest.mark.parametrize("vp", [-20.0, -7.5, -0.5, 0.0, 0.5, 7.3, 19.9, 20.0])
def test_two_hot_is_a_distribution_with_the_right_mean(vp: float) -> None:
    net = _net(True)
    t = net.two_hot(torch.tensor([vp]))
    assert t.shape == (1, VALUE_ATOMS)
    assert float(t.sum()) == pytest.approx(1.0, abs=1e-6)
    assert int((t > 0).sum()) <= 2, "a two-hot target puts mass on at most two atoms"
    assert float((t[0] * net.value_support).sum()) == pytest.approx(vp, abs=1e-5)


def test_two_hot_clamps_rather_than_crashing_out_of_range() -> None:
    # The engine cannot produce these, so anything outside is an upstream bug -- it should pile
    # up visibly on the end atom rather than raise or silently wrap.
    net = _net(True)
    for vp, expect in ((999.0, 20.0), (-999.0, -20.0)):
        t = net.two_hot(torch.tensor([vp]))
        assert float((t[0] * net.value_support).sum()) == pytest.approx(expect)


def test_two_hot_handles_a_batch() -> None:
    net = _net(True)
    vals = torch.tensor([[-3.5], [0.0], [11.25]])
    t = net.two_hot(vals)
    assert t.shape == (3, VALUE_ATOMS)
    assert torch.allclose((t * net.value_support).sum(-1), vals.reshape(-1), atol=1e-5)


# --- the checkpoint contract -------------------------------------------------------------------

def test_the_two_variants_are_not_interchangeable_checkpoints() -> None:
    """A categorical checkpoint must not silently load as a scalar one, or the reverse."""
    scalar, categorical = _net(False), _net(True)
    assert "value_dist_head.0.weight" in categorical.state_dict()
    assert "value_dist_head.0.weight" not in scalar.state_dict()
    assert "val_win_head.0.weight" in scalar.state_dict()
    assert "val_win_head.0.weight" not in categorical.state_dict()
    with pytest.raises(RuntimeError):
        scalar.load_state_dict(categorical.state_dict())


def test_create_like_preserves_the_head() -> None:
    # The failure this guards has happened twice on other dimensions: a frozen evaluation copy
    # built by listing arguments missed one, and the arm died at its first snapshot.
    for categorical in (False, True):
        src = _net(categorical)
        copy = create_like(src, "cpu")
        assert copy.categorical_value is categorical
        assert set(copy.state_dict()) == set(src.state_dict())


# --- advantage filtering ------------------------------------------------------------------------

def test_filtering_keeps_the_high_advantage_tail() -> None:
    """The quantile is over the minibatch, so it adapts as the advantage scale shrinks."""
    adv = torch.tensor([-5.0, 0.1, -0.05, 3.0, 0.02, -4.0])
    q = 0.5
    keep = adv.abs() >= torch.quantile(adv.abs().float(), q)
    assert keep.tolist() == [True, False, False, True, False, True]


def test_a_zero_quantile_keeps_everything() -> None:
    adv = torch.tensor([-5.0, 0.1, 3.0])
    keep = adv.abs() >= torch.quantile(adv.abs().float(), 0.0)
    assert bool(keep.all())


def test_the_trainer_accepts_both_options() -> None:
    from ai.training.nash_pg import NashPGTrainer

    assert "adv_filter_quantile" in NashPGTrainer.__init__.__code__.co_varnames
    net = _net(True)
    assert hasattr(net, "two_hot") and hasattr(net, "forward_with_value_logits")


def test_the_value_loss_is_the_cross_entropy_it_claims_to_be() -> None:
    """Wiring, not statistics: the trainer's loss must equal the hand-computed CE.

    An earlier version of this test asserted the untrained head sits near ln(41). It does, but
    only loosely and depending on the initialisation -- it measured 3.6 on one seed and 2.97 on
    another, which makes it a flaky proxy for the thing actually worth pinning: that the loss is
    computed from the right logits against the right target.
    """
    import torch.nn.functional as F

    from ai.training.nash_pg import NashPGTrainer
    from bindings.ts_env import TsVectorizedEnv

    net = _net(True)
    env = TsVectorizedEnv(num_envs=4, base_seed=7, layout="v2.3")
    trainer = NashPGTrainer(active_net=net, env=env, num_envs=4, buffer_size=8, batch_size=8)
    # The trainer moves the net to its device; follow it rather than assuming CPU.
    device = next(net.parameters()).device

    obs, mask = _batch(net, 16)
    obs, mask = obs.to(device), mask.to(device)
    ret_vp = torch.empty(16, device=device).uniform_(-20.0, 20.0)
    with torch.no_grad():
        _, v_win, v_vp, logits = net.forward_with_value_logits(obs, mask)
        assert logits is not None
        got = trainer._value_loss(v_win.squeeze(-1), v_vp.squeeze(-1),
                                  torch.zeros(16, device=device), ret_vp, logits)
        want = -(net.two_hot(ret_vp) * F.log_softmax(logits, dim=-1)).sum(-1).mean()
    assert float(got) == pytest.approx(float(want), abs=1e-6)


def test_a_confident_wrong_prediction_costs_more_than_a_confident_right_one() -> None:
    """The direction the loss has to push, independent of initialisation."""
    import torch.nn.functional as F

    net = _net(True)
    target_vp = torch.tensor([10.0])
    target = net.two_hot(target_vp)

    right = torch.full((1, VALUE_ATOMS), -10.0)
    right[0, int((10.0 + VP_LIMIT))] = 10.0            # mass on the correct atom
    wrong = torch.full((1, VALUE_ATOMS), -10.0)
    wrong[0, int((-10.0 + VP_LIMIT))] = 10.0           # mass on the opposite one

    ce_right = -(target * F.log_softmax(right, dim=-1)).sum()
    ce_wrong = -(target * F.log_softmax(wrong, dim=-1)).sum()
    assert float(ce_right) < float(ce_wrong)
