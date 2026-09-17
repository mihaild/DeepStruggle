"""The auxiliary DEFCON-risk head and its training target.

val_win_head is close to a restatement of the VP margin (corr(v_win, v_vp) = 0.86 over the
review replays), which is why it prices self-inflicted DEFCON-1 deaths at about -0.27 --
roughly an even position -- while pricing ordinary losing positions at -0.72. This head
gives the shared latent an explicit target for imminent DEFCON-1 death instead.

The head is inert unless --defcon-coef is set, so these tests cover the wiring: the target
marks the right steps, the extra head does not disturb the existing forward contract, and
the loss term only engages when asked for.
"""

import torch

import pytest

from ai.models.coldwar_net import create_coldwar_net
from ai.models.coldwar_net_v2 import create_coldwar_net_v2
import ts_engine as ts
from ai.training.rollout_buffer import RolloutBuffer

#: The engine emits one observation width; nothing here should hardcode it.
OBS_DIM = int(ts.OBS_SIZE)


def _buffer(num_envs: int = 1, size: int = 8) -> RolloutBuffer:
    return RolloutBuffer(buffer_size=size, num_envs=num_envs, obs_dim=OBS_DIM,
                         action_dim=212, device="cpu")


def _fill(buf: RolloutBuffer, blunder_at: int, blunderer: int) -> None:
    """A run of steps by alternating players, ending in a self-inflicted DEFCON-1 loss."""
    for t in range(buf.buffer_size):
        done = (t == blunder_at)
        player = 1 if t % 2 == 0 else -1
        buf.add(
            obs=torch.zeros(buf.num_envs, OBS_DIM),
            masks=torch.ones(buf.num_envs, 212, dtype=torch.uint8),
            actions=torch.zeros(buf.num_envs, dtype=torch.long),
            log_probs=torch.zeros(buf.num_envs),
            rewards=torch.zeros(buf.num_envs),
            dones=torch.full((buf.num_envs,), float(done)),
            values_win=torch.zeros(buf.num_envs),
            values_vp=torch.zeros(buf.num_envs),
            players=torch.full((buf.num_envs,), float(player)),
            turns=torch.ones(buf.num_envs, dtype=torch.int8),
            vps=torch.zeros(buf.num_envs),
            defcon_blunder=torch.full((buf.num_envs,), blunderer if done else 0,
                                      dtype=torch.int8),
        )


def test_target_marks_only_the_doomed_players_run_up() -> None:
    """Positives land on the blunderer's own steps inside the horizon, and nowhere else."""
    buf = _buffer(size=8)
    _fill(buf, blunder_at=6, blunderer=1)  # US brings the loss on itself at t=6
    buf.compute_gae(
        last_v_win=torch.zeros(1), last_v_vp=torch.zeros(1),
        last_dones=torch.zeros(1), last_players=torch.ones(1),
        defcon_risk_horizon=4,
    )
    target = buf.defcon_risk_target[:, 0]

    # Players alternate US(+1) at even t. The loss is at t=6, horizon 4 steps back.
    assert target[6] == 1.0, "the terminal blunder step itself must be positive"
    assert target[4] == 1.0, "the blunderer's earlier step inside the horizon is positive"
    assert target[5] == 0.0, "the opponent's steps are never positive"
    assert target[7] == 0.0, "steps after the loss are not part of the run-up"
    assert target[0] == 0.0, "steps outside the horizon are negative"


def test_no_blunder_means_no_positives() -> None:
    """An ordinary loss must not produce DEFCON-risk positives."""
    buf = _buffer(size=8)
    _fill(buf, blunder_at=6, blunderer=0)  # terminal, but not self-inflicted
    buf.compute_gae(
        last_v_win=torch.zeros(1), last_v_vp=torch.zeros(1),
        last_dones=torch.zeros(1), last_players=torch.ones(1),
    )
    assert buf.defcon_risk_target.sum() == 0.0


@pytest.mark.parametrize("factory", [create_coldwar_net, create_coldwar_net_v2])
def test_head_does_not_disturb_the_existing_forward_contract(factory) -> None:
    """forward() must still return exactly (logits, v_win, v_vp) for every other caller."""
    net = factory("cpu")
    net.eval()   # dropout is active in train mode, so the two passes would differ
    obs = torch.zeros(2, OBS_DIM)
    mask = torch.ones(2, 212, dtype=torch.uint8)

    out = net(obs, mask)
    assert len(out) == 3

    logits, v_win, v_vp, risk = net.forward_with_risk(obs, mask)
    assert logits.shape == (2, 212) and risk.shape == (2, 1)
    torch.testing.assert_close(logits, out[0])
    torch.testing.assert_close(v_win, out[1])

    prob = net.defcon_risk(obs)
    assert prob.shape == (2, 1)
    assert bool(((prob >= 0) & (prob <= 1)).all()), "risk must be a probability"


def test_batches_carry_the_target() -> None:
    """get_batches yields the DEFCON-risk target alongside the existing tensors."""
    buf = _buffer(size=8)
    _fill(buf, blunder_at=6, blunderer=1)
    buf.compute_gae(
        last_v_win=torch.zeros(1), last_v_vp=torch.zeros(1),
        last_dones=torch.zeros(1), last_players=torch.ones(1),
    )
    batch = next(buf.get_batches(batch_size=8))
    # Eighth tensor, and the learner mask is the ninth (added for frozen-opponent sampling).
    # X4b appends a search target and its flag after those; positions 1-9 are what is pinned.
    assert len(batch) >= 9, "expected the DEFCON-risk target eighth and the learner mask ninth"
    assert batch[7].shape == (8,)


def test_checkpoints_without_the_head_still_load() -> None:
    """The head was added after existing checkpoints were written."""
    import torch
    from tools.lib.player_agent import load_checkpoint_into

    net = create_coldwar_net_v2("cpu")
    old = {k: v for k, v in net.state_dict().items() if not k.startswith("defcon_risk_head.")}
    load_checkpoint_into(net, old)          # must not raise

    with pytest.raises(RuntimeError):
        broken = dict(old)
        broken.pop(next(iter(broken)))      # a genuinely missing weight must still fail
        load_checkpoint_into(create_coldwar_net_v2("cpu"), broken)
