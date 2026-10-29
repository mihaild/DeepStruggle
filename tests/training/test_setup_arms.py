"""P4 arms. A (--setup-block-lambda): lambda = 1 between consecutive setup placements, exactly the
normal recursion elsewhere; off is bitwise the old update, and eager equals graphed. B
(--inject-setup-only): the injector keeps only setup rows and trains the policy term alone."""

from __future__ import annotations

import pathlib
from typing import NamedTuple

import pytest
import torch

from ai.models import create_coldwar_net
from ai.models.ladder_net import create_ladder_net
from ai.training import NashPGTrainer
from ai.training.generic_trainer import _HumanInjector
from ai.training.human_corpus_dataset import HumanCorpusWriter
from ai.training.rollout_buffer import RolloutBuffer, setup_phase_slot
from ai.training.train import build_parser
from bindings.ts_env import TsVectorizedEnv
from tests.replay_corpus import ReplayCorpus

T, N = 24, 16


class _Last(NamedTuple):
    last_v_win: torch.Tensor
    last_v_vp: torch.Tensor
    last_players: torch.Tensor


def _fill(b: RolloutBuffer, setup_len: int = 6, seed: int = 0) -> _Last:
    g = torch.Generator().manual_seed(seed)
    b.players.copy_(torch.where(torch.rand(T, N, generator=g) < 0.5, 1, -1).to(torch.int8))
    b.dones.zero_()
    b.turns.fill_(1)
    b.rewards.zero_()
    b.values_win.copy_(torch.rand(T, N, generator=g) * 2 - 1)
    b.vps.zero_()
    b.obs.zero_()
    b.obs[:, :, setup_phase_slot()] = 2.0 / 6.0            # action round
    b.obs[:setup_len, :, setup_phase_slot()] = 0.0         # the first steps are setup
    return _Last(torch.rand(N, generator=g), torch.zeros(N),
                 torch.ones(N, dtype=torch.int8))


def _raw_adv(on: bool, lam: float = 0.9) -> torch.Tensor:
    b = RolloutBuffer(T, N, device="cpu")
    b.setup_block_lambda = on
    last = _fill(b)
    key = (1.0, lam, False, True, 4, False, "setup" if on else "off", False)
    b._gae_backward(key, last.last_v_win, last.last_v_vp, last.last_players)
    return b.advantages.clone()


def test_off_leaves_the_recursion_unchanged_and_on_changes_only_the_setup_block() -> None:
    off, on = _raw_adv(False), _raw_adv(True)
    assert torch.equal(off[6:], on[6:])                  # after the block: identical
    assert not torch.equal(off[:6], on[:6])              # inside: lambda 1


def test_inside_the_block_the_advantage_telescopes_to_the_first_real_position() -> None:
    lam = 0.9
    b = RolloutBuffer(T, N, device="cpu")
    b.setup_block_lambda = True
    last = _fill(b)
    b._gae_backward((1.0, lam, False, True, 4, False, "setup", False), last.last_v_win, last.last_v_vp,
                    last.last_players)
    a = b.advantages
    # step 5 is the block's last placement: ordinary lambda from there on
    # step t < 5: A_t = delta_t + sign_t * A_{t+1} (lambda = 1)
    v, p = b.values_win, b.players.float()
    for t in range(5):
        sign = p[t] * p[t + 1]
        delta = sign * v[t + 1] - v[t]
        assert torch.allclose(a[t], delta + sign * a[t + 1], atol=1e-6)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
def test_graphed_equals_eager_with_the_setup_block() -> None:
    eager = RolloutBuffer(T, N, device="cuda")
    graphed = RolloutBuffer(T, N, device="cuda")
    for b in (eager, graphed):
        b.setup_block_lambda = True
    graphed.graph_gae = True
    for seed in range(3):
        for b in (eager, graphed):
            last = _fill(b, seed=seed)
            b.compute_gae(last.last_v_win.cuda(), last.last_v_vp.cuda(),
                          last.last_players.cuda())
        assert torch.equal(eager.advantages, graphed.advantages)


def test_per_player_gae_is_refused() -> None:
    b = RolloutBuffer(T, N, device="cpu")
    b.setup_block_lambda = True
    last = _fill(b)
    with pytest.raises(ValueError):
        b.compute_gae(*last, per_player_gae=True)


#: Games of the corpus that convert end to end (pinned by tests/replayer); enough of them that the
#: injector's 80% training split holds more setup placements than one batch.
_HUMAN_GAMES = (100, 105, 111, 114, 237, 78, 96)


def _human_dataset(corpus: ReplayCorpus, directory: str) -> str:
    """A human-corpus dataset of a few games, written the way tools/build_human_dataset.py
    writes one. The test used to read the full dataset from the shared data tree, which is
    built rather than committed, so it held only where someone had built it."""
    writer = HumanCorpusWriter(directory)
    for replay_id in _HUMAN_GAMES:
        conv = corpus.converted(replay_id)
        assert conv.failure is None and conv.samples, f"replay {replay_id}: {conv.failure}"
        writer.add_game(conv.samples, us_utility=conv.us_utility, final_vp=conv.final_victory_points)
    writer.write()
    return directory


def test_the_setup_only_injector_keeps_setup_rows_and_the_policy_term(tmp_path: pathlib.Path,
                                                                     corpus: ReplayCorpus) -> None:
    ds = [_human_dataset(corpus, str(tmp_path / "human"))]
    net = create_ladder_net("cpu", input_mode="grouped", aggregation="flatten", entity_dim=16,
                            entity_proj_dim=64, card_self_attention=False, cross_attention=False,
                            per_entity_heads=16, head_context=True, head_static=True,
                            head_entities="country", identity_dim=0, card_lookup=False,
                            card_lookup_heads=0, card_lookup_dim=0, card_lookup_identity_dim=0,
                            drop_static=True, hidden_dim=64, num_res_blocks=1, num_attn_heads=4,
                            categorical_value=False, head_center=True)
    inj = _HumanInjector(ds[0], net, torch.device("cpu"), every=1, weight=1.0, batch_size=64,
                         setup_only=True)
    phase = inj._obs[inj.train_idx][:, setup_phase_slot()]
    assert len(inj.train_idx) >= 64 and float(abs(phase).max()) == 0.0
    assert inj.maybe_step(0) > 0.0


def _lam_one(mode: str, b: RolloutBuffer) -> torch.Tensor:
    """Which transitions a mode chains, read back through the recursion's own mask."""
    b.block_lambda = mode
    last = _Last(torch.zeros(N), torch.zeros(N),
                 torch.ones(N, dtype=torch.int8))
    b.values_win.zero_(); b.rewards.zero_()
    # with V = 0 and rewards 0 except at the end, the advantage at t is non-zero only if the chain
    # from t reaches the step whose reward is set -- so probe one transition at a time instead:
    out = torch.zeros(T - 1, N, dtype=torch.bool)
    for t in range(T - 1):
        b.rewards.zero_(); b.rewards[t + 1] = 1.0
        b.values_win.zero_()
        b._gae_backward((1.0, 0.0, False, False, 4, False, mode, False), last.last_v_win,
                        last.last_v_vp, last.last_players)
        # with gae_lambda = 0 the advantage at t sees reward t+1 only through a lambda-1 link
        out[t] = b.advantages[t] != 0
    return out


def test_setup_side_cuts_the_block_where_the_mover_changes() -> None:
    b = RolloutBuffer(T, N, device="cpu")
    _fill(b, setup_len=8)
    b.players[:4] = -1; b.players[4:] = 1              # USSR 0-3, US 4-...
    a = _lam_one("setup", b)
    s = _lam_one("setup-side", b)
    assert bool(a[3].all()) and not bool(s[3].any())   # the hand-over: chained only in "setup"
    assert bool(s[0].all()) and bool(s[5].all())       # inside each side's run: chained
    assert not bool(s[7].any())                        # setup -> first real position: not chained


def test_same_side_chains_same_mover_with_unchanged_rng_only() -> None:
    b = RolloutBuffer(T, N, device="cpu")
    _fill(b, setup_len=0)
    b.players.fill_(1)
    b.rngs.zero_()
    b.rngs[5:] = 7                                     # a chance node between steps 4 and 5
    b.players[10:] = -1                                # the mover changes between 9 and 10
    s = _lam_one("same-side", b)
    assert bool(s[0].all()) and bool(s[3].all())
    assert not bool(s[4].any())                        # rng changed
    assert not bool(s[9].any())                        # mover changed
    assert bool(s[12].all())


def test_the_trainer_records_rng_states_only_for_same_side() -> None:
    for mode, expect in (("off", False), ("same-side", True)):
        torch.manual_seed(0)
        env = TsVectorizedEnv(num_envs=8, base_seed=123)
        tr = NashPGTrainer(active_net=create_coldwar_net(torch.device("cpu")), env=env, num_envs=8,
                           buffer_size=64, batch_size=64, cuda_graphs=False, device="cpu",
                           block_lambda=mode)
        tr.collect_rollouts()
        assert bool((tr.buffer.rngs != 0).any()) is expect
        if expect:
            # a real rollout changes the rng somewhere (the deal, dice) and keeps it somewhere
            same = tr.buffer.rngs[:-1] == tr.buffer.rngs[1:]
            assert bool(same.any()) and bool((~same).any())


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
def test_graphed_equals_eager_for_same_side() -> None:
    eager = RolloutBuffer(T, N, device="cuda")
    graphed = RolloutBuffer(T, N, device="cuda")
    graphed.graph_gae = True
    for seed in range(3):
        for b in (eager, graphed):
            b.block_lambda = "same-side"
            last = _fill(b, seed=seed)
            g = torch.Generator().manual_seed(seed + 10)
            b.rngs.copy_(torch.randint(0, 3, (T, N), generator=g).cumsum(0))
            b.compute_gae(last.last_v_win.cuda(), last.last_v_vp.cuda(),
                          last.last_players.cuda())
        assert torch.equal(eager.advantages, graphed.advantages)


def test_off_is_the_default_and_the_block_modes_stay_reachable() -> None:
    p = build_parser()
    a = p.parse_args([])
    assert a.block_lambda is None                       # resolved to off in main()
    assert p.parse_args(["--block-lambda", "setup-side"]).block_lambda == "setup-side"
    assert p.parse_args(["--block-lambda", "off"]).block_lambda == "off"
    assert p.parse_args(["--setup-block-lambda"]).setup_block_lambda is True
