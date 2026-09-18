"""Credit assignment around blunder losses.

An *unprovoked* blunder -- holding a scoring card past the end of a turn, or driving DEFCON
to 1 by one's own choice -- is a local mistake. The play that preceded it was not
necessarily bad, and the opponent did not earn the resulting win. So credit for it is
confined to the turn the blunder happened in, the blunderer is penalised there, and the
opponent is shielded from the windfall.

Two cases must NOT be windowed:

* A clean win or loss, where the outcome reflects the whole game and must propagate back
  through every turn. The older global ``slice_turn_boundaries`` truncated these too,
  which strips the outcome signal from all but the final turn of the ~80% of games that
  end normally.
* A *provoked* DEFCON-1 loss, where the phasing player was forced to fire an
  opponent-associated event because they had run out of safe cards. There the earlier card
  management is precisely the cause, so credit must keep flowing backwards.
"""

from typing import List

import torch

import ts_engine as ts
from ai.training.rollout_buffer import RolloutBuffer
from bindings.action_encoder import ActionEncoder

#: The engine emits one observation width; nothing here should hardcode it.
OBS_DIM = int(ts.OBS_SIZE)

TURNS: List[int] = [1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4]
PLAYERS: List[int] = [1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1]  # 1=US, -1=USSR; USSR last
VALS: List[float] = [0.5 * p for p in PLAYERS]  # perspective-consistent, US ahead by 0.5
BLUNDER_TURN = 4


def _build(term_reward: float, held_us: bool = False, held_ussr: bool = False,
           defcon_blunder: int = 0) -> RolloutBuffer:
    b = RolloutBuffer(buffer_size=len(TURNS), num_envs=1, device="cpu")
    for t in range(len(TURNS)):
        done = (t == len(TURNS) - 1)
        b.add(obs=torch.zeros(1, OBS_DIM), masks=torch.zeros(1, ActionEncoder.FLAT_ACTION_SIZE, dtype=torch.uint8),
              actions=torch.zeros(1, dtype=torch.long), log_probs=torch.zeros(1),
              rewards=torch.tensor([term_reward if done else 0.0]),
              dones=torch.tensor([float(done)]),
              values_win=torch.tensor([VALS[t]]), values_vp=torch.zeros(1),
              players=torch.tensor([PLAYERS[t]], dtype=torch.int8),
              turns=torch.tensor([TURNS[t]], dtype=torch.int8), vps=torch.tensor([5.0]),
              held_scoring_us=torch.tensor([held_us and done]),
              held_scoring_ussr=torch.tensor([held_ussr and done]),
              defcon_blunder=torch.tensor([defcon_blunder if done else 0], dtype=torch.int8))
    return b


def _gae(b: RolloutBuffer, **kw: object) -> List[float]:
    b.compute_gae(last_v_win=torch.tensor([-0.5]), last_v_vp=torch.zeros(1),
                  last_dones=torch.tensor([1.0]),
                  last_players=torch.tensor([1], dtype=torch.int8),
                  gamma=0.999, gae_lambda=0.98, **kw)  # pyrefly: ignore
    return [float(x) for x in b.returns_win[:, 0]]


def _prefix_indices() -> List[int]:
    return [i for i, t in enumerate(TURNS) if t != BLUNDER_TURN]


def _blunder_turn_indices() -> List[int]:
    return [i for i, t in enumerate(TURNS) if t == BLUNDER_TURN]


def test_clean_win_propagates_credit_back_to_the_first_turn() -> None:
    """A normal outcome must reach every turn; this is the ~80% case."""
    ret = _gae(_build(1.0))
    for i in _prefix_indices():
        assert abs(ret[i] - VALS[i]) > 0.1, (
            f"step {i} (turn {TURNS[i]}) got return {ret[i]:.3f}, indistinguishable from its "
            f"value {VALS[i]:.3f} - the outcome signal did not reach it"
        )
    assert abs(ret[-1]) == 1.0


def test_global_slicing_strips_credit_from_a_clean_win() -> None:
    """Documents why the blunt flag is off by default."""
    ret = _gae(_build(1.0), slice_turn_boundaries=True)
    # At a truncated boundary delta == -V, so advantage == -V and return == 0: the value
    # head is regressed toward zero and the outcome is invisible.
    for i in _prefix_indices():
        assert abs(ret[i]) < 0.05, (
            f"expected global slicing to zero out step {i}, got {ret[i]:.3f}"
        )


def test_held_scoring_blunder_is_confined_to_its_turn_and_shields_the_opponent() -> None:
    ret = _gae(_build(-1.0, held_ussr=True))
    for i in _prefix_indices():
        assert abs(ret[i] - VALS[i]) < 0.02, (
            f"step {i} before the blunder turn should be neutral (return ~= V, up to gamma/lambda "
            f"drift within the turn), got {ret[i]:.3f} vs V {VALS[i]:.3f}"
        )
    for i in _blunder_turn_indices():
        if PLAYERS[i] == -1:                      # the blunderer
            assert abs(ret[i] - (-1.0)) < 1e-4, f"blunderer step {i} should be -1, got {ret[i]:.3f}"
        else:                                     # the opponent, shielded
            assert abs(ret[i] - VALS[i]) < 1e-4, (
                f"opponent step {i} should be shielded (return == V), got {ret[i]:.3f}"
            )


def test_unprovoked_defcon_suicide_is_windowed_like_held_scoring() -> None:
    """The DEFCON path was previously relying on the global flag; it now windows per episode."""
    held = _gae(_build(-1.0, held_ussr=True))
    defcon = _gae(_build(-1.0, defcon_blunder=-1))
    assert defcon == held, "unprovoked DEFCON-1 should get the same treatment as held scoring"


def test_provoked_defcon_suicide_keeps_full_backward_credit() -> None:
    """Forced by an opponent event: the prefix card management is the actual cause."""
    ret = _gae(_build(-1.0, defcon_blunder=0))
    for i in _prefix_indices():
        assert abs(ret[i] - VALS[i]) > 0.1, (
            f"step {i} should carry outcome credit for a forced loss, got {ret[i]:.3f} vs V {VALS[i]:.3f}"
        )
    assert abs(ret[-1] - (-1.0)) < 1e-4


def test_blunder_window_can_be_disabled() -> None:
    ret = _gae(_build(-1.0, defcon_blunder=-1), blunder_window=False)
    for i in _prefix_indices():
        assert abs(ret[i] - VALS[i]) > 0.1, "disabling the window should restore full propagation"
