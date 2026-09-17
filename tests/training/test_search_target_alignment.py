"""P15-X4b: a search target must describe the position it is stored against.

`_search_targets` reads the runner's *current* state, and `collect_rollouts` files the answer into
the buffer alongside `obs_t`. Those are the same position only while the call happens **before**
`env.step`. Taken afterwards the runner already holds s_{t+1}, so every target describes the
position after the observation it is stored with.

The targets do not generally become *illegal* when this happens -- consecutive decisions share
most of their legal set -- they become legal and wrong, which is worse, because a mask cannot
catch it.

That is the failure this file exists for, and it is worth spelling out why it needed a test. It
does not raise, it does not warn, and it does not produce a malformed tensor. It trains the policy
toward noise with a real gradient behind it, and the only symptom is at the training-curve level:
two 20M-step arms collapsed (entropy 1.03 -> 0.36, critic_auc to chance at 0.505 where the
step-matched control holds 0.87) before the cause was found. The existing `test_search_ce.py`
covers the buffer mechanics with synthetic tensors and passes either way, because nothing there
ever compares a target against a real game state.
"""

from __future__ import annotations

import inspect
from types import SimpleNamespace
from typing import Any, List, Tuple, cast

import numpy as np
import torch
import ts_engine as ts

from ai.training.nash_pg import BaseNashPGTrainer
from bindings import TsVectorizedEnv

N_ENVS = 8
ACTION_DIM = 212


class _LegalHereSearcher:
    """A stand-in teacher that can only answer about the state it is handed.

    It returns that state's own legal actions with uniform visits. That makes the alignment
    question decidable: if the target was taken at the position it is stored with, its support is
    inside that position's mask by construction, and if it was taken one step later it is inside
    the *next* position's mask instead.
    """

    def should_search(self, state: Any) -> bool:
        return True

    def run(self, states: List[Any]) -> List[Tuple[List[int], List[float]]]:
        out = []
        for st in states:
            legal = np.flatnonzero(np.asarray(ts.get_flat_action_mask(st))).tolist()
            out.append((legal, [1.0] * len(legal)))
        return out


def _shim(env: TsVectorizedEnv) -> SimpleNamespace:
    """The attributes `_search_targets` actually touches, and nothing else."""
    return SimpleNamespace(
        _searcher=_LegalHereSearcher(),
        num_envs=N_ENVS,
        buffer=SimpleNamespace(action_dim=ACTION_DIM),
        device=torch.device("cpu"),
        env=env,
    )


def _targets(env: TsVectorizedEnv) -> Tuple[torch.Tensor, torch.Tensor]:
    # The shim is deliberately not a BaseNashPGTrainer: building a real one needs an optimiser,
    # a network and a pool, none of which `_search_targets` touches. Cast rather than construct.
    fn = cast(Any, BaseNashPGTrainer._search_targets)
    pi, flag = fn(_shim(env))
    assert pi is not None and flag is not None, "the searcher was configured off"
    return pi, flag


def _support_outside(pi: torch.Tensor, masks: np.ndarray, flag: torch.Tensor) -> int:
    """How many searched rows put mass on an action the stored mask calls illegal."""
    bad = 0
    m = torch.from_numpy(np.asarray(masks)).float()
    for i in range(pi.shape[0]):
        if float(flag[i]) <= 0.5:
            continue
        if float((pi[i] * (1.0 - m[i])).sum()) > 1e-6:
            bad += 1
    return bad


def test_a_target_taken_before_the_step_is_legal_where_it_is_stored() -> None:
    env = TsVectorizedEnv(num_envs=N_ENVS, base_seed=4242)
    obs, masks, _ = env.reset_all()

    for _ in range(6):
        pi, flag = _targets(env)          # as collect_rollouts does: BEFORE the step
        assert float(flag.sum()) > 0, "the fixture searched nothing; it is not testing anything"
        assert _support_outside(pi, masks, flag) == 0, (
            "a target put mass on an action illegal in the state it is filed against")

        actions = [int(np.flatnonzero(np.asarray(masks[i]))[0]) for i in range(N_ENVS)]
        obs, masks, _r, _d, _info = env.step(actions)


def test_the_two_orderings_give_different_answers() -> None:
    """Without this the first test could pass against a searcher that answers anything.

    Note what this does *not* assert. The first draft checked that a post-step target puts mass
    on an action illegal at s_t, and that check never fired: consecutive decisions here share
    most of their legal set, so the misalignment does not generally produce illegal mass. It
    produces mass that is legal and *wrong* -- the right answer to the following position -- and
    a mask cannot see the difference. That is precisely why the bug reached two training arms.

    So the property worth pinning is the weaker, true one: the orderings disagree. If they did
    not, the call site could sit on either side of `env.step` and the first test would prove
    nothing.
    """
    env = TsVectorizedEnv(num_envs=N_ENVS, base_seed=99)
    _obs, masks_t, _ = env.reset_all()

    differing = 0
    compared = 0
    for _ in range(8):
        pi_before, flag_before = _targets(env)

        actions = [int(np.flatnonzero(np.asarray(masks_t[i]))[0]) for i in range(N_ENVS)]
        _obs, masks_next, _r, _d, _info = env.step(actions)

        pi_after, flag_after = _targets(env)

        for i in range(N_ENVS):
            if float(flag_before[i]) <= 0.5 or float(flag_after[i]) <= 0.5:
                continue
            compared += 1
            if float((pi_before[i] - pi_after[i]).abs().sum()) > 1e-6:
                differing += 1
        masks_t = masks_next

    assert compared > 0, "nothing was searched; the fixture is not testing anything"
    assert differing > 0, (
        "the target at s_t and the target at s_{t+1} were identical everywhere, so this file "
        "cannot tell the aligned call site from the misaligned one")


def test_collect_rollouts_takes_the_targets_before_stepping() -> None:
    """The behavioural tests above exercise `_search_targets`; the bug was in its call site.

    Guarding the ordering directly is what stops the call drifting back below `env.step`, which
    is exactly what happened and what nothing else here would catch.
    """
    src = inspect.getsource(BaseNashPGTrainer.collect_rollouts)
    call = src.index("self._search_targets()")
    step = src.index("self.env.step(actions_np)")
    assert call < step, (
        "_search_targets() is called after env.step(): the runner holds s_{t+1} by then, so "
        "every target describes the position after the observation it is stored with")
