"""An E4 policy translated into the E4.1 merged-influence view (P23), exactly.

At an op-choice node the E4.1 view offers "influence, first point in X" as one action, defined as
the two E4 steps it names: commit ops to influence, then place the first point in X. An E4
policy therefore implies an E4.1 policy by factorising through the step the merge removes:

    P_fact(X)             = P_E4(OPS_INFLUENCE | s) * P_E4(X | s')          s' = s after the commit
    P_fact(OPS_INFLUENCE) = P_E4(OPS_INFLUENCE | s) * P_E4(CONFIRM_DONE | s')   (only where the
                             E4.1 view still offers the bare commit: where the commit ends the game)
    P_fact(a)             = P_E4(a | s)                                      every other action

renormalised over the E4.1 legal set. `tools/scripts/merged_view_warmstart.py` measures how far a
raw E4 network is from this; `tools/generate_policy_targets.py --merged-view` records it as a
distillation target, so an E4.1 network can start from an E4 policy instead of from scratch.
"""

from __future__ import annotations

from typing import Callable, List

import numpy as np
import ts_engine as ts

#: Flat slots: SELECT_OP_MODE's influence commit, the NODE block, and the POINT_NODE "done".
INFL, CONFIRM = 112, 208
NODE = slice(116, 200)
OP_CHOICE = (ts.DecisionType.SELECT_PLAY_MODE, ts.DecisionType.SELECT_OP_MODE)


def is_merged_op_choice(state: "ts.GameState") -> bool:
    """True at an op-choice node where influence is legal in E4, i.e. where the views differ."""
    if state.ctx().decision_type not in OP_CHOICE:
        return False
    return bool(np.asarray(ts.Engine.get_flat_action_mask(state, False))[INFL])


def after_influence_commit(state: "ts.GameState") -> "ts.GameState":
    """A copy of `state` advanced by the E4 influence commit: where P_E4(X | s') is asked."""
    t = state.clone()
    ts.Engine.step_flat(t, INFL, False, False)
    return t


def post_commit_policy(states_after: List["ts.GameState"],
                       policy_fn: Callable[[List["ts.GameState"]], np.ndarray]) -> np.ndarray:
    """P_E4(. | s') for each post-commit state, rows over the flat actions.

    Where the commit itself ended the game (We Will Bury You's VP falling due, say) s' is terminal
    and has no decision: there is no second step, so the whole of P_E4(OPS_INFLUENCE) belongs to
    the bare commit, which the E4.1 view still offers exactly there. That is a one-hot on
    CONFIRM_DONE, which factorised_policy routes to the commit. Asking the network at a terminal
    state instead gives a softmax over an empty mask, NaN, which poisoned targets before this.
    `policy_fn(states) -> probs` evaluates the non-terminal ones in one batch."""
    out = np.zeros((len(states_after), 220), dtype=np.float64)
    live = [j for j, t in enumerate(states_after) if not ts.Engine.is_terminal(t)]
    for j in range(len(states_after)):
        if j not in set(live):
            out[j, CONFIRM] = 1.0
    if live:
        out[live] = np.asarray(policy_fn([states_after[j] for j in live]), dtype=np.float64)
    return out


def factorised_policy(p_e4: np.ndarray, p_post: np.ndarray, merged_mask: np.ndarray) -> np.ndarray:
    """P_fact over the E4.1 legal set, rows = positions. `p_e4` is the E4 policy at s (E4 mask),
    `p_post` the E4 policy at s' (after the commit), `merged_mask` the E4.1 mask at s."""
    p_e4 = np.asarray(p_e4, dtype=np.float64)
    p_post = np.asarray(p_post, dtype=np.float64)
    legal = np.asarray(merged_mask) > 0
    out = p_e4.copy()
    commit = p_e4[:, INFL:INFL + 1]
    out[:, INFL] = 0.0
    out[:, NODE] = commit * p_post[:, NODE] * legal[:, NODE]
    out[:, INFL] = commit[:, 0] * p_post[:, CONFIRM] * legal[:, INFL]
    out = np.where(legal, out, 0.0)
    tot = out.sum(axis=1, keepdims=True)
    if np.any(tot <= 0):
        raise ValueError("a factorised policy has no mass on the E4.1 legal set")
    return out / tot
