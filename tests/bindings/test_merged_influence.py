"""P23 / E4.1: the merged-influence view is the two E4 steps it names, and nothing else.

The whole safety argument of the change is composition: a NODE slot at an op-choice node is
DEFINED as `step(OPS_INFLUENCE); step(X)`, and OPS_INFLUENCE in that view as
`step(OPS_INFLUENCE); step(CONFIRM_DONE)`. So these tests check the definition directly, at every
op-choice node of many random games, rather than checking outcomes that a wrong implementation
could also produce:

* the merged mask is the E4 mask with influence replaced by exactly the placements (and the
  early stop) that follow it;
* every composed action leaves a GameState byte-identical to the two E4 steps;
* with the view off, nothing changes (the frozen decision stream covers that end to end; here it
  is the mask and step per node);
* a mixed game -- one side merged, one not -- runs to the end in the batch runner.
"""

from __future__ import annotations

from typing import List, Tuple

import numpy as np
import pytest
import ts_engine as ts

INFL = 112
NODE, NODE_END, CONFIRM = 116, 200, 208
OP_CHOICE = (ts.DecisionType.SELECT_PLAY_MODE, ts.DecisionType.SELECT_OP_MODE)


def _settle(s: ts.GameState) -> None:
    while (not ts.Engine.is_terminal(s) and s.ctx().decision_player == ts.Player.NONE
           and s.ctx().decision_type == ts.DecisionType.ROLL_DIE):
        ts.Engine.step(s, ts.MicroAction(ts.DecisionType.ROLL_DIE, 0, 0, 0))


def _op_choice_nodes(games: int, seed: int) -> List[ts.GameState]:
    """States at op-choice nodes where E4 offers influence, from random legal play."""
    rng = np.random.default_rng(seed)
    out: List[ts.GameState] = []
    for g in range(games):
        s = ts.GameState()
        ts.Engine.init_game(s, seed + g)
        for _ in range(3000):
            _settle(s)
            if ts.Engine.is_terminal(s):
                break
            m = np.asarray(ts.Engine.get_flat_action_mask(s))
            legal = np.flatnonzero(m)
            if len(legal) == 0:
                break
            if s.ctx().decision_type in OP_CHOICE and m[INFL]:
                out.append(s.clone())
            ts.Engine.step_flat(s, int(rng.choice(legal)), False)
    return out


@pytest.fixture(scope="module")
def nodes() -> List[ts.GameState]:
    found = _op_choice_nodes(60, 2323_000)
    assert len(found) > 500, "the sample must actually reach op-choice nodes"
    kinds = {n.ctx().decision_type for n in found}
    assert kinds == set(OP_CHOICE), "both op-choice node types must be covered"
    return found


def _after_influence(s: ts.GameState) -> Tuple[ts.GameState, np.ndarray]:
    t = s.clone()
    ts.Engine.step_flat(t, INFL, False)
    return t, np.asarray(ts.Engine.get_flat_action_mask(t))


def test_the_merged_mask_is_the_e4_mask_with_influence_replaced(nodes: List[ts.GameState]) -> None:
    for s in nodes:
        e4 = np.asarray(ts.Engine.get_flat_action_mask(s))
        merged = np.asarray(ts.Engine.get_flat_action_mask(s, merged_influence=True))
        t, post = _after_influence(s)
        if t.ctx().decision_type != ts.DecisionType.POINT_NODE:
            # The composition does not exist here (the E4 card-32 dead end): no influence offered.
            assert not merged[NODE:NODE_END].any() and not merged[INFL]
            continue
        # Everything that is not influence is untouched.
        others = np.ones(len(e4), dtype=bool)
        others[NODE:NODE_END] = False
        others[INFL] = False
        assert np.array_equal(merged[others], e4[others])
        # E4 offers no NODE slot at an op-choice node, so these are exactly the placements.
        assert not e4[NODE:NODE_END].any()
        assert np.array_equal(merged[NODE:NODE_END], post[NODE:NODE_END])
        assert bool(merged[INFL]) == bool(post[CONFIRM])


def test_every_composed_action_is_the_two_e4_steps(nodes: List[ts.GameState]) -> None:
    checked = 0
    for s in nodes:
        merged = np.asarray(ts.Engine.get_flat_action_mask(s, merged_influence=True))
        for a in np.flatnonzero(merged):
            a = int(a)
            if not ts.ActionMask.is_merged_influence_action(s, a):
                continue
            composed = s.clone()
            ts.Engine.step_flat(composed, a, False, merged_influence=True)
            reference = s.clone()
            ts.Engine.step_flat(reference, INFL, False)
            ts.Engine.step_flat(reference, CONFIRM if a == INFL else a, False)
            assert composed.raw_bytes() == reference.raw_bytes(), f"action {a} diverges from E4"
            checked += 1
    assert checked > 5000


def test_the_view_off_changes_nothing(nodes: List[ts.GameState]) -> None:
    for s in nodes[:200]:
        assert np.array_equal(np.asarray(ts.Engine.get_flat_action_mask(s)),
                              np.asarray(ts.Engine.get_flat_action_mask(s, merged_influence=False)))
        a = ts.Engine.step_flat
        x, y = s.clone(), s.clone()
        a(x, INFL, False)
        a(y, INFL, False, merged_influence=False)
        assert x.raw_bytes() == y.raw_bytes()


def test_a_merged_node_slot_is_refused_in_the_e4_view(nodes: List[ts.GameState]) -> None:
    """The E4 view never accepts a composed action. (E4's own refusal still clears the per-step
    `last_roll` diagnostic, as it does for any refused step, so only the decision is compared.)"""
    for s in nodes[:100]:
        merged = np.asarray(ts.Engine.get_flat_action_mask(s, merged_influence=True))
        node_slots = np.flatnonzero(merged[NODE:NODE_END])
        if len(node_slots):
            t = s.clone()
            assert not ts.Engine.try_step_flat(t, NODE + int(node_slots[0]), False)
            assert t.ctx().decision_type == s.ctx().decision_type
            assert np.array_equal(np.asarray(ts.Engine.get_flat_action_mask(t)),
                                  np.asarray(ts.Engine.get_flat_action_mask(s)))


def test_a_refused_merged_step_leaves_the_state_untouched(nodes: List[ts.GameState]) -> None:
    """The merged path validates before it steps, so refusal changes nothing -- not even
    `last_roll`. That is the atomicity P23 promises for its two-step actions."""
    refused = 0
    for s in nodes[:300]:
        merged = np.asarray(ts.Engine.get_flat_action_mask(s, merged_influence=True))
        illegal_nodes = [NODE + i for i in range(NODE_END - NODE) if not merged[NODE + i]]
        for a in illegal_nodes[:3]:
            t = s.clone()
            assert not ts.Engine.try_step_flat(t, a, False, merged_influence=True)
            assert t.raw_bytes() == s.raw_bytes()
            refused += 1
    assert refused > 100


def test_a_mixed_game_runs_to_the_end_in_the_batch_runner() -> None:
    """US decides in the merged view, USSR in E4's, in the same games."""
    n = 16
    runner = ts.VectorizedBatchRunner(n, 777)
    runner.set_merged_influence([True] * n, [False] * n)
    rng = np.random.default_rng(5)
    composed_us = 0
    op_choice_ussr = 0
    for _ in range(4000):
        terminals = np.asarray(runner.get_terminals())
        if terminals.all():
            break
        masks = np.asarray(runner.get_action_masks())
        players = np.asarray(runner.get_decision_players())
        actions = []
        for i in range(n):
            legal = np.flatnonzero(masks[i])
            if terminals[i] or len(legal) == 0:
                actions.append(0)
                continue
            a = int(rng.choice(legal))
            if runner.get_state(i).ctx().decision_type in OP_CHOICE:
                if players[i] == int(ts.Player.USSR):
                    # The E4 side must never be offered a composed action.
                    assert not masks[i][NODE:NODE_END].any()
                    op_choice_ussr += 1
                elif NODE <= a < NODE_END:
                    composed_us += 1
            actions.append(a)
        results = runner.step_flat_all(actions, False)
        for i in range(n):
            if not terminals[i] and np.flatnonzero(masks[i]).size:
                assert results[i] in (1, 2), f"env {i} refused a masked action {actions[i]}"
    assert composed_us > 0 and op_choice_ussr > 0, "both views must actually be exercised"
    assert np.asarray(runner.get_terminals()).all(), "every mixed game should finish"
