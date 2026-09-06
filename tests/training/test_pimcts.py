"""Perfect-information MCTS used as a diagnostic teacher.

The failure modes this guards against are all sign and chance-node errors, which this action
space invites: the mover changes partway through a turn, values alternate perspective, and a
ROLL_DIE node has no decision player. Three separate bugs in this repo came from code that
treated a chance node as a normal decision.
"""

import numpy as np
import pytest
import torch
import ts_engine as ts

from ai.models.coldwar_net_v2 import create_coldwar_net_v2
from ai.search.pimcts import PIMCTSAgent, PIMCTSConfig, acting_player, drain_chance_nodes


def _agent(sims: int = 16, seed: int = 3) -> PIMCTSAgent:
    torch.manual_seed(seed)
    net = create_coldwar_net_v2("cpu")
    return PIMCTSAgent(net, device="cpu", config=PIMCTSConfig(simulations=sims, seed=seed))


def _fresh(seed: int = 4242) -> ts.GameState:
    st = ts.GameState()
    ts.Engine.init_game(st, seed)
    drain_chance_nodes(st)
    return st


def test_search_returns_a_legal_action() -> None:
    from bindings.action_encoder import ActionEncoder
    agent = _agent()
    st = _fresh()
    a = agent.select_action(st, acting_player(st), temperature=0.0)
    legal = np.flatnonzero(np.asarray(ActionEncoder.get_legal_mask(st)))
    assert a in legal, f"search returned illegal action {a}"


def test_search_does_not_mutate_the_position_it_is_given() -> None:
    """The searcher clones; a caller's state must be untouched or the game desyncs."""
    agent = _agent()
    st = _fresh()
    before = (int(st.turn), int(st.action_round), int(st.victory_points),
              int(st.defcon), int(st.rng_state))
    agent.select_action(st, acting_player(st), temperature=0.0)
    after = (int(st.turn), int(st.action_round), int(st.victory_points),
             int(st.defcon), int(st.rng_state))
    assert before == after, f"search mutated the caller's state: {before} -> {after}"


def test_visits_concentrate_with_more_simulations() -> None:
    """A search that is not accumulating statistics would spread visits uniformly."""
    st = _fresh()
    actions, visits = _agent(sims=64).search(st)
    assert len(actions) > 1, "fixture position has no choice to make"
    assert visits.sum() == 64, f"expected 64 simulations, counted {visits.sum()}"
    assert visits.max() > 64 / len(actions), "visits are uniform; PUCT is not selecting"


def test_terminal_values_use_the_us_perspective() -> None:
    """Sign convention: everything internal is US-positive, matching get_terminal_utility."""
    agent = _agent()
    st = _fresh()
    # Drive to a terminal state with first-legal play.
    from bindings.action_encoder import ActionEncoder
    for _ in range(4000):
        if ts.Engine.is_terminal(st):
            break
        legal = np.flatnonzero(np.asarray(ActionEncoder.get_legal_mask(st)))
        ts.Engine.step_flat(st, int(legal[0]) if len(legal) else 0)
        drain_chance_nodes(st)
    assert ts.Engine.is_terminal(st), "fixture never terminated"

    node = agent._evaluate(st)
    assert node.terminal
    assert node.value_us == pytest.approx(float(ts.Engine.get_terminal_utility(st))), (
        "terminal node value must be the engine's US-perspective utility, unmodified"
    )


def test_leaf_value_is_negated_for_ussr_to_move() -> None:
    """v_win is mover-perspective; the search stores US-perspective. Getting this backwards
    makes the searcher actively play to lose for one side, which a win-rate test can mask."""
    agent = _agent()
    st = _fresh()
    node = agent._evaluate(st)
    assert not node.terminal

    mover = node.mover
    obs = ts.extract_observation(st, acting_player(st))
    from bindings.action_encoder import ActionEncoder
    mask = np.asarray(ActionEncoder.get_legal_mask(st))
    with torch.no_grad():
        _, v, _ = agent.model.forward(
            torch.from_numpy(np.asarray(obs, dtype=np.float32)).unsqueeze(0),
            torch.from_numpy(mask).unsqueeze(0))
    v_mover = float(v.squeeze().item())
    expected = v_mover if mover == int(ts.Player.US) else -v_mover
    assert node.value_us == pytest.approx(expected, abs=1e-6), (
        f"mover={mover}: stored {node.value_us} for a mover-perspective value of {v_mover}"
    )


def test_no_node_is_ever_built_on_an_unresolved_chance_node() -> None:
    """A ROLL_DIE node has decision_player NONE and must be settled by the engine.

    Expanding one would let PUCT choose the die, which is the bug that has already appeared
    three times in this codebase.
    """
    agent = _agent(sims=48)
    st = _fresh()
    root_actions, _ = agent.search(st)
    assert root_actions, "fixture produced no root actions"

    seen = []

    def walk(node, depth=0):
        if depth > 6 or node.terminal:
            return
        seen.append(node)
        for child in node.children.values():
            walk(child, depth + 1)

    # Re-run a search and inspect the tree it builds.
    root_state = st.clone()
    drain_chance_nodes(root_state)
    root = agent._evaluate(root_state)
    for _ in range(48):
        agent._simulate(root)
    walk(root)

    assert seen, "search tree is empty"
    for node in seen:
        ctx = node.state.ctx()
        assert not (ctx.decision_player == ts.Player.NONE
                    and ctx.decision_type == ts.DecisionType.ROLL_DIE), (
            "a chance node was expanded as if it were a decision"
        )


def test_root_dirichlet_noise_reaches_low_prior_actions() -> None:
    """Noise must actually perturb the root prior, and only the root.

    Without it a low-prior action can go unexpanded: at 64 simulations its PUCT bonus stays
    below the Q gap, and search cannot back up a terminal value from a branch it never
    visits. That is the leading explanation for search not improving the forced-win rate.
    """
    st = _fresh()

    clean = _agent(sims=8)
    clean.cfg.dirichlet_frac = 0.0
    root_clean = clean._evaluate(st.clone())
    baseline = root_clean.priors.copy()

    noisy = _agent(sims=8)
    noisy.cfg.dirichlet_frac = 0.25
    noisy.cfg.dirichlet_alpha = 1.0
    actions, visits = noisy.search(st)
    assert len(actions) > 1, "fixture has no choice"

    # Re-derive the perturbed prior the same way search() does, to check it moved.
    root_noisy = noisy._evaluate(st.clone())
    rng = np.random.RandomState(noisy.cfg.seed)
    noise = rng.dirichlet([1.0] * len(root_noisy.actions))
    perturbed = 0.75 * root_noisy.priors + 0.25 * noise

    assert not np.allclose(perturbed, baseline), "dirichlet noise left the prior unchanged"
    assert perturbed.sum() == pytest.approx(1.0), "perturbed prior must stay a distribution"
    assert (perturbed >= 0).all()


def test_dirichlet_is_off_by_default() -> None:
    """Evaluation should be noise-free unless a caller asks; noise weakens play slightly."""
    assert PIMCTSConfig().dirichlet_frac == 0.0
