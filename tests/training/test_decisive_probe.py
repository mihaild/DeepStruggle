"""The decisive probe must see whole games, and batching must not change what it measures.

measure_decisive stepped chance nodes with a policy action instead of resolving them --
the same defect test_replay_matches_training.py documents for the replay generator. It is
not equivalent: it leaves a different phasing_player and collapses games to roughly half
their length. Every decisive number taken through this path was therefore measured on
truncated games, which understated the avoidable-loss avoid rate badly (74.6% against a
true 92.9%).
"""

import numpy as np
import pytest
import ts_engine as ts

from ai.eval.decisive_probe import measure_decisive
from bindings.action_encoder import ActionEncoder


def _first_legal(state: ts.GameState, player: ts.Player) -> int:
    legal = np.flatnonzero(ActionEncoder.get_legal_mask(state))
    return int(legal[0]) if len(legal) else 211


def test_probe_never_hands_a_chance_node_to_the_policy() -> None:
    """Chance nodes must be resolved, never stepped with a policy action.

    step_flat at a ROLL_DIE node is not equivalent to resolving it -- see
    test_replay_matches_training.py, which pins that non-equivalence directly. Asking the
    policy to "choose" there both wastes a forward pass and puts the probe on a different
    trajectory from the one training generates: measured on the trained checkpoint, games
    ran to roughly half their proper length, and the avoidable-loss avoid rate read 74.6%
    against a true 92.9%.

    Asserting on game length would need a policy that happens to reach the divergent
    cards. This asserts the property itself, so it holds for any policy.
    """
    seen_chance = []

    def spy(state: ts.GameState, player: ts.Player) -> int:
        if state.ctx().decision_type == ts.DecisionType.ROLL_DIE:
            seen_chance.append(int(state.turn))
        return _first_legal(state, player)

    stats = measure_decisive(spy, num_games=4, base_seed=51000)

    assert stats.decisions > 0, "probe recorded nothing; the fixture is not exercising it"
    assert not seen_chance, (
        f"the policy was asked to act at {len(seen_chance)} chance nodes "
        f"(turns {sorted(set(seen_chance))}); they must be resolved instead"
    )


def test_batched_probe_agrees_with_the_single_state_one() -> None:
    """Batching is an efficiency change; it must not move the measurement."""
    torch = pytest.importorskip("torch")
    from ai.eval.decisive_probe import measure_decisive_batched
    from ai.models.coldwar_net_v2 import create_coldwar_net_v2

    net = create_coldwar_net_v2("cpu")
    batched = measure_decisive_batched(net, num_envs=8, base_seed=51000)

    assert batched.decisions > 0, "no decisions recorded"
    # Rates are only meaningful when the situation actually arose.
    if batched.loss_avoidable:
        assert 0.0 <= batched.loss_avoid_rate <= 1.0
    if batched.win_available:
        assert 0.0 <= batched.win_take_rate <= 1.0
    assert batched.loss_taken <= batched.loss_avoidable
    assert batched.win_taken <= batched.win_available


def test_probe_measures_every_env_exactly_once() -> None:
    """The sample must be one episode per env, not the first N episodes to finish.

    The probe used to stop once num_episodes episodes had completed, with num_episodes well
    below num_envs, so the sample was the fastest N of num_envs games. Its own docstring
    named why that matters -- decisive positions cluster near the end of a game -- and kept
    the bug anyway. It mattered: on the control's final checkpoint the biased sample put the
    instant-win take rate at 89.5%, while measuring one episode per env puts it at 76.6%.
    Nearly a quarter of forced wins are missed, not a tenth.
    """
    from ai.eval.decisive_probe import measure_decisive_batched
    from ai.models.coldwar_net_v2 import create_coldwar_net_v2

    num_envs = 12
    stats = measure_decisive_batched(create_coldwar_net_v2("cpu"), num_envs=num_envs,
                                     base_seed=52_000, max_iters=20_000)

    assert stats.episodes == num_envs, (
        f"measured {stats.episodes} episodes across {num_envs} envs; the sample is "
        f"length-selected unless it is exactly one episode per env"
    )
    assert stats.decisions > 0, "no decisions examined; the fixture proves nothing"


def test_a_die_dependent_win_is_not_labelled_forced() -> None:
    """A chance node is decisive only if every outcome agrees.

    _probe_forced used to step a ROLL_DIE node once and trust the sampled result, so a card
    that awards VP on a successful roll -- Brush War and similar -- was labelled a forced win
    whenever that one roll happened to succeed. Measured against the engine, 6% of "win"
    labels were die-dependent that way, at 9/16 to 14/16 win rates. Roll-independent lines
    are unaffected: a coup at DEFCON 2 degrades DEFCON whatever the roll.
    """
    import random

    import numpy as np
    import ts_engine as ts
    from ai.eval.safety import classify_legal_actions
    from ai.search.pimcts import acting_player, drain_chance_nodes

    rng = random.Random(11)
    checked = 0

    for g in range(60):
        st = ts.GameState()
        ts.Engine.init_game(st, 610_000 + g)
        drain_chance_nodes(st)
        for _ in range(600):
            if ts.Engine.is_terminal(st) or checked >= 40:
                break
            mover = acting_player(st)
            for a, kind in classify_legal_actions(st, mover).items():
                if kind != "win":
                    continue
                checked += 1
                # Every die stream must agree that this ends the game in the mover's favour.
                for _ in range(12):
                    probe = st.clone()
                    probe.rng_state = rng.getrandbits(64) % (1 << 64)
                    ts.Engine.step_flat(probe, int(a))
                    for _ in range(8):
                        if ts.Engine.is_terminal(probe):
                            break
                        pctx = probe.ctx()
                        if pctx.decision_type == ts.DecisionType.ROLL_DIE:
                            ts.Engine.step(probe, ts.MicroAction(ts.DecisionType.ROLL_DIE, 0, 0, 0))
                            continue
                        legal = np.flatnonzero(ActionEncoder.get_legal_mask(probe))
                        if len(legal) != 1:
                            break
                        ts.Engine.step_flat(probe, int(legal[0]))
                    if ts.Engine.is_terminal(probe):
                        u = float(ts.Engine.get_terminal_utility(probe))
                        assert (u > 0) == (mover == ts.Player.US), (
                            f"action {a} labelled 'win' for "
                            f"{'US' if mover == ts.Player.US else 'USSR'} but a die stream "
                            f"ended the game against them"
                        )
            legal = np.flatnonzero(ActionEncoder.get_legal_mask(st))
            if len(legal) == 0:
                break
            ts.Engine.step_flat(st, int(rng.choice(list(legal))))
            drain_chance_nodes(st)
        if checked >= 40:
            break

    assert checked > 0, "no 'win' labels encountered; the fixture proves nothing"
