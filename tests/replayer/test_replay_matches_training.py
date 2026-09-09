"""Replays must reproduce the games training actually plays.

Three code paths reach the network -- the vectorized training env, the replay generator,
and the bot loop -- and they had drifted apart. Two defects came out of that drift:

* the replay generator stepped with ``step_flat`` and never resolved the ROLL_DIE chance
  nodes that follow, which the vectorized runner does internally. For Summit (#45) the two
  are not equivalent: they leave a different ``phasing_player``, and since that decides who
  acts next and who loses a DEFCON-1 ending, games collapsed to roughly half their length.
* ``play_match`` bypassed the generator entirely for its bot loop.

These tests pin the stepping contract rather than any particular policy's behaviour, so
they stay meaningful as the agent changes.
"""

import numpy as np
import pytest
import ts_engine as ts

from bindings.action_encoder import ActionEncoder


def _drain(state: ts.GameState) -> None:
    while (not ts.Engine.is_terminal(state)
           and state.ctx().decision_player == ts.Player.NONE
           and state.ctx().decision_type == ts.DecisionType.ROLL_DIE):
        ts.Engine.step(state, ts.MicroAction(ts.DecisionType.ROLL_DIE, 0, 0, 0))


def test_flat_action_at_a_chance_node_now_matches_resolving_it() -> None:
    """The trap this file was written for is gone; the two paths agree everywhere.

    They used to disagree at Summit (#45) on phasing_player. That divergence came from
    allow_early_stop leaking into Summit's branch decision from whatever decision preceded
    it, so the two ways of resolving the chance node could reach different masks. Every
    decision now sets that flag deliberately instead of inheriting it, and across 790 chance
    nodes in 120 seeded games the two paths produce identical states.

    The assertion is inverted rather than deleted, so that a future change reintroducing the
    divergence is caught. The drain in the replay converter is now belt-and-braces rather
    than load-bearing, and is left in place.
    """
    mismatches = 0
    checked = 0
    for seed in range(2000, 2120):
        st = ts.GameState()
        ts.Engine.init_game(st, seed)
        for _ in range(600):
            if ts.Engine.is_terminal(st):
                break
            if st.ctx().decision_type == ts.DecisionType.ROLL_DIE:
                legal = np.flatnonzero(ActionEncoder.get_legal_mask(st))
                a, b = st.clone(), st.clone()
                ts.Engine.step_flat(a, int(legal[0]) if len(legal) else 211)
                ts.Engine.step(b, ts.MicroAction(ts.DecisionType.ROLL_DIE, 0, 0, 0))
                checked += 1
                if a.to_json() != b.to_json():
                    mismatches += 1
                ts.Engine.step(st, ts.MicroAction(ts.DecisionType.ROLL_DIE, 0, 0, 0))
                continue
            legal = np.flatnonzero(ActionEncoder.get_legal_mask(st))
            if not len(legal):
                break
            ts.Engine.step_flat(st, int(legal[0]))
    assert checked > 100, "fixture reached too few chance nodes to be meaningful"
    assert mismatches == 0, (
        f"{mismatches} of {checked} chance nodes resolve differently through step_flat than "
        "through an explicit ROLL_DIE; the two were made equivalent by giving every decision "
        "its own allow_early_stop instead of inheriting the previous one's"
    )


def test_draining_chance_nodes_lengthens_games() -> None:
    """A policy-independent check that the stepping contract matters."""
    def play(seed: int, drain: bool) -> int:
        st = ts.GameState()
        ts.Engine.init_game(st, seed)
        for _ in range(2500):
            if ts.Engine.is_terminal(st):
                break
            legal = np.flatnonzero(ActionEncoder.get_legal_mask(st))
            if not len(legal):
                break
            ts.Engine.step_flat(st, int(legal[0]))
            if drain:
                _drain(st)
        return int(st.turn)

    seeds = range(3000, 3030)
    drained = [play(s, True) for s in seeds]
    undrained = [play(s, False) for s in seeds]
    assert np.mean(drained) >= np.mean(undrained), (
        f"draining should not shorten games: drained {np.mean(drained):.2f} "
        f"vs undrained {np.mean(undrained):.2f}"
    )


def test_generator_never_asks_the_policy_at_a_chance_node(tmp_path) -> None:
    """The contract, stated policy-independently.

    A chance node is not a decision. If any logged step carries decision_type ROLL_DIE
    then the generator queried the network there instead of resolving it, which is exactly
    the defect that halved replay game length.
    """
    pytest.importorskip("torch")
    from ai.models.coldwar_net import create_coldwar_net
    from tools.lib.self_play import generate_self_play_replay

    # Write into tmp_path: the workbench lists data/replays non-recursively, and a test
    # artefact appearing beside real games is how a stale replay gets reviewed by mistake.
    log, _ = generate_self_play_replay(
        model=create_coldwar_net("cpu"), seed=91004, temperature=0.3,
        game_id="pytest_replay_contract", device="cpu", verbose=False,
        output_path=str(tmp_path / "pytest_replay_contract.tslog.json"),
    )
    steps = log.get("steps", [])
    assert steps, "generator produced no steps"

    roll_die = int(ts.DecisionType.ROLL_DIE)
    offenders = [s["step_index"] for s in steps
                 if int(s.get("action", {}).get("decision_type", -1)) == roll_die]
    assert not offenders, (
        f"{len(offenders)} logged steps were chance nodes handed to the policy "
        f"(first at step {offenders[:3]}); they must be resolved instead"
    )
