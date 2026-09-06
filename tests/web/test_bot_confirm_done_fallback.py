"""Regression tests for the CONFIRM_DONE fallback all bots must honor.

The engine guarantees that whenever a POINT_NODE (or any other) decision has no legal
targets, CONFIRM_DONE (flags bit 0x80) is legal -- its own anti-deadlock safety net
(engine/src/action_mask.cpp), independent of allow_early_stop. Several bots instead
special-cased on allow_early_stop and either returned None (crashing their caller) or
synthesized a bogus {"primary_id": 0, "flags": 0} action, which for POINT_NODE actually
means "target country 0", not a pass -- the engine legally rejects it since valid_ids is
empty, silently stalling the decision.

This gap was latent (reachable via Truman Doctrine/Muslim Revolution, which have always
been mandatory) until Suez Crisis and East European Unrest were also made mandatory
(allow_early_stop=0), which made it common enough to break test_bot_vs_bot_simulation
deterministically.
"""

import pytest

import ts_engine as ts
from bot.event_heavy_bot import EventHeavyBot
from bot.exploratory_bot import ExploratoryBot
from bot.heuristic_bot import HeuristicBot
from bot.neural_bot import NeuralBot
from bot.random_bot import RandomBot
from bot.strategic_bot import StrategicBot

CONFIRM_DONE = 0x80

# UK, France, Israel -- the three Suez Crisis targets (see test_game_invariants.py).
SUEZ_COUNTRIES = (1, 8, 23)


def _is_confirm_done(action):
    return action is not None and (action.get("flags", 0) & CONFIRM_DONE) != 0


SELECT_ACTION_BOTS = {
    "RandomBot": lambda: RandomBot(role="US"),
    "HeuristicBot": lambda: HeuristicBot(role="US"),
    "ExploratoryBot": lambda: ExploratoryBot(role="US", rng_seed=1),
    "NeuralBot": lambda: NeuralBot(role="US", device="cpu"),
}

CHOOSE_ACTION_BOTS = {
    "EventHeavyBot": lambda: EventHeavyBot(role="US", rng_seed=1),
    "StrategicBot": lambda: StrategicBot(role="US", rng_seed=1),
}


@pytest.mark.parametrize("bot_name", list(SELECT_ACTION_BOTS))
@pytest.mark.parametrize("allow_early_stop", [False, True])
def test_select_action_confirms_done_with_no_legal_targets(bot_name, allow_early_stop):
    bot = SELECT_ACTION_BOTS[bot_name]()
    legal_actions = {
        "decision_type": int(ts.DecisionType.POINT_NODE),
        "valid_ids": [],
        "allow_early_stop": allow_early_stop,
    }
    action = bot.select_action({}, legal_actions)
    assert _is_confirm_done(action), f"{bot_name} (allow_early_stop={allow_early_stop}) returned {action!r}"


@pytest.mark.parametrize("bot_name", list(CHOOSE_ACTION_BOTS))
@pytest.mark.parametrize("allow_early_stop", [False, True])
def test_choose_action_confirms_done_with_no_legal_targets(bot_name, allow_early_stop):
    bot = CHOOSE_ACTION_BOTS[bot_name]()
    legal_actions = {
        "decision_type": int(ts.DecisionType.POINT_NODE),
        "valid_ids": [],
        "allow_early_stop": allow_early_stop,
    }
    action, _, _ = bot.choose_action({}, legal_actions)
    assert _is_confirm_done(action), f"{bot_name} (allow_early_stop={allow_early_stop}) returned {action!r}"


def test_heuristic_bot_closes_out_suez_crisis_with_no_targets():
    """End-to-end regression for the original test_bot_vs_bot_simulation failure:
    Suez Crisis triggered with zero eligible targets must be closeable by a bot driven
    purely off the engine's own serialized legal_actions, not a hand-built fixture."""
    state = ts.GameState()
    ts.Engine.init_game(state, 42)
    state.current_phase = ts.Phase.ACTION_ROUND
    for cid in SUEZ_COUNTRIES:
        state.set_country(cid, 0, state.get_country(cid).ussr_influence)

    ts.CardHandlers.trigger_event(state, 28, ts.Player.USSR)  # Suez Crisis
    assert state.ctx().resolving_card == 28

    legal_actions = state.to_dict()["legal_actions"]
    assert legal_actions["valid_ids"] == []

    bot = HeuristicBot(role="USSR")
    action = bot.select_action(state.to_dict(), legal_actions)
    assert _is_confirm_done(action)
    assert action is not None  # narrows for the type checker; _is_confirm_done already checked this

    micro_action = ts.MicroAction(
        ts.DecisionType(action["decision_type"]),
        action["primary_id"],
        action["secondary_id"],
        action["flags"],
    )
    assert ts.Engine.step(state, micro_action)
    assert state.ctx().resolving_card == 0
