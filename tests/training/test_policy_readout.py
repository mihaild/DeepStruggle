"""Guards for the policy readout that the replay trace is built from.

The load-bearing one is `test_read_policy_samples_exactly_like_sample_action`. `read_policy`
duplicates `sample_action`'s three lines of sampling rather than replacing them, because
`sample_action` is on the training hot path. If the duplicate ever drifts, a traced replay
becomes a *different game* from the untraced one with the same seed, and every comparison
between a replay and a training rollout quietly changes meaning -- with nothing failing.
"""
from __future__ import annotations

import numpy as np
import pytest
import torch
import ts_engine as ts

from ai.eval.policy_readout import TRACE_TOP_K, read_policy, unasked_readout
from ai.models.coldwar_net_v2 import create_coldwar_net_v2
from bindings.action_encoder import ActionEncoder


@pytest.fixture(scope="module")
def model():
    torch.manual_seed(11)
    m = create_coldwar_net_v2("cpu")
    m.eval()
    return m


def _node(seed: int = 7, steps: int = 0):
    """A real decision node: observation, legal mask, and the state they came from."""
    state = ts.GameState()
    ts.Engine.init_game(state, seed)
    for _ in range(steps):
        mask = np.asarray(ActionEncoder.get_legal_mask(state))
        legal = np.flatnonzero(mask)
        if not len(legal):
            break
        ts.Engine.step_flat(state, int(legal[0]))
    p = (state.ctx().decision_player if state.ctx().decision_player != ts.Player.NONE
         else state.phasing_player)
    obs = torch.from_numpy(
        np.asarray(ts.extract_observation(state, p), dtype=np.float32).reshape(1, -1)).float()
    mask_t = torch.from_numpy(np.asarray(ActionEncoder.get_legal_mask(state)).reshape(1, -1))
    return state, obs, mask_t


@pytest.mark.parametrize("temperature", [1.0, 0.3, 0.05])
def test_read_policy_samples_exactly_like_sample_action(model, temperature: float) -> None:
    """Same seed, same node, same draw -- or a traced game is not the game it traced."""
    _state, obs, mask = _node()

    torch.manual_seed(2024)
    expected, _lp, _vw, _vvp, _ent = model.sample_action(
        obs, mask, temperature=temperature, deterministic=False)

    torch.manual_seed(2024)
    got, _readout = read_policy(model, obs, mask, temperature=temperature, deterministic=False)

    assert got == int(expected.item())


def test_read_policy_deterministic_matches_argmax(model) -> None:
    _state, obs, mask = _node()
    torch.manual_seed(1)
    expected, *_ = model.sample_action(obs, mask, temperature=1.0, deterministic=True)
    got, readout = read_policy(model, obs, mask, deterministic=True)
    assert got == int(expected.item()) == readout["argmax_idx"]
    assert readout["p_chosen_sampled"] == 1.0


def test_the_listed_mass_plus_the_tail_is_the_whole_distribution(model) -> None:
    state, obs, mask = _node()
    _idx, readout = read_policy(model, obs, mask, state=state)
    listed = sum(e["p"] for e in readout["top"])
    assert listed + readout["p_tail"] == pytest.approx(1.0, abs=2e-4)
    assert readout["p_tail"] >= 0.0


def test_top_is_descending_legal_and_named(model) -> None:
    state, obs, mask = _node()
    _idx, readout = read_policy(model, obs, mask, state=state)
    legal = set(np.flatnonzero(np.asarray(mask[0])).tolist())
    ps = [e["p"] for e in readout["top"]]
    assert ps == sorted(ps, reverse=True)
    assert all(e["idx"] in legal for e in readout["top"])
    assert all(e.get("name") for e in readout["top"])
    assert readout["n_legal"] == len(legal)
    # The default lists every legal action: the workbench paints each probability on its own
    # card, button or country, so a truncated distribution leaves most of the board blank.
    assert len(readout["top"]) == readout["n_legal"]


def test_the_chosen_action_is_listed_even_below_the_floor(model) -> None:
    """Under a cap, the 0.001 choice is the one worth looking at -- it must survive the floor."""
    state, obs, mask = _node()
    # A floor above every probability: only the chosen action can remain.
    _idx, readout = read_policy(model, obs, mask, state=state, top_k=12, p_floor=1.1)
    assert [e["idx"] for e in readout["top"]] == [readout["chosen_idx"]]
    assert readout["p_tail"] == pytest.approx(1.0 - readout["p_chosen"], abs=2e-4)


def test_include_forces_an_action_into_the_listing(model) -> None:
    state, obs, mask = _node()
    legal = np.flatnonzero(np.asarray(mask[0])).tolist()
    target = int(legal[-1])
    _idx, readout = read_policy(model, obs, mask, state=state, top_k=12, p_floor=1.1,
                                include=target)
    assert target in [e["idx"] for e in readout["top"]]


def test_the_default_lists_every_legal_action(model) -> None:
    """TRACE_TOP_K is 0 -- no cap -- so every option gets a number of its own."""
    assert TRACE_TOP_K == 0
    state, obs, mask = _node()
    _idx, readout = read_policy(model, obs, mask, state=state)
    assert len(readout["top"]) == readout["n_legal"]
    assert readout["p_tail"] == 0.0


def test_a_cap_truncates_and_reports_the_missing_mass(model) -> None:
    """The cap still works for someone who wants a smaller file."""
    state, obs, mask = _node(steps=0)
    _idx, readout = read_policy(model, obs, mask, state=state, top_k=3)
    if readout["n_legal"] > 4:
        assert len(readout["top"]) <= 4          # +1: the chosen action is never dropped
        assert readout["p_tail"] > 0.0


def test_temperature_changes_the_sampling_probability_not_the_belief(model) -> None:
    """`p_chosen` is the model's own distribution; only `p_chosen_sampled` follows temperature."""
    state, obs, mask = _node()
    torch.manual_seed(3)
    _i1, hot = read_policy(model, obs, mask, temperature=1.0, state=state)
    torch.manual_seed(3)
    _i2, cold = read_policy(model, obs, mask, temperature=0.05, state=state)
    same = hot["chosen_idx"] == cold["chosen_idx"]
    if same:
        assert hot["p_chosen"] == pytest.approx(cold["p_chosen"])
        assert cold["p_chosen_sampled"] != pytest.approx(hot["p_chosen_sampled"])
    assert hot["temperature"] == 1.0 and cold["temperature"] == 0.05


def test_a_forced_step_is_certain_and_a_scripted_one_is_not(model) -> None:
    """A node with one option is certain; an overridden choice has no distribution to report."""
    state, _obs, _mask = _node()
    forced = unasked_readout(5, state)
    assert forced["source"] == "forced" and forced["p_chosen"] == 1.0
    assert forced["top"][0]["idx"] == 5

    scripted = unasked_readout(5, state, source="scripted", n_legal=40)
    assert scripted["source"] == "scripted" and scripted["n_legal"] == 40
    # No invented certainty: the policy was overruled, not consulted.
    assert "p_chosen" not in scripted and "top" not in scripted


def test_a_bot_that_answers_without_the_network_reports_no_readout() -> None:
    """A stale readout attached to the next node would describe a different decision.

    `NeuralBot.select_action` short-circuits when there is nothing to choose between, so the
    field has to be cleared on entry rather than only on the path that fills it.
    """
    from bot.neural_bot import NeuralBot

    bot = NeuralBot("US", model_path=None, trace=True)
    bot.last_readout = {"source": "policy", "chosen_idx": 99}   # left over from an earlier node

    action = bot.select_action({}, {"valid_ids": [], "decision_type": 5})

    assert action is not None and action["flags"] == 0x80      # the CONFIRM_DONE fallback
    assert bot.last_readout is None


def test_a_batch_is_refused(model) -> None:
    _state, obs, mask = _node()
    with pytest.raises(ValueError, match="one node at a time"):
        read_policy(model, obs.repeat(2, 1), mask.repeat(2, 1))
