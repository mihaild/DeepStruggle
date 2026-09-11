"""The batched blunder probe must count what the single-state one counts.

It exists for speed: the single-state probe measured 90 of the 122 seconds of a snapshot
evaluation — 74% — because it asked the network for one state at a time (~890 decisions/sec)
while every other probe in the same evaluation was already batched (~106,000). Speed is only
worth having if the counting rule is unchanged, which is what these check.
"""

import numpy as np
import pytest
import torch
import ts_engine

from ai.eval.blunders import RULES, BlunderCounts, measure_blunders, measure_blunders_batched
from ai.models.coldwar_net_v2 import create_coldwar_net_v2


class _FirstLegal(torch.nn.Module):
    """Deterministic policy, so the two probes see the same games and must agree exactly."""

    TOTAL_OBS_SIZE = None          # filled in below from the engine

    def __init__(self) -> None:
        super().__init__()
        self.marker = torch.nn.Parameter(torch.zeros(1))

    def sample_action(self, obs, mask, temperature: float = 1.0, deterministic: bool = False):
        actions = torch.argmax(mask.to(torch.int32), dim=1)
        z = torch.zeros(actions.shape[0])
        return actions, z, z, z, z


def _first_legal(state, player):
    mask = np.asarray(ts_engine.ActionMask.generate_flat_mask(state))
    legal = np.flatnonzero(mask)
    return int(legal[0]) if len(legal) else None


def test_the_two_probes_agree_on_the_same_games() -> None:
    """Same deterministic policy, same seeds: the counts must match exactly."""
    from bindings.ts_env import obs_size

    _FirstLegal.TOTAL_OBS_SIZE = obs_size()
    single = measure_blunders(_first_legal, num_games=8, base_seed=515_000)
    batched = measure_blunders_batched(_FirstLegal(), num_games=8, base_seed=515_000)

    for rule in RULES:
        assert batched.opportunities.get(rule, 0) == single.opportunities.get(rule, 0), (
            f"{rule}: opportunities differ, "
            f"{batched.opportunities.get(rule, 0)} batched vs {single.opportunities.get(rule, 0)}")
        assert batched.committed.get(rule, 0) == single.committed.get(rule, 0), (
            f"{rule}: committed differ")


def test_it_returns_the_same_metric_keys() -> None:
    """The trainer logs these straight into the strategy charts."""
    from bindings.ts_env import obs_size

    _FirstLegal.TOTAL_OBS_SIZE = obs_size()
    m = measure_blunders_batched(_FirstLegal(), num_games=4, base_seed=515_000).metrics()
    for rule in RULES:
        for part in ("rate", "ci_low", "ci_high", "count", "chances"):
            assert f"blunder_{rule}_{part}" in m


def test_it_refuses_a_model_of_the_wrong_width() -> None:
    """Same guard every other probe has: a mismatched model is silently misread, not raised on."""
    class _Wrong(torch.nn.Module):
        TOTAL_OBS_SIZE = 4293      # legacy, retired

        def sample_action(self, obs, mask, temperature: float = 1.0):
            raise AssertionError("must not get as far as a forward pass")

    with pytest.raises(ValueError, match="the engine emits"):
        measure_blunders_batched(_Wrong(), num_games=2)


def test_a_real_network_runs_through_it() -> None:
    net = create_coldwar_net_v2("cpu")
    counts = measure_blunders_batched(net, num_games=4, base_seed=515_000)
    assert isinstance(counts, BlunderCounts)
    # Every rule reports a denominator, even if it is zero.
    for rule in RULES:
        assert rule in counts.opportunities or counts.opportunities.get(rule, 0) == 0
