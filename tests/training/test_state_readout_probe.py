"""A probe must recover an answer that is sitting in front of it.

This test exists because a probe failed that bar and the failure was reported as a finding. A
least-squares-onto-one-hot classifier, handed the noiseless `influence / 10` column the engine
puts in board slot 0, scored 79% against a 72% constant baseline -- closing 24% of the gap on a
perfectly informative feature. "The trunk closes about a fifth of the gap" was that ceiling, not
the network.

So any estimator used to answer "can X be read off Y" is gated here first: feed it the known
answer, require near-perfect recovery. A probe that cannot pass this cannot be trusted to
distinguish a representation that lost the information from one that kept it.
"""
from __future__ import annotations

import numpy as np
import pytest

from ai.eval.state_readout import MAX_LEVEL, multinomial_accuracy

#: The measured battleground influence histogram: ~72% zeros with a tail. The imbalance is the
#: point -- it is what a least-squares indicator fit mishandles.
LEVEL_SHARES = np.array([0.72, 0.117, 0.046, 0.065, 0.026, 0.012, 0.014])


def _sample(n: int = 6000, seed: int = 0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    lev = rng.choice(np.arange(MAX_LEVEL + 1), size=(n, 1), p=LEVEL_SHARES / LEVEL_SHARES.sum())
    perm = rng.permutation(n)
    return lev, perm[: int(0.7 * n)], perm[int(0.7 * n):]


def test_recovers_a_noiseless_influence_column() -> None:
    """Exactly how the engine encodes influence: slot 0 is `my_influence / 10`."""
    lev, tr, te = _sample()
    acc = multinomial_accuracy(lev.astype(np.float64) / 10.0, lev, tr, te)
    assert acc[0] > 0.99, f"probe recovered only {acc[0]:.3f} of a noiseless answer"


def test_survives_being_buried_in_noise_columns() -> None:
    """The real slice is 26 wide and only one column carries the influence."""
    rng = np.random.default_rng(1)
    lev, tr, te = _sample()
    x = np.hstack([lev.astype(np.float64) / 10.0, rng.normal(size=(len(lev), 25))])
    acc = multinomial_accuracy(x, lev, tr, te)
    assert acc[0] > 0.97, f"probe recovered only {acc[0]:.3f} with 25 noise columns"


def test_pure_noise_does_not_beat_the_constant_baseline() -> None:
    """The other half of the bar: no signal must not read as signal."""
    rng = np.random.default_rng(2)
    lev, tr, te = _sample()
    acc = multinomial_accuracy(rng.normal(size=(len(lev), 26)), lev, tr, te)
    baseline = float((lev[te] == 0).mean())
    assert acc[0] <= baseline + 0.02, f"probe scored {acc[0]:.3f} on noise, baseline {baseline:.3f}"


@pytest.mark.parametrize("shape", ["shared", "per_country"])
def test_both_feature_shapes_are_fitted(shape: str) -> None:
    """(N, D) features shared across countries, and (N, 84, D) per-country ones."""
    rng = np.random.default_rng(3)
    n, n_country = 3000, 4
    lev = rng.choice(np.arange(MAX_LEVEL + 1), size=(n, n_country),
                     p=LEVEL_SHARES / LEVEL_SHARES.sum())
    perm = rng.permutation(n)
    tr, te = perm[:2100], perm[2100:]
    if shape == "per_country":
        x = (lev.astype(np.float64) / 10.0)[:, :, None]
    else:
        x = lev[:, :1].astype(np.float64) / 10.0
    acc = multinomial_accuracy(x, lev, tr, te)
    assert acc.shape == (n_country,)
    assert acc[0] > 0.99
