"""Small statistical helpers shared by the evaluation probes.

Torch-free and dependency-free on purpose: `bindings/ts_env.py` and the probes import from here
on the training path, and `ai/eval/__init__` pulls in torch and the model zoo.
"""

from typing import Final, Tuple

#: 1.96 standard deviations, i.e. a 95% interval.
Z95: Final[float] = 1.959963984540054


def wilson_interval(successes: int, trials: int, z: float = Z95) -> Tuple[float, float]:
    """A binomial confidence interval for `successes / trials`, by the Wilson score method.

    Wilson rather than the normal approximation, because these rates live exactly where the
    normal approximation is worst: at 1 of 204 chances, `p +/- z*sqrt(p(1-p)/n)` reaches below
    zero, and at 0 of n it collapses to the single point 0.0, which reads as certainty from no
    evidence at all. Wilson stays inside [0, 1] and keeps a sensible width at both ends.

    With no trials the interval is the whole of [0, 1]: nothing has been observed, so nothing is
    excluded. That is the honest answer, and it is why the interval can replace the separate
    numerator and denominator series on a chart -- a rule the policy never had the chance to
    break shows as a band spanning everything rather than as a reassuring 0.0.
    """
    if trials <= 0:
        return 0.0, 1.0
    n = float(trials)
    p = successes / n
    denom = 1.0 + z * z / n
    centre = (p + z * z / (2.0 * n)) / denom
    half = (z / denom) * ((p * (1.0 - p) / n + z * z / (4.0 * n * n)) ** 0.5)
    return max(0.0, centre - half), min(1.0, centre + half)
