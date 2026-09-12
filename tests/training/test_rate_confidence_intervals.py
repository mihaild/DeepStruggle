"""Rates are charted with a binomial confidence interval instead of a numerator and denominator.

The interval is not decoration: it is what lets a rate be read on its own. A blunder rule with
three chances and a rule with four hundred produce rates that look alike and mean very different
things, and the band is where that difference shows.
"""

import math

import pytest

from ai.eval.blunders import RULES, Blunder, BlunderCounts
from ai.eval.decisive_probe import DecisiveStats
from ai.stats import Z95, wilson_interval


def test_the_interval_brackets_the_rate() -> None:
    for successes, trials in [(1, 204), (25, 362), (3, 52), (66, 85), (1794, 1849)]:
        lo, hi = wilson_interval(successes, trials)
        assert lo <= successes / trials <= hi


def test_no_trials_gives_the_whole_range_rather_than_a_confident_zero() -> None:
    # A rule the policy never had the chance to break has demonstrated nothing. Reporting 0.0
    # with no band would read as evidence of virtue.
    assert wilson_interval(0, 0) == (0.0, 1.0)


def test_zero_successes_does_not_collapse_to_a_point() -> None:
    # Where the normal approximation gives [0, 0] and claims certainty, Wilson keeps a width
    # that shrinks with the sample.
    lo_small, hi_small = wilson_interval(0, 20)
    lo_big, hi_big = wilson_interval(0, 2000)
    assert lo_small == 0.0 and hi_small > 0.0
    assert hi_big < hi_small, "the interval must tighten as evidence accumulates"


def test_the_interval_stays_inside_the_unit_range() -> None:
    for successes, trials in [(0, 1), (1, 1), (0, 7), (7, 7), (1, 3)]:
        lo, hi = wilson_interval(successes, trials)
        assert 0.0 <= lo <= hi <= 1.0


def test_width_carries_the_sample_size() -> None:
    """The property that lets the band replace the separate denominator series."""
    narrow = wilson_interval(60, 600)
    wide = wilson_interval(6, 60)          # same rate, a tenth of the evidence
    assert (wide[1] - wide[0]) > 2 * (narrow[1] - narrow[0])


def test_a_larger_z_gives_a_wider_interval() -> None:
    lo95, hi95 = wilson_interval(25, 362, z=Z95)
    lo99, hi99 = wilson_interval(25, 362, z=2.576)
    assert lo99 < lo95 and hi99 > hi95


def test_blunder_metrics_report_rate_and_interval() -> None:
    counts = BlunderCounts()
    for _ in range(52):
        counts.note_opportunity("spaced_own_or_neutral")
    for _ in range(3):
        counts.note_blunder(Blunder("spaced_own_or_neutral", "US", 1, "x", 3, 2, "d"))

    m = counts.metrics()
    rule = "spaced_own_or_neutral"
    assert m[f"blunder_{rule}_rate"] == pytest.approx(3 / 52)
    assert m[f"blunder_{rule}_ci_low"] < m[f"blunder_{rule}_rate"] < m[f"blunder_{rule}_ci_high"]
    # The raw counts stay in the record even though they are no longer charted.
    assert m[f"blunder_{rule}_count"] == 3.0
    assert m[f"blunder_{rule}_chances"] == 52.0
    # Every rule reports the full set, including ones with no chances at all.
    for r in RULES:
        for part in ("rate", "ci_low", "ci_high", "count", "chances"):
            assert f"blunder_{r}_{part}" in m


def test_a_rule_with_no_chances_reports_the_widest_band() -> None:
    m = BlunderCounts().metrics()
    for r in RULES:
        assert m[f"blunder_{r}_ci_low"] == 0.0
        assert m[f"blunder_{r}_ci_high"] == 1.0
        assert m[f"blunder_{r}_chances"] == 0.0


def test_decisive_metrics_report_both_rates_with_intervals() -> None:
    stats = DecisiveStats(win_available=85, win_taken=66,
                          loss_avoidable=1849, loss_taken=55)
    m = stats.as_metrics()
    assert m["decisive_win_take_rate"] == pytest.approx(66 / 85)
    assert m["decisive_win_take_ci_low"] < 66 / 85 < m["decisive_win_take_ci_high"]

    # Avoiding is not taking: the successes are the losses NOT walked into.
    avoid_rate = 1.0 - 55 / 1849
    assert m["decisive_loss_avoid_rate"] == pytest.approx(avoid_rate)
    assert m["decisive_loss_avoid_ci_low"] < avoid_rate < m["decisive_loss_avoid_ci_high"]

    # Forced wins are ~20x rarer than avoidable losses, so their rate is far less certain --
    # which a bare pair of numbers would not show.
    win_width = m["decisive_win_take_ci_high"] - m["decisive_win_take_ci_low"]
    loss_width = m["decisive_loss_avoid_ci_high"] - m["decisive_loss_avoid_ci_low"]
    assert win_width > 5 * loss_width


def test_decisive_metrics_survive_a_probe_that_saw_no_decisive_moments() -> None:
    m = DecisiveStats().as_metrics()
    assert math.isnan(m["decisive_win_take_rate"])
    assert (m["decisive_win_take_ci_low"], m["decisive_win_take_ci_high"]) == (0.0, 1.0)


def test_the_banded_charts_are_registered_and_the_counts_are_not() -> None:
    from ai.training.generic_trainer import MULTILINE_CHARTS, TB_TAGS, _TB_SUPPRESSED

    for rule in RULES:
        chart = MULTILINE_CHARTS[f"strategy/blunder_{rule}"]
        assert set(chart) == {"rate", "ci_low", "ci_high"}
        # The numerator and denominator are kept in the JSONL but deliberately not charted.
        assert f"blunder_{rule}_count" in _TB_SUPPRESSED
        assert f"blunder_{rule}_chances" in _TB_SUPPRESSED
        assert f"blunder_{rule}_count" not in TB_TAGS

    for name in ("win_take", "loss_avoid"):
        assert set(MULTILINE_CHARTS[f"strategy/decisive_{name}"]) == {"rate", "ci_low", "ci_high"}
    assert "decisive_win_available" in _TB_SUPPRESSED
    assert "decisive_loss_avoidable" in _TB_SUPPRESSED
