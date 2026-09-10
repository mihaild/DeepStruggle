"""Per-winner game statistics, and the human reference lines drawn beside them.

The reference values are constants copied out of a corpus analysis, which is exactly the kind of
thing that rots silently: a renamed metric leaves an orphan key that draws no line, and a value
on the wrong scale draws a line in the wrong place without erroring. Both are checked here.
"""

import ai.itsc_reference as itsc
import pytest
from ai.game_length import FULL_GAME_PLIES
from ai.training.generic_trainer import (
    EPISODE_DEPENDENT_KEYS,
    TB_TAGS,
    _tb_tag,
    episode_dependent_in,
    summarize_completed_episodes,
)


def _ep(winner: str, turn: int, ply: int, reason: str, utility: float) -> dict:
    return {"winner": winner, "turn": turn, "ply": ply, "ending_reason": reason,
            "terminal_utility": utility, "start_turn": 1}


SCORED = [
    {"winner": "US", "turn": 11, "ply": 154, "ending_reason": "final_scoring",
     "terminal_utility": 1.0, "victory_points": 6, "start_turn": 1},
    {"winner": "USSR", "turn": 9, "ply": 128, "ending_reason": "20vp",
     "terminal_utility": -1.0, "victory_points": -12, "start_turn": 1},
]

EPISODES = [
    _ep("US", 11, 154, "final_scoring", 1.0),
    _ep("US", 8, 110, "20vp", 1.0),
    _ep("USSR", 6, 82, "defcon1_self", -1.0),
    _ep("USSR", 9, 128, "20vp", -1.0),
    _ep("USSR", 7, 95, "wargames", -1.0),
    _ep("DRAW", 11, 154, "final_scoring", 0.0),
]


def test_win_rates_cover_both_sides_and_are_over_decided_games() -> None:
    s = summarize_completed_episodes(EPISODES)
    # 5 decided (the draw is excluded), 2 US and 3 USSR.
    assert s["us_win_rate"] == pytest.approx(2 / 5)
    assert s["ussr_win_rate"] == pytest.approx(3 / 5)
    assert s["us_win_rate"] + s["ussr_win_rate"] == pytest.approx(1.0)
    # The draw rate is over *all* episodes, not the decided ones.
    assert s["draw_rate"] == pytest.approx(1 / 6)


def test_length_is_split_by_winning_side() -> None:
    s = summarize_completed_episodes(EPISODES)
    assert s["episodes_completed_won_us"] == 2
    assert s["episodes_completed_won_ussr"] == 3
    assert s["mean_ply_won_us"] == pytest.approx((154 + 110) / 2)
    assert s["mean_ply_won_ussr"] == pytest.approx((82 + 128 + 95) / 3)
    assert s["mean_turn_won_us"] == pytest.approx((11 + 8) / 2)


def test_ending_mix_is_split_by_winning_side() -> None:
    s = summarize_completed_episodes(EPISODES)
    assert s["ending_frac_20vp_won_us"] == pytest.approx(1 / 2)
    assert s["ending_frac_20vp_won_ussr"] == pytest.approx(1 / 3)
    assert s["ending_frac_wargames_won_ussr"] == pytest.approx(1 / 3)
    assert s["ending_frac_final_scoring_won_us"] == pytest.approx(1 / 2)


def test_the_per_winner_groups_do_not_report_a_degenerate_win_rate() -> None:
    # Who won defines the group, so a ussr_win_rate of 1.0 every iteration is not a measurement.
    s = summarize_completed_episodes(EPISODES)
    for key in ("us_win_rate", "ussr_win_rate", "draw_rate", "mean_terminal_utility"):
        assert f"{key}_won_us" not in s
        assert f"{key}_won_ussr" not in s


def test_defcon1_is_also_reported_combined() -> None:
    # ITS records DEFCON 1 without saying whose decision caused it, so only the sum can be
    # compared against a human number.
    s = summarize_completed_episodes(EPISODES)
    assert s["ending_frac_defcon1"] == pytest.approx(
        s["ending_frac_defcon1_self"] + s["ending_frac_defcon1_provoked"])
    assert s["ending_frac_defcon1"] == pytest.approx(1 / 6)


def test_a_group_with_no_episodes_is_held_back_rather_than_logged_as_zero() -> None:
    only_us = [e for e in EPISODES if e["winner"] == "US"]
    s = summarize_completed_episodes(only_us)
    held = episode_dependent_in(s)
    # The USSR won nothing this iteration; its mean turn is absent, not 0.
    assert "mean_turn_won_ussr" in held
    assert "mean_ply_won_ussr" in held


def test_every_reference_key_is_a_metric_the_trainer_emits() -> None:
    """An orphan reference key draws no line and says nothing about it."""
    emitted = set(summarize_completed_episodes(EPISODES))
    orphans = sorted(k for k in itsc.ITSC_REFERENCE if k not in emitted)
    assert not orphans, f"reference keys with no matching metric: {orphans}"


def test_every_reference_key_has_a_tensorboard_tag() -> None:
    """A reference without a tag lands in misc/ and never shares a chart with its metric."""
    from ai.training.generic_trainer import POPULATIONS

    def base_of(key: str) -> str:
        for suffix, _run in POPULATIONS:
            if suffix and key.endswith(suffix):
                return key[: -len(suffix)]
        return key

    untagged = sorted(k for k in itsc.ITSC_REFERENCE if base_of(k) not in TB_TAGS)
    assert not untagged, f"reference keys missing a TB tag: {untagged}"
    # Pooled series, per-winner series and human lines all share one chart, written from
    # sibling runs under the same tag -- that is what puts six lines on `endgame/turn`.
    assert _tb_tag("mean_ply") == "endgame/ply"
    assert _tb_tag(base_of("ending_frac_20vp_won_ussr")) == "endgame/ending_20vp"


def test_reference_values_are_on_the_right_scales() -> None:
    for key, value in itsc.ITSC_REFERENCE.items():
        if key.startswith("ending_frac_") or key.endswith("win_rate") or key == "draw_rate":
            assert 0.0 <= value <= 1.0, f"{key}={value} is not a fraction"
        elif "_turn" in key:
            # Engine scale: 11 is a game that played all ten turns out.
            assert 1.0 <= value <= 11.0, f"{key}={value} is not a turn"
        elif "_ply" in key:
            assert 1.0 <= value <= FULL_GAME_PLIES, f"{key}={value} is not a ply"
        else:
            raise AssertionError(f"unclassified reference key {key}")


def test_the_ending_mix_reference_sums_to_one() -> None:
    for suffix in ("", "_won_us", "_won_ussr"):
        total = sum(v for k, v in itsc.ITSC_REFERENCE.items()
                    if k.startswith("ending_frac_") and k.endswith(suffix)
                    and (suffix != "" or "_won_" not in k))
        assert total == pytest.approx(1.0, abs=2e-3), f"suffix {suffix!r} sums to {total}"


def test_the_split_defcon_reasons_deliberately_have_no_reference() -> None:
    for key in itsc.NO_REFERENCE_SPLIT:
        assert itsc.reference_for(f"ending_frac_{key}") is None
    assert itsc.reference_for("ending_frac_defcon1") is not None


def test_mean_ply_reference_sits_inside_its_stated_bounds() -> None:
    lo, hi = itsc.ITSC_MEAN_PLY_BOUNDS
    assert lo <= itsc.ITSC_REFERENCE["mean_ply"] <= hi


def test_episode_dependent_keys_cover_the_new_series() -> None:
    for key in ("us_win_rate", "mean_ply", "ending_frac_defcon1",
                "mean_victory_points", "mean_vp_margin"):
        assert key in EPISODE_DEPENDENT_KEYS


def test_final_score_is_reported_pooled_and_per_winner() -> None:
    s = summarize_completed_episodes(SCORED)
    # US-positive, so a +6 US win and a -12 USSR win average to -3.
    assert s["mean_victory_points"] == pytest.approx(-3.0)
    # The margin ignores who it favoured, which is what says "narrow or a blowout".
    assert s["mean_vp_margin"] == pytest.approx(9.0)
    assert s["mean_vp_margin_won_us"] == pytest.approx(6.0)
    assert s["mean_vp_margin_won_ussr"] == pytest.approx(12.0)


def test_final_score_has_no_human_reference() -> None:
    # ITS records the ending and the turn, never the final score.
    assert itsc.reference_for("mean_victory_points") is None
    assert itsc.reference_for("mean_vp_margin") is None
