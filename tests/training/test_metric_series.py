"""What the training run publishes as metrics.

Both properties here were violated in a 320M run and neither showed up as an error: ten of its
series were exact copies of another series, and five were constant zero for the whole run. A
duplicate and a flat zero are indistinguishable from a real measurement until someone reads them
carefully, which is why they are asserted rather than eyeballed.
"""

from typing import Any, Dict, List

from ai.training.generic_trainer import (
    EPISODE_DEPENDENT_KEYS,
    _tb_tag,
    episode_dependent_in,
    summarize_completed_episodes,
)


def _ep(turn: int, winner: str, start_turn: int = 1, reason: str = "20vp") -> Dict[str, Any]:
    return {
        "turn": turn,
        "winner": winner,
        "terminal_utility": 1.0 if winner == "US" else (-1.0 if winner == "USSR" else 0.0),
        "ending_reason": reason,
        "start_turn": start_turn,
    }


class TestPerStartTurnSeries:
    def test_no_suffixed_series_when_every_game_starts_at_turn_one(self) -> None:
        """The ordinary case: --start-pool-frac 0, so a _start1 series would duplicate the pooled
        one exactly. A 320M run logged ten such copies across 1,220 iterations."""
        stats = summarize_completed_episodes([_ep(7, "US"), _ep(5, "USSR"), _ep(9, "US")])
        suffixed = [k for k in stats if "_start" in k]
        assert suffixed == [], f"emitted duplicate per-start series with no pool in use: {suffixed}"

    def test_suffixed_series_appear_when_starts_actually_differ(self) -> None:
        """The case they exist for: a pooled mean turn then describes neither population."""
        eps = [_ep(7, "US"), _ep(9, "USSR", start_turn=6), _ep(10, "US", start_turn=6)]
        stats = summarize_completed_episodes(eps)
        assert stats["mean_turn_start1"] == 7.0
        assert stats["mean_turn_start6"] == 9.5
        assert stats["mean_turn"] != stats["mean_turn_start1"], "pooled must not equal a group"
        assert stats["episodes_completed_start6"] == 2.0

    def test_suffixed_keys_are_held_back_with_the_group_they_belong_to(self) -> None:
        eps = [_ep(7, "US"), _ep(9, "USSR", start_turn=6)]
        held = episode_dependent_in(summarize_completed_episodes(eps))
        assert "mean_turn_start6" in held
        assert "mean_turn" in held
        assert "episodes_completed" not in held, "a count of zero episodes is a real zero"


class TestSideOutcomeMetrics:
    def test_ussr_win_rate_counts_only_decided_games(self) -> None:
        stats = summarize_completed_episodes(
            [_ep(7, "USSR"), _ep(7, "USSR"), _ep(7, "US"), _ep(10, "DRAW")])
        assert stats["ussr_win_rate"] == 2.0 / 3.0, "a draw is not half a USSR win"
        assert stats["draw_rate"] == 0.25
        assert stats["mean_terminal_utility"] == (-1.0 - 1.0 + 1.0 + 0.0) / 4.0

    def test_no_episodes_does_not_publish_a_zero_win_rate(self) -> None:
        """Zero would read as "the USSR won nothing this iteration" rather than "nothing ended"."""
        assert "ussr_win_rate" in EPISODE_DEPENDENT_KEYS
        assert "ussr_win_rate" in episode_dependent_in(summarize_completed_episodes([]))


class TestTagMapping:
    def test_evaluation_win_rates_get_their_own_namespace(self) -> None:
        assert _tb_tag("eval_win_rate_HeuristicBot") == "eval/win_rate_vs_HeuristicBot"
        assert _tb_tag("eval_win_rate_RandomBot_as_ussr") == "eval/win_rate_vs_RandomBot_as_ussr"

    def test_unknown_keys_land_in_misc_rather_than_vanishing(self) -> None:
        assert _tb_tag("something_new") == "misc/something_new"

    def test_known_keys_still_map(self) -> None:
        # Four groups: progress / internal / endgame / strategy. mean_turn and mean_ply share
        # one chart each with their per-winner and human lines, hence the bare chart names.
        assert _tb_tag("mean_turn") == "endgame/turn"
        assert _tb_tag("mean_ply") == "endgame/ply"
        assert _tb_tag("ussr_win_rate") == "endgame/ussr_win_rate"
        assert _tb_tag("loss") == "internal/loss"
        # The four battleground series share one chart, so they are suppressed as individual
        # scalars and reach TensorBoard only as lines of `strategy/battlegrounds`.
        from ai.training.generic_trainer import MULTILINE_CHARTS, _TB_SUPPRESSED

        assert "diag/uncontrolled_battlegrounds_turn8" in _TB_SUPPRESSED
        assert set(MULTILINE_CHARTS["strategy/battlegrounds"].values()) == {
            "diag/empty_battlegrounds_turn5", "diag/empty_battlegrounds_turn8",
            "diag/uncontrolled_battlegrounds_turn5", "diag/uncontrolled_battlegrounds_turn8"}
        assert _tb_tag("diag/salvageable_frac_turn6") == "strategy/salvageable_frac_turn6"
