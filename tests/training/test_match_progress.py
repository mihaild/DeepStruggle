"""Progress lines while a long matchup plays (owner, 2026-11-17: a searched match ran 2.5 hours with nothing in
its log). Off by default; with an interval, lines name the games finished and the first agent's score in them,
and they change nothing about the result."""
from __future__ import annotations

import pytest

from tools.lib.batch_tournament import BatchMatchRunner, MatchProgress, MatchupResult
from tools.lib.player_agent import load_agent


def test_progress_reports_games_score_and_live_turns(capsys: pytest.CaptureFixture[str]) -> None:
    p = MatchProgress("A vs B", 4, every=1e-9)
    p.finish(1.0)
    p.finish(0.5)
    p.tick([3, 5])
    out = capsys.readouterr().out
    assert "[progress] A vs B: 2/4 games finished, the first 75.0% of them; 2 live at turn 4.0 on average" in out


def test_progress_is_silent_by_default(capsys: pytest.CaptureFixture[str]) -> None:
    p = MatchProgress("A vs B", 4, every=0.0)
    p.finish(1.0)
    p.tick([3])
    assert capsys.readouterr().out == ""


@pytest.mark.parametrize("packed", [False, True])
def test_progress_lines_leave_the_result_unchanged(packed: bool, capsys: pytest.CaptureFixture[str]) -> None:
    # Deterministic agents: a random bot's draws carry over between the two plays and would differ for that alone.
    a, b = load_agent("heuristic", device="cpu"), load_agent("heuristic", device="cpu")

    def play(every: float) -> MatchupResult:
        if packed:
            return BatchMatchRunner.play_packed_matchups([(a, b)], games_per_side=3, device="cpu",
                                                        temperature=0.0, progress_every=every)[0]
        return BatchMatchRunner.play_parallel_matchup(a, b, games_per_side=3, device="cpu", temperature=0.0,
                                                      progress_every=every)

    quiet = play(0.0)
    assert "[progress]" not in capsys.readouterr().out
    loud = play(1e-9)
    assert "[progress]" in capsys.readouterr().out
    for k in ("a_wins", "b_wins", "draws", "a_wins_as_us", "a_wins_as_ussr"):
        assert loud.get(k) == quiet.get(k)
