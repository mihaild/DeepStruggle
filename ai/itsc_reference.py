"""Human reference values from the ITS Junta results database, for plotting beside training.

`/workspace/data/itsc-games` is a scrape of the public game-results table at
twilight-struggle.com — 47,928 digital (Playdek, Deluxe) games, of which **44,136** carry a
rules ending. Those are the games behind every number here. Excluded: 1,464 forfeits and
timeouts, which are abandonment rather than a result, and 2,328 rows with neither field.

Why this corpus and not ts-replayer. The ts-replayer corpus has moves and so is the only thing
that can train anything, but only 119 of its 274 logs reach a terminal state, and that filter is
not independent of how the game ended — a game that blows up at turn 6 leaves a log that stops
mid-turn and lands in the fragment pile, while one that goes the distance is recorded to the end.
It reports 67.2% of games going the distance against ITS's 29.7%, and DEFCON 1 at 1.7% against
11.7%, which is not sampling noise at these sizes. See `research/metrics.md` §1.5.1.

Two cautions on reading a line drawn from these:

* **Turn numbers are on the engine's scale already.** The site records a game that plays all ten
  turns out as ending on turn 11 — 12,787 of its 12,792 Final Scoring games sit there — which is
  exactly what `finish_end_turn` produces. No rescaling is applied or needed.
* **The ply values are estimates, the turn values are measurements.** The rows give the ending
  turn but not the action round, which pins a game's ply only to a 14- or 16-wide interval. The
  values below apply the mean within-turn offset per ending kind measured on ts-replayer (20 VP
  lands late in a turn, 13.7 of 16; Wargames early, 5.5; pooled 9.2). The mean is bounded to
  [112.1, 122.5] whatever the offsets are, so treat `mean_ply` as "about 120", not as 119.85.

These are a target to read a run against, not a target to optimise: a policy could match every
number here and play badly, and the arms already match the 20 VP share almost exactly while
losing four times as many games to DEFCON 1.
"""

from typing import Dict, Final

#: Games behind the numbers below, and the ending mix they were computed from.
ITSC_GAMES: Final[int] = 44_136
ITSC_DECIDED: Final[int] = 43_685
ITSC_SOURCE: Final[str] = "ITS Junta results database (twilight-struggle.com), 2005-12-31..2026-09-10"

#: Metric key (as emitted by `_episode_group_stats`) -> the human value for it.
#: `_won_us` / `_won_ussr` suffixes mirror the per-winner split the trainer logs.
ITSC_REFERENCE: Final[Dict[str, float]] = {
    "us_win_rate": 0.50104,
    "ussr_win_rate": 0.49896,
    "draw_rate": 0.01022,

    "mean_turn": 8.33,
    "median_turn": 9.0,
    "mean_ply": 119.85,
    "median_ply": 127.5,

    "ending_frac_20vp": 0.41513,
    "ending_frac_europe_control": 0.01550,
    "ending_frac_final_scoring": 0.28983,
    "ending_frac_wargames": 0.14884,
    "ending_frac_defcon1": 0.11712,
    "ending_frac_held_scoring": 0.01359,

    "mean_turn_won_us": 8.619,
    "median_turn_won_us": 9.0,
    "mean_ply_won_us": 124.05,
    "median_ply_won_us": 135.7,
    "ending_frac_20vp_won_us": 0.34288,
    "ending_frac_europe_control_won_us": 0.01933,
    "ending_frac_final_scoring_won_us": 0.32127,
    "ending_frac_wargames_won_us": 0.14474,
    "ending_frac_defcon1_won_us": 0.15561,
    "ending_frac_held_scoring_won_us": 0.01617,

    "mean_turn_won_ussr": 7.998,
    "median_turn_won_ussr": 8.0,
    "mean_ply_won_ussr": 115.10,
    "median_ply_won_ussr": 119.7,
    "ending_frac_20vp_won_ussr": 0.49622,
    "ending_frac_europe_control_won_ussr": 0.01197,
    "ending_frac_final_scoring_won_ussr": 0.25031,
    "ending_frac_wargames_won_ussr": 0.14933,
    "ending_frac_defcon1_won_ussr": 0.08088,
    "ending_frac_held_scoring_won_ussr": 0.01129,
}

#: The ply figures are imputed, not measured (see the module docstring). The mean is bounded to
#: this interval regardless of the imputation, which is the honest width of the claim.
ITSC_MEAN_PLY_BOUNDS: Final[tuple[float, float]] = (112.1, 122.5)

#: ITS records DEFCON 1 as one outcome and cannot say whose decision caused it, so there is no
#: reference for `defcon1_self` / `defcon1_provoked` separately -- only for their sum, which is
#: why the trainer emits a combined `ending_frac_defcon1`.
#:
#: Europe Control, by contrast, *is* recorded separately by ITS (684 games, 1.55% of those with
#: a rules ending) and used to be folded into `20vp` here, because the engine could not tell the
#: two apart: controlling Europe when Europe is scored ends the game at +/-20 VP like any other
#: 20 VP win. `effect_bits::EUROPE_CONTROL_WIN` now records which it was, so the two are split.
NO_REFERENCE_SPLIT: Final[tuple[str, ...]] = ("defcon1_self", "defcon1_provoked")


def reference_for(metric_key: str) -> float | None:
    """The human value for a metric key, or None where the corpus cannot speak to it."""
    return ITSC_REFERENCE.get(metric_key)
