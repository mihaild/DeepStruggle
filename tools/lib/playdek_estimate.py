"""The leaderboard's players on Playdek's rating scale, from the games people played against them.

The leaderboard's Elo comes only from games between models. The stats database (web/stats_worker/)
holds games against people, each with the human's Playdek rating, and those games place the players
on Playdek's scale:

    P(human scores) = 1 / (1 + 10^((position + b + d - rating) / 400))

`position` is the player's leaderboard Elo on the newest epoch's scale (an older epoch's rating
reaches it through the newer epoch's anchor, which is rated on both), `b` is one offset shared by
every player, and `d` is the player's own deviation, drawn from N(0, deviation_sd^2). So a gap
between two players is not simply carried over from the leaderboard: people's games against a
configuration move it, as far as their number justifies. A player nobody has played has d = 0, the
leaderboard's gap, and carries deviation_sd in its uncertainty.

se and interval come from a bootstrap over the people, whose games are not independent of each
other. Humans play at their stated rating, with no seat advantage. Which games count is the
caller's (tools/playdek_estimate.py), under the same check as `tools/game_stats.py rating`; takebacks
are allowed (owner, 2026-10-31).

The output holds aggregates only. A login is read to group a person's games and never leaves
`estimate`: logins are private (web/stats_worker/README.md).
"""

from __future__ import annotations

import math
import random
from collections import Counter
from dataclasses import dataclass
from typing import Callable, Dict, List, Mapping, Optional, Sequence, Tuple, TypedDict

from tools.lib.leaderboard import Board, LeaderboardError, fit_epoch
from tools.lib.player_spec import spec_from_dict, workbench_search

_C = math.log(10.0) / 400.0
#: The sd of a player's deviation from its leaderboard gap, in rating points.
DEFAULT_DEVIATION_SD = 100.0
#: The prior on the offset, around the mean of (human rating - position).
OFFSET_PRIOR_SD = 400.0
BOOTSTRAP = 1000

#: The first build whose bare `rollout` means the keyed root; before it, the 32-world root it replaced.
#: A build since 620d80f spells the search out, so this is history and does not move.
KEYED_DEFAULT_COMMIT = "d954317"
OLD_DEFAULT_SEARCH = "rollout:4:32:4:z2:earlier:own"
KEYED_DEFAULT_SEARCH = "rollout:4:32:4:z2:earlier:own:keyed:aligned:box4"

NOT_ON_BOARD = "not a player on the leaderboard"


class PlayerEstimate(TypedDict):
    rating: float
    se: float
    lo: float
    hi: float
    #: Games people played against this player, and the model's record in them.
    games: int
    humans: int
    w: int
    d: int
    l: int


class Estimate(TypedDict):
    #: The day the newest game in the database arrived.
    as_of: str
    games: int
    humans: int
    #: Why a game does not count, with how many.
    excluded: Dict[str, int]
    offset: float
    deviation_sd: float
    #: The epoch whose scale `position` is on.
    reference_epoch: str
    players: Dict[str, PlayerEstimate]


@dataclass(frozen=True)
class PlayedGame:
    """A game that passed the check `rating` applies (tools/lib/game_stats.rating_check), as the
    fit needs it. The caller reads the submission; this module needs no engine."""
    login: str
    rating: float
    #: The human's score: 1 win, 0.5 draw, 0 loss.
    score: float
    model_label: str
    search: str
    commit: str


@dataclass(frozen=True)
class HumanGame:
    login: str
    rating: float
    #: The human's score: 1 win, 0.5 draw, 0 loss.
    score: float
    player: str


@dataclass(frozen=True)
class Fit:
    offset: float
    #: Only the players with games in the fit.
    deviations: Dict[str, float]


def positions(board: Board) -> Tuple[str, Dict[str, float]]:
    """(the newest epoch, every player it can place on that epoch's scale).

    A player keeps its newest epoch's rating. Walking back, an older epoch joins through the newer
    one's anchor; an epoch that does not rate that anchor cannot be joined, nor can any before it."""
    names = list(board.epochs)
    out: Dict[str, float] = {}
    shift = 0.0
    newer_anchor: Optional[Tuple[str, float]] = None
    for name in reversed(names):
        epoch_fit = fit_epoch(board, name)
        elo = {r.player: r.elo for r in epoch_fit.ratings}
        if newer_anchor is not None:
            player, on_newer = newer_anchor
            if player not in elo:
                break
            shift += on_newer - elo[player]
        for player, e in elo.items():
            out.setdefault(player, e + shift)
        newer_anchor = (epoch_fit.anchor, epoch_fit.anchor_elo)
    return names[-1], out


def player_index(board: Board) -> Dict[Tuple[str, str], str]:
    """(a network's name, its workbench `search=` value) -> the player it plays."""
    index: Dict[Tuple[str, str], str] = {}
    for pid, player in board.players.items():
        search = workbench_search(spec_from_dict(player["inference"]))
        network = player["network"]
        if search is None or network is None:
            continue
        for name in [network, *board.networks[network]["old_names"]]:
            other = index.setdefault((name, search), pid)
            if other != pid:
                raise LeaderboardError(f"the workbench plays {other!r} and {pid!r} the same way: {name} {search}")
    return index


def rated_games(played: Sequence[PlayedGame], board: Board,
                is_before_keyed_default: Callable[[str], Optional[bool]]) -> Tuple[List[HumanGame], Dict[str, int]]:
    """Each game with the player it was against, and how many have none and why.
    `is_before_keyed_default` says whether a build precedes KEYED_DEFAULT_COMMIT, None for a build it
    does not know."""
    index = player_index(board)
    games: List[HumanGame] = []
    excluded: Counter[str] = Counter()
    for g in played:
        search = g.search
        if search == "rollout":
            before = is_before_keyed_default(g.commit)
            if before is None:
                excluded["unknown build"] += 1
                continue
            search = OLD_DEFAULT_SEARCH if before else KEYED_DEFAULT_SEARCH
        player = index.get((g.model_label, search))
        if player is None:
            excluded[NOT_ON_BOARD] += 1
            continue
        games.append(HumanGame(g.login, g.rating, g.score, player))
    return games, dict(excluded)


def fit(games: Sequence[HumanGame], position: Mapping[str, float], deviation_sd: float) -> Fit:
    """The posterior mode of the offset and the played players' deviations (Newton: an arrowhead
    system, the offset against one diagonal entry per player). deviation_sd 0 is the plain shift."""
    players = sorted({g.player for g in games})
    column = {p: i for i, p in enumerate(players)}
    rows = [(g.rating - position[g.player], g.score, column[g.player]) for g in games]
    mu = sum(r[0] for r in rows) / len(rows)
    n = len(players) if deviation_sd > 0.0 else 0
    prior = 1.0 / deviation_sd ** 2 if n else 0.0
    b = mu
    d = [0.0] * len(players)
    for _ in range(100):
        grad_b = -(b - mu) / OFFSET_PRIOR_SD ** 2
        info_b = 1.0 / OFFSET_PRIOR_SD ** 2
        grad_d = [-d[k] * prior for k in range(n)]
        info_d = [prior] * n
        for margin, score, k in rows:
            p = 1.0 / (1.0 + math.exp(-_C * (margin - b - d[k])))
            g = -_C * (score - p)
            w = _C ** 2 * p * (1.0 - p)
            grad_b += g
            info_b += w
            if n:
                grad_d[k] += g
                info_d[k] += w
        # The cross term of the offset and a deviation is that deviation's information from games.
        cross = [info_d[k] - prior for k in range(n)]
        schur = info_b - sum(cross[k] ** 2 / info_d[k] for k in range(n))
        step_b = (grad_b - sum(cross[k] * grad_d[k] / info_d[k] for k in range(n))) / schur
        steps = [(grad_d[k] - cross[k] * step_b) / info_d[k] for k in range(n)]
        b += step_b
        for k in range(n):
            d[k] += steps[k]
        if max([abs(step_b), *map(abs, steps)]) < 1e-7:
            break
    return Fit(b, dict(zip(players, d)))


def _percentile(ordered: Sequence[float], q: float) -> float:
    at = q * (len(ordered) - 1)
    lo = int(at)
    hi = min(lo + 1, len(ordered) - 1)
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (at - lo)


def estimate(played: Sequence[PlayedGame], left_out: Mapping[str, int], as_of: str, board: Board,
             is_before_keyed_default: Callable[[str], Optional[bool]], deviation_sd: float = DEFAULT_DEVIATION_SD,
             bootstrap: int = BOOTSTRAP, seed: int = 0) -> Estimate:
    """Every player the newest epoch's scale reaches, on Playdek's scale. `left_out` counts the games
    that did not pass the check, by reason. Raises when no game counts."""
    reference, position = positions(board)
    games, excluded = rated_games(played, board, is_before_keyed_default)
    excluded = {**left_out, **excluded}
    placed = [g for g in games if g.player in position]
    if len(placed) != len(games):
        excluded[NOT_ON_BOARD] = excluded.get(NOT_ON_BOARD, 0) + len(games) - len(placed)
    if not placed:
        raise LeaderboardError("no game against a person counts towards the Playdek estimate")
    point = fit(placed, position, deviation_sd)

    by_login: Dict[str, List[HumanGame]] = {}
    for g in placed:
        by_login.setdefault(g.login, []).append(g)
    logins = sorted(by_login)
    rng = random.Random(seed)
    draws: Dict[str, List[float]] = {p: [] for p in position}
    for _ in range(bootstrap):
        sample = [g for _ in logins for g in by_login[logins[rng.randrange(len(logins))]]]
        one = fit(sample, position, deviation_sd)
        for p, e in position.items():
            # A player absent from this draw has no games to say how it deviates: the prior does.
            d = one.deviations[p] if p in one.deviations else rng.gauss(0.0, deviation_sd)
            draws[p].append(e + one.offset + d)

    players: Dict[str, PlayerEstimate] = {}
    for p, e in position.items():
        mine = [g for g in placed if g.player == p]
        ordered = sorted(draws[p])
        mean = sum(ordered) / len(ordered)
        players[p] = {
            "rating": round(e + point.offset + point.deviations.get(p, 0.0), 1),
            "se": round(math.sqrt(sum((x - mean) ** 2 for x in ordered) / max(1, len(ordered) - 1)), 1),
            "lo": round(_percentile(ordered, 0.025), 1), "hi": round(_percentile(ordered, 0.975), 1),
            "games": len(mine), "humans": len({g.login for g in mine}),
            # The record is the model's; a game's score is the human's.
            "w": sum(g.score == 0.0 for g in mine), "d": sum(g.score == 0.5 for g in mine),
            "l": sum(g.score == 1.0 for g in mine),
        }
    return {"as_of": as_of[:10], "games": len(placed), "humans": len(logins),
            "excluded": excluded, "offset": round(point.offset, 1), "deviation_sd": deviation_sd,
            "reference_epoch": reference, "players": players}
