"""The leaderboard's players on Playdek's rating scale, from the games people played against them.

The leaderboard's Elo comes only from games between models. The stats database (web/stats_worker/)
holds games against people, each with the human's Playdek rating, and those games place the players
on Playdek's scale:

    P(human scores) = 1 / (1 + 10^((position + b + n + c - rating) / 400))

`position` is the player's leaderboard Elo on the newest epoch's scale (an older epoch's rating
reaches it through the newer epoch's anchor, which is rated on both), so the models' strength
against each other is where every player starts. `b` is one offset shared by every player. `n`,
drawn from N(0, network_sd^2), is shared by all the players on one network, and `c`, drawn from
N(0, config_sd^2), is the player's own: people's games against one search of a network say most
about that network's other searches too, and less about how the searches differ, which the
leaderboard has measured over thousands of games. So a gap moves as far as the number of games
against people justifies, and a player with none takes its network's `n` (0 if nobody has played
that network) with the priors in its uncertainty.

se and interval come from a bootstrap over the people, whose games are not independent of each
other. Humans play at their stated rating. There is no seat term: how a model plays each seat is
its own property, part of its strength (owner, 2026-10-31). Which games count is the
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
from typing import Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, TypedDict

import numpy as np

from tools.lib.leaderboard import Board, LeaderboardError, fit_epoch
from tools.lib.player_spec import spec_from_dict, workbench_search

_C = math.log(10.0) / 400.0
#: The sds of a network's and of a configuration's deviation from the leaderboard, in rating points.
#: Tight on purpose: a gap the leaderboard measured over thousands of games should yield to many games
#: against people, not to one. At these, removing any one game moves a gap by about 3 points; at 100
#: and 50, one loss took a player from 50 points above another to 26 below (owner, 2026-10-31).
DEFAULT_NETWORK_SD = 20.0
DEFAULT_CONFIG_SD = 10.0
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
    network_sd: float
    config_sd: float
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
    #: Only what has games in the fit.
    networks: Dict[str, float]
    configs: Dict[str, float]


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


def fit(games: Sequence[HumanGame], position: Mapping[str, float], network_of: Mapping[str, str],
        network_sd: float = DEFAULT_NETWORK_SD, config_sd: float = DEFAULT_CONFIG_SD) -> Fit:
    """The posterior mode of the offset and the deviations of what was played (Newton). An sd of 0
    fixes that part at 0; both 0 is the plain shift."""
    players = sorted({g.player for g in games}) if config_sd > 0.0 else []
    networks = sorted({network_of[g.player] for g in games}) if network_sd > 0.0 else []
    net_at = {n: 1 + k for k, n in enumerate(networks)}
    config_at = {p: 1 + len(networks) + k for k, p in enumerate(players)}
    x = np.zeros((len(games), 1 + len(networks) + len(players)))
    x[:, 0] = 1.0
    for row, g in enumerate(games):
        if networks:
            x[row, net_at[network_of[g.player]]] = 1.0
        if players:
            x[row, config_at[g.player]] = 1.0
    margin = np.array([g.rating - position[g.player] for g in games])
    score = np.array([g.score for g in games])
    mu = float(margin.mean())
    prior = np.array([1.0 / OFFSET_PRIOR_SD ** 2] + [1.0 / max(network_sd, 1e-9) ** 2] * len(networks)
                     + [1.0 / max(config_sd, 1e-9) ** 2] * len(players))
    centre = np.zeros(x.shape[1])
    centre[0] = mu
    theta = centre.copy()
    for _ in range(100):
        p = 1.0 / (1.0 + np.exp(-_C * (margin - x @ theta)))
        grad = -_C * (x.T @ (score - p)) - prior * (theta - centre)
        info = _C ** 2 * (x.T * (p * (1.0 - p))) @ x + np.diag(prior)
        step = np.linalg.solve(info, grad)
        theta = theta + step
        if float(np.abs(step).max()) < 1e-7:
            break
    return Fit(float(theta[0]), {n: float(theta[k]) for n, k in net_at.items()},
               {p: float(theta[k]) for p, k in config_at.items()})


def _percentile(ordered: Sequence[float], q: float) -> float:
    at = q * (len(ordered) - 1)
    lo = int(at)
    hi = min(lo + 1, len(ordered) - 1)
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (at - lo)


def _networks(board: Board, players: Iterable[str]) -> Dict[str, str]:
    """Each player's network; a player with none (a bot) stands alone."""
    return {p: board.players[p]["network"] or p for p in players}


def estimate(played: Sequence[PlayedGame], left_out: Mapping[str, int], as_of: str, board: Board,
             is_before_keyed_default: Callable[[str], Optional[bool]], network_sd: float = DEFAULT_NETWORK_SD,
             config_sd: float = DEFAULT_CONFIG_SD, bootstrap: int = BOOTSTRAP, seed: int = 0) -> Estimate:
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
    network_of = _networks(board, position)
    point = fit(placed, position, network_of, network_sd, config_sd)

    by_login: Dict[str, List[HumanGame]] = {}
    for g in placed:
        by_login.setdefault(g.login, []).append(g)
    logins = sorted(by_login)
    rng = random.Random(seed)
    draws: Dict[str, List[float]] = {p: [] for p in position}
    for _ in range(bootstrap):
        sample = [g for _ in logins for g in by_login[logins[rng.randrange(len(logins))]]]
        one = fit(sample, position, network_of, network_sd, config_sd)
        # What has no games in this draw has nothing to say how it deviates: the prior does.
        shared = {n: one.networks[n] if n in one.networks else rng.gauss(0.0, network_sd)
                  for n in set(network_of.values())}
        for p, e in position.items():
            own = one.configs[p] if p in one.configs else rng.gauss(0.0, config_sd)
            draws[p].append(e + one.offset + shared[network_of[p]] + own)

    players: Dict[str, PlayerEstimate] = {}
    for p, e in position.items():
        mine = [g for g in placed if g.player == p]
        ordered = sorted(draws[p])
        mean = sum(ordered) / len(ordered)
        players[p] = {
            "rating": round(e + point.offset + point.networks.get(network_of[p], 0.0) + point.configs.get(p, 0.0), 1),
            "se": round(math.sqrt(sum((x - mean) ** 2 for x in ordered) / max(1, len(ordered) - 1)), 1),
            "lo": round(_percentile(ordered, 0.025), 1), "hi": round(_percentile(ordered, 0.975), 1),
            "games": len(mine), "humans": len({g.login for g in mine}),
            # The record is the model's; a game's score is the human's.
            "w": sum(g.score == 0.0 for g in mine), "d": sum(g.score == 0.5 for g in mine),
            "l": sum(g.score == 1.0 for g in mine),
        }
    return {"as_of": as_of[:10], "games": len(placed), "humans": len(logins),
            "excluded": excluded, "offset": round(point.offset, 1), "network_sd": network_sd, "config_sd": config_sd,
            "reference_epoch": reference, "players": players}
