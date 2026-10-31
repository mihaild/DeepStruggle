"""The leaderboard's players on Playdek's scale: which player a game was against, the fit with its
deviations by network and by player, and what may be published. Synthetic games on a small synthetic board; the
database and its logins are private, so no test reads them."""
from __future__ import annotations

import json
import pathlib
from typing import Any, Dict, List, Optional

import pytest
import yaml

from tools import playdek_estimate as cli
from tools.lib import leaderboard as lb
from tools.lib.game_stats import RatedGame, RatingCheck, SubmissionDict, _map_rating
from tools.lib.playdek_estimate import (KEYED_DEFAULT_SEARCH, OLD_DEFAULT_SEARCH, HumanGame, PlayedGame, estimate, fit,
                                        player_index, positions, rated_games)

REPO = pathlib.Path(__file__).resolve().parents[2]
NET = "E7-A8-R1-S44@6400M+(R34,R34-S45,R34-S46)@7120..7200M"
POLICY = NET
KEYED = (f"{NET}~rollout(k=4,worlds=32,horizon=4,rule=z2,stop=earlier,leaf=own,"
         "chance=keyed,aligned=true,box4=true)")
OLD = f"{NET}~rollout(k=4,worlds=32,horizon=4,rule=z2,stop=earlier,leaf=own)"


@pytest.fixture(scope="module")
def board() -> lb.Board:
    return lb.load()


def played(login: str = "someone", rating: float = 1600.0, score: float = 1.0, search: str = KEYED_DEFAULT_SEARCH,
           label: str = NET, commit: str = "c") -> PlayedGame:
    return PlayedGame(login, rating, score, label, search, commit)


def test_a_game_names_the_player_it_was_against(board: lb.Board) -> None:
    index = player_index(board)
    assert index[(NET, KEYED_DEFAULT_SEARCH)] == KEYED
    assert index[(NET, "policy")] == POLICY


def test_a_bare_rollout_means_what_it_meant_on_its_build(board: lb.Board) -> None:
    games = [played(search="rollout", commit="old"), played(search="rollout", commit="new")]
    known: Dict[str, Optional[bool]] = {"old": True, "new": False}
    got, excluded = rated_games(games, board, known.__getitem__)
    assert [g.player for g in got] == [OLD, KEYED] and not excluded
    assert player_index(board)[(NET, OLD_DEFAULT_SEARCH)] == OLD


def test_a_game_with_no_player_is_counted_and_left_out(board: lb.Board) -> None:
    games = [played(search="rollout", commit="stranger"), played(label="a dropped file")]
    got, excluded = rated_games(games, board, lambda _: None)
    assert not got
    assert excluded == {"unknown build": 1, "not a player on the leaderboard": 1}


def test_an_older_epoch_reaches_the_newest_scale_through_its_anchor(board: lb.Board) -> None:
    reference, position = positions(board)
    names = list(board.epochs)
    assert reference == names[-1]
    newest = lb.fit_epoch(board, names[-1])
    older = lb.fit_epoch(board, names[-2])
    on_older = {r.player: r.elo for r in older.ratings}
    shift = newest.anchor_elo - on_older[newest.anchor]
    only_older = next(p for p in on_older if p not in {r.player for r in newest.ratings})
    assert position[only_older] == pytest.approx(on_older[only_older] + shift)
    assert position[newest.anchor] == pytest.approx(newest.anchor_elo)


def test_no_deviations_is_the_plain_shift() -> None:
    """Both sds 0 fit one offset, what fit_rating's one rating is for a single player."""
    games = [HumanGame("a", 1500.0, 1.0, "p"), HumanGame("b", 1700.0, 0.0, "p"), HumanGame("c", 1600.0, 0.5, "p")]
    got = fit(games, {"p": 1400.0}, {"p": "n"}, 0.0, 0.0)
    want, _ = _map_rating([RatedGame(g.login, g.rating, "US", g.score) for g in games], 0.0, 1600.0, 400.0)
    assert got.offset + 1400.0 == pytest.approx(want, abs=1e-3)
    assert not got.networks and not got.configs


def _beaten(player: str, games: int, rating: float = 1700.0, wins_of_ten: int = 9) -> List[HumanGame]:
    return [HumanGame(f"{player}{k}", rating, 1.0 if k % 10 < wins_of_ten else 0.0, player) for k in range(games)]


def test_people_can_move_a_gap_the_leaderboard_gave() -> None:
    """Humans beat `weak` more than its gap says: the gap widens as far as the prior allows, which the
    shift alone could not do at all."""
    position = {"strong": 1700.0, "weak": 1600.0}
    networks = {"strong": "A", "weak": "B"}
    games = _beaten("strong", 40, wins_of_ten=5) + _beaten("weak", 40)
    shifted = fit(games, position, networks, 0.0, 0.0)
    moved = fit(games, position, networks)
    assert moved.networks["B"] + moved.configs["weak"] < moved.networks["A"] + moved.configs["strong"] - 15.0
    assert not shifted.networks and not shifted.configs
    tight = fit(games, position, networks, 10.0, 5.0)
    assert abs(tight.networks["B"]) < abs(moved.networks["B"]), "a tighter prior moves it less"


def test_one_game_cannot_override_a_measured_gap() -> None:
    """One loss to someone rated well below it barely moves a player, and a 50-point gap holds."""
    position = {"measured": 1600.0, "newcomer": 1650.0}
    networks = {"measured": "A", "newcomer": "B"}
    games = _beaten("measured", 60, rating=1650.0, wins_of_ten=5)
    rating = lambda f, p: position[p] + f.offset + f.networks.get(networks[p], 0.0) + f.configs.get(p, 0.0)
    without = fit(games, position, networks)
    with_loss = fit(games + [HumanGame("one", 1300.0, 1.0, "newcomer")], position, networks)
    gap = rating(with_loss, "newcomer") - rating(with_loss, "measured")
    assert gap == pytest.approx(rating(without, "newcomer") - rating(without, "measured"), abs=6.0)
    assert gap > 40.0


def test_many_games_can_still_move_a_gap() -> None:
    position = {"measured": 1600.0, "newcomer": 1650.0}
    networks = {"measured": "A", "newcomer": "B"}
    games = _beaten("measured", 200, rating=1650.0, wins_of_ten=5) + _beaten("newcomer", 200, rating=1650.0, wins_of_ten=9)
    moved = fit(games, position, networks)
    assert moved.networks["B"] + moved.configs["newcomer"] < moved.networks["A"] + moved.configs["measured"] - 20.0


def test_games_against_one_search_move_its_network_more_than_its_own_part() -> None:
    position = {"a32": 1600.0, "other": 1600.0}
    networks = {"a32": "A", "other": "B"}
    moved = fit(_beaten("a32", 40) + _beaten("other", 40, wins_of_ten=5), position, networks)
    assert abs(moved.networks["A"]) > abs(moved.configs["a32"]) > 0.0


def test_a_players_strength_among_its_network_survives_a_few_games(board: lb.Board) -> None:
    """The stronger search of a network stays above the weaker when people played only the weaker,
    the order the models' own games measured; the unshared fit put it below after two games."""
    _, position = positions(board)
    networks = {p: board.players[p]["network"] or p for p in position}
    strong = next(p for p in position if "race=16" in p and "@12720..12800M" in p)
    weak = next(p for p in position if networks[p] == networks[strong] and "worlds=32" in p)
    assert position[strong] > position[weak]
    games = [played(login=f"h{k}", rating=1600.0, score=float(k % 2)) for k in range(30)]
    two = [played(login="one", rating=1787.0, score=0.0, label=networks[weak]),
           played(login="one", rating=1787.0, score=0.5, label=networks[weak])]
    est = estimate(games + two, {}, "2026-10-31T00:00:00Z", board, lambda _: False, bootstrap=20)
    assert est["players"][strong]["rating"] > est["players"][weak]["rating"]
    assert est["players"][strong]["games"] == 0 and est["players"][weak]["games"] == 2
    # The network's games reach the player nobody faced.
    assert est["players"][strong]["rating"] > position[strong] + est["offset"]


def test_a_player_of_an_unplayed_network_sits_at_its_gap_with_a_wider_interval(board: lb.Board) -> None:
    _, position = positions(board)
    networks = {p: board.players[p]["network"] or p for p in position}
    unplayed = next(p for p in position if networks[p] != NET)
    games = [played(login=f"h{k}", rating=1600.0 + 10 * k, score=float(k % 2)) for k in range(30)]
    est = estimate(games, {}, "2026-10-31T00:00:00Z", board, lambda _: False, bootstrap=100)
    assert est["players"][unplayed]["rating"] == pytest.approx(position[unplayed] + est["offset"], abs=0.11)
    assert est["players"][unplayed]["games"] == 0
    assert est["players"][unplayed]["se"] > est["players"][KEYED]["se"]
    assert est["players"][KEYED]["games"] == 30 and est["players"][KEYED]["humans"] == 30


def test_the_estimate_is_reproducible_and_holds_no_login(board: lb.Board) -> None:
    secret = "a.private.login@example.com"
    games = [played(login=secret, score=1.0), played(login="other", score=0.0), played(login="third", score=0.5)]
    one = estimate(games, {"not clean": 2}, "2026-10-31T01:02:03Z", board, lambda _: False, bootstrap=20)
    two = estimate(games, {"not clean": 2}, "2026-10-31T01:02:03Z", board, lambda _: False, bootstrap=20)
    assert one == two
    text = json.dumps(one) + cli.summary(one)
    assert secret not in text and "other" not in one["players"] and "third" not in text
    assert one["excluded"] == {"not clean": 2}
    assert one["as_of"] == "2026-10-31" and one["games"] == 3 and one["humans"] == 3
    keyed = one["players"][KEYED]
    assert (keyed["w"], keyed["d"], keyed["l"]) == (1, 1, 1), "the record is the model's"


def test_no_counted_game_refuses_rather_than_publishing_nothing(board: lb.Board) -> None:
    with pytest.raises(lb.LeaderboardError):
        estimate([], {"not clean": 1}, "2026-10-31", board, lambda _: False)


def test_a_dump_reads_with_or_without_wrangler_banner() -> None:
    body = json.dumps({"id": "x"})
    rows = json.dumps([{"results": [{"received_at": "t", "body": body}, {"received_at": "later", "body": body}]}])
    want = [("t", {"id": "x"})]
    assert cli.rows_from_text(rows) == want
    assert cli.rows_from_text("\n wrangler banner\n Executed 1 command\n" + rows) == want


def test_only_a_game_already_here_may_skip_replaying(monkeypatch: pytest.MonkeyPatch) -> None:
    """An old engine's game that does not replay cannot be told from a forgery. The Worker stamps
    received_at, so what arrived before the cutoff is known, and nothing new can claim to have."""
    seen: List[bool] = []

    def check(sub: SubmissionDict, fingerprint: str, *, include_unverified: bool = False) -> RatingCheck:
        seen.append(include_unverified)
        return RatingCheck(None, "not clean")

    monkeypatch.setattr(cli, "rating_check", check)
    sub: Any = {}
    played, left_out = cli.counted_games([("2026-10-30T23:59:59Z", sub), (cli.UNVERIFIED_BEFORE + "T00:00:00Z", sub)], "fp")
    assert seen == [True, False]
    assert not played and left_out == {"not clean": 2}


def test_the_query_never_asks_for_the_address_hash() -> None:
    assert "ip_hash" not in cli.QUERY and "*" not in cli.QUERY


def test_only_the_fit_step_of_the_pages_workflow_has_the_token() -> None:
    """The build runs every npm package's scripts, and its log and artifact are public."""
    text = (REPO / ".github" / "workflows" / "pages.yml").read_text()
    flow = yaml.safe_load(text)
    holders = [(job, step.get("name")) for job, spec in flow["jobs"].items() for step in spec["steps"]
               if "secrets." in json.dumps(step)]
    assert holders == [("playdek", "Fit the players on Playdek's scale")]
    for job, spec in flow["jobs"].items():
        assert "secrets." not in json.dumps({k: v for k, v in spec.items() if k != "steps"}), job
    assert "secrets." not in json.dumps(flow.get("env", {}))
