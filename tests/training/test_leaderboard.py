"""The leaderboard: player ids, the record checks, and the two-stage rating fit."""

from __future__ import annotations

import dataclasses
import json
import math
import pathlib
import subprocess
import sys
from typing import Any, Dict, List, Optional, Tuple, cast

import numpy as np
import pytest
from huggingface_hub.hf_api import RepoFile

import tools.publish_hf as publish
from tools.lib import engine_behaviour
from tools.lib import leaderboard as lb
from tools.lib.player_agent import search_spec_config
from tools.lib.player_spec import (Bot, Gumbel, Policy, PlayerSpecError, Rollout, Search, load_spec,
                                   parse_player_id, player_id, spec_from_dict, spec_to_dict, workbench_search)
from ai.search.rollout_root import rollout_spec_config
from tools.leaderboard_play import plan_pairs

ROOT = pathlib.Path(__file__).resolve().parents[2]


# ---- player ids -------------------------------------------------------------------------------


@pytest.mark.parametrize("network,spec,want", [
    ("E7-A4-R1-S44@4800M", Policy(), "E7-A4-R1-S44@4800M"),
    ("E7-A4-R1-S44@4800M", Policy(0.5), "E7-A4-R1-S44@4800M~policy(temperature=0.5)"),
    ("N@1M", Gumbel(), "N@1M~gumbel(sims=256,k=8,fpu=0.2)"),
    ("N@1M", Rollout(), "N@1M~rollout(k=4,worlds=16,horizon=4,rule=z2)"),
    ("N@1M", Rollout(k=3, worlds=32, horizon=2, rule="kl0.05"), "N@1M~rollout(k=3,worlds=32,horizon=2,rule=kl0.05)"),
    ("N@1M", Search(sims=128, determinize=True),
     "N@1M~search(sims=128,determinize=true,node_filter=all,subsample=1,backend=cpp,fpu=0)"),
    (None, Bot("heuristic"), "HeuristicBot"),
])
def test_player_id_round_trips(network, spec, want) -> None:
    assert player_id(network, spec) == want
    assert parse_player_id(want) == (network, spec)
    assert spec_from_dict(spec_to_dict(spec)) == spec


def test_a_search_id_may_omit_fields_but_the_canonical_id_spells_them_all() -> None:
    network, spec = parse_player_id("N@1M~gumbel(k=4)")
    assert player_id(network, spec) == "N@1M~gumbel(sims=256,k=4,fpu=0.2)"


@pytest.mark.parametrize("bad", ["N@1M~gumbel(width=3)", "N@1M~mystery", "N@1M~search(node_filter=some)",
                                 "N@1M~policy(temperature=-1)", "N@1M~search(determinize=yes)"])
def test_bad_ids_are_refused(bad: str) -> None:
    with pytest.raises(PlayerSpecError):
        parse_player_id(bad)


def test_a_stored_spec_must_name_every_field() -> None:
    with pytest.raises(PlayerSpecError):
        spec_from_dict({"kind": "gumbel", "sims": 256, "k": 8})


@pytest.mark.parametrize("spec", [Gumbel(), Gumbel(sims=64, k=4, fpu=0.0), Search(),
                                  Search(sims=128, determinize=True, node_filter="card", subsample=0.125,
                                         backend="python", fpu=0.2)])
def test_load_spec_plays_the_spec_it_names(spec) -> None:
    """The agent string must configure the searcher exactly as the spec says."""
    path, cfg, _ = search_spec_config(load_spec(spec, "x.pt"))
    assert path == "x.pt"
    assert cfg.simulations == spec.sims
    assert cfg.fpu_reduction == spec.fpu
    if isinstance(spec, Gumbel):
        assert cfg.gumbel_k == spec.k
    else:
        assert cfg.determinize == spec.determinize
        assert cfg.node_filter == ("all" if spec.node_filter == "all" else "card_playmode")
        assert cfg.subsample == spec.subsample
        assert cfg.backend == spec.backend


@pytest.mark.parametrize("spec", [Rollout(), Rollout(k=3, worlds=32, horizon=0, rule="argmax")])
def test_a_rollout_spec_plays_the_spec_it_names(spec) -> None:
    path, cfg, _ = rollout_spec_config(load_spec(spec, "x.pt"))
    assert path == "x.pt"
    assert (cfg.k, cfg.worlds, cfg.horizon, cfg.rule) == (spec.k, spec.worlds, spec.horizon, spec.rule)


@pytest.mark.parametrize("bad", ["N~rollout(rule=best)", "N~rollout(worlds=0)", "N~rollout(horizon=-1)"])
def test_a_bad_rollout_spec_is_refused(bad: str) -> None:
    with pytest.raises(PlayerSpecError):
        parse_player_id(bad)


def test_a_policy_pins_its_own_temperature() -> None:
    assert load_spec(Policy(), "x.pt") == "temp:0:x.pt"
    assert load_spec(Policy(0.3), "x.pt") == "temp:0.3:x.pt"


# ---- the fit ----------------------------------------------------------------------------------

ENGINE = "e" * 64
BEHAVIOUR: lb.Behaviour = {"version": lb.BEHAVIOUR_VERSION, "runner": "1" * 64, "state": "2" * 64,
                           "rollout": "3" * 64, "mcts": "4" * 64}
BEHAVIOUR_ID = lb.behaviour_id(BEHAVIOUR)


def _board(tmp_path: pathlib.Path, mains: List[str], others: List[str]) -> lb.Board:
    networks = {f"{p}@1M": {"file": f"{p}.pt", "sha256": "0" * 64, "description": "", "old_names": [],
                            "hf": None, "report": None} for p in mains + others}
    players = {f"{p}@1M": {"network": f"{p}@1M", "inference": {"kind": "policy", "temperature": 0.0},
                           "description": "", "report": None} for p in mains + others}
    epochs = {"T": {"description": "", "behaviour": dict(BEHAVIOUR), "anchor": f"{mains[0]}@1M",
                    "anchor_elo": 1500.0, "main": [f"{p}@1M" for p in mains]}}
    (tmp_path / "matches").mkdir(exist_ok=True)
    board = lb.Board(networks, players, epochs, {"T": []}, tmp_path)  # type: ignore[arg-type]
    lb.save_registry(board)
    return board


def _expected(r_a: float, r_b: float, seat: float, n: int) -> Tuple[lb.Seat, lb.Seat]:
    """Win counts that are exactly the model's expectation (no draws), so the fit should return
    the generating ratings up to rounding."""
    def seat_counts(x: float) -> lb.Seat:
        w = round(n / (1 + 10 ** (-x / 400)))
        return {"w": w, "l": n - w, "d": 0}
    return seat_counts(r_a - r_b + seat), seat_counts(r_a - r_b - seat)


def _play(board: lb.Board, truth: Dict[str, float], pairs: List[Tuple[str, str]], seat: float = 40.0,
          n: int = 20000, seed: int = 10000) -> None:
    for a, b in pairs:
        us, ussr = _expected(truth[a], truth[b], seat, n)
        lb.append_record(board, "T", lb.make_record(f"{a}@1M", f"{b}@1M", us, ussr, seed, ENGINE, BEHAVIOUR_ID, "c", "d"))


def test_the_fit_recovers_known_ratings_and_the_seat_term(tmp_path: pathlib.Path) -> None:
    truth = {"A": 1500.0, "B": 1600.0, "C": 1420.0, "D": 1550.0, "E": 1700.0}
    board = _board(tmp_path, ["A", "B", "C"], ["D", "E"])
    _play(board, truth, [("A", "B"), ("A", "C"), ("B", "C"), ("D", "A"), ("D", "B"), ("E", "D")])
    assert lb.validate(board) == []
    fit = lb.fit_epoch(board, "T")
    got = {r.player[:-3]: r.elo for r in fit.ratings}
    for p, want in truth.items():
        assert abs(got[p] - want) < 2.0, (p, got[p], want)
    assert abs(fit.seat_us - 40.0) < 2.0
    assert next(r for r in fit.ratings if r.player == "A@1M").se == 0.0
    assert all(r.se > 0 for r in fit.ratings if r.player != "A@1M")


def test_games_without_a_main_player_never_move_the_main_ratings(tmp_path: pathlib.Path) -> None:
    truth = {"A": 1500.0, "B": 1600.0, "C": 1420.0, "D": 1550.0, "E": 1700.0}
    board = _board(tmp_path, ["A", "B", "C"], ["D", "E"])
    _play(board, truth, [("A", "B"), ("A", "C"), ("B", "C"), ("D", "A")])
    before = {r.player: r.elo for r in lb.fit_epoch(board, "T").ratings if r.main}
    truth_shifted = dict(truth, D=1800.0)       # a non-main player who now plays very differently
    _play(board, truth_shifted, [("D", "B"), ("E", "D")], seed=20000)
    after = {r.player: r.elo for r in lb.fit_epoch(board, "T").ratings if r.main}
    assert before == after


def test_a_player_unconnected_to_the_main_players_is_unrated(tmp_path: pathlib.Path) -> None:
    truth = {"A": 1500.0, "B": 1600.0, "D": 1550.0, "E": 1700.0}
    board = _board(tmp_path, ["A", "B"], ["D", "E"])
    _play(board, truth, [("A", "B"), ("D", "E")])
    fit = lb.fit_epoch(board, "T")
    assert sorted(fit.unrated) == ["D@1M", "E@1M"]
    assert {r.player for r in fit.ratings} == {"A@1M", "B@1M"}


def test_a_main_player_cut_off_from_the_anchor_is_an_error(tmp_path: pathlib.Path) -> None:
    board = _board(tmp_path, ["A", "B", "C"], [])
    _play(board, {"A": 1500.0, "B": 1600.0, "C": 1400.0}, [("A", "B")])
    with pytest.raises(lb.LeaderboardError, match="C@1M"):
        lb.fit_epoch(board, "T")


def test_a_networks_training_engine_need_not_be_the_epochs(tmp_path: pathlib.Path) -> None:
    """A name's E<n> is the engine its network was trained on; the epoch is the engine whose rules its
    games are played under. An E7 network anchors E8 beside one trained on E8 (owner, 2026-10-27)."""
    old, new = "E7-A8-R1-S44@7200M", "E8-A8-R1-S44@12800M"
    anchor = player_id(old, Rollout(worlds=32))
    networks = {n: {"file": f"{n}.pt", "sha256": "0" * 64, "description": "", "old_names": [], "hf": None,
                    "report": None} for n in (old, new)}
    players = {pid: {"network": net, "inference": spec_to_dict(spec), "description": "", "report": None}
               for pid, net, spec in ((anchor, old, Rollout(worlds=32)), (old, old, Policy()), (new, new, Policy()))}
    epochs = {"E8": {"description": "", "behaviour": dict(BEHAVIOUR), "anchor": anchor, "anchor_elo": 1600.0,
                     "main": [anchor, old]}}
    (tmp_path / "matches").mkdir()
    board = lb.Board(networks, players, epochs, {"E8": []}, tmp_path)  # type: ignore[arg-type]
    seats: lb.Seat = {"w": 6, "l": 4, "d": 0}
    for n, (a, b) in enumerate(((anchor, old), (new, anchor), (new, old))):
        lb.append_record(board, "E8", lb.make_record(a, b, seats, seats, n, ENGINE, BEHAVIOUR_ID, "c", "d"))
    assert lb.validate(board) == []
    fit = lb.fit_epoch(board, "E8")
    assert {r.player for r in fit.ratings} == {anchor, old, new}
    assert next(r for r in fit.ratings if r.player == anchor).elo == 1600.0


def test_views_filter_one_fit(tmp_path: pathlib.Path) -> None:
    truth = {"A": 1500.0, "B": 1600.0, "D": 1550.0, "E": 1700.0}
    board = _board(tmp_path, ["A", "B"], ["D", "E"])
    _play(board, truth, [("A", "B"), ("D", "A"), ("E", "B")])
    fit = lb.fit_epoch(board, "T")
    every = {r.player: r.elo for r in lb.view(board, fit)}
    some = {r.player: r.elo for r in lb.view(board, fit, lineages=["D"])}
    assert set(some) == {"A@1M", "B@1M", "D@1M"}
    assert all(every[p] == some[p] for p in some)
    assert set(r.player for r in lb.view(board, fit, main_only=True)) == {"A@1M", "B@1M"}


# ---- the record checks ------------------------------------------------------------------------


def test_the_same_deals_cannot_be_recorded_twice(tmp_path: pathlib.Path) -> None:
    board = _board(tmp_path, ["A", "B"], [])
    _play(board, {"A": 1500.0, "B": 1600.0}, [("A", "B")])
    with pytest.raises(lb.LeaderboardError, match="already recorded"):
        _play(board, {"A": 1500.0, "B": 1600.0}, [("B", "A")])       # either side as `a`
    _play(board, {"A": 1500.0, "B": 1600.0}, [("B", "A")], seed=20000)  # other deals are fine


def test_a_line_merged_in_twice_and_an_edited_record_are_caught(tmp_path: pathlib.Path) -> None:
    board = _board(tmp_path, ["A", "B"], [])
    _play(board, {"A": 1500.0, "B": 1600.0}, [("A", "B")])
    path = tmp_path / "matches" / "T.jsonl"
    line = path.read_text()
    path.write_text(line + line)                                      # what a union merge can leave
    assert any("appears twice" in e for e in lb.validate(lb.load(tmp_path)))
    rec = json.loads(line)
    rec["a_as_us"]["w"] += 1
    path.write_text(json.dumps(rec) + "\n")
    assert any("does not match its content" in e for e in lb.validate(lb.load(tmp_path)))


def test_an_also_accepted_digest_validates_records_and_nothing_else_does(tmp_path: pathlib.Path) -> None:
    """A build the owner ruled plays the epoch's games (`also_accepts`) may record pairings under its
    own digest; any other digest is still refused, and an accepted entry must give its reason."""
    board = _board(tmp_path, ["A", "B"], [])
    other = cast(lb.Behaviour, dict(BEHAVIOUR, runner="5" * 64))
    seats: lb.Seat = {"w": 6, "l": 4, "d": 0}
    lb.append_record(board, "T", lb.make_record("A@1M", "B@1M", seats, seats, 1, ENGINE,
                                                lb.behaviour_id(other), "c", "d"))
    assert any("not one T accepts" in e for e in lb.validate(board))

    board.epochs["T"]["also_accepts"] = [{"behaviour": other, "why": "only an ending label moved"}]
    assert lb.validate(board) == []
    assert lb.epoch_behaviours(board.epochs["T"]) == [BEHAVIOUR, other]

    board.epochs["T"]["also_accepts"] = [{"behaviour": other, "why": " "}]
    assert any("also_accepts must be" in e for e in lb.validate(board))


def test_a_non_canonical_player_id_is_caught(tmp_path: pathlib.Path) -> None:
    board = _board(tmp_path, ["A", "B"], [])
    board.players["A@1M~gumbel"] = {"network": "A@1M", "description": "", "report": None,
                                    "inference": spec_to_dict(Gumbel())}  # type: ignore[assignment]
    assert any("canonical id" in e for e in lb.validate(board))


def _report(root: pathlib.Path, subject: str, generated: bool = True, played: bool = True) -> str:
    """A behaviour report on `subject`, as tools/scripts/checkpoint_report.py heads one, under `root`;
    `played` False for the searched-leak census written alone (`--games 0`), which has no Games line."""
    rel = f"{lb.REPORT_DIR}/{''.join(c if c.isalnum() else '_' for c in subject)}.md"
    (root / lb.REPORT_DIR).mkdir(parents=True, exist_ok=True)
    by = " by `tools/scripts/checkpoint_report.py` (commit abc1234)" if generated else ""
    games = "* **Games:** 16,384 greedy self-play games of the checkpoint against itself.\n" if played else ""
    (root / rel).write_text(f"# {subject}: behaviour report\n\nGenerated 2026-10-17{by}.\n\n"
                            f"## The checkpoint\n\n* **File:** `x.pt`\n{games}", encoding="utf-8")
    return rel


def test_a_search_player_never_links_its_networks_greedy_report(tmp_path: pathlib.Path) -> None:
    """A network's report is of its greedy play: the greedy player links it, a search or a sampled
    policy of the same weights does not -- only a report of its own, once there is one."""
    board = _board(tmp_path, ["A"], [])
    net_report = _report(tmp_path, "A@1M")
    board.networks["A@1M"]["report"] = net_report
    search = player_id("A@1M", Gumbel())
    board.players[search] = {"network": "A@1M", "inference": spec_to_dict(Gumbel()), "description": "",
                             "report": None}
    sampled = player_id("A@1M", Policy(temperature=0.5))
    board.players[sampled] = {"network": "A@1M", "inference": spec_to_dict(Policy(temperature=0.5)),
                              "description": "", "report": None}
    assert lb.validate(board, tmp_path) == []
    assert lb.player_report(board, "A@1M") == net_report
    assert lb.player_report(board, search) is None and lb.player_report(board, sampled) is None
    own = _report(tmp_path, search)
    board.players[search]["report"] = own
    assert lb.validate(board, tmp_path) == []
    assert lb.player_report(board, search) == own
    # The page's data carries the same link.
    players = cast(Dict[str, Dict[str, object]], lb.site_data(board)["players"])
    assert players[search]["report"] == own and players["A@1M"]["report"] == net_report
    board.players[search]["report"] = f"{lb.REPORT_DIR}/no_such_report.md"
    assert any("does not exist" in e for e in lb.validate(board, tmp_path))


def test_a_report_link_is_the_scripts_behaviour_report_on_that_network_or_player(tmp_path: pathlib.Path) -> None:
    board = _board(tmp_path, ["A", "B"], [])
    search = player_id("A@1M", Gumbel())
    board.players[search] = {"network": "A@1M", "inference": spec_to_dict(Gumbel()), "description": "",
                             "report": None}
    # A plan or a research log that mentions the network is not its report.
    (tmp_path / "research" / "log").mkdir(parents=True)
    (tmp_path / "research" / "log" / "P33_plan.md").write_text("# P33: the plan\n", encoding="utf-8")
    board.players[search]["report"] = "research/log/P33_plan.md"
    assert any("not a behaviour report from tools/scripts/checkpoint_report.py" in e
               for e in lb.validate(board, tmp_path))
    # Nor is the report on another player -- here, its network's greedy play -- or on another network.
    board.players[search]["report"] = _report(tmp_path, "A@1M")
    assert any("not a report on" in e for e in lb.validate(board, tmp_path))
    board.players[search]["report"] = None
    board.networks["A@1M"]["report"] = _report(tmp_path, "B@1M")
    assert any("not a report on A@1M" in e for e in lb.validate(board, tmp_path))
    # Nor is the searched-leak census written alone: it has no self-play.
    board.networks["A@1M"]["report"] = None
    board.players[search]["report"] = _report(tmp_path, search, played=False)
    assert any("has no self-play" in e for e in lb.validate(board, tmp_path))
    board.players[search]["report"] = None
    # A file in the directory that the script did not write is not one either.
    board.networks["A@1M"]["report"] = _report(tmp_path, "A@1M", generated=False)
    assert any("does not say it was generated" in e for e in lb.validate(board, tmp_path))
    # A report written under a network's earlier name stays its report.
    board.networks["A@1M"]["old_names"] = ["A-old@1M"]
    board.networks["A@1M"]["report"] = _report(tmp_path, "A-old@1M")
    board.players[search]["report"] = _report(tmp_path, player_id("A-old@1M", Gumbel()))
    assert lb.validate(board, tmp_path) == []


def test_site_data_aggregates_a_pair_from_the_first_players_side(tmp_path: pathlib.Path) -> None:
    board = _board(tmp_path, ["A", "B"], [])
    lb.append_record(board, "T", lb.make_record("B@1M", "A@1M", {"w": 7, "l": 3, "d": 0},
                                                {"w": 6, "l": 3, "d": 1}, 1, ENGINE, BEHAVIOUR_ID, "c", "d"))
    pair = lb.site_data(board)["epochs"]["T"]["pairs"][0]  # type: ignore[index]
    assert (pair["a"], pair["b"]) == ("A@1M", "B@1M")
    # A as US is B as USSR (B won 6, lost 3): A won 3, lost 6.
    assert pair["a_as_us"] == {"w": 3, "l": 6, "d": 1}
    assert pair["a_as_ussr"] == {"w": 3, "l": 7, "d": 0}


def test_a_record_is_of_its_epoch_by_behaviour_or_by_a_frozen_pre_digest_engine(tmp_path: pathlib.Path) -> None:
    board = _board(tmp_path, ["A", "B"], [])
    win: lb.Seat = {"w": 1, "l": 0, "d": 0}
    board.matches["T"] = [lb.make_record("A@1M", "B@1M", win, win, 1, ENGINE, BEHAVIOUR_ID, "c", "d")]
    assert lb.validate(board) == []
    # Another engine's behaviour is another epoch, whatever the engine's sources.
    other = lb.behaviour_id({**BEHAVIOUR, "mcts": "5" * 64})
    board.matches["T"] = [lb.make_record("A@1M", "B@1M", win, win, 1, ENGINE, other, "c", "d")]
    assert any("not one T accepts" in e for e in lb.validate(board))
    # A record from before behaviour digests is held to the epoch's frozen list of engines.
    old = {k: v for k, v in lb.make_record("A@1M", "B@1M", win, win, 1, ENGINE, "", "c", "d").items()
           if k not in ("id", "behaviour")}
    board.matches["T"] = [cast(lb.Record, {"id": lb.record_id(old), **old})]
    assert any("pre_digest_engines" in e for e in lb.validate(board))
    board.epochs["T"]["pre_digest_engines"] = [ENGINE]
    assert lb.validate(board) == []
    # An epoch's behaviour names every section.
    board.epochs["T"]["behaviour"] = cast(lb.Behaviour, {k: v for k, v in BEHAVIOUR.items() if k != "mcts"})
    assert any("behaviour must be" in e for e in lb.validate(board))


def test_a_digest_of_another_version_differs_in_every_section() -> None:
    assert lb.behaviour_mismatch(BEHAVIOUR, BEHAVIOUR) == []
    assert lb.behaviour_mismatch(BEHAVIOUR, {**BEHAVIOUR, "state": "9" * 64}) == ["state"]
    assert lb.behaviour_mismatch(BEHAVIOUR, {**BEHAVIOUR, "version": 0}) == list(lb.BEHAVIOUR_SECTIONS)


def test_the_stand_in_network_is_a_function_of_each_row_alone() -> None:
    rng = np.random.default_rng(0)
    obs = rng.standard_normal((3, 40)).astype(np.float32)
    masks = (rng.random((3, 12)) < 0.5).astype(np.uint8)
    masks[:, 0] = 1
    logits, probs, values = engine_behaviour.stand_in(obs, masks)
    # A row's answer does not depend on the rows beside it.
    one = engine_behaviour.stand_in(obs[1:2], masks[1:2])
    assert np.array_equal(one[0][0], logits[1]) and np.array_equal(one[1][0], probs[1]) and one[2][0] == values[1]
    # Integer logits, exact in float32; probabilities on the legal moves only, summing to one.
    assert np.array_equal(logits, np.round(logits)) and logits.max() < 2 ** 24
    assert np.all(probs[masks == 0] == 0) and np.allclose(probs.sum(axis=1), 1.0)
    assert np.all(np.abs(values) <= 1)
    # Any change to a row changes its answer.
    obs[1, 7] += 1.0
    assert not np.array_equal(engine_behaviour.stand_in(obs[1:2], masks[1:2])[0][0], logits[1])


def test_the_behaviour_digest_is_the_same_every_time_it_is_taken() -> None:
    first = engine_behaviour.measure()
    assert set(first) == {"version", *lb.BEHAVIOUR_SECTIONS} and first["version"] == lb.BEHAVIOUR_VERSION
    assert engine_behaviour.measure() == first


def test_the_keyed_rollout_batch_moves_with_each_of_its_flags() -> None:
    """keyed, aligned and box4 each change what the deployed root's batch hashes, so a change to any of
    them moves the rollout section (box4 acts only at a Space Race box 4 headline, which the positions
    taken by cadence never are)."""
    positions: List[engine_behaviour.ts.GameState] = []
    box4: List[engine_behaviour.ts.GameState] = []
    engine_behaviour._runner(engine_behaviour._Stream(), positions, box4)
    assert len(box4) == engine_behaviour._BOX4_POSITIONS
    assert all(engine_behaviour.ts.revealed_headline(st, st.ctx().decision_player) for st in box4)
    deployed = engine_behaviour._ROLLOUTS[-1]
    assert deployed.keyed and deployed.aligned and deployed.box4 and deployed.box4_headlines

    def digest(case: engine_behaviour._RolloutCase) -> str:
        out = engine_behaviour._Stream()
        engine_behaviour._rollout(out, positions, box4, [case])
        return out.hexdigest()

    full = digest(deployed)
    assert digest(dataclasses.replace(deployed, keyed=False)) != full
    assert digest(dataclasses.replace(deployed, aligned=False)) != full
    assert digest(dataclasses.replace(deployed, box4=False)) != full


def test_a_network_is_trained_under_the_last_code_of_each_kind_in_its_name() -> None:
    assert lb.network_codes("E7-A8-R1-S44@6400M+R34@7200M+R48@9170M+A13@11920..12000M") == ("A13", "R48")
    assert lb.network_codes("E7-A8-R1-S44@6400M+(R34,R34-S45,R34-S46)@7120..7200M") == ("A8", "R34")
    assert lb.network_codes("E7-A8-R1-S44@6400M+(S44,45,46)@6720..6800M") == ("A8", "R1")
    assert lb.network_codes("E7-20-44-4390M.45") == (None, None)         # the old scheme names no codes


def test_the_codes_tab_lists_run_codes_with_the_networks_trained_under_each(tmp_path: pathlib.Path) -> None:
    board = _board(tmp_path, ["A", "B"], [])
    board.networks = {"E7-A8-R1-S44@6800M": board.networks["A@1M"],
                      "E7-A8-R1-S44@6400M+R34@7200M+A13@8000M": board.networks["B@1M"]}
    codes_file = tmp_path / "run_codes.json"
    codes_file.write_text(json.dumps({
        "architectures": {"A13": {"description": "wide", "flags": {"ladder_entity_proj_dim": 512}},
                          "A8": {"description": "heads", "flags": {}}},
        "recipes": {"R34": {"description": "gap", "flags": {"logit_gap_coef": 0.001}, "first": "run_x"},
                    "R1": {"description": "plain", "flags": {}}, "R2": {"description": "unused", "flags": {}}}}))
    codes = cast(Dict[str, Dict[str, Dict[str, object]]], lb.codes_data(board, codes_file))
    assert list(codes["architectures"]) == ["A8", "A13"]                  # in numeric order
    assert codes["architectures"]["A13"]["networks"] == ["E7-A8-R1-S44@6400M+R34@7200M+A13@8000M"]
    assert codes["architectures"]["A8"]["networks"] == ["E7-A8-R1-S44@6800M"]
    assert codes["recipes"]["R34"]["flags"] == {"logit_gap_coef": 0.001} and codes["recipes"]["R34"]["first"] == "run_x"
    assert codes["recipes"]["R2"]["networks"] == []
    assert codes["by_network"]["E7-A8-R1-S44@6400M+R34@7200M+A13@8000M"] == {"architecture": "A13", "recipe": "R34"}
    # The real file: every code a network on the real board names is defined there.
    real = lb.load()
    data = cast(Dict[str, Dict[str, Dict[str, object]]], lb.site_data(real)["codes"])
    for net, pair in data["by_network"].items():
        for kind, code in (("architectures", pair["architecture"]), ("recipes", pair["recipe"])):
            assert code is None or code in data[kind], f"{net} names {code}, which run_codes.json does not define"


def test_plan_pairs_skips_what_is_recorded(tmp_path: pathlib.Path) -> None:
    board = _board(tmp_path, ["A", "B", "C"], ["D"])
    _play(board, {"A": 1500.0, "B": 1600.0, "D": 1550.0}, [("D", "A")])
    play, skipped = plan_pairs(board, "T", ["D@1M"], None, False, None)
    assert skipped == [("D@1M", "A@1M")]
    assert play == [("D@1M", "B@1M"), ("D@1M", "C@1M")]
    play, _ = plan_pairs(board, "T", ["D@1M"], None, False, 20000)
    assert len(play) == 3
    with pytest.raises(lb.LeaderboardError):
        plan_pairs(board, "T", ["D@1M"], None, False, 10000)


# ---- the committed records --------------------------------------------------------------------


def test_the_committed_leaderboard_is_consistent() -> None:
    board = lb.load()
    assert lb.validate(board) == []
    for name in board.epochs:
        if board.matches[name]:
            lb.fit_epoch(board, name)


def test_the_fit_needs_neither_torch_nor_the_engine() -> None:
    """CI builds the page's data with plain python3 (tools/scripts/build_web.sh)."""
    code = ("import sys; import tools.leaderboard; "
            "bad = [m for m in ('torch', 'numpy', 'ts_engine') if m in sys.modules]; "
            "assert not bad, bad")
    subprocess.run([sys.executable, "-c", code], check=True, cwd=ROOT, env={"PYTHONPATH": str(ROOT)})
    assert math.isfinite(lb.ELO_SCALE)


# ---- publishing -------------------------------------------------------------------------------


def test_the_hf_repo_mirrors_data_checkpoints(tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A run's files keep their run directory; a model under _models/ keeps its readable name
    even when it is a symlink into _soups/."""
    data = tmp_path / "data"
    (data / "checkpoints" / "R_20260101_000000").mkdir(parents=True)
    (data / "checkpoints" / "_soups").mkdir()
    (data / "checkpoints" / "_models").mkdir()
    (data / "checkpoints" / "_soups" / "soup.pt").write_bytes(b"x")
    (data / "checkpoints" / "_models" / "N@1M+(S1,2)@2M.pt").symlink_to("../_soups/soup.pt")
    monkeypatch.setattr(publish, "data_path", lambda *p: str(data.joinpath(*p)))
    run_file = data / "checkpoints" / "R_20260101_000000" / "snapshot_10steps.pt"
    assert publish.repo_path(str(run_file)) == "R_20260101_000000/snapshot_10steps.pt"
    assert publish.repo_path(str(data / "checkpoints" / "_models" / "N@1M+(S1,2)@2M.pt")) == "_models/N@1M+(S1,2)@2M.pt"
    with pytest.raises(lb.LeaderboardError):
        publish.repo_path(str(tmp_path / "elsewhere.pt"))


def test_only_the_final_snapshot_the_final_swa_and_leaderboard_files_get_an_onnx() -> None:
    files = ["snapshot_10000steps.pt", "snapshot_990000steps.pt", "snapshot_1000000steps.pt",
             "swa_80-160M.pt", "swa_880-960M.pt", "swa_920-1000M.pt", "snapshot_0s.pt"]
    assert publish.onnx_selection(files, set()) == ["snapshot_1000000steps.pt", "swa_920-1000M.pt"]
    assert publish.onnx_selection(files, {"snapshot_10000steps.pt", "elsewhere.pt"}) == [
        "snapshot_1000000steps.pt", "snapshot_10000steps.pt", "swa_920-1000M.pt"]
    assert publish.onnx_selection(["snapshot_5steps.pt"], set()) == ["snapshot_5steps.pt"]


def test_snapshot_final_is_never_published_and_never_registered(tmp_path: pathlib.Path) -> None:
    """Snapshots are referred to by step; `snapshot_final.pt` is a copy that names none."""
    assert publish._excluded("snapshot_final.pt") and publish._excluded("snapshot_final.onnx")
    board = _board(tmp_path, ["A", "B"], [])
    board.networks["A@1M"]["file"] = "checkpoints/R_20260101_000000/snapshot_final.pt"
    assert any("by step" in e for e in lb.validate(board))



def test_a_rollout_players_stop_and_leaf_are_named_only_when_not_the_measured_root() -> None:
    """Rollout ids predate the stop and leaf settings: an existing id and stored spec keep meaning the
    measured root, and a non-default stop or leaf is spelled out in the id and in the agent string."""
    old = "E7-A8-R1-S44@6800M~rollout(k=4,worlds=16,horizon=4,rule=z2)"
    net, spec = parse_player_id(old)
    assert player_id(net, spec) == old
    assert spec_to_dict(spec) == {"kind": "rollout", "k": 4, "worlds": 16, "horizon": 4, "rule": "z2"}
    assert spec_from_dict({"kind": "rollout", "k": 4, "worlds": 16, "horizon": 4, "rule": "z2"}) == spec
    assert load_spec(spec, "m.pt") == "rollout:m.pt:4:16:4:z2"
    new = "E7-A8-R1-S44@6800M~rollout(k=4,worlds=16,horizon=4,rule=z2,stop=earlier,leaf=own)"
    net, spec = parse_player_id(new)
    assert isinstance(spec, Rollout) and (spec.stop, spec.leaf) == ("earlier", "own")
    assert player_id(net, spec) == new
    assert spec_from_dict(spec_to_dict(spec)) == spec
    assert load_spec(spec, "m.pt") == "rollout:m.pt:4:16:4:z2:earlier:own"
    with pytest.raises(PlayerSpecError):
        parse_player_id("E7-A8-R1-S44@6800M~rollout(stop=never)")



def test_a_rollout_players_leaf_net_is_a_registered_network_named_only_when_set() -> None:
    """P34's two-network root: `leaf_net` names the evaluator that reads the leaves. Ids without it
    keep meaning the one-network root; with it the id, the stored spec and the agent string carry it,
    the agent string as the leaf checkpoint rollout_root already reads, and validate refuses an
    evaluator that is not registered."""
    one = "E7-A8-R1-S44@6800M~rollout(k=4,worlds=16,horizon=4,rule=z2,stop=earlier,leaf=own)"
    net, spec = parse_player_id(one)
    assert isinstance(spec, Rollout) and spec.leaf_net == "" and player_id(net, spec) == one
    two = one[:-1] + ",leaf_net=E7-A8-R1-S44@6800M+A15-R52)"
    net, spec = parse_player_id(two)
    assert isinstance(spec, Rollout) and spec.leaf_net == "E7-A8-R1-S44@6800M+A15-R52"
    assert player_id(net, spec) == two
    assert spec_from_dict(spec_to_dict(spec)) == spec
    agent = load_spec(spec, "m.pt", "leaf.pt")
    assert agent == "rollout:m.pt:4:16:4:z2:earlier:own:leaf.pt"
    path, cfg, _ = rollout_spec_config(agent)
    assert path == "m.pt" and (cfg.stop, cfg.leaf_side) == ("earlier", "own")
    with pytest.raises(PlayerSpecError):
        load_spec(spec, "m.pt")              # the evaluator's file is required
    with pytest.raises(PlayerSpecError):
        parse_player_id(one[:-1] + ",leaf_net=(S44,45,46)@6800M)")
    board = lb.load()
    board.players[two] = {"network": net, "inference": spec_to_dict(spec), "description": "", "report": None}
    assert any("leaf_net" in e and "not in networks.json" in e for e in lb.validate(board))


def test_a_rollout_players_chance_is_named_only_when_keyed() -> None:
    """`chance=keyed` (bindings/keyed_chance.hpp) is spelled in the id and the agent string, and the
    rollout root reads it; ids without it keep meaning the shared-stream root."""
    one = "E7-A8-R1-S44@6800M~rollout(k=4,worlds=16,horizon=4,rule=z2,stop=earlier,leaf=own)"
    net, spec = parse_player_id(one)
    assert isinstance(spec, Rollout) and spec.chance == "shared" and player_id(net, spec) == one
    keyed = one[:-1] + ",chance=keyed)"
    net, spec = parse_player_id(keyed)
    assert isinstance(spec, Rollout) and spec.chance == "keyed" and player_id(net, spec) == keyed
    assert spec_from_dict(spec_to_dict(spec)) == spec
    agent = load_spec(spec, "m.pt")
    assert agent == "rollout:m.pt:4:16:4:z2:earlier:own:keyed"
    path, cfg, label = rollout_spec_config(agent)
    assert cfg.keyed and path == "m.pt" and "keyed" in label
    assert not rollout_spec_config(load_spec(parse_player_id(one)[1], "m.pt"))[1].keyed
    # keyed2, the packing that keeps every situation field (bindings/keyed_chance.hpp), is its own
    # chance, so the keyed players already measured keep their meaning.
    keyed2 = one[:-1] + ",chance=keyed2)"
    net, spec = parse_player_id(keyed2)
    assert isinstance(spec, Rollout) and spec.chance == "keyed2" and player_id(net, spec) == keyed2
    assert spec_from_dict(spec_to_dict(spec)) == spec
    agent = load_spec(spec, "m.pt")
    assert agent == "rollout:m.pt:4:16:4:z2:earlier:own:keyed2"
    _, cfg, label = rollout_spec_config(agent)
    assert cfg.keyed2 and not cfg.keyed and label.endswith("-keyed2-z2")
    assert not rollout_spec_config("rollout:m.pt:4:16:4:z2:earlier:own:keyed")[1].keyed2
    # keyedsym, keyed's USSR packing on both sides, is a third chance; it cannot be spelled beside keyed2.
    sym = one[:-1] + ",chance=keyedsym)"
    net, spec = parse_player_id(sym)
    assert isinstance(spec, Rollout) and spec.chance == "keyedsym" and player_id(net, spec) == sym
    assert spec_from_dict(spec_to_dict(spec)) == spec
    agent = load_spec(spec, "m.pt")
    assert agent == "rollout:m.pt:4:16:4:z2:earlier:own:keyedsym"
    _, cfg, label = rollout_spec_config(agent)
    assert cfg.keyedsym and not cfg.keyed and not cfg.keyed2 and label.endswith("-keyedsym-z2")
    assert not rollout_spec_config("rollout:m.pt:4:16:4:z2:earlier:own:keyed2")[1].keyedsym
    with pytest.raises(ValueError, match="exclusive"):
        rollout_spec_config("rollout:m.pt:4:16:4:z2:earlier:own:keyed2:keyedsym")
    # B6's soup under the raced 64-world root, keyedsym: the id the leaderboard registers round-trips.
    b6 = ("E8.1-A8-R132-S45@11160M+(S45,46,47)@11520..11600M~rollout(k=4,worlds=64,horizon=4,rule=z2,stop=earlier,"
          "leaf=own,chance=keyedsym,aligned=true,box4=true,race=16)")
    net, spec = parse_player_id(b6)
    assert isinstance(spec, Rollout) and spec.chance == "keyedsym" and spec.race == 16 and player_id(net, spec) == b6
    assert workbench_search(spec) == "rollout:4:64:4:z2:earlier:own:keyedsym:aligned:box4:race16"
    with pytest.raises(PlayerSpecError):
        parse_player_id(one[:-1] + ",chance=fresh)")


def test_the_oracle_root_is_a_tournament_spec_only() -> None:
    """`oracle` (the opponent's true hand in every world, P35 item 3's ceiling) is read from a rollout
    agent string and labelled; it is not a leaderboard player, so no player id spells it."""
    path, cfg, label = rollout_spec_config("rollout:m.pt:4:16:4:z2:earlier:own:keyed:oracle")
    assert cfg.oracle and cfg.keyed and label.endswith("-keyed-oracle-z2") and path == "m.pt"
    assert not rollout_spec_config("rollout:m.pt:4:16:4:z2:earlier:own:keyed")[1].oracle


def test_an_epochs_strong_tier_must_be_main_players() -> None:
    """epochs.json's optional `strong` list: distinct main players, or validate reports it."""
    board = lb.load()
    ep = board.epochs["E7"]
    good: lb.Epoch = {**ep, "strong": list(ep["main"][-2:])}
    bad: lb.Epoch = {**ep, "strong": ["not-a-main-player"]}
    assert not [e for e in lb.validate(dataclasses.replace(board, epochs={"E7": good})) if "strong" in e]
    assert [e for e in lb.validate(dataclasses.replace(board, epochs={"E7": bad})) if "strong" in e]


def test_the_belief_root_is_spelled_in_the_id_and_the_agent_string() -> None:
    """`belief=<hands>` (P35 3b, bindings/belief.hpp) names the belief root; ids without it keep meaning
    the root with uniform redeals."""
    one = "E7-A8-R1-S44@6800M~rollout(k=4,worlds=16,horizon=4,rule=z2,stop=earlier,leaf=own,chance=keyed)"
    net, spec = parse_player_id(one)
    assert isinstance(spec, Rollout) and spec.belief == 0 and player_id(net, spec) == one
    withb = one[:-1] + ",belief=2048)"
    net, spec = parse_player_id(withb)
    assert isinstance(spec, Rollout) and spec.belief == 2048 and player_id(net, spec) == withb
    assert spec_from_dict(spec_to_dict(spec)) == spec
    agent = load_spec(spec, "m.pt")
    assert agent == "rollout:m.pt:4:16:4:z2:earlier:own:keyed:belief2048"
    assert rollout_spec_config(agent)[1].belief == 2048
    with pytest.raises(PlayerSpecError):
        parse_player_id(one[:-1] + ",belief=-1)")


def test_the_aligned_horizon_is_spelled_in_the_id_and_the_agent_string() -> None:
    """`aligned=true` (bindings/rollout_batch.hpp's Config::aligned) counts every candidate's horizon from
    the end of the mover's own action round; ids without it keep meaning the count they were measured with."""
    one = "E7-A8-R1-S44@6800M~rollout(k=4,worlds=16,horizon=4,rule=z2,stop=earlier,leaf=own,chance=keyed)"
    net, spec = parse_player_id(one)
    assert isinstance(spec, Rollout) and not spec.aligned and player_id(net, spec) == one
    assert spec_from_dict({k: v for k, v in spec_to_dict(spec).items() if k != "aligned"}) == spec
    witha = one[:-1] + ",aligned=true)"
    net, spec = parse_player_id(witha)
    assert isinstance(spec, Rollout) and spec.aligned and player_id(net, spec) == witha
    assert spec_from_dict(spec_to_dict(spec)) == spec
    agent = load_spec(spec, "m.pt")
    assert agent == "rollout:m.pt:4:16:4:z2:earlier:own:keyed:aligned"
    assert rollout_spec_config(agent)[1].aligned
    withb = one[:-1] + ",aligned=true,box4=true)"
    net, spec = parse_player_id(withb)
    assert isinstance(spec, Rollout) and spec.box4 and player_id(net, spec) == withb
    assert spec_from_dict(spec_to_dict(spec)) == spec
    agent = load_spec(spec, "m.pt")
    assert agent == "rollout:m.pt:4:16:4:z2:earlier:own:keyed:aligned:box4"
    assert rollout_spec_config(agent)[1].box4
    withr = one[:-1] + ",aligned=true,box4=true,race=8)"
    net, spec = parse_player_id(withr)
    assert isinstance(spec, Rollout) and spec.race == 8 and player_id(net, spec) == withr
    assert spec_from_dict(spec_to_dict(spec)) == spec
    agent = load_spec(spec, "m.pt")
    assert agent == "rollout:m.pt:4:16:4:z2:earlier:own:keyed:aligned:box4:race8"
    assert rollout_spec_config(agent)[1].race == 8
    with pytest.raises(PlayerSpecError):
        parse_player_id(one[:-1] + ",race=16)")


def test_the_workbench_plays_a_player_as_its_search_spelled_out() -> None:
    """`workbench_search` writes the web workbench's `search=` value with every field spelled out: the
    workbench's defaults for a field left out are the measured root's, not this module's. What it
    cannot run is None."""
    assert workbench_search(Policy()) == "policy"
    assert workbench_search(Policy(temperature=0.5)) is None
    assert workbench_search(Rollout()) == "rollout:4:16:4:z2:boundaries:mover"
    assert workbench_search(Rollout(worlds=32, stop="earlier", leaf="own", chance="keyed", aligned=True,
                                    box4=True)) == "rollout:4:32:4:z2:earlier:own:keyed:aligned:box4"
    assert workbench_search(Rollout(belief=512, rule="kl1.5")) == "rollout:4:16:4:kl1.5:boundaries:mover:belief512"
    assert workbench_search(Rollout(worlds=32, race=16)) == "rollout:4:32:4:z2:boundaries:mover:race16"
    assert workbench_search(Rollout(leaf_net="E7-A1-R1-S44@1M")) is None
    assert workbench_search(Gumbel()) is None and workbench_search(Search()) is None and workbench_search(Bot()) is None


class _FakeHub:
    """The calls `build_manifest` makes, over a repo holding `files` and `manifest` as its models.json."""

    def __init__(self, tmp: pathlib.Path, files: List[str], manifest: Optional[Dict[str, Any]]) -> None:
        self.tmp, self.files, self.manifest = tmp, files, manifest

    def list_repo_tree(self, repo: str, recursive: bool) -> List[RepoFile]:
        return [RepoFile(path=f, size=1, oid="o", lfs=None, last_commit=None, security=None) for f in self.files]

    def get_paths_info(self, repo: str, paths: List[str], expand: bool) -> List[RepoFile]:
        return []

    def file_exists(self, repo: str, name: str) -> bool:
        return name == publish.MANIFEST and self.manifest is not None

    def hf_hub_download(self, repo: str, name: str) -> str:
        out = self.tmp / name
        out.write_text(json.dumps(self.manifest), encoding="utf-8")
        return str(out)


def test_the_default_search_is_kept_until_a_default_names_one(tmp_path: pathlib.Path) -> None:
    """models.json's default_search: a publish keeps the repo's, `default <onnx>` keeps it with the new
    model, `default --player` (both given) replaces both."""
    keyed = "rollout:4:32:4:z2:earlier:own:keyed:aligned:box4"
    hub = cast(Any, _FakeHub(tmp_path, ["a.onnx", "b.onnx"], {"schema": 1, "default": "a.onnx", "default_search": keyed,
                                                                "models": []}))
    m = publish.build_manifest(hub, "r", [])
    assert (m["default"], m["default_search"]) == ("a.onnx", keyed)
    m = publish.build_manifest(hub, "r", [], "b.onnx")
    assert (m["default"], m["default_search"]) == ("b.onnx", keyed)
    m = publish.build_manifest(hub, "r", [], "b.onnx", "policy")
    assert (m["default"], m["default_search"]) == ("b.onnx", "policy")
    # A repo whose models.json predates the field names none.
    m = publish.build_manifest(cast(Any, _FakeHub(tmp_path, ["a.onnx"], None)), "r", [], "a.onnx")
    assert m["default_search"] is None

