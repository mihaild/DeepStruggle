"""tools/scripts/league.py: which resume state a main-latest exploiter generation starts from; and P45's seat
league driver (tools/scripts/league_seats.py, check 10)."""

from __future__ import annotations

import json
import os
from typing import Dict, List, Tuple

from tools.scripts.league import exploiter_run_name, latest_resume
from tools.scripts.league_seats import LINK_RE, Ops, SeatLeague


def _touch(d: str, name: str) -> None:
    with open(os.path.join(d, name), "w") as f:
        f.write("x")


def test_latest_resume_takes_the_newest_tagged_state_and_ignores_the_rolling_one(tmp_path) -> None:
    d = str(tmp_path)
    for n in ("resume_10027008steps.pt", "resume_40042496steps.pt", "resume_state.pt",
              "snapshot_50000000steps.pt", "resume_9000000steps.pt"):
        _touch(d, n)
    assert latest_resume(d) == os.path.join(d, "resume_40042496steps.pt")


def test_latest_resume_waits_for_the_start_threshold(tmp_path) -> None:
    d = str(tmp_path)
    _touch(d, "resume_10027008steps.pt")
    assert latest_resume(d, 40_000_000) is None
    _touch(d, "resume_40042496steps.pt")
    assert latest_resume(d, 40_000_000) == os.path.join(d, "resume_40042496steps.pt")


def test_the_driver_refuses_a_non_conforming_exploiter_name_before_launching(tmp_path) -> None:
    import pytest
    from tools.scripts.league import main
    base = ["--main-name", "E5-21-43", "--main-steps", "1", "--seed", "43", "--exploiter-reset", "main-latest",
            "--league-dir", str(tmp_path / "l"), "--log-dir", str(tmp_path / "g"), "--train-args", "",
            "--main-description", "x", "--exploiter-description", "x"]
    with pytest.raises(SystemExit, match="not a run name"):
        main(base + ["--exploiter-name", "E5-22-43b"])
    assert not (tmp_path / "l").exists()


def test_a_continued_league_keeps_counting_generations(tmp_path) -> None:
    """--first-generation must be at least 1, and is checked before anything launches."""
    import pytest
    from tools.scripts.league import main
    base = ["--main-name", "E5-21-43", "--main-steps", "1", "--seed", "43", "--exploiter-reset", "main-latest",
            "--league-dir", str(tmp_path / "l"), "--log-dir", str(tmp_path / "g"), "--train-args", "",
            "--main-description", "x", "--exploiter-description", "x", "--exploiter-name", "E5-22-43"]
    with pytest.raises(SystemExit, match="first-generation"):
        main(base + ["--first-generation", "0"])
    assert not (tmp_path / "l").exists()


def test_a_step_template_names_each_generation_by_its_branch_point() -> None:
    """In the run-name grammar a generation is a branch of the main agent at its reset state."""
    t = "E7-A8-R1-S44@6400M+R19@{step}M+R3"
    assert exploiter_run_name(t, 1, 6_410_053_632) == "E7-A8-R1-S44@6400M+R19@6410M+R3"
    assert exploiter_run_name(t, 7, 6_719_987_712) == "E7-A8-R1-S44@6400M+R19@6720M+R3"
    assert exploiter_run_name("E7-22-44", 3, 2_910_000_000) == "E7-22-44-3"


# ---- P45 check 10: the seat league driver, its world faked ------------------------------------------------

class _World:
    """A fake main and fake exploiter runs: directories with snapshots, resume states and metadata."""

    def __init__(self, root: str, margins: Dict[Tuple[str, int], float], main_gens: int = 3) -> None:
        self.root, self.margins = root, margins
        self.main_dir = os.path.join(root, "main")
        os.makedirs(self.main_dir)
        self.main_step = 1_000_000_000
        self._snap(self.main_dir, self.main_step)
        _touch(self.main_dir, f"resume_{self.main_step}steps.pt")
        self.trained: List[Tuple[str, str, str, str, int]] = []
        self.gen = 0
        self.main_gens = main_gens

    @staticmethod
    def _snap(d: str, n: int) -> None:
        _touch(d, f"snapshot_{n}steps.pt")

    def train(self, seat: str, start: str, target: str, name: str, steps: int, desc: str) -> str:
        d = os.path.join(self.root, f"{name}_20261112_000000")
        os.makedirs(d)
        with open(os.path.join(d, "metadata.json"), "w") as f:
            json.dump({"opponent_lock_side": seat}, f)
        for n in (steps - 40_000_000, steps):
            self._snap(d, n)
            _touch(d, f"resume_{n}steps.pt")
        self.trained.append((seat, start, target, name, steps))
        return d

    def swa(self, run_dir: str, hi: int) -> str:
        return os.path.join(run_dir, f"swa_{hi // 1_000_000}.pt")

    def match(self, a: str, b: str) -> Dict[str, float]:
        if a == b:                                     # the target's self-play bar
            return {"a_as_us": 0.49, "a_as_ussr": 0.49}
        seat = "us" if "/us_" in a or "+R108" in a else "ussr"
        m = self.margins.get((seat, self.gen_of(a)), 0.05)
        return {"a_as_us": 0.49 + m, "a_as_ussr": 0.49 + m}

    def gen_of(self, path: str) -> int:
        return 1 + sum(1 for t in self.trained[:-1] if t[0] == self.trained[-1][0])

    def main_steps(self) -> int:
        return self.main_step

    def alive(self) -> bool:
        return self.gen < self.main_gens

    def sleep(self, _: float) -> None:            # the main trains 800M between polls
        self.main_step += 800_000_000
        self._snap(self.main_dir, self.main_step)
        _touch(self.main_dir, f"resume_{self.main_step}steps.pt")
        self.gen += 1


def _league(tmp_path, margins: Dict[Tuple[str, int], float], gens: int = 3, capacity: int = 4) -> Tuple[SeatLeague, _World]:
    w = _World(str(tmp_path / "w"), margins, gens)
    league_dir = str(tmp_path / "league")
    os.makedirs(league_dir)
    ops = Ops(train=w.train, swa=w.swa, match=w.match, main_steps=w.main_steps, main_alive=w.alive, sleep=w.sleep)
    lg = SeatLeague(main_dir=w.main_dir, league_dir=league_dir, every=800_000_000, gen_steps=400_000_000,
                    starts={"us": "/r108/resume_15610019840steps.pt", "ussr": "/r109/resume_15610019840steps.pt"},
                    names={"us": "E8-A8-R102-S44@{step}M+R108", "ussr": "E8-A8-R102-S44@{step}M+R109"},
                    description="test", ops=ops, capacity=capacity, first_at=w.main_step, poll=0.0)
    return lg, w


def _events(lg: SeatLeague) -> List[dict]:
    return [json.loads(line) for line in open(os.path.join(lg.league_dir, "events.jsonl"))]


def test_10_a_generation_trains_us_then_ussr_against_the_swa_at_its_start(tmp_path) -> None:
    lg, w = _league(tmp_path, {}, gens=2)
    lg.run()
    seats = [t[0] for t in w.trained]
    assert seats == ["us", "ussr", "us", "ussr"]
    assert w.trained[0][2] == w.trained[1][2] != w.trained[2][2] == w.trained[3][2]   # one target a generation
    assert w.trained[0][2].endswith("swa_1000.pt")


def test_10_each_lineage_continues_from_its_own_previous_generation(tmp_path) -> None:
    lg, w = _league(tmp_path, {}, gens=2)
    lg.run()
    us1, ussr1, us2, ussr2 = w.trained
    assert us1[1] == "/r108/resume_15610019840steps.pt" and ussr1[1] == "/r109/resume_15610019840steps.pt"
    assert us2[1].endswith(f"resume_{us1[4]}steps.pt") and os.path.dirname(us2[1]).endswith(us1[3] + "_20261112_000000")
    assert ussr2[1].endswith(f"resume_{ussr1[4]}steps.pt")
    assert us2[4] == us1[4] + 400_000_000                                  # 400M more on the lineage's own counter


def test_10_every_generation_is_published_with_its_margin(tmp_path) -> None:
    lg, w = _league(tmp_path, {("us", 1): 0.055, ("ussr", 1): -0.012}, gens=1)
    lg.run()
    pub = [e for e in _events(lg) if e["event"] == "published"]
    assert [(e["seat"], round(e["margin"], 3)) for e in pub] == [("us", 0.055), ("ussr", -0.012)]
    links = sorted(f for f in os.listdir(lg.league_dir) if LINK_RE.match(f))
    assert len(links) == 2
    assert all(os.path.islink(os.path.join(lg.league_dir, f)) for f in links)


def test_10_a_lineage_negative_twice_restarts_from_the_main(tmp_path) -> None:
    lg, w = _league(tmp_path, {("ussr", 1): -0.01, ("ussr", 2): -0.02}, gens=3)
    lg.run()
    ussr = [t for t in w.trained if t[0] == "ussr"]
    assert os.path.dirname(ussr[2][1]) == w.main_dir                      # the third generation starts from the main
    us = [t for t in w.trained if t[0] == "us"]
    assert os.path.dirname(us[2][1]) != w.main_dir                         # the other lineage continues
    assert any(e["event"] == "lineage_restart" and e["seat"] == "ussr" for e in _events(lg))


def test_10_the_league_keeps_capacity_members_a_seat(tmp_path) -> None:
    lg, w = _league(tmp_path, {}, gens=3, capacity=2)
    lg.run()
    links = [LINK_RE.match(f) for f in os.listdir(lg.league_dir)]
    by_seat: Dict[str, List[int]] = {"us": [], "ussr": []}
    for m in links:
        if m:
            by_seat[m.group(1)].append(int(m.group(2)))
    assert sorted(by_seat["us"]) == [2, 3] and sorted(by_seat["ussr"]) == [2, 3]   # the oldest generation evicted
    assert sum(e["event"] == "evicted" for e in _events(lg)) == 2
