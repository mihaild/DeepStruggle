"""The Elo leaderboard: its records, their validation, and the rating fit.

Everything lives in `leaderboard/` at the repository root and is reviewed like code:

* `networks.json` -- the weights: file (relative to the shared `data/` tree), sha256, where it is
  published on Hugging Face, its behaviour report, a description and any older names.

A report, a network's or a player's, is always the behaviour report `tools/scripts/checkpoint_report.py`
generated for that network or player (`REPORT_DIR`: the openings, how games end, what each side does
with every card, ... and its searched-leak census first, when it has one) -- never a plan or a research
log that mentions it, nor the leak census written alone (`--games 0`), which has no self-play
(`report_problem`).
* `players.json` -- a network plus the way it is run (tools/lib/player_spec.py), keyed by the
  player's canonical id; a bot has no network.
* `epochs.json` -- one engine epoch per entry: the engine behaviour whose games it accepts (the
  digest of tools/lib/engine_behaviour.py: any build that reproduces it plays the epoch's games,
  whatever its sources), the main players, and the anchor that fixes the scale. `also_accepts` lists
  further digests the owner ruled play the same games -- a build whose digest moved only in what it
  records, not in what is played -- each with the reason; a record names the digest it reproduced.
* `matches/<epoch>.jsonl` -- one line per pairing played, append-only: both seats' wins, losses
  and draws, the first deal seed, the engine (its source fingerprint), its behaviour digest's id
  and the commit. Records from before epochs were defined by behaviour carry no behaviour; their
  engine must be one of the epoch's `pre_digest_engines`, a frozen list. `.gitattributes` merges these files
  by union, so two branches that both appended merge cleanly; a record's `id` is a hash of its
  content, so a line that arrives twice is caught here.

The fit is Bradley-Terry with a seat term: P(a beats b, a as US) = logistic((r_a - r_b + h) / s),
s = 400 / ln 10, a draw half a win. It runs in two stages so that the main players' ratings do not
move when anyone else plays:

1. the main players, on main-vs-main pairings only, with the anchor pinned and h estimated;
2. everyone else, on every pairing that involves a non-main player (non-main against non-main
   included), with the main players and h held at stage 1's values.

A view ("main only", "main + lineage X") is a filter over this one fit, never a refit, so a
number means the same in every view. A non-main player is rated only if pairings connect it to
the main players. Standard library only: CI fits the leaderboard without torch or the engine.
"""

from __future__ import annotations

import hashlib
import json
import math
import pathlib
import re
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Mapping, NotRequired, Optional, Sequence, Set, Tuple, TypedDict

from tools.lib.player_spec import (GREEDY_TEMPERATURE, Inference, PlayerSpecError, Policy, Rollout, kind_of,
                                   player_id, spec_from_dict, workbench_search)

ROOT = pathlib.Path(__file__).resolve().parents[2]
LEADERBOARD_DIR = ROOT / "leaderboard"
#: The architecture and recipe codes (`tools/scripts/run_codes.py`): each code's flag set and description.
RUN_CODES = ROOT / "research" / "run_codes.json"
HF_REPO = "mihaild/deepstruggle"
GITHUB_REPO = "mihaild/DeepStruggle"
#: Elo per natural-log unit of odds.
ELO_SCALE = 400.0 / math.log(10.0)
#: A weak Gaussian prior (standard deviation, Elo) on every fitted value, so that a player who
#: has won or lost every game still gets a finite rating. Against 2,000 games it moves a rating by
#: well under 0.1 Elo.
PRIOR_SD = 1000.0
_HEX64 = re.compile(r"^[0-9a-f]{64}$")
#: Where tools/scripts/checkpoint_report.py writes its behaviour reports, the only reports linked.
REPORT_DIR = "research/log/per_checkpoint"
_REPORT_SCRIPT = "tools/scripts/checkpoint_report.py"
#: Of tools/lib/engine_behaviour.py's digest: bumped whenever what it hashes changes, so a digest
#: taken one way is never compared with one taken another. 2: the rollout section plays a keyed
#: batch (2026-10-24); before, no section built one, so a change to how keyed chance draws moved
#: nothing. 3: that batch also plays Space Race box 4 headlines (2026-10-25), where its box4 acts;
#: under 2 it never met one. No epoch was taken under 2.
BEHAVIOUR_VERSION = 3
BEHAVIOUR_SECTIONS: Tuple[str, ...] = ("runner", "state", "rollout", "mcts")


class Network(TypedDict):
    file: str
    sha256: str
    description: str
    old_names: List[str]
    hf: Optional[str]
    report: Optional[str]


class Player(TypedDict):
    network: Optional[str]
    inference: Dict[str, object]
    description: str
    #: A report on this player's own play -- for a search, of the search, not of its network. A
    #: greedy policy that has none shows its network's report, which is of greedy play.
    report: Optional[str]


class Behaviour(TypedDict):
    """tools/lib/engine_behaviour.py's digest: a sha256 per section (`BEHAVIOUR_SECTIONS`)."""
    version: int
    runner: str
    state: str
    rollout: str
    mcts: str


class AcceptedBehaviour(TypedDict):
    """A further digest an epoch accepts, and why its games are the epoch's."""
    behaviour: Behaviour
    why: str


class Epoch(TypedDict):
    description: str
    behaviour: Behaviour
    #: Builds whose digest differs from `behaviour` only where the owner ruled no game differs.
    also_accepts: NotRequired[List[AcceptedBehaviour]]
    #: The source fingerprints of the builds the epoch's first records were played on, before
    #: epochs were defined by behaviour: a frozen list that only validates those records.
    pre_digest_engines: NotRequired[List[str]]
    anchor: str
    anchor_elo: float
    main: List[str]
    #: Main players a strong newcomer plays instead of the whole main set (`leaderboard_play --strong`):
    #: against a much weaker opponent a game carries less information and the fit's prediction misses
    #: (owner, 2026-10-16; research/log/P33_diagnostics.md).
    strong: NotRequired[List[str]]


class Seat(TypedDict):
    w: int
    l: int
    d: int


class Record(TypedDict):
    id: str
    a: str
    b: str
    a_as_us: Seat
    a_as_ussr: Seat
    base_seed: int
    engine: str
    #: `behaviour_id` of the epoch's behaviour, which the engine reproduced; absent from records
    #: played before epochs were defined by behaviour.
    behaviour: NotRequired[str]
    commit: str
    date: str


class LeaderboardError(ValueError):
    pass


@dataclass
class Board:
    networks: Dict[str, Network]
    players: Dict[str, Player]
    epochs: Dict[str, Epoch]
    matches: Dict[str, List[Record]]
    root: pathlib.Path = LEADERBOARD_DIR


def lineage(network: Optional[str]) -> Optional[str]:
    """The run a network descends from: its name up to the first `@` (`E7-A8-R1-S44@6400M+...`
    is lineage `E7-A8-R1-S44`)."""
    return network.split("@", 1)[0] if network else None


#: A code in a network's name: `A8`, `R34` (not the engine `E7`, a seed `S44` or a step `@6400M`).
_CODE = re.compile(r"(?<![A-Za-z0-9])([AR])(\d+)(?![0-9])")


def network_codes(network: str) -> Tuple[Optional[str], Optional[str]]:
    """The architecture and recipe a network was trained under: the last A code and the last R code in
    its name, since each branch point names what changed (`E7-A8-R1-S44@6400M+R34@7200M+R48@9170M+A13`
    is A13 under R48; a soup of `(R34,R34-S45,R34-S46)` is R34). Names in the old scheme carry none."""
    last: Dict[str, Optional[str]] = {"A": None, "R": None}
    for m in _CODE.finditer(network):
        last[m.group(1)] = m.group(1) + m.group(2)
    return last["A"], last["R"]


def codes_data(board: Board, path: pathlib.Path = RUN_CODES) -> Dict[str, object]:
    """The page's tab of codes: every architecture and recipe with its description and flags, the
    networks on the board trained under it, and each network's pair -- read from run_codes.json, the
    one place the codes are defined, so the tab cannot drift from them."""
    codes = json.loads(path.read_text(encoding="utf-8"))
    by_network: Dict[str, Dict[str, Optional[str]]] = {}
    users: Dict[str, Set[str]] = {}
    for net in board.networks:
        a, r = network_codes(net)
        by_network[net] = {"architecture": a, "recipe": r}
        for c in (a, r):
            if c:
                users.setdefault(c, set()).add(net)
    out: Dict[str, object] = {"by_network": by_network}
    for kind in ("architectures", "recipes"):
        out[kind] = {code: {"description": str(v.get("description", "")), "flags": v.get("flags", {}),
                            "first": v.get("first"), "networks": sorted(users.get(code, set()))}
                     for code, v in sorted(codes.get(kind, {}).items(), key=lambda kv: int(kv[0][1:]))}
    return out


def player_spec(board: Board, pid: str) -> Inference:
    return spec_from_dict(board.players[pid]["inference"])


def player_lineage(board: Board, pid: str) -> Optional[str]:
    return lineage(board.players[pid]["network"])


# ---- reading and writing ----------------------------------------------------------------------


def _read_json(path: pathlib.Path) -> object:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise LeaderboardError(f"missing {path}") from None
    except json.JSONDecodeError as e:
        raise LeaderboardError(f"{path}: not JSON ({e})") from None


def load(root: pathlib.Path = LEADERBOARD_DIR) -> Board:
    networks = _read_json(root / "networks.json")
    players = _read_json(root / "players.json")
    epochs = _read_json(root / "epochs.json")
    if not (isinstance(networks, dict) and isinstance(players, dict) and isinstance(epochs, dict)):
        raise LeaderboardError("networks.json, players.json and epochs.json must each hold an object")
    # Every match file, not only those of known epochs: a file whose epoch is missing from
    # epochs.json is reported by `validate`, which it could not be if it were never read.
    matches: Dict[str, List[Record]] = {name: [] for name in epochs}
    for path in sorted((root / "matches").glob("*.jsonl")):
        rows: List[Record] = []
        for n, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if line.strip():
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError as e:
                    raise LeaderboardError(f"{path}:{n}: not JSON ({e})") from None
        matches[path.stem] = rows
    return Board(networks, players, epochs, matches, root)  # type: ignore[arg-type]


def _dump(obj: object) -> str:
    return json.dumps(obj, indent=2, ensure_ascii=False) + "\n"


def save_registry(board: Board) -> None:
    """Write networks.json, players.json and epochs.json, keys sorted so diffs stay small."""
    (board.root / "networks.json").write_text(_dump(dict(sorted(board.networks.items()))), encoding="utf-8")
    (board.root / "players.json").write_text(_dump(dict(sorted(board.players.items()))), encoding="utf-8")
    (board.root / "epochs.json").write_text(_dump(board.epochs), encoding="utf-8")


def record_id(rec: Mapping[str, object]) -> str:
    """A hash of everything in the record but its id."""
    body = {k: v for k, v in rec.items() if k != "id"}
    return hashlib.sha256(json.dumps(body, sort_keys=True).encode()).hexdigest()[:16]


def behaviour_id(b: Behaviour) -> str:
    """One hash of a behaviour digest's sections and version, for a record to name."""
    return hashlib.sha256(json.dumps(dict(b), sort_keys=True).encode()).hexdigest()


def epoch_behaviours(ep: Epoch) -> List[Behaviour]:
    """Every digest whose games the epoch accepts: its own first, then `also_accepts` in order."""
    return [ep["behaviour"], *(a["behaviour"] for a in ep.get("also_accepts", []))]


def _behaviour_ok(beh: object) -> bool:
    fields: Mapping[str, object] = beh if isinstance(beh, dict) else {}
    return (set(fields) == {"version", *BEHAVIOUR_SECTIONS} and isinstance(fields["version"], int)
            and all(isinstance(v, str) and bool(_HEX64.match(v)) for v in (fields[s] for s in BEHAVIOUR_SECTIONS)))


def behaviour_mismatch(want: Behaviour, have: Behaviour) -> List[str]:
    """The sections in which `have` differs from `want`; every section when the versions differ."""
    if want["version"] != have["version"]:
        return list(BEHAVIOUR_SECTIONS)
    w: Mapping[str, object] = want
    h: Mapping[str, object] = have
    return [s for s in BEHAVIOUR_SECTIONS if w[s] != h[s]]


def make_record(a: str, b: str, a_as_us: Seat, a_as_ussr: Seat, base_seed: int, engine: str,
                behaviour: str, commit: str, date: str) -> Record:
    body = {"a": a, "b": b, "a_as_us": dict(a_as_us), "a_as_ussr": dict(a_as_ussr),
            "base_seed": int(base_seed), "engine": engine, "behaviour": behaviour, "commit": commit,
            "date": date}
    return {"id": record_id(body), **body}  # type: ignore[return-value]


def append_record(board: Board, epoch: str, rec: Record) -> None:
    """Append one pairing to the epoch's match file (and to `board`), refusing a repeat."""
    key = _pair_key(rec)
    for old in board.matches.setdefault(epoch, []):
        if old["id"] == rec["id"] or _pair_key(old) == key:
            raise LeaderboardError(f"{rec['a']} vs {rec['b']} at base seed {rec['base_seed']} is "
                                   f"already recorded ({old['id']})")
    path = board.root / "matches" / f"{epoch}.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    board.matches[epoch].append(rec)


def _pair_key(rec: Record) -> Tuple[str, str, int]:
    """A pairing at a base seed plays the same deals whichever side is `a`, so this is what must
    be unique: a second record of it would count the same games twice."""
    x, y = sorted((rec["a"], rec["b"]))
    return x, y, int(rec["base_seed"])


# ---- validation -------------------------------------------------------------------------------


def _seat_ok(s: object) -> bool:
    return (isinstance(s, dict) and set(s) == {"w", "l", "d"}
            and all(isinstance(s[k], int) and not isinstance(s[k], bool) and s[k] >= 0 for k in s))


def report_problem(repo_root: pathlib.Path, report: str, subjects: Sequence[str]) -> Optional[str]:
    """Why `report` is not a behaviour report checkpoint_report.py generated for one of `subjects`
    (a network's names, or a player's ids), or None when it is: a file in REPORT_DIR titled
    `# <subject>: behaviour report`, saying it was generated by the script, and played from self-play
    (its `* **Games:**` line) -- the searched-leak census written alone has none."""
    if not report.startswith(REPORT_DIR + "/"):
        return (f"report {report} is not a behaviour report from {_REPORT_SCRIPT} (in {REPORT_DIR}/); "
                f"generate one, or leave it null")
    path = repo_root / report
    if not path.is_file():
        return f"report {report} does not exist"
    lines = path.read_text(encoding="utf-8").splitlines()
    head = lines[:5]
    if not head or head[0] not in {f"# {s}: behaviour report" for s in subjects}:
        return f"report {report} is titled {head[0] if head else '(empty)'!r}, not a report on {subjects[0]}"
    if not any(f"by `{_REPORT_SCRIPT}`" in line for line in head):
        return f"report {report} does not say it was generated by {_REPORT_SCRIPT}"
    if not any(line.startswith("* **Games:** ") for line in lines):
        return (f"report {report} has no self-play (the searched-leak census alone, `--games 0`); "
                f"play the behaviour report, or leave it null")
    return None


def validate(board: Board, repo_root: pathlib.Path = ROOT) -> List[str]:
    """Every problem in the records, as messages; empty when they are consistent."""
    errs: List[str] = []
    for name, net in board.networks.items():
        where = f"networks.json[{name}]"
        if "~" in name or ":" in name or "@" not in name:
            errs.append(f"{where}: a network name has an '@' and no '~' or ':'")
        want = {"file", "sha256", "description", "old_names", "hf", "report"}
        if set(net) != want:
            errs.append(f"{where}: fields must be {sorted(want)}, got {sorted(net)}")
            continue
        if not isinstance(net["sha256"], str) or not _HEX64.match(net["sha256"]):
            errs.append(f"{where}: sha256 must be 64 lowercase hex digits")
        if not isinstance(net["old_names"], list) or not all(isinstance(x, str) for x in net["old_names"]):
            errs.append(f"{where}: old_names must be a list of strings")
        if isinstance(net["file"], str) and net["file"].rsplit("/", 1)[-1] == "snapshot_final.pt":
            errs.append(f"{where}: register the snapshot by step (snapshot_<N>steps.pt), not "
                        f"snapshot_final.pt, a copy that names no step")
        if net["report"] is not None:
            bad = report_problem(repo_root, net["report"], [name, *net["old_names"]])
            if bad:
                errs.append(f"{where}: {bad}")
        if net["hf"] is not None and (not isinstance(net["hf"], str) or net["hf"].startswith("/")):
            errs.append(f"{where}: hf must be a path inside {HF_REPO}, or null")
    for pid, pl in board.players.items():
        where = f"players.json[{pid}]"
        if set(pl) != {"network", "inference", "description", "report"}:
            errs.append(f"{where}: fields must be ['description', 'inference', 'network', 'report'], got {sorted(pl)}")
            continue
        if pl["report"] is not None:
            network = pl["network"]
            olds = board.networks[network]["old_names"] if network is not None and network in board.networks else []
            # A report written under an earlier name of the network names the player that way.
            ids = [pid] + [old + pid[len(network):] for old in olds if network is not None]
            bad = report_problem(repo_root, pl["report"], ids)
            if bad:
                errs.append(f"{where}: {bad}")
        try:
            spec = spec_from_dict(pl["inference"])
            canon = player_id(pl["network"], spec)
        except PlayerSpecError as e:
            errs.append(f"{where}: {e}")
            continue
        if canon != pid:
            errs.append(f"{where}: the canonical id of this network and inference is {canon!r}")
        if pl["network"] is not None and pl["network"] not in board.networks:
            errs.append(f"{where}: network {pl['network']!r} is not in networks.json")
        leaf_net = spec.leaf_net if isinstance(spec, Rollout) else ""
        if leaf_net and leaf_net not in board.networks:
            errs.append(f"{where}: leaf_net {leaf_net!r} is not in networks.json")
    for name, ep in board.epochs.items():
        where = f"epochs.json[{name}]"
        want = {"description", "behaviour", "anchor", "anchor_elo", "main"}
        if not want <= set(ep) <= want | {"strong", "pre_digest_engines", "also_accepts"}:
            errs.append(f"{where}: fields must be {sorted(want)} (and optionally strong, "
                        f"pre_digest_engines, also_accepts), got {sorted(ep)}")
            continue
        strong = ep.get("strong", [])
        if any(pid not in ep["main"] for pid in strong) or len(set(strong)) != len(strong):
            errs.append(f"{where}: the strong tier must be distinct main players")
        beh = ep["behaviour"]
        if not _behaviour_ok(beh):
            errs.append(f"{where}: behaviour must be {{version, {', '.join(BEHAVIOUR_SECTIONS)}}}, "
                        f"each section a sha256 (tools/lib/engine_behaviour.py)")
            continue
        accepted = ep.get("also_accepts", [])
        if not isinstance(accepted, list) or not all(
                isinstance(a, dict) and set(a) == {"behaviour", "why"} and _behaviour_ok(a["behaviour"])
                and isinstance(a["why"], str) and a["why"].strip() for a in accepted):
            errs.append(f"{where}: also_accepts must be a list of {{behaviour, why}}, each behaviour a "
                        f"digest like `behaviour` and each why a non-empty reason")
            continue
        accepted_ids = {behaviour_id(b) for b in epoch_behaviours(ep)}
        if len(accepted_ids) != 1 + len(accepted):
            errs.append(f"{where}: also_accepts repeats a digest")
        legacy = ep.get("pre_digest_engines", [])
        if not all(isinstance(e, str) and _HEX64.match(e) for e in legacy):
            errs.append(f"{where}: pre_digest_engines must be engine fingerprints")
        for pid in ep["main"]:
            if pid not in board.players:
                errs.append(f"{where}: main player {pid!r} is not in players.json")
        if len(set(ep["main"])) != len(ep["main"]):
            errs.append(f"{where}: a main player is listed twice")
        if ep["anchor"] not in ep["main"]:
            errs.append(f"{where}: the anchor {ep['anchor']!r} must be a main player")
        seen_ids: Set[str] = set()
        seen_pairs: Dict[Tuple[str, str, int], str] = {}
        for n, rec in enumerate(board.matches.get(name, []), 1):
            at = f"matches/{name}.jsonl:{n}"
            want_r = {"id", "a", "b", "a_as_us", "a_as_ussr", "base_seed", "engine", "commit", "date"}
            if not isinstance(rec, dict) or not want_r <= set(rec) <= want_r | {"behaviour"}:
                errs.append(f"{at}: fields must be {sorted(want_r)} and behaviour")
                continue
            typed = [f for f, ok in (
                ("id, a, b, commit and date must be strings",
                 all(isinstance(rec[k], str) for k in ("id", "a", "b", "commit", "date"))),
                ("base_seed must be an integer",
                 isinstance(rec["base_seed"], int) and not isinstance(rec["base_seed"], bool)),
                ("engine must be an engine fingerprint (64 lowercase hex digits)",
                 isinstance(rec["engine"], str) and bool(_HEX64.match(rec["engine"]))),
                ("behaviour must be a string", isinstance(rec.get("behaviour", ""), str)),
            ) if not ok]
            if typed:
                # The checks below index and hash these fields; report, do not crash on them.
                errs.extend(f"{at}: {f}" for f in typed)
                continue
            if rec["a"] == rec["b"]:
                errs.append(f"{at}: a player against itself")
            for side in ("a", "b"):
                if rec[side] not in board.players:
                    errs.append(f"{at}: player {rec[side]!r} is not in players.json")
            if not (_seat_ok(rec["a_as_us"]) and _seat_ok(rec["a_as_ussr"])):
                errs.append(f"{at}: a_as_us / a_as_ussr must be {{w, l, d}} of non-negative integers")
            elif sum(int(s[k]) for s in (rec["a_as_us"], rec["a_as_ussr"]) for k in ("w", "l", "d")) == 0:
                errs.append(f"{at}: no games")
            if "behaviour" in rec:
                if rec["behaviour"] not in accepted_ids:
                    errs.append(f"{at}: played on an engine of behaviour {str(rec['behaviour'])[:12]}…, "
                                f"not one {name} accepts")
            elif rec["engine"] not in legacy:
                errs.append(f"{at}: a record with no behaviour, on engine {rec['engine'][:12]}…, which "
                            f"is not one of {name}'s pre_digest_engines")
            if rec["id"] != record_id(rec):
                errs.append(f"{at}: id {rec['id']} does not match its content ({record_id(rec)}); "
                            f"records are never edited")
            if rec["id"] in seen_ids:
                errs.append(f"{at}: record {rec['id']} appears twice")
            seen_ids.add(rec["id"])
            key = _pair_key(rec)
            if key in seen_pairs and seen_pairs[key] != rec["id"]:
                errs.append(f"{at}: {key[0]} vs {key[1]} at base seed {key[2]} is already recorded "
                            f"({seen_pairs[key]}) -- the same deals, counted twice")
            seen_pairs.setdefault(key, rec["id"])
    for name in board.matches:
        if name not in board.epochs:
            errs.append(f"matches/{name}.jsonl: no epoch {name!r} in epochs.json")
    return errs


# ---- the fit ----------------------------------------------------------------------------------


@dataclass
class Rating:
    player: str
    elo: float
    se: float
    main: bool
    games: int
    score: float
    opponents: int
    main_opponents: int


@dataclass
class EpochFit:
    epoch: str
    anchor: str
    anchor_elo: float
    seat_us: float
    seat_us_se: float
    ratings: List[Rating]
    #: Players with games that no pairing connects to the main players.
    unrated: List[str] = field(default_factory=list)


@dataclass(frozen=True)
class _Binomial:
    """One pairing's games from one seat."""
    a: str
    b: str
    #: +1 when a played the US, -1 when a played the USSR.
    sign: int
    #: a's score, draws half a win.
    score: float
    games: int


@dataclass(frozen=True)
class _Fit:
    """`_newton`'s optimum, in Elo: the free ratings, the seat term, and their standard errors."""
    ratings: Dict[str, float]
    seat: float
    se: Dict[str, float]
    seat_se: float


def _observations(records: Iterable[Record]) -> List[_Binomial]:
    out: List[_Binomial] = []
    for r in records:
        for sign, s in ((1, r["a_as_us"]), (-1, r["a_as_ussr"])):
            n = s["w"] + s["l"] + s["d"]
            if n:
                out.append(_Binomial(r["a"], r["b"], sign, s["w"] + 0.5 * s["d"], n))
    return out


def _cholesky_solve_and_inverse_diag(m: List[List[float]], g: List[float]) -> Tuple[List[float], List[float]]:
    """x with m x = g, and diag(m^-1), for a symmetric positive definite m."""
    n = len(m)
    low = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1):
            acc = m[i][j] - sum(low[i][k] * low[j][k] for k in range(j))
            if i == j:
                if acc <= 0:
                    raise LeaderboardError("the fit's information matrix is not positive definite")
                low[i][i] = math.sqrt(acc)
            else:
                low[i][j] = acc / low[j][j]

    def solve(b: List[float]) -> List[float]:
        y = [0.0] * n
        for i in range(n):
            y[i] = (b[i] - sum(low[i][k] * y[k] for k in range(i))) / low[i][i]
        x = [0.0] * n
        for i in reversed(range(n)):
            x[i] = (y[i] - sum(low[k][i] * x[k] for k in range(i + 1, n))) / low[i][i]
        return x

    x = solve(g)
    diag = []
    for i in range(n):
        e = [0.0] * n
        e[i] = 1.0
        diag.append(solve(e)[i])
    return x, diag


def _newton(free: Sequence[str], fixed: Mapping[str, float], obs: Sequence[_Binomial], fit_seat: bool,
            seat: float, prior_mean: float) -> _Fit:
    """Maximise the binomial log-likelihood plus the weak prior over the `free` ratings (and the
    seat term when `fit_seat`), the rest held at `fixed` / `seat`."""
    idx = {p: i for i, p in enumerate(free)}
    n_var = len(free) + (1 if fit_seat else 0)
    h_i = len(free)
    v = [prior_mean] * len(free) + ([0.0] if fit_seat else [])
    mu = [prior_mean] * len(free) + ([0.0] if fit_seat else [])
    inv_var = 1.0 / PRIOR_SD ** 2

    def rating(p: str, vec: List[float]) -> float:
        return vec[idx[p]] if p in idx else fixed[p]

    def seat_of(vec: List[float]) -> float:
        return vec[h_i] if fit_seat else seat

    def loglik(vec: List[float]) -> float:
        ll = -0.5 * inv_var * sum((x - m) ** 2 for x, m in zip(vec, mu))
        h = seat_of(vec)
        for o in obs:
            a, b, sign, s, n = o.a, o.b, o.sign, o.score, o.games
            x = (rating(a, vec) - rating(b, vec) + sign * h) / ELO_SCALE
            # log p = -log(1 + e^-x), log(1 - p) = -log(1 + e^x), computed stably
            ll += -s * _softplus(-x) - (n - s) * _softplus(x)
        return ll

    if n_var == 0:
        return _Fit({}, seat, {}, 0.0)
    ll = loglik(v)
    for _ in range(200):
        grad = [-(x - m) * inv_var for x, m in zip(v, mu)]
        hess = [[inv_var if i == j else 0.0 for j in range(n_var)] for i in range(n_var)]
        h = seat_of(v)
        for o in obs:
            a, b, sign, s, n = o.a, o.b, o.sign, o.score, o.games
            x = (rating(a, v) - rating(b, v) + sign * h) / ELO_SCALE
            p = 1.0 / (1.0 + math.exp(-x))
            d: List[Tuple[int, float]] = []
            if a in idx:
                d.append((idx[a], 1.0 / ELO_SCALE))
            if b in idx:
                d.append((idx[b], -1.0 / ELO_SCALE))
            if fit_seat:
                d.append((h_i, sign / ELO_SCALE))
            w = n * p * (1.0 - p)
            for i, di in d:
                grad[i] += (s - n * p) * di
                for j, dj in d:
                    hess[i][j] += w * di * dj
        step, _ = _cholesky_solve_and_inverse_diag(hess, grad)
        t = 1.0
        while True:
            cand = [x + t * dx for x, dx in zip(v, step)]
            cand_ll = loglik(cand)
            if cand_ll >= ll - 1e-12 or t < 1e-6:
                break
            t *= 0.5
        v, ll = cand, cand_ll
        if max(abs(t * dx) for dx in step) < 1e-7:
            break
    else:
        raise LeaderboardError("the rating fit did not converge in 200 Newton steps")
    # Standard errors from the curvature at the optimum.
    hess = [[inv_var if i == j else 0.0 for j in range(n_var)] for i in range(n_var)]
    h = seat_of(v)
    for o in obs:
        a, b, sign, n = o.a, o.b, o.sign, o.games
        x = (rating(a, v) - rating(b, v) + sign * h) / ELO_SCALE
        p = 1.0 / (1.0 + math.exp(-x))
        d = [(idx[a], 1.0 / ELO_SCALE)] if a in idx else []
        if b in idx:
            d.append((idx[b], -1.0 / ELO_SCALE))
        if fit_seat:
            d.append((h_i, sign / ELO_SCALE))
        for i, di in d:
            for j, dj in d:
                hess[i][j] += n * p * (1.0 - p) * di * dj
    _, var = _cholesky_solve_and_inverse_diag(hess, [0.0] * n_var)
    ratings = {p: v[i] for p, i in idx.items()}
    ses = {p: math.sqrt(var[i]) for p, i in idx.items()}
    return _Fit(ratings, seat_of(v), ses, math.sqrt(var[h_i]) if fit_seat else 0.0)


def _softplus(x: float) -> float:
    return x + math.log1p(math.exp(-x)) if x > 0 else math.log1p(math.exp(x))


def _components(nodes: Iterable[str], edges: Iterable[Tuple[str, str]]) -> Dict[str, str]:
    parent = {n: n for n in nodes}

    def find(x: str) -> str:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for a, b in edges:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb
    return {n: find(n) for n in parent}


def fit_epoch(board: Board, epoch: str) -> EpochFit:
    ep = board.epochs[epoch]
    records = board.matches.get(epoch, [])
    mains = list(ep["main"])
    main_set = set(mains)
    anchor, anchor_elo = ep["anchor"], float(ep["anchor_elo"])

    # Stage 1: the main players among themselves.
    main_recs = [r for r in records if r["a"] in main_set and r["b"] in main_set]
    comp = _components(mains, ((r["a"], r["b"]) for r in main_recs))
    cut_off = sorted(p for p in mains if comp[p] != comp[anchor])
    if cut_off:
        raise LeaderboardError(f"{epoch}: no main-vs-main pairings connect these main players to the "
                               f"anchor {anchor}: {', '.join(cut_off)}")
    free1 = [p for p in mains if p != anchor]
    fit1 = _newton(free1, {anchor: anchor_elo}, _observations(main_recs),
                   fit_seat=bool(main_recs), seat=0.0, prior_mean=anchor_elo)
    seat, seat_se = fit1.seat, fit1.seat_se
    elo: Dict[str, float] = {anchor: anchor_elo, **fit1.ratings}
    se: Dict[str, float] = {anchor: 0.0, **fit1.se}

    # Stage 2: everyone else, against the main players held fixed.
    other_recs = [r for r in records if not (r["a"] in main_set and r["b"] in main_set)]
    others = sorted({p for r in other_recs for p in (r["a"], r["b"])} - main_set)
    hub = "\0main"
    comp2 = _components([hub, *others],
                        ((hub if r["a"] in main_set else r["a"], hub if r["b"] in main_set else r["b"])
                         for r in other_recs))
    rated = [p for p in others if comp2[p] == comp2[hub]]
    unrated = [p for p in others if comp2[p] != comp2[hub]]
    rated_set = set(rated) | main_set
    obs2 = _observations(r for r in other_recs if r["a"] in rated_set and r["b"] in rated_set)
    fit2 = _newton(rated, {p: elo[p] for p in mains}, obs2, fit_seat=False, seat=seat, prior_mean=anchor_elo)
    elo.update(fit2.ratings)
    se.update(fit2.se)

    # Per-player tallies over every record of the epoch.
    games: Dict[str, int] = {}
    score: Dict[str, float] = {}
    opps: Dict[str, Set[str]] = {}
    for r in records:
        for sign, s in ((1, r["a_as_us"]), (-1, r["a_as_ussr"])):
            n, pts = s["w"] + s["l"] + s["d"], s["w"] + 0.5 * s["d"]
            for p, q, got in ((r["a"], r["b"], pts), (r["b"], r["a"], n - pts)):
                games[p] = games.get(p, 0) + n
                score[p] = score.get(p, 0.0) + got
                opps.setdefault(p, set()).add(q)
    ratings = [Rating(player=p, elo=elo[p], se=se[p], main=p in main_set, games=games.get(p, 0),
                      score=score.get(p, 0.0), opponents=len(opps.get(p, ())),
                      main_opponents=len(opps.get(p, set()) & main_set))
               for p in elo]
    ratings.sort(key=lambda x: (-x.elo, x.player))
    return EpochFit(epoch, anchor, anchor_elo, seat, seat_se, ratings, unrated)


def view(board: Board, fit: EpochFit, *, main_only: bool = False,
         lineages: Sequence[str] = ()) -> List[Rating]:
    """The ratings a view shows: the main players, plus every player of the given lineages
    (all players when neither filter is given). A filter, never a refit."""
    if not main_only and not lineages:
        return list(fit.ratings)
    want = set(lineages)
    return [r for r in fit.ratings if r.main or player_lineage(board, r.player) in want]


# ---- the page's data --------------------------------------------------------------------------


def player_report(board: Board, pid: str) -> Optional[str]:
    """The report the page links for a player: its own; else, for the greedy policy only, its
    network's -- a behaviour report of the network's greedy play, which says nothing about how the
    network plays under a search or sampled, so those players have none until one is written."""
    pl = board.players[pid]
    if pl["report"] is not None:
        return pl["report"]
    spec = spec_from_dict(pl["inference"])
    if pl["network"] is None or not (isinstance(spec, Policy) and spec.temperature <= GREEDY_TEMPERATURE):
        return None
    return board.networks[pl["network"]]["report"]


def site_data(board: Board, commit: Optional[str] = None, playdek: Optional[object] = None) -> Dict[str, object]:
    """Everything the leaderboard page shows, fitted: `leaderboard.json`, built by CI from the
    records rather than committed. `playdek` is tools/playdek_estimate.py's output, when the build
    could read the stats database; without it the page has only the anchored scale."""
    epochs: Dict[str, object] = {}
    for name, ep in board.epochs.items():
        fit = fit_epoch(board, name)
        pairs: Dict[Tuple[str, str], Dict[str, object]] = {}
        for r in board.matches.get(name, []):
            a, b = r["a"], r["b"]
            flip = a > b
            key = (b, a) if flip else (a, b)
            us, ussr = (r["a_as_ussr"], r["a_as_us"]) if flip else (r["a_as_us"], r["a_as_ussr"])
            # From the first player's side: their record as US and as USSR. Flipping a record swaps
            # who `a` is, so a's games as USSR become the new a's games as US, with w and l swapped.
            us_t = {"w": us["l"], "l": us["w"], "d": us["d"]} if flip else dict(us)
            ussr_t = {"w": ussr["l"], "l": ussr["w"], "d": ussr["d"]} if flip else dict(ussr)
            acc = pairs.setdefault(key, {"a": key[0], "b": key[1],
                                         "a_as_us": {"w": 0, "l": 0, "d": 0},
                                         "a_as_ussr": {"w": 0, "l": 0, "d": 0}, "records": []})
            for tgt, src in (("a_as_us", us_t), ("a_as_ussr", ussr_t)):
                for k in ("w", "l", "d"):
                    acc[tgt][k] += src[k]  # type: ignore[index]
            acc["records"].append(r["id"])  # type: ignore[union-attr]
        epochs[name] = {
            "description": ep["description"], "behaviour": behaviour_id(ep["behaviour"]), "anchor": ep["anchor"],
            "anchor_elo": ep["anchor_elo"], "main": ep["main"], "seat_us": round(fit.seat_us, 2),
            "seat_us_se": round(fit.seat_us_se, 2), "unrated": fit.unrated,
            "ratings": [{"player": r.player, "elo": round(r.elo, 1), "se": round(r.se, 1),
                         "main": r.main, "games": r.games, "score": r.score,
                         "opponents": r.opponents, "main_opponents": r.main_opponents}
                        for r in fit.ratings],
            "pairs": list(pairs.values()),
        }
    players = {}
    for pid, pl in board.players.items():
        spec = spec_from_dict(pl["inference"])
        players[pid] = {"network": pl["network"], "lineage": lineage(pl["network"]),
                        "kind": kind_of(spec), "inference": pl["inference"],
                        "description": pl["description"], "report": player_report(board, pid),
                        # how the workbench plays it: its `play` link's search=, null if it cannot
                        "workbench_search": workbench_search(spec)}
    data: Dict[str, object] = {"schema": 1, "commit": commit, "hf_repo": HF_REPO, "github": GITHUB_REPO,
                               "networks": board.networks, "players": players, "epochs": epochs,
                               "codes": codes_data(board)}
    if playdek is not None:
        data["playdek"] = playdek
    return data


def markdown_table(board: Board, ratings: Sequence[Rating]) -> str:
    rows = ["| # | player | Elo | ± | games | score | main opps | |", "|---:|:---|---:|---:|---:|---:|---:|:---|"]
    for k, r in enumerate(ratings, 1):
        pct = 100.0 * r.score / r.games if r.games else 0.0
        rows.append(f"| {k} | `{r.player}` | {r.elo:.0f} | {r.se:.0f} | {r.games:,} | {pct:.1f}% | "
                    f"{r.main_opponents} | {'main' if r.main else ''} |")
    return "\n".join(rows)
