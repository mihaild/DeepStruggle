"""Games people played against the models in the web workbench, from the stats database.

The page records a game (web/ui/src/game/record.ts) as its seed and every step -- MicroAction,
forced die, who made it, the settings in force -- and on request submits it, with the player's
Playdek login and rating, to a Cloudflare Worker that keeps it in D1 (web/stats_worker/). This
module reads an export of that table and

* decodes the compact log and summarizes how each game was played -- the same rules as the
  page's `summarize()`, recomputed here rather than trusted from the row;
* re-drives a game in this build's engine and checks it lands on the submitted final position
  (`final_hash`), writing the full .tslog.json on request;
* fits each model player's rating on Playdek's scale from the clean games (`fit_rating`).

The rating model is Elo's logistic on a 400-point scale -- the scale Playdek's rating is reported
to follow, not one it documents -- with the human's self-reported rating as the opponent's:

    P(human scores) = 1 / (1 + 10 ** (-(R_human + d * s - R_model) / 400))

where s is +1 when the human plays the USSR and -1 for the US, and d the USSR's advantage in
rating points (a parameter, not fitted: a handful of games cannot separate it from R_model). A
weak normal prior keeps a perfect score finite; the interval is a bootstrap over *players*, so
one person's many games count as one person's evidence.
"""
from __future__ import annotations

import base64
import hashlib
import json
import math
import struct
import zlib
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Literal, Optional, Sequence, Tuple, TypedDict, cast

import numpy as np

import ts_engine as ts
from tools.lib.game_step import drain_chance, step_checked

LOG_VERSION = 1
_STEP_BYTES = 7
_SOURCES = ("human", "suggested", "auto")
_MAX_LOG_BYTES = 256 * 1024

Side = Literal["US", "USSR"]


class SettingsDict(TypedDict):
    view: str
    auto: str
    model: str
    search: str


class ModelInfoDict(TypedDict):
    label: str
    source: str
    checkpoint: str


class EventDict(TypedDict):
    step: int
    kind: str
    text: str


class ResultDict(TypedDict):
    winner: str
    margin: int
    end_turn: int
    reason: str


class BuildDict(TypedDict):
    commit: str
    dirty: bool
    engine_fingerprint: str


class PlaydekDict(TypedDict):
    login: str
    rating: int


class SummaryDict(TypedDict):
    model_side: Optional[str]
    human_side: Optional[str]
    human_moves_for_model: int
    suggested_moves: int
    analysis_moves: int
    dev_view_moves: int
    forced_dice: int
    undos: int
    debug_overrides: int
    loaded_position: bool
    models: List[str]
    searches: List[str]
    clean: bool


class SubmissionDict(TypedDict):
    """One game as the page submits it (record.ts `Submission`)."""
    schema: int
    id: str
    started_at: str
    ended_at: str
    build: BuildDict
    playdek: PlaydekDict
    note: str
    result: ResultDict
    summary: SummaryDict
    settings: List[SettingsDict]
    models: Dict[str, ModelInfoDict]
    events: List[EventDict]
    start_position: Optional[str]
    log: str
    n_steps: int
    final_hash: str


@dataclass(frozen=True)
class Step:
    decision_type: int
    primary_id: int
    secondary_id: int
    flags: int
    forced_die: int
    source: str
    analysis: bool
    mover: str
    rolled: bool
    setting: int

    def micro_action(self) -> ts.MicroAction:
        return ts.MicroAction(ts.DecisionType(self.decision_type), self.primary_id, self.secondary_id, self.flags)


def _from_base64url(token: str) -> bytes:
    return base64.urlsafe_b64decode(token + "=" * (-len(token) % 4))


def decode_log(token: str) -> Tuple[int, List[Step]]:
    """The seed and steps of a game log (record.ts `encodeLog`). Raises ValueError on anything else."""
    d = zlib.decompressobj()
    raw = d.decompress(_from_base64url(token), _MAX_LOG_BYTES)
    if d.unconsumed_tail:
        raise ValueError("game log inflates past the size limit")
    if len(raw) < 9 or raw[0] != LOG_VERSION:
        raise ValueError(f"not a version-{LOG_VERSION} game log")
    seed, n = struct.unpack_from("<II", raw, 1)
    if len(raw) != 9 + _STEP_BYTES * n:
        raise ValueError(f"game log of {len(raw)} bytes for {n} steps")
    steps: List[Step] = []
    for i in range(n):
        dt, p, s, f, die, info, setting = raw[9 + _STEP_BYTES * i: 9 + _STEP_BYTES * (i + 1)]
        mover = {1: "US", 2: "USSR"}.get((info >> 3) & 3, "")
        steps.append(Step(dt, p, s, f, die, _SOURCES[info & 3] if (info & 3) < 3 else "human",
                          bool(info & 4), mover, bool(info & 32), setting))
    return seed, steps


def summarize(steps: Sequence[Step], settings: Sequence[SettingsDict], events: Sequence[EventDict],
              start_position: Optional[str], finished: bool) -> SummaryDict:
    """How a game was played: record.ts `summarize()`, rule for rule."""
    model_sides = set()
    models = set()
    searches = set()
    for st in steps:
        if st.source != "auto" or not st.mover:
            continue
        model_sides.add(st.mover)
        s = settings[st.setting] if st.setting < len(settings) else None
        models.add(s["model"] if s else "")
        searches.add(s["search"] if s else "")
    model_side: Optional[str] = (None if not model_sides else "mixed" if len(model_sides) > 1
                                 else next(iter(model_sides)))
    human_side: Optional[str] = {"US": "USSR", "USSR": "US"}.get(model_side or "")
    for_model = suggested = analysis = dev_view = forced = 0
    for st in steps:
        if st.forced_die > 0 and st.rolled:
            forced += 1
        if st.source == "auto":
            continue
        if model_side in ("US", "USSR") and st.mover == model_side:
            for_model += 1
            continue
        if st.source == "suggested":
            suggested += 1
        if st.analysis:
            analysis += 1
        view = settings[st.setting]["view"] if st.setting < len(settings) else "DEV"
        if view == "DEV":
            dev_view += 1
    undos = sum(1 for e in events if e["kind"] == "undo")
    debug = sum(1 for e in events if e["kind"] == "debug")
    loaded = start_position is not None
    clean = (finished and human_side is not None and for_model == 0 and suggested == 0 and analysis == 0
             and dev_view == 0 and forced == 0 and debug == 0 and not loaded
             and len(models) == 1 and "" not in models and len(searches) == 1)
    return {
        "model_side": model_side,
        "human_side": human_side,
        "human_moves_for_model": for_model,
        "suggested_moves": suggested,
        "analysis_moves": analysis,
        "dev_view_moves": dev_view,
        "forced_dice": forced,
        "undos": undos,
        "debug_overrides": debug,
        "loaded_position": loaded,
        "models": sorted(models),
        "searches": sorted(searches),
        "clean": clean,
    }


def summarize_submission(sub: SubmissionDict) -> SummaryDict:
    _, steps = decode_log(sub["log"])
    return summarize(steps, sub["settings"], sub["events"], sub["start_position"], finished=True)


def start_state(seed: int, start_position: Optional[str]) -> ts.GameState:
    """The position a game began from: the seed's deal, or the loaded position (a `pos=` token)."""
    if start_position is not None:
        state = ts.state_from_save_json(zlib.decompress(_from_base64url(start_position)).decode())
    else:
        state = ts.GameState()
        ts.Engine.init_game(state, seed)
    drain_chance(state)
    return state


def play_step(state: ts.GameState, st: Step) -> None:
    """One recorded step as the page made it: the MicroAction, then its dice (forced or rolled)."""
    step_checked(state, st.micro_action(), context="recorded step")
    drain_chance(state, forced_die=st.forced_die, context="recorded step's dice")


def final_hash(state: ts.GameState) -> str:
    return hashlib.sha256(state.to_save_json().encode()).hexdigest()


def outcome(state: ts.GameState) -> ResultDict:
    util = ts.Engine.get_terminal_utility(state)
    return {
        "winner": "US" if util > 0 else "USSR" if util < 0 else "DRAW",
        "margin": int(state.victory_points),
        "end_turn": int(state.turn),
        "reason": ts.game_ending_reason(state),
    }


@dataclass
class Verdict:
    """What re-driving a submitted game in this build found."""
    status: Literal["ok", "mismatch", "refused", "other-engine", "bad-log"]
    detail: str = ""


#: What decode_log raises on a log that is not one.
BAD_LOG = (ValueError, zlib.error, struct.error)


def verify(sub: SubmissionDict, engine_fingerprint: str, *, replay_other_engine: bool = False) -> Verdict:
    """Re-drive `sub` in this build and check it ends where the page said, as the page said.

    A game the record says was played on another engine is "other-engine" without being re-driven,
    unless `replay_other_engine`: then it is re-driven here all the same, and "ok" if it ends where
    the page said. The fingerprint is the submitter's own claim, so it cannot excuse a record from
    the check; a game that does not replay here stays "other-engine", since from this build an
    engine change and a forged record look alike.
    """
    where = (f"played on engine {sub['build']['engine_fingerprint'][:12]} "
             f"(commit {sub['build']['commit'][:12] or '?'}); this build is {engine_fingerprint[:12]}")
    if sub["build"]["engine_fingerprint"] == engine_fingerprint:
        return _replay(sub)
    if not replay_other_engine:
        return Verdict("other-engine", where)
    v = _replay(sub)
    if v.status == "ok":
        return Verdict("ok", f"{where}; replays here")
    return Verdict("other-engine", f"{where}; does not replay here ({v.status}: {v.detail})")


def _replay(sub: SubmissionDict) -> Verdict:
    """Re-drive `sub` in this build, whatever engine it names."""
    try:
        seed, steps = decode_log(sub["log"])
    except BAD_LOG as e:
        return Verdict("bad-log", str(e))
    try:
        state = start_state(seed, sub["start_position"])
        for st in steps:
            play_step(state, st)
    except Exception as e:  # the engine's refusal, whatever its type: it is the finding
        return Verdict("refused", str(e))
    problems = []
    if not ts.Engine.is_terminal(state):
        problems.append("the game is not over after its steps")
    elif outcome(state) != sub["result"]:
        problems.append(f"result {outcome(state)} != submitted {sub['result']}")
    if final_hash(state) != sub["final_hash"]:
        problems.append("final position differs from the submitted one")
    recomputed = summarize_submission(sub)
    if recomputed != sub["summary"]:
        problems.append("the page's summary differs from this one")
    return Verdict("mismatch", "; ".join(problems)) if problems else Verdict("ok")


def load_export(path: str) -> List[SubmissionDict]:
    """The submissions in an export of the `games` table.

    Either `wrangler d1 execute ... --json` output (a list of `{"results": [rows]}`), a list of
    rows, or a list of submissions; a row carries its submission in `body`. One per id.
    """
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    rows: List[Any] = []
    for item in data if isinstance(data, list) else [data]:
        if isinstance(item, dict) and isinstance(item.get("results"), list):
            rows.extend(item["results"])
        elif isinstance(item, dict):
            rows.append(item)
    out: Dict[str, SubmissionDict] = {}
    for row in rows:
        body = row.get("body", row) if isinstance(row, dict) else None
        sub = json.loads(body) if isinstance(body, str) else body
        if isinstance(sub, dict) and isinstance(sub.get("id"), str):
            out.setdefault(sub["id"], cast(SubmissionDict, sub))
    return list(out.values())


# ---- the rating on Playdek's scale -----------------------------------------------------------

_C = math.log(10.0) / 400.0


@dataclass(frozen=True)
class RatedGame:
    login: str
    human_rating: float
    human_side: Side
    #: The human's score: 1 win, 0.5 draw, 0 loss.
    score: float


def player_key(sub: SubmissionDict, summary: SummaryDict) -> str:
    """The model player a clean game was against: the model file and how it searched."""
    sha = summary["models"][0]
    info = sub["models"].get(sha)
    label = info["label"] if info else "?"
    return f"{label} [{sha[:12]}] {summary['searches'][0]}"


def rated_game(sub: SubmissionDict, summary: SummaryDict) -> RatedGame:
    side = summary["human_side"]
    assert side in ("US", "USSR")
    w = sub["result"]["winner"]
    return RatedGame(sub["playdek"]["login"].strip().lower(), float(sub["playdek"]["rating"]),
                     "US" if side == "US" else "USSR", 0.5 if w == "DRAW" else 1.0 if w == side else 0.0)


@dataclass(frozen=True)
class RatingCheck:
    """Whether a submitted game counts towards a rating."""
    summary: Optional[SummaryDict]
    #: Why it does not count; None when it does.
    why: Optional[str]
    other_engine_replayed: bool = False
    unverified: bool = False


def rating_check(sub: SubmissionDict, fingerprint: str, *, max_undos: Optional[int] = None,
                 include_unverified: bool = False) -> RatingCheck:
    """Every game that would count is re-driven on this build, whatever engine it names: the record's
    fingerprint is the submitter's claim. Most games outlive an engine change and replay here; one
    that does not cannot be told from a forgery, so it counts only with `include_unverified`."""
    try:
        s = summarize_submission(sub)
    except BAD_LOG:
        return RatingCheck(None, "verification: bad-log")
    # The cheap checks first, so a game left out anyway is never re-driven.
    why = ("not clean" if not s["clean"] else
           "too many undos" if max_undos is not None and s["undos"] > max_undos else
           "no rating" if sub["playdek"]["rating"] <= 0 else None)
    if why is not None:
        return RatingCheck(s, why)
    verdict = verify(sub, fingerprint, replay_other_engine=True)
    if verdict.status == "ok":
        return RatingCheck(s, None, other_engine_replayed=sub["build"]["engine_fingerprint"] != fingerprint)
    if verdict.status == "other-engine" and include_unverified:
        return RatingCheck(s, None, unverified=True)
    return RatingCheck(s, f"verification: {verdict.status}")


def _map_rating(games: Sequence[RatedGame], ussr_advantage: float, prior_mean: float,
                prior_sd: float) -> Tuple[float, float]:
    """The posterior mode of the model's rating and its curvature's standard error (Newton)."""
    h = np.array([g.human_rating + ussr_advantage * (1.0 if g.human_side == "USSR" else -1.0) for g in games])
    y = np.array([g.score for g in games])
    r = prior_mean
    info = 1.0 / prior_sd ** 2
    for _ in range(100):
        p = 1.0 / (1.0 + np.exp(-_C * (h - r)))
        grad = -_C * float(np.sum(y - p)) - (r - prior_mean) / prior_sd ** 2
        info = _C ** 2 * float(np.sum(p * (1.0 - p))) + 1.0 / prior_sd ** 2
        step = grad / info
        r += step
        if abs(step) < 1e-6:
            break
    return r, 1.0 / math.sqrt(info)


@dataclass
class RatingFit:
    player: str
    games: int
    humans: int
    #: The model's record: wins, draws, losses.
    record: Tuple[int, int, int]
    mean_opponent: float
    rating: float
    se: float
    #: A 95% bootstrap interval over the human players.
    interval: Tuple[float, float]
    by_side: Dict[str, Tuple[int, int, int]] = field(default_factory=dict)


def fit_rating(player: str, games: Sequence[RatedGame], *, ussr_advantage: float = 0.0,
               prior_mean: Optional[float] = None, prior_sd: float = 400.0,
               bootstrap: int = 1000, seed: int = 0) -> RatingFit:
    """The model player's rating on its opponents' (Playdek's) scale, from its games against them."""
    if not games:
        raise ValueError(f"no games for {player}")
    mean_opp = float(np.mean([g.human_rating for g in games]))
    mu = mean_opp if prior_mean is None else prior_mean
    rating, se = _map_rating(games, ussr_advantage, mu, prior_sd)
    by_login: Dict[str, List[RatedGame]] = {}
    for g in games:
        by_login.setdefault(g.login, []).append(g)
    logins = sorted(by_login)
    rng = np.random.default_rng(seed)
    draws = []
    for _ in range(bootstrap):
        pick = rng.integers(0, len(logins), len(logins))
        sample = [g for i in pick for g in by_login[logins[int(i)]]]
        draws.append(_map_rating(sample, ussr_advantage, mu, prior_sd)[0])
    lo, hi = (float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))) if draws else (rating, rating)

    def tally(gs: Iterable[RatedGame]) -> Tuple[int, int, int]:
        gs = list(gs)
        return (sum(g.score == 0.0 for g in gs), sum(g.score == 0.5 for g in gs), sum(g.score == 1.0 for g in gs))

    return RatingFit(
        player=player, games=len(games), humans=len(logins), record=tally(games), mean_opponent=mean_opp,
        rating=rating, se=se, interval=(lo, hi),
        # Keyed by the model's side, as the record is the model's.
        by_side={"US": tally(g for g in games if g.human_side == "USSR"),
                 "USSR": tally(g for g in games if g.human_side == "US")},
    )
