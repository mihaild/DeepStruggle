"""Games people played against the models in the web workbench, from the stats database.

The page records a game (web/ui/src/game/record.ts) as its seed and every step -- MicroAction,
forced die, who made it, the settings in force -- and on request submits it, with the player's
Playdek login and rating, to a Cloudflare Worker that keeps it in D1 (web/stats_worker/). This
module reads an export of that table and

* decodes the compact log and summarizes how each game was played -- the same rules as the
  page's `summarize()`, recomputed here rather than trusted from the row;
* re-drives a game in this build's engine with each side seated as the page seated it -- the
  auto-play side with its model's obs_features, read from the model's checkpoint, a person's side
  with its build's person rules -- and checks its end (`verify`): exactly, final position included,
  for a game played on this engine; on winner, margin, end turn and the ending's name (through
  E9's renames) for one played on another, whose save JSON no longer hashes alike;
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
import os
import pathlib
import struct
import subprocess
import zlib
from dataclasses import dataclass, field
from typing import (Any, Callable, Dict, Iterable, List, Literal, Mapping, Optional, Sequence, Tuple, TypedDict,
                    cast)

import numpy as np
import onnx
from huggingface_hub import hf_hub_download

import ts_engine as ts
from bindings.action_encoder import ActionEncoder
from tools.lib.data_root import checkpoints_dir, data_path
from tools.lib.engine_fingerprint import commits_by_fingerprint
from tools.lib.game_step import drain_chance, step_checked
from tools.lib.player_agent import NeuralAgent

_ROOT = pathlib.Path(__file__).resolve().parents[2]

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


# ---- the seats: which rules each side played ---------------------------------------------------
#
# The page seats each side with rule bits (web/ui/src/main.ts `seatSides`, engine/AGENTS.md on rule
# bits): the auto-play side with its model's obs_features, a person's side with the person rules of
# that build (WasmEngine.currentRules, obs_features::PERSON_RULES). The rules decide what is offered
# -- an Event that would do nothing, a discard question asked or not -- so a game re-driven unseated
# plays the pre-rule-bit engine and is refused where the person was asked something it does not ask.
# A replay seats both sides as the page did, step by step: each step's `setting` names the auto side
# and model then in force.

OPP_EVENT_GATED = int(ts.OBS_FEATURE_OPP_EVENT_GATED)
OWN_EVENT_GATED = int(ts.OBS_FEATURE_OWN_EVENT_GATED)
ALWAYS_ASK_DISCARD = int(ts.OBS_FEATURE_ALWAYS_ASK_DISCARD)
_RULE_NAMES = ((OPP_EVENT_GATED, "opp_event_gated"), (OWN_EVENT_GATED, "own_event_gated"),
               (ALWAYS_ASK_DISCARD, "always_ask_discard"))


def rule_names(bits: int) -> str:
    """`bits`' person rules by name, "none" for 0."""
    return "|".join(name for bit, name in _RULE_NAMES if bits & bit) or "none"


@dataclass(frozen=True)
class EnvyRules:
    """How a build's engine asked Missile Envy's giver -- the one question whose asking changed in place
    after games were played, so a replay must answer it as that build did (`replay`).

    `effective_ops`: ties are among the giver's highest *effective* Ops (4dd0ead, 2026-11-18, owner's
    ruling); before, among the highest printed Ops. `asks_single`: under always_ask_discard the giver
    is asked even when one card qualifies (f088816, 2026-11-18); before, that card was exchanged at
    once. An engine that did not ask exchanged exactly the card the question's one answer gives
    (f088816: "answering it makes the same exchange"), so a replay answers it itself, from the
    rules, never from a guess."""
    effective_ops: bool
    asks_single: bool


CURRENT_ENVY = EnvyRules(True, True)
_PRINTED_ENVY = EnvyRules(False, False)
MISSILE_ENVY = 49


@dataclass(frozen=True)
class PersonRules:
    """The rule bits the page seated a person's side with, from commit `since` on."""
    since: str
    #: None where what the page seated a person with cannot be reproduced.
    bits: Optional[int]
    why: str
    envy: EnvyRules = _PRINTED_ENVY

    @property
    def name(self) -> str:
        return rule_names(self.bits or 0) + ("" if self.envy == CURRENT_ENVY else
                                              " (Missile Envy as before 4dd0ead)" if self.envy.asks_single else
                                              " (Missile Envy as before f088816)")


#: What a person's side played, by build, newest first, from the git history of web/ui/src/main.ts
#: (`seatSides`), bindings/wasm/ts_engine_wasm.cpp (`ts_obs_features_rules`, the page's currentRules)
#: and engine/include/ts/game_state.hpp (obs_features::RULES, PERSON_RULES).
PERSON_RULES_HISTORY: Tuple[PersonRules, ...] = (
    PersonRules("4dd0ead4662ef62e87df711a35dbeea8b0913a22", OPP_EVENT_GATED | OWN_EVENT_GATED | ALWAYS_ASK_DISCARD,
                "Missile Envy's ties are the giver's highest effective Ops (2026-11-18)", CURRENT_ENVY),
    PersonRules("f08881664c9b585c9bf74944e03ccd2bf5068da9", OPP_EVENT_GATED | OWN_EVENT_GATED | ALWAYS_ASK_DISCARD,
                "always_ask_discard asks Missile Envy's giver even when one card qualifies (2026-11-18)",
                EnvyRules(False, True)),
    PersonRules("464bb90916f72a6ca8144e1bdd439ca55b9d9d68", OPP_EVENT_GATED | OWN_EVENT_GATED | ALWAYS_ASK_DISCARD,
                "PERSON_RULES gains own_event_gated and always_ask_discard (2026-11-11)"),
    PersonRules("f383abd28cfc57c3cb60442d3a388ace90e0a063", OPP_EVENT_GATED,
                "the page seats each side (seatSides); a person plays every rule bit then, opp_event_gated "
                "(bd7a3eb's PERSON_RULES left out the canonical orders it added to RULES; 2026-11-06)"),
    PersonRules("d2cc85f65c6a22a472ef4601150455a3d2a05afd", None,
                "opp_event_gated is a rule bit, and the page's mask seats a person's side with whichever "
                "model last analysed it (2026-11-06, until f383abd)"),
    PersonRules("", 0, "no rule bits: every seat plays the same rules"),
)

#: E9's ending classifier (audit T-10): from the engine's records, where an older build classified by
#: the score. For a reason the current engine gives, the names an older build gave the same ending:
#: final scoring clamps VP to +/-20, so a game decided there read "20 VP"; a held scoring card read
#: "20 VP" (one holder) or "wargames" (both, VP 0). Every other name is unchanged since the
#: classifier was written (f21985c, 2026-09-25, before the first stored game).
E9_REASONS_COMMIT = "f95503ced34cb0f82fe7835f1ab74edbf1288db3"
PRE_E9_REASON_NAMES: Dict[str, Tuple[str, ...]] = {
    "final scoring": ("20 VP",),
    "held scoring card": ("20 VP", "wargames"),
}


def same_reason(replayed: str, submitted: str, old_names: bool) -> bool:
    """Whether a submitted ending reason names the replayed ending, through E9's renames when the
    build may have used the old names."""
    return submitted == replayed or (old_names and submitted in PRE_E9_REASON_NAMES.get(replayed, ()))


class Unverifiable(Exception):
    """What a game was played with cannot be established here: its replay could only guess."""


def _sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


#: Fetches `path` at `revision` of the Hugging Face repo `repo` into `cache_dir`; returns the local file.
Download = Callable[[str, str, str, str], str]


def hf_download(repo: str, path: str, revision: str, cache_dir: str) -> str:
    return str(hf_hub_download(repo, path, revision=revision, cache_dir=cache_dir))


def parse_hf_source(source: str) -> Optional[Tuple[str, str, str]]:
    """(repo, revision, path) of a model source the page writes as hf:<repo>@<revision>:<path>
    (web/ui/src/analysis/model.ts), None for any other source."""
    if not source.startswith("hf:") or "@" not in source:
        return None
    repo, rest = source[3:].split("@", 1)
    revision, sep, path = rest.partition(":")
    return (repo, revision, path) if repo and ":" not in repo and revision and sep and path else None


def _onnx_meta(path: str) -> Dict[str, str]:
    return {p.key: p.value for p in onnx.load(path, load_external_data=False).metadata_props}


class ModelFiles:
    """The obs_features bits of the models games were played with, read from each model itself.

    A submission names a model by its ONNX file's sha256, with the checkpoint it was exported from
    (`ts.checkpoint`, relative to the checkpoints directory) and where the page loaded it (`source`).

    * The checkpoint, when it is here: the bits are read from its weights (NeuralAgent: LadderNet's
      obs_features buffer). Where the ONNX the page loaded lies beside it -- a file of the recorded
      sha256 -- the chain is checked too: that ONNX must name the checkpoint's own sha256
      (`ts.checkpoint_sha256`), and the rule bits the page read from its metadata (`ts.obs_features`)
      must be the weights'.
    * Otherwise (CI has no checkpoints) the ONNX itself, from its Hugging Face source, downloaded into
      `cache_dir` and required to hash to the recorded sha256: its `ts.obs_features` is what the page
      read and seated.

    Neither available -- no checkpoint here and no Hugging Face source, a failed download, a file of
    another sha256 -- makes the game unverifiable: its seat's rules would be a guess.
    """

    def __init__(self, root: str, *, cache_dir: Optional[str] = None, download: Optional[Download] = None) -> None:
        self.root = root
        self.cache_dir = cache_dir or os.path.join(os.path.dirname(os.path.abspath(root)), "hf_models")
        self._download: Download = download or hf_download
        self._known: Dict[str, int] = {}

    def features(self, sha: str, info: Optional[ModelInfoDict]) -> int:
        if sha in self._known:
            return self._known[sha]
        if info is None:
            raise Unverifiable(f"model {sha[:12]} is not among the record's models")
        label, ck = info["label"], info["checkpoint"]
        safe = bool(ck) and not os.path.isabs(ck) and ".." not in pathlib.PurePath(ck).parts
        path = os.path.join(self.root, ck) if safe else ""
        bits = self._from_checkpoint(sha, label, ck, path) if path and os.path.isfile(path) else \
            self._from_source(sha, label, ck, info["source"])
        if bits & int(ts.OBS_FEATURE_CANONICAL_LEARNED):
            raise Unverifiable(f"model {label} plays canonical_learned, whose order is the network's own logits at "
                               f"each run's start: a replay would have to run it")
        self._known[sha] = bits
        return bits

    def _from_checkpoint(self, sha: str, label: str, ck: str, path: str) -> int:
        bits = int(NeuralAgent.from_checkpoint(path, device="cpu").obs_features)
        onnx_path = os.path.splitext(path)[0] + ".onnx"
        if os.path.isfile(onnx_path) and _sha256_file(onnx_path) == sha:
            meta = _onnx_meta(onnx_path)
            exported_from = meta.get("ts.checkpoint_sha256")
            if exported_from is not None and exported_from != _sha256_file(path):
                raise Unverifiable(f"model {label}: the ONNX the page played was exported from another "
                                   f"checkpoint than {ck}")
            read = int(meta.get("ts.obs_features", "0"))
            if read & int(ts.OBS_FEATURES_RULES) != bits & int(ts.OBS_FEATURES_RULES):
                raise Unverifiable(f"model {label}: the page read rule bits {read:#x} from its ONNX, the "
                                   f"checkpoint's weights hold {bits:#x}")
        return bits

    def _from_source(self, sha: str, label: str, ck: str, source: str) -> int:
        missing = f"model {label}: its checkpoint {ck or '(none named)'} is not in {self.root}"
        hf = parse_hf_source(source)
        if hf is None:
            raise Unverifiable(f"{missing}, and its source {source or '(a dropped file)'} is not on Hugging Face, "
                               f"so the rule bits its seat played cannot be read")
        repo, revision, file = hf
        try:
            local = self._download(repo, file, revision, self.cache_dir)
        except Exception as e:   # whatever the hub or the network raised: the finding, with its reason
            raise Unverifiable(f"{missing}, and downloading {source} failed: {type(e).__name__}: {e}") from e
        got = _sha256_file(local)
        if got != sha:
            raise Unverifiable(f"{missing}, and {source} is now another file (sha256 {got[:12]}, the page played "
                               f"{sha[:12]})")
        return int(_onnx_meta(local).get("ts.obs_features", "0"))


def model_files(checkpoints: Optional[str] = None) -> ModelFiles:
    """The models' checkpoints under `checkpoints` (default: the shared data tree's checkpoints/), with
    Hugging Face downloads cached in the data tree's hf_models/."""
    return ModelFiles(checkpoints or checkpoints_dir(), cache_dir=data_path("hf_models"))


class BuildHistory:
    """What this clone's git history says about the build a game was played on.

    `commits(fingerprint)` are the commits whose engine sources hash to the build's engine fingerprint
    (engine_fingerprint.commits_by_fingerprint, computed on first use); `descends(a, c)` whether
    commit `c` contains `a` -- None where git cannot say (a commit this clone lacks). The page's engine
    and the page are built from one checkout, so a build's engine sources name the rules its page
    seated.
    """

    def __init__(self, index: Optional[Mapping[str, Sequence[str]]] = None,
                 descends: Optional[Callable[[str, str], Optional[bool]]] = None,
                 root: Optional[pathlib.Path] = None) -> None:
        self._index = index
        self._descends = descends
        self._root = root
        self._cache: Dict[Tuple[str, str], Optional[bool]] = {}

    def commits(self, fingerprint: str) -> Sequence[str]:
        if self._index is None:
            self._index = commits_by_fingerprint(self._root)
        return self._index.get(fingerprint, ())

    def descends(self, ancestor: str, commit: str) -> Optional[bool]:
        if self._descends is not None:
            return self._descends(ancestor, commit)
        key = (ancestor, commit)
        if key not in self._cache:
            code = subprocess.run(["git", "merge-base", "--is-ancestor", ancestor, commit],
                                  cwd=str(self._root or _ROOT), capture_output=True).returncode
            self._cache[key] = True if code == 0 else False if code == 1 else None
        return self._cache[key]


@dataclass(frozen=True)
class ReplayContext:
    """What re-driving games needs besides the games: this build's engine fingerprint, the models'
    files and the git history that names other builds."""
    fingerprint: str
    models: ModelFiles
    builds: BuildHistory


@dataclass(frozen=True)
class BuildRules:
    """What a game's build seated a person with: the candidates to replay, newest first."""
    candidates: Tuple[PersonRules, ...]
    #: How they were chosen, for the verdict.
    basis: str
    #: Whether the build may have named endings as before E9 (PRE_E9_REASON_NAMES).
    old_reason_names: bool


def build_rules(sub: SubmissionDict, ctx: ReplayContext) -> BuildRules:
    """The person rules `sub`'s build seated, from its engine fingerprint.

    This build: its own PERSON_RULES. A fingerprint some commits here hash to: the newest PERSON_RULES_HISTORY
    entry any of them descends from (they share their sources). Any other -- a build of a tree this clone lacks, such as the
    twilightstruggle.app deployments, built from a repository whose history is not here -- every rule
    set the page has had that can be reproduced, newest first, whichever its date: that repository
    carried the 2026-11-11 person rules (464bb90, made for it) before they were merged here, and its
    games of 2026-10-31..11-10 replay only with them. The set that replays is named in the verdict.
    Raises Unverifiable for a build whose page seated a person in a way that cannot be reproduced.
    """
    fp = sub["build"]["engine_fingerprint"]
    if fp == ctx.fingerprint:
        return BuildRules((PersonRules("", int(ts.OBS_FEATURES_PERSON_RULES), "this build", CURRENT_ENVY),),
                          "this build", False)
    eras: List[PersonRules] = []
    e9: List[Optional[bool]] = []
    commits = list(ctx.builds.commits(fp))
    for commit in commits:
        era = None
        for entry in PERSON_RULES_HISTORY:
            known = True if not entry.since else ctx.builds.descends(entry.since, commit)
            if known is None:
                era = None
                break
            if known:
                era = entry
                break
        if era is None:
            eras = []
            break
        if era not in eras:
            eras.append(era)
        e9.append(ctx.builds.descends(E9_REASONS_COMMIT, commit))
    if eras:
        # Every commit here with this fingerprint has the same engine sources, so they hold every change
        # any of them descends from: a rebased copy of a commit (7655378 is 464bb90's) descends from less
        # than its twin, not from older rules. The newest era wins, and E9's names if any one has them.
        era = min(eras, key=PERSON_RULES_HISTORY.index)
        where = f"engine of commit {commits[0][:10]}" + (f" and {len(commits) - 1} more" if len(commits) > 1 else "")
        if era.bits is None:
            raise Unverifiable(f"{where}, from {era.since[:7]} on: {era.why}")
        return BuildRules((era,), where, not any(x is True for x in e9))
    every = tuple(e for e in PERSON_RULES_HISTORY if e.bits is not None)
    return BuildRules(every, "an engine no commit here has; every person rule set the page has had, newest first",
                      True)


def _setting(sub: SubmissionDict, i: int, st: Step) -> SettingsDict:
    if st.setting >= len(sub["settings"]):
        raise ValueError(f"step {i + 1} names setting {st.setting} of {len(sub['settings'])}")
    return sub["settings"][st.setting]


@dataclass(frozen=True)
class Seating:
    """Each step's seats, as the page seated them: the auto-play side its model's obs_features, a
    person's side (and an auto side with no model) the person rules."""
    person: int
    #: Per step: (US, USSR) obs_features.
    seats: Tuple[Tuple[int, int], ...]
    #: How the build asked Missile Envy's giver.
    envy: EnvyRules = CURRENT_ENVY
    name: str = ""

    def seat(self, state: ts.GameState, i: int) -> None:
        us, ussr = self.seats[i]
        state.seat_obs_features(ts.Player.US, us)
        state.seat_obs_features(ts.Player.USSR, ussr)


def seating(sub: SubmissionDict, steps: Sequence[Step], person: int, models: ModelFiles,
            envy: EnvyRules = CURRENT_ENVY, name: str = "") -> Seating:
    """The seats of every step of `sub` with `person` on a person's side. Raises Unverifiable when a
    model's rule bits cannot be read, ValueError for a step naming a setting the record lacks."""
    seats: List[Tuple[int, int]] = []
    for i, st in enumerate(steps):
        s = _setting(sub, i, st)
        auto = models.features(s["model"], sub["models"].get(s["model"])) if s["auto"] and s["model"] else person
        seats.append((auto if s["auto"] == "US" else person, auto if s["auto"] == "USSR" else person))
    return Seating(person, tuple(seats), envy, name or rule_names(person))


def _unasked_envy_answer(state: ts.GameState, envy: EnvyRules, seats: Tuple[int, int]) -> Optional[int]:
    """The flat action answering a Missile Envy giver's question the build's engine never asked, or None.

    This engine asks the giver to choose among the highest effective-Ops cards, and under
    always_ask_discard even when one qualifies. A build before 4dd0ead compared printed Ops -- the
    giver's modifiers apply to every card alike and never lower a higher card below a lower one, so its
    candidates are the printed-highest among these -- and one before f088816 asked only a real tie.
    Where the build would have had one candidate and not asked, it exchanged that card at once."""
    if envy == CURRENT_ENVY:
        return None
    ctx = state.ctx()
    if int(ctx.decision_type) != int(ts.DecisionType.SELECT_CARD) or int(ctx.resolving_card) != MISSILE_ENVY:
        return None
    legal = {int(ActionEncoder.decode(state, int(f)).primary_id): int(f)
             for f in np.flatnonzero(ActionEncoder.get_legal_mask(state))}
    if not envy.effective_ops:
        top = max(int(ts.CardData.get_card_info(c)["ops"]) for c in legal)
        legal = {c: f for c, f in legal.items() if int(ts.CardData.get_card_info(c)["ops"]) == top}
    if len(legal) != 1:
        return None
    # A single candidate: asked by a build after f088816 only when a person's seat played the bit.
    asked = envy.asks_single and bool((seats[0] | seats[1]) & ALWAYS_ASK_DISCARD)
    return None if asked else next(iter(legal.values()))


class StepRefused(Exception):
    """The engine refused a recorded step: which one, the decision, its card, and why."""


def describe_step(state: ts.GameState, i: int, n: int, st: Step) -> str:
    """Step `i` (0-based) of `n` as it was offered: turn, round, mover, decision, card."""
    dt = ts.DecisionType(st.decision_type) if st.decision_type <= int(ts.DecisionType.ROLL_DIE) else None
    card = st.primary_id if st.decision_type in (1, 2, 3, 4) else int(state.ctx().pending_op_card)
    where = f"T{int(state.turn)} AR{int(state.action_round)}"
    what = f"{dt.name if dt is not None else st.decision_type} {st.primary_id}/{st.secondary_id}"
    return (f"step {i + 1} of {n} ({where}, {st.mover or 'nobody'} {what}"
            + (f", card {ts.CardData.get_card_name(card)}" if 0 < card <= 110 else "") + ")")


def replay(sub: SubmissionDict, seed: int, steps: Sequence[Step], seats: Seating,
           on_step: Optional[Callable[[int, Step, ts.GameState], None]] = None) -> ts.GameState:
    """Re-drive `sub` with `seats`; `on_step` sees each step's state before it is played. Raises
    StepRefused where the engine refuses a step."""
    state = start_state(seed, sub["start_position"])
    for i, st in enumerate(steps):
        seats.seat(state, i)
        while (answer := _unasked_envy_answer(state, seats.envy, seats.seats[i])) is not None:
            step_checked(state, answer, context="Missile Envy's question the build did not ask")
            drain_chance(state, context="Missile Envy's exchange")
        if on_step is not None:
            on_step(i, st, state)
        where = describe_step(state, i, len(steps), st)
        try:
            play_step(state, st)
        except Exception as e:   # the engine's refusal, whatever its type: it is the finding
            raise StepRefused(f"{where}: {e}") from e
    return state


@dataclass
class Verdict:
    """What re-driving a submitted game in this build found."""
    status: Literal["ok", "mismatch", "refused", "other-engine", "unverifiable", "bad-log"]
    detail: str = ""
    #: The record names another engine than this build's.
    other_engine: bool = False
    #: The seats it replayed with, when it replayed.
    seating: Optional[Seating] = None
    #: Whether it landed on the submitted final position (informational for another engine).
    hash_matches: Optional[bool] = None


#: What decode_log raises on a log that is not one.
BAD_LOG = (ValueError, zlib.error, struct.error)


def verify(sub: SubmissionDict, ctx: ReplayContext) -> Verdict:
    """Re-drive `sub` in this build, its sides seated as the page seated them, and check its end.

    Played on this build's engine: it must end where the page said, exactly -- terminal, the same
    result (reason included), the submitted final position (`final_hash`) and the page's summary.

    Played on another engine (its fingerprint is the submitter's claim, which cannot excuse it from
    being re-driven): it is "ok" when it replays here, seated with its build's person rules
    (`build_rules`), to a terminal position with the submitted winner, margin and end turn, its reason
    the submitted one or its pre-E9 name (PRE_E9_REASON_NAMES), and the same summary. The final hash
    is informational: it hashes the save JSON, whose format grows new fields, so it never matches
    across builds even when every move is the same. One that does not replay so is "other-engine",
    with where each candidate was refused: from here an engine change and a forged record look alike.

    "unverifiable": what the game was played with cannot be established here (a model's checkpoint
    is missing, or the build seated a person in a way that cannot be reproduced).
    """
    fp = sub["build"]["engine_fingerprint"]
    other = fp != ctx.fingerprint
    commit = sub["build"]["commit"].split("@")[-1][:12] or "?"
    where = f"played on engine {fp[:12]} (commit {commit}); this build is {ctx.fingerprint[:12]}" if other else ""

    def said(text: str) -> str:
        return f"{where}; {text}" if where else text

    try:
        seed, steps = decode_log(sub["log"])
        for i, st in enumerate(steps):
            _setting(sub, i, st)
    except BAD_LOG as e:
        return Verdict("bad-log", str(e), other)
    try:
        rules = build_rules(sub, ctx)
        candidates = [seating(sub, steps, cast(int, era.bits), ctx.models, era.envy, era.name)
                      for era in rules.candidates]
    except Unverifiable as e:
        return Verdict("unverifiable", said(str(e)), other)
    summary_ok = summarize_submission(sub) == sub["summary"]
    failures: List[str] = []
    for seats in candidates:
        try:
            state = replay(sub, seed, steps, seats)
        except StepRefused as e:
            failures.append(f"person rules {seats.name}: refused at {e}")
            continue
        got = outcome(state)
        hash_ok = final_hash(state) == sub["final_hash"]
        want = sub["result"]
        problems = []
        if not ts.Engine.is_terminal(state):
            problems.append("the game is not over after its steps")
        elif not other and got != want:
            problems.append(f"result {got} != submitted {want}")
        elif other and ((got["winner"], got["margin"], got["end_turn"]) != (want["winner"], want["margin"], want["end_turn"])
                        or not same_reason(got["reason"], want["reason"], rules.old_reason_names)):
            problems.append(f"ends {got['winner']} {got['margin']:+d} T{got['end_turn']} ({got['reason']}), "
                            f"submitted {want['winner']} {want['margin']:+d} T{want['end_turn']} ({want['reason']})")
        if not other and not hash_ok:
            problems.append("final position differs from the submitted one")
        if not summary_ok:
            problems.append("the page's summary differs from this one")
        if problems:
            failures.append(f"person rules {seats.name}: " + "; ".join(problems))
            continue
        if not other:
            return Verdict("ok", "", False, seats, True)
        renamed = f"; reason {want['reason']!r} is {got['reason']!r} here" if got["reason"] != want["reason"] else ""
        return Verdict("ok", said(f"replays here with person rules {seats.name} ({rules.basis}){renamed}; "
                                  f"final position hash {'matches' if hash_ok else 'differs (informational)'}"),
                       True, seats, hash_ok)
    if not candidates:
        failures.append(f"no person rules to try ({rules.basis})")
    if not other:
        last = failures[-1].split(": ", 1)[1]
        return Verdict("refused" if "refused at" in failures[-1] else "mismatch", last, False)
    return Verdict("other-engine", said(f"does not replay here ({rules.basis}): " + " / ".join(failures)), True)


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


def rating_check(sub: SubmissionDict, ctx: ReplayContext, *, max_undos: Optional[int] = None,
                 include_unverified: bool = False) -> RatingCheck:
    """Whether `sub` counts: clean, rated, and `verify`'s "ok" -- the same check, whatever engine the
    record names. One played on another engine that is not verified here ("other-engine", or
    "unverifiable" for want of a model file) cannot be told from a forgery, so it counts only with
    `include_unverified`; a game on this build's engine always has to verify."""
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
    verdict = verify(sub, ctx)
    if verdict.status == "ok":
        return RatingCheck(s, None, other_engine_replayed=verdict.other_engine)
    if verdict.other_engine and verdict.status in ("other-engine", "unverifiable") and include_unverified:
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
