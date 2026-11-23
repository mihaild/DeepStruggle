"""Re-driving a game people played against a model: each side seated as the page seated it.

The page seats the auto-play side with its model's obs_features and a person's side with the
person rules of its build (web/ui/src/main.ts `seatSides`); the rules decide what the engine asks,
so a replay must seat both, step by step, from the model's own checkpoint. A game from another
engine verifies on what it shares with this one -- winner, margin, end turn, the ending's name
through E9's renames -- not on the hash of a save JSON whose format changed. Synthetic games only:
the database's games are real people's.
"""
from __future__ import annotations

import copy
import hashlib
import io
import pathlib
import subprocess
import tarfile
from typing import Any, Callable, Dict, List, Optional, Tuple, cast

import onnx
import pytest
import ts_engine as ts
from onnx import helper

from tests.game_stats_helpers import CHECKPOINT, MODEL, PERSON, SHA, play_seated, submission, tiny_checkpoint
from tools.lib.engine_fingerprint import commits_by_fingerprint, fingerprint
from tools.lib.game_stats import (E9_REASONS_COMMIT, PERSON_RULES_HISTORY, BuildHistory, ModelFiles, ReplayContext,
                                  Seating, StepRefused, SubmissionDict, Unverifiable, decode_log, final_hash, outcome,
                                  rating_check, replay, same_reason, seating, verify)

HERE = fingerprint()
OTHER = "f" * 64
OPP = int(ts.OBS_FEATURE_OPP_EVENT_GATED)


@pytest.fixture(scope="module")
def models(tmp_path_factory: pytest.TempPathFactory) -> ModelFiles:
    root = tmp_path_factory.mktemp("checkpoints")
    tiny_checkpoint(str(root), CHECKPOINT, MODEL)
    return ModelFiles(str(root))


def _ctx(models: ModelFiles, builds: Optional[BuildHistory] = None) -> ReplayContext:
    return ReplayContext(HERE, models, builds or BuildHistory({}))


def _unseated_differs(sub: Dict[str, Any], seats: Seating) -> bool:
    seed, steps = decode_log(sub["log"])
    try:
        state = replay(cast(SubmissionDict, sub), seed, steps, seats)
    except StepRefused:
        return True
    return not ts.Engine.is_terminal(state) or outcome(state) != sub["result"] or final_hash(state) != sub["final_hash"]


def _first_game(want: Callable[[Dict[str, Any]], bool], auto_sides: List[str], *,
                switch_at: Optional[int] = None, model: int = MODEL) -> Dict[str, Any]:
    """The first seed's seated game that `want` accepts: a search over seeds, deterministic for a build."""
    for seed in range(1, 80):
        steps, state = play_seated(seed, auto_sides, model, PERSON, switch_at=switch_at)
        sub = submission(seed, steps, state, auto_sides, HERE, game=f"game-{seed:04d}")
        if want(sub):
            return sub
    raise AssertionError("no seed in 1..79 plays such a game")


@pytest.fixture(scope="module")
def seated_game() -> Dict[str, Any]:
    """A game whose seats matter: re-driven unseated, it is refused or ends elsewhere."""
    def matters(sub: Dict[str, Any]) -> bool:
        return _unseated_differs(sub, Seating(0, ((0, 0),) * sub["n_steps"]))
    return _first_game(matters, ["US"])


def test_a_game_replays_only_with_its_seats(seated_game: Dict[str, Any], models: ModelFiles) -> None:
    sub = cast(SubmissionDict, seated_game)
    assert sub["summary"]["clean"]
    v = verify(sub, _ctx(models))
    assert v.status == "ok", v.detail
    assert not v.other_engine and v.hash_matches and v.seating is not None
    us, ussr = v.seating.seats[0]
    assert (us, ussr) == (MODEL, PERSON), "the auto side its model's bits, the person the person rules"
    assert _unseated_differs(seated_game, Seating(0, ((0, 0),) * sub["n_steps"]))


def test_a_change_of_auto_side_mid_game_moves_the_seats(tmp_path: pathlib.Path) -> None:
    """Each step's `setting` names the auto side then in force: from step 20 the model plays the USSR
    and the person the US, and a replay that kept the first setting's seats is not this game. The
    model places its influence in canonical order (P43's rule bit, which a person never plays), so
    whichever side it seats is visible at every placement."""
    model = int(ts.OBS_FEATURE_HEADLINE_REVEAL) | int(ts.OBS_FEATURE_CANONICAL_FIRST)
    tiny_checkpoint(str(tmp_path), CHECKPOINT, model)
    models = ModelFiles(str(tmp_path))

    def matters(sub: Dict[str, Any]) -> bool:
        seed, steps = decode_log(sub["log"])
        kept = seating(cast(SubmissionDict, {**sub, "settings": sub["settings"][:1] * 2}), steps, PERSON, models)
        return _unseated_differs(sub, kept)

    sub = cast(SubmissionDict, _first_game(matters, ["US", "USSR"], switch_at=20, model=model))
    v = verify(sub, _ctx(models))
    assert v.status == "ok", v.detail
    assert v.seating is not None
    assert v.seating.seats[19] == (model, PERSON) and v.seating.seats[20] == (PERSON, model)


def test_another_engines_game_verifies_on_winner_margin_and_turn(seated_game: Dict[str, Any],
                                                                 models: ModelFiles) -> None:
    """The final hash hashes a save JSON whose format changes across builds: informational only."""
    sub = copy.deepcopy(seated_game)
    sub["build"]["engine_fingerprint"] = OTHER
    sub["final_hash"] = "0" * 64
    v = verify(cast(SubmissionDict, sub), _ctx(models))
    assert v.status == "ok" and v.other_engine and v.hash_matches is False, v.detail
    assert "every person rule set" in v.detail and "differs (informational)" in v.detail
    # This build's own record of it must still land on the exact position.
    here = copy.deepcopy(seated_game)
    here["final_hash"] = "0" * 64
    assert verify(cast(SubmissionDict, here), _ctx(models)).status == "mismatch"


def test_another_engines_game_must_end_as_submitted(seated_game: Dict[str, Any], models: ModelFiles) -> None:
    for field, value in (("winner", "DRAW"), ("margin", 7), ("end_turn", 11), ("reason", "Europe Control")):
        sub = copy.deepcopy(seated_game)
        sub["build"]["engine_fingerprint"] = OTHER
        sub["result"][field] = value
        v = verify(cast(SubmissionDict, sub), _ctx(models))
        assert v.status == "other-engine", (field, v.detail)
        assert "submitted" in v.detail


def test_a_known_build_is_replayed_with_its_own_person_rules(seated_game: Dict[str, Any], models: ModelFiles) -> None:
    """A fingerprint some commit here hashes to names the build's person rules; nothing else is tried."""
    sub = copy.deepcopy(seated_game)
    sub["build"]["engine_fingerprint"] = OTHER
    newest, opp_era = PERSON_RULES_HISTORY[0], PERSON_RULES_HISTORY[1]
    assert newest.bits == PERSON and opp_era.bits == OPP

    def builds(era_commit: str) -> BuildHistory:
        return BuildHistory({OTHER: ["b" * 40]}, lambda a, c: a in (era_commit, ""))

    v = verify(cast(SubmissionDict, sub), _ctx(models, builds(newest.since)))
    assert v.status == "ok" and "engine of commit bbbbbbbbbb" in v.detail, v.detail
    # Built before the page gave a person own_event_gated and always_ask_discard: this game, which
    # needs them, is not that build's.
    v = verify(cast(SubmissionDict, sub), _ctx(models, builds(opp_era.since)))
    assert v.status == "other-engine" and "person rules opp_event_gated:" in v.detail, v.detail
    assert "always_ask_discard" not in v.detail


def test_a_build_whose_seating_cannot_be_reproduced_is_unverifiable(seated_game: Dict[str, Any],
                                                                      models: ModelFiles) -> None:
    sub = copy.deepcopy(seated_game)
    sub["build"]["engine_fingerprint"] = OTHER
    window = next(e for e in PERSON_RULES_HISTORY if e.bits is None)
    v = verify(cast(SubmissionDict, sub), _ctx(models, BuildHistory({OTHER: ["b" * 40]},
                                                                     lambda a, c: a in (window.since, ""))))
    assert v.status == "unverifiable" and "whichever model last analysed it" in v.detail


def test_the_ending_renames_of_e9() -> None:
    assert same_reason("final scoring", "20 VP", old_names=True)
    assert same_reason("held scoring card", "wargames", old_names=True)
    assert same_reason("held scoring card", "20 VP", old_names=True)
    assert not same_reason("final scoring", "20 VP", old_names=False)
    assert not same_reason("20 VP", "final scoring", old_names=True), "the renames go one way"
    assert not same_reason("wargames", "Europe Control", old_names=True)


def test_a_pre_e9_name_is_accepted_only_from_a_pre_e9_build(models: ModelFiles) -> None:
    def held_scoring(sub: Dict[str, Any]) -> bool:
        return bool(sub["result"]["reason"] == "held scoring card")

    played = _first_game(held_scoring, ["US"])
    sub = copy.deepcopy(played)
    sub["build"]["engine_fingerprint"] = OTHER
    sub["result"]["reason"] = "20 VP"   # what a build before f95503c called a loss to a held scoring card
    v = verify(cast(SubmissionDict, sub), _ctx(models))
    assert v.status == "ok" and "reason '20 VP' is 'held scoring card' here" in v.detail, v.detail

    def builds(e9: bool) -> BuildHistory:
        return BuildHistory({OTHER: ["b" * 40]},
                            lambda a, c: a in (PERSON_RULES_HISTORY[0].since, "") or (e9 and a == E9_REASONS_COMMIT))

    assert verify(cast(SubmissionDict, sub), _ctx(models, builds(False))).status == "ok"
    assert verify(cast(SubmissionDict, sub), _ctx(models, builds(True))).status == "other-engine"


def test_a_model_whose_checkpoint_is_not_here_is_unverifiable(seated_game: Dict[str, Any],
                                                              tmp_path: pathlib.Path, models: ModelFiles) -> None:
    empty = ReplayContext(HERE, ModelFiles(str(tmp_path)), BuildHistory({}))
    v = verify(cast(SubmissionDict, seated_game), empty)
    assert v.status == "unverifiable" and f"its checkpoint {CHECKPOINT} is not in" in v.detail
    # It never counts on this build's engine; from another engine only as an unverified game.
    assert rating_check(cast(SubmissionDict, seated_game), empty, include_unverified=True).why == \
        "verification: unverifiable"
    other = copy.deepcopy(seated_game)
    other["build"]["engine_fingerprint"] = OTHER
    assert rating_check(cast(SubmissionDict, other), empty).why == "verification: unverifiable"
    assert rating_check(cast(SubmissionDict, other), empty, include_unverified=True).unverified
    assert rating_check(cast(SubmissionDict, other), _ctx(models)).other_engine_replayed


def _onnx_beside(path: pathlib.Path, meta: Dict[str, str]) -> str:
    model = helper.make_model(helper.make_graph([], "g", [], []))
    for k, v in meta.items():
        entry = model.metadata_props.add()
        entry.key, entry.value = k, v
    onnx_path = path.with_suffix(".onnx")
    onnx.save_model(model, str(onnx_path))
    return hashlib.sha256(onnx_path.read_bytes()).hexdigest()


def test_the_onnx_the_page_played_must_be_the_checkpoints(tmp_path: pathlib.Path) -> None:
    """The page read the rule bits from its ONNX; where that file lies beside the checkpoint, it must
    be an export of that checkpoint and say the bits its weights hold."""
    ck = pathlib.Path(tiny_checkpoint(str(tmp_path), "run/net.pt", MODEL))
    ck_sha = hashlib.sha256(ck.read_bytes()).hexdigest()
    info: Any = {"label": "net", "source": "", "checkpoint": "run/net.pt"}
    cases: List[Tuple[Dict[str, str], Optional[str]]] = [
        ({"ts.checkpoint_sha256": ck_sha, "ts.obs_features": str(MODEL)}, None),
        ({"ts.checkpoint_sha256": ck_sha, "ts.obs_features": str(int(ts.OBS_FEATURE_HEADLINE_REVEAL))}, "rule bits"),
        ({"ts.checkpoint_sha256": "0" * 64, "ts.obs_features": str(MODEL)}, "another checkpoint"),
    ]
    for meta, refused in cases:
        sha = _onnx_beside(ck, meta)
        files = ModelFiles(str(tmp_path))
        if refused is None:
            assert files.features(sha, info) == MODEL
        else:
            with pytest.raises(Unverifiable, match=refused):
                files.features(sha, info)
    # An ONNX of another sha256 beside it is some other export: the weights alone are read.
    assert ModelFiles(str(tmp_path)).features(SHA, info) == MODEL


def test_a_refusal_names_the_step_the_decision_and_the_card(seated_game: Dict[str, Any],
                                                            models: ModelFiles) -> None:
    sub = cast(SubmissionDict, seated_game)
    seed, steps = decode_log(sub["log"])
    with pytest.raises(StepRefused, match=r"^step \d+ of \d+ \(T\d+ AR\d+, (US|USSR|nobody) [A-Z_]+ \d+/\d+"):
        replay(sub, seed, steps, Seating(0, ((0, 0),) * len(steps)))


def test_a_commits_engine_fingerprint_is_computed_from_git(tmp_path: pathlib.Path) -> None:
    """commits_by_fingerprint hashes each commit's tree with fingerprint()'s rule: the newest engine
    commit's key is the fingerprint of that commit's sources, checked out."""
    root = pathlib.Path(__file__).resolve().parents[2]
    newest = subprocess.run(["git", "log", "-1", "--format=%H", "--", "CMakeLists.txt", "engine", "bindings"],
                            cwd=root, capture_output=True, text=True, check=True).stdout.strip()
    archive = subprocess.run(["git", "archive", "--format=tar", newest, "CMakeLists.txt", "engine", "bindings"],
                             cwd=root, capture_output=True, check=True).stdout
    with tarfile.open(fileobj=io.BytesIO(archive)) as tar:
        tar.extractall(tmp_path, filter="data")
    index = commits_by_fingerprint(root)
    assert newest in index[fingerprint(tmp_path)]
