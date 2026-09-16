"""The replay schema is pinned as source, so an accidental change to it fails loudly.

`test_replays_directory_schema_conformance` generates a replay and checks the writer's output
against the TypedDicts. That covers "the writer agrees with the schema" -- but it cannot catch a
change to the schema itself, because the writer and the definitions move together: rename a field
in `ReplayLogDict`, update the writer, and every generated-data test still passes while every
replay already on disk, and every consumer of them, silently stops matching.

What catches that is a golden copy of the schema, committed as source. It is a few kilobytes,
it shows up as a readable diff in review, and it makes a schema change a deliberate act rather
than a side effect. Deliberately the *schema* and not a sample replay: sample replays run to
several megabytes each and go stale, which is why `data/replays/` is git-ignored.

To accept an intended change, run:

    PYTHONPATH=.:build/release .venv/bin/python tests/bindings/test_replay_schema_golden.py

and commit the regenerated file alongside the change that caused it.
"""

from __future__ import annotations

import json
import os
import pathlib
import typing
from typing import Any, Dict, get_type_hints

from web.server import replay_types

GOLDEN_PATH = pathlib.Path(__file__).resolve().parent / "replay_schema.golden.json"

#: The types a replay file is actually made of. Other TypedDicts in replay_types cover metrics,
#: audits and websocket traffic, which are not part of the on-disk replay contract.
REPLAY_TYPES = (
    "ReplayLogDict",
    "ReplayMetadataDict",
    "ReplayPlayersDict",
    "ReplayResultDict",
    "ReplayStepDict",
    "ReplayActionDict",
    "ReplayPolicyDict",
    "PolicyChoiceDict",
    "ReplayCriticDict",
    "ReplayTraceMetaDict",
    "ReplayInitialStateDict",
    "ReplaySummaryDict",
    "GameStateDict",
    "CountryStateDict",
    "DecisionContextDict",
    "LegalActionsDict",
    "DieRollDict",
    "ChinaCardDict",
    "ActionLogEntryDict",
)


def _render(annotation: Any) -> str:
    """A stable, readable name for a type annotation.

    `repr` of a typing object embeds module paths and varies between Python versions, which
    would make the golden file churn for reasons that are not schema changes.
    """
    origin = typing.get_origin(annotation)
    if origin is None:
        name = getattr(annotation, "__name__", None)
        return str(name) if name else str(annotation).replace("typing.", "")
    args = typing.get_args(annotation)
    origin_name = getattr(origin, "__name__", str(origin)).replace("typing.", "")
    if origin_name in ("UnionType", "Union"):
        return " | ".join(sorted(_render(a) for a in args))
    return f"{origin_name}[{', '.join(_render(a) for a in args)}]"


def build_schema() -> Dict[str, Dict[str, str]]:
    """Field name -> rendered type, for each replay TypedDict, sorted for a stable diff."""
    schema: Dict[str, Dict[str, str]] = {}
    for type_name in sorted(REPLAY_TYPES):
        td = getattr(replay_types, type_name)
        hints = get_type_hints(td)
        schema[type_name] = {field: _render(hints[field]) for field in sorted(hints)}
    return schema


def test_replay_schema_matches_the_committed_golden_copy() -> None:
    assert GOLDEN_PATH.exists(), (
        f"{GOLDEN_PATH.name} is missing. Generate it with "
        f"'python {os.path.relpath(__file__)}' and commit it."
    )
    current = build_schema()
    golden = json.loads(GOLDEN_PATH.read_text(encoding="utf-8"))

    added = sorted(set(current) - set(golden))
    removed = sorted(set(golden) - set(current))
    assert not added and not removed, (
        f"replay schema types changed: added={added} removed={removed}. "
        f"If intended, regenerate {GOLDEN_PATH.name} and commit it with the change."
    )

    for type_name in sorted(current):
        assert current[type_name] == golden[type_name], (
            f"{type_name} changed.\n"
            f"  committed: {golden[type_name]}\n"
            f"  current:   {current[type_name]}\n"
            f"If intended, regenerate {GOLDEN_PATH.name} and commit it with the change."
        )


def test_the_golden_copy_is_not_empty() -> None:
    """Guards the guard: an empty schema would make the comparison above vacuous."""
    schema = build_schema()
    assert len(schema) == len(REPLAY_TYPES)
    assert schema["ReplayLogDict"], "ReplayLogDict has no fields"
    assert "steps" in schema["ReplayLogDict"]
    assert len(schema["GameStateDict"]) > 5


if __name__ == "__main__":
    GOLDEN_PATH.write_text(json.dumps(build_schema(), indent=2, sort_keys=True) + "\n",
                           encoding="utf-8")
    print(f"wrote {GOLDEN_PATH}")
