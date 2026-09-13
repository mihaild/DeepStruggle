"""The public snapshot must not refer to what the public snapshot omits.

`publish_snapshot.sh` strips the private paths and proves none survived. That is only half the
job: a file that *does* survive can still cite `research/metrics.md`, name a checkpoint directory
nobody outside has, or recount an experiment by arm and seed, and a public reader follows that
reference into nothing. `check_public_hygiene.sh` reads the contents of the tree about to be
published and refuses those.

These tests guard the guard. Two ways it could quietly stop working:

* the two scripts' exclude lists drift apart, so the checker skips a directory that is now
  published, or checks one that is not;
* a rule's regex stops matching -- which has already happened twice here, once by splitting
  tab-separated rules on `|` (regex alternation) and once by running the patterns through
  `printf '%b'`, which turns a `\\b` word boundary into a backspace. Both failures were silent and
  both left the check reporting success.
"""
from __future__ import annotations

import re
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
PUBLISH = ROOT / "tools" / "scripts" / "publish_snapshot.sh"
HYGIENE = ROOT / "tools" / "scripts" / "check_public_hygiene.sh"


def _excluded_in_publish() -> set[str]:
    body = PUBLISH.read_text()
    block = body.split("EXCLUDED_PATHS=(", 1)[1].split(")", 1)[0]
    out: set[str] = set()
    for raw in block.splitlines():
        entry = raw.split("#", 1)[0].strip().strip('"')
        if not entry:
            continue
        if entry == "$SCRIPT_REL_PATH":
            entry = "tools/scripts/publish_snapshot.sh"
        out.add(entry.rstrip("/"))
    return out


def _excluded_in_hygiene() -> set[str]:
    body = HYGIENE.read_text()
    line = next(ln for ln in body.splitlines() if ln.startswith("EXCLUDED_PREFIXES="))
    pattern = line.split("'", 1)[1].rsplit("'", 1)[0]
    inner = pattern.removeprefix("^(").removesuffix(")")
    out: set[str] = set()
    for alt in inner.split("|"):
        if "publish_snapshot" in alt or "check_public_hygiene" in alt:
            continue
        out.add(alt.replace("\\", "").rstrip("/"))
    out.add("tools/scripts/publish_snapshot.sh")
    return out


def test_both_scripts_agree_on_what_is_private() -> None:
    """A path published but unchecked is the hole this whole mechanism exists to close."""
    publish, hygiene = _excluded_in_publish(), _excluded_in_hygiene()
    # The hygiene checker also skips itself, which publish_snapshot does not need to list.
    hygiene.discard("tools/scripts/check_public_hygiene.sh")
    assert publish == hygiene, (
        f"exclude lists have drifted.\n"
        f"  only in publish_snapshot.sh: {sorted(publish - hygiene)}\n"
        f"  only in check_public_hygiene.sh: {sorted(hygiene - publish)}")


def test_the_publish_script_runs_the_hygiene_check() -> None:
    assert "check_public_hygiene.sh" in PUBLISH.read_text(), \
        "publishing is no longer gated on the content check"


def _rules() -> list[tuple[str, str]]:
    body = HYGIENE.read_text()
    block = body.split("RULES=(", 1)[1].split("\n)", 1)[0]
    out: list[tuple[str, str]] = []
    for raw in block.splitlines():
        raw = raw.strip()
        if not raw.startswith('"'):
            continue
        parts = raw.strip('"').split("\t")
        if len(parts) >= 2:
            out.append((parts[0], parts[1]))
    return out


def test_every_rule_has_a_usable_regex() -> None:
    """The two silent failures were both broken regexes that matched nothing."""
    rules = _rules()
    assert len(rules) >= 8, f"expected the full rule set, parsed {len(rules)}"
    for name, pattern in rules:
        assert "\t" not in pattern
        re.compile(pattern)          # raises if the rule was truncated mid-group


@pytest.mark.parametrize("name,sample", [
    ("research-log", "see research/metrics.md section 21"),
    ("docs-dir", "listed in docs/strategy/index.md"),
    ("claude-dir", "a worktree under .claude/worktrees/"),
    ("checkpoint-path", "load data/checkpoints/run_v2/snapshot.pt"),
    ("snapshot-file", "resume from snapshot_80019456steps.pt"),
    ("run-shortname", "measured on E3-12-21 at 80M"),
    ("run-dirname", "the p1_scalar_nofilter directory"),
    ("arm-label", "this is what killed arm E"),
    ("elo-delta", "worth +53 Elo against the control"),
])
def test_each_rule_catches_what_it_is_for(name: str, sample: str) -> None:
    pattern = dict(_rules())[name]
    assert re.search(pattern, sample), f"rule {name!r} no longer catches {sample!r}"


def test_ordinary_prose_is_not_flagged() -> None:
    """The check forbids dangling references and bookkeeping, not explanation. A docstring that
    says *why* the code is shaped as it is must survive, or the rule becomes unusable and gets
    switched off."""
    allowed = (
        "A plain graph convolution applies one weight matrix to a node and its neighbours "
        "alike, so a country's own influence is attenuated in proportion to its degree.",
        "Influence is a non-negative integer, so rounding a least-squares fit is the wrong "
        "estimator for naming the exact value.",
        "The research into this showed the docstring approach works.",
    )
    patterns = [p for _, p in _rules()]
    for text in allowed:
        hits = [p for p in patterns if re.search(p, text)]
        assert not hits, f"ordinary prose flagged by {hits}: {text!r}"


def test_the_checker_actually_fails_on_a_planted_reference(tmp_path: Path) -> None:
    """End to end: the script must exit non-zero, not merely print."""
    script = HYGIENE.read_text()
    bad = tmp_path / "repo"
    (bad / "tools" / "scripts").mkdir(parents=True)
    (bad / "tools" / "scripts" / "check_public_hygiene.sh").write_text(script)
    (bad / "leaky.py").write_text('"""See research/metrics.md for why."""\n')
    subprocess.run(["git", "init", "-q"], cwd=bad, check=True)
    subprocess.run(["git", "add", "-A"], cwd=bad, check=True)
    proc = subprocess.run(
        ["bash", "tools/scripts/check_public_hygiene.sh"],
        cwd=bad, capture_output=True, text=True)
    assert proc.returncode != 0, "a planted research reference did not fail the check"
    assert "leaky.py" in proc.stderr
