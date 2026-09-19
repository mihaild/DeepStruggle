"""Every relative markdown link under `research/` must resolve.

Archiving the E3 ladder moved ~20 findings, logs and plans under
`research/archive/E3_ladder/` without rewriting the links pointing at them, leaving **105 dead
links across 68 files** -- including `findings/training/seed_variance.md`, which five live method
and plan documents cite as the authority on how much evidence an effect needs. Nothing failed:
the record simply stopped being navigable, and it stayed that way until a link happened to be
followed.

That is the same shape as the other bookkeeping faults in this repo -- a rule ("update the links
when you archive") that has to be remembered every time, applied by hand, and is silent when
skipped. This test replaces the rule.

Anchors (`#section`) are not checked, only the file part. External links are ignored.
"""

from __future__ import annotations

import os
import re
from typing import List, Tuple

_HERE = os.path.dirname(os.path.abspath(__file__))
RESEARCH = os.path.normpath(os.path.join(_HERE, "..", "..", "research"))
LINK = re.compile(r"\[[^\]]*\]\(([^)]+)\)")


def _broken() -> List[Tuple[str, str]]:
    out: List[Tuple[str, str]] = []
    for dirpath, _dirnames, filenames in os.walk(RESEARCH):
        for name in filenames:
            if not name.endswith(".md"):
                continue
            path = os.path.join(dirpath, name)
            with open(path, encoding="utf-8") as f:
                text = f.read()
            for match in LINK.finditer(text):
                target = match.group(1).split("#")[0].strip()
                if not target or target.startswith(("http://", "https://", "mailto:")):
                    continue
                if not os.path.exists(os.path.normpath(os.path.join(dirpath, target))):
                    out.append((os.path.relpath(path, RESEARCH), target))
    return out


def test_no_broken_links_in_the_research_record() -> None:
    broken = _broken()
    if broken:
        listing = "\n".join(f"  {src} -> {target}" for src, target in sorted(broken))
        raise AssertionError(
            f"{len(broken)} broken link(s) under research/.\n"
            f"If a file was archived, repoint the links at its new home rather than "
            f"deleting them -- the content still exists.\n{listing}")


def test_the_checker_can_actually_fail(tmp_path) -> None:
    """A link checker that silently matches nothing would pass forever."""
    doc = tmp_path / "x.md"
    doc.write_text("[gone](./definitely_not_here.md)\n", encoding="utf-8")
    found = LINK.findall(doc.read_text(encoding="utf-8"))
    assert found == ["./definitely_not_here.md"]
    assert not os.path.exists(os.path.join(str(tmp_path), found[0]))


def test_it_is_scanning_a_real_corpus() -> None:
    """Guard against the walk silently covering nothing (a wrong RESEARCH path)."""
    count = sum(1 for dp, _dn, fn in os.walk(RESEARCH) for n in fn if n.endswith(".md"))
    assert count > 50, f"only {count} markdown files found under {RESEARCH}"
