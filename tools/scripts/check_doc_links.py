#!/usr/bin/env python3
"""Check every markdown cross-reference, the way Obsidian resolves them.

Run from the repository root:

    .venv/bin/python tools/scripts/check_doc_links.py

Exits non-zero if anything is broken, so it can gate a commit that touches documentation.

Obsidian differs from GitHub in ways that matter here:
  * a relative link resolves from the FILE's directory, so `../x.md` must be right per file;
  * a link to a directory does not open anything;
  * an anchor `#Some Heading` matches the heading TEXT, not a GitHub-style slug, so
    `#3-what-search-costs` resolves on GitHub and fails in Obsidian;
  * a bare `file.md` with no path is resolved by vault-wide filename search, which works only if
    the name is unique.

Reports broken targets, ambiguous bare names, and anchors that do not match a heading.
"""
import os
import re
from collections import defaultdict

ROOT = "."
SCAN = ["research", "engine", "bindings", "bot", "web", "tools", "ai", "tests", "."]
LINK = re.compile(r"\[([^\]]*)\]\(([^)]+)\)")
HEADING = re.compile(r"^(#{1,6})\s+(.*?)\s*$", re.M)

md_files = []
for base in SCAN:
    for dirpath, dirnames, filenames in os.walk(base):
        if any(p in dirpath for p in (".git", "node_modules", "build", "external", ".venv")):
            continue
        for fn in filenames:
            if fn.endswith(".md"):
                md_files.append(os.path.normpath(os.path.join(dirpath, fn)))
md_files = sorted(set(md_files))

by_name = defaultdict(list)
headings = {}
for f in md_files:
    by_name[os.path.basename(f)].append(f)
    try:
        text = open(f, encoding="utf-8").read()
    except Exception:
        text = ""
    headings[f] = [m.group(2) for m in HEADING.finditer(text)]


def obsidian_anchor_ok(target_file, anchor):
    """Obsidian matches the heading text, case-insensitively, ignoring markdown emphasis."""
    want = anchor.replace("-", " ").strip().lower()
    for h in headings.get(target_file, []):
        clean = re.sub(r"[`*_]", "", h).strip().lower()
        if clean == anchor.strip().lower() or clean == want:
            return True
        # GitHub-style slug of the heading
        slug = re.sub(r"[^a-z0-9\s-]", "", clean).replace(" ", "-")
        if slug == anchor.strip().lower():
            return "github-only"
    return False


broken, ambiguous, bad_anchor, dir_links = [], [], [], []
total = 0
for f in md_files:
    try:
        text = open(f, encoding="utf-8").read()
    except Exception:
        continue
    for m in LINK.finditer(text):
        label, target = m.group(1), m.group(2).strip()
        if target.startswith(("http://", "https://", "mailto:", "#")):
            continue
        total += 1
        path, _, anchor = target.partition("#")
        path = path.strip()
        if not path:
            continue
        resolved = os.path.normpath(os.path.join(os.path.dirname(f), path))
        if os.path.isdir(resolved):
            dir_links.append((f, target))
            continue
        if not os.path.exists(resolved):
            if "/" not in path and path in by_name:
                if len(by_name[path]) > 1:
                    ambiguous.append((f, target, by_name[path]))
            else:
                broken.append((f, target))
            continue
        if anchor and resolved.endswith(".md"):
            ok = obsidian_anchor_ok(resolved, anchor)
            if ok is not True:
                bad_anchor.append((f, target, "github-slug only" if ok else "no such heading"))

print(f"{len(md_files)} markdown files, {total} internal links\n")
for name, rows in (("BROKEN TARGET", broken), ("LINK TO A DIRECTORY", dir_links),
                   ("AMBIGUOUS BARE NAME", ambiguous), ("ANCHOR NOT OBSIDIAN-RESOLVABLE", bad_anchor)):
    print(f"{name}: {len(rows)}")
    for r in rows[:25]:
        print("   ", " -> ".join(str(x) for x in r))
    if len(rows) > 25:
        print(f"    ... and {len(rows) - 25} more")
    print()


import sys
sys.exit(1 if (broken or dir_links or ambiguous or bad_anchor) else 0)
