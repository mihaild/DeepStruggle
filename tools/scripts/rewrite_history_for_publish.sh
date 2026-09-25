#!/usr/bin/env bash
# tools/scripts/rewrite_history_for_publish.sh
#
# Rewrite the repository's history for publishing: every local branch, minus the big and the
# third-party files, in a fresh bare clone. The source repository is only read.
#
# Deterministic: a commit's rewritten SHA depends only on its content, metadata and parents, so
# re-running this later over a repository that has grown reproduces exactly the SHAs already
# published, and the commits made since land on top of them -- the later push is a fast-forward.
# That holds only for the same filter-repo and the same path list, hence the version pin below:
# change the list and every SHA from the first affected commit on changes with it.
#
# Removed from every commit:
#   data/warmup_5k_games.jsonl.gz, data/datasets/archive/   demonstration datasets (18.5 MB)
#   docs/card_strategies.md, docs/strategy/                 articles scraped from third-party sites
#   rules/map_rendered.png                                  a 1.3 MB render, deleted at HEAD already
#
# Left out of the clone (no SHA depends on them): the stash, remote-tracking refs, the old
# snapshot branch `main` (an unrelated one-commit history the publish replaces), and the
# `published/*` tags that belonged to it.
#
# Usage: rewrite_history_for_publish.sh <source-repo> <dest.git>
#   FILTER_REPO=/path/to/git-filter-repo   (pip install git-filter-repo==2.47.0)
set -euo pipefail

SRC="${1:?source repository}"
DEST="${2:?destination bare repository (must not exist)}"
FILTER_REPO="${FILTER_REPO:-git-filter-repo}"

want="a40bce548d2c"   # git-filter-repo 2.47.0
have="$("$FILTER_REPO" --version)"
if [ "$have" != "$want" ]; then
    echo "ERROR: git-filter-repo $have; this rewrite is pinned to $want (2.47.0)." >&2
    exit 1
fi
if [ -e "$DEST" ]; then
    echo "ERROR: $DEST exists; the rewrite needs a fresh clone." >&2
    exit 1
fi

git clone --quiet --no-local --bare "$SRC" "$DEST"
cd "$DEST"
git branch -D main >/dev/null 2>&1 || true
git tag -l | xargs -r git tag -d >/dev/null

"$FILTER_REPO" --invert-paths \
    --path data/warmup_5k_games.jsonl.gz \
    --path data/datasets/archive/ \
    --path docs/card_strategies.md \
    --path docs/strategy/ \
    --path rules/map_rendered.png

# filter-repo drops the origin remote so the rewrite cannot be pushed back by accident.
git gc --quiet --prune=now --aggressive
echo "rewritten: $DEST"
echo "old->new commit map: $DEST/filter-repo/commit-map"
