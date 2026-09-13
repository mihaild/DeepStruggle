#!/usr/bin/env bash
# tools/scripts/check_public_hygiene.sh
#
# Fails if anything destined for the public snapshot refers to something the public snapshot does
# not contain.
#
# `publish_snapshot.sh` already proves that no *excluded path* survives into the published tree.
# That is only half the job: a file that survives can still point at `research/metrics.md`, name a
# checkpoint directory nobody outside has, or recount an experiment by arm and seed. The reader of
# the public repository then follows a reference into nothing.
#
# The split this enforces:
#
#   * **Why the code is the way it is** belongs in the code, and is welcome publicly. "A plain
#     graph convolution applies one weight matrix to a node and its neighbours alike, so a
#     country's own influence is attenuated in proportion to its degree" explains a design and
#     needs no private context.
#   * **Experiment bookkeeping** -- run names, seeds, Elo deltas, checkpoint paths, section
#     citations into the research log -- belongs in `research/`, which is not published.
#
# So this does not forbid explanation. It forbids *references that dangle* and *bookkeeping*.
#
# Usage:
#   check_public_hygiene.sh                # check the working tree
#   check_public_hygiene.sh <tree-ish>     # check a git tree (what publish_snapshot.sh does)

set -euo pipefail

TREE="${1:-}"

# Paths stripped by publish_snapshot.sh. Kept in sync by `test_public_hygiene.py`, which reads
# both files and fails if they disagree -- the two lists drifting apart is how this stops working.
EXCLUDED_PREFIXES='^(Dockerfile|data/|research/|checkpoints|replays|\.claude/|docs/|\.agents/|tools/scripts/publish_snapshot\.sh|tools/scripts/check_public_hygiene\.sh|tests/training/test_public_hygiene\.py)'

# Files allowed to name the stripped layout, with the reason.
#   .gitignore  -- must name data/ and .claude/ to ignore them; that is its job
#   .gitmodules -- records the submodule path
ALLOWLIST='^(\.gitignore|\.gitmodules)$'

# Strings that are legitimate in published code, exempted individually rather than by file, so a
# *new* private reference in the same file is still caught.
#
# Two kinds:
#   * runtime paths the published code genuinely reads and writes -- a public user has these
#     directories too, and the code has to name them to work;
#   * the naming convention's own synthetic example, which any valid run name matches by
#     construction.
ALLOWED_LINE='(coldwar_net_bc[.]pt|coldwar_net_[^ ]*warmup[.]pt|warmup_regenerated[.]jsonl[.]gz|datasets/ts_replayer|snapshot_0s[.]pt|data/checkpoints/run_|E9-99-01|/api/replays/|replays/ is a symlink|run_x|docs/strategy|docs/card_strategies|scratch/|data/replays|default: data/checkpoints)'

# Each rule is  <name><TAB><regex><TAB><why>.
#
# Tab-separated, not pipe-separated: `|` is regex alternation, and splitting rules on it silently
# truncated every multi-branch pattern to its first branch. The symptom was `grep: Unmatched (`
# and a check that passed the patterns it had quietly broken.
RULES=(
  "research-log	(^|[^a-zA-Z0-9_-])research/	cites the research log, which is not published"
  "docs-dir	(^|[^a-zA-Z_])docs/	cites docs/, which is not published"
  "claude-dir	[.]claude/	cites the private agent directory"
  "checkpoint-path	(data/)?(checkpoints|datasets|replays)/	names a checkpoint, dataset or replay a reader cannot have"
  "snapshot-file	snapshot_[0-9]+(steps|s)?[.]pt	names a specific training snapshot file"
  "run-shortname	[Ee][0-9]-[0-9]{2}-[0-9]{2}	names a private run by its nomenclature short name"
  "run-dirname	(p1_[a-z_]+|arm_[A-Z][0-9A-Za-z_]*)	names a private run directory"
  "arm-label	arm [A-Z][0-9]?\b	recounts an experiment arm"
  "elo-delta	[+-][0-9]+ Elo\b	quotes a measured Elo delta, which is experiment bookkeeping"
)

fail=0
report() {
  if [ "$fail" -eq 0 ]; then
    echo "" >&2
    echo "Public hygiene check FAILED. The public snapshot must not refer to what it omits." >&2
    echo "" >&2
  fi
  fail=1
  printf '  %-16s %s\n' "[$1]" "$2" >&2
}

if [ -n "$TREE" ]; then
    mapfile -t FILES < <(git ls-tree -r --name-only "$TREE")
    read_file() { git cat-file blob "${TREE}:$1" 2>/dev/null || true; }
else
    mapfile -t FILES < <(git ls-files)
    read_file() { cat "$1" 2>/dev/null || true; }
fi

checked=0
for f in "${FILES[@]}"; do
    [[ "$f" =~ $EXCLUDED_PREFIXES ]] && continue
    [[ "$f" =~ $ALLOWLIST ]] && continue
    # Binary and asset files carry no prose.
    case "$f" in *.png|*.jpg|*.pdf|*.ico|*.gz|*.so|*.npz|*.npy|*.pt|*.woff|*.woff2|*.ttf) continue;; esac
    content="$(read_file "$f")"
    [ -z "$content" ] && continue
    checked=$((checked + 1))
    for rule in "${RULES[@]}"; do
        # No printf '%b' here: it would read the `\b` word-boundary in a regex as a
        # backspace character, and those two rules would then match nothing at all.
        IFS=$'\t' read -r name regex why <<< "$rule"
        while IFS= read -r hit; do
            [ -z "$hit" ] && continue
            line="${hit%%:*}"
            text="${hit#*:}"
            printf '%s' "$text" | grep -qE "$ALLOWED_LINE" && continue
            text="$(printf '%s' "$text" | sed 's/^[[:space:]]*//' | cut -c1-90)"
            report "$name" "$f:$line  $text    <-- $why"
        done < <(printf '%s\n' "$content" | grep -nE "$regex" || true)
    done
done

if [ "$fail" -ne 0 ]; then
    echo "" >&2
    echo "Fix by moving the bookkeeping into research/ and leaving the reasoning behind." >&2
    echo "A docstring may say *why* the code is shaped this way; it may not cite the log," >&2
    echo "name a run or checkpoint, or quote a result." >&2
    exit 1
fi

echo "--> Public hygiene: ${checked} files carry no dangling reference."
