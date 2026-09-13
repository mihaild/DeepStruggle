#!/usr/bin/env bash
# tools/scripts/publish_snapshot.sh
#
# Generates a clean public snapshot of master on the main branch.
# Excludes data/, research/, Dockerfile, root symlinks, and this script itself.
# Keeps CLAUDE.md and .agents/.
# Labels the source commit on master with an annotated tag, while keeping
# main completely free of any master commit metadata or hashes.

set -euo pipefail

SOURCE_BRANCH="${SOURCE_BRANCH:-master}"
TARGET_BRANCH="${TARGET_BRANCH:-main}"

# Clean commit message for main (public facing, no master metadata)
PUBLIC_MSG="${1:-"Release: $(date +%Y-%m-%d)"}"

# Label tag to mark master
DATE_TAG="$(date +%Y-%m-%d)"
TAG_NAME="${2:-"published/${DATE_TAG}"}"

SCRIPT_REL_PATH="tools/scripts/publish_snapshot.sh"

echo "================================================================================"
echo " Publishing Snapshot from '${SOURCE_BRANCH}' to '${TARGET_BRANCH}'"
echo " Public Commit Message: '${PUBLIC_MSG}'"
echo " Master Label Tag:      '${TAG_NAME}'"
echo "================================================================================"

# 1. Capture source commit on master
if ! git rev-parse --verify --quiet "refs/heads/${SOURCE_BRANCH}" >/dev/null; then
    echo "ERROR: source branch '${SOURCE_BRANCH}' does not exist." >&2
    exit 1
fi
SOURCE_COMMIT=$(git rev-parse "${SOURCE_BRANCH}")
SOURCE_SHORT=$(git rev-parse --short "${SOURCE_BRANCH}")

# 2. Build clean tree in a temporary index
TMP_INDEX=$(mktemp)
trap 'rm -f "$TMP_INDEX"' EXIT

GIT_INDEX_FILE="$TMP_INDEX" git read-tree "$SOURCE_COMMIT"

# 3. Strip all excluded paths
#
# Every path here is one this repository must not publish. The strip used to run as
#
#     git rm -r --cached --ignore-unmatch ... 2>/dev/null || true
#
# which suppressed the error *and* forced success: a renamed path silently stopped being
# stripped and its contents went public with nothing said. `ts_engine.pyi` had already drifted
# that way -- the stub became the package `bindings/ts_engine/` and the entry matched nothing --
# which is how the failure mode announces itself, i.e. not at all. So the removal is checked
# against the written tree below, and a survivor is fatal.
EXCLUDED_PATHS=(
    Dockerfile
    data
    research
    checkpoints
    replays
    .claude
    docs
    .agents
    "$SCRIPT_REL_PATH"
)

echo "--> Stripping excluded paths..."
GIT_INDEX_FILE="$TMP_INDEX" git rm -r --cached --ignore-unmatch --quiet "${EXCLUDED_PATHS[@]}"

TREE=$(GIT_INDEX_FILE="$TMP_INDEX" git write-tree)
echo "--> Clean tree created: $TREE"

# 3b. Prove it. --ignore-unmatch cannot tell "already absent" from "no longer matches", so the
# only trustworthy check is what the tree actually contains.
echo "--> Verifying the published tree..."
TREE_FILES=$(git ls-tree -r --name-only "$TREE")
SURVIVORS=()
for path in "${EXCLUDED_PATHS[@]}"; do
    if printf '%s\n' "$TREE_FILES" | grep -qE "^${path}(/|$)"; then
        SURVIVORS+=("$path")
    fi
done
if [ ${#SURVIVORS[@]} -gt 0 ]; then
    echo "" >&2
    echo "ERROR: these excluded paths are still in the tree that would be published:" >&2
    printf '  %s\n' "${SURVIVORS[@]}" >&2
    echo "" >&2
    echo "Nothing has been published and no ref was moved. Fix the exclude list and rerun." >&2
    exit 1
fi

# A tree that lost everything is also a failure, and an empty commit on main would look fine.
FILE_COUNT=$(printf '%s\n' "$TREE_FILES" | grep -c . || true)
if [ "$FILE_COUNT" -lt 100 ]; then
    echo "ERROR: the published tree has only ${FILE_COUNT} files, which is not a whole repository." >&2
    echo "Nothing has been published and no ref was moved." >&2
    exit 1
fi

# The things a public checkout is useless without. A typo in the exclude list that took one of
# these out would otherwise publish quietly.
for required in README.md LICENSE requirements.txt CMakeLists.txt engine bindings tools; do
    if ! printf '%s\n' "$TREE_FILES" | grep -qE "^${required}(/|$)"; then
        echo "ERROR: '${required}' is missing from the published tree." >&2
        echo "Nothing has been published and no ref was moved." >&2
        exit 1
    fi
done
echo "--> ${FILE_COUNT} files, no excluded path survived."

# 3c. Absence of the excluded paths is only half of it. A file that survives can still point at
# the research log, name a checkpoint nobody outside has, or recount an experiment by arm and
# seed -- and the public reader follows that reference into nothing. This checks the contents of
# what is about to be published, and is fatal for the same reason the path check is.
echo "--> Checking public hygiene..."
"$(dirname "${BASH_SOURCE[0]}")/check_public_hygiene.sh" "$TREE"

# 4. Chain to previous main commit (if any)
PARENT_ARGS=()
PARENT=$(git rev-parse --verify --quiet "refs/heads/${TARGET_BRANCH}" || true)
if [ -n "$PARENT" ]; then
    PARENT_ARGS=(-p "$PARENT")
    echo "--> Chaining to previous snapshot on ${TARGET_BRANCH}: ${PARENT:0:7}"
else
    echo "--> Initializing ${TARGET_BRANCH} with orphan root commit"
fi

# 5. Create clean commit on main (strictly the public message)
MAIN_COMMIT=$(git commit-tree "$TREE" "${PARENT_ARGS[@]}" -m "$PUBLIC_MSG")
git update-ref "refs/heads/${TARGET_BRANCH}" "$MAIN_COMMIT"
echo "--> Created snapshot commit on ${TARGET_BRANCH}: ${MAIN_COMMIT:0:7}"

# 6. Label master with an annotated tag recording the main commit
if git rev-parse "$TAG_NAME" >/dev/null 2>&1; then
    TAG_NAME="${TAG_NAME}_$(date +%H%M%S)"
fi

git tag -a "$TAG_NAME" "$SOURCE_COMMIT" -m "main: ${MAIN_COMMIT} (${PUBLIC_MSG})"
echo "--> Labeled ${SOURCE_BRANCH} at ${SOURCE_SHORT} with tag '${TAG_NAME}'"

echo "================================================================================"
echo " Snapshot Complete!"
echo "   Branch '${TARGET_BRANCH}': commit ${MAIN_COMMIT:0:7} (\"${PUBLIC_MSG}\")"
echo "   Branch '${SOURCE_BRANCH}': tagged as '${TAG_NAME}'"
echo ""
echo " To publish:"
echo "   git push public ${TARGET_BRANCH}"
echo "================================================================================"
