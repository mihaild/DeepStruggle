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
SOURCE_COMMIT=$(git rev-parse "${SOURCE_BRANCH}")
SOURCE_SHORT=$(git rev-parse --short "${SOURCE_BRANCH}")

# 2. Build clean tree in a temporary index
TMP_INDEX=$(mktemp)
trap 'rm -f "$TMP_INDEX"' EXIT

GIT_INDEX_FILE="$TMP_INDEX" git read-tree "$SOURCE_COMMIT"

# 3. Strip all excluded paths
echo "--> Stripping excluded paths..."
GIT_INDEX_FILE="$TMP_INDEX" git rm -r --cached --ignore-unmatch \
    Dockerfile \
    data \
    research \
    checkpoints \
    replays \
    ts_engine.pyi \
    .claude \
    "$SCRIPT_REL_PATH" \
    2>/dev/null || true

TREE=$(GIT_INDEX_FILE="$TMP_INDEX" git write-tree)
echo "--> Clean tree created: $TREE"

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
