#!/usr/bin/env bash
# Bring a remote run home. Metrics continuously, checkpoints as they appear.
#
#   remote_sync.sh <ssh-target> <run-name> [--once] [--with-resume]
#
# Sizing, measured on E3-17-22 (160M steps, 34 snapshots): a whole run directory is 2.0GB, of
# which the resume files are 1.6GB (48MB each) and the snapshots 442MB (13MB each). Metrics and
# TensorBoard together are 25MB.
#
# So by default this brings home the metrics, the TensorBoard tree, the hardware record and every
# snapshot -- about 500MB per run, and everything a tournament or an offline probe needs. The
# resume files stay remote unless --with-resume is given: they exist to restart an interrupted
# run *on that host*, and only the newest is ever useful.
#
# **Works without rsync.** rsync is not installed on every machine this has to run from (it is
# absent, and not installable, on the workstation this was written on), so there is a tar-over-ssh
# fallback. It transfers at file granularity -- fetching only files not already present locally --
# which suits an append-only directory of snapshots. rsync is still preferred when available
# because it resumes partial transfers.
set -euo pipefail

TARGET="${1:?usage: remote_sync.sh <ssh-target> <run-name> [--once] [--with-resume]}"
NAME="${2:?missing run name}"
shift 2

ONCE=0
WITH_RESUME=0
for arg in "$@"; do
    case "$arg" in
        --once) ONCE=1 ;;
        --with-resume) WITH_RESUME=1 ;;
        *) echo "unknown option: $arg" >&2; exit 2 ;;
    esac
done

REMOTE_DIR="${REMOTE_DIR:-/workspace/ts}"
LOCAL_ROOT="${LOCAL_CHECKPOINTS:-/workspace/data/checkpoints}"
RSRC="$REMOTE_DIR/runs/$NAME"
DST="$LOCAL_ROOT/$NAME"
INTERVAL="${SYNC_INTERVAL:-300}"

mkdir -p "$DST"

# Files that are always re-fetched because they grow in place, versus files that are immutable
# once written and so only need fetching if absent locally.
MUTABLE=(training_metrics.jsonl metadata.json hardware.json train.log)

sync_rsync() {
    local -a filters=(
        --include 'training_metrics.jsonl' --include 'metadata.json'
        --include 'hardware.json' --include 'train.log'
        --include 'tb/***' --include 'snapshot_*.pt'
    )
    [ "$WITH_RESUME" = "1" ] && filters+=(--include 'resume_*.pt')
    filters+=(--exclude '*')
    rsync -az --partial "${filters[@]}" "$TARGET:$RSRC/" "$DST/"
}

sync_tar() {
    # 1. What immutable files exist remotely?
    local pattern='snapshot_*.pt'
    [ "$WITH_RESUME" = "1" ] && pattern='snapshot_*.pt resume_*.pt'
    local remote_list
    remote_list=$(ssh "$TARGET" "cd '$RSRC' 2>/dev/null && ls -1 $pattern 2>/dev/null || true")

    # 2. Which of them are missing here?
    local -a want=()
    local f
    while IFS= read -r f; do
        [ -n "$f" ] || continue
        [ -f "$DST/$f" ] || want+=("$f")
    done <<< "$remote_list"

    # 3. The growing files, every time; plus the TensorBoard tree, which is small.
    want+=("${MUTABLE[@]}" tb)

    # `tar --ignore-failed-read` because a run that has not written train.log or hardware.json yet
    # must not abort the whole sync.
    ssh "$TARGET" "cd '$RSRC' && tar czf - --ignore-failed-read $(printf '%q ' "${want[@]}") 2>/dev/null" \
        | tar xzf - -C "$DST" 2>/dev/null || return 1
}

sync_once() {
    if command -v rsync >/dev/null 2>&1; then
        sync_rsync
    else
        sync_tar
    fi
}

report() {
    local steps="?"
    if [ -s "$DST/training_metrics.jsonl" ]; then
        steps=$(tail -1 "$DST/training_metrics.jsonl" \
                | python3 -c 'import json,sys; print(json.load(sys.stdin).get("total_steps","?"))' \
                2>/dev/null || echo "?")
    fi
    echo "SYNCED $NAME: $steps steps, $(du -sh "$DST" 2>/dev/null | cut -f1) local"
}

if [ "$ONCE" = "1" ]; then
    sync_once
    report
    exit 0
fi

echo "==> syncing $NAME every ${INTERVAL}s into $DST (ctrl-c to stop)"
while true; do
    # A transient SSH failure must not end the sync loop; the run is still going.
    if sync_once; then report; else echo "SYNC-RETRY $NAME: transfer failed, retrying"; fi
    sleep "$INTERVAL"
done
