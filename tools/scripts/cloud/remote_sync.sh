#!/usr/bin/env bash
# Bring a remote run home. Metrics continuously, checkpoints as they appear.
#
#   remote_sync.sh <ssh-target> <run-name> [--once] [--all-resume]
#
# RESUME_STRIDE=<steps> sets which step-tagged resume files come home (default 80000000, 0 for
# none); --all-resume brings every one.
#
# Sizing, measured on E3-17-22 (160M steps, 34 snapshots): a whole run directory is 2.0GB, of
# which the resume files are 1.6GB (48MB each) and the snapshots 442MB (13MB each). Metrics and
# TensorBoard together are 25MB.
#
# So this brings home the metrics, the TensorBoard tree, the hardware record, every snapshot,
# the newest resume_state.pt, and a step-tagged resume roughly every RESUME_STRIDE steps.
#
# Resume files are not only for restarting an interrupted host -- they are what lets an
# experiment branch from a partial result, which is how the current P10/P11 arms were started
# from E3-17-22's 80M. Bringing every one home is wasteful at 48MB each; bringing none means
# fetching a branch point by hand later. Every 80M is the useful granularity for branching.
#
# **Works without rsync**, which is absent and not installable on the workstation this was
# written on. The tar-over-ssh fallback transfers at file granularity, fetching only files not
# already present locally, which suits an append-only directory. rsync is preferred when
# available because it resumes partial transfers.
set -euo pipefail

TARGET="${1:?usage: remote_sync.sh <ssh-target> <run-name> [--once] [--all-resume]}"

# `user@host:port` -> ssh -p port user@host. Every Vast host is reached on a high port, and a bare
# `ssh "$TARGET"` silently tries 22. A bare user@host or an ssh-config alias is unchanged.
SSH_PORT=""
case "$TARGET" in
    *:*) SSH_PORT="${TARGET##*:}"; TARGET="${TARGET%:*}" ;;
esac
SSH_BASE=(ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null)
[ -n "$SSH_PORT" ] && SSH_BASE+=(-p "$SSH_PORT")
# rsync takes the port through -e, not on the ssh argv it builds itself.
RSYNC_RSH="ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null"
[ -n "$SSH_PORT" ] && RSYNC_RSH="$RSYNC_RSH -p $SSH_PORT"
NAME="${2:?missing run name}"
shift 2

ONCE=0
ALL_RESUME=0
for arg in "$@"; do
    case "$arg" in
        --once) ONCE=1 ;;
        --all-resume) ALL_RESUME=1 ;;
        *) echo "unknown option: $arg" >&2; exit 2 ;;
    esac
done

RESUME_STRIDE="${RESUME_STRIDE:-80000000}"
REMOTE_DIR="${REMOTE_DIR:-/workspace/ts}"
LOCAL_ROOT="${LOCAL_CHECKPOINTS:-/workspace/data/checkpoints}"
RSRC="$REMOTE_DIR/runs/$NAME"
DST="$LOCAL_ROOT/$NAME"
INTERVAL="${SYNC_INTERVAL:-300}"

mkdir -p "$DST"

# Files that grow in place and so are re-fetched every pass. Everything else is immutable once
# written and only fetched when absent locally.
MUTABLE=(training_metrics.jsonl metadata.json hardware.json train.log resume_state.pt)

# Which step-tagged resume files to keep, from a list of filenames on stdin.
#
# The stride CANNOT be a modulo test. Real step counts are 5046272, 40042496, 80019456 -- never
# exact multiples of anything -- so `steps % stride == 0` keeps nothing at all. The rule is the
# same one the trainer uses to write them: walk in step order and keep one whenever it is at
# least `stride` past the last kept.
select_resumes() {
    awk -v stride="$RESUME_STRIDE" -v all="$ALL_RESUME" '
        {
            s = $0
            sub(/^resume_/, "", s); sub(/steps\.pt$/, "", s)
            if (s !~ /^[0-9]+$/) next
            n++; name[n] = $0; val[n] = s + 0
        }
        END {
            if (all == 1) { for (i = 1; i <= n; i++) print name[i]; exit }
            if (stride + 0 == 0) exit
            for (i = 1; i <= n; i++) ord[i] = i
            for (i = 1; i <= n; i++)
                for (j = i + 1; j <= n; j++)
                    if (val[ord[j]] < val[ord[i]]) { t = ord[i]; ord[i] = ord[j]; ord[j] = t }
            last = -1
            for (i = 1; i <= n; i++) {
                k = ord[i]
                if (last < 0 || val[k] - last >= stride) { print name[k]; last = val[k] }
            }
        }'
}

remote_resumes() {
    "${SSH_BASE[@]}" "$TARGET" "cd '$RSRC' 2>/dev/null && ls -1 resume_*steps.pt 2>/dev/null || true"
}

sync_rsync() {
    local -a filters=(
        --include 'training_metrics.jsonl' --include 'metadata.json'
        --include 'hardware.json' --include 'train.log'
        --include 'resume_state.pt' --include 'tb/***' --include 'snapshot_*.pt'
    )
    local f
    while IFS= read -r f; do
        [ -n "$f" ] && filters+=(--include "$f")
    done < <(remote_resumes | select_resumes)
    filters+=(--exclude '*')
    rsync -az --partial -e "$RSYNC_RSH" "${filters[@]}" "$TARGET:$RSRC/" "$DST/"
}

sync_tar() {
    local -a want=()
    local f

    while IFS= read -r f; do
        [ -n "$f" ] || continue
        [ -f "$DST/$f" ] || want+=("$f")
    done < <("${SSH_BASE[@]}" "$TARGET" "cd '$RSRC' 2>/dev/null && ls -1 snapshot_*.pt 2>/dev/null || true")

    while IFS= read -r f; do
        [ -n "$f" ] || continue
        [ -f "$DST/$f" ] || want+=("$f")
    done < <(remote_resumes | select_resumes)

    want+=("${MUTABLE[@]}" tb)

    # --ignore-failed-read: a run that has not written train.log or hardware.json yet must not
    # abort the whole sync.
    "${SSH_BASE[@]}" "$TARGET" "cd '$RSRC' && tar czf - --ignore-failed-read $(printf '%q ' "${want[@]}") 2>/dev/null" \
        | tar xzf - -C "$DST" 2>/dev/null || return 1
}

sync_once() {
    if command -v rsync >/dev/null 2>&1; then sync_rsync; else sync_tar; fi
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
    # A transient SSH failure must not end the loop; the run is still going.
    if sync_once; then report; else echo "SYNC-RETRY $NAME: transfer failed, retrying"; fi
    sleep "$INTERVAL"
done
