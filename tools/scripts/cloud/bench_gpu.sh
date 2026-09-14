#!/usr/bin/env bash
# Rent one GPU, measure training throughput on it, destroy it. Prints $ per 80M steps.
#
#   bench_gpu.sh <offer-id> <label> [max-minutes]
#
# The question this answers cannot be answered from spec sheets. This workload is a small network
# (a 512-wide trunk over 3,824 inputs) plus a C++ simulator that runs on the host CPU, which is
# exactly the shape that fails to scale with headline FLOPS. A card at half the price wins only if
# it is less than twice as slow, and that ratio has to be measured.
#
# **The instance is destroyed on every exit path** -- success, failure, or interrupt. A leaked
# instance bills silently until someone notices, which is the expensive failure here.
set -euo pipefail

OFFER="${1:?usage: bench_gpu.sh <offer-id> <label> [max-minutes]}"
LABEL="${2:?missing label}"
MAX_MIN="${3:-25}"

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/../../.." && pwd)"
VAST="$HERE/vast.sh"
OUT="/workspace/data/tournaments/gpu_bench_${LABEL}.json"
# Ubuntu 24.04 for Python 3.12. The codebase uses PEP 701 f-strings (nested quotes inside an
# f-string, e.g. tools/lib/self_play.py), which are a SyntaxError on 3.11 -- and the stock
# pytorch/pytorch images ship 3.11, where the engine builds and imports fine and then the first
# `import tools.lib` fails. torch is pip-installed instead of baked in, costing ~2.5GB once.
# The tag is verified against Docker Hub before renting: a nonexistent tag leaves the
# instance stuck in `loading` with `manifest unknown` until something times out, and the
# rental bills the whole time. 12.4.1-devel-ubuntu24.04 does not exist -- Ubuntu 24.04
# images start at CUDA 12.6.
IMAGE="${IMAGE:-nvidia/cuda:12.6.3-devel-ubuntu24.04}"

INSTANCE=""
cleanup() {
    [ -n "$INSTANCE" ] || return 0
    echo "==> destroying instance $INSTANCE"
    # -y is load-bearing: `destroy instance` prompts for confirmation, and without it the
    # trap's destroy aborted while its output was redirected to /dev/null. The instance kept
    # running and kept billing, and nothing said so. Verify afterwards rather than trust it.
    "$VAST" destroy instance "$INSTANCE" -y >/dev/null 2>&1 || true
    sleep 5
    if "$VAST" show instances --raw 2>/dev/null | grep -q "\"id\": *$INSTANCE"; then
        echo "!! INSTANCE $INSTANCE STILL RUNNING -- destroy it by hand:" >&2
        echo "!!   tools/scripts/cloud/vast.sh destroy instance $INSTANCE -y" >&2
    else
        echo "==> confirmed destroyed"
    fi
}
trap cleanup EXIT INT TERM

# Verify the image tag exists before paying for a machine to discover it does not.
REPO="${IMAGE%%:*}"; TAG="${IMAGE##*:}"
/workspace/.venv/bin/python - "$REPO" "$TAG" <<'PYEOF' || exit 1
import json, sys, urllib.request
repo, tag = sys.argv[1], sys.argv[2]
try:
    with urllib.request.urlopen(
            f"https://hub.docker.com/v2/repositories/{repo}/tags/{tag}", timeout=20) as r:
        json.load(r)
    print(f"==> image {repo}:{tag} exists")
except Exception as e:
    print(f"!! image {repo}:{tag} not found ({e}) -- refusing to rent", file=sys.stderr)
    sys.exit(1)
PYEOF

echo "==> creating instance from offer $OFFER ($LABEL)"
CREATE=$("$VAST" create instance "$OFFER" --image "$IMAGE" --disk 40 --ssh --direct --raw)
INSTANCE=$(echo "$CREATE" | /workspace/.venv/bin/python -c \
    'import json,sys; d=json.load(sys.stdin); print(d.get("new_contract",""))')
[ -n "$INSTANCE" ] || { echo "could not create instance: $CREATE" >&2; exit 1; }
echo "==> instance $INSTANCE"

# Wait for ssh. A host that never comes up must time out rather than hang, or it bills forever.
SSH_URL=""
for _ in $(seq 1 "$((MAX_MIN * 4))"); do
    INFO=$("$VAST" show instance "$INSTANCE" --raw 2>/dev/null || echo '{}')
    read -r STATUS HOST PORT <<< "$(echo "$INFO" | /workspace/.venv/bin/python -c \
        'import json,sys
d=json.load(sys.stdin)
d=d[0] if isinstance(d,list) and d else d
print(d.get("actual_status",""), d.get("ssh_host",""), d.get("ssh_port",""))')"
    if [ "$STATUS" = "running" ] && [ -n "$HOST" ] && [ "$HOST" != "None" ]; then
        SSH_URL="root@${HOST}"
        if ssh -p "$PORT" -o StrictHostKeyChecking=no -o ConnectTimeout=10 \
               -o UserKnownHostsFile=/dev/null "$SSH_URL" true 2>/dev/null; then
            echo "==> ssh up at $SSH_URL:$PORT"
            break
        fi
    fi
    # Log the status each poll. Without this a failure says only "never became reachable",
    # which does not distinguish a slow image pull from a bad tag from an ssh problem -- and
    # the answer is in status_msg, which the API reports and we were discarding.
    echo "    [$(date +%H:%M:%S)] status=$STATUS host=${HOST:-none} $(echo "$INFO" | \
        /workspace/.venv/bin/python -c '
import json,sys
try: d=json.load(sys.stdin)
except Exception: raise SystemExit
d=d[0] if isinstance(d,list) and d else d
m=str(d.get("status_msg") or "").strip().replace(chr(10)," ")
print("msg=" + m[:90] if m else "")' 2>/dev/null)"
    SSH_URL=""
    sleep 15
done
[ -n "$SSH_URL" ] || { echo "instance never became reachable (see status lines above)" >&2; exit 1; }

SSH=(ssh -p "$PORT" -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null "$SSH_URL")

echo "==> shipping source (4.4MB) and building"
"${SSH[@]}" "mkdir -p /workspace/ts"
tar czf - -C "$ROOT" --exclude './.git' --exclude './.venv' --exclude './build' \
    --exclude './data' --exclude './external' --exclude './__pycache__' \
    --exclude './.triton_cache' --exclude './web/ui/node_modules' . \
    | "${SSH[@]}" "tar xzf - -C /workspace/ts"

"${SSH[@]}" bash -s <<'REMOTE'
set -euo pipefail
cd /workspace/ts
export DEBIAN_FRONTEND=noninteractive
apt-get update -qq
apt-get install -y -qq python3 python3-pip python3-venv cmake build-essential >/dev/null
PY=python3
$PY -c 'import sys; assert sys.version_info >= (3,12), sys.version' \
    || { echo "python <3.12: the codebase uses PEP 701 f-strings" >&2; exit 1; }
$PY -m venv /opt/venv
/opt/venv/bin/pip install -q --upgrade pip
/opt/venv/bin/pip install -q torch --index-url https://download.pytorch.org/whl/cu124
/opt/venv/bin/pip install -q nanobind 'numpy>=2.0' tensorboard
cmake -B build/release -S . -DPython_EXECUTABLE=/opt/venv/bin/python >/dev/null
cmake --build build/release -j "$(nproc)" >/dev/null
PYTHONPATH=.:build/release /opt/venv/bin/python -c "
import ts_engine, torch
print('engine OK | torch', torch.__version__, '|', torch.cuda.get_device_name(0))
import tools.lib  # the import that failed on 3.11
print('tools.lib OK')"
REMOTE

echo "==> benchmarking"
"${SSH[@]}" bash -s <<'REMOTE' | tee "/tmp/bench_${LABEL}.txt"
set -euo pipefail
cd /workspace/ts
PYTHONPATH=.:build/release /opt/venv/bin/python tools/scripts/bench_throughput.py \
    --num-envs 256 512 1024 --buffer-size 64 --iterations 3
REMOTE

cp "/tmp/bench_${LABEL}.txt" "${OUT%.json}.txt" 2>/dev/null || true
echo "==> done: ${OUT%.json}.txt"
