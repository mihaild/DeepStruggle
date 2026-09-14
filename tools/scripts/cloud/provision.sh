#!/usr/bin/env bash
# Rent a host, prepare it, verify it, and LEAVE IT RUNNING.
#
#   provision.sh <offer-id>
#
# The difference from bench_gpu.sh, and the reason it exists: setup is the whole problem with
# rented capacity here. A 3GB image pull plus an engine build costs ten to fifteen minutes, and
# bench_gpu.sh threw that away after a two-minute measurement -- two attempts produced zero steps.
# Amortising it over many runs is the only way this pays: at ~$0.27/hr, fifteen minutes of setup
# is 1.4% overhead across ten 1.8h arms, against 100% overhead for one-shot rentals.
#
# The image is pytorch/pytorch:2.13.0-cuda13.0-cudnn9-runtime -- 3.0GB rather than the 6.2GB the
# previous approach downloaded, and it carries *exactly* the torch this project runs locally
# (2.13.0+cu130), which removes a version difference between local and rented results. Only cmake
# and a compiler are added, since the C++ engine links no CUDA.
#
# **Does NOT destroy the instance.** That is the point. `leak_guard.sh` is the safety net; the
# cleanup trap is not sufficient on its own, having failed once under `pkill -TERM` while the
# script was blocked in ssh.
set -euo pipefail

OFFER="${1:?usage: provision.sh <offer-id>}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/../../.." && pwd)"
VAST="$HERE/vast.sh"
PY=/workspace/.venv/bin/python
IMAGE="${IMAGE:-pytorch/pytorch:2.13.0-cuda13.0-cudnn9-runtime}"
MAX_MIN="${MAX_MIN:-25}"

REPO="${IMAGE%%:*}"; TAG="${IMAGE##*:}"
"$PY" - "$REPO" "$TAG" <<'PYEOF' || exit 1
import json, sys, urllib.request
repo, tag = sys.argv[1], sys.argv[2]
try:
    with urllib.request.urlopen(
            f"https://hub.docker.com/v2/repositories/{repo}/tags/{tag}", timeout=20) as r:
        print(f"==> image {repo}:{tag} exists ({json.load(r)['full_size']/1e9:.1f} GB)")
except Exception as e:
    print(f"!! image {repo}:{tag} not found ({e}) -- refusing to rent", file=sys.stderr)
    sys.exit(1)
PYEOF

echo "==> creating instance from offer $OFFER"
CREATE=$("$VAST" create instance "$OFFER" --image "$IMAGE" --disk 60 --ssh --direct --raw)
INSTANCE=$(echo "$CREATE" | "$PY" -c 'import json,sys; print(json.load(sys.stdin).get("new_contract",""))')
[ -n "$INSTANCE" ] || { echo "could not create: $CREATE" >&2; exit 1; }
echo "==> instance $INSTANCE  (destroy with: $VAST destroy instance $INSTANCE -y)"

HOST=""; PORT=""
for _ in $(seq 1 $((MAX_MIN * 4))); do
    INFO=$("$VAST" show instance "$INSTANCE" --raw 2>/dev/null || echo '{}')
    read -r ST H P MSG <<< "$(echo "$INFO" | "$PY" -c '
import json,sys
try: d=json.load(sys.stdin)
except Exception: print("? ? ? ?"); raise SystemExit
d=d[0] if isinstance(d,list) and d else d
m=str(d.get("status_msg") or "").strip().replace(chr(10)," ")[:70] or "-"
print(d.get("actual_status","?"), d.get("ssh_host","?"), d.get("ssh_port","?"), m)')"
    echo "    [$(date +%H:%M:%S)] $ST  $MSG"
    if [ "$ST" = "running" ] && [ "$H" != "?" ] && [ "$H" != "None" ]; then
        if ssh -p "$P" -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null \
               -o ConnectTimeout=10 "root@$H" true 2>/dev/null; then
            HOST=$H; PORT=$P; break
        fi
    fi
    sleep 15
done
[ -n "$HOST" ] || { echo "never became reachable; instance $INSTANCE left for inspection" >&2; exit 1; }
echo "==> ssh up: root@$HOST:$PORT"

SSH=(ssh -p "$PORT" -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null "root@$HOST")
"${SSH[@]}" "mkdir -p /workspace/ts"
tar czf - -C "$ROOT" --exclude './.git' --exclude './.venv' --exclude './build' \
    --exclude './data' --exclude './external' --exclude './__pycache__' \
    --exclude './.triton_cache' --exclude './web/ui/node_modules' . \
    | "${SSH[@]}" "tar xzf - -C /workspace/ts"
echo "==> source shipped"

"${SSH[@]}" bash -s <<'REMOTE'
set -euo pipefail
cd /workspace/ts
export DEBIAN_FRONTEND=noninteractive
python -c 'import sys; assert sys.version_info >= (3,12), f"python {sys.version} <3.12: this codebase uses PEP 701 f-strings"'
command -v cmake >/dev/null || { apt-get update -qq; apt-get install -y -qq cmake build-essential >/dev/null; }
pip install -q nanobind 'numpy>=2.0' tensorboard 2>&1 | tail -1 || true
cmake -B build/release -S . -DPython_EXECUTABLE="$(command -v python)" >/dev/null
cmake --build build/release -j "$(nproc)" >/dev/null
PYTHONPATH=.:build/release python -c "
import ts_engine, torch, tools.lib, os
print('engine OK | torch', torch.__version__, '|', torch.cuda.get_device_name(0),
      '| cores', os.cpu_count())"
REMOTE

cat > "$ROOT/../.vast_host" <<EOF
INSTANCE=$INSTANCE
HOST=$HOST
PORT=$PORT
SSH="ssh -p $PORT -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null root@$HOST"
EOF
echo "==> READY. Details written to ../.vast_host"
echo "==> instance $INSTANCE is STILL RUNNING and billing. Destroy when done:"
echo "==>   $VAST destroy instance $INSTANCE -y"
