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
    # With --direct the reachable endpoint is public_ipaddr:direct_port_start.
    # ssh_host:ssh_port is the PROXY, and it timed out during banner exchange on a host that was
    # plainly `running` -- which is why an otherwise healthy instance looked unreachable for
    # seven minutes. Fall back to the proxy only when no direct port is published.
    read -r ST H P MSG <<< "$(echo "$INFO" | "$PY" -c '
import json,sys
try: d=json.load(sys.stdin)
except Exception: print("? ? ? ?"); raise SystemExit
d=d[0] if isinstance(d,list) and d else d
m=str(d.get("status_msg") or "").strip().replace(chr(10)," ")[:70] or "-"
port = d.get("direct_port_start")
if port and port != -1:
    host = d.get("public_ipaddr") or "?"
else:
    host, port = d.get("ssh_host","?"), d.get("ssh_port","?")
print(d.get("actual_status","?"), host or "?", port or "?", m)')"
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
# Both, not just cmake: the pytorch runtime image ships cmake through conda but has no
# compiler, so guarding on cmake alone skipped build-essential and the configure step
# failed with "CMAKE_CXX_COMPILER not set".
if ! command -v cmake >/dev/null || ! command -v g++ >/dev/null; then
    apt-get update -qq
    apt-get install -y -qq cmake build-essential >/dev/null 2>&1
fi
pip install --break-system-packages -q nanobind 'numpy>=2.0' tensorboard
# --break-system-packages because the image's python is PEP 668 externally-managed and pip
# refuses otherwise. NOT suppressed and NOT `|| true`: the first attempt hid this failure, and a
# missing nanobind makes CMake skip the bindings target *silently* -- the engine core, tests and
# benchmark all build, `cmake --build` exits 0, and only the python module is absent. The import
# error that follows names ts_engine, which points nowhere near the real cause.
python -c "import nanobind" || { echo "nanobind still missing after install" >&2; exit 1; }
cmake -B build/release -S . -DPython_EXECUTABLE="$(command -v python)" >/dev/null
cmake --build build/release -j "$(nproc)" > /tmp/build.log 2>&1 \
    || { tail -30 /tmp/build.log >&2; exit 1; }
# Assert the artefact exists. `cmake --build` exits 0 when the bindings target was
# never configured, so a zero exit code does not mean the module was produced.
ls build/release/ts_engine*.so >/dev/null 2>&1 \
    || { echo "no ts_engine*.so produced -- bindings target was skipped" >&2; exit 1; }
PYTHONPATH=.:build/release python -c "
import ts_engine, torch, tools.lib, os
print('engine OK | torch', torch.__version__, '|', torch.cuda.get_device_name(0),
      '| cores', os.cpu_count())"
REMOTE

# One file per host. The default keeps the single-host workflow unchanged; provisioning several
# hosts in parallel must override it, or each overwrites the last and every run is launched onto
# whichever host finished provisioning last.
HOSTFILE="${HOSTFILE:-$ROOT/../.vast_host}"
cat > "$HOSTFILE" <<EOF
INSTANCE=$INSTANCE
HOST=$HOST
PORT=$PORT
SSH="ssh -p $PORT -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null root@$HOST"
EOF
echo "==> READY. Details written to $HOSTFILE"
echo "==> instance $INSTANCE is STILL RUNNING and billing. Destroy when done:"
echo "==>   $VAST destroy instance $INSTANCE -y"
