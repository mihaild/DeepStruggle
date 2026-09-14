#!/usr/bin/env bash
# Thin wrapper around the vast.ai CLI.
#
# Invoked through `python -c` rather than the `vastai` console script on purpose: the venv's bin
# scripts carry an absolute shebang from wherever the venv was first created, so they are broken
# in this checkout (the same reason CLAUDE.md says to use `python -m pytest`).
#
#   export VAST_API_KEY=...            # or put it in ~/.vast_api_key
#   tools/scripts/cloud/vast.sh search offers '...'
#
# Common calls, with the filter this workload actually needs -- see README.md for why the CPU
# floor matters (the engine is a C++ simulator on the host, not on the GPU):
#
#   vast.sh search offers 'gpu_name=RTX_4090 num_gpus=1 cpu_cores_effective>=8 \
#       cpu_ram>=32 disk_space>=100 reliability>0.98 rentable=true' -o 'dph+'
#   vast.sh create instance <ID> --image <image> --disk 100 --ssh --direct
#   vast.sh show instances
#   vast.sh destroy instance <ID>
set -euo pipefail
PY="${PY:-/workspace/.venv/bin/python}"
exec "$PY" -c 'from vastai.cli.main import main; main()' "$@"
