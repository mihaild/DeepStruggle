#!/usr/bin/env python3
"""Read `vastai show instances --raw` on stdin; print a warning per over-age instance.

Split out of leak_guard.sh rather than inlined with `python -c`, because the nested quoting
needed to build the warning string inside a shell heredoc is a reliable source of SyntaxErrors --
one of which is why this file exists.
"""
import json
import sys
import time


def main() -> int:
    limit_min = float(sys.argv[1]) if len(sys.argv) > 1 else 40.0
    try:
        data = json.load(sys.stdin)
    except Exception:
        return 0
    rows = data if isinstance(data, list) else [data]
    now = time.time()
    for inst in rows:
        if not isinstance(inst, dict):
            continue
        started = inst.get("start_date") or 0
        # An instance with no start date is suspicious rather than fine: report it.
        age_min = (now - started) / 60.0 if started else -1.0
        if age_min < 0 or age_min > limit_min:
            iid = inst.get("id")
            age = "unknown" if age_min < 0 else f"{age_min:.0f} min"
            print(f"LEAK WARNING: instance {iid} ({inst.get('gpu_name')}, "
                  f"${inst.get('dph_total', 0):.3f}/hr) running {age} -- destroy with: "
                  f"tools/scripts/cloud/vast.sh destroy instance {iid} -y")
    return 0


if __name__ == "__main__":
    sys.exit(main())
