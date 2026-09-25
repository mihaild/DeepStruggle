"""Run a TypeScript test entry from tests/web/js under node: bundle it with the esbuild that ships
with the web UI's toolchain, then execute it. Used by the tests that hold the page's TypeScript to
what the Python stack does."""
from __future__ import annotations

import json
import os
import shutil
import subprocess
from typing import Any, Dict

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
UI = os.path.join(REPO, "web", "ui")
PUBLIC = os.path.join(UI, "public") + os.sep
ESBUILD = os.path.join(UI, "node_modules", ".bin", "esbuild")


def run_ts(entry: str, out_dir: str, *args: str, timeout: int = 600) -> Dict[str, Any]:
    assert os.path.exists(ESBUILD), "web/ui dependencies missing: cd web/ui && npm ci"
    assert os.path.exists(os.path.join(PUBLIC, "engine", "ts_engine.mjs")), (
        "no WebAssembly engine: build it with tools/scripts/build_web.sh --engine")
    node = shutil.which("node")
    assert node, "node is needed to run the page's TypeScript"
    bundle = os.path.join(out_dir, os.path.basename(entry).replace(".ts", ".mjs"))
    built = subprocess.run(
        [ESBUILD, os.path.join(REPO, "tests", "web", "js", entry), "--bundle", "--platform=node",
         "--format=esm", "--target=node20", f"--outfile={bundle}", "--log-level=error",
         # The page's modules read Vite's import.meta.env; under node there is none.
         "--define:import.meta.env.BASE_URL=\"/\""],
        capture_output=True, text=True, cwd=UI)
    assert built.returncode == 0, built.stderr
    out = subprocess.run([node, bundle, *args], capture_output=True, text=True, timeout=timeout)
    assert out.returncode == 0, out.stderr
    return json.loads(out.stdout.strip().splitlines()[-1])
