#!/usr/bin/env python3
"""Massive Parallel Tournament & Elo Rating Benchmark (Vectorized Batch Simulation)."""

import os
import sys

_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _root not in sys.path:
    sys.path.insert(0, _root)

from scripts.massive_tournament import main

if __name__ == "__main__":
    main()
