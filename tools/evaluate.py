#!/usr/bin/env python3
"""Head-to-head Match Evaluator with side-specific win rate and loss cause breakdown."""

import os
import sys

_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _root not in sys.path:
    sys.path.insert(0, _root)

from scripts.evaluate import main

if __name__ == "__main__":
    main()
