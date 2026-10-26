"""Audit of the gold signals tables. See README.md.

The modules read the daily caches of src/features and the profiling of eda as plain modules.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
for folder in ("src/features", "eda"):
    if str(ROOT / folder) not in sys.path:
        sys.path.insert(0, str(ROOT / folder))
