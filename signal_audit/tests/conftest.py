"""The repo root on the path, so the tests import signal_audit as a package."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
