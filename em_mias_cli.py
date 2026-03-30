#!/usr/bin/env python3
"""Convenience launcher for EM-MIAs CLI.

Allows running EM-MIAs without relying on `python -m em_mias.cli` module discovery.
"""

from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from em_mias.cli import main


if __name__ == "__main__":
    main()
