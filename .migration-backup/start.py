#!/usr/bin/env python3
"""Backward-compatible launcher for Quantlytics.

Prefer using the installed entrypoint:
    quantlytics --mode dashboard
"""

from pathlib import Path
import runpy


if __name__ == "__main__":
    cli_path = Path(__file__).parent / "src" / "quantlytics" / "cli.py"
    runpy.run_path(str(cli_path), run_name="__main__")
