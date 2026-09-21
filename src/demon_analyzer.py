#!/usr/bin/env python3
"""DEMON Analyzer (legacy wrapper)

The original `demon_analyzer.py` contained overlapping DEMON + plotting logic.
It has been refactored into:

- src/demon_core.py      (signal-processing + detection)
- src/demon_plotting.py  (visual standard)
- src/demon_cli.py       (CLI entry point)

This wrapper forwards to the canonical CLI so existing commands keep working.
"""

from demon_cli import main

if __name__ == "__main__":
    raise SystemExit(main())
