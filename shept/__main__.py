"""Allow running Shept as `python3 -m shept`."""

from __future__ import annotations

import sys

from shept.app import main

if __name__ == "__main__":
    sys.exit(main())
