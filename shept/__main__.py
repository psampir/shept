"""Allow running Shept as `python3 -m shept`."""

from __future__ import annotations

import sys

#: Printed when PyGObject is unavailable. `pip install shept` alone is not
#: enough: `gi` comes from the system packages, never from PyPI.
MISSING_GTK = (
    "Shept needs GTK 4, libadwaita and PyGObject (the 'gi' module), which are "
    "installed by your distribution, not with pip.\n"
    "See https://github.com/psampir/shept#requirements for the command your "
    "distribution needs."
)


def main() -> int:
    try:
        import gi  # noqa: F401
    except ImportError:
        print(MISSING_GTK, file=sys.stderr)
        return 1

    from shept.app import main as run_app

    return run_app()


if __name__ == "__main__":
    sys.exit(main())
