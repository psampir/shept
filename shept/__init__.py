"""Shept: a GTK4 + libadwaita frontend for whisper-cli.

This module must stay importable without GTK: the test suite runs on machines
where PyGObject is not installed. Everything that needs `gi` lives in
`shept.app`.
"""

from __future__ import annotations

__version__ = "0.2.0"

APP_ID = "io.github.psampir.Shept"
