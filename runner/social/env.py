"""Tiny .env file reader.

Not python-dotenv — this project's dependency list is deliberately short
(requests, PyYAML, mcp, websocket-client), and a KEY=VALUE line reader is
about five lines of actual logic, so a new dependency isn't worth it for
this.
"""

from __future__ import annotations

from pathlib import Path


def load_env(path: Path) -> dict[str, str]:
    """Parse a KEY=VALUE-per-line file. Blank lines and lines starting with
    # are ignored. No quoting/escaping support - these files are always
    written by us or by copy-pasting a token, never hand-edited with
    special characters."""
    values: dict[str, str] = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        values[key.strip()] = value.strip()
    return values
