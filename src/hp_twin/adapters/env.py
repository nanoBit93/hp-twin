"""Minimaler .env-Loader (ohne externe Abhängigkeit). Secrets kommen aus der gitignored .env."""

from __future__ import annotations

import os
from pathlib import Path


def load_dotenv(path: str | Path = ".env") -> dict[str, str]:
    """Liest KEY=VALUE-Zeilen, setzt sie in os.environ (ohne vorhandene zu überschreiben)."""
    p = Path(path)
    values: dict[str, str] = {}
    if not p.exists():
        return values
    for raw in p.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, val = line.partition("=")
        key, val = key.strip(), val.strip().strip('"').strip("'")
        values[key] = val
        os.environ.setdefault(key, val)
    return values
