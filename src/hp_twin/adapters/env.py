"""Minimaler .env-Loader (ohne externe Abhängigkeit). Secrets kommen aus der gitignored .env."""

from __future__ import annotations

import os
from pathlib import Path


def load_dotenv(
    path: str | Path = ".env",
    prefixes: tuple[str, ...] | None = None,
) -> dict[str, str]:
    """Liest KEY=VALUE-Zeilen; setzt hp-twin-relevante Keys in os.environ (setdefault).

    `prefixes=None` setzt alle erfassten Keys; mit Prefix-Liste (z. B. ("HA_",)) lädt
    der Aufrufer nur seine Namen — eine gemischte Dev-.env (HA-Zugang + Remote-API-
    Token) vermischt sich sonst und das API-Token färbt unbeabsichtigt die lokale
    API (security.api_token) bzw. umgekehrt.
    """
    p = Path(path)
    values: dict[str, str] = {}
    if not p.exists():
        return values
    for raw in p.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, val = line.partition("=")
        key = key.strip()
        val = val.strip().strip('"').strip("'")
        values[key] = val
        if prefixes is None or key.startswith(prefixes):
            os.environ.setdefault(key, val)
    return values
