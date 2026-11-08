"""Home-Assistant-REST-Adapter: Entities auflisten und Historie holen.

Nutzt nur die Standardbibliothek (urllib), damit kein weiteres Secret-Handling/Dep nötig ist.
Secrets (HA_BASE_URL, HA_TOKEN) kommen aus der gitignored .env bzw. den Umgebungsvariablen.
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from hp_twin.adapters.env import load_dotenv


class HaError(RuntimeError):
    pass


@dataclass(frozen=True)
class HaClient:
    base_url: str
    token: str
    timeout_s: float = 30.0

    @classmethod
    def from_env(cls, dotenv_path: str = ".env", timeout_s: float = 30.0) -> HaClient:
        load_dotenv(dotenv_path)
        base = os.environ.get("HA_BASE_URL", "").rstrip("/")
        token = os.environ.get("HA_TOKEN", "")
        if not base or not token:
            raise HaError("HA_BASE_URL und HA_TOKEN müssen in .env / Umgebung gesetzt sein")
        return cls(base_url=base, token=token, timeout_s=timeout_s)

    def _get(self, path: str) -> Any:
        req = urllib.request.Request(
            f"{self.base_url}{path}",
            headers={"Authorization": f"Bearer {self.token}", "Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout_s) as resp:  # noqa: S310
                return json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as e:  # pragma: no cover - netzwerkabhängig
            raise HaError(f"HTTP {e.code} bei {path}: {e.read().decode('utf-8')[:200]}") from e
        except urllib.error.URLError as e:  # pragma: no cover
            raise HaError(f"Verbindung fehlgeschlagen zu {self.base_url}: {e.reason}") from e

    def ping(self) -> str:
        return str(self._get("/api/")["message"])

    def state(self, entity_id: str) -> dict[str, Any]:
        """Einzelner Entity-State (entity_id, state, attributes)."""
        return dict(self._get(f"/api/states/{urllib.parse.quote(entity_id)}"))

    def states(self) -> list[dict[str, Any]]:
        """Alle aktuellen Entity-States (entity_id, state, attributes)."""
        return list(self._get("/api/states"))

    def post_response(self, path: str, payload: dict[str, Any]) -> Any:
        """Service-Aufruf mit Antwort (`?return_response`); liefert `service_response`."""
        sep = "&" if "?" in path else "?"
        return self._post_json(f"{path}{sep}return_response", payload)

    def _post_json(self, path: str, payload: dict[str, Any]) -> Any:
        req = urllib.request.Request(
            f"{self.base_url}{path}",
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {self.token}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout_s) as resp:  # noqa: S310
                body = json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as e:  # pragma: no cover - netzwerkabhängig
            raise HaError(f"HTTP {e.code} bei {path}: {e.read().decode('utf-8')[:200]}") from e
        except urllib.error.URLError as e:  # pragma: no cover
            raise HaError(f"Verbindung fehlgeschlagen zu {self.base_url}: {e.reason}") from e
        if isinstance(body, dict) and set(body) == {"changed_states", "service_response"}:
            return body["service_response"]
        return body

    def history(
        self, entity_ids: list[str], start: datetime, end: datetime, minimal: bool = True
    ) -> list[list[dict[str, Any]]]:
        """Historie für Entities im Zeitraum (UTC-Zeitstempel empfohlen).

        minimal=True liefert nur state (kompakt); minimal=False liefert auch attributes
        (für climate: temperature/hvac_action).
        """
        params = {"filter_entity_id": ",".join(entity_ids), "end_time": end.isoformat()}
        if minimal:
            params["minimal_response"] = ""
        query = urllib.parse.urlencode(params)
        path = f"/api/history/period/{urllib.parse.quote(start.isoformat())}?{query}"
        return list(self._get(path))
