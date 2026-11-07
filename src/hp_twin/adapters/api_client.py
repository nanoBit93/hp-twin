"""Client für die Analyse-/Dev-API einer laufenden hp-twin-Instanz (Add-on in Home Assistant).

Damit liest die Entwickler-Maschine den Zustand/die Daten der Produktivinstanz, holt
Zwischenschritt-Traces und fährt Was-wäre-wen-Szenarien – ohne lokale Messdaten oder Konfig.

Konfiguration (Reihenfolge: explizite Argumente > Env > .env):
- HP_TWIN_API_URL   Basis-URL der Instanz (z. B. http://homeassistant.local:8000)
- HP_TWIN_API_TOKEN API-Token (Add-on-Option `api_token`), optional

Nur Standardbibliothek (urllib), keine neuen Abhängigkeiten – wie die HA-/EOS-Adapter.
Alle GET-Endpunkte sind lesend; POST /api/dev/scenario(s) berechnet nur (kein Systemzustand).
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

DEFAULT_TIMEOUT_S = 180.0  # Szenarien mit Trace können einige Sekunden dauern


def _read_dotenv(path: str) -> dict[str, str]:
    """Liest KEY=VALUE-Zeilen (ohne os.environ zu verändern); Umgebung gewohnt vorgezogen."""
    p = Path(path)
    if not p.exists():
        return {}
    out: dict[str, str] = {}
    for raw in p.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, val = line.partition("=")
        out[key.strip()] = val.strip().strip('"').strip("'")
    return out


@dataclass(frozen=True)
class DevApiClient:
    """Dünner HTTP-Client auf die Dev-API (`/api/dev/*`) einer hp-twin-Instanz."""

    base_url: str
    token: str | None = None
    timeout_s: float = DEFAULT_TIMEOUT_S

    @classmethod
    def from_env(
        cls, dotenv_path: str = ".env", timeout_s: float = DEFAULT_TIMEOUT_S
    ) -> DevApiClient:
        """URL/Token aus Umgebung, ergänzt um Werte aus der .env (Umgebung hat Vorrang)."""
        dotenv = _read_dotenv(dotenv_path)
        base = os.environ.get("HP_TWIN_API_URL") or dotenv.get("HP_TWIN_API_URL") or ""
        base = base.rstrip("/")
        if not base:
            raise ValueError(
                "HP_TWIN_API_URL nicht gesetzt (in .env eintragen: URL der hp-twin-Instanz)"
            )
        token = os.environ.get("HP_TWIN_API_TOKEN") or dotenv.get("HP_TWIN_API_TOKEN") or None
        return cls(base_url=base, token=token, timeout_s=timeout_s)

    # ------------------------------------------------------------------ HTTP-Basis
    def _headers(self) -> dict[str, str]:
        h = {"Accept": "application/json"}
        if self.token:
            h["Authorization"] = "Bearer " + str(self.token)
        return h

    def get(self, path: str) -> dict[str, Any]:
        """GET-JSON (typisiert); alle Dev-Endpunkte antworten mit einem JSON-Objekt."""
        result = self._get_json(path)
        if not isinstance(result, dict):
            raise RuntimeError(f"Unerwartete Antwort (kein JSON-Objekt) bei GET {path}")
        return result

    def post(self, path: str, body: dict[str, Any] | list[dict[str, Any]]) -> dict[str, Any]:
        """POST-JSON (typisiert); Dev-Endpunkte antworten immer mit einem JSON-Objekt."""
        result = self._post_json(path, body)
        if not isinstance(result, dict):
            raise RuntimeError(f"Unerwartete Antwort (kein JSON-Objekt) bei POST {path}")
        return result

    def _get_json(self, path: str) -> Any:
        req = urllib.request.Request(f"{self.base_url}{path}", headers=self._headers())
        try:
            with urllib.request.urlopen(req, timeout=self.timeout_s) as resp:  # noqa: S310
                return json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            detail = e.read().decode("utf-8")[:300]
            raise RuntimeError(f"HTTP {e.code} bei GET {path}: {detail}") from e
        except urllib.error.URLError as e:
            raise RuntimeError(f"Verbindung fehlgeschlagen zu {self.base_url}: {e.reason}") from e

    def _post_json(self, path: str, body: dict[str, Any] | list[dict[str, Any]]) -> Any:
        data = json.dumps(body, ensure_ascii=False).encode("utf-8")
        headers = {**self._headers(), "Content-Type": "application/json"}
        req = urllib.request.Request(
            f"{self.base_url}{path}", data=data, headers=headers, method="POST"
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout_s) as resp:  # noqa: S310
                return json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            detail = e.read().decode("utf-8")[:300]
            raise RuntimeError(f"HTTP {e.code} bei POST {path}: {detail}") from e
        except urllib.error.URLError as e:
            raise RuntimeError(f"Verbindung fehlgeschlagen zu {self.base_url}: {e.reason}") from e

    # ------------------------------------------------------------------ Dev-Endpunkte
    def health(self) -> dict[str, Any]:
        return self.get("/health")

    def version(self) -> dict[str, Any]:
        return self.get("/api/version")

    def state(self) -> dict[str, Any]:
        """Startzustand 'jetzt' (Knotentemperaturen, Datenalter, Reglermodus)."""
        return self.get("/api/dev/state")

    def calibration(self) -> dict[str, Any]:
        """Aktive Kalibrierung + Modellparameter."""
        return self.get("/api/dev/calibration")

    def data(
        self,
        start: str | None = None,
        end: str | None = None,
    ) -> dict[str, Any]:
        """Datenbestand; mit start/end (UTC-ISO) ein Ausschnitt der Messreihen."""
        quoted = (
            f"{k}={urllib.parse.quote(v, safe='')}"
            for k, v in (("start", start), ("end", end))
            if v
        )
        q = "&".join(quoted)
        return self.get(f"/api/dev/data{'?' + q if q else ''}")

    def trace(
        self,
        start: str | None = None,
        horizon_h: float = 6.0,
        trace_step_s: float = 300.0,
    ) -> dict[str, Any]:
        """Zwischenschritte des unveränderten Laufs ab `start` (UTC-ISO), mit Vergleich."""
        q = urllib.parse.urlencode(
            {
                k: v
                for k, v in (
                    ("start", start),
                    ("horizon_h", horizon_h),
                    ("trace_step_s", trace_step_s),
                )
                if v
            }
        )
        return self.get(f"/api/dev/trace?{q}")

    def scenario(
        self,
        horizon_h: float = 24.0,
        start: str | None = None,
        overrides: dict[str, Any] | None = None,
        trace: bool = False,
        trace_step_s: float = 300.0,
    ) -> dict[str, Any]:
        """Ein Was-wäre-wen-Lauf.

        overrides: Eingänge (t_outdoor_c, t_outdoor_offset_k, ghi_wm2, room_setpoint_c,
        setpoint_hk1_offset_k, setpoint_dhw_offset_k, dhw_setpoint_cap_c) und Modell
        (controller={min_off_s, ...}, heat_pump={p_el_ref_w, eta_carnot, ...}, c_building_j_per_k).
        """
        body: dict[str, Any] = {
            "horizon_h": horizon_h,
            "trace": trace,
            "trace_step_s": trace_step_s,
            "overrides": overrides or {},
        }
        if start:
            body["start"] = start
        return self.post("/api/dev/scenario", body)

    def scenarios(self, items: list[dict[str, Any]]) -> dict[str, Any]:
        """Batch (1..20 Szenarien, gleicher Body wie /scenario); Antwort mit Kurzvergleich."""
        return self.post("/api/dev/scenarios", items)

    # ------------------------------------------------------------------ Komfort
    def compare_scenarios(self, items: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
        """Batch ausführen und je Szenario die Kennzahlen extrahieren (für schnelle Übersicht)."""
        res = self.scenarios(items)
        out: dict[str, dict[str, Any]] = {}
        for r in res.get("results", []):
            s = r.get("summary", {})
            out[str(r.get("index"))] = {
                "energy_kwh": s.get("energy_kwh"),
                "n_cycles": s.get("n_cycles"),
                "overrides": s.get("overrides"),
                "metrics": s.get("metrics"),
            }
        return out


@dataclass
class RemoteOverview:
    """Eine kompakte Lageübersicht (health, state, Datenalter, Version) in einem Aufruf."""

    version: str | None
    health: dict[str, Any] = field(default_factory=dict)
    state: dict[str, Any] = field(default_factory=dict)

    def summary(self) -> str:
        st = self.state
        nodes = st.get("nodes_c", {})
        return (
            f"hp-twin {self.version or '?'} | letzte Messung {st.get('last_measurement', '?')} "
            f"(Alter {st.get('data_age_minutes', '?')} min) | "
            f"Puffer {nodes.get('buffer')} °C, DHW {nodes.get('dhw')} °C | "
            f"regler: {st.get('controller_state', {}).get('mode')}"
        )


def overview(client: DevApiClient, at: datetime | None = None) -> RemoteOverview:
    """Version + Health + Startzustand in einem Objekt (für 'wie steht die Anlage?')."""
    _ = at  # bewusst ohne Parameter: Instanz liefert den aktuellen Stand selbst
    return RemoteOverview(
        version=client.version().get("version"),
        health=client.health(),
        state=client.state(),
    )
