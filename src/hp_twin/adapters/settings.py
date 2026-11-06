"""Persistente Laufzeit-Einstellungen (über das Dashboard-UI editierbar).

Umfasst EOS-Connect-Verbindung, Scheduler/Automatik und Prognose-Parameter. Wird als JSON
gespeichert (Default `config/runtime.json`, via Env HP_TWIN_RUNTIME überschreibbar). Die Datei
enthält ggf. ein EOS-Token und ist deshalb gitignored.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

from pydantic import BaseModel, Field


def runtime_path() -> str:
    return os.environ.get("HP_TWIN_RUNTIME", "config/runtime.json")


class RuntimeSettings(BaseModel):
    # Pfade / Datenquelle
    topology_path: str = "config/example_topology.yaml"
    model_path: str = "config/model.yaml"
    ha_mapping_path: str = "config/ha_mapping.yaml"
    heizlast_path: str = "config/heizlast.yaml"
    calibration_path: str = "config/calibration.json"
    use_stored_calibration: bool = False  # True: gespeicherte Kalibrierung statt Fit aus CSV
    env_path: str = ".env"
    csv_path: str = "data/ha_export/measurements.csv"
    # Home-Assistant-Verbindung (überschreibt .env, falls gesetzt)
    ha_base_url: str = ""
    ha_token: str | None = None
    # Prognose
    forecast_runs: int = Field(default=100, ge=1, le=1000)
    horizon_h: float = Field(default=48.0, gt=0, le=48)
    dhw_setpoint_cap_c: float | None = 45.0
    # EOS Connect
    eos_base_url: str = ""
    eos_managed_load_id: str = ""
    eos_token: str | None = None
    eos_ttl_minutes: int | None = None
    # Automatik / Scheduler
    auto_push_enabled: bool = False
    schedule_interval_minutes: float = Field(default=15.0, ge=1)
    export_before_push: bool = True
    export_window_hours: float = Field(default=6.0, gt=0, description="Export-Fenster je Zyklus")

    @classmethod
    def load(cls, path: str | None = None) -> RuntimeSettings:
        p = Path(path or runtime_path())
        if not p.exists():
            return cls()
        return cls.model_validate_json(p.read_text(encoding="utf-8"))

    def save(self, path: str | None = None) -> None:
        p = Path(path or runtime_path())
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(self.model_dump(), indent=2, ensure_ascii=False), encoding="utf-8")

    def public_dict(self) -> dict[str, object]:
        """Wie model_dump, aber ohne Tokens offenzulegen (nur Flags, ob gesetzt)."""
        d = self.model_dump()
        d["eos_token"] = None
        d["eos_token_set"] = bool(self.eos_token)
        d["ha_token"] = None
        d["ha_token_set"] = bool(self.ha_token)
        return d

    def apply_ha_env(self) -> None:
        """Setzt HA-Zugang aus den Settings in die Umgebung (überschreibt .env), falls gesetzt."""
        import os

        if self.ha_base_url:
            os.environ["HA_BASE_URL"] = self.ha_base_url
        if self.ha_token:
            os.environ["HA_TOKEN"] = self.ha_token
