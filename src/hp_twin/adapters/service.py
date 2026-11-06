"""Automatik-Service: ein Zyklus = frischer Export -> Kalibrierung -> Prognose -> EOS-Push.

Der frische Export direkt vor der Prognose stellt sicher, dass der simulierte Startzustand
~ "jetzt" entspricht; `data_age_minutes` macht die Synchronität sichtbar. Der EOS-Payload wird
mit `start:"now"` gepusht, sodass beide Systeme dieselbe Zeitachse verwenden.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from hp_twin.adapters.settings import RuntimeSettings


@dataclass
class ServiceStatus:
    last_run_iso: str | None = None
    ok: bool = False
    error: str | None = None
    http_status: int | None = None
    planned_kwh: float | None = None
    n_events: int | None = None
    data_age_minutes: float | None = None
    exported_rows: int | None = None
    running: bool = False
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "last_run_iso": self.last_run_iso,
            "ok": self.ok,
            "error": self.error,
            "http_status": self.http_status,
            "planned_kwh": self.planned_kwh,
            "n_events": self.n_events,
            "data_age_minutes": self.data_age_minutes,
            "exported_rows": self.exported_rows,
            "running": self.running,
            **self.extra,
        }


def data_age_minutes(csv_path: str) -> float | None:
    """Alter des letzten Messzeitpunkts in Minuten (für den Start-Abgleich)."""
    p = Path(csv_path)
    if not p.exists():
        return None
    try:
        last = pd.read_csv(p, usecols=["timestamp"])["timestamp"].iloc[-1]
        ts = pd.to_datetime(last, utc=True)
        return float((datetime.now(UTC) - ts.to_pydatetime()).total_seconds() / 60.0)
    except Exception:  # pragma: no cover - defensive
        return None


def missing_config(settings: RuntimeSettings) -> list[str]:
    """Für einen Zyklus nötige Konfigurationsdateien, die noch fehlen."""
    need = [settings.topology_path]
    if settings.export_before_push:
        need.append(settings.ha_mapping_path)
    return [f for f in need if not Path(f).exists()]


def run_cycle(settings: RuntimeSettings, do_push: bool = True) -> ServiceStatus:
    """Führt einen Automatik-Zyklus aus und liefert den Status zurück."""
    from hp_twin.adapters.eos_connect import push_rest
    from hp_twin.adapters.forecast import run_forecast
    from hp_twin.adapters.ha_history import export_to_csv
    from hp_twin.domain.model_config import ModelConfig

    st = ServiceStatus(running=True, last_run_iso=datetime.now(UTC).isoformat())
    try:
        settings.apply_ha_env()  # HA-Zugang aus den Settings (überschreibt .env)
        missing = missing_config(settings)
        if missing:
            raise ValueError(
                "Konfiguration unvollständig, es fehlt: "
                + ", ".join(missing)
                + " – im Dashboard unter 'Portable Konfiguration' ein Config-Bundle importieren."
            )
        # 1) frischer Export (Startzustand ~ jetzt)
        if settings.export_before_push and Path(settings.ha_mapping_path).exists():
            st.exported_rows = export_to_csv(
                settings.ha_mapping_path,
                settings.env_path,
                days=settings.export_window_hours / 24.0,
                out_path=settings.csv_path,
                append=True,
            )

        # 2) Modell + Kalibrierung (gespeichert oder frisch aus CSV)
        from hp_twin.cli import resolve_calibration

        model = ModelConfig.from_path(settings.model_path)
        model.montecarlo.n_runs = settings.forecast_runs
        model.forecast.dhw_setpoint_cap_c = settings.dhw_setpoint_cap_c
        model.forecast.eos_ttl_minutes = settings.eos_ttl_minutes
        topo, df, calib = resolve_calibration(settings, model)
        if df is None:
            raise ValueError("Keine Messdaten (CSV) vorhanden – Export zuerst durchführen")

        # 3) Prognose ab Startzustand = Ende der (frischen) Daten
        start_idx = max(len(df) - int(round(settings.horizon_h * 60)) - 1, 0)
        fc = run_forecast(df, topo, calib, start_idx, horizon_h=settings.horizon_h, model=model)
        st.n_events = len(fc.events)
        st.planned_kwh = round(float(fc.slots.energy_wh.sum()) / 1000, 2)
        st.data_age_minutes = data_age_minutes(settings.csv_path)

        # 4) Push an EOS Connect
        if do_push:
            if not settings.eos_base_url or not settings.eos_managed_load_id:
                raise ValueError("EOS-Verbindung unvollständig (base_url / managed_load_id)")
            st.http_status = push_rest(
                settings.eos_base_url,
                settings.eos_managed_load_id,
                fc.eos,
                token=settings.eos_token,
            )
        st.ok = True
    except Exception as e:  # pragma: no cover - IO/Netz-abhängig
        st.error = str(e)
        st.ok = False
    finally:
        st.running = False
    return st
