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
    phase: str | None = None
    weather_source: str | None = None
    weather_hours: float | None = None
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
            "phase": self.phase,
            "weather_source": self.weather_source,
            "weather_hours": self.weather_hours,
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


def history_span_days(csv_path: str) -> float:
    """Zeitliche Länge der vorhandenen Messdaten in Tagen (0 = keine)."""
    p = Path(csv_path)
    if not p.exists():
        return 0.0
    try:
        ts = pd.to_datetime(pd.read_csv(p, usecols=["timestamp"])["timestamp"], utc=True)
        return float((ts.iloc[-1] - ts.iloc[0]).total_seconds() / 86400.0)
    except Exception:  # pragma: no cover - defensive
        return 0.0


def plan_export(settings: RuntimeSettings) -> tuple[float, bool, str]:
    """Wieviel exportiert wird: (Tage, anhängen?, Phase).

    - keine/zu kurze Historie -> Backfill der letzten `backfill_days`
    - Lücke zu groß für den HA-Recorder -> neu beginnen (sonst ungleichmäßiges Raster)
    - sonst nur das Delta seit dem letzten Messpunkt (mind. `export_window_hours`)
    """
    age_min = data_age_minutes(settings.csv_path)
    if age_min is None or history_span_days(settings.csv_path) < settings.min_history_days:
        return settings.backfill_days, False, "Historie laden"
    if age_min / 1440.0 > 0.9 * settings.backfill_days:
        return settings.backfill_days, False, "Historie neu laden (Lücke)"
    hours = max(settings.export_window_hours, age_min / 60.0 + 1.0)
    return hours / 24.0, True, "Export"


def run_cycle(
    settings: RuntimeSettings, do_push: bool = True, status: ServiceStatus | None = None
) -> ServiceStatus:
    """Ein Automatik-Zyklus. Aktualisiert `status` fortlaufend (für die Fortschrittsanzeige)."""
    from hp_twin.adapters.eos_connect import push_rest
    from hp_twin.adapters.forecast import fetch_live_weather, run_live_forecast
    from hp_twin.adapters.ha_history import export_to_csv
    from hp_twin.cli import resolve_calibration
    from hp_twin.domain.model_config import ModelConfig

    st = status or ServiceStatus()
    st.running, st.ok, st.error, st.http_status = True, False, None, None
    st.last_run_iso = datetime.now(UTC).isoformat()
    try:
        settings.apply_ha_env()  # HA-Zugang aus den Settings (überschreibt .env)
        missing = missing_config(settings)
        if missing:
            raise ValueError(
                "Konfiguration unvollständig, es fehlt: "
                + ", ".join(missing)
                + " – im Dashboard unter 'Portable Konfiguration' ein Config-Bundle importieren."
            )
        # 1) Export: Backfill / Lückenschluss / Delta  (Startzustand = jetzt)
        if settings.export_before_push and Path(settings.ha_mapping_path).exists():
            days, append, phase = plan_export(settings)
            st.phase = f"{phase} ({days:.1f} d)"
            st.exported_rows = export_to_csv(
                settings.ha_mapping_path,
                settings.env_path,
                days=days,
                out_path=settings.csv_path,
                append=append,
            )

        # 2) Modell + Kalibrierung (gespeichert oder frisch aus CSV)
        st.phase = "Kalibrierung"
        model = ModelConfig.from_path(settings.model_path)
        model.montecarlo.n_runs = settings.forecast_runs
        model.forecast.dhw_setpoint_cap_c = settings.dhw_setpoint_cap_c
        model.forecast.eos_ttl_minutes = settings.eos_ttl_minutes
        topo, df, calib = resolve_calibration(settings, model)
        if df is None:
            raise ValueError("Keine Messdaten (CSV) vorhanden – Export zuerst durchführen")

        # 3) ECHTE Vorwärtsprognose ab dem letzten Messpunkt
        st.phase = "Prognose"
        weather = fetch_live_weather(settings, df)
        if weather is not None:
            st.weather_source = str(settings.weather_entity)
            st.weather_hours = round(
                float((weather.index.max() - weather.index.min()).total_seconds() / 3600.0), 1
            )
        fc = run_live_forecast(
            df, topo, calib, horizon_h=settings.horizon_h, model=model, weather=weather
        )
        st.n_events = len(fc.events)
        st.planned_kwh = round(float(fc.slots.energy_wh.sum()) / 1000, 2)
        st.data_age_minutes = data_age_minutes(settings.csv_path)

        # 4) Push an EOS Connect
        if do_push:
            st.phase = "Push an EOS"
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
        st.phase = None
    return st
