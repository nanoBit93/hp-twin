"""Wetterprognose-Adapter über Home Assistant (DWD-Integration).

Quelle: POST /api/services/weather/get_forecasts (stündliche DWD-Vorhersage):
`temperature` -> Außentemperatur, `sun_irradiance` -> Globalstrahlung (W/m²).

Trennung: fetch_weather_forecast (IO, netzabhängig) vs. merge_forecast_columns (reine,
testbare Transformation). Nach dem Ende der Vorhersage bleibt der Fallback des
24-h-Wiederholungs-Frames (build_forecast_frame) aktiv; Merge ist reine Verbesserung.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from hp_twin.adapters.home_assistant import HaClient, HaError

FORECAST_SERVICE_PATH = "/api/services/weather/get_forecasts"
FORECAST_COLUMNS = ("t_outdoor_c", "ghi_wm2")


def fetch_weather_forecast(
    client: HaClient, entity_id: str, timeout_s: float | None = None
) -> pd.DataFrame:
    """Stündliche DWD-Vorhersage als DataFrame (UTC-Index, Spalten t_outdoor_c/ghi_wm2).

    Wirft HaError, wenn die Entity keine Vorhersagewerte liefert. Der Serviceaufruf nutzt
    `?return_response`, damit die Prognose direkt in der Antwort zurückkommt.
    """
    if timeout_s is not None:
        client = HaClient(base_url=client.base_url, token=client.token, timeout_s=timeout_s)
    data = client.post_response(FORECAST_SERVICE_PATH, {"entity_id": entity_id, "type": "hourly"})
    entries: list[dict[str, Any]] = []
    if isinstance(data, dict) and data:
        item = next(iter(data.values()), None)
        entries = (item or {}).get("forecast") or []
    elif isinstance(data, list) and data:
        entries = (data[0] or {}).get("forecast") or []
    rows: dict[pd.Timestamp, dict[str, float | None]] = {}
    for e in entries:
        if not isinstance(e, dict) or not e.get("datetime"):
            continue
        ts = pd.Timestamp(str(e.get("datetime")), tz="UTC")
        rows[ts] = {"t_outdoor_c": e.get("temperature"), "ghi_wm2": e.get("sun_irradiance")}
    if not rows:
        raise HaError(f"Weather-Entity '{entity_id}' liefert keine Vorhersagewerte")
    df = pd.DataFrame.from_dict(rows, orient="index").sort_index()
    df.index = pd.DatetimeIndex(df.index, tz="UTC")
    return df


def merge_forecast_columns(
    fut: pd.DataFrame, forecast: pd.DataFrame, columns: tuple[str, ...] = FORECAST_COLUMNS
) -> pd.DataFrame:
    """Ersetzt im Zukunfts-Frame die Wetter-Eingänge durch echte Prognosewerte.

    `fut` = Frame aus build_forecast_frame (dichtes Raster, UTC-Index), `forecast` =
    stündlicher Vorhersage-Frame (UTC-Index). Innerhalb des Vorhersagezeitraums werden
    die Spalten stündlich linear aufs Raster interpoliert und ersetzt; danach bleibt der
    Wiederholungs-Fallback aktiv. Von Frame-Start bis zum ersten Vorhersagewert gilt der
    erste Wert konstant (DWD startet typischerweise mit der laufenden Stunde).
    """
    out = fut.copy()
    if forecast is None or forecast.empty:
        return out
    cols = [c for c in columns if c in fut.columns and c in forecast.columns]
    if not cols:
        return out
    fc = forecast[list(cols)].astype(float).dropna(how="all")
    if fc.empty:
        return out
    grid = fc.index.union(out.index).sort_values()
    src = fc.reindex(grid).interpolate(method="time", limit_direction="forward")
    dense = src.reindex(out.index).interpolate(method="time", limit_direction="both")
    mask = np.asarray((out.index >= fc.index.min()) & (out.index <= fc.index.max()))
    fallback = out[list(cols)].to_numpy(dtype=float)
    values = np.column_stack([dense[c].to_numpy(dtype=float) for c in cols])
    merged = np.where(mask[:, None], values, fallback)
    for j, c in enumerate(cols):
        out[c] = pd.Series(merged[:, j], index=out.index).ffill().bfill().to_numpy(dtype=float)
    return out


def forecast_horizon_hours(forecast: pd.DataFrame) -> float:
    """Länge der Vorhersage in Stunden (0, wenn leer)."""
    if forecast is None or forecast.empty:
        return 0.0
    return float((forecast.index.max() - forecast.index.min()).total_seconds() / 3600.0)
