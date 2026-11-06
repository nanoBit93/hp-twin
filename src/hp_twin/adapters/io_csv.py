"""CSV-Schema für historische Messdaten (siehe data/schema.md)."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

TIME_COLUMN = "timestamp"
MODES = ("idle", "heat", "dhw", "defrost")

# Pflichtspalten: Name -> (Einheit, Beschreibung)
REQUIRED_COLUMNS: dict[str, tuple[str, str]] = {
    "t_outdoor_c": ("°C", "Außentemperatur gemessen"),
    "ghi_wm2": ("W/m²", "Globalstrahlung (Prognose/Messung)"),
    "compressor_on": ("0/1", "Verdichter läuft"),
    "mode": ("-", "idle|heat|dhw|defrost"),
    "p_el_w": ("W", "elektrische Leistung WP gesamt"),
    "q_heat_wh": ("Wh", "Wärmemenge im Intervall"),
    "t_flow_c": ("°C", "Vorlauf WP"),
    "t_return_c": ("°C", "Rücklauf WP"),
    "t_dhw_c": ("°C", "Speicherfühler DHW"),
    "t_hk1_c": ("°C", "Speicherfühler HK1 (Puffer)"),
    "t_hk2_c": ("°C", "HK2-Ist"),
    "setpoint_dhw_c": ("°C", "Soll DHW"),
    "setpoint_hk1_c": ("°C", "Soll HK1"),
    "setpoint_hk2_c": ("°C", "Soll HK2"),
}
OPTIONAL_COLUMNS: dict[str, tuple[str, str]] = {
    "t_outdoor_fc_c": ("°C", "Außentemperatur-Prognose"),
    "circuit_active": ("-", "hk1|hk2|dhw|none"),
}
ROOM_PREFIX, ROOM_SUFFIX = "t_room_", "_c"  # t_room_<name>_c


class CsvSchemaError(ValueError):
    pass


def room_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c.startswith(ROOM_PREFIX) and c.endswith(ROOM_SUFFIX)]


def validate_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Prüft Schema, erzwingt UTC-DatetimeIndex, monotone Zeit und gleichmäßiges Raster."""
    missing = [c for c in (TIME_COLUMN, *REQUIRED_COLUMNS) if c not in df.columns]
    if missing:
        raise CsvSchemaError(f"Fehlende Spalten: {missing}")
    out = df.copy()
    out[TIME_COLUMN] = pd.to_datetime(out[TIME_COLUMN], utc=True)
    out = out.set_index(TIME_COLUMN).sort_index()
    if out.index.has_duplicates:
        raise CsvSchemaError("Doppelte Zeitstempel")
    if not room_columns(out.reset_index()):
        raise CsvSchemaError("Mindestens eine Raumtemperatur-Spalte t_room_<name>_c erforderlich")
    bad_modes = set(out["mode"].unique()) - set(MODES)
    if bad_modes:
        raise CsvSchemaError(f"Unbekannte Modi: {bad_modes}")
    if len(out) > 1:
        steps = out.index.to_series().diff().dropna().unique()
        if len(steps) != 1:
            raise CsvSchemaError("Zeitraster ist nicht gleichmäßig (Lücken/Wechsel)")
    return out


def read_measurements(path: str | Path) -> pd.DataFrame:
    return validate_frame(pd.read_csv(path))
