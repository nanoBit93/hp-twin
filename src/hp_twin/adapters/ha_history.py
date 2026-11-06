"""HA-History -> CSV-Schema (data/schema.md).

Trennung: fetch_measurements (IO) vs. build_frame (reine, testbare Transformation).
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from hp_twin.adapters.home_assistant import HaClient

# Wasser: Dichte*spez. Wärmekapazität ~ 1 kg/L * 4186 J/(kg*K)
RHO_CP_J_PER_L_K = 4186.0


@dataclass(frozen=True)
class HaMapping:
    sensors: dict[str, str]
    rooms: dict[str, str]
    derived: dict[str, Any]
    resample: str = "1min"
    scales: dict[str, float] | None = None
    climate: dict[str, str] | None = None  # Raumname -> climate-Entity (Attribut-Historie)
    sg_status: str | None = None  # sensor.warmepumpe_sg_ready_status

    @classmethod
    def load(cls, path: str | Path) -> HaMapping:
        d = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
        return cls(
            sensors=d.get("sensors", {}),
            rooms=d.get("rooms", {}),
            derived=d.get("derived", {}),
            resample=d.get("resample", "1min"),
            scales=d.get("scales") or {},
            climate=d.get("climate") or {},
            sg_status=d.get("sg_status"),
        )

    def all_entities(self) -> list[str]:
        ents: set[str] = set(self.sensors.values()) | set(self.rooms.values())
        ents.add(self.derived["compressor_on"])
        for grp in ("mode", "circuit_active"):
            ents.update(self.derived.get(grp, {}).values())
        return sorted(ents)


def _series_from_history(block: list[dict[str, Any]]) -> pd.Series:
    """Ein HA-History-Block (minimal_response) -> Series mit UTC-Index."""
    raw_ts = [p.get("last_changed") or p.get("lu") or p.get("lc") for p in block]
    ts = pd.to_datetime(pd.Series(raw_ts), utc=True)
    vals = [p.get("state") for p in block]
    return pd.Series(vals, index=pd.DatetimeIndex(ts.to_numpy())).sort_index()


def _to_numeric_grid(s: pd.Series, grid: pd.DatetimeIndex) -> pd.Series:
    num = pd.to_numeric(s, errors="coerce").dropna()
    if num.empty:
        return pd.Series(index=grid, dtype=float)
    return num.reindex(num.index.union(grid)).sort_index().ffill().reindex(grid)


def _bool_grid(s: pd.Series, grid: pd.DatetimeIndex) -> pd.Series:
    b = s.map(lambda v: 1 if str(v).lower() in {"on", "true", "1", "heat", "heating"} else 0)
    b = b.reindex(b.index.union(grid)).sort_index().ffill().reindex(grid).fillna(0)
    return b.astype(int)


def build_frame(
    histories: dict[str, pd.Series], mapping: HaMapping, grid: pd.DatetimeIndex
) -> pd.DataFrame:
    """Reine Transformation (netzunabhängig, testbar): rohe Entity-Series -> Schema-DataFrame."""
    out = pd.DataFrame(index=grid)
    out.index.name = "timestamp"

    required = {
        "t_outdoor_c",
        "ghi_wm2",
        "p_el_w",
        "t_flow_c",
        "t_return_c",
        "t_dhw_c",
        "t_hk1_c",
        "t_hk2_c",
        "setpoint_dhw_c",
        "setpoint_hk1_c",
        "setpoint_hk2_c",
    }
    missing = required - set(mapping.sensors)
    if missing:
        raise ValueError(f"Mapping 'sensors' fehlt Pflicht-Spalten: {sorted(missing)}")

    scales = mapping.scales or {}
    for col, ent in mapping.sensors.items():
        if col in {"p_el_heater_w", "flow_rate_lpm"}:
            continue
        out[col] = _to_numeric_grid(histories.get(ent, pd.Series(dtype=float)), grid) * scales.get(
            col, 1.0
        )
    for name, ent in mapping.rooms.items():
        out[f"t_room_{name}_c"] = _to_numeric_grid(histories.get(ent, pd.Series(dtype=float)), grid)

    # compressor_on
    out["compressor_on"] = _bool_grid(
        histories.get(mapping.derived["compressor_on"], pd.Series(dtype=object)), grid
    )

    # mode: Priorität defrost > dhw > heat > idle
    mode = pd.Series("idle", index=grid)
    for m in ("heat", "dhw", "defrost"):
        ent = mapping.derived.get("mode", {}).get(m)
        if ent:
            on = _bool_grid(histories.get(ent, pd.Series(dtype=object)), grid) == 1
            mode = mode.mask(on, m)
    out["mode"] = mode.where(out["compressor_on"] == 1, "idle")

    # circuit_active: Priorität dhw > hk1 > hk2 > none
    circ = pd.Series("none", index=grid)
    for c in ("hk2", "hk1", "dhw"):
        ent = mapping.derived.get("circuit_active", {}).get(c)
        if ent:
            on = _bool_grid(histories.get(ent, pd.Series(dtype=object)), grid) == 1
            circ = circ.mask(on, c)
    out["circuit_active"] = circ.where(out["compressor_on"] == 1, "none")

    # Q_th aus Volumenstrom * rho*cp * (VL-RL), über Intervall integriert
    flow_ent = mapping.sensors.get("flow_rate_lpm")
    flow = (
        _to_numeric_grid(histories.get(flow_ent, pd.Series(dtype=float)), grid)
        if flow_ent
        else pd.Series(0.0, index=grid)
    )
    dt_s = (grid[1] - grid[0]).total_seconds() if len(grid) > 1 else 60.0
    power_th_w = (flow.fillna(0) / 60.0) * RHO_CP_J_PER_L_K * (out["t_flow_c"] - out["t_return_c"])
    power_th_w = power_th_w.clip(lower=0).where(out["compressor_on"] == 1, 0.0)
    out["q_heat_wh"] = (power_th_w * dt_s / 3600.0).round(1)
    out["flow_rate_lpm"] = flow.round(2)

    heater_ent = mapping.sensors.get("p_el_heater_w")
    if heater_ent:
        out["p_el_heater_w"] = _to_numeric_grid(
            histories.get(heater_ent, pd.Series(dtype=float)), grid
        ).round(1)

    return out.reset_index()


def export_to_csv(
    mapping_path: str,
    env_path: str,
    days: float,
    out_path: str,
    append: bool = True,
    timeout_s: float = 120.0,
    chunk_hours: float = 12.0,
) -> int:
    """Holt die letzten `days` Tage aus HA und schreibt/ergänzt die CSV. Rückgabe: Zeilenzahl."""
    from datetime import UTC
    from pathlib import Path

    from hp_twin.adapters.home_assistant import HaClient

    client = HaClient.from_env(env_path, timeout_s=timeout_s)
    mapping = HaMapping.load(mapping_path)
    end = datetime.now(UTC)
    start = end - timedelta(days=days)
    df = fetch_measurements(client, mapping, start, end, chunk_hours=chunk_hours)
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    if append and out.exists():
        old = pd.read_csv(out)
        old["timestamp"] = pd.to_datetime(old["timestamp"], utc=True)
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
        df = (
            pd.concat([old, df], ignore_index=True)
            .drop_duplicates(subset="timestamp", keep="last")
            .sort_values("timestamp")
        )
    df.to_csv(out, index=False)
    return len(df)


def fetch_measurements(
    client: HaClient,
    mapping: HaMapping,
    start: datetime,
    end: datetime,
    chunk_hours: float = 6.0,
    progress: bool = False,
) -> pd.DataFrame:
    """Holt History in Zeitchunks (schonend) und baut den Schema-DataFrame."""
    entities = mapping.all_entities()
    series: dict[str, list[pd.Series]] = {e: [] for e in entities}
    cur = start
    while cur < end:
        nxt = min(cur + timedelta(hours=chunk_hours), end)
        if progress:
            print(f"  ... {cur:%Y-%m-%d %H:%M} .. {nxt:%H:%M}", flush=True)
        blocks = client.history(entities, cur, nxt)
        for block in blocks:
            if not block:
                continue
            eid = block[0].get("entity_id")
            if eid in series:
                series[eid].append(_series_from_history(block))
        cur = nxt
    merged = {
        e: (pd.concat(parts).sort_index() if parts else pd.Series(dtype=object))
        for e, parts in series.items()
    }
    # Grid auf das Raster runden (volle Minuten), damit aufeinanderfolgende Exporte zusammenpassen
    g0 = pd.Timestamp(start).tz_convert("UTC").floor(mapping.resample)
    g1 = pd.Timestamp(end).tz_convert("UTC").floor(mapping.resample)
    grid = pd.date_range(g0, g1, freq=mapping.resample, tz="UTC")
    df = build_frame(merged, mapping, grid)
    clim = _fetch_climate(client, mapping, start, end, grid, chunk_hours, progress)
    if clim is not None:
        for col in clim.columns:
            df[col] = clim[col].to_numpy()
    return df


def _attr_series(block: list[dict[str, Any]], attr: str) -> pd.Series:
    raw_ts = [p.get("last_changed") or p.get("lu") or p.get("lc") for p in block]
    ts = pd.to_datetime(pd.Series(raw_ts), utc=True)
    vals = [p.get("attributes", {}).get(attr) for p in block]
    return pd.Series(vals, index=pd.DatetimeIndex(ts.to_numpy())).sort_index()


def _fetch_climate(
    client: HaClient,
    mapping: HaMapping,
    start: datetime,
    end: datetime,
    grid: pd.DatetimeIndex,
    chunk_hours: float,
    progress: bool,
) -> pd.DataFrame | None:
    """Per-Raum climate-Attribute (Sollwert, hvac_action) + SG-Status als Grid-Spalten."""
    climate = mapping.climate or {}
    if not climate and not mapping.sg_status:
        return None
    ents = list(climate.values()) + ([mapping.sg_status] if mapping.sg_status else [])
    sp_parts: dict[str, list[pd.Series]] = {e: [] for e in climate.values()}
    act_parts: dict[str, list[pd.Series]] = {e: [] for e in climate.values()}
    sg_parts: list[pd.Series] = []
    cur = start
    while cur < end:
        nxt = min(cur + timedelta(hours=chunk_hours), end)
        if progress:
            print(f"  (climate) {cur:%Y-%m-%d %H:%M} .. {nxt:%H:%M}", flush=True)
        for block in client.history(ents, cur, nxt, minimal=False):
            if not block:
                continue
            eid = block[0].get("entity_id")
            if eid in sp_parts:
                sp_parts[eid].append(_attr_series(block, "temperature"))
                act_parts[eid].append(_attr_series(block, "hvac_action"))
            elif eid == mapping.sg_status:
                sg_parts.append(_series_from_history(block))
        cur = nxt

    out = pd.DataFrame(index=grid)
    for name, ent in climate.items():
        sp = pd.concat(sp_parts[ent]).sort_index() if sp_parts[ent] else pd.Series(dtype=float)
        act = pd.concat(act_parts[ent]).sort_index() if act_parts[ent] else pd.Series(dtype=object)
        out[f"setpoint_room_{name}_c"] = _to_numeric_grid(sp, grid).round(1).to_numpy()
        act_grid = act.reindex(act.index.union(grid)).sort_index().ffill().reindex(grid)
        out[f"heating_room_{name}"] = (act_grid == "heating").astype(int).to_numpy()
    if mapping.sg_status and sg_parts:
        sg = pd.concat(sg_parts).sort_index()
        out["sg_status"] = _to_numeric_grid(sg, grid).round(0).to_numpy()
    return out
