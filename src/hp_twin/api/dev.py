"""Analyse-/Dev-API: Fern-Diagnose der Prognose im Produktivsystem (Home Assistant Add-on).

Der lokale Entwicklungsrechner fragt die laufende Add-on-Instanz per HTTP ab (Bearer-Token);
dieselben Endpunkte stehen im Dashboard-Ingress zur Verfügung. Nur lesend/rechnend – Konfig und
Produktivdaten werden nicht verändert.

Endpunkte:
- GET  /api/dev/data                  – Datenbestand (Zeitbereich, Raster, Spalten, Kalibrierung)
- GET  /api/dev/data?start=&end=      – Ausschnitt der Messreihen (JSON, gerundet)
- GET  /api/dev/state                 – aktueller Startzustand (Knotentemperaturen, Modus)
- GET  /api/dev/calibration           – aktive Kalibrierung (WP-Kennfeld, Schwellen, Entladung)
- GET  /api/dev/trace    – Zwischenschritte eines Referenzlaufs (Reglerentscheid je Takt)
- POST /api/dev/scenario           – Was-wäre-wen: Overrides auf Eingänge und Modellparameter
- POST /api/dev/scenarios         – Batch: mehrere Szenarien in einem Aufruf (direkt vergleichbar)
"""

from __future__ import annotations

from typing import Annotated, Any

import numpy as np
import pandas as pd
from fastapi import APIRouter, Depends, HTTPException, Query, Request
from pydantic import BaseModel, ConfigDict, Field

from hp_twin.adapters.dev_sim import Overrides, ScenarioResult, run_scenario
from hp_twin.adapters.settings import RuntimeSettings
from hp_twin.api.security import request_meta, require_token

router = APIRouter(prefix="/api/dev", tags=["dev"])

# Obergrenzen, damit ein Aufruf die Instanz nicht blockiert bzw. die Antwort handhabbar bleibt.
_MAX_SCENARIOS = 20
_MAX_TRACE_POINTS = 4000
_MAX_WINDOW_POINTS = 5000

# Serie -> Rundung: diskrete/wenig gestufte Werte kompakt, Leistungen/Arbeit als ganze Zahl.
_SERIES_ND: dict[str, int] = {
    "t_outdoor_c": 2,
    "ghi_wm2": 0,
    "compressor_on": 0,
    "p_el_w": 0,
    "q_heat_wh": 0,
    "t_flow_c": 1,
    "t_return_c": 1,
    "t_dhw_c": 1,
    "t_hk1_c": 1,
    "t_hk2_c": 1,
    "setpoint_dhw_c": 1,
    "setpoint_hk1_c": 1,
    "setpoint_hk2_c": 1,
}


def _load_context(settings: RuntimeSettings) -> tuple[Any, pd.DataFrame, Any, Any]:
    """(topology, messframe, kalibrierung, model) wie die Prognose sie benutzt."""
    from hp_twin.cli import resolve_calibration
    from hp_twin.domain.model_config import ModelConfig

    model = ModelConfig.from_path(settings.model_path)
    try:
        topo, df, calib = resolve_calibration(settings, model)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e
    if df is None:
        raise HTTPException(
            status_code=404, detail="Keine Messdaten (CSV leer) – erst Export/Bundle-Import"
        )
    return topo, df, calib, model


def _num(x: Any, nd: int = 1) -> float | None:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return round(v, nd) if np.isfinite(v) else None


def _series(a: Any, nd: int = 1) -> list[float | None]:
    arr = np.asarray(a, dtype=float)
    return [round(float(v), nd) if np.isfinite(v) else None for v in arr]


def _parse_utc(ts: str) -> pd.Timestamp:
    t = pd.Timestamp(ts)
    return t.tz_localize("UTC") if t.tzinfo is None else t.tz_convert("UTC")


# --------------------------------------------------------------------------- Datenbestand
@router.get("/data")
def data_endpoint(
    request: Request,
    start: str | None = Query(default=None),
    end: str | None = Query(default=None),
    _auth: None = Depends(require_token),
) -> dict[str, Any]:
    """Datenbestand/Zeitbereich/Spalten/Kalibrierung; mit start/end ein Messreihen-Ausschnitt."""
    settings = RuntimeSettings.load()
    topo, df, calib, _model = _load_context(settings)
    dt_s = float(df.index.to_series().diff().dropna().iloc[0].total_seconds())
    last = df.iloc[-1]
    out: dict[str, Any] = {
        "meta": request_meta(request),
        "topology": topo.name,
        "csv_path": settings.csv_path,
        "n_rows": len(df),
        "first_iso": str(df.index[0]),
        "last_iso": str(df.index[-1]),
        "span_days": _num((df.index[-1] - df.index[0]).total_seconds() / 86400.0, 2),
        "step_s": dt_s,
        "columns": sorted(df.reset_index().columns.tolist()),
        "state": {
            "buffer_c": _num(last["t_hk1_c"]),
            "dhw_c": _num(last["t_dhw_c"]),
            "t_outdoor_c": _num(last["t_outdoor_c"]),
            "compressor_on": int(last["compressor_on"] or 0),
            "mode": str(last["mode"]),
            "p_el_w": _num(last["p_el_w"], 0),
        },
        "calibration": calib.to_dict(),
    }
    if start or end:
        s = _parse_utc(start) if start else df.index[0]
        e = _parse_utc(end) if end else df.index[-1]
        win_full = df.loc[s:e]
        if win_full.empty:
            raise HTTPException(status_code=404, detail="Zeitfenster außerhalb der Daten")
        downsampled_by = 1
        win = win_full
        if len(win) > _MAX_WINDOW_POINTS:
            downsampled_by = int(np.ceil(len(win) / _MAX_WINDOW_POINTS))
            win = win.iloc[::downsampled_by]
        cols = [c for c in _SERIES_ND if c in win.columns]
        out["window"] = {
            "start_iso": str(win_full.index[0]),
            "end_iso": str(win_full.index[-1]),
            "n": len(win),
            "downsampled_by": downsampled_by,
            "timestamps": [i.isoformat() for i in win.index],
            "series": {c: _series(win[c].to_numpy(), _SERIES_ND[c]) for c in cols},
        }
    return out


# --------------------------------------------------------------------------- Startzustand
@router.get("/state")
def state_endpoint(request: Request, _auth: None = Depends(require_token)) -> dict[str, Any]:
    """Startzustand 'jetzt': Knotentemperaturen und Regler-Bewertung des letzten Messpunkts."""
    settings = RuntimeSettings.load()
    topo, df, _calib, _model = _load_context(settings)
    from hp_twin.adapters.replay import initial_controller_state

    cs = initial_controller_state(df, float(topo.controller.min_off_s))
    return {
        "meta": request_meta(request),
        "last_measurement": str(df.index[-1]),
        "data_age_minutes": _num(
            (pd.Timestamp.now(tz="UTC") - df.index[-1]).total_seconds() / 60.0, 1
        ),
        "nodes_c": {
            "dhw": _num(df["t_dhw_c"].iloc[-1]),
            "buffer": _num(df["t_hk1_c"].iloc[-1]),
        },
        "t_outdoor_c": _num(df["t_outdoor_c"].iloc[-1]),
        "compressor_on": bool(df["compressor_on"].iloc[-1]),
        "measured_mode": str(df["mode"].iloc[-1]),
        "controller_state": {
            "mode": cs.mode.value,
            "time_in_mode_s": cs.time_in_mode_s,
            "time_since_stop_s": cs.time_since_stop_s,
            "active_node": cs.active_node,
        },
    }


# --------------------------------------------------------------------------- Kalibrierung
@router.get("/calibration")
def calibration_endpoint(request: Request, _auth: None = Depends(require_token)) -> dict[str, Any]:
    """Aktive Kalibrierung (WP-Kennfeld, Regler-Schwellen, Gebäude-Entladung) + Modellparameter."""
    settings = RuntimeSettings.load()
    _topo, _df, calib, model = _load_context(settings)
    return {
        "meta": request_meta(request),
        "calibration": calib.to_dict(),
        "model": model.model_dump(),
    }


# --------------------------------------------------------------------------- Szenarien
class OverridesIn(BaseModel):
    """Was-wäre-wen-Overrides (alle optional; Feld weglassen = unverändert)."""

    model_config = ConfigDict(extra="forbid")

    t_outdoor_c: float | None = None
    ghi_wm2: float | None = None
    t_outdoor_offset_k: float = 0.0
    room_setpoint_c: float | None = None
    setpoint_hk1_offset_k: float = 0.0
    setpoint_dhw_offset_k: float = 0.0
    controller: dict[str, Any] | None = (
        None  # Merge auf ControllerConfig (min_off_s, min_on_s, ...)
    )
    heat_pump: dict[str, Any] | None = None  # Merge auf das kalibrierte WP-Kennfeld
    c_building_j_per_k: float | None = None
    dhw_setpoint_cap_c: float | None = None


class ScenarioIn(BaseModel):
    model_config = ConfigDict(extra="forbid")

    start: str | None = None  # UTC-ISO; None = Historienanfang
    horizon_h: Annotated[float, Field(gt=0, le=48)] = 24.0
    trace: bool = False
    trace_step_s: Annotated[float, Field(gt=0)] = 300.0
    overrides: OverridesIn = OverridesIn()


def _run_single(settings: RuntimeSettings, body: ScenarioIn) -> ScenarioResult:
    topo, df, calib, model = _load_context(settings)
    ov = Overrides(**body.overrides.model_dump())
    return run_scenario(
        df,
        topo,
        calib,
        model,
        ov,
        start_iso=body.start,
        horizon_h=body.horizon_h,
        trace=body.trace,
        trace_step_s=body.trace_step_s,
    )


def _scenario_payload(res: ScenarioResult, with_trace: bool) -> dict[str, Any]:
    return {
        "summary": res.summary,
        "compare": res.compare,
        **({"trace": res.trace} if with_trace else {}),
    }


@router.post("/scenario")
def scenario_endpoint(
    request: Request,
    body: ScenarioIn,
    _auth: None = Depends(require_token),
) -> dict[str, Any]:
    """Ein Szenario-Lauf: Overrides anwenden, vorwärts simulieren, gegen die Messung vergleichen."""
    settings = RuntimeSettings.load()
    res = _run_single(settings, body)
    return _scenario_payload(res, body.trace)


@router.post("/scenarios")
def scenarios_endpoint(
    request: Request,
    bodies: list[ScenarioIn],
    _auth: None = Depends(require_token),
) -> dict[str, Any]:
    """Batch-Variante von /scenario (1..20): alle Szenarien auf denselben Daten, vergleichbar."""
    if not bodies or len(bodies) > _MAX_SCENARIOS:
        raise HTTPException(
            status_code=422, detail=f"1..{_MAX_SCENARIOS} Szenarien im Batch erwartet"
        )
    settings = RuntimeSettings.load()
    results = [
        {"index": i, **_scenario_payload(_run_single(settings, b), b.trace)}
        for i, b in enumerate(bodies)
    ]
    return {
        "n": len(results),
        "results": results,
        "energy_kwh": [r["summary"]["energy_kwh"] for r in results],
        "n_cycles": [r["summary"]["n_cycles"] for r in results],
    }


# --------------------------------------------------------------------------- Trace (Referenz)
@router.get("/trace")
def trace_endpoint(
    request: Request,
    start: str | None = Query(default=None),
    horizon_h: Annotated[float, Query(gt=0, le=24)] = 6.0,
    trace_step_s: Annotated[float, Query(gt=0)] = 300.0,
    _auth: None = Depends(require_token),
) -> dict[str, Any]:
    """Zwischenschritte des UNVERÄNDERTEN Laufs: Regler/Sollwerte/Vorlauf/Q_th/P_el je Takt."""
    settings = RuntimeSettings.load()
    res = _run_single(
        settings,
        ScenarioIn(start=start, horizon_h=horizon_h, trace=True, trace_step_s=trace_step_s),
    )
    trace = res.trace
    if len(trace) > _MAX_TRACE_POINTS:
        keep = int(np.ceil(len(trace) / _MAX_TRACE_POINTS))
        trace = trace[::keep]
    return {
        "meta": request_meta(request),
        "summary": res.summary,
        "trace": trace,
        "compare": res.compare,
    }
