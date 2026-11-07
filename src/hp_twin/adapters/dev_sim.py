"""Szenario-Simulation (Was-wäre-wen) für die Analyse-API: Overrides auf Eingänge und Modell.

Wiederverwendet die Replay-Maschinerie (`build_replay`) und legt die Szenario-Overrides auf die
Erzeugung der Eingangsfunktionen. Overrides betreffen nur diesen Lauf:

- Eingänge: Außentemperatur/Strahlung (Offset oder eigene Reihe), Sollwerte (Reihe/Offset),
  Raumsoll, SG-Status
- Modell: Controller-Config (min_off/min_on/Hysterese), WP-Kennfeld, Gebäude-C, DHW-Cap
- Trace: pro Zeitschritt alle Zwischengrößen über den Simulator-Callback

Deterministisch (kein Zufall, ein Lauf); Vergleich gegen die Messdaten im Fenster inklusive.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from hp_twin.adapters.replay import build_replay, initial_controller_state
from hp_twin.domain.calibration import CalibrationReport
from hp_twin.domain.model_config import ModelConfig
from hp_twin.domain.simulator import StepTrace
from hp_twin.domain.topology import ControllerConfig, Topology


@dataclass
class Overrides:
    """Szenario-Overrides; None = unverändert aus Messdaten/Kalibrierung."""

    # Konstante Eingänge fürs ganze Fenster
    t_outdoor_c: float | None = None
    ghi_wm2: float | None = None
    room_setpoint_c: float | None = None  # ein aktiver Raumsollwert statt climate-Signal
    # Additionale Offsets auf die Messreihen
    t_outdoor_offset_k: float = 0.0
    setpoint_hk1_offset_k: float = 0.0
    setpoint_dhw_offset_k: float = 0.0
    # Modell-Parameter
    controller: dict[str, Any] | None = None  # merge auf ControllerConfig (min_off_s, ...)
    heat_pump: dict[str, Any] | None = None  # merge auf kalibriertes WP-Kennfeld
    c_building_j_per_k: float | None = None
    dhw_setpoint_cap_c: float | None = None


@dataclass
class ScenarioResult:
    summary: dict[str, Any]
    trace: list[dict[str, Any]]
    compare: dict[str, Any]


def _patch_config(
    topo: Topology, calib: CalibrationReport, ov: Overrides
) -> tuple[Topology, CalibrationReport]:
    """Kopien mit überschriebener Controller-Config bzw. WP-Kennfeld."""
    if ov.controller:
        merged = {**topo.controller.model_dump(), **ov.controller}
        topo = topo.model_copy(update={"controller": ControllerConfig.model_validate(merged)})
    if ov.heat_pump:
        import dataclasses

        hp = calib.heat_pump.heat_pump
        hp_eff = dataclasses.replace(hp, **ov.heat_pump)
        hp_fit = dataclasses.replace(calib.heat_pump, heat_pump=hp_eff)
        calib = CalibrationReport(
            heat_pump=hp_fit,
            buffer_threshold=calib.buffer_threshold,
            discharge=calib.discharge,
            notes=[*calib.notes, "WP-Kennfeld per Szenario überschrieben"],
        )
    return topo, calib


def _effective_setpoints(
    df_win: pd.DataFrame, ov: Overrides, dhw_cap: float | None
) -> tuple[pd.DataFrame, float | None]:
    """Wendet Sollwert-Offsets/DHW-Cap als Kopie des Fenster-DataFrames an."""
    df = df_win.copy()
    if ov.setpoint_hk1_offset_k:
        df["setpoint_hk1_c"] = df["setpoint_hk1_c"] + ov.setpoint_hk1_offset_k
    if ov.setpoint_dhw_offset_k:
        df["setpoint_dhw_c"] = df["setpoint_dhw_c"] + ov.setpoint_dhw_offset_k
    if dhw_cap is not None:
        df["setpoint_dhw_c"] = df["setpoint_dhw_c"].clip(upper=dhw_cap)
    return df, dhw_cap


def run_scenario(
    df: pd.DataFrame,
    topology: Topology,
    calib: CalibrationReport,
    model: ModelConfig,
    ov: Overrides,
    start_iso: str | None = None,
    horizon_h: float = 24.0,
    trace: bool = False,
    trace_step_s: float = 300.0,
) -> ScenarioResult:
    """Ein Szenario-Lauf über die geladene Historie. Liefert Summary + optionalen Trace + Vergleich.

    start_iso: Fensterstart (UTC; None = Anfang der Historie); horizon_h: Fensterlänge.
    trace: pro trace_step_s ein Takt-Schnappschuss (Regler/Soll/Vorlauf/Q/P_el/Knotentemp).
    """
    dt_s = float(df.index.to_series().diff().dropna().iloc[0].total_seconds())
    n = int(round(horizon_h * 3600 / dt_s))
    si = 0
    if start_iso:
        ts = pd.Timestamp(start_iso)
        ts = ts.tz_localize("UTC") if ts.tzinfo is None else ts.tz_convert("UTC")
        pos: int = int(df.index.searchsorted(ts))
        si = min(pos, len(df) - 1)
    df_win = df.iloc[si : si + n].copy()
    if df_win.empty:
        raise ValueError("Szenario-Fenster leer (start_iso hinter Datenende?)")
    n = len(df_win)

    topo, calib_eff = _patch_config(topology, calib, ov)
    cap = (
        model.forecast.dhw_setpoint_cap_c
        if ov.dhw_setpoint_cap_c is None
        else ov.dhw_setpoint_cap_c
    )
    df_use, cap = _effective_setpoints(df_win, ov, cap)

    if ov.t_outdoor_offset_k:
        df_use["t_outdoor_c"] = df_use["t_outdoor_c"] + ov.t_outdoor_offset_k
    if ov.t_outdoor_c is not None or ov.ghi_wm2 is not None:
        df_use = df_use.copy()
        if ov.t_outdoor_c is not None:
            df_use["t_outdoor_c"] = float(ov.t_outdoor_c)
        if ov.ghi_wm2 is not None:
            df_use["ghi_wm2"] = float(ov.ghi_wm2)

    traces: list[StepTrace] = []

    def on_trace(t_s: float, i: int, tr: StepTrace) -> None:
        traces.append(tr)

    ctrl_state = initial_controller_state(df_win, topo.controller.min_off_s)
    room_ov = None if ov.room_setpoint_c is None else np.full(n, float(ov.room_setpoint_c))
    rep = build_replay(
        df_use,
        topo,
        calib_eff,
        0,
        horizon_h,
        dhw_setpoint_cap_c=None,  # Cap läuft über _effective_setpoints (Offsets zuerst)
        c_building=ov.c_building_j_per_k,
        model=model,
        ctrl_state=ctrl_state,
        room_setpoint_override=room_ov,
        trace_callback=on_trace if trace else None,
    )

    res, meas = rep.result, rep.measured
    meas = meas.iloc[:n]

    # Trace drosseln und formatieren
    trace_out = _format_traces(traces, dt_s, trace_step_s) if trace else []

    # Vergleich mit Messung
    from hp_twin.domain.backtest import compute_metrics

    n_cmp = min(len(res.time_s), len(meas))
    on = np.array([1 if m in ("heat", "dhw") else 0 for m in res.mode[:n_cmp]])
    meas_on = meas["compressor_on"].to_numpy()[:n_cmp]
    met = compute_metrics(
        res.time_s[:n_cmp],
        on,
        meas_on,
        res.p_el_w[:n_cmp],
        meas["p_el_w"].to_numpy()[:n_cmp],
        res.node_temps_c[rep.buffer_node][:n_cmp],
        meas["t_hk1_c"].to_numpy()[:n_cmp],
    )

    events = []
    starts = np.where(np.diff(on) == 1)[0] + 1
    stops = np.where(np.diff(on) == -1)[0] + 1
    for k, s_i in enumerate(starts):
        e_i = int(stops[k]) if k < len(stops) else n_cmp
        seg = res.p_el_w[s_i:e_i]
        events.append(
            {
                "start_min": round(float(res.time_s[s_i]) / 60.0, 1),
                "end_min": round(float(res.time_s[e_i - 1] + dt_s) / 60.0, 1),
                "mode": res.mode[s_i],
                "mean_p_el_w": round(float(np.mean(seg)) if len(seg) else 0.0, 0),
                "energy_wh": round(float(seg.sum() * dt_s / 3600.0), 0),
            }
        )

    t0 = df.index[si]
    summary = {
        "topology": topology.name,
        "start_iso": str(t0),
        "horizon_h": round(n_cmp * dt_s / 3600.0, 3),
        "timestep_s": dt_s,
        "overrides": {
            k: (v if not isinstance(v, dict) else {kk: vv for kk, vv in v.items()})
            for k, v in ov.__dict__.items()
            if v is not None
        },
        "controller_effective": topo.controller.model_dump(),
        "heat_pump_effective": {
            f: round(float(getattr(calib_eff.heat_pump.heat_pump, f)), 3)
            if isinstance(getattr(calib_eff.heat_pump.heat_pump, f), float)
            else getattr(calib_eff.heat_pump.heat_pump, f)
            for f in (
                "p_el_ref_w",
                "dp_dt_flow_w_per_k",
                "dp_dt_out_w_per_k",
                "eta_carnot",
                "standby_w",
                "t_flow_ref_c",
                "t_out_ref_c",
                "p_comp_min_w",
                "p_comp_max_w",
                "cop_min",
                "cop_max",
            )
        },
        "c_building_j_per_k": ov.c_building_j_per_k or model.load.c_building_j_per_k,
        "dhw_setpoint_cap_c": cap,
        "energy_kwh": round(float(res.p_el_w[:n_cmp].sum() * dt_s / 3600.0 / 1000.0), 3),
        "n_cycles": len(starts),
        "notes": list(calib_eff.notes),
        "metrics": {
            "next_start_error_min": _round_opt(met.next_start_error_min),
            "start_errors_min": [_round_opt(x) for x in met.start_errors_min],
            "energy_6h_mae_wh": _round_opt(met.energy_6h_mae_wh),
            "energy_6h_bias_wh": _round_opt(met.energy_6h_bias_wh),
            "buffer_rmse_c": _round_opt(met.buffer_rmse_c),
            "cycles_per_day_sim": _round_opt(met.cycles_per_day_sim, 2),
            "cycles_per_day_measured": _round_opt(met.cycles_per_day_meas, 2),
        },
    }

    hours = (np.arange(n_cmp) * dt_s / 3600.0).round(3).tolist()
    compare = {
        "start_iso": str(t0),
        "timestep_s": dt_s,
        "hours": hours,
        "buffer_sim_c": _series(res.node_temps_c[rep.buffer_node][:n_cmp]),
        "dhw_sim_c": _series(res.node_temps_c[rep.dhw_node][:n_cmp]),
        "buffer_measured_c": _series(meas["t_hk1_c"].to_numpy()[:n_cmp]),
        "dhw_measured_c": _series(meas["t_dhw_c"].to_numpy()[:n_cmp]),
        "p_el_sim_w": _series(res.p_el_w[:n_cmp], 0),
        "p_el_measured_w": _series(meas["p_el_w"].to_numpy()[:n_cmp], 0),
        "q_th_sim_w": _series(res.q_th_w[:n_cmp], 0),
        "mode_sim": list(res.mode[:n_cmp]),
        "compressor_measured": [int(v) for v in meas_on],
    }

    return ScenarioResult(summary=summary, trace=trace_out, compare=compare)


def _round_opt(x: float, nd: int = 1) -> float | None:
    x = float(x)
    return round(x, nd) if np.isfinite(x) else None


def _series(a: Any, nd: int = 1) -> list[float | None]:
    arr = np.asarray(a, dtype=float)
    return [round(float(v), nd) if np.isfinite(v) else None for v in arr]


def _format_traces(
    traces: list[StepTrace], dt_s: float, trace_step_s: float
) -> list[dict[str, Any]]:
    """StepTrace-Objekte als JSON-Liste, mit Sampling-Rate trace_step_s (>= 1 Zeitschritt)."""
    step = max(int(round(trace_step_s / dt_s)), 1) if trace_step_s else 1
    out: list[dict[str, Any]] = []
    for tr in traces:
        if tr.index % step != 0:
            continue
        out.append(
            {
                "t_s": round(tr.t_s, 1),
                "index": tr.index,
                "mode": tr.mode,
                "charging": tr.charging,
                "target_node": tr.target_node,
                "flow_setpoint_c": round(tr.flow_setpoint_c, 2),
                "t_flow_eff_c": round(tr.t_flow_eff_c, 2),
                "p_el_w": round(tr.p_el_w, 1),
                "q_th_w": round(tr.q_th_w, 1),
                "cop_est": (
                    round(tr.cop_carnot_based, 2) if np.isfinite(tr.cop_carnot_based) else None
                ),
                "setpoints": {
                    "dhw_c": round(tr.setpoints.dhw_c, 2),
                    "buffer_c": round(tr.setpoints.buffer_c, 2),
                    "dhw_hysteresis_k": tr.setpoints.dhw_hysteresis_k,
                    "buffer_hysteresis_k": tr.setpoints.buffer_hysteresis_k,
                    "dhw_overshoot_k": tr.setpoints.dhw_overshoot_k,
                    "buffer_overshoot_k": tr.setpoints.buffer_overshoot_k,
                },
                "controller": {
                    "mode": tr.controller_state.mode.value,
                    "time_in_mode_s": tr.controller_state.time_in_mode_s,
                    "time_since_stop_s": tr.controller_state.time_since_stop_s,
                    "active_node": tr.controller_state.active_node,
                },
                "nodes_before_c": {k: round(v, 3) for k, v in tr.node_temps_before_c.items()},
                "nodes_after_c": {k: round(v, 3) for k, v in tr.node_temps_after_c.items()},
                "q_in_w": {k: round(v, 1) for k, v in tr.q_in_w.items()},
                "q_out_w": {k: round(v, 1) for k, v in tr.q_out_w.items()},
            }
        )
    return out
