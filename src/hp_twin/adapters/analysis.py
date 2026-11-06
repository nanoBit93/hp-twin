"""Analyse-Bundle fürs Dashboard: Prognose + Einflussfaktoren + Zwischenergebnisse als JSON.

Liefert alles, was zum Troubleshooting nötig ist: Kalibrierparameter (wie die Berechnung
zustande kommt), die Eingangs-Zeitreihen (Wetter, Sollwerte, SG, aktive Räume), den
deterministischen Simulationsverlauf (Puffer, P_el, Modus) und die Monte-Carlo-Prognose.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from hp_twin.adapters.forecast import run_forecast
from hp_twin.adapters.replay import build_replay
from hp_twin.domain.calibration import CalibrationReport
from hp_twin.domain.model_config import ModelConfig
from hp_twin.domain.topology import Topology


def _series(a: Any, nd: int = 1) -> list[float]:
    return [round(float(x), nd) for x in np.asarray(a, dtype=float)]


def build_analysis(
    df: pd.DataFrame,
    topology: Topology,
    calib: CalibrationReport,
    start_idx: int,
    model: ModelConfig,
) -> dict[str, Any]:
    horizon_h = model.sim.horizon_h
    det = build_replay(
        df,
        topology,
        calib,
        start_idx,
        horizon_h,
        dhw_setpoint_cap_c=model.forecast.dhw_setpoint_cap_c,
        model=model,
    )
    fc = run_forecast(df, topology, calib, start_idx, horizon_h=horizon_h, model=model)
    res, sl = det.result, det.measured
    n = len(res.time_s)
    hours = _series(res.time_s / 3600.0, 3)

    room_cols = [c for c in sl.columns if c.startswith("setpoint_room_") and c.endswith("_c")]
    active_rooms = {
        c[len("setpoint_room_") : -len("_c")]: _series(
            (sl[c].to_numpy()[:n] > model.load.frost_threshold_c).astype(float), 0
        )
        for c in room_cols
    }

    factors: dict[str, Any] = {
        "hours": hours,
        "t_outdoor_c": _series(sl["t_outdoor_c"].to_numpy()[:n]),
        "ghi_wm2": _series(sl["ghi_wm2"].to_numpy()[:n], 0),
        "setpoint_hk1_c": _series(sl["setpoint_hk1_c"].to_numpy()[:n]),
        "setpoint_dhw_c": _series(sl["setpoint_dhw_c"].to_numpy()[:n]),
        "active_rooms": active_rooms,
    }
    if "sg_status" in sl.columns:
        factors["sg_status"] = _series(sl["sg_status"].to_numpy()[:n], 0)

    sim = {
        "hours": hours,
        "buffer_sim_c": _series(res.node_temps_c[det.buffer_node]),
        "dhw_sim_c": _series(res.node_temps_c[det.dhw_node]),
        "buffer_meas_c": _series(sl["t_hk1_c"].to_numpy()[:n]),
        "dhw_meas_c": _series(sl["t_dhw_c"].to_numpy()[:n]),
        "p_el_sim_w": _series(res.p_el_w, 0),
        "q_th_sim_w": _series(res.q_th_w, 0),
        "mode": res.mode,
    }

    slot_hours = _series(fc.slots.slot_start_s / 3600.0, 3)
    forecast = {
        "slot_hours": slot_hours,
        "p_el_mean_w": _series(fc.slots.p_el_mean_w, 0),
        "p_el_p10_w": _series(fc.slots.p_el_p10_w, 0),
        "p_el_p90_w": _series(fc.slots.p_el_p90_w, 0),
        "run_probability": _series(fc.slots.run_probability, 3),
        "energy_wh": _series(fc.slots.energy_wh, 0),
        "expected_energy_kwh": round(float(fc.slots.energy_wh.sum()) / 1000, 2),
        "events": fc.to_dict()["events"],
        "eos_payload": fc.eos.to_payload(),
    }

    return {
        "config": model.model_dump(),
        "calibration": calib.to_dict(),
        "factors": factors,
        "sim": sim,
        "forecast": forecast,
    }
