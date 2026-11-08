"""Analyse-Bundle fürs Dashboard: Prognose + Einflussfaktoren + Zwischenergebnisse als JSON.

Liefert alles, was zum Troubleshooting nötig ist: Kalibrierparameter (wie die Berechnung
zustande kommt), die Eingangs-Zeitreihen (Wetter, Sollwerte, SG, aktive Räume), den
deterministischen Simulationsverlauf (Puffer, P_el, Modus) und die Monte-Carlo-Prognose.

live=True: echte Vorwärtsprognose ab dem letzten Messpunkt (Eingänge = Wetterprognose via HA,
sofern konfiguriert, sonst letzte 24 h wiederholt), dazu die letzten 24 h Messung als Kontext.
live=False: Replay ab Datenanfang mit gemessenem Wetter.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from hp_twin.adapters.forecast import run_forecast, run_live_forecast
from hp_twin.adapters.replay import build_forecast_frame, build_replay, initial_controller_state
from hp_twin.adapters.settings import RuntimeSettings
from hp_twin.domain.calibration import CalibrationReport
from hp_twin.domain.model_config import ModelConfig
from hp_twin.domain.topology import Topology


def _series(a: Any, nd: int = 1) -> list[float | None]:
    """Zeitreihe als JSON-taugliche Liste (NaN/inf -> None)."""
    arr = np.asarray(a, dtype=float)
    return [round(float(x), nd) if np.isfinite(x) else None for x in arr]


def build_analysis(
    df: pd.DataFrame,
    topology: Topology,
    calib: CalibrationReport,
    model: ModelConfig,
    live: bool = True,
    start_idx: int = 0,
    settings: RuntimeSettings | None = None,
) -> dict[str, Any]:
    horizon_h = model.sim.horizon_h
    cap = model.forecast.dhw_setpoint_cap_c
    weather: pd.DataFrame | None = None
    weather_meta: dict[str, Any] = {}
    sg_plan_meta: dict[str, Any] = {"enabled": False}
    if live and settings is not None and settings.weather_entity:
        from hp_twin.adapters.forecast import fetch_live_weather
        from hp_twin.adapters.weather_forecast import forecast_horizon_hours

        weather = fetch_live_weather(settings, df)
        if weather is not None:
            weather_meta = {
                "entity": settings.weather_entity,
                "horizon_h": round(forecast_horizon_hours(weather), 1),
            }
    if live:
        frame = build_forecast_frame(df, horizon_h, weather=weather)
        cs = initial_controller_state(df, topology.controller.min_off_s)
        det = build_replay(
            frame, topology, calib, 0, horizon_h, dhw_setpoint_cap_c=cap, model=model, ctrl_state=cs
        )
        fc = run_live_forecast(
            df, topology, calib, horizon_h=horizon_h, model=model, weather=weather
        )
    else:
        det = build_replay(
            df, topology, calib, start_idx, horizon_h, dhw_setpoint_cap_c=cap, model=model
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

    # Kontext: die letzten 24 h Messung (negative Stunden = Vergangenheit)
    history: dict[str, Any] = {}

    sg_plan_meta = {"enabled": bool(settings is not None and settings.sg_enabled)}
    if live and sg_plan_meta["enabled"] and settings is not None:
        from hp_twin.adapters.evcc import fetch_market_series_safe
        from hp_twin.adapters.sg_orchestrator import SgPlanConfig, run_sg_cycle

        sg_cfg = SgPlanConfig(
            enabled=True,
            sg3_load_id=settings.sg_load_id,
            profile_load_id=settings.profile_load_id,
            min_pv_surplus_w=float(settings.sg_min_pv_surplus_w),
            max_price_eur=settings.sg_max_price_eur,
            dry_run=True,  # Analysis ist lesend -> nichts pushen; Placement via Folgetakt
        )
        try:
            _, sg_res = run_sg_cycle(
                df,
                topology,
                calib,
                model,
                forecast_fn=run_live_forecast,
                market=fetch_market_series_safe(settings),
                cfg=sg_cfg,
                eos_base_url="",
            )
            sg_plan_meta.update(sg_res.to_dict())
        except (ValueError, RuntimeError) as e:
            sg_plan_meta["notes"] = [f"SG-Zyklus fehlgeschlagen: {e}"]

    if live:
        dt_s = float(df.index.to_series().diff().dropna().iloc[0].total_seconds())
        h24 = df.iloc[-int(round(86400 / dt_s)) :]
        history = {
            "hours": _series(-(len(h24) - 1 - np.arange(len(h24))) * dt_s / 3600.0, 3),
            "buffer_meas_c": _series(h24["t_hk1_c"].to_numpy()),
            "dhw_meas_c": _series(h24["t_dhw_c"].to_numpy()),
            "p_el_w": _series(h24["p_el_w"].to_numpy(), 0),
        }

    forecast = {
        "mode": "live" if live else "replay",
        "start_state": {
            "buffer_c": round(float(df["t_hk1_c"].iloc[-1]), 1),
            "dhw_c": round(float(df["t_dhw_c"].iloc[-1]), 1),
            "last_measurement": str(df.index[-1]),
        },
        "weather": weather_meta,
        "slot_hours": _series(fc.slots.slot_start_s / 3600.0, 3),
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
        "history": history,
        "forecast": forecast,
        "sg_plan": sg_plan_meta,
    }
