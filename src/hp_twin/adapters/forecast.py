"""Prognose-Orchestrator: deterministischer Lauf + Monte-Carlo -> Events, Slots, EOS-Payload.

Nutzt die kalibrierte Replay-Maschinerie (build_replay) und streut für Monte-Carlo die
Entladerate, Solargewinne und Gebäudemasse. Alle Parameter kommen aus ModelConfig.
"""

from __future__ import annotations

import copy
import logging
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from hp_twin.adapters.eos_connect import (
    EosLoadProfile,
    load_profile_from_slots,
    thermal_sidecar,
)
from hp_twin.adapters.replay import (
    build_forecast_frame,
    build_replay,
    initial_controller_state,
)
from hp_twin.adapters.settings import RuntimeSettings
from hp_twin.domain.calibration import CalibrationReport
from hp_twin.domain.controller import ControllerState
from hp_twin.domain.model_config import ModelConfig
from hp_twin.domain.montecarlo import (
    ForecastEvent,
    SlotForecast,
    aggregate_slots,
    extract_events,
)
from hp_twin.domain.sg_plan import DEFAULT_SG3_MIN_SETPOINT_C
from hp_twin.domain.topology import Topology

log = logging.getLogger(__name__)


@dataclass
class ForecastResult:
    events: list[ForecastEvent]
    slots: SlotForecast
    eos: EosLoadProfile
    deterministic_p_el_w: np.ndarray
    time_s: np.ndarray
    buffer_sim_c: np.ndarray | None = None  # det. Pufferverlauf (SG-Kandidaten)
    dhw_sim_c: np.ndarray | None = None
    frame_start: pd.Timestamp | None = None
    outdoor_c: np.ndarray | None = None
    heat_pump: Any = None
    controller_mode: list[str] | None = None

    def to_dict(self) -> dict[str, Any]:
        s = self.slots
        return {
            "events": [
                {
                    "start_min": round(e.start_s / 60, 1),
                    "end_min": round(e.end_s / 60, 1),
                    "mode": e.mode,
                    "mean_p_el_w": round(e.mean_p_el_w, 0),
                }
                for e in self.events
            ],
            "slots_15min": {
                "p_el_mean_w": [round(float(x), 1) for x in s.p_el_mean_w],
                "p_el_p10_w": [round(float(x), 1) for x in s.p_el_p10_w],
                "p_el_p50_w": [round(float(x), 1) for x in s.p_el_p50_w],
                "p_el_p90_w": [round(float(x), 1) for x in s.p_el_p90_w],
                "run_probability": [round(float(x), 3) for x in s.run_probability],
                "energy_wh": [round(float(x), 1) for x in s.energy_wh],
            },
            "eos_payload": self.eos.to_payload(),
        }


def run_forecast(
    df: pd.DataFrame,
    topology: Topology,
    calib: CalibrationReport,
    start_idx: int,
    horizon_h: float | None = None,
    model: ModelConfig | None = None,
    ctrl_state: ControllerState | None = None,
) -> ForecastResult:
    model = model or ModelConfig()
    horizon_h = horizon_h if horizon_h is not None else model.sim.horizon_h
    cap = model.forecast.dhw_setpoint_cap_c
    mc = model.montecarlo

    det = build_replay(
        df,
        topology,
        calib,
        start_idx,
        horizon_h,
        dhw_setpoint_cap_c=cap,
        model=model,
        ctrl_state=ctrl_state,
    )
    time_s = det.result.time_s
    events = extract_events(time_s, det.result.mode, det.result.p_el_w)

    rng = np.random.default_rng(mc.seed)
    p_el_runs: list[np.ndarray] = [det.result.p_el_w]
    for _ in range(mc.n_runs):
        rep = build_replay(
            df,
            topology,
            calib,
            start_idx,
            horizon_h,
            dhw_setpoint_cap_c=cap,
            demand_scale=float(max(0.3, rng.normal(1.0, mc.demand_sigma))),
            solar_scale=float(max(0.0, rng.normal(1.0, mc.solar_sigma))),
            c_building=float(
                model.load.c_building_j_per_k * np.exp(rng.normal(0.0, mc.c_building_sigma))
            ),
            model=model,
            ctrl_state=ctrl_state,
        )
        p_el_runs.append(rep.result.p_el_w)

    slots = aggregate_slots(
        time_s,
        p_el_runs,
        slot_s=model.forecast.slot_s,
        running_threshold_w=model.forecast.running_threshold_w,
    )
    n_slots = int(round(horizon_h * 3600 / model.forecast.slot_s))
    expected = 192 if n_slots > 96 else 96
    eos = load_profile_from_slots(
        slots, expected_slots=expected, ttl_minutes=model.forecast.eos_ttl_minutes
    )
    return ForecastResult(
        events=events, slots=slots, eos=eos, deterministic_p_el_w=det.result.p_el_w, time_s=time_s
    )


def fetch_live_weather(settings: RuntimeSettings, df: pd.DataFrame) -> pd.DataFrame | None:
    """DWD-Vorhersage über Home Assistant abholen (Start = letzte Messung).

    Liefert None, wenn keine Weather-Entity konfiguriert oder HA nicht erreichbar ist
    (Prognose fällt in dem Fall auf den 24-h-Wiederholungs-Fallback zurück).
    """
    if not settings.weather_entity:
        return None
    from hp_twin.adapters.home_assistant import HaClient, HaError
    from hp_twin.adapters.weather_forecast import fetch_weather_forecast

    settings.apply_ha_env()
    try:
        client = HaClient.from_env(settings.env_path)
        return fetch_weather_forecast(client, settings.weather_entity)
    except HaError as e:
        if settings.weather_required:
            raise
        log.warning(" Wetterprognose nicht verfügbar, Fallback 'gestern wie heute': %s", e)
        return None


def run_live_forecast(
    df: pd.DataFrame,
    topology: Topology,
    calib: CalibrationReport,
    horizon_h: float | None = None,
    model: ModelConfig | None = None,
    weather: pd.DataFrame | None = None,
    sg3_start_s: float | None = None,
    sg3_end_s: float | None = None,
    sg3_top_c: float | None = None,
    sg3_windows: list[tuple[float, float]] | None = None,
    thermal_plan: dict[str, Any] | None = None,
) -> ForecastResult:
    """ECHTE Vorwärtsprognose ab dem letzten Messpunkt ("jetzt").

    Startzustand: letzte gemessene Speichertemperaturen + Reglerzustand (läuft/Pause seit ...).
    Zukünftige Eingänge: Wetter optional über einen Vorhersage-Frame (DWD via Home Assistant),
    Rest (Soll-/Raumsollwerte, SG) letzte 24 h täglich wiederholt (siehe build_forecast_frame).

    ``sg3_windows`` injiziert alle geplanten Laufblöcke in einem einzigen Lauf, damit
    min_on/min_off und Speicherzustand zwischen den Blöcken erhalten bleiben.
    """
    model = model or ModelConfig()
    horizon_h = horizon_h if horizon_h is not None else model.sim.horizon_h
    if sg3_windows is None and sg3_start_s is not None and sg3_end_s is not None:
        sg3_windows = [(float(sg3_start_s), float(sg3_end_s))]
    frame = build_forecast_frame(df, horizon_h, weather=weather, strip_sg3=not sg3_windows)
    if sg3_windows:
        frame = _inject_sg3_windows(frame, sg3_windows, float(sg3_top_c or 46.0))
        if calib.sg3_buffer_threshold is not None:
            # The SG3 target has its own measured completion threshold. The
            # mixed SG2/SG3 median is not valid for the high-temperature run.
            model.controller.buffer_overshoot_k = calib.sg3_buffer_threshold.stop_offset_k
    cs = initial_controller_state(df, topology.controller.min_off_s)
    result = run_forecast(
        frame, topology, calib, 0, horizon_h=horizon_h, model=model, ctrl_state=cs
    )
    # Pufferverlauf für die SG-Kandidatenfindung mitliefern (det. Lauf)
    from hp_twin.adapters.replay import build_replay

    det = build_replay(
        frame,
        topology,
        calib,
        0,
        horizon_h,
        dhw_setpoint_cap_c=model.forecast.dhw_setpoint_cap_c,
        model=model,
        ctrl_state=copy.copy(cs),
    )
    result.buffer_sim_c = np.asarray(det.result.node_temps_c["buffer"], dtype=float)
    result.dhw_sim_c = np.asarray(det.result.node_temps_c["dhw"], dtype=float)
    result.frame_start = frame.index[0]
    result.outdoor_c = frame["t_outdoor_c"].astype(float).ffill().bfill().to_numpy()
    result.heat_pump = calib.heat_pump.heat_pump
    result.controller_mode = list(det.result.mode)
    reference_slots = aggregate_slots(
        result.time_s,
        [result.deterministic_p_el_w],
        slot_s=model.forecast.slot_s,
        running_threshold_w=model.forecast.running_threshold_w,
    )
    if result.frame_start is not None:
        result.eos = EosLoadProfile(
            values_w=result.eos.values_w,
            unit=result.eos.unit,
            start=result.frame_start.isoformat(),
            ttl_minutes=result.eos.ttl_minutes,
            thermal=thermal_sidecar(
                buffer_c=result.buffer_sim_c,
                dhw_c=result.dhw_sim_c,
                outdoor_c=result.outdoor_c,
                mode=result.controller_mode,
                run_probability=reference_slots.run_probability,
                running_threshold_w=model.forecast.running_threshold_w,
                time_s=result.time_s,
                start=result.frame_start,
                slot_s=model.forecast.slot_s,
                plan=thermal_plan,
            ),
        )
    return result


def _inject_sg3_windows(
    frame: pd.DataFrame, windows: list[tuple[float, float]], top_c: float
) -> pd.DataFrame:
    """Inject every scheduled run block into one consistent future frame."""
    from hp_twin.domain.sg_plan import inject_sg3_start

    dt_s = float(frame.index.to_series().diff().dropna().iloc[0].total_seconds())
    hk1 = frame["setpoint_hk1_c"].to_numpy(dtype=float)
    for start_s, end_s in sorted(windows):
        i0 = int(round(start_s / dt_s))
        i1 = int(round(end_s / dt_s))
        hk1 = inject_sg3_start(
            hk1,
            i0,
            i1,
            top_c,
            ramp_s=900.0,
            min_setpoint_c=DEFAULT_SG3_MIN_SETPOINT_C,
            dt_s=dt_s,
        )
    frame["setpoint_hk1_c"] = hk1
    return frame
