"""Replay/Backtest-Orchestrierung: baut aus Messdaten + Kalibrierung einen Lauf und vergleicht.

Startzustand aus Messdaten, Vorwärtssimulation mit GEMESSENEM Wetter und GEMESSENEN Sollwerten
(trennt Modell- von Prognosefehlern). Gibt simulierte + gemessene Zeitreihen zurück.
"""

from __future__ import annotations

import copy
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from hp_twin.adapters.io_csv import room_columns
from hp_twin.adapters.weather_forecast import merge_forecast_columns
from hp_twin.domain.building import RC2R2C, BuildingState
from hp_twin.domain.calibration import CalibrationReport
from hp_twin.domain.consumers import CompositeLoad, HeatingCircuitLoad, MultiZoneHeatingLoad
from hp_twin.domain.controller import Controller, ControllerState, Mode, Setpoints
from hp_twin.domain.model_config import ModelConfig
from hp_twin.domain.simulator import SimConfig, SimResult, Simulator, StepTrace
from hp_twin.domain.storage import StorageState, StratifiedStorage
from hp_twin.domain.topology import Topology


def _heat_load_ua_by_key(
    heat_load_path: str | None, ha_mapping_path: str | None
) -> dict[str, float]:
    """Design-UA je climate-Mapping-Key (z. B. 'wohnzimmer') aus Heizlast + ha_mapping.climate."""
    if not heat_load_path or not ha_mapping_path:
        return {}
    if not Path(heat_load_path).exists() or not Path(ha_mapping_path).exists():
        return {}
    from hp_twin.adapters.ha_history import HaMapping
    from hp_twin.adapters.io_yaml import load_heat_load

    heat = load_heat_load(heat_load_path)
    climate = HaMapping.load(ha_mapping_path).climate or {}
    entity_to_key = {ent: key for key, ent in climate.items()}
    out: dict[str, float] = {}
    for room in heat.rooms:
        if room.climate_entity and room.climate_entity in entity_to_key:
            key = entity_to_key[room.climate_entity]
            out[key] = out.get(key, 0.0) + room.ua_w_per_k(heat.design_outdoor_c)
    return out


@dataclass
class ReplayOutput:
    result: SimResult
    measured: pd.DataFrame  # gleicher Zeitraum, gemessene Spalten
    buffer_node: str
    dhw_node: str


def _ua_split(
    ua_total: float, ua_air_mass: float = 300.0, air_out_fraction: float = 0.5
) -> tuple[float, float, float]:
    """Verteilt effektives Gebäude-UA auf die drei 2R2C-Pfade (plausible Aufteilung)."""
    ua_air_out = air_out_fraction * ua_total
    rest = ua_total - ua_air_out  # = Serienschaltung(air_mass, mass_out)
    # 1/rest = 1/ua_air_mass + 1/ua_mass_out  ->  ua_mass_out
    inv = max(1.0 / rest - 1.0 / ua_air_mass, 1.0 / 1e6)
    return ua_air_out, ua_air_mass, 1.0 / inv


def initial_controller_state(df: pd.DataFrame, min_off_s: float) -> ControllerState:
    """Reglerzustand "jetzt" aus der Historie: läuft die WP, bzw. wie lange steht sie schon.

    Entscheidend für den Zeitpunkt des nächsten Starts (Mindeststillstand/Pause).
    """
    dt_s = float(df.index.to_series().diff().dropna().iloc[0].total_seconds())
    on = df["compressor_on"].fillna(0).to_numpy().astype(int)
    if len(on) == 0 or not on.any():
        return ControllerState()  # nie gelaufen -> darf sofort starten
    if on[-1] == 1:
        start = len(on) - 1
        while start > 0 and on[start - 1] == 1:
            start -= 1
        dhw = str(df["mode"].iloc[-1]) == "dhw"
        return ControllerState(
            mode=Mode.HEAT_DHW if dhw else Mode.HEAT_BUFFER,
            time_in_mode_s=(len(on) - 1 - start) * dt_s,
            time_since_stop_s=0.0,
            active_node="dhw" if dhw else "buffer",
        )
    last_on = int(np.where(on == 1)[0][-1])
    since_stop = (len(on) - 1 - last_on) * dt_s
    return ControllerState(
        mode=Mode.PAUSE if since_stop < min_off_s else Mode.IDLE,
        time_in_mode_s=since_stop,
        time_since_stop_s=since_stop,
    )


def build_forecast_frame(
    df: pd.DataFrame,
    horizon_h: float,
    weather: pd.DataFrame | None = None,
    strip_sg3: bool = True,
) -> pd.DataFrame:
    """Eingangsrahmen für eine ECHTE Vorwärtsprognose ab dem letzten Messpunkt.

    Anfangszustand = letzte Messwerte (Speicher). Zukünftige Eingänge (Außentemperatur, Strahlung,
    Soll-/Raumsollwerte, SG) = die letzten 24 h, täglich wiederholt ("gestern wie heute"). Ein
    Wetter-Vorhersage-Frame (z. B. DWD über Home Assistant, adapters/weather_forecast) ersetzt
    t_outdoor_c/ghi_wm2 im Vorhersagezeitraum; danach gilt weiter der Wiederholungs-Fallback.

    strip_sg3=True (Standard) ersetzt die SG3-Blöcke in den Sollwert-Spuren durch das
    SG2-Grundprogramm (domain/sg_plan): Die Basis-Prognose ist dann "SG2 only" und SG3-
    Läufe werden nur dort eingeplant, wo der SG-Orchestrator sie bewusst setzt - nicht
    als stille Wiederholung des gestrigen Tagesverlaufs.
    """
    dt_s = float(df.index.to_series().diff().dropna().iloc[0].total_seconds())
    per_day = int(round(86400.0 / dt_s))
    if len(df) < per_day:
        raise ValueError("Für die Prognose werden mindestens 24 h Messdaten benötigt")
    n = int(round(horizon_h * 3600.0 / dt_s))
    d24 = df.iloc[-per_day:]
    idx = (np.arange(1, n + 1) - 1) % per_day
    fut = d24.iloc[idx].copy()
    fut.index = df.index[-1] + pd.to_timedelta(np.arange(1, n + 1) * dt_s, unit="s")
    last = df.iloc[-1]
    # Messgrößen der Zukunft gibt es nicht; Anfangszustand = jetzt gemessen
    for col in ("t_dhw_c", "t_hk1_c", "t_hk2_c", "compressor_on", "p_el_w", "q_heat_wh"):
        if col in fut.columns:
            fut[col] = np.nan
    fut["t_dhw_c"] = fut["t_dhw_c"].astype(float)
    fut["t_hk1_c"] = fut["t_hk1_c"].astype(float)
    first = fut.index[0]
    fut.loc[first, "t_dhw_c"] = last["t_dhw_c"]
    fut.loc[first, "t_hk1_c"] = last["t_hk1_c"]
    for col in room_columns(fut.reset_index()):
        fut[col] = last[col]
    if weather is not None and len(weather):
        fut = merge_forecast_columns(fut, weather)
    if strip_sg3:
        fut = _strip_sg3_columns(fut)
    return fut


def _strip_sg3_columns(fut: pd.DataFrame) -> pd.DataFrame:
    """SG3-Blöcke in den Sollwert-Spuren des Zukunftsframes durch SG2-Grundprogramm ersetzen."""
    from hp_twin.domain.sg_plan import sg3_mask, strip_sg3_setpoints

    if "setpoint_hk1_c" not in fut.columns or fut.empty:
        return fut
    hk1 = fut["setpoint_hk1_c"].to_numpy(dtype=float)
    sg_status = fut["sg_status"].to_numpy(dtype=float) if "sg_status" in fut.columns else None
    mask = sg3_mask(sg_status, hk1)
    if not mask.any():
        return fut
    series = {"setpoint_hk1_c": hk1}
    if "setpoint_dhw_c" in fut.columns:
        series["setpoint_dhw_c"] = fut["setpoint_dhw_c"].to_numpy(dtype=float)
    for col in [c for c in fut.columns if c.startswith("setpoint_room_") and c.endswith("_c")]:
        series[col] = fut[col].to_numpy(dtype=float)
    stripped = strip_sg3_setpoints(series, mask)
    for name, vals in stripped.items():
        fut[name] = vals
    return fut


def build_replay(
    df: pd.DataFrame,
    topology: Topology,
    calib: CalibrationReport,
    start_idx: int,
    horizon_h: float,
    heat_load_path: str | None = "config/heizlast.yaml",
    ha_mapping_path: str | None = "config/ha_mapping.yaml",
    c_building: float | None = None,
    solar_scale: float = 1.0,
    dhw_setpoint_cap_c: float | None = None,
    demand_scale: float = 1.0,
    model: ModelConfig | None = None,
    ctrl_state: ControllerState | None = None,
    room_setpoint_override: np.ndarray | None = None,
    trace_callback: Callable[[float, int, StepTrace], None] | None = None,
) -> ReplayOutput:
    model = model or ModelConfig()
    lp = model.load
    c_building = lp.c_building_j_per_k if c_building is None else c_building
    frost_c = lp.frost_threshold_c
    dt_s = float(df.index.to_series().diff().dropna().iloc[0].total_seconds())
    n = int(round(horizon_h * 3600.0 / dt_s))
    sl = df.iloc[start_idx : start_idx + n]
    if len(sl) < n:
        n = len(sl)

    # Knoten
    dhw_node, buffer_node = "dhw", "buffer"
    storage = StratifiedStorage.from_topology(topology.storage)
    state0 = StorageState(
        np.array([float(sl["t_dhw_c"].iloc[0]), float(sl["t_hk1_c"].iloc[0])], dtype=float)
    )

    # Messarrays -> Provider (Index = round(t_s/dt)). Lücken (NaN) füllen, sonst schlagen sie durch.
    def _filled(col: str) -> np.ndarray:
        s = sl[col].astype(float).ffill().bfill()
        return s.fillna(0.0).to_numpy()

    out_arr = _filled("t_outdoor_c")
    ghi_arr = _filled("ghi_wm2")
    sp_buf = _filled("setpoint_hk1_c")
    sp_dhw = _filled("setpoint_dhw_c")
    if dhw_setpoint_cap_c is not None:
        # neue Konfig: DHW-Soll gedeckelt (kein SG3-DHW-Boost) -> DHW passiv über HK1/Kopplung
        sp_dhw = np.minimum(sp_dhw, dhw_setpoint_cap_c)
    rooms = room_columns(sl.reset_index())

    # Raumsoll-Verlauf aus gemessenen climate-Sollwerten der AKTIVEN Räume (Soll > Frostschutz).
    # Dieser enthält bereits den SG3-+2°C-Boost -> Masse lädt -> längerer Leerlauf danach.
    sp_room_cols = [c for c in sl.columns if c.startswith("setpoint_room_") and c.endswith("_c")]
    if sp_room_cols:
        sp_room = sl[sp_room_cols].to_numpy(dtype=float)
        active = sp_room > frost_c
        with np.errstate(invalid="ignore"):
            room_set_arr = np.where(
                active.any(axis=1),
                np.nansum(np.where(active, sp_room, 0.0), axis=1)
                / np.clip(active.sum(axis=1), 1, None),
                np.nan,
            )
        fallback = float(np.nanmedian(room_set_arr)) if np.isfinite(room_set_arr).any() else 20.5
        room_set_arr = np.where(np.isfinite(room_set_arr), room_set_arr, fallback)
        if room_setpoint_override is not None:
            room_set_arr = np.asarray(room_setpoint_override, dtype=float)[:n]
    else:
        room_set_arr = np.full(n, float(np.median(sl[rooms].to_numpy())))
        if room_setpoint_override is not None:
            room_set_arr = np.asarray(room_setpoint_override, dtype=float)[:n]
    room0 = float(room_set_arr[0])

    def idx(t_s: float) -> int:
        return min(int(round(t_s / dt_s)), n - 1)

    def outdoor(t_s: float) -> float:
        return float(out_arr[idx(t_s)])

    def ghi(t_s: float) -> float:
        return float(ghi_arr[idx(t_s)])

    cp = model.controller
    over = (
        cp.buffer_overshoot_k
        if cp.buffer_overshoot_k is not None
        else calib.buffer_threshold.stop_offset_k
    )
    hyst = (
        cp.buffer_hysteresis_k
        if cp.buffer_hysteresis_k is not None
        else abs(calib.buffer_threshold.start_offset_k)
    )

    def room_setpoint(t_s: float) -> float:
        return float(room_set_arr[idx(t_s)])

    def setpoints(t_s: float) -> Setpoints:
        i = idx(t_s)
        return Setpoints(
            dhw_c=float(sp_dhw[i]),
            buffer_c=float(sp_buf[i]),
            buffer_hysteresis_k=hyst,
            buffer_overshoot_k=over,
            dhw_hysteresis_k=cp.dhw_hysteresis_k,
            dhw_overshoot_k=cp.dhw_overshoot_k,
        )

    # Lastmodell: Multi-Zone (Heizlast + aktive Räume) falls verfügbar, sonst kalibriertes 1R2C.
    solar = max(calib.discharge.solar_w_per_wm2, 0.0) * solar_scale
    key_to_ua = _heat_load_ua_by_key(heat_load_path, ha_mapping_path)
    col_keys = [c[len("setpoint_room_") : -len("_c")] for c in sp_room_cols]
    usable = sp_room_cols and any(k in key_to_ua for k in col_keys)
    if usable:
        ua_cols = np.array([key_to_ua.get(k, 0.0) for k in col_keys])
        sp_room = sl[sp_room_cols].to_numpy(dtype=float)
        ua_active_arr = ((sp_room > frost_c) * ua_cols).sum(axis=1)
        ua_active_arr = np.where(ua_active_arr > 0, ua_active_arr, float(np.median(ua_active_arr)))
        # Höhe auf kalibrierte Gesamt-UA skalieren (mehrere beheizte Räume noch ohne Thermostat-
        # Mapping) -> zeitliche Struktur aus aktiven Räumen bleibt, Magnitude stimmt zur Messung.
        mean_active = float(np.mean(ua_active_arr[ua_active_arr > 0]))
        if mean_active > 0 and calib.discharge.ua_w_per_k > 0:
            ua_active_arr = ua_active_arr * (calib.discharge.ua_w_per_k / mean_active)
        ua_active_arr = ua_active_arr * demand_scale  # MC-Streuung auf die Entladerate

        def ua_active(t_s: float) -> float:
            return float(ua_active_arr[idx(t_s)])

        load: CompositeLoad = CompositeLoad(
            [
                MultiZoneHeatingLoad(
                    node=buffer_node,
                    c_building_j_per_k=c_building,
                    ua_active=ua_active,
                    room_setpoint=room_setpoint,
                    outdoor=outdoor,
                    ghi=ghi,
                    t_room_c=room0,
                    solar_w_per_wm2=solar,
                    emission_ua_w_per_k=lp.emission_ua_w_per_k,
                    throttle_band_k=lp.throttle_band_k,
                    internal_gain_w=lp.internal_gain_w,
                )
            ]
        )
    else:
        ua_air_out, ua_air_mass, ua_mass_out = _ua_split(
            calib.discharge.ua_w_per_k, lp.ua_air_mass_w_per_k, lp.ua_air_out_fraction
        )
        building = RC2R2C(
            ua_air_out_w_per_k=ua_air_out,
            ua_air_mass_w_per_k=ua_air_mass,
            ua_mass_out_w_per_k=ua_mass_out,
            solar_aperture_m2=solar,
        )
        load = CompositeLoad(
            [
                HeatingCircuitLoad(
                    node=buffer_node,
                    building=building,
                    state=BuildingState(room0, room0),
                    outdoor=outdoor,
                    ghi=ghi,
                    room_setpoint=room_setpoint,
                    emission_ua_w_per_k=lp.fallback_emission_ua_w_per_k,
                    throttle_band_k=lp.throttle_band_k,
                    internal_gain_w=max(calib.discharge.base_load_w, 0.0),
                )
            ]
        )

    hp = calib.heat_pump.heat_pump
    controller = Controller(config=topology.controller, dhw_node=dhw_node, buffer_node=buffer_node)
    sim = Simulator(
        storage,
        hp,
        controller,
        SimConfig(timestep_s=dt_s, horizon_s=n * dt_s),
        condenser_approach_k=model.sim.condenser_approach_k,
    )
    result = sim.run(
        state0,
        setpoints=setpoints,
        load=load,
        outdoor=outdoor,
        ctrl_state=copy.copy(ctrl_state) if ctrl_state is not None else None,
        trace_callback=trace_callback,
    )
    return ReplayOutput(result=result, measured=sl, buffer_node=buffer_node, dhw_node=dhw_node)
