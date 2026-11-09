"""Echte Vorwärtsprognose: Startzustand, Eingangsrahmen, Export-Planung."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from hp_twin.adapters.io_csv import validate_frame
from hp_twin.adapters.replay import build_forecast_frame, initial_controller_state
from hp_twin.adapters.service import plan_export
from hp_twin.adapters.settings import RuntimeSettings
from hp_twin.domain.controller import Mode
from scripts_helper import synthetic_frame


def _df(days: int = 3) -> pd.DataFrame:
    return validate_frame(synthetic_frame(days=days))


def test_forecast_frame_starts_after_last_measurement() -> None:
    df = _df()
    fr = build_forecast_frame(df, horizon_h=48.0)
    dt = df.index[1] - df.index[0]
    assert fr.index[0] == df.index[-1] + dt  # Zukunft beginnt direkt nach "jetzt"
    assert len(fr) == int(48 * 3600 / dt.total_seconds())
    # Anfangszustand = letzte Messung, danach keine Messgrößen mehr
    assert fr["t_hk1_c"].iloc[0] == df["t_hk1_c"].iloc[-1]
    assert fr["t_dhw_c"].iloc[0] == df["t_dhw_c"].iloc[-1]
    assert fr["t_hk1_c"].iloc[1:].isna().all()


def test_forecast_frame_repeats_last_24h() -> None:
    df = _df()
    fr = build_forecast_frame(df, horizon_h=48.0)
    per_day = int(86400 / (df.index[1] - df.index[0]).total_seconds())
    # Eingänge des nächsten Tages = gestern, und der Tag danach identisch
    assert np.allclose(fr["t_outdoor_c"].iloc[:per_day], df["t_outdoor_c"].iloc[-per_day:])
    assert np.allclose(
        fr["t_outdoor_c"].iloc[per_day : 2 * per_day], fr["t_outdoor_c"].iloc[:per_day]
    )


def test_forecast_frame_strips_sg3_by_default() -> None:
    df = _df()
    # SG3 in der Sollwertspur der letzten 24 h: HK1 32 -> 44 °C Block
    df.loc[df.index[-400:-200], "setpoint_hk1_c"] = 44.0
    fr_stripped = build_forecast_frame(df, horizon_h=48.0, strip_sg3=True)
    fr_raw = build_forecast_frame(df, horizon_h=48.0, strip_sg3=False)
    per_day = int(86400 / (df.index[1] - df.index[0]).total_seconds())
    # Ohne Strip: Spur wiederholt gestern inkl. 44-°C-Block; mit Strip: max auf SG2-Niveau
    assert fr_raw["setpoint_hk1_c"].max() == pytest.approx(44.0)
    assert fr_stripped["setpoint_hk1_c"].max() < fr_raw["setpoint_hk1_c"].max()
    assert np.allclose(
        fr_raw["setpoint_hk1_c"].iloc[:per_day], df["setpoint_hk1_c"].iloc[-per_day:]
    )


def test_eos_thermal_sidecar_uses_the_same_final_forecast() -> None:
    from hp_twin.adapters.forecast import run_live_forecast
    from hp_twin.adapters.io_yaml import load_topology
    from hp_twin.domain.calibration import (
        CalibrationReport,
        DischargeFit,
        HeatPumpFit,
        ThresholdEstimate,
    )
    from hp_twin.domain.heat_source import HeatPumpAirWater
    from hp_twin.domain.model_config import ModelConfig

    df = _df()
    calib = CalibrationReport(
        heat_pump=HeatPumpFit(HeatPumpAirWater(), 1, 0.0, 0.0, 0.0, 0.0, 3.0),
        buffer_threshold=ThresholdEstimate("buffer", 1, 1, -5.0, 2.0),
        discharge=DischargeFit(100.0, 0.0, 0.0, 1, 1.0),
    )
    model = ModelConfig()
    model.montecarlo.n_runs = 0
    result = run_live_forecast(
        df, load_topology("config/example_topology.yaml"), calib, model=model
    )
    payload = result.eos.to_payload()
    thermal = payload["thermal"]
    assert thermal["start"] == result.frame_start.isoformat()
    assert thermal["resolution"] == "15min"
    assert thermal["series"]["buffer_c"][0] == pytest.approx(
        float(np.mean(result.buffer_sim_c[:3]))
    )
    assert thermal["series"]["dhw_c"][0] == pytest.approx(
        float(np.mean(result.dhw_sim_c[:3])), abs=0.002
    )
    assert thermal["series"]["mode"][0] in ("idle", "heat", "dhw", "pause")
    assert len(thermal["series"]["buffer_c"]) == len(payload["values"])


def test_adopted_thermal_payload_carries_the_actual_eos_plan() -> None:
    from hp_twin.adapters.eos_connect import plan_signature

    status = {
        "plan_anchor": "2026-01-13T00:00:00+00:00",
        "time_frame_base": 900,
        "fallback_active": False,
        "plan": [0.0] * 8 + [400.0, 400.0] + [0.0] * 182,
    }
    plan = plan_signature(status, pd.Timestamp("2026-01-13T00:00:00Z"))
    assert plan["signature"] == "2026-01-13T00:00:00+00:00|8-10"
    assert plan["start_s"] == 7200.0
    assert plan["end_s"] == 9000.0
    assert plan["contingent_id"] == "sg3_req"


def test_adopted_plan_signature_keeps_midnight_slot_indices() -> None:
    from hp_twin.adapters.eos_connect import plan_signature

    status = {
        "plan_anchor": "2026-01-13T00:00:00+00:00",
        "time_frame_base": 900,
        "fallback_active": False,
        "plan": [0.0] * 20 + [400.0, 400.0] + [0.0] * 170,
    }
    plan = plan_signature(status, pd.Timestamp("2026-01-13T02:00:00Z"))
    assert plan["signature"] == "2026-01-13T00:00:00+00:00|20-22"
    assert plan["start_s"] == 10800.0
    assert plan["end_s"] == 12600.0


def test_partial_first_heating_slot_is_visible_in_thermal_mode() -> None:
    from hp_twin.adapters.eos_connect import thermal_sidecar

    times = np.arange(60) * 60.0
    buffer = np.full(60, 28.0)
    buffer[12:] += np.arange(48) * 0.1
    mode = ["idle"] * 12 + ["heat"] * 48
    result = thermal_sidecar(
        buffer_c=buffer,
        dhw_c=np.full(60, 45.0),
        outdoor_c=np.full(60, 8.0),
        mode=mode,
        time_s=times,
        start=pd.Timestamp("2026-01-13T00:03:00Z"),
    )
    assert result["series"]["buffer_c"][1] > result["series"]["buffer_c"][0]
    assert result["series"]["mode"][1] == "heat"
    # Three minutes of real operation at the end of the previous slot count too.
    assert result["series"]["mode"][0] == "heat"


def test_thermal_operating_band_uses_the_electrical_slot_occupancy() -> None:
    from hp_twin.adapters.eos_connect import thermal_sidecar

    times = np.arange(60) * 60.0
    result = thermal_sidecar(
        buffer_c=np.full(60, 30.0),
        dhw_c=np.full(60, 45.0),
        outdoor_c=np.full(60, 8.0),
        mode=["idle"] * 60,
        run_probability=[0.0, 1 / 15, 1.0, 0.0],
        running_threshold_w=200.0,
        time_s=times,
        start=pd.Timestamp("2026-01-13T00:03:00Z"),
    )
    # Less than one simulation step in a slot is still visible through the
    # electrical profile's run probability; the band cannot start one slot late.
    assert result["run_probability"][1] > 0
    assert result["run_probability"][0] == 0.0
    assert result["running_threshold_w"] == 200.0


def test_thermal_occupancy_follows_the_same_reference_run_as_temperature() -> None:
    from hp_twin.adapters.forecast import run_live_forecast
    from hp_twin.adapters.io_yaml import load_topology
    from hp_twin.domain.calibration import (
        CalibrationReport,
        DischargeFit,
        HeatPumpFit,
        ThresholdEstimate,
    )
    from hp_twin.domain.heat_source import HeatPumpAirWater
    from hp_twin.domain.model_config import ModelConfig
    from hp_twin.domain.montecarlo import aggregate_slots

    model = ModelConfig()
    model.montecarlo.n_runs = 3
    calib = CalibrationReport(
        heat_pump=HeatPumpFit(HeatPumpAirWater(), 1, 0.0, 0.0, 0.0, 0.0, 3.0),
        buffer_threshold=ThresholdEstimate("buffer", 1, 1, -5.0, 2.0),
        discharge=DischargeFit(100.0, 0.0, 0.0, 1, 1.0),
    )
    result = run_live_forecast(
        _df(), load_topology("config/example_topology.yaml"), calib, model=model
    )
    reference = aggregate_slots(
        result.time_s,
        [result.deterministic_p_el_w],
        slot_s=model.forecast.slot_s,
        running_threshold_w=model.forecast.running_threshold_w,
    )
    assert np.allclose(
        result.eos.thermal["run_probability"], reference.run_probability, atol=0.0001
    )


def test_forecast_needs_24h() -> None:
    df = _df().iloc[:100]
    try:
        build_forecast_frame(df, 48.0)
    except ValueError as e:
        assert "24 h" in str(e)
    else:  # pragma: no cover
        raise AssertionError("ValueError erwartet")


def test_initial_controller_state_running_and_pause() -> None:
    df = _df()
    dt = (df.index[1] - df.index[0]).total_seconds()
    df["compressor_on"] = 0
    df["mode"] = "idle"
    # läuft seit 4 Schritten
    df.iloc[-4:, df.columns.get_loc("compressor_on")] = 1
    df.iloc[-4:, df.columns.get_loc("mode")] = "heat"
    cs = initial_controller_state(df, min_off_s=1800)
    assert cs.mode is Mode.HEAT_BUFFER and abs(cs.time_in_mode_s - 3 * dt) < 1e-6

    # gerade gestoppt (< min_off) -> PAUSE, später -> IDLE
    df2 = _df()
    df2["compressor_on"] = 0
    df2["mode"] = "idle"
    df2.iloc[-30:-2, df2.columns.get_loc("compressor_on")] = 1  # lief bis vor 2 Schritten
    cs2 = initial_controller_state(df2, min_off_s=1800)
    assert cs2.mode is Mode.PAUSE and cs2.time_since_stop_s < 1800
    df2.iloc[-30:-2, df2.columns.get_loc("compressor_on")] = 0
    df2.iloc[-100:-80, df2.columns.get_loc("compressor_on")] = 1  # lange her
    assert initial_controller_state(df2, min_off_s=1800).mode is Mode.IDLE


def test_plan_export_backfill_then_delta(tmp_path: Path) -> None:
    csv = tmp_path / "m.csv"
    s = RuntimeSettings(csv_path=str(csv), backfill_days=10, min_history_days=3)
    days, append, phase = plan_export(s)  # keine Datei -> Backfill
    assert days == 10 and append is False and "Historie" in phase

    # frische, lange Historie -> nur Delta anhängen
    df = synthetic_frame(days=5)
    now = pd.Timestamp.now(tz="UTC").floor("5min")
    df["timestamp"] = (now - pd.to_timedelta(np.arange(len(df))[::-1] * 5, unit="min")).strftime(
        "%Y-%m-%dT%H:%M:%SZ"
    )
    df.to_csv(csv, index=False)
    days, append, _ = plan_export(s)
    assert append is True and days <= 1.0

    # zu kurze Historie -> wieder Backfill
    df.tail(50).to_csv(csv, index=False)
    assert plan_export(s)[1] is False


def test_short_nan_gap_does_not_break_calibration() -> None:
    """Regression: NaN-Zeilen am Export-Ende ließen den Entlade-Fit mit LinAlgError scheitern."""
    from hp_twin.domain.calibration import fit_discharge

    n = 3000
    rng = np.random.default_rng(0)
    tb = 40 - np.cumsum(np.abs(rng.normal(0.01, 0.005, n)))
    drv = np.full(n, 20.0)
    drv[-3:] = np.nan  # Sensorlücke am Ende
    fit = fit_discharge(tb, drv, np.zeros(n), dt_s=60.0, volume_l=300.0, ghi_w_m2=np.full(n, 100.0))
    assert np.isfinite(fit.ua_w_per_k) and np.isfinite(fit.base_load_w)


def test_csv_reader_fills_short_gaps() -> None:
    df = synthetic_frame(days=1)
    df.loc[df.index[-3:], "t_outdoor_c"] = np.nan
    out = validate_frame(df)
    assert not out["t_outdoor_c"].isna().any()
