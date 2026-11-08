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
