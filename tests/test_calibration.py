"""Kalibrierung: Rückgewinnung bekannter Parameter aus synthetischen Daten."""

from __future__ import annotations

import numpy as np
import pytest

from hp_twin.domain.calibration import (
    estimate_thresholds,
    fit_discharge,
    fit_heatpump_map,
)
from hp_twin.domain.heat_source import HeatPumpAirWater


def test_heatpump_recovers_power_slopes() -> None:
    true = HeatPumpAirWater(
        p_el_ref_w=2000, standby_w=80, dp_dt_flow_w_per_k=60, dp_dt_out_w_per_k=-25, eta_carnot=0.55
    )
    rng = np.random.default_rng(0)
    to = rng.uniform(0, 12, 2000)
    tf = rng.uniform(30, 45, 2000)
    pe = np.array([true.output(o, f, True).p_el_w for o, f in zip(to, tf, strict=True)])
    qt = np.array([true.output(o, f, True).q_th_w for o, f in zip(to, tf, strict=True)])
    fit = fit_heatpump_map(to, tf, pe, qt, standby_w=80)
    assert abs(fit.heat_pump.dp_dt_flow_w_per_k - 60) < 1.0
    assert abs(fit.heat_pump.dp_dt_out_w_per_k - (-25)) < 1.0
    assert abs(fit.heat_pump.eta_carnot - 0.55) < 0.02
    assert fit.p_el_r2 > 0.99 and fit.q_th_r2 > 0.97


def test_threshold_estimate() -> None:
    # zwei Takte: Start bei T-Soll = -4, Stopp bei +8
    sp = np.full(20, 30.0)
    ts = np.full(20, 30.0)
    on = np.zeros(20, int)
    # Takt 1: an von 3..6
    ts[2] = 26.0  # Start-Offset -4 (Schritt vor dem Einschalten)
    on[3:7] = 1
    ts[6] = 38.0  # Stopp-Offset +8 (letzter An-Schritt)
    # Takt 2: an von 12..15
    ts[11] = 26.0
    on[12:16] = 1
    ts[15] = 38.0
    est = estimate_thresholds(on, ts, sp, circuit="buffer")
    assert est.n_starts == 2 and est.n_stops == 2
    assert abs(est.start_offset_k - (-4.0)) < 1e-6
    assert abs(est.stop_offset_k - 8.0) < 1e-6


def test_sg3_threshold_uses_only_completed_elevated_setpoint_runs() -> None:
    sp = np.full(40, 30.0)
    ts = np.full(40, 30.0)
    on = np.zeros(40, int)
    sp[5:15] = 50.0
    on[5:15] = 1
    ts[14] = 51.6
    # An early abort below the elevated target is not a completed SG3 run.
    sp[20:30] = 50.0
    on[20:25] = 1
    ts[24] = 38.0
    est = estimate_thresholds(
        on, ts, sp, circuit="sg3_buffer", min_setpoint_c=38.0, target_tolerance_k=0.5
    )
    assert est.n_stops == 1
    assert est.stop_offset_k == pytest.approx(1.6)


def test_stored_calibration_gains_only_sg3_threshold_from_history(tmp_path) -> None:
    import json

    from hp_twin.adapters.settings import RuntimeSettings
    from hp_twin.cli import _calibrate_report, resolve_calibration
    from hp_twin.domain.calibration import calibration_to_json
    from hp_twin.domain.model_config import ModelConfig
    from scripts_helper import synthetic_frame

    csv = tmp_path / "measurements.csv"
    calibration = tmp_path / "calibration.json"
    frame = synthetic_frame(days=6)
    frame["flow_rate_lpm"] = np.where(frame["compressor_on"] == 1, 20.0, 0.0)
    frame["setpoint_hk1_c"] = np.where(frame["compressor_on"] == 1, 49.8, 31.5)
    frame["t_hk1_c"] = np.where(frame["compressor_on"] == 1, 51.4, 25.95)
    frame.to_csv(csv, index=False)
    _, _, calib = _calibrate_report("config/example_topology.yaml", str(csv))
    original_stop = calib.buffer_threshold.stop_offset_k
    calib.sg3_buffer_threshold = None
    calibration.write_text(json.dumps(calibration_to_json(calib)), encoding="utf-8")
    settings = RuntimeSettings(
        topology_path="config/example_topology.yaml",
        csv_path=str(csv),
        calibration_path=str(calibration),
        use_stored_calibration=True,
    )
    _, _, restored = resolve_calibration(settings, ModelConfig())
    assert restored.buffer_threshold.stop_offset_k == pytest.approx(original_stop)
    assert restored.sg3_buffer_threshold is not None
    assert restored.sg3_buffer_threshold.stop_offset_k == pytest.approx(1.6)


def test_discharge_recovers_ua() -> None:
    ua_true, c = 120.0, 300 * 4186.0
    dt = 60.0
    room = 21.0
    n = 2000
    tb = np.zeros(n)
    tb[0] = 40.0
    for i in range(1, n):
        tb[i] = tb[i - 1] - ua_true * (tb[i - 1] - room) / c * dt
    # Treiber = (T_puffer - T_raum), der die Entladung im Synthetik-Modell antreibt
    fit = fit_discharge(tb, tb - room, np.zeros(n), dt_s=dt, volume_l=300.0)
    assert abs(fit.ua_w_per_k - ua_true) < 5.0
    assert fit.r2 > 0.98
