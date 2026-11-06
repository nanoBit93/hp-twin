"""Kalibrierung: Rückgewinnung bekannter Parameter aus synthetischen Daten."""

from __future__ import annotations

import numpy as np

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
