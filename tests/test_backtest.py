"""Backtest-Metriken: Selbstkonsistenz und Start-Fehler."""

from __future__ import annotations

import numpy as np

from hp_twin.domain.backtest import (
    compute_metrics,
    cycles_per_day,
    energy_per_window_wh,
    next_start_error_s,
    start_times_s,
)


def _square_on(n: int, period: int, duty: int, dt: float) -> np.ndarray:
    on = np.zeros(n, int)
    for s in range(0, n, period):
        on[s : s + duty] = 1
    return on


def test_start_detection_and_error() -> None:
    t = np.arange(100) * 60.0
    on = np.zeros(100, int)
    on[10:20] = 1
    on[50:60] = 1
    starts = start_times_s(t, on)
    assert list(starts) == [600.0, 3000.0]
    # Verschiebung um +5 min beim ersten Start
    on2 = np.zeros(100, int)
    on2[15:25] = 1
    assert next_start_error_s(start_times_s(t, on2), starts) == 300.0


def test_energy_window() -> None:
    t = np.arange(720) * 60.0  # 12 h
    p = np.full(720, 1000.0)
    e = energy_per_window_wh(t, p, window_s=6 * 3600)
    assert len(e) == 2
    assert abs(e[0] - 6000.0) < 1e-6  # 1000 W * 6 h = 6000 Wh


def test_cycles_per_day() -> None:
    dt = 60.0
    n = 1440  # 1 Tag
    on = np.zeros(n, int)
    for s in range(100, n, 288):  # führendes "aus" -> 5 Einschaltflanken/Tag
        on[s : s + 30] = 1
    t = np.arange(n) * dt
    assert abs(cycles_per_day(t, on) - 5.0) < 0.3


def test_self_consistency_zero_error() -> None:
    """Identische sim/meas -> alle Fehler null."""
    t = np.arange(600) * 60.0
    on = _square_on(600, 120, 30, 60.0)
    p = on * 2000.0 + 80.0
    buf = 30 + 5 * np.sin(t / 3600)
    m = compute_metrics(t, on, on, p, p, buf, buf)
    assert m.next_start_error_min == 0.0
    assert m.buffer_rmse_c == 0.0
    assert abs(m.energy_6h_mae_wh) < 1e-6
    assert m.cycles_per_day_sim == m.cycles_per_day_meas
