"""Backtest-Metriken: Vergleich simulierter vs. gemessener WP-Verläufe.

Fokus auf das, was für die Lastprognose zählt:
- Fehler beim Zeitpunkt des nächsten Starts und der Starts 2-5
- Energiefehler je 6-h-Fenster
- Abweichung der Zyklenzahl pro Tag
- stündlicher RMSE (ergänzend)
Alle Funktionen arbeiten auf numpy-Arrays (IO-frei).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def start_indices(compressor_on: np.ndarray) -> np.ndarray:
    """Indizes der Einschaltflanken (0 -> 1)."""
    on = np.asarray(compressor_on, float).astype(int)
    idx: np.ndarray = np.where(np.diff(on) == 1)[0] + 1
    return idx


def start_times_s(time_s: np.ndarray, compressor_on: np.ndarray) -> np.ndarray:
    times: np.ndarray = np.asarray(time_s, float)[start_indices(compressor_on)]
    return times


def next_start_error_s(sim_starts_s: np.ndarray, meas_starts_s: np.ndarray) -> float:
    """Signierter Fehler (sim - gemessen) des ersten Starts in Sekunden (nan wenn keiner)."""
    if len(sim_starts_s) == 0 or len(meas_starts_s) == 0:
        return float("nan")
    return float(sim_starts_s[0] - meas_starts_s[0])


def start_errors_s(sim_starts_s: np.ndarray, meas_starts_s: np.ndarray, k: int = 5) -> list[float]:
    """Fehler der ersten k Starts (paarweise nach Reihenfolge), nan wo einer fehlt."""
    out: list[float] = []
    for i in range(k):
        if i < len(sim_starts_s) and i < len(meas_starts_s):
            out.append(float(sim_starts_s[i] - meas_starts_s[i]))
        else:
            out.append(float("nan"))
    return out


def energy_per_window_wh(
    time_s: np.ndarray, p_el_w: np.ndarray, window_s: float = 6 * 3600.0
) -> np.ndarray:
    """Energie (Wh) je Zeitfenster."""
    t = np.asarray(time_s, float)
    p = np.asarray(p_el_w, float)
    dt = float(t[1] - t[0]) if len(t) > 1 else 60.0
    per = max(int(round(window_s / dt)), 1)
    n_win = len(p) // per
    return np.array([p[i * per : (i + 1) * per].sum() * dt / 3600.0 for i in range(n_win)])


def cycles_per_day(time_s: np.ndarray, compressor_on: np.ndarray) -> float:
    n = len(start_indices(compressor_on))
    span_days = (float(time_s[-1] - time_s[0]) / 86400.0) if len(time_s) > 1 else 1.0
    return n / span_days if span_days > 0 else float(n)


def rmse(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    d = (a - b).astype(float)
    # (a-b)**2 kann bei sehr großen Werten in float64 überlaufen -> mathematisch stabil rechnen.
    with np.errstate(over="ignore", invalid="ignore"):
        sq = np.square(d)
    if not np.isfinite(sq).all():
        return float("inf")
    return float(np.sqrt(np.mean(sq)))


@dataclass
class BacktestMetrics:
    next_start_error_min: float
    start_errors_min: list[float]
    energy_6h_mae_wh: float
    energy_6h_bias_wh: float
    cycles_per_day_sim: float
    cycles_per_day_meas: float
    buffer_rmse_c: float
    p_el_hourly_rmse_w: float

    def summary(self) -> str:
        se = ", ".join("nan" if np.isnan(x) else f"{x:+.0f}" for x in self.start_errors_min)
        return (
            f"nächster Start: {self.next_start_error_min:+.0f} min | "
            f"Starts 1-5 [min]: {se} | "
            f"Energie 6h MAE {self.energy_6h_mae_wh:.0f} Wh (Bias {self.energy_6h_bias_wh:+.0f}) | "
            f"Zyklen/Tag {self.cycles_per_day_sim:.1f} vs {self.cycles_per_day_meas:.1f} | "
            f"Puffer-RMSE {self.buffer_rmse_c:.1f} K"
        )


def compute_metrics(
    time_s: np.ndarray,
    sim_compressor_on: np.ndarray,
    meas_compressor_on: np.ndarray,
    sim_p_el_w: np.ndarray,
    meas_p_el_w: np.ndarray,
    sim_buffer_c: np.ndarray,
    meas_buffer_c: np.ndarray,
) -> BacktestMetrics:
    sim_starts = start_times_s(time_s, sim_compressor_on)
    meas_starts = start_times_s(time_s, meas_compressor_on)
    e_sim = energy_per_window_wh(time_s, sim_p_el_w)
    e_meas = energy_per_window_wh(time_s, meas_p_el_w)
    k = min(len(e_sim), len(e_meas))
    e_err = e_sim[:k] - e_meas[:k]
    return BacktestMetrics(
        next_start_error_min=next_start_error_s(sim_starts, meas_starts) / 60.0,
        start_errors_min=[x / 60.0 for x in start_errors_s(sim_starts, meas_starts)],
        energy_6h_mae_wh=float(np.mean(np.abs(e_err))) if k else float("nan"),
        energy_6h_bias_wh=float(np.mean(e_err)) if k else float("nan"),
        cycles_per_day_sim=cycles_per_day(time_s, sim_compressor_on),
        cycles_per_day_meas=cycles_per_day(time_s, meas_compressor_on),
        buffer_rmse_c=rmse(sim_buffer_c, meas_buffer_c),
        p_el_hourly_rmse_w=rmse(sim_p_el_w, meas_p_el_w),
    )
