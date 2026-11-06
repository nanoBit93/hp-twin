"""Monte-Carlo-Aggregation und Ereignis-Extraktion für die Lastprognose.

- extract_events: diskrete Start/Ende/Modus/mittlere-Leistung-Ereignisse aus einem Lauf.
- aggregate_slots: Ensemble vieler Läufe -> 15-min-Slots mit Erwartungswert, P10/P50/P90,
  Energie und Laufwahrscheinlichkeit je Slot.
Die eigentliche Streuung (Parameter/Wetter/Zapfungen) erzeugt der Aufrufer, der run_once liefert.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ForecastEvent:
    start_s: float
    end_s: float
    mode: str
    mean_p_el_w: float

    @property
    def duration_min(self) -> float:
        return (self.end_s - self.start_s) / 60.0


def extract_events(
    time_s: np.ndarray,
    mode: list[str],
    p_el_w: np.ndarray,
    running_modes: tuple[str, ...] = ("heat", "dhw"),
) -> list[ForecastEvent]:
    """Zusammenhängende Laufphasen als Ereignisse (Start, Ende, Modus, mittlere P_el)."""
    t = np.asarray(time_s, float)
    p = np.asarray(p_el_w, float)
    on = np.array([m in running_modes for m in mode])
    dt = float(t[1] - t[0]) if len(t) > 1 else 60.0
    events: list[ForecastEvent] = []
    start = None
    for i in range(len(on)):
        if on[i] and start is None:
            start = i
        if (not on[i]) and start is not None:
            seg = slice(start, i)
            dom = max(set(mode[start:i]), key=mode[start:i].count)
            events.append(ForecastEvent(t[start], t[i], dom, float(np.mean(p[seg]))))
            start = None
    if start is not None:
        seg = slice(start, len(on))
        dom = max(set(mode[start:]), key=mode[start:].count)
        events.append(ForecastEvent(t[start], t[-1] + dt, dom, float(np.mean(p[seg]))))
    return events


@dataclass
class SlotForecast:
    slot_start_s: np.ndarray  # linke Slotgrenze (s ab Prognosebeginn)
    slot_s: float
    p_el_mean_w: np.ndarray
    p_el_p10_w: np.ndarray
    p_el_p50_w: np.ndarray
    p_el_p90_w: np.ndarray
    energy_wh: np.ndarray  # Erwartungswert je Slot
    run_probability: np.ndarray  # Anteil Läufe mit WP-Betrieb im Slot

    @property
    def n_slots(self) -> int:
        return len(self.slot_start_s)


def _to_slots(time_s: np.ndarray, series: np.ndarray, slot_s: float) -> np.ndarray:
    """Mittelt eine Zeitreihe (festes Raster) auf Slots (Slot-Mittelwert)."""
    t = np.asarray(time_s, float)
    dt = float(t[1] - t[0]) if len(t) > 1 else 60.0
    per = max(int(round(slot_s / dt)), 1)
    n_slots = len(series) // per
    return np.array([series[i * per : (i + 1) * per].mean() for i in range(n_slots)])


def aggregate_slots(
    time_s: np.ndarray,
    p_el_runs: list[np.ndarray],
    slot_s: float = 900.0,
    running_threshold_w: float = 200.0,
) -> SlotForecast:
    """Ensemble (Liste P_el-Läufe, gleiches Raster) -> Slot-Statistik."""
    if not p_el_runs:
        raise ValueError("p_el_runs ist leer")
    slot_matrix = np.stack([_to_slots(time_s, p, slot_s) for p in p_el_runs])  # (n_runs, n_slots)
    run_matrix = np.stack(
        [_to_slots(time_s, (p > running_threshold_w).astype(float), slot_s) for p in p_el_runs]
    )
    n_slots = slot_matrix.shape[1]
    mean = slot_matrix.mean(axis=0)
    return SlotForecast(
        slot_start_s=np.arange(n_slots) * slot_s,
        slot_s=slot_s,
        p_el_mean_w=mean,
        p_el_p10_w=np.percentile(slot_matrix, 10, axis=0),
        p_el_p50_w=np.percentile(slot_matrix, 50, axis=0),
        p_el_p90_w=np.percentile(slot_matrix, 90, axis=0),
        energy_wh=mean * slot_s / 3600.0,
        run_probability=(run_matrix > 0.5).mean(axis=0),
    )


def run_ensemble(
    run_once: Callable[[np.random.Generator], np.ndarray], n_runs: int, seed: int = 0
) -> list[np.ndarray]:
    """Führt run_once n_runs-mal mit eigenem RNG aus und sammelt die P_el-Läufe."""
    rng = np.random.default_rng(seed)
    return [run_once(np.random.default_rng(rng.integers(0, 2**32 - 1))) for _ in range(n_runs)]
