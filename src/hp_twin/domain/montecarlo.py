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
    p_el_det_w: np.ndarray | None = None  # deterministischer Lauf als separate Spur

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


def phase_locked_mean(
    time_s: np.ndarray,
    p_el_runs: list[np.ndarray],
    det_window: np.ndarray,
) -> np.ndarray:
    """Erwartungswert mit deterministischer Phase: Ensemble-Energie, det-Fenster.

    Monte-Carlo-Streuung jittert die Laufstarts, ein slotweises Mittel verschmiert deshalb
    jeden Lauf. Hingegen ist die ENERGIE je Lauf eine solide Ensemble-Aussage. Diese Funktion
    hält die Phase (Start/Stop der Läufe) aus dem deterministischen Lauf fest und verteilt
    die Ensemble-Energie in diese Fenster:

    - Je det-Fenster zählt jeder Ensemble-Lauf mit seiner GESAMT-Energie im erweiterten
      Fenster (Erweiterung bis zur Mitte der Lücke zum Nachbarfenster). Ein Lauf, der vor
      oder nach dem det-Fenster auskeilt, gibt seine Überhang-Energie damit ans Fenster.
    - Gemittelt über ALLE Läufe: Läufe, die den Zyklus ausfallen lassen, tragen 0 bei und
      senken die erwartete Energie proportional zur Ausfallwahrscheinlichkeit.
    - Der Mittelwert-Shape im det-Fenster wird auf diese Ziel-Energie skaliert
      (Phasen-Form bleibt, Energie = Ensemble-Erwartung je Zyklus).

    ``det_window``: 1 = deterministisch läuft hier, 0 = aus (gleiche Länge wie die Läufe).
    """
    dt = float(time_s[1] - time_s[0]) if len(time_s) > 1 else 60.0
    n = len(det_window)
    if n == 0 or not p_el_runs:
        return np.zeros(n)
    det = np.asarray(det_window, dtype=int)
    p_matrix = np.stack([np.asarray(p[:n], dtype=float) for p in p_el_runs])
    mean_profile = p_matrix.mean(axis=0)
    out = np.zeros(n)
    # zusammenhängende det-Laufblöcke als Fenster
    d = np.diff(det)
    starts = np.where(d == 1)[0] + 1
    stops = np.where(d == -1)[0] + 1
    if det[0] == 1:
        starts = np.concatenate([[0], starts])
    if det[-1] == 1:
        stops = np.concatenate([stops, [n]])
    bounds_L = starts.tolist()
    bounds_R = stops.tolist()
    for i, (s, e) in enumerate(zip(bounds_L, bounds_R, strict=True)):
        # Erweiterung bis zur Mitte der Lücke zum jeweiligen Nachbarfenster
        lo = (bounds_R[i - 1] + s) // 2 if i > 0 else 0
        hi = (e + bounds_L[i + 1]) // 2 if i + 1 < len(bounds_L) else n

        # mittlere Energie je Lauf im erweiterten Fenster (Ensemble-Aussage)
        E_ens = float(p_matrix[:, lo:hi].sum(axis=1).mean()) * dt / 3600.0
        shape_sum = float(mean_profile[s:e].sum()) * dt / 3600.0
        if shape_sum <= 0:
            out[s:e] = 0.0
            continue
        scale = E_ens / shape_sum if E_ens > 0 else 0.0
        out[s:e] = mean_profile[s:e] * scale
    return out


def aggregate_slots(
    time_s: np.ndarray,
    p_el_runs: list[np.ndarray],
    slot_s: float = 900.0,
    running_threshold_w: float = 200.0,
    deterministic: np.ndarray | None = None,
    det_blend_hours: float | None = None,
    band_runs: list[np.ndarray] | None = None,
) -> SlotForecast:
    """Ensemble (Liste P_el-Läufe, gleiches Raster) -> Slot-Statistik.

    Erwartungswert/Laufwahrscheinlichkeit kommen aus ``p_el_runs``; ``band_runs`` (optional)
    liefert die Läufe für P10/P50/P90 — bei phase-locked Aggregation (ein geformter Lauf im
    ``p_el_runs``) stammt das Streuband aus dem ORIGINAL-Ensemble. ``deterministic`` +
    ``det_blend_hours``: Das Nahzeit-Band wächst weich mit dem Horizont (0 = volles Ensemble).
    """
    if not p_el_runs:
        raise ValueError("p_el_runs ist leer")
    mean = np.stack([_to_slots(time_s, p, slot_s) for p in p_el_runs]).mean(axis=0)
    src_matrix = np.stack(
        [_to_slots(time_s, p, slot_s) for p in (band_runs if band_runs is not None else p_el_runs)]
    )
    run_matrix = np.stack(
        [
            _to_slots(time_s, (p > running_threshold_w).astype(float), slot_s)
            for p in (band_runs if band_runs is not None else p_el_runs)
        ]
    )
    n_slots = src_matrix.shape[1]
    det_slots: np.ndarray | None = None
    if deterministic is not None:
        det_slots = _to_slots(time_s, np.asarray(deterministic, dtype=float), slot_s)
        if det_blend_hours:
            n_det = min(len(det_slots), n_slots)
            # Slot-Stunden (Mitte je Slot) für den weichen Übergang
            h = (np.arange(n_det) * slot_s + slot_s / 2.0) / 3600.0
            w = np.clip(h / det_blend_hours, 0.0, 1.0)
            mean[:n_det] = w * mean[:n_det] + (1.0 - w) * det_slots[:n_det]
            # Nahzeit-Band eingrenzen: Das Band wächst mit dem Horizont — nah ist der
            # deterministische Lauf bester Einzelschätzer (Startzustand gemessen; die
            # Ensemble-Streuung akkumuliert erst mit der Simulationsdauer).
            p10 = np.percentile(src_matrix, 10, axis=0).copy()
            p90 = np.percentile(src_matrix, 90, axis=0).copy()
            p10[:n_det] = mean[:n_det] + (p10[:n_det] - mean[:n_det]) * w
            p90[:n_det] = mean[:n_det] + (p90[:n_det] - mean[:n_det]) * w
    return SlotForecast(
        slot_start_s=np.arange(n_slots) * slot_s,
        slot_s=slot_s,
        p_el_mean_w=mean,
        p_el_p10_w=(
            p10
            if det_slots is not None and det_blend_hours
            else np.percentile(src_matrix, 10, axis=0)
        ),
        p_el_p50_w=np.percentile(src_matrix, 50, axis=0),
        p_el_p90_w=(
            p90
            if det_slots is not None and det_blend_hours
            else np.percentile(src_matrix, 90, axis=0)
        ),
        energy_wh=mean * slot_s / 3600.0,
        # Fraction of each slot genuinely above the operating threshold, not
        # the fraction of ensemble runs whose majority is active.
        run_probability=run_matrix.mean(axis=0),
        p_el_det_w=det_slots,
    )


def run_ensemble(
    run_once: Callable[[np.random.Generator], np.ndarray], n_runs: int, seed: int = 0
) -> list[np.ndarray]:
    """Führt run_once n_runs-mal mit eigenem RNG aus und sammelt die P_el-Läufe."""
    rng = np.random.default_rng(seed)
    return [run_once(np.random.default_rng(rng.integers(0, 2**32 - 1))) for _ in range(n_runs)]
