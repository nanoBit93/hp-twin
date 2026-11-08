"""SG-Ready-Planung (rein): Sollwertspur normalisieren (Basis) und zurücksetzen (Injection).

Die Funktionen sind IO-frei und deterministisch. Die SG3-Semantik steht in den Sollwert-
Spuren selbst: bei SG3 hebt der WPM den HK1-Sollwert an (real 32 → 44-50 °C), die climate-
Thermostaten ihre Raum-Sollwerte (+2 K) und der DHW-Soll geht auf 45 °C (gedeckelt per
dhw_setpoint_cap).

- `strip_sg3_setpoints` ersetzt die SG3-Blöcke in allen Spuren durch das SG2-Grundprogramm
  (Interpolation der Randwerte) -> Basis-Lauf ohne SG3.
- `inject_sg3_start` setzt an einem geplanten Fenster den SG3-Verlauf zurück (Anstieg,
  Plateau) -> Endprognose mit SG3; die exakte Taktung (min_off, Ladeziel) behält der
  Regler selbst, weil Sollwerte schon Zeitreihen-Eingänge sind.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

DEFAULT_SG3_MIN_SETPOINT_C = 38.0  # SG2-Plateau ~32 °C, SG3 real 44-50 °C


def sg3_mask(
    sg_status: np.ndarray | None,
    setpoint_hk1_c: np.ndarray,
    sg3_min_setpoint_c: float = DEFAULT_SG3_MIN_SETPOINT_C,
    dilate: int = 3,
) -> np.ndarray:
    """True, wo SG3 aktiv ist: Status-Sensor >= 3 ODER HK1-Sollwert >= Schwelle.

    Beide Quellen kombinieren (ODER), weil beide allein zu kurz sind: der Status-Sensor
    schaltet am Block-Ende zu spät zurück (der WPM hält den SG3-Soll bis zur Umschaltung),
    der Sollwert-Trigger verpasst ggf. ein manuelles SG2-Plateau über der Schwelle.
    Nach der ODER-Verknüpfung wird die Maske um `dilate` Punkte dilatiert, damit
    auslaufende Sensor-/Sollwert-Übergänge beim Strip nicht als Randwerte wirken.
    """
    sp = np.asarray(setpoint_hk1_c, dtype=float)
    m_soll = np.isfinite(sp) & (sp >= sg3_min_setpoint_c)
    m = m_soll.copy()
    if sg_status is not None and len(sg_status) == len(sp):
        stat = np.asarray(sg_status, dtype=float)
        if np.isfinite(stat).any() and float(np.nanmax(stat)) >= 3.0:
            m = m_soll | (np.nan_to_num(stat, nan=0.0) >= 3.0)
    if dilate > 0 and m.any():
        kernel = np.ones(2 * int(dilate) + 1, dtype=bool)
        m = np.convolve(m, kernel, mode="same")
    return m


def blocks(mask: np.ndarray) -> list[tuple[int, int]]:
    """Zusammenhängende True-Blöcke als (lo, hi) inklusive; leere Liste wenn keiner."""
    m = np.asarray(mask, dtype=bool)
    if m.size == 0 or not m.any():
        return []
    pad = np.concatenate(([False], m, [False])).astype(int)
    edges = np.diff(pad)
    starts = np.where(edges == 1)[0]
    ends = np.where(edges == -1)[0]
    return [(int(s), int(e - 1)) for s, e in zip(starts, ends, strict=False)]


def strip_sg3_setpoints(
    series: dict[str, np.ndarray],
    mask: np.ndarray,
    min_floor_c: float = 5.0,
) -> dict[str, np.ndarray]:
    """Ersetzt SG3-Blöcke in jeder Sollwert-Spur durch das SG2-Grundprogramm.

    `series` benennt die Spuren (setpoint_hk1_c, setpoint_dhw_c, setpoint_room_*_c).
    Grundprogramm = linear zwischen den Randwerten des Blocks (Zeitachse = Index-Abstand);
    fehlt ein Rand (Block am Zeilenanfang/-ende), bleibt der nächste vorhandene Wert
    konstant. Räume unter `min_floor_c` (nicht beheizt) bleiben ausgeschaltet.
    """
    m = np.asarray(mask, dtype=bool)
    out: dict[str, np.ndarray] = {}
    for name, vals in series.items():
        arr = np.asarray(vals, dtype=float).copy()
        if len(arr) != len(m):
            out[name] = arr
            continue
        s = pd.Series(arr, dtype=float)
        for lo, hi in blocks(m):
            s.iloc[lo : hi + 1] = np.nan
        filled = s.interpolate(limit_direction="both").bfill().ffill()
        clean = filled.to_numpy(dtype=float)
        fallback = float(np.nanmean(arr)) if np.isfinite(arr).any() else 0.0
        safe = np.where(np.isfinite(clean), clean, fallback)
        powered = np.isfinite(arr) & (arr > min_floor_c)
        out[name] = np.where(powered, np.maximum(safe, min_floor_c), arr)
    return out


def inject_sg3_start(
    setpoint_hk1_c: np.ndarray,
    start_idx: int,
    end_idx: int,
    sg3_top_c: float,
    min_ramp_k: float = 5.0,
) -> np.ndarray:
    """SG3-Sollverlauf für [start_idx, end_idx) zurücksetzen (Return = neue HK1-Spur).

    Anstieg auf `sg3_top_c` über die ersten ~1/3 des Fensters (die WPM rampet), danach
    Plateau bis Fenstende; danach gilt die Basis-Spur weiter.
    """
    n = len(setpoint_hk1_c)
    sp = np.asarray(setpoint_hk1_c, dtype=float).copy()
    start_idx = max(0, int(start_idx))
    end_idx = int(np.clip(end_idx, start_idx + 1, n))
    span = end_idx - start_idx
    base_level = float(sp[start_idx - 1]) if start_idx > 0 else float(sp[start_idx])
    top = max(float(sg3_top_c), base_level + min_ramp_k)
    ramp = np.clip((np.arange(span) + 1) / max(span / 3.0, 1.0), 0.0, 1.0)
    sp[start_idx:end_idx] = base_level + (top - base_level) * ramp
    return sp


# --- Kandidaten-Fenster (thermische Feasibility) -------------------------------------------


def sg3_candidates(
    t_buffer_c: np.ndarray,
    buffer_setpoint_c: np.ndarray,
    buffer_overshoot_k: float,
    running_w: np.ndarray,
    running_threshold_w: float,
    min_off_s: float,
    dt_s: float,
    allowed_start_s: float,
    allowed_end_s: float,
    sg3_top_c: float,
    max_duration_s: float = 3 * 3600.0,
) -> list[tuple[float, float]]:
    """Thermisch mögliche SG3-Startfenster im Basis-Lauf ([start_s, end_s], Simulationszeit).

    Ein Start ist möglich, wenn der Puffer in der Basis-Prognose durch Ladung/Entladung
    den Startbereich erreicht (t_buffer < sg3_top_c - Hysterese-Ersatz) bzw. geladen
    würde — vereinfacht: Startfenster, in denen die WP im Basis-Lauf bereits läuft oder
    der Puffer weit unter SG3-Top steht (dann lohnt SG3), und nicht im min_off-Lock.
    """
    n = min(len(t_buffer_c), len(buffer_setpoint_c), len(running_w))
    t_buf = np.asarray(t_buffer_c, dtype=float)[:n]
    running = np.asarray(running_w, dtype=float)[:n] > running_threshold_w
    # min_off-Lock nach jedem Basis-Stopp
    stop_idx = np.where(np.diff(running.astype(int)) == -1)[0]
    lock_until = min_off_s  # absolute Zeitstempel
    locks = np.zeros(n, dtype=bool)
    for i in stop_idx:
        lo = i + 1
        hi = min(n, i + 1 + int(round(min_off_s / dt_s)))
        locks[lo:hi] = True
    del lock_until

    cands: list[tuple[float, float]] = []
    in_window = False
    start_i = 0
    for i in range(n):
        t_s = i * dt_s
        inside = allowed_start_s <= t_s < allowed_end_s
        # Ein SG3-Start braucht die freie WP (keine laufende Basis-Ladung, kein min_off-Lock);
        # Puffer unter SG3-Top ist die zweite Bedingung (sonst ist nichts zu holen)
        free = not running[i] and not locks[i]
        lohnt = t_buf[i] < sg3_top_c - buffer_overshoot_k
        if inside and free and lohnt:
            if not in_window:
                in_window = True
                start_i = i
        else:
            if in_window:
                cands.append((start_i * dt_s, i * dt_s))
            in_window = False
    if in_window:
        cands.append((start_i * dt_s, n * dt_s))
    # Fenster kappen auf max_duration
    capped: list[tuple[float, float]] = []
    for lo, hi in cands:
        while lo < hi:
            hi_c = min(hi, lo + max_duration_s)
            capped.append((lo, hi_c))
            lo = hi_c
            if hi_c >= hi:
                break
    return capped


# --- Contingent-Bildung (wirtschaftliche Bewertung) ------------------------------------------


@dataclass(frozen=True)
class SgCandidate:
    """Ein bewertetes SG3-Fenster mit Contingent-Daten (EOS-push-fähig)."""

    start_s: float
    end_s: float
    energy_wh: float
    max_power_w: float
    reason: str

    def deadline_hours(self) -> float:
        """Letzter Slot des Fensters relativ zum Frame-Start, in Stunden aufgerundet."""
        return float(np.ceil(self.end_s / 3600.0))


def evaluate_candidate(
    start_s: float,
    end_s: float,
    energy_wh: float,
    max_power_w: float,
    slot_s: float,
    solar_surplus_w: np.ndarray | None,
    min_pv_surplus_w: float,
    price_eur_per_kwh: np.ndarray | None = None,
    max_price_eur: float | None = None,
) -> SgCandidate | None:
    """Bewertet ein thermisches Fenster gegen die Markt-Signale; None = nicht sinnvoll.

    Sinnvoll, wenn im Fenster (a) der PV-Überschuss mindestens einmal >= Schwelle ist
    ODER (b) ein Preis <= Preisdach vorliegt. Fehlt das Markt-Signal (None/NaN), zählt
    das Fenster als unentschieden und wird nicht empfohlen (kein blindes SG3).
    """

    def _slots(a: np.ndarray) -> np.ndarray:
        i0 = int(round(start_s / slot_s))
        i1 = max(i0 + 1, int(np.ceil(end_s / slot_s)))
        return np.asarray(a, dtype=float)[i0:i1]

    reasons: list[str] = []
    if solar_surplus_w is not None and len(solar_surplus_w):
        peak = float(np.nanmax(_slots(solar_surplus_w)))
        if peak >= min_pv_surplus_w:
            reasons.append(f"pv_surplus>={min_pv_surplus_w:.0f}W (peak {peak:.0f}W)")
    if price_eur_per_kwh is not None and max_price_eur is not None and max_price_eur > 0:
        prices = _slots(price_eur_per_kwh)
        prices = prices[np.isfinite(prices)]
        if len(prices) and float(np.min(prices)) <= max_price_eur:
            reasons.append(f"price<={max_price_eur:.2f}€ (min {float(np.min(prices)):.2f}€)")
    if not reasons:
        return None
    return SgCandidate(start_s, end_s, energy_wh, max_power_w, "; ".join(reasons))
