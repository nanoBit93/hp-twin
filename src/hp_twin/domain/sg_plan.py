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
from typing import Any

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
    *,
    ramp_s: float | None = None,
    min_setpoint_c: float | None = None,
    dt_s: float = 60.0,
) -> np.ndarray:
    """SG3-Sollverlauf für [start_idx, end_idx) zurücksetzen (Return = neue HK1-Spur).

    Der gemessene Regler schaltet den Sollwert direkt auf das SG3-Niveau. Ein
    optionaler, zeitbasierter Anstieg bleibt für einen simulierten Lauf unabhängig
    von dessen späterer EOS-Slotdauer. `min_setpoint_c` ist die explizite
    Zielverriegelung im Regler und verhindert unnötige Starts im selben SG3-Fenster.
    """
    n = len(setpoint_hk1_c)
    sp = np.asarray(setpoint_hk1_c, dtype=float).copy()
    start_idx = max(0, int(start_idx))
    end_idx = int(np.clip(end_idx, start_idx + 1, n))
    span = end_idx - start_idx
    base_level = float(sp[start_idx - 1]) if start_idx > 0 else float(sp[start_idx])
    top = max(float(sg3_top_c), base_level + min_ramp_k)
    if ramp_s is not None:
        ramp_steps = max(int(np.ceil(ramp_s / dt_s)), 1)
    else:
        ramp_steps = max(int(np.ceil(span / 3.0)), 1)
    ramp = np.clip((np.arange(span) + 1) / ramp_steps, 0.0, 1.0)
    sp[start_idx:end_idx] = base_level + (top - base_level) * ramp
    if min_setpoint_c is not None:
        sp[start_idx:end_idx] = np.maximum(sp[start_idx:end_idx], min_setpoint_c)
    return sp


# --- Kandidaten-Fenster (thermische Feasibility) -------------------------------------------


def sg3_candidates(
    t_buffer_c: np.ndarray,
    buffer_overshoot_k: float,
    running_w: np.ndarray,
    running_threshold_w: float,
    min_off_s: float,
    dt_s: float,
    allowed_start_s: float,
    allowed_end_s: float,
    sg3_top_c: float,
) -> list[tuple[float, float]]:
    """Thermisch mögliche SG3-Startbereiche im Basis-Lauf ([start_s, end_s]).

    Die Enden beschreiben nur den freien Startbereich. Die tatsächliche Dauer und
    der Zielabschluss werden anschließend von der vollständigen Simulation bestimmt.
    """
    n = min(len(t_buffer_c), len(running_w))
    t_buf = np.asarray(t_buffer_c, dtype=float)[:n]
    running = np.asarray(running_w, dtype=float)[:n] > running_threshold_w
    # min_off-Lock nach jedem Basis-Stopp
    stop_idx = np.where(np.diff(running.astype(int)) == -1)[0]
    locks = np.zeros(n, dtype=bool)
    for i in stop_idx:
        lo = i + 1
        hi = min(n, i + 1 + int(round(min_off_s / dt_s)))
        locks[lo:hi] = True

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
    # The free region identifies a possible start, not the compressor duration.
    # Its end is irrelevant for the EOS offer: the thermal simulation runs from
    # the first allowed start to target completion or the configured limit.
    return [(lo, hi) for lo, hi in cands if hi > lo]


# --- Contingent-Bildung (wirtschaftliche Bewertung) ------------------------------------------


@dataclass(frozen=True)
class SgCandidate:
    """Ein bewertetes SG3-Fenster mit Contingent-Daten (EOS-push-fähig)."""

    start_s: float
    end_s: float
    energy_wh: float
    max_power_w: float
    reason: str
    feasible: tuple[bool, ...] | None = None
    value_eur_per_kwh: float | None = None
    saving_eur: float | None = None
    avoided_eur: float | None = None
    baseline_cost_eur: float | None = None
    sg3_cost_eur: float | None = None
    thermal: dict[str, Any] | None = None

    @property
    def runtime_s(self) -> float:
        """Offered compressor operation, distinct from the setpoint deadline."""
        if self.thermal is not None and self.thermal.get("offer_end_s") is not None:
            return float(self.thermal["offer_end_s"]) - self.start_s
        return self.end_s - self.start_s

    @property
    def setpoint_end_s(self) -> float:
        """End of elevated setpoint, including EOS's final partial-slot rounding."""
        if self.thermal is not None and self.thermal.get("setpoint_end_s") is not None:
            return float(self.thermal["setpoint_end_s"])
        return self.end_s

    def deadline_hours(self) -> float:
        """Absolute candidate end as hours from the forecast frame start.

        EOS measures this duration from its current planning slot. Since the
        twin's frame_start is that same slot, end_s is already the correct
        absolute endpoint; adding start_s again would incorrectly shorten or
        lengthen the permissible placement period.
        """
        return float(max(1 / 60, self.end_s / 3600.0))


def evaluate_sg3_economics(
    basis_power_w: np.ndarray,
    sg3_power_w: np.ndarray,
    price_eur_per_kwh: np.ndarray,
    *,
    dt_s: float,
    slot_s: float,
    outdoor_c: np.ndarray,
    offered_wh: float | None = None,
) -> dict[str, Any]:
    """Value the full scenario while EOS prices the complete offered window.

    SG3 is a temporal shift, not an energy-saving measure. ``offered_wh`` is the
    additional electrical energy inside the SG3 setpoint window, before displaced
    baseline runs elsewhere in the horizon. Only displaced baseline energy carries
    avoided cost; unchanged operation and standby do not inflate the offer value.
    A zero or negative horizon-wide net-energy difference must not suppress it.
    """
    baseline = np.asarray(basis_power_w, dtype=float)
    sg3 = np.asarray(sg3_power_w, dtype=float)
    prices = np.asarray(price_eur_per_kwh, dtype=float)
    if not (len(baseline) == len(sg3) == len(outdoor_c)):
        raise ValueError(
            "SG3 scenarios and outdoor forecast must share one time axis: "
            f"baseline={len(baseline)}, sg3={len(sg3)}, outdoor={len(outdoor_c)}"
        )
    if len(baseline) < 2 or slot_s < dt_s:
        raise ValueError("SG3 economics requires a forecast and compatible slot lengths")
    steps_per_slot = int(round(slot_s / dt_s))
    if abs(steps_per_slot * dt_s - slot_s) > 1e-6:
        raise ValueError("SG3 simulation steps do not fit the EOS slot length")
    prices = np.repeat(prices, steps_per_slot)[: len(baseline)]
    if len(prices) != len(baseline):
        raise ValueError(
            f"SG3 price slots do not cover the forecast: prices={len(prices)}, "
            f"required={len(baseline)}"
        )

    # Do not invent missing tariffs. Evaluate only complete, priced EOS slots;
    # a later publication gap must not erase a fully priced morning window.
    slot_count = int(np.ceil(len(baseline) * dt_s / slot_s))
    priced_steps = np.zeros(len(baseline), dtype=bool)
    for slot in range(slot_count):
        i0 = int(slot * slot_s / dt_s)
        i1 = min(len(baseline), int((slot + 1) * slot_s / dt_s))
        if i1 > i0 and np.isfinite(prices[i0:i1]).all():
            priced_steps[i0:i1] = True
    covered_baseline_wh = float(np.sum(baseline[priced_steps] * dt_s / 3600.0))
    covered_sg3_wh = float(np.sum(sg3[priced_steps] * dt_s / 3600.0))
    total_baseline_wh = float(np.sum(baseline * dt_s / 3600.0))
    if covered_baseline_wh <= 0.0 or covered_sg3_wh <= 0.0:
        raise ValueError("SG3 economics has no fully priced energy in the forecast")
    baseline = baseline[priced_steps]
    sg3 = sg3[priced_steps]
    prices = prices[priced_steps]
    baseline_cost = float(np.sum(baseline * prices * dt_s / 3600.0) / 1000.0)
    sg3_cost = float(np.sum(sg3 * prices * dt_s / 3600.0) / 1000.0)
    extra_kwh = (sg3 - baseline) * dt_s / 3600.0 / 1000.0
    displaced_w = np.maximum(baseline - sg3, 0.0)
    avoided = float(np.sum(displaced_w * prices * dt_s / 3600.0) / 1000.0)
    total_extra_wh = float(np.sum(extra_kwh) * 1000.0)
    denominator_wh = offered_wh if offered_wh is not None else total_extra_wh
    if denominator_wh <= 0.0:
        raise ValueError("SG3 offer has no electrical energy in its operating window")
    if avoided <= 0.0:
        raise ValueError("SG3 scenario has no positive avoided-cost value")

    return {
        "baseline_cost_eur": baseline_cost,
        "sg3_cost_eur": sg3_cost,
        "saving_eur": baseline_cost - sg3_cost,
        "avoided_eur": avoided,
        "value_eur_per_kwh": avoided / (denominator_wh / 1000.0),
        "total_extra_wh": total_extra_wh,
        "offered_wh": denominator_wh,
        "priced_baseline_fraction": covered_baseline_wh / max(total_baseline_wh, 1e-9),
        "priced_slots": int(np.sum(priced_steps) / steps_per_slot),
        "total_slots": slot_count,
    }


def sg3_start_feasible_slots(
    *,
    t_buffer_c: np.ndarray,
    running_w: np.ndarray,
    outdoor_c: np.ndarray,
    initial_controller_state: Any,
    heat_pump: Any,
    controller_config: Any,
    sg3_top_c: float,
    flow_offset_k: float,
    buffer_overshoot_k: float,
    running_threshold_w: float,
    dt_s: float,
    slot_s: float,
    allowed_start_s: float,
    allowed_end_s: float,
    max_duration_s: float,
    min_cop: float = 1.5,
    bivalence_outdoor_c: float = 0.0,
) -> tuple[list[bool], list[str]]:
    """Return EOS-aligned *start* feasibility and a reason for every slot.

    EOS expands a permitted start by its configured minimum runtime. Here the full
    allowed run must fit min_on, max_duration and the remaining horizon; min_off at
    the current measurement and after a predicted baseline stop is also respected.
    """
    n = min(len(t_buffer_c), len(running_w), len(outdoor_c))
    if n == 0 or slot_s <= 0 or dt_s <= 0:
        return [], []
    temps = np.asarray(t_buffer_c[:n], dtype=float)
    running = np.asarray(running_w[:n], dtype=float) > running_threshold_w
    outside = np.asarray(outdoor_c[:n], dtype=float)
    state = initial_controller_state
    locked_now = bool(state.mode.value == "pause")
    remaining_off_s = (
        max(
            0.0,
            float(controller_config.min_off_s) - float(state.time_since_stop_s),
        )
        if locked_now
        else 0.0
    )
    if state.mode.value in ("heat", "dhw") and not np.any(running):
        # No predicted stop: the current baseline run covers the whole horizon.
        remaining_off_s = float("inf")

    n_slots = int(np.ceil(n * dt_s / slot_s))
    feasible: list[bool] = []
    reasons: list[str] = []
    for slot in range(n_slots):
        start_s = slot * slot_s
        start_i = int(round(start_s / dt_s))
        reason = "ok"
        if start_s < allowed_start_s or start_s >= allowed_end_s:
            reason = "outside_start_window"
        elif start_s < remaining_off_s:
            reason = "min_off"
        elif start_i >= n or start_i + int(np.ceil(controller_config.min_on_s / dt_s)) > n:
            reason = "min_on_beyond_horizon"
        elif running[start_i]:
            reason = "baseline_running"
        else:
            stop_indices = np.where(np.diff(running.astype(int)) == -1)[0] + 1
            recent_stops = stop_indices[
                (stop_indices <= start_i)
                & (start_s - stop_indices * dt_s < controller_config.min_off_s)
            ]
            if len(recent_stops):
                reason = "min_off"
            elif temps[start_i] >= sg3_top_c - buffer_overshoot_k:
                reason = "buffer_at_sg3_target"
            else:
                duration_s = min(max_duration_s, n * dt_s - start_s)
                end_i = min(n, start_i + int(np.ceil(duration_s / dt_s)))
                min_on_i = min(n, start_i + int(np.ceil(controller_config.min_on_s / dt_s)))
                if np.any(running[start_i:min_on_i]):
                    reason = "baseline_running_during_min_on"
                else:
                    cop = np.asarray(
                        [
                            heat_pump.cop(float(t_out), float(sg3_top_c + flow_offset_k))
                            for t_out in outside[start_i:end_i]
                        ]
                    )
                    if (
                        not len(cop)
                        or not np.isfinite(cop).all()
                        or float(np.min(cop)) < min_cop
                        or float(np.min(outside[start_i:end_i])) < bivalence_outdoor_c
                    ):
                        reason = "cop_bivalence"
        feasible.append(reason == "ok")
        reasons.append(reason)
    return feasible, reasons
