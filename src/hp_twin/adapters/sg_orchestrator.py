"""SG3-Orchestrator: Basis-Lauf (SG2) -> Kandidaten -> EOS-Contingent -> Placement -> Endprognose.

Gerichteter Zyklus pro 15-min-Periode (keine Rückkopplung innerhalb eines Zyklus):

1. BASIS-Lauf:   Forecast-Frame mit strip_sg3=True, DWD-Wetter -> deterministische Basis
                 (WP-Frei-Zeiten, Pufferverlauf) ohne SG3.
2. KANDIDATEN:   thermisch mögliche Fenster (sg3_candidates) + Wirtschaftlichkeit
                 (evaluate_candidate gegen PV-Schwelle/Preisdach).
3. CONTINGENT:   Differenzenergie des besten Kandidaten als EosContingent an <sg3_load>
                 pushen (dry_run=True oder ohne base_url: skip).
4. PLACEMENT:    gelegte Slots aus dem Load-Status lesen (plan[]); leer -> Folgetakt.
5. ENDPROGNOSE:  SG3-Sollverlauf am Placement injecten -> Forecast -> Profil-Push wird
                 vom Aufrufer erledigt (service.run_cycle).

Der Orchestrator kennt HA/EOS nur über injizierte Callables/Client-Objekte — im Test
werden Fakes übergeben, produktiv die echten Funktionen aus forecast/eos_connect.
Degraded: ohne EOS oder ohne Kandidat läuft alles weiter, nur eben ohne SG3.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from hp_twin.adapters.eos_connect import (
    EosContingent,
    planned_slots_from_status,
    push_contingent,
    read_load_status,
)
from hp_twin.adapters.evcc import MarketSeries
from hp_twin.domain.calibration import CalibrationReport
from hp_twin.domain.model_config import ModelConfig
from hp_twin.domain.sg_plan import (
    SgCandidate,
    evaluate_candidate,
    sg3_candidates,
)
from hp_twin.domain.topology import Topology

log = logging.getLogger(__name__)


@dataclass
class SgPlanConfig:
    """SG3-Planungs-Parameter (runtime.json 'sg'-Sektion)."""

    enabled: bool = False
    sg3_load_id: str = "sg3_req"
    profile_load_id: str = "hp_ext"
    min_pv_surplus_w: float = 2500.0
    max_price_eur: float | None = None  # None = nur PV-Schwelle
    windows_utc_h: list[tuple[int, int]] = field(default_factory=lambda: [(7, 8), (10, 14)])
    # Erlaubte UTC-Startstunden (MESZ-1): 7-8 = 8-9 Uhr, 10-14 = 11-15 Uhr
    max_daily_runs: int = 1
    max_duration_s: float = 3 * 3600.0
    hysteresis_energy_frac: float = 0.10
    sg3_top_setpoint_c: float = 46.0  # HK1-Soll im SG3-Fenster (real 44-50, konservativ)
    dry_run: bool = False


@dataclass
class SgPlanResult:
    """Ergebnis eines Zyklus fürs Dashboard und die Endprognose."""

    basis_p_el_w: np.ndarray | None = None
    final_p_el_w: np.ndarray | None = None
    candidates: list[SgCandidate] = field(default_factory=list)
    chosen: SgCandidate | None = None
    contingency_pushed: bool = False
    placement_slots: list[tuple[int, float]] = field(default_factory=list)
    sg3_injected: bool = False
    notes: list[str] = field(default_factory=list)
    eos_status: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        c = self.chosen
        return {
            "n_candidates": len(self.candidates),
            "candidates": [
                {
                    "start_min": round(x.start_s / 60, 1),
                    "end_min": round(x.end_s / 60, 1),
                    "energy_wh": round(x.energy_wh, 0),
                    "max_power_w": round(x.max_power_w, 0),
                    "reason": x.reason,
                }
                for x in self.candidates
            ],
            "chosen": None
            if c is None
            else {
                "start_min": round(c.start_s / 60, 1),
                "end_min": round(c.end_s / 60, 1),
                "energy_wh": round(c.energy_wh, 0),
                "reason": c.reason,
            },
            "contingency_pushed": self.contingency_pushed,
            "placement_slots": self.placement_slots,
            "sg3_injected": self.sg3_injected,
            "notes": self.notes,
            "eos_status": self.eos_status,
        }


def allowed_window_s(
    frame_start: pd.Timestamp, cfg: SgPlanConfig, horizon_s: float
) -> tuple[float, float] | None:
    """Aktuelles UTC-Startfenster, in Simulationszeit [start_s, end_s); None wenn außerhalb."""
    h0 = frame_start.hour + frame_start.minute / 60.0
    for lo, hi in cfg.windows_utc_h:
        if lo <= h0 < hi:
            end_offset = min((hi - h0) * 3600.0, horizon_s)
            if end_offset <= 0:
                return None
            return 0.0, end_offset
    return None


def sg3_energy_vs_basis(
    basis: ForecastResultLike,  # type: ignore[name-defined]  # noqa: F821
    start_s: float,
    end_s: float,
    running_threshold_w: float,
    rated_w: float,
    dt_s: float = 60.0,
) -> float:
    """Differenzenergie (Wh) eines SG3-Laufs gegen die Basis im selben Fenster.

    Erste Näherung (konservativ): Im Fenster liefert der SG3-Lauf ~rated_w (WP läuft
    durch bis SG3-Top); die im Basis-Lauf während des Fensters bereits geplanten Läufe
    werden abgezogen (der SG3-Lauf übernimmt sie). Feinjustierung passiert in der
    Endprognose — das Contingent dient nur der Slot-Platzierung durch EOS.
    """
    i0 = int(round(start_s / dt_s))
    i1 = max(i0 + 1, int(round(end_s / dt_s)))
    p = np.asarray(basis.deterministic_p_el_w, dtype=float)
    seg = p[i0:i1]
    basis_energy = float(np.sum(np.where(seg > running_threshold_w, seg, 0.0))) * dt_s / 3600.0
    window_energy = rated_w * (i1 - i0) * dt_s / 3600.0
    return float(max(0.0, window_energy - basis_energy))


def simulate_sg3_run(
    forecast_fn: Callable[..., Any],
    df: pd.DataFrame,
    topology: Topology,
    calib: CalibrationReport,
    model: ModelConfig,
    start_s: float,
    end_s: float,
    top_c: float,
    running_threshold_w: float,
) -> tuple[float, float, float]:
    """Simuliert einen SG3-Lauf am Kandidat-Fenster: (energy_wh, max_power_w, run_end_s).

    Kette: Basis-Lauf + injizierter SG3-Lauf (run_live_forecast, je ein deterministischer
    Replay). Der Regler entscheidet Start/Stopp exakt (Ladeziel + Overshoot, min_off);
    daraus folgt:

    - energy_wh: elektrische Energie des SG3-Laufs MINUS die Basis-Energie im selben
      Intervall (verdrängte Basis-Läufe abgezogen — EOS soll nur die EXTRA-Energie
      planen, nicht doppelt)
    - max_power_w: echte Spitzenleistung aus dem Kennfeld-Lauf
    - run_end_s: tatsächliches Laufende (Regler-Stopp) -> Deadline-Basis
    """
    dt_s = float(df.index.to_series().diff().dropna().iloc[0].total_seconds())
    basis_like = forecast_fn(df, topology, calib, model=model)
    final = forecast_fn(
        df, topology, calib, model=model, sg3_start_s=start_s, sg3_end_s=end_s, sg3_top_c=top_c
    )
    p_basis = np.asarray(basis_like.deterministic_p_el_w, dtype=float)
    p_final = np.asarray(final.deterministic_p_el_w, dtype=float)

    run_start, run_end = start_s, end_s
    seg_final = p_final[int(start_s / dt_s) : min(len(p_final), int(end_s / dt_s) + 1)]
    max_power = float(np.max(seg_final)) if len(seg_final) else 0.0
    for ev in final.events:
        # Erster Lauf, der das Fenster überlappt (Start VOR Fensterende, Ende NACH Start)
        if ev.start_s <= end_s and ev.end_s > start_s:
            run_start, run_end = float(ev.start_s), float(ev.end_s)
            break
    i0 = int(run_start / dt_s)
    i1 = max(i0 + 1, min(int(run_end / dt_s), len(p_basis)))
    sg3_energy = float(np.sum(p_final[i0 : min(i1, len(p_final))])) * dt_s / 3600.0
    basis_energy = (
        float(np.sum(np.where(p_basis[i0:i1] > running_threshold_w, p_basis[i0:i1], 0.0)))
        * dt_s
        / 3600.0
    )
    return float(max(0.0, sg3_energy - basis_energy)), max_power, run_end


def push_and_collect_placement(
    chosen: SgCandidate,
    cfg: SgPlanConfig,
    eos_base_url: str,
    eos_token: str | None,
    last_total_wh: float | None = None,
) -> tuple[bool, list[tuple[int, float]], dict[str, Any], list[str]]:
    """Contingent pushen (mit Hysterese) und Placement lesen.

    Gibt (gepusht, slots, status, notes) — Slots als (index, Wh), leer wenn nicht legbar.
    """
    notes: list[str] = []
    if not eos_base_url:
        notes.append("Keine EOS-Basis-URL -> Contingent nicht gepusht")
        return False, [], {}, notes
    if last_total_wh is not None and abs(
        chosen.energy_wh - last_total_wh
    ) <= cfg.hysteresis_energy_frac * max(last_total_wh, 1.0):
        notes.append(
            f"Hysterese: {chosen.energy_wh:.0f} Wh ≈ zuletzt {last_total_wh:.0f} Wh -> kein Push"
        )
        return False, [], {}, notes
    status: dict[str, Any] = {}
    if not cfg.dry_run:
        try:
            resp = push_contingent(
                eos_base_url,
                cfg.sg3_load_id,
                EosContingent(chosen.energy_wh, chosen.deadline_hours(), chosen.reason),
                token=eos_token,
            )
            notes.append(f"Contingent {resp.get('total_wh', '?')} Wh -> EOS akzeptiert")
            status = dict(resp)
        except RuntimeError as e:
            notes.append(f"EOS-Push fehlgeschlagen ({e})")
            return False, [], status, notes
    else:
        notes.append("Dry-Run: Contingent nicht gepusht")
        return False, [], status, notes
    # Placement lesen (sofort; EOS plant im 30-s-Zyklus, Folgetakt ergänzt)
    placeholders: list[tuple[int, float]] = []
    try:
        st = read_load_status(eos_base_url, cfg.sg3_load_id, token=eos_token)
        placeholders = planned_slots_from_status(st)
        status = st if isinstance(st, dict) else status
    except RuntimeError as e:
        notes.append(f"EOS-Status-Lesen fehlgeschlagen ({e})")
    return True, placeholders, status, notes


def run_sg_cycle(
    df: pd.DataFrame,
    topology: Topology,
    calib: CalibrationReport,
    model: ModelConfig,
    forecast_fn: Callable[..., Any],
    market: MarketSeries | None = None,
    cfg: SgPlanConfig | None = None,
    eos_base_url: str = "",
    eos_token: str | None = None,
) -> tuple[Any, SgPlanResult]:
    """Ein voller SG-Zyklus. forecast_fn(df, topology, calib, *, model=..., weather=...,
    sg3_start_start_s=..., sg3_end_s=...) -> ForecastResult (forecast.run_live_forecast).

    Rgibt (ForecastResult der Endprognose, SgPlanResult).
    """
    cfg = cfg or SgPlanConfig()
    res = SgPlanResult()
    horizon_h = model.sim.horizon_h

    # 1) Basis-Lauf
    basis = forecast_fn(df, topology, calib, horizon_h=horizon_h, model=model)
    res.basis_p_el_w = np.asarray(basis.deterministic_p_el_w, dtype=float).copy()
    if not cfg.enabled:
        res.notes.append("SG-Planung deaktiviert -> Endprognose = Basis (SG2 only)")
        return basis, res

    dt_s = float(df.index.to_series().diff().dropna().iloc[0].total_seconds())
    frame_start = df.index[-1] + pd.Timedelta(seconds=dt_s)
    window = allowed_window_s(frame_start, cfg, horizon_h * 3600.0)
    if window is None:
        res.notes.append("Jetzt außerhalb der SG3-UTC-Startfenster -> kein Kandidat")
        return basis, res
    w_start, w_end = window

    # 2) Kandidaten über den Basis-Lauf (Pufferverlauf kommt aus dem Forecaast-Objekt)
    buffer_temps = np.asarray(basis.buffer_sim_c, dtype=float)
    running_w = res.basis_p_el_w
    cands_time = sg3_candidates(
        buffer_temps,
        np.full(len(buffer_temps), 32.0),
        buffer_overshoot_k=abs(calib.buffer_threshold.stop_offset_k),
        running_w=running_w,
        running_threshold_w=model.forecast.running_threshold_w,
        min_off_s=topology.controller.min_off_s,
        dt_s=dt_s,
        allowed_start_s=w_start,
        allowed_end_s=w_end,
        sg3_top_c=cfg.sg3_top_setpoint_c,
        max_duration_s=cfg.max_duration_s,
    )
    if not cands_time:
        res.notes.append("Keine thermisch möglichen Fenster im Basis-Lauf")
        return basis, res

    solar = np.asarray(market.solar_surplus_w, dtype=float) if market is not None else None
    prices = (
        market.price_at(_slot_index(frame_start, len(buffer_temps))) if market is not None else None
    )
    for start_s, end_s in cands_time:
        # Echter simulierter SG3-Lauf pro Kandidat (energie, Leistung, Stopzeit)
        energy_wh, max_power_w, run_end_s = simulate_sg3_run(
            forecast_fn,
            df,
            topology,
            calib,
            model,
            start_s,
            end_s,
            cfg.sg3_top_setpoint_c,
            model.forecast.running_threshold_w,
        )
        if energy_wh <= 0 or max_power_w <= model.forecast.running_threshold_w:
            continue  # SG3 bringt kein extra Signal (Puffer schon oben) -> kein Kandidat
        cand = evaluate_candidate(
            start_s,
            max(end_s, run_end_s),
            energy_wh,
            max_power_w,
            model.forecast.slot_s,
            solar,
            cfg.min_pv_surplus_w,
            prices,
            cfg.max_price_eur,
        )
        if cand is not None:
            res.candidates.append(cand)
    if not res.candidates:
        res.notes.append("Kandidaten vorhanden, aber ohne wirtschaftliches Signal (PV/Preis)")
        return basis, res

    # 3+4) Contingent pushen & Placement lesen
    chosen = res.candidates[0]
    res.chosen = chosen
    pushed, slots, status, notes = push_and_collect_placement(
        chosen, cfg, eos_base_url, eos_token, last_total_wh=None
    )
    res.contingency_pushed = pushed
    res.placement_slots = slots
    res.eos_status = {k: v for k, v in status.items() if k in ("planned_wh", "kind", "valid_until")}
    res.notes.extend(notes)
    if not pushed:
        return basis, res
    if not slots:
        res.notes.append("Contingent akzeptiert, Placement noch leer -> Endprognose im Folgetakt")
        return basis, res

    # 5) Endprognose: erste gelegte Startzeit injizieren
    slot0 = min(s for s, wh in slots)
    start_dt = frame_start + pd.Timedelta(seconds=slot0 * model.forecast.slot_s)
    start_s = (start_dt - frame_start).total_seconds()
    final = forecast_fn(
        df,
        topology,
        calib,
        horizon_h=horizon_h,
        model=model,
        sg3_start_s=start_s,
        sg3_end_s=start_s + cfg.max_duration_s,
        sg3_top_c=cfg.sg3_top_setpoint_c,
    )
    res.final_p_el_w = np.asarray(final.deterministic_p_el_w, dtype=float).copy()
    res.sg3_injected = True
    return final, res


def _slot_index(frame_start: pd.Timestamp, n_slots: int) -> pd.DatetimeIndex:
    """15-min-Slot-Achse ab Frame-Start (Ausrichtung auf Viertelstunden)."""
    anchor = frame_start.floor("15min")
    return pd.date_range(anchor, periods=n_slots, freq="15min", tz="UTC")
