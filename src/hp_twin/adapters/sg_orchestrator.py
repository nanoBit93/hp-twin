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
    plan_signature,
    planned_slots_from_status,
    planned_windows_from_status,
    push_contingent,
    read_load_status,
)
from hp_twin.adapters.evcc import MarketSeries
from hp_twin.adapters.replay import initial_controller_state
from hp_twin.domain.calibration import CalibrationReport
from hp_twin.domain.model_config import ModelConfig
from hp_twin.domain.sg_plan import (
    SgCandidate,
    evaluate_sg3_economics,
    sg3_candidates,
    sg3_start_feasible_slots,
)
from hp_twin.domain.topology import Topology

log = logging.getLogger(__name__)


@dataclass
class SgPlanConfig:
    """SG3-Planungs-Parameter (runtime.json 'sg'-Sektion)."""

    enabled: bool = False
    sg3_load_id: str = "sg3_req"
    windows_utc_h: list[tuple[int, int]] = field(default_factory=lambda: [(7, 8), (10, 14)])
    max_duration_s: float = 3 * 3600.0
    hysteresis_energy_frac: float = 0.10
    # Measured median SG3 setpoint of the reference installation (49.8 °C).
    # Compressor cycling remains dynamic inside the SG3 window.
    sg3_top_setpoint_c: float = 49.8
    min_cop: float = 1.5
    bivalence_outdoor_c: float = 0.0
    contingent_ttl_minutes: float = 60.0
    dry_run: bool = False


@dataclass
class Sg3CycleState:
    """Offer/placement state shared by the service across scheduler cycles."""

    offered_total_wh: float | None = None
    offered_duration_s: float | None = None
    offered_setpoint_end_s: float | None = None
    offered_setpoint_end_iso: str | None = None
    offered_frame_start_iso: str | None = None
    pending: bool = False
    adopted: bool = False


@dataclass
class SgPlanResult:
    """Ergebnis eines Zyklus fürs Dashboard und die Endprognose."""

    basis_p_el_w: np.ndarray | None = None
    final_p_el_w: np.ndarray | None = None
    candidates: list[SgCandidate] = field(default_factory=list)
    rejected_candidates: list[dict[str, Any]] = field(default_factory=list)
    chosen: SgCandidate | None = None
    contingency_pushed: bool = False
    contingency_withdrawn: bool = False
    placement_slots: list[tuple[int, float]] = field(default_factory=list)
    placement_rejected: bool = False
    sg3_injected: bool = False
    feasible: list[bool] = field(default_factory=list)
    feasible_reasons: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    eos_status: dict[str, Any] = field(default_factory=dict)
    frame_start: pd.Timestamp | None = None
    allowed_windows: list[tuple[float, float]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        c = self.chosen
        frame_start = self.frame_start.isoformat() if self.frame_start is not None else None
        return {
            "frame_start": frame_start,
            "allowed_windows": [
                {
                    "start_s": start_s,
                    "end_s": end_s,
                    "start_iso": (
                        (self.frame_start + pd.Timedelta(seconds=start_s)).isoformat()
                        if self.frame_start is not None
                        else None
                    ),
                    "end_iso": (
                        (self.frame_start + pd.Timedelta(seconds=end_s)).isoformat()
                        if self.frame_start is not None
                        else None
                    ),
                }
                for start_s, end_s in self.allowed_windows
            ],
            "n_candidates": len(self.candidates),
            "rejected_candidates": list(self.rejected_candidates),
            "candidates": [
                {
                    "start_min": round(x.start_s / 60, 1),
                    "end_min": round(x.end_s / 60, 1),
                    "runtime_min": round(x.runtime_s / 60, 1),
                    "setpoint_end_min": round(x.setpoint_end_s / 60, 1),
                    "energy_wh": round(x.energy_wh, 0),
                    "max_power_w": round(x.max_power_w, 0),
                    "saving_eur": x.saving_eur,
                    "value_eur_per_kwh": x.value_eur_per_kwh,
                    "reason": x.reason,
                    "thermal": x.thermal,
                }
                for x in self.candidates
            ],
            "chosen": None
            if c is None
            else {
                "start_min": round(c.start_s / 60, 1),
                "end_min": round(c.end_s / 60, 1),
                "runtime_min": round(c.runtime_s / 60, 1),
                "setpoint_end_min": round(c.setpoint_end_s / 60, 1),
                "energy_wh": round(c.energy_wh, 0),
                "max_power_w": round(c.max_power_w, 0),
                "saving_eur": c.saving_eur,
                "baseline_cost_eur": c.baseline_cost_eur,
                "sg3_cost_eur": c.sg3_cost_eur,
                "avoided_eur": c.avoided_eur,
                "value_eur_per_kwh": c.value_eur_per_kwh,
                "reason": c.reason,
                "thermal": c.thermal,
            },
            "feasible": self.feasible,
            "feasible_reasons": self.feasible_reasons,
            "feasible_slots": sum(self.feasible),
            "contingency_pushed": self.contingency_pushed,
            "contingency_withdrawn": self.contingency_withdrawn,
            "placement_slots": self.placement_slots,
            "placement_rejected": self.placement_rejected,
            "sg3_injected": self.sg3_injected,
            "notes": self.notes,
            "eos_status": self.eos_status,
        }


def allowed_windows_s(
    frame_start: pd.Timestamp, cfg: SgPlanConfig, horizon_s: float
) -> list[tuple[float, float]]:
    """All configured future start windows within the forecast horizon."""
    start = pd.Timestamp(frame_start)
    candidates: list[tuple[float, float]] = []
    for day_offset in range(3):
        day = (start + pd.Timedelta(days=day_offset)).normalize()
        for lo, hi in cfg.windows_utc_h:
            begin = day + pd.Timedelta(hours=lo)
            end = day + pd.Timedelta(hours=hi)
            if end <= start:
                continue
            begin = max(begin, start)
            begin_s = (begin - start).total_seconds()
            end_s = min((end - start).total_seconds(), horizon_s)
            if begin_s < end_s:
                candidates.append((max(0.0, begin_s), end_s))
    return sorted(candidates)


def simulate_sg3_run(
    forecast_fn: Callable[..., Any],
    df: pd.DataFrame,
    topology: Topology,
    calib: CalibrationReport,
    model: ModelConfig,
    start_s: float,
    end_s: float,
    top_c: float,
    *,
    basis: Any | None = None,
    weather: pd.DataFrame | None = None,
) -> tuple[float, float, float, Any, dict[str, Any]]:
    """Simulate a full target-driven SG3 operation and return its extra energy.

    `end_s` limits the permitted SG3 setpoint operation; it is not presumed to be
    the compressor runtime. A real elevated-target stop is mandatory. The rounded
    compressor end is the EOS runtime, while the later setpoint-window end remains
    the placement deadline so EOS cannot postpone a run beyond its thermal model.
    """
    dt_s = float(df.index.to_series().diff().dropna().iloc[0].total_seconds())
    target_c = top_c
    if calib.sg3_buffer_threshold is not None:
        target_c += calib.sg3_buffer_threshold.stop_offset_k
    kwargs = {"horizon_h": model.sim.horizon_h, "model": model, "weather": weather}
    basis_like = basis or forecast_fn(df, topology, calib, **kwargs)
    final = forecast_fn(
        df,
        topology,
        calib,
        **kwargs,
        sg3_windows=[(start_s, end_s)],
        sg3_top_c=top_c,
    )
    p_basis = np.asarray(basis_like.deterministic_p_el_w, dtype=float)
    p_final = np.asarray(final.deterministic_p_el_w, dtype=float)
    if len(p_basis) != len(p_final):
        raise ValueError("SG2 and SG3 scenarios use different time axes")
    diagnostics: dict[str, Any] = {
        "start_s": start_s,
        "setpoint_end_s": end_s,
        "target_buffer_c": target_c,
        "target_tolerance_k": 0.5,
        "target_reached": False,
        "completion_s": None,
        "following_off_s": None,
        "next_normal_start_s": None,
    }
    if final.buffer_sim_c is None or getattr(final, "controller_mode", None) is None:
        raise ValueError("SG3 thermal trace is unavailable; no shortened offer is allowed")
    buffer = np.asarray(final.buffer_sim_c, dtype=float)
    modes = list(final.controller_mode)
    if not (len(p_final) == len(buffer) == len(modes)):
        raise ValueError("SG3 thermal traces use different time axes")
    i0 = int(np.floor(start_s / dt_s))
    if i0 >= len(p_final) or end_s <= start_s:
        raise ValueError("SG3 operation lies outside the forecast")
    diagnostics["start_buffer_c"] = (
        float(df["t_hk1_c"].iloc[-1])
        if i0 == 0
        else float(basis_like.buffer_sim_c[i0])
    )
    i_end = min(len(p_final), int(np.ceil(end_s / dt_s)))
    completion_i: int | None = None
    stop_i: int | None = None
    for index in range(i0, i_end):
        if modes[index] != "heat":
            continue
        next_index = index + 1
        if buffer[index] >= target_c and (
            next_index >= len(modes)
            or next_index >= i_end
            or modes[next_index] not in ("heat", "dhw")
        ):
            completion_i = index
            stop_i = index + 1
            diagnostics["target_reached"] = True
            diagnostics["completion_s"] = stop_i * dt_s
            diagnostics["end_buffer_c"] = float(buffer[index])
            break
    if completion_i is None or stop_i is None:
        diagnostics["max_buffer_c"] = float(np.max(buffer[i0:i_end]))
        raise ValueError("SG3 target not reached before its setpoint window ends")
    if buffer[stop_i - 1] < target_c - 1e-6:
        raise ValueError("SG3 compressor stopped before reaching its thermal target")
    next_start_i = next(
        (index for index in range(stop_i, len(modes)) if modes[index] in ("heat", "dhw")),
        None,
    )
    diagnostics["next_normal_start_s"] = (
        next_start_i * dt_s if next_start_i is not None else None
    )
    diagnostics["following_off_s"] = (
        (next_start_i - stop_i) * dt_s if next_start_i is not None else None
    )
    run_end_s = start_s + max(1, int(np.ceil((stop_i * dt_s - start_s) / 900.0))) * 900.0
    diagnostics["offer_end_s"] = run_end_s
    diagnostics["setpoint_end_s"] = max(end_s, run_end_s)
    i1 = min(len(p_final), int(np.ceil(run_end_s / dt_s)))
    seg_final = p_final[i0:i1]
    # The flexible offer is the extra energy of the full compressor operation,
    # not the whole window and not a later displaced SG2 run.
    energy_wh = float(np.sum(np.maximum(seg_final - p_basis[i0:i1], 0.0)) * dt_s / 3600.0)
    max_power = float(np.max(seg_final)) if len(seg_final) else 0.0
    return energy_wh, max_power, run_end_s, final, diagnostics


def push_and_collect_placement(
    chosen: SgCandidate,
    cfg: SgPlanConfig,
    eos_base_url: str,
    eos_token: str | None,
    last_total_wh: float | None = None,
    *,
    frame_start: pd.Timestamp | None = None,
    state: Sg3CycleState | None = None,
) -> tuple[bool, list[tuple[int, float]], dict[str, Any], list[str]]:
    """Contingent pushen (mit Hysterese) und Placement lesen.

    Gibt (gepusht, slots, status, notes) — Slots als (index, Wh), leer wenn nicht legbar.
    """
    notes: list[str] = []
    if not eos_base_url:
        notes.append("Keine EOS-Basis-URL -> Contingent nicht gepusht")
        return False, [], {}, notes
    if last_total_wh is not None and chosen.energy_wh > 0 and frame_start is not None:
        # The same physical run must not be re-offered on every poll; a different
        # forecast day, however, is a new offer even at exactly the same energy.
        current_day = pd.Timestamp(frame_start).floor("D")
        reference_day = (
            pd.Timestamp(state.offered_frame_start_iso).floor("D")
            if state is not None and state.offered_frame_start_iso is not None
            else None
        )
        if reference_day == current_day and abs(
            chosen.energy_wh - last_total_wh
        ) <= cfg.hysteresis_energy_frac * max(last_total_wh, 1.0):
            notes.append(
                f"Hysterese: {chosen.energy_wh:.0f} Wh ≈ "
                f"zuletzt {last_total_wh:.0f} Wh -> kein Push"
            )
            return False, [], {}, notes
    status: dict[str, Any] = {}
    if not cfg.dry_run:
        try:
            resp = push_contingent(
                eos_base_url,
                cfg.sg3_load_id,
                EosContingent(
                    chosen.energy_wh,
                    chosen.deadline_hours(),
                    chosen.reason,
                    feasible=list(chosen.feasible) if chosen.feasible is not None else None,
                    value_eur_per_kwh=chosen.value_eur_per_kwh,
                    max_power_w=chosen.max_power_w,
                    start=(frame_start or pd.Timestamp("1970-01-01T00:00:00Z")).isoformat(),
                    deadline=(frame_start + pd.Timedelta(seconds=chosen.setpoint_end_s)).isoformat()
                    if frame_start is not None
                    else None,
                    ttl_minutes=cfg.contingent_ttl_minutes,
                    min_runtime_slots=max(1, int(round(chosen.runtime_s / 900.0))),
                    contiguous=True,
                ),
                token=eos_token,
            )
            notes.append(f"Contingent {resp.get('total_wh', '?')} Wh -> EOS akzeptiert")
            status = dict(resp)
            if state is not None:
                state.offered_total_wh = chosen.energy_wh
                state.offered_duration_s = chosen.runtime_s
                state.offered_setpoint_end_s = chosen.setpoint_end_s
                state.offered_frame_start_iso = frame_start.isoformat()
                state.offered_setpoint_end_iso = (
                    frame_start + pd.Timedelta(seconds=chosen.setpoint_end_s)
                ).isoformat()
                state.pending = True
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


def withdraw_contingent(
    eos_base_url: str, cfg: SgPlanConfig, eos_token: str | None, frame_start: pd.Timestamp
) -> dict[str, Any]:
    """Remove adopted energy before the final replacement profile becomes authoritative."""
    return push_contingent(
        eos_base_url,
        cfg.sg3_load_id,
        EosContingent(0.0, 1 / 60, start=frame_start.isoformat(), ttl_minutes=1),
        token=eos_token,
    )


def _adopt_placement(
    *,
    df: pd.DataFrame,
    topology: Topology,
    calib: CalibrationReport,
    model: ModelConfig,
    forecast_fn: Callable[..., Any],
    weather: pd.DataFrame | None,
    chosen: SgCandidate,
    cfg: SgPlanConfig,
    status: dict[str, Any],
    frame_start: pd.Timestamp,
    res: SgPlanResult,
    state: Sg3CycleState,
    eos_base_url: str = "",
    eos_token: str | None = None,
) -> Any | None:
    """Adopt exactly one uninterrupted SG3 window of the offered duration."""
    res.placement_rejected = False
    try:
        windows = planned_windows_from_status(status, frame_start, model.sim.horizon_h * 3600.0)
    except (TypeError, ValueError) as exc:
        res.notes.append(f"Placement verworfen: {exc}")
        res.placement_rejected = True
        return None
    if not windows:
        return None
    expected_s = chosen.runtime_s
    if len(windows) != 1 or (windows[0][1] - windows[0][0]) != expected_s:
        res.notes.append(
            "Placement verworfen: SG3 benötigt genau einen ununterbrochenen "
            f"Block von {expected_s / 60:.0f} Minuten"
        )
        res.placement_rejected = True
        return None
    if chosen.end_s > 0 and windows[0][1] > chosen.setpoint_end_s + 1e-6:
        res.notes.append("Placement verworfen: liegt nach dem simulierten Sollfenster")
        res.placement_rejected = True
        return None
    final = forecast_fn(
        df,
        topology,
        calib,
        horizon_h=model.sim.horizon_h,
        model=model,
        weather=weather,
        sg3_windows=windows,
        sg3_top_c=cfg.sg3_top_setpoint_c,
    )
    if chosen.thermal is not None:
        # The adopted start can be later than the simulated offer. Validate its
        # actual target stop instead of merely trusting the slot block length.
        try:
            _, _, _, final, thermal = simulate_sg3_run(
                forecast_fn,
                df,
                topology,
                calib,
                model,
                windows[0][0],
                windows[0][1],
                cfg.sg3_top_setpoint_c,
                weather=weather,
            )
        except (AttributeError, TypeError, ValueError) as exc:
            res.notes.append(f"Placement verworfen: Ziel nicht erreicht ({exc})")
            res.placement_rejected = True
            return None
        chosen.thermal["adopted_start_s"] = windows[0][0]
        chosen.thermal["adopted_completion_s"] = thermal["completion_s"]
        chosen.thermal["adopted_following_off_s"] = thermal["following_off_s"]
        chosen.thermal["adopted_target_reached"] = thermal["target_reached"]
        chosen.thermal["adopted_target_buffer_c"] = thermal["target_buffer_c"]
        chosen.thermal["adopted_end_buffer_c"] = thermal["end_buffer_c"]
        if not thermal["target_reached"]:
            res.notes.append("Placement verworfen: Ziel nicht erreicht")
            res.placement_rejected = True
            return None
    res.placement_slots = planned_slots_from_status(status)
    res.final_p_el_w = np.asarray(final.deterministic_p_el_w, dtype=float).copy()
    if hasattr(final, "eos") and final.eos.thermal is not None:
        final.eos.thermal.update({
            "plan": plan_signature(status, frame_start, cfg.sg3_load_id) or {}
        })
    res.sg3_injected = True
    res.notes.append("Ununterbrochener SG3-Block mit nachgewiesenem Zielabschluss injiziert")
    state.pending = False
    state.adopted = True
    if eos_base_url and not cfg.dry_run:
        try:
            withdraw_contingent(eos_base_url, cfg, eos_token, frame_start)
            res.contingency_withdrawn = True
            res.notes.append(
                "Kontingent nach der Endsimulation zurückgezogen; "
                "das folgende Gesamtprofil ist alleinige Wahrheit"
            )
        except RuntimeError as exc:
            # A failed withdrawal leaves the old offer authoritative. Publishing
            # the replacement would count the same SG3 energy twice.
            res.notes.append(f"Kontingent-Rückzug fehlgeschlagen ({exc}); Basis bleibt")
            state.pending = True
            state.adopted = False
            res.sg3_injected = False
            return None
    return final


def _public_status(status: dict[str, Any]) -> dict[str, Any]:
    return {
        key: status.get(key)
        for key in (
            "planned_wh",
            "kind",
            "valid_until",
            "fallback_active",
            "plan_anchor",
            "time_frame_base",
        )
        if key in status
    }


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
    weather: pd.DataFrame | None = None,
    state: Sg3CycleState | None = None,
) -> tuple[Any, SgPlanResult]:
    """Ein voller SG-Zyklus. forecast_fn(df, topology, calib, *, model=..., weather=...,
    sg3_start_start_s=..., sg3_end_s=...) -> ForecastResult (forecast.run_live_forecast).

    Rgibt (ForecastResult der Endprognose, SgPlanResult).
    """
    cfg = cfg or SgPlanConfig()
    state = state or Sg3CycleState()
    res = SgPlanResult()
    horizon_h = model.sim.horizon_h

    # 1) Basis-Lauf
    basis = forecast_fn(df, topology, calib, horizon_h=horizon_h, model=model, weather=weather)
    res.basis_p_el_w = np.asarray(basis.deterministic_p_el_w, dtype=float).copy()
    if not cfg.enabled:
        res.notes.append("SG-Planung deaktiviert -> Endprognose = Basis (SG2 only)")
        return basis, res
    if horizon_h != 48 or model.forecast.slot_s != 900:
        res.notes.append("SG3 benötigt den 48-h-Horizont mit 15-min-Slots -> SG2")
        return basis, res

    dt_s = float(df.index.to_series().diff().dropna().iloc[0].total_seconds())
    frame_start = (
        pd.Timestamp(basis.frame_start)
        if basis.frame_start is not None
        else (df.index[-1] + pd.Timedelta(seconds=dt_s)).floor("15min")
    )
    res.frame_start = frame_start
    if state.adopted:
        # The completed adoption already withdrew the contingent. Keep its energy
        # as the hysteresis reference so the same run is not offered again merely
        # because the contingent itself is no longer active.
        state.adopted = False
        state.offered_duration_s = None
        state.offered_setpoint_end_s = None
    if state.pending and eos_base_url and not cfg.dry_run:
        try:
            status = read_load_status(eos_base_url, cfg.sg3_load_id, token=eos_token)
            res.eos_status = _public_status(status)
            # EOS intentionally keeps pushed data in memory. After its restart
            # there is no live offer to wait for; create a fresh candidate rather
            # than retaining pending forever. An expired push is equally stale.
            if (
                not status.get("pushed", True)
                or not planned_slots_from_status(status)
                or status.get("reason") in ("nothing pushed yet", "pushed data expired")
            ):
                state.pending = False
                state.offered_total_wh = None
                state.offered_duration_s = None
                state.offered_setpoint_end_s = None
                res.notes.append("Kein gültiges EOS-Kontingent mehr; SG3 wird neu geplant")
            else:
                offered_deadline_s = state.offered_setpoint_end_s
                if state.offered_setpoint_end_iso is not None:
                    offered_deadline_s = max(
                        0.0,
                        (
                            pd.Timestamp(state.offered_setpoint_end_iso) - frame_start
                        ).total_seconds(),
                    )
                chosen = SgCandidate(
                    0.0,
                    state.offered_duration_s or cfg.max_duration_s,
                    state.offered_total_wh or 0.0,
                    0.0,
                    "previous offer",
                    thermal={
                        "setpoint_end_s": offered_deadline_s
                        or state.offered_duration_s
                        or cfg.max_duration_s,
                    },
                )
                res.chosen = chosen
                final = _adopt_placement(
                    df=df,
                    topology=topology,
                    calib=calib,
                    model=model,
                    forecast_fn=forecast_fn,
                    weather=weather,
                    chosen=chosen,
                    cfg=cfg,
                    status=status,
                    frame_start=frame_start,
                    res=res,
                    state=state,
                    eos_base_url=eos_base_url,
                    eos_token=eos_token,
                )
                return (final or basis), res
        except RuntimeError as exc:
            res.notes.append(f"Offenes Placement nicht lesbar ({exc}); SG2 bis Folgetakt")
            return basis, res
    windows = allowed_windows_s(frame_start, cfg, horizon_h * 3600.0)
    res.allowed_windows = list(windows)
    if not windows:
        res.notes.append("Kein SG3-Startfenster im Prognosehorizont -> kein Kandidat")
        return basis, res
    if market is None:
        res.notes.append("Keine vollständigen Marktdaten -> kein SG3-Angebot")
        return basis, res
    if basis.outdoor_c is None or basis.heat_pump is None or basis.buffer_sim_c is None:
        res.notes.append("Thermische Prognosedaten unvollständig -> kein SG3-Angebot")
        return basis, res
    try:
        feasible, feasible_reasons = sg3_start_feasible_slots(
            t_buffer_c=basis.buffer_sim_c,
            running_w=res.basis_p_el_w,
            outdoor_c=basis.outdoor_c,
            initial_controller_state=initial_controller_state(df, topology.controller.min_off_s),
            heat_pump=basis.heat_pump,
            controller_config=topology.controller,
            sg3_top_c=cfg.sg3_top_setpoint_c,
            flow_offset_k=10.0,
            buffer_overshoot_k=abs(calib.buffer_threshold.stop_offset_k),
            running_threshold_w=model.forecast.running_threshold_w,
            dt_s=dt_s,
            slot_s=900.0,
            allowed_start_s=0.0,
            allowed_end_s=horizon_h * 3600.0,
            max_duration_s=cfg.max_duration_s,
            min_cop=cfg.min_cop,
            bivalence_outdoor_c=cfg.bivalence_outdoor_c,
        )
    except (AttributeError, TypeError, ValueError) as exc:
        res.notes.append(f"Feasibility nicht bestimmbar ({exc}) -> SG2")
        return basis, res
    res.feasible = feasible
    res.feasible_reasons = feasible_reasons
    if not any(feasible):
        res.notes.append("Kein thermisch ausführbarer SG3-Startslot")
        return basis, res

    # 2) Kandidaten über den Basis-Lauf (Pufferverlauf kommt aus dem Forecaast-Objekt)
    buffer_temps = np.asarray(basis.buffer_sim_c, dtype=float)
    running_w = res.basis_p_el_w
    cands_time = []
    for w_start, w_end in windows:
        for free_start, _free_end in sg3_candidates(
            buffer_temps,
            buffer_overshoot_k=abs(calib.buffer_threshold.stop_offset_k),
            running_w=running_w,
            running_threshold_w=model.forecast.running_threshold_w,
            min_off_s=topology.controller.min_off_s,
            dt_s=dt_s,
            allowed_start_s=w_start,
            allowed_end_s=w_end,
            sg3_top_c=cfg.sg3_top_setpoint_c,
        ):
            start_s = float(np.ceil(free_start / 900.0) * 900.0)
            end_s = start_s + cfg.max_duration_s
            if end_s <= horizon_h * 3600.0:
                cands_time.append((start_s, end_s))
    if not cands_time:
        res.notes.append("Keine thermisch möglichen Fenster im Basis-Lauf")
        return basis, res

    prices = market.price_at(_slot_index(frame_start, int(np.ceil(horizon_h * 3600 / 900.0))))
    for start_s, end_s in cands_time:
        if not feasible[int(np.floor(start_s / 900.0))]:
            res.rejected_candidates.append(
                {
                    "start_s": start_s,
                    "reason": "start_slot_not_feasible",
                }
            )
            continue
        try:
            energy_wh, max_power_w, run_end_s, final, thermal = simulate_sg3_run(
                forecast_fn,
                df,
                topology,
                calib,
                model,
                start_s,
                end_s,
                cfg.sg3_top_setpoint_c,
                basis=basis,
                weather=weather,
            )
            economics = evaluate_sg3_economics(
                basis.deterministic_p_el_w,
                final.deterministic_p_el_w,
                prices,
                dt_s=dt_s,
                slot_s=900.0,
                outdoor_c=basis.outdoor_c,
                offered_wh=energy_wh,
            )
        except (AttributeError, TypeError, ValueError) as exc:
            res.rejected_candidates.append(
                {
                    "start_s": start_s,
                    "reason": "economics",
                    "detail": str(exc),
                }
            )
            res.notes.append(f"SG3-Kandidat verworfen: {exc}")
            continue
        if energy_wh <= 0 or max_power_w <= model.forecast.running_threshold_w:
            res.rejected_candidates.append(
                {
                    "start_s": start_s,
                    "reason": "no_sg3_window_energy",
                    "energy_wh": energy_wh,
                    "max_power_w": max_power_w,
                }
            )
            continue
        run_mask = list(feasible)
        # The mask describes permitted *starts*, not only the earliest simulated
        # completion. EOS may delay the full run until the original deadline.
        latest_start_s = thermal["setpoint_end_s"] - (run_end_s - start_s)
        for index in range(len(run_mask)):
            if index * 900.0 > latest_start_s + 1e-6:
                run_mask[index] = False
        if economics["priced_slots"] < economics["total_slots"]:
            res.notes.append(
                f"Preisabdeckung {economics['priced_slots']}/{economics['total_slots']} Slots; "
                "unbepreiste Energie nicht bewertet"
            )
        cand = SgCandidate(
            start_s,
            run_end_s,
            energy_wh,
            max_power_w,
            "twin gross avoided-cost value",
            feasible=tuple(run_mask),
            value_eur_per_kwh=economics["value_eur_per_kwh"],
            saving_eur=economics["saving_eur"],
            avoided_eur=economics["avoided_eur"],
            baseline_cost_eur=economics["baseline_cost_eur"],
            sg3_cost_eur=economics["sg3_cost_eur"],
            thermal=thermal,
        )
        res.candidates.append(cand)
    if not res.candidates:
        res.notes.append("Kein wirtschaftlich bewertbarer SG3-Kandidat")
        return basis, res

    # 3+4) Contingent pushen & Placement lesen
    chosen = max(res.candidates, key=lambda item: item.value_eur_per_kwh or 0.0)
    res.chosen = chosen
    pushed, slots, status, notes = push_and_collect_placement(
        chosen,
        cfg,
        eos_base_url,
        eos_token,
        last_total_wh=state.offered_total_wh,
        frame_start=frame_start,
        state=state,
    )
    res.contingency_pushed = pushed
    res.placement_slots = slots
    res.eos_status = _public_status(status)
    res.notes.extend(notes)
    if not pushed:
        return basis, res
    final = _adopt_placement(
        df=df,
        topology=topology,
        calib=calib,
        model=model,
        forecast_fn=forecast_fn,
        weather=weather,
        chosen=chosen,
        cfg=cfg,
        status=status,
        frame_start=frame_start,
        res=res,
        state=state,
        eos_base_url=eos_base_url,
        eos_token=eos_token,
    )
    if final is None and not res.placement_slots:
        res.notes.append("Contingent akzeptiert, Placement noch leer -> Endprognose im Folgetakt")
    return (final or basis), res


def _slot_index(frame_start: pd.Timestamp, n_slots: float) -> pd.DatetimeIndex:
    """UTC quarter-hour axis beginning at the explicitly aligned forecast start."""
    return pd.date_range(frame_start, periods=int(np.ceil(n_slots)), freq="15min", tz="UTC")
