"""Output-Adapter für EOS Connect: Prognose -> "External load profile" (Managed Load).

Format: {"values": [...], "unit": "W", "start": "now", "ttl_minutes"?} mit 192 Viertelstunden-
werten (48 h). Es wird die gesamte WP-Leistung übergeben (in EOS "Replaces sensor" auf den
WP-Leistungssensor). Push (REST/MQTT) folgt später; hier nur die Formaterzeugung.

Zusätzlich die Schnittstelle EnergyBudgetAdapter (SG3, "External energy budget") als Stub.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from hp_twin.domain.montecarlo import SlotForecast


@dataclass(frozen=True)
class EosLoadProfile:
    values_w: list[float]
    unit: str = "W"
    start: str = "now"
    ttl_minutes: int | None = None
    thermal: dict[str, Any] | None = None

    def to_payload(self) -> dict[str, Any]:
        payload: dict[str, Any] = {"values": self.values_w, "unit": self.unit, "start": self.start}
        if self.ttl_minutes is not None:
            payload["ttl_minutes"] = self.ttl_minutes
        if self.thermal is not None:
            payload["thermal"] = self.thermal
        return payload


def thermal_sidecar(
    *,
    buffer_c: Any,
    dhw_c: Any,
    outdoor_c: Any,
    mode: Any,
    time_s: Any,
    start: pd.Timestamp,
    slot_s: float = 900.0,
    plan: dict[str, Any] | None = None,
    run_probability: Any = None,
    running_threshold_w: float | None = None,
) -> dict[str, Any] | None:
    """Deterministic final simulation state, on EOS's 15-minute grid.

    The sidecar belongs to the same ForecastResult as the electrical profile. It
    therefore describes the actual adopted SG3 placement, including the subsequent
    SG2 operation and cooldown, rather than a separate baseline diagnostic.
    """
    times = np.asarray(time_s, dtype=float)
    modes = list(mode)
    series = {
        "buffer_c": np.asarray(buffer_c, dtype=float),
        "dhw_c": np.asarray(dhw_c, dtype=float),
        "outdoor_c": np.asarray(outdoor_c, dtype=float),
    }
    if not len(times) or any(len(values) != len(times) for values in series.values()):
        return None
    if len(modes) != len(times):
        return None
    dt_s = float(np.diff(times)[0]) if len(times) > 1 else 60.0
    origin_s = float(times[0])
    n_slots = int(np.ceil((times[-1] + dt_s - origin_s) / slot_s))
    result: dict[str, Any] = {
        "buffer_c": [],
        "dhw_c": [],
        "outdoor_c": [],
        "mode": [],
    }
    for slot in range(n_slots):
        lo = int(np.floor(slot * slot_s / dt_s))
        hi = min(len(times), int(np.ceil((slot + 1) * slot_s / dt_s)))
        for name, values in series.items():
            segment = values[lo:hi]
            finite = segment[np.isfinite(segment)]
            result[name].append(round(float(np.mean(finite)), 3) if len(finite) else None)
        mode_segment = modes[lo:hi]
        # Temperatures and electrical load both respond to a partial first slot.
        # A majority-idle vote would hide precisely that real compressor start.
        operating = next((mode for mode in mode_segment if mode in ("dhw", "heat")), None)
        if operating is not None:
            result["mode"].append(operating)
        else:
            counts = pd.Series(mode_segment).value_counts()
            result["mode"].append(str(counts.index[0]) if len(counts) else "unknown")
    return {
        "start": pd.Timestamp(start).isoformat(),
        "resolution": "15min",
        "series": result,
        "plan": plan or {},
        "run_probability": (
            [round(float(value), 4) for value in np.asarray(run_probability, dtype=float)]
            if run_probability is not None
            else None
        ),
        "running_threshold_w": running_threshold_w,
    }


def plan_signature(
    status: dict[str, Any], frame_start: Any, contingent_id: str = "sg3_req"
) -> dict[str, Any] | None:
    """Identity of the adopted EOS block, including its midnight time anchor."""
    windows = planned_windows_from_status(status, frame_start, 48 * 3600.0)
    if not windows:
        return None
    start, end = windows[0]
    slot0 = int(
        round(
            (pd.Timestamp(frame_start) - pd.Timestamp(status["plan_anchor"])).total_seconds()
            / 900.0
        )
    )
    slots = [int(round(start / 900.0)) + slot0, int(round(end / 900.0)) + slot0]
    return {
        "signature": f"{status['plan_anchor']}|{slots[0]}-{slots[1]}",
        "contingent_id": contingent_id,
        "plan_anchor": status["plan_anchor"],
        "start_s": start,
        "end_s": end,
    }


def load_profile_from_slots(
    slots: SlotForecast, expected_slots: int = 192, ttl_minutes: int | None = None
) -> EosLoadProfile:
    """Erwartungswert-Leistung je 15-min-Slot -> EOS-Load-Profile (Default 192 Werte = 48 h).

    EOS erlaubt 24/48 Stundenwerte oder 96/192 Viertelstundenwerte. Wir liefern den
    Erwartungswert (Mittel über die Monte-Carlo-Läufe) der gesamten WP-Leistung.
    """
    vals = np.nan_to_num(np.asarray(slots.p_el_mean_w, float), nan=0.0, posinf=0.0, neginf=0.0)
    if slots.slot_s != 900.0:
        raise ValueError("EOS-Viertelstundenprofil erwartet 15-min-Slots (slot_s=900)")
    if len(vals) >= expected_slots:
        vals = vals[:expected_slots]
    else:
        vals = np.concatenate(
            [vals, np.full(expected_slots - len(vals), vals[-1] if len(vals) else 0.0)]
        )
    if expected_slots not in (24, 48, 96, 192):
        raise ValueError("EOS erlaubt 24/48/96/192 Werte")
    return EosLoadProfile(values_w=[round(float(v), 1) for v in vals], ttl_minutes=ttl_minutes)


def push_rest(
    base_url: str,
    managed_load_id: str,
    profile: EosLoadProfile,
    token: str | None = None,
    timeout_s: float = 15.0,
) -> int:
    """Pusht das Load-Profile per REST an EOS Connect: POST <base>/api/managed_loads/<id>/push."""
    url = f"{base_url.rstrip('/')}/api/managed_loads/{managed_load_id}/push"
    body = json.dumps(profile.to_payload()).encode("utf-8")
    headers = {"Content-Type": "application/json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    req = urllib.request.Request(url, data=body, headers=headers, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:  # noqa: S310
            return int(resp.status)
    except urllib.error.HTTPError as e:  # pragma: no cover - netzwerkabhängig
        raise RuntimeError(f"EOS-Push HTTP {e.code}: {e.read().decode('utf-8')[:200]}") from e
    except urllib.error.URLError as e:  # pragma: no cover
        raise RuntimeError(f"EOS-Push Verbindung fehlgeschlagen: {e.reason}") from e


def mqtt_topic(managed_load_id: str) -> str:
    """MQTT-Alternative (retained): eos_connect/managed_load/<id>/set mit demselben Payload."""
    return f"eos_connect/managed_load/{managed_load_id}/set"


class EosContingent:
    """Flexibles SG3-Kontingent mit explizitem Zeit-, Wert- und Leistungsvertrag."""

    def __init__(
        self,
        total_wh: float,
        deadline_hours: float,
        reason: str = "",
        *,
        feasible: list[bool] | None = None,
        value_eur_per_kwh: float | None = None,
        max_power_w: float | None = None,
        start: str = "now",
        deadline: str | None = None,
        resolution: str = "15min",
        ttl_minutes: float = 60.0,
        min_runtime_slots: int | None = None,
        contiguous: bool = False,
    ) -> None:
        self.total_wh = max(0.0, float(total_wh))
        self.deadline_hours = max(1 / 60, float(deadline_hours))
        self.deadline = deadline
        self.reason = reason
        self.feasible = feasible
        self.value_eur_per_kwh = value_eur_per_kwh
        self.max_power_w = max_power_w
        self.start = start
        self.resolution = resolution
        self.ttl_minutes = max(1.0, float(ttl_minutes))
        self.min_runtime_slots = min_runtime_slots
        self.contiguous = bool(contiguous)

    def to_payload(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "total_wh": round(self.total_wh, 1),
            "start": self.start,
            "resolution": self.resolution,
            "ttl_minutes": self.ttl_minutes,
            "contiguous": self.contiguous,
        }
        if self.deadline is not None:
            payload["deadline"] = self.deadline
        else:
            payload["deadline_hours"] = round(self.deadline_hours, 3)
        if self.min_runtime_slots is not None:
            payload["min_runtime_slots"] = self.min_runtime_slots
        if self.feasible is not None:
            payload["feasible"] = self.feasible
        if self.value_eur_per_kwh is not None:
            payload["value_eur_per_kwh"] = round(self.value_eur_per_kwh, 6)
        if self.max_power_w is not None:
            payload["max_power_w"] = round(self.max_power_w, 1)
        return payload


def push_contingent(
    base_url: str,
    managed_load_id: str,
    contingent: EosContingent,
    token: str | None = None,
    timeout_s: float = 15.0,
) -> dict[str, Any]:
    """Pusht das Contingent (SG3-Anforderung): POST <base>/api/managed_loads/<id>/push.

    EOS normalisiert auf sein 15-min-Raster und antwortet mit kind='contingent',
    total_wh, deadline_slot, valid_until (Echo zur Verifizierung).
    """
    url = f"{base_url.rstrip('/')}/api/managed_loads/{managed_load_id}/push"
    body = json.dumps(contingent.to_payload()).encode("utf-8")
    headers = {"Content-Type": "application/json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    req = urllib.request.Request(url, data=body, headers=headers, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:  # noqa: S310
            return dict(json.loads(resp.read().decode("utf-8")))
    except urllib.error.HTTPError as e:
        raise RuntimeError(
            f"EOS-Contingent-Push HTTP {e.code}: {e.read().decode('utf-8')[:200]}"
        ) from e
    except urllib.error.URLError as e:
        raise RuntimeError(f"EOS-Contingent-Push Verbindung fehlgeschlagen: {e.reason}") from e


def read_load_status(
    base_url: str, managed_load_id: str, token: str | None = None, timeout_s: float = 15.0
) -> dict[str, Any]:
    """Status eines Managed Loads inkl. 'plan' (Wh je Slot) und 'release' (Gate-Signal)."""
    url = f"{base_url.rstrip('/')}/api/managed_loads/{managed_load_id}"
    headers = {"Accept": "application/json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    req = urllib.request.Request(url, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:  # noqa: S310
            return dict(json.loads(resp.read().decode("utf-8")))
    except urllib.error.HTTPError as e:
        raise RuntimeError(f"EOS-Status HTTP {e.code}: {e.read().decode('utf-8')[:120]}") from e
    except urllib.error.URLError as e:
        raise RuntimeError(f"EOS-Status Verbindung fehlgeschlagen: {e.reason}") from e


def planned_slots_from_status(status: dict[str, Any]) -> list[tuple[int, float]]:
    """Gelegte Slots aus einem Load-Status: [(slot_index, wh)], Wh > 0."""
    plan = status.get("plan") or []
    return [(i, float(v)) for i, v in enumerate(plan) if float(v) > 0.0]


def planned_windows_from_status(
    status: dict[str, Any], frame_start: Any, horizon_s: float
) -> list[tuple[float, float]]:
    """Translate midnight-anchored EOS plan slots to forecast-relative run blocks.

    The anchor is part of the EOS response. Without it a plan index cannot safely be
    interpreted as a future timestamp; refusing is safer than injecting yesterday.
    A forecast that begins inside an EOS slot owns that slot from its boundary.
    """
    if status.get("fallback_active"):
        raise ValueError("EOS placement belongs to the fallback planner")
    if status.get("time_frame_base") != 900:
        raise ValueError("SG3 placement requires EOS 15-minute slots")
    if status.get("plan_anchor") is None:
        raise ValueError("EOS placement has no plan_anchor; update EOS Connect")
    anchor = pd.Timestamp(status["plan_anchor"])
    start = pd.Timestamp(frame_start)
    if start.tzinfo is None:
        start = start.tz_localize("UTC")
    slot0 = int(np.floor((start - anchor).total_seconds() / 900.0))
    windows: list[tuple[float, float]] = []
    active_start: int | None = None
    previous: int | None = None
    for index, _ in planned_slots_from_status(status):
        if index < slot0:
            raise ValueError("EOS placement contains past slots")
        start_s = (index - slot0) * 900.0
        if start_s >= horizon_s:
            continue
        if active_start is None:
            active_start = index
        elif previous is not None and index != previous + 1:
            windows.append(((active_start - slot0) * 900.0, (previous - slot0 + 1) * 900.0))
            active_start = index
        previous = index
    if active_start is not None and previous is not None:
        windows.append(((active_start - slot0) * 900.0, (previous - slot0 + 1) * 900.0))
    return windows
