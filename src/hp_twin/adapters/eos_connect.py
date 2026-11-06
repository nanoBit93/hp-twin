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
from typing import Any, Protocol

import numpy as np

from hp_twin.domain.montecarlo import SlotForecast


@dataclass(frozen=True)
class EosLoadProfile:
    values_w: list[float]
    unit: str = "W"
    start: str = "now"
    ttl_minutes: int | None = None

    def to_payload(self) -> dict[str, Any]:
        payload: dict[str, Any] = {"values": self.values_w, "unit": self.unit, "start": self.start}
        if self.ttl_minutes is not None:
            payload["ttl_minutes"] = self.ttl_minutes
        return payload


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


class EnergyBudgetAdapter(Protocol):
    """Spätere SG3-Schnittstelle: External energy budget {"total_wh":..., "deadline_hours":...}."""

    def to_payload(self) -> dict[str, Any]: ...
