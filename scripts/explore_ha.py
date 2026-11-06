"""Listet Home-Assistant-Entities auf, die für die WP-Simulation relevant sein könnten.

Aufruf:  .venv/bin/python scripts/explore_ha.py
Secrets aus .env (HA_BASE_URL, HA_TOKEN). Gibt eine gruppierte Übersicht
und eine CSV-Mappingvorlage aus.
"""

from __future__ import annotations

import csv
import re
from pathlib import Path

from hp_twin.adapters.home_assistant import HaClient

# Heuristik: Schlüsselwörter → Rolle im Modell (nur Vorschlag, du korrigierst danach)
KEYWORDS: dict[str, list[str]] = {
    "außentemperatur": ["outdoor", "aussen", "outside", "ambient"],
    "vorlauf": ["vorlauf", "flow", "supply"],
    "rücklauf": ["ruecklauf", "rücklauf", "return"],
    "warmwasser/dhw": ["dhw", "warmwasser", "ww", "domestic"],
    "puffer/hk1": ["puffer", "buffer", "hk1", "heiz", "heating_circuit"],
    "hk2": ["hk2", "mix", "misch"],
    "verdichter/kompressor": ["compressor", "verdichter"],
    "leistung_elektrisch": ["power", "leistung", "watt", "_w", "p_el"],
    "energie/wärmemenge": ["energy", "wh", "kwh", "wärme", "heat_quantity", "waermemenge"],
    "modus/betrieb": ["mode", "betrieb", "operating", "status"],
    "sollwert": ["soll", "setpoint", "target"],
    "raumtemperatur": ["room", "raum", "temperatur", "climate"],
    "globalstrahlung": ["radiation", "ghi", "solar", "strahlung", "irradiance"],
}


def classify(entity_id: str, name: str) -> list[str]:
    text = f"{entity_id} {name}".lower()
    return [role for role, kws in KEYWORDS.items() if any(re.search(k, text) for k in kws)]


def main() -> None:
    client = HaClient.from_env()
    print(f"Verbunden: {client.ping()}  @ {client.base_url}\n")
    states = client.states()
    print(f"{len(states)} Entities gesamt.\n")

    rows: list[dict[str, str]] = []
    buckets: dict[str, list[str]] = {}
    for s in states:
        eid = s["entity_id"]
        attrs = s.get("attributes", {})
        name = attrs.get("friendly_name", "")
        unit = attrs.get("unit_of_measurement", "")
        domain = eid.split(".", 1)[0]
        roles = classify(eid, name)
        # Nur plausibel relevante Domains/Treffer behalten
        if domain in {"sensor", "binary_sensor", "climate", "number", "select", "weather"} and (
            roles or unit in {"°C", "W", "kW", "Wh", "kWh", "W/m²"}
        ):
            for r in roles or ["(unklassifiziert)"]:
                buckets.setdefault(r, []).append(f"{eid}  [{unit}]  {name}")
            rows.append(
                {
                    "role": ";".join(roles),
                    "entity_id": eid,
                    "unit": unit,
                    "friendly_name": name,
                    "state_example": str(s.get("state", ""))[:30],
                    "map_to": "",
                }
            )

    for role in sorted(buckets):
        print(f"### {role}")
        for line in sorted(set(buckets[role])):
            print(f"   {line}")
        print()

    out = Path("data/ha_entities.csv")
    out.parent.mkdir(exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(
            f, fieldnames=["role", "entity_id", "unit", "friendly_name", "state_example", "map_to"]
        )
        w.writeheader()
        w.writerows(sorted(rows, key=lambda r: r["role"]))
    print(f"\nMappingvorlage geschrieben: {out}  (Spalte 'map_to' mit CSV-Schemanamen füllen)")


if __name__ == "__main__":
    main()
