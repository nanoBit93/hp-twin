"""Strukturierte Gebäude-Heizlast (aus Norm-Heizlastberechnung, DIN EN 12831).

Pro Raum die Norm-Heizlast -> UA_Raum = Heizlast / (Raumsoll - Norm-Außentemperatur).
Welche Räume aktuell aktiv heizen, kommt zur Laufzeit aus den climate-`hvac_action`-Daten;
die momentane Gesamt-UA ist die Summe der UA der aktiven Räume. Das löst die zeitlich/räumlich
variable Last auf (Morgenlast, tageweise unterschiedlich beheizte Räume).
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class _Model(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class RoomHeatLoad(_Model):
    name: str
    design_heat_load_w: float = Field(gt=0, description="Norm-Heizlast des Raums (W)")
    design_room_temp_c: float = Field(default=20.0, description="Norm-Innentemperatur")
    area_m2: float | None = Field(default=None, ge=0)
    circuit: str = Field(default="hk1", description="zugeordneter Heizkreis")
    climate_entity: str | None = Field(
        default=None, description="HA climate-Entity für hvac_action"
    )
    heated: bool = Field(default=True, description="wird dieser Raum grundsätzlich beheizt")

    def ua_w_per_k(self, design_outdoor_c: float) -> float:
        dt = self.design_room_temp_c - design_outdoor_c
        return self.design_heat_load_w / dt if dt > 0 else 0.0


class BuildingHeatLoad(_Model):
    design_outdoor_c: float = Field(default=-10.0, description="Norm-Außentemperatur")
    flow_design_c: float | None = Field(default=None, description="Auslegungs-Vorlauf (FBH)")
    stated_total_load_w: float | None = Field(
        default=None, description="in der PDF genannte Gesamt-Heizlast (ggf. inkl. Reserve)"
    )
    rooms: list[RoomHeatLoad] = Field(default_factory=list)

    @property
    def summed_room_load_w(self) -> float:
        return sum(r.design_heat_load_w for r in self.rooms)

    def total_ua_w_per_k(self, only: set[str] | None = None) -> float:
        """Gesamt-UA über alle (oder nur die genannten aktiven) Räume."""
        return sum(
            r.ua_w_per_k(self.design_outdoor_c)
            for r in self.rooms
            if only is None or r.name in only
        )

    def ua_by_room(self) -> dict[str, float]:
        return {r.name: r.ua_w_per_k(self.design_outdoor_c) for r in self.rooms}
