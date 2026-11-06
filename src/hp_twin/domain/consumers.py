"""Wärmesenken: Heizkreis (mit Thermostat-Drosselung) und Lastmodell-Schnittstelle.

Ein LoadModel entnimmt pro Zeitschritt Wärme aus Speicherknoten. Der Heizkreis koppelt den
Puffer an das Gebäudemodell: die entnommene Leistung hängt von (Puffertemperatur - Raumluft)
und der Thermostatstellung ab. Erreichen die Räume ihren Sollwert, drosseln die Thermostate
die Entnahme -> der Puffer entlädt langsamer -> die WP startet später.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Protocol

from hp_twin.domain.building import BuildingInputs, BuildingModel, BuildingState


class LoadModel(Protocol):
    def draw(
        self, t_s: float, dt_s: float, storage_temps: dict[str, float]
    ) -> dict[str, float]: ...


def _clip(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


@dataclass
class ConstantLoad:
    """Konstante Entnahme je Knoten (für Tests/Szenarien ohne Gebäudemodell)."""

    loads_w: dict[str, float]

    def draw(self, t_s: float, dt_s: float, storage_temps: dict[str, float]) -> dict[str, float]:
        return dict(self.loads_w)


@dataclass
class CompositeLoad:
    """Summiert mehrere Lastmodelle (Heizkreise + Warmwasser)."""

    models: list[LoadModel]

    def draw(self, t_s: float, dt_s: float, storage_temps: dict[str, float]) -> dict[str, float]:
        total: dict[str, float] = {}
        for m in self.models:
            for node, w in m.draw(t_s, dt_s, storage_temps).items():
                total[node] = total.get(node, 0.0) + w
        return total


@dataclass
class HeatingCircuitLoad:
    """Heizkreis: Fußboden-Emission vom Puffer ins Gebäude, mit Thermostat-Drosselung.

    q = emission_ua * max(T_puffer - T_raum, 0) * ventil, ventil in [0,1] (proportional unter Soll).
    Die entnommene Wärme wird dem Gebäudemodell als Heizleistung zugeführt (Zustandskopplung).
    """

    node: str
    building: BuildingModel
    state: BuildingState
    outdoor: Callable[[float], float]
    ghi: Callable[[float], float]
    room_setpoint: Callable[[float], float]
    emission_ua_w_per_k: float = 350.0
    throttle_band_k: float = 2.0
    internal_gain_w: float = 300.0
    last_q_w: float = field(default=0.0, init=False)

    def valve(self, t_s: float) -> float:
        sp = self.room_setpoint(t_s)
        # voll offen bei T_raum <= Soll - Band, geschlossen bei T_raum >= Soll
        return _clip((sp - self.state.t_air_c) / self.throttle_band_k, 0.0, 1.0)

    def draw(self, t_s: float, dt_s: float, storage_temps: dict[str, float]) -> dict[str, float]:
        t_buf = storage_temps[self.node]
        q = self.emission_ua_w_per_k * max(t_buf - self.state.t_air_c, 0.0) * self.valve(t_s)
        self.state = self.building.step(
            self.state,
            dt_s,
            BuildingInputs(
                t_outdoor_c=self.outdoor(t_s),
                ghi_w_m2=self.ghi(t_s),
                q_heat_w=q,
                internal_gain_w=self.internal_gain_w,
            ),
        )
        self.last_q_w = q
        return {self.node: q}


@dataclass
class MultiZoneHeatingLoad:
    """Heizkreis-Last mit zeitvariabler UA aus den aktiv beheizten Räumen (Heizlast + climate).

    1R1C-Gebäude: eine Gebäudemasse t_room, Verlust = UA_aktiv(t)*(t_room - T_außen), solare
    Gewinne, Wärmeeintrag vom Puffer über Thermostat-Ventil. UA_aktiv steigt/fällt, je nachdem
    welche Räume gerade beheizt werden (Morgen-Phase, tageweise andere Räume).
    """

    node: str
    c_building_j_per_k: float
    ua_active: Callable[[float], float]
    room_setpoint: Callable[[float], float]
    outdoor: Callable[[float], float]
    ghi: Callable[[float], float]
    t_room_c: float
    solar_w_per_wm2: float = 1.0
    emission_ua_w_per_k: float = 1000.0
    throttle_band_k: float = 1.5
    internal_gain_w: float = 0.0
    last_q_w: float = field(default=0.0, init=False)

    def valve(self, t_s: float) -> float:
        return _clip((self.room_setpoint(t_s) - self.t_room_c) / self.throttle_band_k, 0.0, 1.0)

    def draw(self, t_s: float, dt_s: float, storage_temps: dict[str, float]) -> dict[str, float]:
        t_buf = storage_temps[self.node]
        q = self.emission_ua_w_per_k * max(t_buf - self.t_room_c, 0.0) * self.valve(t_s)
        loss = self.ua_active(t_s) * (self.t_room_c - self.outdoor(t_s))
        solar = self.solar_w_per_wm2 * self.ghi(t_s)
        self.t_room_c += (q - loss + solar + self.internal_gain_w) * dt_s / self.c_building_j_per_k
        self.last_q_w = q
        return {self.node: q}
