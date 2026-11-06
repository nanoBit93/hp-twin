"""Gebäudemodell (pluggable). Strategie A: Grey-Box 2R2C (Raumluft + Gebäudemasse).

Die Raumtemperatur ist eine ZUSTANDSGRÖSSE, keine Störgröße: thermische Masse speichert
Sonnen-/Heizwärme, hält die Räume warm und lässt den Puffer langsamer entladen (Thermostat-
Drosselung). Damit springt die WP nach einem sonnigen Tag später an.

2R2C-Knoten:
  T_air  (C_air):  Raumluft + schnelle Möblierung
  T_mass (C_mass): träge Gebäudemasse (Estrich, Wände)
Wärmeströme:
  Luft  <-> Außen  (ua_air_out,  schnell: Fenster/Lüftung)
  Luft  <-> Masse  (ua_air_mass)
  Masse <-> Außen  (ua_mass_out, langsam: Hülle)
Eingänge: Heizleistung, interne Gewinne, solare Gewinne (GHI * Aperturfläche), Außentemperatur.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


@dataclass
class BuildingState:
    t_air_c: float
    t_mass_c: float

    def copy(self) -> BuildingState:
        return BuildingState(self.t_air_c, self.t_mass_c)


@dataclass(frozen=True)
class BuildingInputs:
    t_outdoor_c: float
    ghi_w_m2: float
    q_heat_w: float  # von den Heizkreisen eingebrachte Leistung
    internal_gain_w: float = 0.0


class BuildingModel(Protocol):
    def step(self, state: BuildingState, dt_s: float, inp: BuildingInputs) -> BuildingState: ...


@dataclass(frozen=True)
class RC2R2C:
    """Aggregiertes 2R2C-Hausmodell (gewichtete mittlere Raumtemperatur)."""

    c_air_j_per_k: float = 5.0e6
    c_mass_j_per_k: float = 6.0e7
    ua_air_out_w_per_k: float = 70.0
    ua_air_mass_w_per_k: float = 300.0
    ua_mass_out_w_per_k: float = 40.0
    solar_aperture_m2: float = 3.0  # effektive Fläche * g-Wert
    solar_to_mass_frac: float = 0.5  # Anteil der Solarstrahlung, der in die Masse geht

    def step(self, state: BuildingState, dt_s: float, inp: BuildingInputs) -> BuildingState:
        t_air, t_mass = state.t_air_c, state.t_mass_c
        solar = inp.ghi_w_m2 * self.solar_aperture_m2
        solar_mass = solar * self.solar_to_mass_frac
        solar_air = solar - solar_mass

        q_air = (
            inp.q_heat_w
            + inp.internal_gain_w
            + solar_air
            + self.ua_air_mass_w_per_k * (t_mass - t_air)
            + self.ua_air_out_w_per_k * (inp.t_outdoor_c - t_air)
        )
        q_mass = (
            solar_mass
            + self.ua_air_mass_w_per_k * (t_air - t_mass)
            + self.ua_mass_out_w_per_k * (inp.t_outdoor_c - t_mass)
        )
        return BuildingState(
            t_air_c=t_air + q_air * dt_s / self.c_air_j_per_k,
            t_mass_c=t_mass + q_mass * dt_s / self.c_mass_j_per_k,
        )

    def steady_loss_w(self, t_air_c: float, t_outdoor_c: float) -> float:
        """Stationärer Wärmeverlust (nur zur Orientierung/Parametrierung)."""
        ua_total = self.ua_air_out_w_per_k + 1.0 / (
            1.0 / self.ua_air_mass_w_per_k + 1.0 / self.ua_mass_out_w_per_k
        )
        return ua_total * (t_air_c - t_outdoor_c)
