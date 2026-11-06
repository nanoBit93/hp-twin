"""Wärmeerzeuger mit Kennfeld. Phase 1: Luft/Wasser-WP (parametrisch) + Heizstab-Stub.

Das Kennfeld (P_el, Q_th über Außen-/Vorlauftemperatur) wird in M4 aus Messdaten gefittet.
Hier ein physikalisch plausibles parametrisches Modell mit manuell gesetzten Defaults (M2).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

KELVIN = 273.15


@dataclass(frozen=True)
class HeatOutput:
    """Momentane Leistungen des Erzeugers."""

    p_el_w: float  # elektrische Aufnahme (inkl. Standby)
    q_th_w: float  # abgegebene Heizleistung


class HeatSource(Protocol):
    def output(self, t_outdoor_c: float, t_flow_c: float, active: bool) -> HeatOutput: ...


@dataclass(frozen=True)
class HeatPumpAirWater:
    """Parametrisches Luft/Wasser-WP-Modell (Inverter), innerhalb eines Takts temperaturabhängig.

    Zwei gekoppelte Effekte über die momentane Vorlauf-/Kondensatortemperatur t_flow:
    - COP sinkt mit steigendem Hub (Carnot-Gütegrad):
      cop = eta_carnot * T_cond / (T_cond - T_evap), begrenzt auf [cop_min, cop_max].
    - elektrische Aufnahme STEIGT mit t_flow (Inverter moduliert hoch) und sinkt mit t_outdoor:
      p_comp = p_el_ref + dp_dt_flow*(t_flow - t_flow_ref) + dp_dt_out*(t_outdoor - t_out_ref).
    Dadurch fällt Q_th = p_comp*cop zum Taktende hin realistisch ab (keine überschätzte Wärme).
    t_flow wird vom Simulator aus der aktuellen Knotentemperatur gebildet, steigt also im Takt mit.
    """

    p_el_ref_w: float = 2000.0  # Verdichter-Aufnahme am Referenzpunkt
    standby_w: float = 80.0
    # Leistungs-Modulation des Inverters
    dp_dt_flow_w_per_k: float = 45.0  # +W je K Vorlauf
    dp_dt_out_w_per_k: float = -25.0  # -W je K Außentemperatur (wärmer -> weniger)
    t_flow_ref_c: float = 35.0
    t_out_ref_c: float = 5.0
    p_comp_min_w: float = 800.0
    p_comp_max_w: float = 5500.0
    # COP-Modell
    eta_carnot: float = 0.45
    dt_cond_k: float = 5.0
    dt_evap_k: float = 6.0
    cop_min: float = 1.5
    cop_max: float = 6.0
    defrost_derate: float = 1.0  # <1 während Abtauung

    def cop(self, t_outdoor_c: float, t_flow_c: float) -> float:
        t_cond = t_flow_c + self.dt_cond_k + KELVIN
        t_evap = t_outdoor_c - self.dt_evap_k + KELVIN
        lift = max(t_cond - t_evap, 1.0)
        raw = self.eta_carnot * t_cond / lift
        return float(min(max(raw, self.cop_min), self.cop_max))

    def compressor_power_w(self, t_outdoor_c: float, t_flow_c: float) -> float:
        p = (
            self.p_el_ref_w
            + self.dp_dt_flow_w_per_k * (t_flow_c - self.t_flow_ref_c)
            + self.dp_dt_out_w_per_k * (t_outdoor_c - self.t_out_ref_c)
        )
        return float(min(max(p, self.p_comp_min_w), self.p_comp_max_w))

    def output(self, t_outdoor_c: float, t_flow_c: float, active: bool) -> HeatOutput:
        if not active:
            return HeatOutput(p_el_w=self.standby_w, q_th_w=0.0)
        p_comp = self.compressor_power_w(t_outdoor_c, t_flow_c)
        cop = self.cop(t_outdoor_c, t_flow_c)
        q_th = p_comp * cop * self.defrost_derate
        return HeatOutput(p_el_w=p_comp + self.standby_w, q_th_w=q_th)


@dataclass(frozen=True)
class ElectricHeater:
    """Heizstab (Phase 1: nur Notbetrieb). COP = 1."""

    power_w: float = 6000.0
    standby_w: float = 0.0

    def output(self, t_outdoor_c: float, t_flow_c: float, active: bool) -> HeatOutput:
        if not active:
            return HeatOutput(p_el_w=self.standby_w, q_th_w=0.0)
        return HeatOutput(p_el_w=self.power_w, q_th_w=self.power_w)
