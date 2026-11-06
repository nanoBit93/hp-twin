"""Topologie-Schema der Anlage (pydantic v2). Die ganze Anlage wird per YAML beschrieben."""

from __future__ import annotations

from enum import StrEnum
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class _Model(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class DhwMode(StrEnum):
    """WPM-Einstellung „Warmwasserbetrieb“."""

    PRIORITY = "priority"  # Vorrangbetrieb: Heizkreise während WW-Ladung gesperrt
    PARALLEL = "parallel"  # Parallelbetrieb: Heizkreise ziehen weiter aus dem Puffer
    PARTIAL = "partial"  # Teilvorrang: eingeschränkter Entzug


# ---------------------------------------------------------------- Speicher
class StorageNode(_Model):
    id: str
    volume_l: float = Field(gt=0)
    loss_ua_w_per_k: float = Field(ge=0, default=1.0, description="Verlust an Umgebung")
    sensor_position: float = Field(
        ge=0, le=1, default=0.5, description="Fühlerhöhe im Knoten, 0 = oben, 1 = unten"
    )
    t_init_c: float = 30.0


class StorageCoupling(_Model):
    upper: str
    lower: str
    k_w_per_k: float = Field(ge=0, default=20.0, description="Kopplung aufwärts (unten wärmer)")
    k_down_w_per_k: float | None = Field(
        default=None,
        ge=0,
        description="Kopplung abwärts (oben wärmer); Schichtung dämpft sie. None = symmetrisch",
    )
    buoyancy_mixing: bool = True


class Storage(_Model):
    id: str
    nodes: Annotated[list[StorageNode], Field(min_length=1)]
    couplings: list[StorageCoupling] = []  # Reihenfolge nicht relevant, upper/lower explizit
    t_ambient_c: float = 15.0


# ---------------------------------------------------------------- Erzeuger
class HeatPumpSource(_Model):
    type: Literal["heat_pump_air_water"] = "heat_pump_air_water"
    id: str
    model: str = "WPL-A 25"
    map_file: str | None = Field(
        default=None, description="CSV-Kennfeld (T_aus, T_VL → P_el, Q_th)"
    )
    single_compressor: bool = True


class ElectricHeaterSource(_Model):
    type: Literal["electric_heater"] = "electric_heater"
    id: str
    power_w: float = Field(gt=0)
    enabled: bool = False
    emergency_only: bool = True


HeatSource = Annotated[HeatPumpSource | ElectricHeaterSource, Field(discriminator="type")]


# ---------------------------------------------------------------- Consumer
class HeatingCurve(_Model):
    slope: float = Field(gt=0, default=0.3)
    base_c: float = Field(default=21.0, description="Fußpunkt/Raumsoll bei 20 °C Außen")


class HeatingCircuit(_Model):
    kind: Literal["heating_circuit"] = "heating_circuit"
    id: str
    node: str = Field(description="Speicherknoten, aus dem entnommen wird")
    mixed: bool = False
    curve: HeatingCurve = HeatingCurve()
    hysteresis_k: float = Field(ge=0, default=5.0)
    leads_charging: bool = Field(
        default=False, description="Fühler ist Führungsgröße für Pufferladen (HK1)"
    )
    sensor_node: str | None = None


class DhwCircuit(_Model):
    kind: Literal["dhw"] = "dhw"
    id: str
    node: str
    setpoint_c: float = 40.5
    hysteresis_k: float = Field(ge=0, default=5.0)
    priority: bool = True
    target_stop_c: float | None = None


Consumer = Annotated[HeatingCircuit | DhwCircuit, Field(discriminator="kind")]


# ---------------------------------------------------------------- Regler
class ControllerConfig(_Model):
    min_off_s: float = Field(ge=0, default=1800)
    mode_switch_pause_s: float | None = Field(default=None, ge=0)
    min_on_s: float = Field(ge=0, default=300)
    dhw_mode: DhwMode = DhwMode.PARALLEL
    partial_priority_factor: float = Field(ge=0, le=1, default=0.5)

    @property
    def effective_mode_switch_pause_s(self) -> float:
        return self.min_off_s if self.mode_switch_pause_s is None else self.mode_switch_pause_s


class SimulationConfig(_Model):
    timestep_s: float = Field(gt=0, default=60)
    horizon_h: float = Field(gt=0, default=48)


class ReturnInjection(_Model):
    """Rücklauf-Einschichtung beim Laden einer Zone (Modell für den HK1-Abfall bei WW-Ladung)."""

    charging_circuit: str
    node: str
    fraction: float = Field(ge=0, le=1, default=0.0)


class Topology(_Model):
    name: str = "installation"
    simulation: SimulationConfig = SimulationConfig()
    storage: Storage
    sources: Annotated[list[HeatSource], Field(min_length=1)]
    consumers: Annotated[list[Consumer], Field(min_length=1)]
    controller: ControllerConfig = ControllerConfig()
    return_injections: list[ReturnInjection] = []

    @model_validator(mode="after")
    def _check_references(self) -> Topology:
        node_ids = [n.id for n in self.storage.nodes]
        if len(set(node_ids)) != len(node_ids):
            raise ValueError("Speicherknoten-IDs müssen eindeutig sein")
        known = set(node_ids)
        for c in self.storage.couplings:
            for ref in (c.upper, c.lower):
                if ref not in known:
                    raise ValueError(f"Kopplung referenziert unbekannten Knoten '{ref}'")
        ids = [c.id for c in self.consumers] + [s.id for s in self.sources]
        if len(set(ids)) != len(ids):
            raise ValueError("Consumer-/Source-IDs müssen eindeutig sein")
        circuits = {c.id for c in self.consumers}
        for consumer in self.consumers:
            if consumer.node not in known:
                raise ValueError(f"Consumer '{consumer.id}': unbekannter Knoten '{consumer.node}'")
            if isinstance(consumer, HeatingCircuit) and consumer.sensor_node not in (None, *known):
                raise ValueError(f"Consumer '{consumer.id}': unbekannter sensor_node")
        for r in self.return_injections:
            if r.node not in known or r.charging_circuit not in circuits:
                raise ValueError("return_injection referenziert Unbekanntes")
        n_dhw = sum(isinstance(c, DhwCircuit) and c.priority for c in self.consumers)
        if n_dhw > 1:
            raise ValueError("Höchstens ein Warmwasser-Kreis mit Vorrang unterstützt")
        return self
