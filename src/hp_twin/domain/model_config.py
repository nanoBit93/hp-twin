"""Zentrale, konfigurierbare Modell-/Simulations-/Monte-Carlo-Parameter.

Alle bisher im Code hartcodierten Werte liegen hier mit Defaults = bisheriges Verhalten.
Per `config/model.yaml` überschreibbar; physikalische Konstanten (z. B. Wasser rho*cp) bleiben
im Code, da sie keine Tuning-Parameter sind.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class _Model(BaseModel):
    model_config = ConfigDict(extra="forbid")


class SimParams(_Model):
    timestep_s: float = Field(default=60.0, gt=0)
    horizon_h: float = Field(default=48.0, gt=0)
    condenser_approach_k: float = Field(
        default=7.0, description="Vorlauf = Knotentemperatur + dieser Wert (gedeckelt am Soll)"
    )


class LoadParams(_Model):
    c_building_j_per_k: float = Field(default=1.5e7, gt=0, description="wirksame Gebäudemasse")
    emission_ua_w_per_k: float = Field(
        default=1000.0, gt=0, description="Fußboden-Emission (Multi-Zone)"
    )
    fallback_emission_ua_w_per_k: float = Field(
        default=600.0, gt=0, description="Emission (1R2C-Fallback)"
    )
    throttle_band_k: float = Field(default=2.0, gt=0, description="Thermostat-Proportionalband")
    frost_threshold_c: float = Field(
        default=6.0, description="Raum-Soll > Wert => Raum aktiv beheizt"
    )
    internal_gain_w: float = Field(default=0.0, ge=0)
    # Aufteilung des effektiven Gebäude-UA auf die 2R2C-Pfade (Fallback ohne Heizlast/climate)
    ua_air_mass_w_per_k: float = Field(default=300.0, gt=0)
    ua_air_out_fraction: float = Field(default=0.5, gt=0, lt=1)


class CalibrationParams(_Model):
    standby_fallback_w: float = Field(default=80.0, ge=0)
    t_flow_ref_c: float = Field(default=35.0)
    t_out_ref_c: float = Field(default=5.0)
    discharge_window_s: float = Field(default=900.0, gt=0)
    op_min_flow_lpm: float = Field(
        default=2.0, ge=0, description="Filter: eingeschwungener Betrieb"
    )
    op_min_p_el_w: float = Field(default=500.0, ge=0)
    min_fit_days: float = Field(
        default=5.0, gt=0, description="weniger Historie: gespeicherte Kalibrierung nutzen"
    )


class MonteCarloParams(_Model):
    n_runs: int = Field(default=100, ge=1)
    seed: int = 0
    demand_sigma: float = Field(default=0.15, ge=0, description="rel. Streuung Entladerate")
    solar_sigma: float = Field(default=0.3, ge=0)
    c_building_sigma: float = Field(default=0.3, ge=0, description="lognormal auf Gebäudemasse")


class ControllerParams(_Model):
    dhw_hysteresis_k: float = Field(default=5.0, ge=0)
    dhw_overshoot_k: float = Field(default=2.0, ge=0)
    # None => aus Kalibrierung (estimate_thresholds); Zahl überschreibt
    buffer_hysteresis_k: float | None = None
    buffer_overshoot_k: float | None = None


class ForecastParams(_Model):
    slot_s: float = Field(default=900.0, gt=0, description="Slot-Breite (900 = 15 min)")
    running_threshold_w: float = Field(default=200.0, ge=0, description="P_el > Wert = WP läuft")
    dhw_setpoint_cap_c: float | None = Field(
        default=45.0,
        description="DHW-Soll deckeln (neue Konfig, passive DHW); None = Messwert nutzen",
    )
    eos_ttl_minutes: int | None = None


class BacktestParams(_Model):
    horizon_h: float = Field(default=24.0, gt=0)
    step_min: float = Field(default=180.0, gt=0)
    energy_window_s: float = Field(default=6 * 3600.0, gt=0)


class ModelConfig(_Model):
    sim: SimParams = SimParams()
    load: LoadParams = LoadParams()
    controller: ControllerParams = ControllerParams()
    calibration: CalibrationParams = CalibrationParams()
    montecarlo: MonteCarloParams = MonteCarloParams()
    forecast: ForecastParams = ForecastParams()
    backtest: BacktestParams = BacktestParams()

    @classmethod
    def from_path(cls, path: str | Path | None) -> ModelConfig:
        """Lädt config/model.yaml; fehlt die Datei, gelten die Defaults."""
        if path is None or not Path(path).exists():
            return cls()
        import yaml

        data: Any = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
        return cls.model_validate(data)
