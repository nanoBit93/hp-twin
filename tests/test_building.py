"""Gebäudemodell 2R2C + Heizkreis-Thermostat + DHW-Zapfprofil."""

from __future__ import annotations

import numpy as np

from hp_twin.domain.building import RC2R2C, BuildingInputs, BuildingState
from hp_twin.domain.consumers import ConstantLoad, HeatingCircuitLoad
from hp_twin.domain.dhw import DhwDraw, DhwDrawProfile, estimate_profile_from_history


def test_building_energy_balance_no_losses() -> None:
    """Ohne Verluste/Solar: zugeführte Heizenergie = Zuwachs der inneren Energie."""
    b = RC2R2C(ua_air_out_w_per_k=0, ua_mass_out_w_per_k=0, solar_aperture_m2=0)
    s = BuildingState(t_air_c=20.0, t_mass_c=20.0)
    u0 = b.c_air_j_per_k * s.t_air_c + b.c_mass_j_per_k * s.t_mass_c
    dt, steps, q = 60.0, 100, 1000.0
    for _ in range(steps):
        s = b.step(s, dt, BuildingInputs(t_outdoor_c=5, ghi_w_m2=0, q_heat_w=q, internal_gain_w=0))
    u1 = b.c_air_j_per_k * s.t_air_c + b.c_mass_j_per_k * s.t_mass_c
    assert abs((u1 - u0) - q * dt * steps) < 1.0


def test_building_cools_toward_outdoor() -> None:
    b = RC2R2C()
    s = BuildingState(t_air_c=22.0, t_mass_c=22.0)
    for _ in range(600):
        s = b.step(s, 60.0, BuildingInputs(t_outdoor_c=0.0, ghi_w_m2=0, q_heat_w=0))
    assert s.t_air_c < 22.0 and s.t_air_c > 0.0


def test_thermostat_throttles_at_setpoint() -> None:
    b = RC2R2C()
    hot = HeatingCircuitLoad(
        node="buffer",
        building=b,
        state=BuildingState(21.5, 21.0),
        outdoor=lambda _t: 5.0,
        ghi=lambda _t: 0.0,
        room_setpoint=lambda _t: 21.0,
        throttle_band_k=2.0,
    )
    cold = HeatingCircuitLoad(
        node="buffer",
        building=b,
        state=BuildingState(18.0, 18.0),
        outdoor=lambda _t: 5.0,
        ghi=lambda _t: 0.0,
        room_setpoint=lambda _t: 21.0,
        throttle_band_k=2.0,
    )
    temps = {"buffer": 35.0}
    q_hot = hot.draw(0.0, 60.0, temps)["buffer"]
    q_cold = cold.draw(0.0, 60.0, temps)["buffer"]
    assert q_hot == 0.0  # Raum über Soll -> Ventil zu
    assert q_cold > 0.0  # Raum unter Soll -> Entnahme


def test_sunny_day_reduces_heat_draw() -> None:
    """Zentraler Effekt: Sonne -> Masse/Räume wärmer -> Thermostat drosselt -> weniger Entnahme."""
    b = RC2R2C()

    def run(ghi: float) -> float:
        hc = HeatingCircuitLoad(
            node="buffer",
            building=b,
            state=BuildingState(20.0, 20.0),
            outdoor=lambda _t: 5.0,
            ghi=lambda _t: ghi,
            room_setpoint=lambda _t: 21.0,
        )
        total = 0.0
        for k in range(600):  # 10 h
            total += hc.draw(k * 60.0, 60.0, {"buffer": 35.0})["buffer"]
        return total

    assert run(ghi=400.0) < run(ghi=0.0)  # mit Sonne weniger Pufferentnahme


def test_dhw_draw_cools_node() -> None:
    profile = DhwDrawProfile.two_peak(liters_per_day=150.0, start_hour=7.0)
    d = DhwDraw(node="dhw", profile=profile)
    q = d.draw(0.0, 60.0, {"dhw": 48.0})["dhw"]
    assert q > 0.0
    # kein Wasser -> keine Entnahme (flat 0)
    d0 = DhwDraw(node="dhw", profile=DhwDrawProfile.flat(0.0))
    assert d0.draw(0.0, 60.0, {"dhw": 48.0})["dhw"] == 0.0


def test_dhw_profile_total() -> None:
    p = DhwDrawProfile.two_peak(liters_per_day=150.0)
    assert abs(p.liters_by_hour.sum() - 150.0) < 1e-6


def test_estimate_profile_from_history() -> None:
    # synthetisch: Abfall um 3 K um 7 Uhr an zwei Tagen
    ts = np.arange("2026-01-10T00:00", "2026-01-12T00:00", 3600, dtype="datetime64[s]")
    t_dhw = np.full(len(ts), 48.0)
    charging = np.zeros(len(ts), dtype=bool)
    for day in (0, 1):
        idx = day * 24 + 7
        t_dhw[idx] = 45.0  # -3 K um 7 Uhr
    prof = estimate_profile_from_history(ts, t_dhw, charging, volume_l=300.0)
    assert prof.liters_by_hour[7] > prof.liters_by_hour[3]


def test_constant_load_passthrough() -> None:
    assert ConstantLoad({"buffer": 2000.0}).draw(0.0, 60.0, {"buffer": 35.0}) == {"buffer": 2000.0}
