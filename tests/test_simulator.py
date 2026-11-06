"""Szenario-Test: reproduziert qualitativ den Pufferlade-Zyklus der realen Anlage.

Plausibilitätsreferenz (nicht hartkodiert): HK1/Puffer pendelt grob zwischen ~27 und ~42 °C,
mehrere Takte pro 12 h, P_el-Pulse im Betrieb, Standby dazwischen, min_off eingehalten.
"""

from __future__ import annotations

import numpy as np

from hp_twin.domain.consumers import ConstantLoad
from hp_twin.domain.controller import Controller, Setpoints
from hp_twin.domain.heat_source import HeatPumpAirWater
from hp_twin.domain.simulator import SimConfig, Simulator
from hp_twin.domain.storage import StratifiedStorage
from hp_twin.domain.topology import ControllerConfig, Storage, StorageCoupling, StorageNode


def _storage() -> tuple[StratifiedStorage, Storage]:
    s = Storage(
        id="kombi",
        t_ambient_c=15.0,
        nodes=[
            StorageNode(id="dhw", volume_l=300, loss_ua_w_per_k=1.0, t_init_c=55.0),
            StorageNode(id="buffer", volume_l=300, loss_ua_w_per_k=1.0, t_init_c=35.0),
        ],
        couplings=[
            StorageCoupling(upper="dhw", lower="buffer", k_w_per_k=0.0, buoyancy_mixing=True)
        ],
    )
    return StratifiedStorage.from_topology(s), s


def _count_starts(modes: list[str]) -> int:
    charging = [m in ("heat", "dhw") for m in modes]
    return sum(1 for i in range(1, len(charging)) if charging[i] and not charging[i - 1])


def test_buffer_sawtooth_cycle() -> None:
    storage, topo = _storage()
    source = HeatPumpAirWater(p_el_ref_w=2000.0, standby_w=80.0)
    controller = Controller(
        config=ControllerConfig(min_off_s=1800, min_on_s=300),
        dhw_node="dhw",
        buffer_node="buffer",
        buffer_charge_offset_k=10.0,
    )
    sim = Simulator(storage, source, controller, SimConfig(timestep_s=60.0, horizon_s=12 * 3600))

    # Konstant: 5 °C außen, Puffer-Soll 31.5, DHW-Soll 40.5 (DHW bleibt warm -> kein DHW-Takt)
    sp = Setpoints(
        dhw_c=40.5,
        buffer_c=31.5,
        dhw_hysteresis_k=5,
        buffer_hysteresis_k=5,
        dhw_overshoot_k=2,
        buffer_overshoot_k=8,
    )
    res = sim.run(
        storage.initial_state(topo),
        setpoints=lambda _t: sp,
        load=ConstantLoad({"buffer": 2300.0}),  # stetige Heizlast auf den Puffer
        outdoor=lambda _t: 5.0,
    )

    buf = res.node_temps_c["buffer"]
    starts = _count_starts(res.mode)

    # (1) mehrere Takte in 12 h (real ~2-3)
    assert 2 <= starts <= 10, f"unerwartete Taktzahl: {starts}"
    # (2) Sägezahn im plausiblen Band ~27..42 °C (Toleranz)
    assert 24.0 <= buf.min() <= 31.0, buf.min()
    assert 38.0 <= buf.max() <= 45.0, buf.max()
    # (3) Leistung: Standby zwischen den Pulsen, Pulse im Betrieb
    assert np.isclose(res.p_el_w.min(), 80.0)
    assert res.p_el_w.max() > 2000.0
    # (4) DHW ohne eigenen Takt (bleibt nahe Startwert, nur Verluste)
    assert "dhw" not in res.mode
    # (5) min_off eingehalten: kein Start näher als 1800 s am vorigen Stopp
    charging = [m in ("heat", "dhw") for m in res.mode]
    last_stop = None
    for i in range(1, len(charging)):
        if charging[i - 1] and not charging[i]:
            last_stop = res.time_s[i]
        if charging[i] and not charging[i - 1] and last_stop is not None:
            assert res.time_s[i] - last_stop >= 1800.0 - 1e-6


def test_energy_accounting_positive() -> None:
    storage, topo = _storage()
    sim = Simulator(
        storage,
        HeatPumpAirWater(),
        Controller(config=ControllerConfig(), dhw_node="dhw", buffer_node="buffer"),
        SimConfig(timestep_s=60.0, horizon_s=6 * 3600),
    )
    res = sim.run(
        storage.initial_state(topo),
        setpoints=lambda _t: Setpoints(dhw_c=40.5, buffer_c=31.5),
        load=ConstantLoad({"buffer": 2000.0}),
        outdoor=lambda _t: 5.0,
    )
    assert res.energy_kwh > 0
    assert len(res.time_s) == 360
