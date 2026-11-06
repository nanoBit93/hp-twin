"""M3-Szenario: Sonne heizt die Gebäudemasse -> Thermostate drosseln -> WP startet seltener/später.

Zwei identische Tage, einmal sonnig, einmal bedeckt. Zeigt den Kerneffekt: Raumtemperatur als
Zustandsgröße koppelt die Pufferentladung an das Wetter.
"""

from __future__ import annotations

import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from hp_twin.domain.building import RC2R2C, BuildingState  # noqa: E402
from hp_twin.domain.consumers import CompositeLoad, HeatingCircuitLoad  # noqa: E402
from hp_twin.domain.controller import Controller, Setpoints  # noqa: E402
from hp_twin.domain.dhw import DhwDraw, DhwDrawProfile  # noqa: E402
from hp_twin.domain.heat_source import HeatPumpAirWater  # noqa: E402
from hp_twin.domain.simulator import SimConfig, Simulator  # noqa: E402
from hp_twin.domain.storage import StratifiedStorage  # noqa: E402
from hp_twin.domain.topology import (  # noqa: E402
    ControllerConfig,
    Storage,
    StorageCoupling,
    StorageNode,
)


def ghi(t_s: float, peak: float) -> float:
    hod = (t_s / 3600.0) % 24
    return peak * max(0.0, math.sin((hod - 7) / 10 * math.pi)) if 7 <= hod <= 17 else 0.0


def run(peak_ghi: float):  # type: ignore[no-untyped-def]
    topo = Storage(
        id="kombi",
        t_ambient_c=18.0,
        nodes=[
            StorageNode(id="dhw", volume_l=300, loss_ua_w_per_k=1.2, t_init_c=48.0),
            StorageNode(id="buffer", volume_l=300, loss_ua_w_per_k=1.2, t_init_c=35.0),
        ],
        couplings=[
            StorageCoupling(upper="dhw", lower="buffer", k_w_per_k=15.0, buoyancy_mixing=True)
        ],
    )
    storage = StratifiedStorage.from_topology(topo)
    hc = HeatingCircuitLoad(
        node="buffer",
        building=RC2R2C(),
        state=BuildingState(20.5, 20.0),
        outdoor=lambda _t: 4.0,
        ghi=lambda t: ghi(t, peak_ghi),
        room_setpoint=lambda _t: 21.0,
        emission_ua_w_per_k=350.0,
    )
    load = CompositeLoad([hc, DhwDraw(node="dhw", profile=DhwDrawProfile.two_peak(150.0))])
    sim = Simulator(
        storage,
        HeatPumpAirWater(),
        Controller(config=ControllerConfig(), dhw_node="dhw", buffer_node="buffer"),
        SimConfig(timestep_s=60.0, horizon_s=24 * 3600),
    )
    # Raumtemperatur mitschreiben
    room = []
    orig_draw = hc.draw

    def draw_rec(t_s, dt_s, temps):  # type: ignore[no-untyped-def]
        r = orig_draw(t_s, dt_s, temps)
        room.append(hc.state.t_air_c)
        return r

    hc.draw = draw_rec  # type: ignore[method-assign]
    res = sim.run(
        storage.initial_state(topo),
        setpoints=lambda _t: Setpoints(dhw_c=40.5, buffer_c=31.5, buffer_overshoot_k=8),
        load=load,
        outdoor=lambda _t: 4.0,
    )
    return res, np.array(room)


def starts(modes: list[str]) -> int:
    c = [m in ("heat", "dhw") for m in modes]
    return sum(1 for i in range(1, len(c)) if c[i] and not c[i - 1])


def main() -> None:
    res_sun, room_sun = run(peak_ghi=550.0)
    res_cloud, room_cloud = run(peak_ghi=0.0)
    h = res_sun.time_s / 3600

    fig, ax = plt.subplots(3, 1, figsize=(13, 8), sharex=True)
    ax[0].plot(h, res_cloud.node_temps_c["buffer"], color="tab:blue", label="Puffer bedeckt")
    ax[0].plot(h, res_sun.node_temps_c["buffer"], color="tab:orange", label="Puffer sonnig")
    ax[0].axhline(31.5, ls="--", color="gray", alpha=0.5, label="Puffer-Soll")
    ax[0].set_ylabel("Puffer °C")
    ax[0].legend(ncol=3, fontsize=8)
    ax[0].grid(alpha=0.3)

    ax[1].plot(h, room_cloud, color="tab:blue", label="Raum bedeckt")
    ax[1].plot(h, room_sun, color="tab:orange", label="Raum sonnig")
    ax[1].axhline(21.0, ls="--", color="gray", alpha=0.5, label="Raumsoll")
    ax[1].set_ylabel("Raumluft °C")
    ax[1].legend(ncol=3, fontsize=8)
    ax[1].grid(alpha=0.3)

    ax[2].fill_between(h, 0, res_cloud.p_el_w, color="tab:blue", alpha=0.4, label="P_el bedeckt")
    ax[2].fill_between(h, 0, res_sun.p_el_w, color="tab:orange", alpha=0.4, label="P_el sonnig")
    ax[2].set_ylabel("P_el [W]")
    ax[2].set_xlabel("Stunden")
    ax[2].legend(ncol=2, fontsize=8)
    ax[2].grid(alpha=0.3)

    sc, ss = starts(res_cloud.mode), starts(res_sun.mode)
    fig.suptitle(
        f"M3: Sonne verschiebt WP-Takte | bedeckt {sc} Takte / {res_cloud.energy_kwh:.1f} kWh"
        f"  vs  sonnig {ss} Takte / {res_sun.energy_kwh:.1f} kWh"
    )
    fig.tight_layout()
    out = Path("data/ha_export/scenario_m3_solar.png")
    fig.savefig(out, dpi=110)
    print(f"Bedeckt: {sc} Takte, {res_cloud.energy_kwh:.1f} kWh, Raum bis {room_cloud.max():.1f}C")
    print(f"Sonnig:  {ss} Takte, {res_sun.energy_kwh:.1f} kWh, Raum bis {room_sun.max():.1f}C")
    print(f"Plot -> {out}")


if __name__ == "__main__":
    main()
