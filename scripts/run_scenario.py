"""M2-Szenario: Pufferlade-Zyklus simulieren und plotten (qualitativer Abgleich zur Realität)."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from hp_twin.domain.consumers import ConstantLoad  # noqa: E402
from hp_twin.domain.controller import Controller, Setpoints  # noqa: E402
from hp_twin.domain.heat_source import HeatPumpAirWater  # noqa: E402
from hp_twin.domain.simulator import SimConfig, Simulator  # noqa: E402
from hp_twin.domain.storage import StratifiedStorage  # noqa: E402
from hp_twin.domain.topology import (  # noqa: E402
    ControllerConfig,
    Storage,
    StorageCoupling,
    StorageNode,
)


def main() -> None:
    topo = Storage(
        id="kombi",
        t_ambient_c=15.0,
        nodes=[
            StorageNode(id="dhw", volume_l=300, loss_ua_w_per_k=1.2, t_init_c=48.0),
            StorageNode(id="buffer", volume_l=300, loss_ua_w_per_k=1.2, t_init_c=35.0),
        ],
        couplings=[
            StorageCoupling(upper="dhw", lower="buffer", k_w_per_k=15.0, buoyancy_mixing=True)
        ],
    )
    storage = StratifiedStorage.from_topology(topo)
    source = HeatPumpAirWater(p_el_ref_w=2000.0, standby_w=80.0)
    controller = Controller(
        config=ControllerConfig(min_off_s=1800, min_on_s=300),
        dhw_node="dhw",
        buffer_node="buffer",
        buffer_charge_offset_k=10.0,
    )
    sim = Simulator(storage, source, controller, SimConfig(timestep_s=60.0, horizon_s=24 * 3600))

    sp = Setpoints(dhw_c=40.5, buffer_c=31.5, buffer_hysteresis_k=5, buffer_overshoot_k=8)
    res = sim.run(
        storage.initial_state(topo),
        setpoints=lambda _t: sp,
        load=ConstantLoad({"buffer": 2300.0}),
        outdoor=lambda _t: 5.0,
    )

    h = res.time_s / 3600
    fig, ax = plt.subplots(2, 1, figsize=(13, 6), sharex=True)
    ax[0].plot(h, res.node_temps_c["buffer"], color="tab:blue", label="Puffer/HK1")
    ax[0].plot(h, res.node_temps_c["dhw"], color="tab:red", label="DHW")
    ax[0].axhline(sp.buffer_c, ls="--", color="tab:blue", alpha=0.4, label="Puffer-Soll")
    ax[0].set_ylabel("°C")
    ax[0].legend(ncol=3, fontsize=8)
    ax[0].grid(alpha=0.3)
    ax[1].fill_between(h, 0, res.p_el_w, color="tab:orange", alpha=0.6)
    ax[1].set_ylabel("P_el [W]")
    ax[1].set_xlabel("Stunden")
    ax[1].grid(alpha=0.3)
    starts = sum(
        1
        for i in range(1, len(res.mode))
        if res.mode[i] in ("heat", "dhw") and res.mode[i - 1] not in ("heat", "dhw")
    )
    fig.suptitle(f"M2-Szenario (simuliert): {starts} Takte/24h, {res.energy_kwh:.1f} kWh")
    fig.tight_layout()
    out = Path("data/ha_export/scenario_m2.png")
    fig.savefig(out, dpi=110)
    print(
        f"Takte/24h: {starts} | Energie: {res.energy_kwh:.1f} kWh | "
        f"Puffer {res.node_temps_c['buffer'].min():.1f}..{res.node_temps_c['buffer'].max():.1f} °C"
    )
    print(f"Plot -> {out}")


if __name__ == "__main__":
    main()
