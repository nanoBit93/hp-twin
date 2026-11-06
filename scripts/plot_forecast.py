"""Plottet eine 48-h-Lastprognose: Erwartungswert + P10/P90-Band + Laufwahrscheinlichkeit."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from hp_twin.adapters.forecast import MonteCarloConfig, run_forecast  # noqa: E402
from hp_twin.cli import _calibrate_report  # noqa: E402


def main() -> None:
    topo, df, calib = _calibrate_report(
        "config/example_topology.yaml", "data/ha_export/measurements.csv"
    )
    fc = run_forecast(
        df, topo, calib, 0, 48.0, mc=MonteCarloConfig(n_runs=100, dhw_setpoint_cap_c=45.0)
    )
    s = fc.slots
    h = s.slot_start_s / 3600
    fig, ax = plt.subplots(2, 1, figsize=(13, 6), sharex=True)
    ax[0].fill_between(
        h, s.p_el_p10_w, s.p_el_p90_w, alpha=0.3, color="tab:orange", label="P10-P90"
    )
    ax[0].plot(h, s.p_el_mean_w, color="tab:red", lw=1.2, label="Erwartungswert")
    ax[0].set_ylabel("P_el [W]")
    ax[0].legend(fontsize=8)
    ax[0].grid(alpha=0.3)
    ax[0].set_title(
        f"48-h-Lastprognose (100 MC-Läufe) | {len(fc.events)} Events | "
        f"{s.energy_wh.sum() / 1000:.1f} kWh erwartet"
    )
    ax[1].fill_between(h, 0, s.run_probability, color="tab:blue", alpha=0.5)
    ax[1].set_ylabel("Laufwahrsch.")
    ax[1].set_xlabel("Stunden ab jetzt")
    ax[1].set_ylim(0, 1)
    ax[1].grid(alpha=0.3)
    fig.tight_layout()
    out = Path("data/ha_export/forecast.png")
    fig.savefig(out, dpi=110)
    print(f"Plot -> {out}")


if __name__ == "__main__":
    main()
