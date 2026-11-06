"""Zyklus-Analyse + Plot der exportierten Messdaten (Plausibilitätsabgleich für M2)."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from hp_twin.adapters.io_csv import read_measurements  # noqa: E402


def analyze(df) -> None:  # type: ignore[no-untyped-def]
    on = (df["compressor_on"] == 1).astype(int)
    starts = int((on.diff() == 1).sum())
    runs, st = [], None
    for t, val in on.items():
        if val == 1 and st is None:
            st = t
        elif val == 0 and st is not None:
            runs.append((t - st).total_seconds() / 60)
            st = None
    runs_a = np.array(runs) if runs else np.array([0.0])
    days = (df.index.max() - df.index.min()).total_seconds() / 86400
    print(f"Zeitraum {df.index.min():%Y-%m-%d} .. {df.index.max():%Y-%m-%d}  ({days:.1f} Tage)")
    print(f"Takte: {starts} = {starts / days:.1f}/Tag")
    print(
        f"Laufdauer min: median={np.median(runs_a):.0f}  p10={np.percentile(runs_a, 10):.0f}  "
        f"p90={np.percentile(runs_a, 90):.0f}"
    )
    print("Modus:\n" + df["mode"].value_counts().to_string())
    for c in ["t_dhw_c", "t_hk1_c", "t_hk2_c", "p_el_w", "setpoint_dhw_c", "setpoint_hk1_c"]:
        s = df[c]
        print(
            f"  {c:16} min={np.nanmin(s):6.1f} max={np.nanmax(s):6.1f} med={np.nanmedian(s):6.1f}"
        )


def plot(df, out: Path, days: float) -> None:  # type: ignore[no-untyped-def]
    end = df.index.max()
    start = end - np.timedelta64(int(days * 24), "h")
    d = df.loc[start:end]
    fig, ax = plt.subplots(3, 1, figsize=(14, 9), sharex=True)
    ax[0].plot(d.index, d["t_dhw_c"], label="DHW", color="tab:red")
    ax[0].plot(d.index, d["setpoint_dhw_c"], "--", color="tab:red", alpha=0.5, label="DHW soll")
    ax[0].plot(d.index, d["t_hk1_c"], label="Puffer/HK1", color="tab:blue")
    ax[0].plot(d.index, d["setpoint_hk1_c"], "--", color="tab:blue", alpha=0.5, label="HK1 soll")
    ax[0].plot(d.index, d["t_hk2_c"], label="HK2", color="tab:green")
    ax[0].set_ylabel("°C")
    ax[0].legend(ncol=3, fontsize=8)
    ax[0].grid(alpha=0.3)
    ax[1].fill_between(d.index, 0, d["p_el_w"], color="tab:orange", alpha=0.6)
    ax[1].set_ylabel("P_el [W]")
    ax[1].grid(alpha=0.3)
    ax[2].step(
        d.index, (d["compressor_on"] == 1).astype(int), where="post", color="k", label="Verdichter"
    )
    for i, m in enumerate(["dhw", "heat", "defrost"], 1):
        ax[2].step(
            d.index, (d["mode"] == m).astype(int) * (i + 1), where="post", alpha=0.6, label=m
        )
    ax[2].plot(d.index, d["t_outdoor_c"] / 10, color="gray", lw=0.8, label="T_aus/10")
    ax[2].set_ylabel("Status")
    ax[2].legend(ncol=5, fontsize=8)
    ax[2].grid(alpha=0.3)
    ax[2].xaxis.set_major_formatter(mdates.DateFormatter("%d.%m %H:%M"))
    fig.autofmt_xdate()
    fig.suptitle("Reale WP-Historie (Home Assistant)")
    fig.tight_layout()
    fig.savefig(out, dpi=110)
    print(f"Plot -> {out}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--csv", default="data/ha_export/measurements.csv")
    p.add_argument("--out", default="data/ha_export/history_overview.png")
    p.add_argument("--plot-days", type=float, default=3.0)
    args = p.parse_args()
    df = read_measurements(args.csv)
    analyze(df)
    plot(df, Path(args.out), args.plot_days)


if __name__ == "__main__":
    main()
