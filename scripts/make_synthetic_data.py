"""Erzeugt synthetische Beispieldaten + Vorlage (grobes Zyklusverhalten, nur für Tests)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from hp_twin.adapters.io_csv import OPTIONAL_COLUMNS, REQUIRED_COLUMNS


def synthetic_frame(days: int = 3, step_min: int = 5, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = pd.date_range(
        "2026-01-10", periods=days * 24 * 60 // step_min, freq=f"{step_min}min", tz="UTC"
    )
    n = len(idx)
    hours = np.arange(n) * step_min / 60
    t_out = 2 + 4 * np.sin((hours - 9) / 24 * 2 * np.pi) + rng.normal(0, 0.2, n)
    ghi = np.clip(300 * np.sin((hours - 6) / 24 * 2 * np.pi), 0, None)
    cycle = (hours % 2.5) < 0.6  # ~35 min Ladung pro 2,5 h
    mode = np.where(cycle, "heat", "idle")
    p_el = np.where(cycle, 3500 + rng.normal(0, 80, n), 0.0)
    hk1 = (
        27
        + 13 * ((hours % 2.5) / 2.5) * 0
        + np.where(cycle, 13 * ((hours % 2.5) / 0.6), 40 - 13 * (((hours % 2.5) - 0.6) / 1.9))
    )
    df = pd.DataFrame(
        {
            "timestamp": idx.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "t_outdoor_c": t_out.round(2),
            "t_outdoor_fc_c": (t_out + rng.normal(0, 0.8, n)).round(2),
            "ghi_wm2": ghi.round(1),
            "t_room_wohnen_c": (21 + 0.5 * np.sin(hours / 24 * 2 * np.pi)).round(2),
            "t_room_schlafen_c": (19 + 0.3 * np.sin(hours / 24 * 2 * np.pi)).round(2),
            "compressor_on": cycle.astype(int),
            "mode": mode,
            "circuit_active": np.where(cycle, "hk1", "none"),
            "p_el_w": p_el.round(0),
            "q_heat_wh": (p_el * 3.5 * step_min / 60).round(1),
            "t_flow_c": np.where(cycle, 38.0, 30.0),
            "t_return_c": np.where(cycle, 33.0, 28.0),
            "t_dhw_c": (43.5 + rng.normal(0, 0.2, n)).round(2),
            "t_hk1_c": hk1.round(2),
            "t_hk2_c": (26.5 + rng.normal(0, 0.4, n)).round(2),
            "setpoint_dhw_c": 40.5,
            "setpoint_hk1_c": 31.5,
            "setpoint_hk2_c": 27.5,
        }
    )
    return df


def main() -> None:
    root = Path(__file__).resolve().parent.parent / "data"
    (root / "synthetic").mkdir(parents=True, exist_ok=True)
    df = synthetic_frame()
    df.to_csv(root / "synthetic" / "synthetic_3d_5min.csv", index=False)
    cols = ["timestamp", *REQUIRED_COLUMNS, *OPTIONAL_COLUMNS, "t_room_wohnen_c"]
    df[cols].head(3).to_csv(root / "template.csv", index=False)


if __name__ == "__main__":
    main()
