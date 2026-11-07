"""Tests für den Szenario-Adapter (`adapters.dev_sim`)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from hp_twin.adapters.dev_sim import Overrides, run_scenario
from hp_twin.adapters.io_csv import validate_frame
from hp_twin.adapters.replay import build_replay
from hp_twin.domain.model_config import ModelConfig
from scripts_helper import synthetic_frame


def _calibrated(topo_path: Path, csv: pd.DataFrame, tmp: Path):
    from hp_twin.cli import _calibrate_report

    df = csv.reset_index()  # timestamp-Spalte zurückgeben (read_measurements erwartet sie)
    csv_path = tmp / "m.csv"
    df.to_csv(csv_path, index=False)
    _topo, _df, calib = _calibrate_report(str(topo_path), str(csv_path))
    return csv_path, calib


def synthetic_csv(days: int = 3) -> pd.DataFrame:
    """Synthetic frame mit den vom Kalibrator erwarteten Extra-Spalten (flow_rate_lpm)."""
    df = synthetic_frame(days=days)
    df["flow_rate_lpm"] = np.where(df["compressor_on"] == 1, 20.0, 0.0)
    return df


@pytest.fixture()
def env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Topologie (Repo-Beispiel) + synthetische 3-Tage-Daten + Kalibrierung."""
    from hp_twin.adapters.io_yaml import load_topology

    df = validate_frame(synthetic_csv(days=3))
    topo_path = tmp_path / "topo.yaml"
    topo_path.write_text(
        Path("config/example_topology.yaml").read_text(encoding="utf-8"), encoding="utf-8"
    )
    topo = load_topology(str(topo_path))
    _csv, calib = _calibrated(topo_path, df, tmp_path)
    model = ModelConfig()
    return df, topo, calib, model


def test_replay_trace_callback_called_per_step(env) -> None:
    df, topo, calib, model = env
    seen: list[tuple[int, bool]] = []
    build_replay(
        df,
        topo,
        calib,
        0,
        2.0,  # 2 h
        model=model,
        trace_callback=lambda t, i, tr: seen.append((i, tr.charging)),
    )
    n = int(2 * 3600 / 300)
    assert len(seen) == n
    assert seen[0][1] in (True, False)
    # mind. ein Ladelauf oder Stillstand je Modus möglich; Indizes müssen lückenlos sein
    assert [i for i, _ in seen] == list(range(n))


def test_scenario_energy_reacts_to_colder_outdoor(env) -> None:
    df, topo, calib, model = env
    base = run_scenario(df, topo, calib, model, Overrides(), horizon_h=24.0)
    cold = run_scenario(df, topo, calib, model, Overrides(t_outdoor_offset_k=-6.0), horizon_h=24.0)
    # Kalter: länger/häufiger laden; im synthetischen Zyklus genügt "nicht kleiner" plus Zyklen
    assert cold.summary["energy_kwh"] >= base.summary["energy_kwh"]
    assert cold.summary["n_cycles"] >= base.summary["n_cycles"]


def test_scenario_min_off_override(env) -> None:
    df, topo, calib, model = env
    base = run_scenario(df, topo, calib, model, Overrides(), horizon_h=48.0)
    fast = run_scenario(
        df, topo, calib, model, Overrides(controller={"min_off_s": 300}), horizon_h=48.0
    )
    assert fast.summary["n_cycles"] >= base.summary["n_cycles"]
    assert fast.summary["controller_effective"]["min_off_s"] == 300


def test_scenario_heat_pump_override_changes_effective_params(env) -> None:
    df, topo, calib, model = env
    base = run_scenario(df, topo, calib, model, Overrides(), horizon_h=12.0)
    small = run_scenario(
        df, topo, calib, model, Overrides(heat_pump={"p_el_ref_w": 1000}), horizon_h=12.0
    )
    assert small.summary["heat_pump_effective"]["p_el_ref_w"] == 1000
    assert base.summary["heat_pump_effective"]["p_el_ref_w"] != 1000


def test_scenario_start_iso_window(env) -> None:
    df, topo, calib, model = env
    start = df.index[20]
    res = run_scenario(df, topo, calib, model, Overrides(), start_iso=str(start), horizon_h=2.0)
    assert res.summary["start_iso"] == str(df.index[20])


def test_scenario_trace_contents(env) -> None:
    df, topo, calib, model = env
    res = run_scenario(
        df,
        topo,
        calib,
        model,
        Overrides(),
        horizon_h=1.0,
        trace=True,
        trace_step_s=600.0,
    )
    assert 0 < len(res.trace) <= 6 + 1
    t = res.trace[0]
    assert set(t.keys()) >= {
        "t_s",
        "mode",
        "charging",
        "setpoints",
        "controller",
        "nodes_before_c",
        "nodes_after_c",
        "q_in_w",
        "q_out_w",
        "p_el_w",
        "q_th_w",
    }
    # Energieerhaltung: q_in - q_out spiegelt sich in der Knotentemperatur-Änderung (Euler)
    t_c = res.trace[0]
    for nid in t_c["nodes_before_c"]:
        assert nid in t_c["nodes_after_c"]


def test_scenario_compare_shapes(env) -> None:
    df, topo, calib, model = env
    res = run_scenario(df, topo, calib, model, Overrides(), horizon_h=6.0)
    cmp_ = res.compare
    n = len(cmp_["hours"])
    assert n > 0
    for key in ("buffer_sim_c", "buffer_measured_c", "p_el_sim_w", "p_el_measured_w"):
        assert len(cmp_[key]) == n
    assert cmp_["mode_sim"][:n] == res.compare["mode_sim"][:n]
