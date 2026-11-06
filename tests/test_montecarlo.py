"""Monte-Carlo-Aggregation, Ereignis-Extraktion, EOS-Output."""

from __future__ import annotations

import numpy as np

from hp_twin.adapters.eos_connect import load_profile_from_slots
from hp_twin.domain.montecarlo import aggregate_slots, extract_events, run_ensemble


def test_extract_events() -> None:
    t = np.arange(120) * 60.0
    mode = ["idle"] * 120
    p = np.full(120, 80.0)
    for i in range(10, 20):
        mode[i] = "heat"
        p[i] = 2300.0
    for i in range(60, 75):
        mode[i] = "dhw"
        p[i] = 3000.0
    ev = extract_events(t, mode, p)
    assert len(ev) == 2
    assert ev[0].mode == "heat" and abs(ev[0].mean_p_el_w - 2300) < 1
    assert ev[1].mode == "dhw" and abs(ev[1].duration_min - 15) < 1e-6


def test_aggregate_slots_percentiles() -> None:
    t = np.arange(192 * 15) * 60.0  # 48 h @ 1 min
    # 3 Läufe mit unterschiedlicher Konstantleistung
    runs = [np.full(len(t), x) for x in (1000.0, 2000.0, 3000.0)]
    sf = aggregate_slots(t, runs, slot_s=900.0)
    assert sf.n_slots == 192
    assert abs(sf.p_el_mean_w[0] - 2000.0) < 1e-6
    assert abs(sf.p_el_p10_w[0] - 1200.0) < 50  # ~10. Perzentil
    assert abs(sf.p_el_p90_w[0] - 2800.0) < 50
    assert np.all(sf.run_probability == 1.0)  # alle > 200 W
    assert abs(sf.energy_wh[0] - 2000.0 * 0.25) < 1e-6


def test_run_probability() -> None:
    t = np.arange(96 * 15) * 60.0
    runs = [np.full(len(t), 80.0), np.full(len(t), 2500.0)]  # 1 idle, 1 läuft
    sf = aggregate_slots(t, runs, slot_s=900.0)
    assert np.allclose(sf.run_probability, 0.5)


def test_eos_profile_192() -> None:
    t = np.arange(192 * 15) * 60.0
    runs = [np.full(len(t), 1500.0)]
    sf = aggregate_slots(t, runs, slot_s=900.0)
    prof = load_profile_from_slots(sf, ttl_minutes=180)
    p = prof.to_payload()
    assert len(p["values"]) == 192
    assert p["unit"] == "W" and p["start"] == "now" and p["ttl_minutes"] == 180
    assert abs(p["values"][0] - 1500.0) < 1e-6


def test_run_ensemble_deterministic() -> None:
    def run_once(rng: np.random.Generator) -> np.ndarray:
        return np.full(10, rng.normal(1000, 100))

    a = run_ensemble(run_once, 50, seed=42)
    b = run_ensemble(run_once, 50, seed=42)
    assert len(a) == 50
    assert np.allclose([x[0] for x in a], [x[0] for x in b])  # reproduzierbar


def test_run_forecast_endtoend() -> None:
    from hp_twin.adapters.forecast import run_forecast
    from hp_twin.cli import _calibrate_report
    from hp_twin.domain.model_config import ModelConfig

    model = ModelConfig()
    model.montecarlo.n_runs = 5
    topo, df, calib = _calibrate_report(
        "config/example_topology.yaml", "data/ha_export/measurements.csv", model=model
    )
    fc = run_forecast(df, topo, calib, 0, horizon_h=24.0, model=model)
    assert fc.slots.n_slots == 96
    assert len(fc.eos.values_w) == 96
    assert fc.slots.energy_wh.sum() > 0
    # P90 >= P10 in jedem Slot
    import numpy as np

    assert np.all(fc.slots.p_el_p90_w >= fc.slots.p_el_p10_w - 1e-6)


def test_eos_push_payload(monkeypatch) -> None:
    import numpy as np

    from hp_twin.adapters import eos_connect
    from hp_twin.adapters.eos_connect import EosLoadProfile, mqtt_topic, push_rest

    captured = {}

    class _Resp:
        status = 200

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def read(self):
            return b"{}"

    def fake_urlopen(req, timeout=0):
        captured["url"] = req.full_url
        captured["body"] = req.data
        return _Resp()

    monkeypatch.setattr(eos_connect.urllib.request, "urlopen", fake_urlopen)
    prof = EosLoadProfile(values_w=[100.0] * 192, ttl_minutes=180)
    status = push_rest("http://eos:8080", "wp1", prof, token="t")
    assert status == 200
    assert captured["url"].endswith("/api/managed_loads/wp1/push")
    body = __import__("json").loads(captured["body"])
    assert len(body["values"]) == 192 and body["unit"] == "W"
    assert mqtt_topic("wp1") == "eos_connect/managed_load/wp1/set"
    _ = np
