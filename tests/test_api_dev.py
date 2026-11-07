"""Tests für die Dev-API (`/api/dev/*`) und die Auth-Schicht."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from hp_twin.api.security import _INGRESS_COOKIE, api_token, require_token  # noqa: F401
from scripts_helper import synthetic_frame


@pytest.fixture()
def client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """TestClient auf isolierter Runtime (Kalibrierung aus synthetischen Daten)."""
    import shutil

    import yaml
    from fastapi.testclient import TestClient

    from hp_twin.api.app import app

    # Isolierte runtime.json + minimale Konfiguration in tmp_path
    shutil.copy("config/example_topology.yaml", tmp_path / "topology.yaml")
    (tmp_path / "model.yaml").write_text(yaml.safe_dump({}), encoding="utf-8")

    df = synthetic_frame(days=3)
    df["flow_rate_lpm"] = np.where(df["compressor_on"] == 1, 20.0, 0.0)
    # Zeit nahe "jetzt" legen, damit Startzustand/Datenalter plausibel sind
    now = pd.Timestamp.now(tz="UTC").floor("5min")
    df["timestamp"] = (now - pd.to_timedelta(np.arange(len(df))[::-1] * 5, unit="min")).strftime(
        "%Y-%m-%dT%H:%M:%SZ"
    )
    df.to_csv(tmp_path / "measurements.csv", index=False)
    monkeypatch.setenv(
        "HP_TWIN_RUNTIME",
        str(tmp_path / "runtime.json"),
    )
    settings = {
        "topology_path": str(tmp_path / "topology.yaml"),
        "model_path": str(tmp_path / "model.yaml"),
        "calibration_path": str(tmp_path / "calibration.json"),
        "csv_path": str(tmp_path / "measurements.csv"),
        "use_stored_calibration": False,
        "export_before_push": False,
    }
    import json

    (tmp_path / "runtime.json").write_text(json.dumps(settings), encoding="utf-8")
    return TestClient(app)


def test_dev_state_and_data(client) -> None:
    r = client.get("/api/dev/state")
    assert r.status_code == 200
    j = r.json()
    assert j["nodes_c"]["dhw"] is not None and j["nodes_c"]["buffer"] is not None
    assert j["controller_state"]["mode"] in {"idle", "pause", "heat", "dhw"}

    r = client.get("/api/dev/data")
    assert r.status_code == 200
    j = r.json()
    assert j["n_rows"] == len(synthetic_frame(days=3))
    assert "calibration" in j and "state" in j
    assert j["span_days"] >= 2.9


def test_dev_data_window_and_downsample(client) -> None:
    r = client.get("/api/dev/data")
    j = r.json()
    first = j["first_iso"]
    # 6-h-Fenster ab first: 72 Punkte à 5 min -> ungekürzt
    end = str(pd.Timestamp(first) + pd.Timedelta(hours=6)).replace(" ", "T")
    r2 = client.get(f"/api/dev/data?start={first}&end={end}")
    assert r2.status_code == 200
    w = r2.json()["window"]
    assert w["n"] <= 73 and w["downsampled_by"] == 1
    assert "p_el_w" in w["series"] and "t_hk1_c" in w["series"]
    assert len(w["series"]["t_hk1_c"]) == w["n"]


def test_dev_calibration_model_present(client) -> None:
    r = client.get("/api/dev/calibration")
    assert r.status_code == 200
    j = r.json()
    assert "heat_pump" in j["calibration"]
    assert "montecarlo" in j["model"]


def test_dev_trace_contains_controller_steps(client) -> None:
    r = client.get("/api/dev/trace?horizon_h=2&trace_step_s=300")
    assert r.status_code == 200
    j = r.json()
    tr = j["trace"]
    assert 0 < len(tr) <= 25
    first = tr[0]
    assert {"mode", "charging", "setpoints", "controller", "nodes_before_c"} <= set(first)
    assert first["controller"]["mode"] in {"idle", "heat", "dhw", "pause"}
    # summary verweist auf dasselbe Fenster
    assert j["summary"]["topology"] == "wpl-a-25-kombispeicher"


def test_dev_scenario_and_batch(client) -> None:
    base = {"horizon_h": 6, "trace": False}
    r = client.post("/api/dev/scenario", json=base)
    assert r.status_code == 200
    j = r.json()
    assert "summary" in j and "compare" in j
    assert "trace" not in j or not j["trace"]

    # MitOverrides: mehr Mindeststillstand -> weniger Zyklen (im synth. 2.5-h-Zyklus)
    r_batch = client.post(
        "/api/dev/scenarios",
        json=[
            base,
            {"horizon_h": 6, "overrides": {"controller": {"min_off_s": 3600}}},
        ],
    )
    assert r_batch.status_code == 200
    jb = r_batch.json()
    assert jb["n"] == 2 and len(jb["results"]) == 2
    assert "energy_kwh" in jb and len(jb["energy_kwh"]) == 2


def test_dev_scenario_trace_on_demand(client) -> None:
    r = client.post(
        "/api/dev/scenario",
        json={"horizon_h": 3, "trace": True, "trace_step_s": 900.0},
    )
    assert r.status_code == 200
    j = r.json()
    assert j["trace"] and 0 < len(j["trace"]) <= 13


def test_dev_batch_bounds(client) -> None:
    r = client.post("/api/dev/scenarios", json=[])
    assert r.status_code == 422
    bodies = [{"horizon_h": 1}] * 21
    r = client.post("/api/dev/scenarios", json=bodies)
    assert r.status_code == 422


def test_auth_token_enforced(monkeypatch: pytest.MonkeyPatch, client) -> None:
    monkeypatch.setenv("HP_TWIN_API_TOKEN", "s3cret")
    c = client
    assert c.get("/api/dev/state").status_code == 401
    assert c.get("/api/dev/state", headers={"Authorization": "Bearer wrong"}).status_code == 401
    assert c.get("/api/dev/state", headers={"Authorization": "Bearer s3cret"}).status_code == 200
    assert c.get("/api/dev/state?token=s3cret").status_code == 200
    # Klassische Schreib-Endpunkte benötigen den Token ebenfalls
    assert c.post("/api/settings", json={}).status_code == 401
    assert (
        c.post(
            "/api/settings", headers={"Authorization": "Bearer s3cret"}, json={"forecast_runs": 5}
        ).status_code
        == 200
    )


def test_auth_ingress_cookie_bypasses(client, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HP_TWIN_API_TOKEN", "s3cret")
    c = client
    c.cookies.set(_INGRESS_COOKIE, "session")
    assert c.get("/api/dev/state").status_code == 200  # HA-Ingress ist authentifiziert
    assert c.post("/api/settings", json={"forecast_runs": 5}).status_code == 200


def test_existing_write_endpoints_have_auth(monkeypatch: pytest.MonkeyPatch, client) -> None:
    monkeypatch.setenv("HP_TWIN_API_TOKEN", "t")
    c = client
    for method, path, kwargs in (
        ("post", "/api/model", {"json": {}}),
        ("post", "/api/bundle", {"json": {}}),
        ("post", "/api/push_now", {"json": None}),
    ):
        r = getattr(c, method)(path, **kwargs)
        assert r.status_code == 401, path
