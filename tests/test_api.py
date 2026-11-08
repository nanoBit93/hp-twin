from fastapi.testclient import TestClient

from hp_twin.api.app import app


def test_health() -> None:
    assert TestClient(app).get("/health").json() == {"status": "ok"}


def test_simulate_endpoint() -> None:
    client = TestClient(app)
    r = client.post("/simulate", json={"runs": 5, "horizon_h": 24, "from_end": True})
    assert r.status_code == 200
    j = r.json()
    assert j["n_events"] >= 0
    assert len(j["slots_15min"]["p_el_mean_w"]) == 96
    assert len(j["eos_payload"]["values"]) == 96
    assert j["eos_payload"]["unit"] == "W"


def test_dashboard_served() -> None:
    client = TestClient(app)
    r = client.get("/")
    assert r.status_code == 200 and "hp-twin" in r.text


def test_analysis_endpoint() -> None:
    client = TestClient(app)
    r = client.get("/api/analysis?runs=5&horizon_h=24&from_end=true")
    assert r.status_code == 200
    j = r.json()
    assert set(j.keys()) >= {"config", "calibration", "factors", "sim", "history", "forecast"}
    assert j["forecast"]["mode"] == "live"
    assert len(j["forecast"]["p_el_mean_w"]) == 96
    assert "active_rooms" in j["factors"]


def test_versions_in_sync() -> None:
    """pyproject- und Add-on-Version müssen gleich sein (Add-on installiert Tag v<version>)."""
    import re
    from pathlib import Path

    root = Path(__file__).parent.parent
    py = re.search(r'^version = "([^"]+)"', (root / "pyproject.toml").read_text(), re.M)
    ad = re.search(r'^version: "([^"]+)"', (root / "hp-twin-addon/config.yaml").read_text(), re.M)
    assert py and ad and py.group(1) == ad.group(1)


def test_api_version() -> None:
    assert "version" in TestClient(app).get("/api/version").json()
