from fastapi.testclient import TestClient

from hp_twin.adapters.settings import RuntimeSettings
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


def test_refresh_while_running_runs_once_more_with_latest_settings(
    monkeypatch,
    tmp_path,
) -> None:
    import threading

    import hp_twin.api.app as api

    monkeypatch.setenv("HP_TWIN_RUNTIME", str(tmp_path / "runtime.json"))
    RuntimeSettings().save()
    api._PENDING_RUNS.clear()
    started = threading.Event()
    release = threading.Event()
    calls = []

    def fake_cycle(settings, do_push, status):
        calls.append((settings.schedule_interval_minutes, do_push))
        started.set()
        release.wait(timeout=2)
        status.running = False

    monkeypatch.setattr(api, "run_cycle", fake_cycle)
    worker = threading.Thread(target=api._run_locked, args=(RuntimeSettings(), True))
    worker.start()
    assert started.wait(timeout=1)
    assert not api._run_locked(RuntimeSettings(schedule_interval_minutes=5.0), True)
    release.set()
    worker.join(timeout=2)
    assert not worker.is_alive()
    assert calls == [(15.0, True), (5.0, True)]
