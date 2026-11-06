"""Laufzeit-Einstellungen: Persistenz und Token-Schutz."""

from __future__ import annotations

from pathlib import Path

from hp_twin.adapters.settings import RuntimeSettings


def test_settings_roundtrip(tmp_path: Path) -> None:
    p = tmp_path / "runtime.json"
    s = RuntimeSettings(
        eos_base_url="http://x:8081", eos_managed_load_id="hp_ext", eos_token="secret"
    )
    s.save(str(p))
    loaded = RuntimeSettings.load(str(p))
    assert loaded.eos_base_url == "http://x:8081"
    assert loaded.eos_managed_load_id == "hp_ext"
    assert loaded.eos_token == "secret"


def test_public_dict_hides_token() -> None:
    s = RuntimeSettings(eos_token="secret")
    d = s.public_dict()
    assert d["eos_token"] is None and d["eos_token_set"] is True


def test_settings_endpoints(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("HP_TWIN_RUNTIME", str(tmp_path / "runtime.json"))
    from fastapi.testclient import TestClient

    from hp_twin.api.app import app

    c = TestClient(app)
    assert c.get("/api/settings").status_code == 200
    r = c.post("/api/settings", json={"eos_base_url": "http://y:8081", "forecast_runs": 20})
    assert r.status_code == 200 and r.json()["eos_base_url"] == "http://y:8081"
    st = c.get("/api/status").json()
    assert "data_age_minutes_live" in st and "eos_configured" in st
