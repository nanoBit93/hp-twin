"""SG3 settings are visible, persisted through the real API, and exported."""

from pathlib import Path

from fastapi.testclient import TestClient

from hp_twin.adapters.bundle import build_bundle
from hp_twin.adapters.settings import RuntimeSettings
from hp_twin.api.app import app


def test_sg3_settings_round_trip_through_api(tmp_path: Path, monkeypatch) -> None:
    path = tmp_path / "runtime.json"
    monkeypatch.setenv("HP_TWIN_RUNTIME", str(path))
    RuntimeSettings(
        sg_enabled=True,
        sg_dry_run=True,
        sg_load_id="sg3_req",
        profile_load_id="hp_ext",
        sg_bivalence_outdoor_c=-2.5,
        sg_contingent_ttl_minutes=45,
    ).save()
    client = TestClient(app)

    before = client.get("/api/settings")
    assert before.status_code == 200
    assert before.json()["sg_bivalence_outdoor_c"] == -2.5
    assert before.json()["sg_contingent_ttl_minutes"] == 45

    response = client.post(
        "/api/settings",
        json={
            "sg_bivalence_outdoor_c": -4.0,
            "sg_contingent_ttl_minutes": 90,
        },
    )
    assert response.status_code == 200
    saved = RuntimeSettings.load()
    assert saved.sg_bivalence_outdoor_c == -4.0
    assert saved.sg_contingent_ttl_minutes == 90

    bundle = build_bundle(saved, include_secrets=False)
    assert bundle["settings"]["sg_bivalence_outdoor_c"] == -4.0
    assert bundle["settings"]["sg_contingent_ttl_minutes"] == 90


def test_sg3_controls_exist_and_are_wired_into_settings_collection() -> None:
    html = TestClient(app).get("/").text
    for field in (
        "sg_enabled",
        "sg_dry_run",
        "sg_load_id",
        "profile_load_id",
        "sg_bivalence_outdoor_c",
        "sg_top_setpoint_c",
        "sg_contingent_ttl_minutes",
    ):
        assert f'id="{field}"' in html
        assert field in html.split("const S_FIELDS", 1)[1].split(";", 1)[0] or field in (
            "sg_enabled",
            "sg_dry_run",
        )
    assert "horizon_h" in html
    for element in (
        "sg3Diagnostics",
        "sg3Meta",
        "sg3Reasons",
        "chartSg3",
        "sg3Candidates",
        "sg3Rejected",
        "sg3Raw",
    ):
        assert f'id="{element}"' in html
    assert "renderSg3(sg)" in html
    assert "allowed_windows" in html
    assert "placement_slots" in html
    assert "feasible_reasons" in html
    assert "sgThermal(chosen, frameStart)" in html
    thermal_script = html.split("function sgThermal", 1)[1].split("function renderSg3", 1)[0]
    assert "sg.frame_start" not in thermal_script


def test_sg3_defaults_are_safe() -> None:
    settings = RuntimeSettings()
    assert settings.sg_enabled is False
    assert settings.sg_dry_run is True
    assert settings.horizon_h == 48.0
    assert settings.sg_load_id == "sg3_req"
    assert settings.profile_load_id == "hp_ext"
    assert settings.sg_load_id != settings.profile_load_id
    assert settings.sg_top_setpoint_c == 49.8
    assert not hasattr(settings, "sg_emergency_gate")
    assert not hasattr(settings, "sg_min_pv_surplus_w")
    assert not hasattr(settings, "sg_max_price_eur")


def test_legacy_gate_settings_do_not_reappear_in_public_config() -> None:
    settings = RuntimeSettings.model_validate(
        {
            "sg_emergency_gate": True,
            "sg_min_pv_surplus_w": 2500,
            "sg_max_price_eur": 0.1,
        }
    )
    assert "sg_emergency_gate" not in settings.public_dict()
    assert "sg_min_pv_surplus_w" not in settings.public_dict()
    assert "sg_max_price_eur" not in settings.public_dict()
