"""Portables Config-Bundle: Export -> Import Round-Trip."""

from __future__ import annotations

import json
from pathlib import Path

import yaml

from hp_twin.adapters.bundle import apply_bundle, build_bundle
from hp_twin.adapters.settings import RuntimeSettings
from hp_twin.domain.model_config import ModelConfig


def _seed(dirp: Path) -> RuntimeSettings:
    # Topologie aus dem Repo-Beispiel übernehmen
    topo = yaml.safe_load(Path("config/example_topology.yaml").read_text(encoding="utf-8"))
    (dirp / "topology.yaml").write_text(yaml.safe_dump(topo), encoding="utf-8")
    (dirp / "model.yaml").write_text(yaml.safe_dump(ModelConfig().model_dump()), encoding="utf-8")
    (dirp / "heizlast.yaml").write_text(
        yaml.safe_dump(
            {
                "design_outdoor_c": -10,
                "rooms": [
                    {"name": "wz", "design_heat_load_w": 1200, "climate_entity": "climate.wz"}
                ],
            }
        ),
        encoding="utf-8",
    )
    (dirp / "ha_mapping.yaml").write_text(
        yaml.safe_dump({"sensors": {}, "climate": {}}), encoding="utf-8"
    )
    return RuntimeSettings(
        topology_path=str(dirp / "topology.yaml"),
        model_path=str(dirp / "model.yaml"),
        heizlast_path=str(dirp / "heizlast.yaml"),
        ha_mapping_path=str(dirp / "ha_mapping.yaml"),
        calibration_path=str(dirp / "calibration.json"),
        eos_base_url="http://eos:8081",
        eos_token="secret",
    )


def test_bundle_roundtrip(tmp_path: Path, monkeypatch) -> None:
    src = tmp_path / "src"
    src.mkdir()
    s = _seed(src)

    # Bundle ohne Secrets exportieren
    b = build_bundle(s, include_secrets=False, include_calibration=False)
    assert b["version"] == 1
    assert b["settings"]["eos_token"] is None  # Secret nicht im Bundle
    assert b["topology"]["name"] == "wpl-a-25-kombispeicher"

    # In ZIELinstallation anwenden (andere Pfade)
    dst = tmp_path / "dst"
    dst.mkdir()
    monkeypatch.setenv("HP_TWIN_RUNTIME", str(dst / "runtime.json"))
    target = RuntimeSettings(
        topology_path=str(dst / "topology.yaml"),
        model_path=str(dst / "model.yaml"),
        heizlast_path=str(dst / "heizlast.yaml"),
        ha_mapping_path=str(dst / "ha_mapping.yaml"),
        calibration_path=str(dst / "calibration.json"),
        eos_token="existing",  # vorhandenes Secret muss erhalten bleiben
    )
    target.save()
    res = apply_bundle(b, target)
    assert str(dst / "topology.yaml") in res["written"]
    # Config kam an
    topo = yaml.safe_load((dst / "topology.yaml").read_text(encoding="utf-8"))
    assert topo["name"] == "wpl-a-25-kombispeicher"
    # Secret blieb erhalten (Bundle brachte keins)
    assert RuntimeSettings.load(str(dst / "runtime.json")).eos_token == "existing"
    # eos_base_url wurde übernommen
    assert RuntimeSettings.load(str(dst / "runtime.json")).eos_base_url == "http://eos:8081"


def test_bundle_with_calibration(tmp_path: Path) -> None:
    from hp_twin.domain.calibration import (
        CalibrationReport,
        DischargeFit,
        HeatPumpFit,
        ThresholdEstimate,
        calibration_to_json,
    )
    from hp_twin.domain.heat_source import HeatPumpAirWater

    s = _seed(tmp_path)
    rep = CalibrationReport(
        HeatPumpFit(HeatPumpAirWater(), 100, 0.7, 250, 0.3, 1400, 4.3),
        ThresholdEstimate("buffer", 56, 56, -5.0, 8.35),
        DischargeFit(192, -390, 1.07, 555, 0.05),
        ["note"],
    )
    Path(s.calibration_path).write_text(json.dumps(calibration_to_json(rep)), encoding="utf-8")
    b = build_bundle(s, include_calibration=True)
    assert "calibration" in b
    assert b["calibration"]["discharge"]["ua_w_per_k"] == 192
