"""Portables Konfigurations-Bundle: gesamte Config (+ optional Kalibrierung) als eine Datei.

Export erzeugt ein JSON mit Topologie, Modellparametern, HA-Mapping, Heizlast, Laufzeit-
Einstellungen und (optional) der Kalibrierung. Import schreibt diese Dateien in einer neuen
Installation zurück und validiert sie vorher. Secrets (Tokens) sind standardmäßig NICHT enthalten.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml

from hp_twin.adapters.settings import RuntimeSettings
from hp_twin.domain.heat_load import BuildingHeatLoad
from hp_twin.domain.model_config import ModelConfig
from hp_twin.domain.topology import Topology

BUNDLE_VERSION = 1


def _read_yaml(path: str) -> Any:
    p = Path(path)
    return yaml.safe_load(p.read_text(encoding="utf-8")) if p.exists() else None


def _write_yaml(path: str, data: Any) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(yaml.safe_dump(data, sort_keys=False, allow_unicode=True), encoding="utf-8")


def build_bundle(
    settings: RuntimeSettings, include_secrets: bool = False, include_calibration: bool = True
) -> dict[str, Any]:
    s = settings.model_dump()
    if not include_secrets:
        s["eos_token"] = None
        s["ha_token"] = None
    bundle: dict[str, Any] = {
        "version": BUNDLE_VERSION,
        "created": datetime.now(UTC).isoformat(),
        "settings": s,
        "model": ModelConfig.from_path(settings.model_path).model_dump(),
        "topology": _read_yaml(settings.topology_path),
        "ha_mapping": _read_yaml(settings.ha_mapping_path),
        "heizlast": _read_yaml(settings.heizlast_path),
    }
    cal = Path(settings.calibration_path)
    if include_calibration and cal.exists():
        bundle["calibration"] = json.loads(cal.read_text(encoding="utf-8"))
    return bundle


def apply_bundle(bundle: dict[str, Any], settings: RuntimeSettings) -> dict[str, Any]:
    """Validiert und schreibt die Bundle-Inhalte in die Zielpfade. Secrets bleiben erhalten."""
    if bundle.get("version") != BUNDLE_VERSION:
        raise ValueError(f"Unbekannte Bundle-Version: {bundle.get('version')}")
    # Validierung vor dem Schreiben
    if bundle.get("model") is not None:
        ModelConfig.model_validate(bundle["model"])
    if bundle.get("topology") is not None:
        Topology.model_validate(bundle["topology"])
    if bundle.get("heizlast") is not None:
        BuildingHeatLoad.model_validate(bundle["heizlast"])

    written: list[str] = []
    for key, path in (
        ("topology", settings.topology_path),
        ("ha_mapping", settings.ha_mapping_path),
        ("heizlast", settings.heizlast_path),
        ("model", settings.model_path),
    ):
        if bundle.get(key) is not None:
            _write_yaml(path, bundle[key])
            written.append(path)

    if bundle.get("calibration") is not None:
        p = Path(settings.calibration_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(
            json.dumps(bundle["calibration"], indent=2, ensure_ascii=False), encoding="utf-8"
        )
        written.append(settings.calibration_path)

    # Einstellungen mergen; vorhandene Secrets nur überschreiben, wenn im Bundle gesetzt
    if bundle.get("settings"):
        cur = settings.model_dump()
        for k, v in bundle["settings"].items():
            if k not in cur:
                continue
            if k in ("eos_token", "ha_token"):
                if v:
                    cur[k] = v
            else:
                cur[k] = v
        RuntimeSettings.model_validate(cur).save()
        written.append("runtime.json")

    return {"written": written, "has_calibration": bundle.get("calibration") is not None}
