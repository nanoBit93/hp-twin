"""YAML-Laden der Topologie."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from hp_twin.domain.heat_load import BuildingHeatLoad
from hp_twin.domain.topology import Topology


def load_topology(path: str | Path) -> Topology:
    data: Any = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    return Topology.model_validate(data)


def load_heat_load(path: str | Path) -> BuildingHeatLoad:
    data: Any = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    return BuildingHeatLoad.model_validate(data)
