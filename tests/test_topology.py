from pathlib import Path

import pytest
from pydantic import ValidationError

from hp_twin.adapters.io_yaml import load_topology
from hp_twin.domain.topology import DhwMode

EXAMPLE = Path(__file__).parent.parent / "config" / "example_topology.yaml"


def test_example_topology_loads() -> None:
    t = load_topology(EXAMPLE)
    assert [n.id for n in t.storage.nodes] == ["dhw", "buffer"]
    assert sum(n.volume_l for n in t.storage.nodes) == 600
    assert t.controller.dhw_mode is DhwMode.PARALLEL
    assert t.controller.effective_mode_switch_pause_s == 1800


def test_unknown_node_reference_rejected() -> None:
    t = load_topology(EXAMPLE).model_dump()
    t["consumers"][1]["node"] = "nope"
    from hp_twin.domain.topology import Topology

    with pytest.raises(ValidationError):
        Topology.model_validate(t)


def test_extra_fields_rejected() -> None:
    from hp_twin.domain.topology import Topology

    t = load_topology(EXAMPLE).model_dump()
    t["bogus"] = 1
    with pytest.raises(ValidationError):
        Topology.model_validate(t)
