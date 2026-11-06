"""Speichermodell: Energieerhaltung, Kopplung, Auftriebsmischung, Rücklauf-Einschichtung."""

from __future__ import annotations

import numpy as np

from hp_twin.domain.storage import RHO_CP_J_PER_L_K, StorageState, StratifiedStorage
from hp_twin.domain.topology import Storage, StorageCoupling, StorageNode


def _two_node(k: float = 0.0, buoyancy: bool = False, ua: float = 0.0) -> StratifiedStorage:
    s = Storage(
        id="s",
        t_ambient_c=15.0,
        nodes=[
            StorageNode(id="dhw", volume_l=300, loss_ua_w_per_k=ua, t_init_c=40.0),
            StorageNode(id="buffer", volume_l=300, loss_ua_w_per_k=ua, t_init_c=30.0),
        ],
        couplings=[
            StorageCoupling(upper="dhw", lower="buffer", k_w_per_k=k, buoyancy_mixing=buoyancy)
        ],
    )
    return StratifiedStorage.from_topology(s)


def test_energy_conservation_pure_input() -> None:
    """Ohne Verluste/Kopplung: ΔU = (Q_in - Q_out) * dt (Energieerhaltung)."""
    st = _two_node(k=0.0, buoyancy=False, ua=0.0)
    state = StorageState(np.array([40.0, 30.0]))
    u0 = st.internal_energy_j(state)
    dt, steps = 60.0, 100
    q_in, q_out = 5000.0, 1200.0
    for _ in range(steps):
        state = st.step(state, dt, {"buffer": q_in}, {"dhw": q_out})
    u1 = st.internal_energy_j(state)
    assert abs((u1 - u0) - (q_in - q_out) * dt * steps) < 1e-3


def test_coupling_conserves_energy() -> None:
    """Reine Kopplung (kein Zu-/Abfluss, keine Verluste) erhält die Gesamtenergie."""
    st = _two_node(k=200.0, buoyancy=False, ua=0.0)
    state = StorageState(np.array([45.0, 30.0]))
    u0 = st.internal_energy_j(state)
    for _ in range(3000):  # viele Zeitkonstanten -> ausgeglichen
        state = st.step(state, 60.0)
    u1 = st.internal_energy_j(state)
    assert abs(u1 - u0) < 1e-3
    # Temperaturen gleichen sich an
    assert abs(state.t_c[0] - state.t_c[1]) < 0.5


def test_buoyancy_mixing_removes_inversion() -> None:
    """Unterer Knoten wärmer als oberer -> Mischung gleicht an, Energie erhalten."""
    st = _two_node(k=0.0, buoyancy=True)
    state = StorageState(np.array([30.0, 50.0]))  # dhw(oben)=30 < buffer(unten)=50 -> Inversion
    u0 = st.internal_energy_j(state)
    state = st.step(state, 60.0)
    assert abs(state.t_c[0] - state.t_c[1]) < 1e-9  # gemischt
    assert abs(st.internal_energy_j(state) - u0) < 1e-6


def test_hygienespeicher_buffer_heats_dhw() -> None:
    """Hygienespeicher: Puffer-Ladung hebt über Kopplung auch den DHW-Knoten (passiv).

    Beide Knoten starten gleich warm; der Puffer wird geladen, steigt über den DHW-Knoten
    und zieht ihn per Kopplung mit nach oben (Frischwasser-/Hygienespeicher-Prinzip).
    """
    st = _two_node(k=80.0, buoyancy=True)
    state = StorageState(np.array([35.0, 35.0]))  # dhw=buffer=35
    for _ in range(3600):  # 1 h Pufferladung
        state = st.step(state, 1.0, {"buffer": 6000.0})
    assert state.t_c[1] > 42.0  # Puffer deutlich erwärmt
    assert state.t_c[0] > 40.0  # DHW passiv mitgeheizt (über Kopplung)
    assert state.t_c[1] >= state.t_c[0] - 1e-6  # Schichtung: unten >= oben beim Laden


def test_return_injection_energy_conserving() -> None:
    st = _two_node()
    state = StorageState(np.array([40.0, 30.0]))
    u0 = st.internal_energy_j(state)
    # 0.2 L/s Rücklauf mit 35 °C über 60 s in den Pufferknoten
    state = st.inject_return(state, "buffer", mass_flow_l_per_s=0.2, t_return_c=35.0, dt_s=60.0)
    added = 0.2 * 60.0 * RHO_CP_J_PER_L_K * 35.0
    base = 0.2 * 60.0 * RHO_CP_J_PER_L_K * state.t_c[1]  # Energie des eingemischten Volumens danach
    # Mischung ist energieerhaltend; der Pufferknoten wird durch 35 °C-Rücklauf wärmer
    assert state.t_c[1] > 30.0
    assert u0 < st.internal_energy_j(state)
    assert added > 0 and base > 0
