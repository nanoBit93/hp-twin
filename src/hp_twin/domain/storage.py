"""Thermischer Schichtspeicher als n-Knoten-Modell (Energiebilanz, Kopplung, Auftrieb).

Energiebilanz je Knoten j:
    C_j * dT_j/dt = Q_in_j - Q_out_j - UA_j*(T_j - T_amb) + sum_k k_couple*(T_k - T_j)
Zusätzlich:
- Auftriebs-/Inversionsmischung: ist der untere Knoten wärmer als der obere, mischen sie
  (Wärme steigt, bis sie auf gleich warmes Wasser trifft).
- Rücklauf-Einschichtung: ein Teil des Lade-Rücklaufs wird in einen Knoten eingeschichtet.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from hp_twin.domain.topology import Storage

# Wasser: Dichte * spez. Wärmekapazität ~ 1 kg/L * 4186 J/(kg*K)
RHO_CP_J_PER_L_K = 4186.0


@dataclass
class StorageState:
    """Zustand: Temperatur je Knoten (°C), Index = Knotenreihenfolge."""

    t_c: np.ndarray

    def copy(self) -> StorageState:
        return StorageState(self.t_c.copy())


@dataclass(frozen=True)
class StratifiedStorage:
    """Speichermodell, aus Topologie erzeugt. Zustandslos; Zustand liegt in StorageState."""

    node_ids: tuple[str, ...]
    capacity_j_per_k: np.ndarray  # C_j
    ua_w_per_k: np.ndarray
    t_ambient_c: float
    # Kopplungen als (idx_upper, idx_lower, k_up, k_down, buoyancy)
    couplings: tuple[tuple[int, int, float, float, bool], ...]

    @classmethod
    def from_topology(cls, storage: Storage) -> StratifiedStorage:
        ids = tuple(n.id for n in storage.nodes)
        idx = {nid: i for i, nid in enumerate(ids)}
        cap = np.array([n.volume_l * RHO_CP_J_PER_L_K for n in storage.nodes], dtype=float)
        ua = np.array([n.loss_ua_w_per_k for n in storage.nodes], dtype=float)
        coup = tuple(
            (
                idx[c.upper],
                idx[c.lower],
                c.k_w_per_k,
                c.k_w_per_k if c.k_down_w_per_k is None else c.k_down_w_per_k,
                c.buoyancy_mixing,
            )
            for c in storage.couplings
        )
        return cls(ids, cap, ua, storage.t_ambient_c, coup)

    def index(self, node_id: str) -> int:
        return self.node_ids.index(node_id)

    def initial_state(self, storage: Storage) -> StorageState:
        return StorageState(np.array([n.t_init_c for n in storage.nodes], dtype=float))

    def step(
        self,
        state: StorageState,
        dt_s: float,
        q_in_w: dict[str, float] | None = None,
        q_out_w: dict[str, float] | None = None,
    ) -> StorageState:
        """Ein Zeitschritt (explizites Euler). q_in/q_out in W je Knoten-ID."""
        q_in_w = q_in_w or {}
        q_out_w = q_out_w or {}
        t = state.t_c.astype(float).copy()
        net = np.zeros_like(t)
        for i, nid in enumerate(self.node_ids):
            net[i] = (
                q_in_w.get(nid, 0.0)
                - q_out_w.get(nid, 0.0)
                - self.ua_w_per_k[i] * (t[i] - self.t_ambient_c)
            )
        for u, lo, k_up, k_down, _b in self.couplings:
            # asymmetrisch: unten wärmer -> k_up (Auftrieb/Hygienespeicher); oben wärmer -> k_down
            # (Schichtung unterdrückt Wärmeleitung nach unten -> DHW-Zone bleibt oben warm)
            k = k_up if t[lo] >= t[u] else k_down
            flux = k * (t[lo] - t[u])  # >0: Wärme vom unteren zum oberen Knoten
            net[u] += flux
            net[lo] -= flux
        t = t + net * dt_s / self.capacity_j_per_k

        # Auftriebsmischung: Inversion (unten wärmer als oben) energieerhaltend ausgleichen.
        for u, lo, _ku, _kd, buoyancy in self.couplings:
            if buoyancy and t[lo] > t[u]:
                cu, cl = self.capacity_j_per_k[u], self.capacity_j_per_k[lo]
                t_mix = (cu * t[u] + cl * t[lo]) / (cu + cl)
                t[u] = t_mix
                t[lo] = t_mix
        return StorageState(t)

    def inject_return(
        self,
        state: StorageState,
        node_id: str,
        mass_flow_l_per_s: float,
        t_return_c: float,
        dt_s: float,
    ) -> StorageState:
        """Rücklauf-Einschichtung: mischt Rücklaufwasser energieerhaltend in einen Knoten."""
        i = self.index(node_id)
        t = state.t_c.copy()
        m_in = mass_flow_l_per_s * dt_s  # L über Intervall
        c_node = self.capacity_j_per_k[i]
        c_in = m_in * RHO_CP_J_PER_L_K
        if c_in <= 0:
            return StorageState(t)
        t[i] = (c_node * t[i] + c_in * t_return_c) / (c_node + c_in)
        return StorageState(t)

    def sensor_temps(self, state: StorageState) -> dict[str, float]:
        """Fühlerwerte je Knoten (bei 1 Fühler/Knoten = Knotentemperatur)."""
        return {nid: float(state.t_c[i]) for i, nid in enumerate(self.node_ids)}

    def internal_energy_j(self, state: StorageState, t_ref_c: float = 0.0) -> float:
        """Innere Energie relativ zu t_ref (für Energieerhaltungstests)."""
        return float(np.sum(self.capacity_j_per_k * (state.t_c - t_ref_c)))
