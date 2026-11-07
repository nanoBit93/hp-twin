"""Deterministischer Vorwärtssimulator: fester Zeitschritt, Horizont (Default 48 h).

Koppelt Speicher + Wärmeerzeuger + Regler + Wärmesenken (LoadModel). Die Last kann konstant
vorgegeben sein (ConstantLoad) oder vom Gebäudemodell/Zapfprofil kommen (HeatingCircuitLoad,
DhwDraw über CompositeLoad).

Für die Analyse-API kann pro Zeitschritt ein Trace-Schnappschuss gesammelt werden
(`trace_callback`): alle Zwischengrößen eines Takt-Entscheids (Reglerzustand, Sollwerte,
Wärmeströme, Knotentemperaturen) – Grundlage für die Zwischenschritt-Endpunkte.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np

from hp_twin.domain.consumers import LoadModel
from hp_twin.domain.controller import Controller, ControllerState, Mode, Setpoints
from hp_twin.domain.heat_source import HeatSource
from hp_twin.domain.storage import StorageState, StratifiedStorage

# (t_s) -> Sollwerte ; (t_s) -> Außentemperatur °C
SetpointProvider = Callable[[float], Setpoints]
OutdoorProvider = Callable[[float], float]
#: callback(t_s, i, ctx) für jeden Zeitschritt; ctx enthält die Zwischengrößen des Takt-Entscheids.
StepTraceCallback = Callable[[float, int, "StepTrace"], None]


@dataclass
class SimConfig:
    timestep_s: float = 60.0
    horizon_s: float = 48 * 3600.0


@dataclass(frozen=True)
class StepTrace:
    """Zwischengrößen EINES Zeitschritts (Froschperspektive des Reglers und Speichers)."""

    t_s: float  # Zeitpunkt (s ab Simulationsstart)
    index: int  # Schrittindex
    # Speicherzustand VOR dem Zeitschritt
    node_temps_before_c: dict[str, float]
    # Reglerentscheid (nach decide, vor dem Schritt)
    mode: str
    controller_state: ControllerState
    setpoints: Setpoints
    charging: bool
    target_node: str | None
    flow_setpoint_c: float
    t_flow_eff_c: float  # effektiver Simulator-Vorlauf (Knotentemp + Approach, gedeckelt)
    # Erzeuger-Ist (nach Koppeln)
    p_el_w: float
    q_th_w: float
    cop_carnot_based: float  # Q_th / P_el im Takt (NaN bei Standby)
    # Wärmeströme im Zeitschritt
    q_in_w: dict[str, float]
    q_out_w: dict[str, float]
    # Speicherzustand NACH dem Zeitschritt
    node_temps_after_c: dict[str, float]


@dataclass
class SimResult:
    time_s: np.ndarray
    p_el_w: np.ndarray
    q_th_w: np.ndarray
    mode: list[str]
    node_temps_c: dict[str, np.ndarray]
    sensor_temps_c: dict[str, np.ndarray]

    @property
    def energy_kwh(self) -> float:
        dt = float(self.time_s[1] - self.time_s[0]) if len(self.time_s) > 1 else 60.0
        return float(self.p_el_w.sum() * dt / 3600.0 / 1000.0)


@dataclass
class Simulator:
    storage: StratifiedStorage
    source: HeatSource
    controller: Controller
    config: SimConfig = field(default_factory=SimConfig)
    condenser_approach_k: float = 7.0  # Vorlauf liegt um diesen Betrag über der Knotentemperatur

    def run(
        self,
        state0: StorageState,
        setpoints: SetpointProvider,
        load: LoadModel,
        outdoor: OutdoorProvider,
        ctrl_state: ControllerState | None = None,
        trace_callback: StepTraceCallback | None = None,
    ) -> SimResult:
        dt = self.config.timestep_s
        n = int(round(self.config.horizon_s / dt))
        state = state0.copy()
        cs = ctrl_state or ControllerState()

        time = np.arange(n, dtype=float) * dt
        p_el = np.zeros(n)
        q_th = np.zeros(n)
        modes: list[str] = []
        node_hist = {nid: np.zeros(n) for nid in self.storage.node_ids}

        for i in range(n):
            t = time[i]
            temps = self.storage.sensor_temps(state)
            temps_before = dict(temps)
            sp = setpoints(t)
            cs, decision = self.controller.decide(cs, temps, sp, dt)
            # Vorlauf folgt der AKTUELLEN Knotentemperatur (steigt im Takt mit), gedeckelt am
            # Soll-Vorlauf. So modulieren COP/Q_th und P_el über den Takt.
            if decision.charging and decision.target_node is not None:
                t_flow_eff = min(
                    temps[decision.target_node] + self.condenser_approach_k,
                    decision.flow_setpoint_c,
                )
            else:
                t_flow_eff = decision.flow_setpoint_c
            out = self.source.output(outdoor(t), t_flow_eff, decision.charging)

            q_in: dict[str, float] = {}
            if decision.charging and decision.target_node is not None:
                q_in[decision.target_node] = out.q_th_w
            q_out = load.draw(t, dt, temps)

            state = self.storage.step(state, dt, q_in_w=q_in, q_out_w=q_out)

            p_el[i] = out.p_el_w
            q_th[i] = out.q_th_w
            modes.append(
                str(decision.mode.value if isinstance(decision.mode, Mode) else decision.mode)
            )
            for nid in self.storage.node_ids:
                node_hist[nid][i] = state.t_c[self.storage.index(nid)]

            if trace_callback is not None:
                temps_after = self.storage.sensor_temps(state)
                trace_callback(
                    t,
                    i,
                    StepTrace(
                        t_s=t,
                        index=i,
                        node_temps_before_c=temps_before,
                        mode=str(
                            decision.mode.value
                            if isinstance(decision.mode, Mode)
                            else decision.mode
                        ),
                        controller_state=cs,
                        setpoints=sp,
                        charging=decision.charging,
                        target_node=decision.target_node,
                        flow_setpoint_c=decision.flow_setpoint_c,
                        t_flow_eff_c=t_flow_eff,
                        p_el_w=out.p_el_w,
                        q_th_w=out.q_th_w,
                        cop_carnot_based=(
                            out.q_th_w / out.p_el_w if out.p_el_w > 1.0 else float("nan")
                        ),
                        q_in_w=q_in,
                        q_out_w=q_out,
                        node_temps_after_c=temps_after,
                    ),
                )

        return SimResult(
            time_s=time,
            p_el_w=p_el,
            q_th_w=q_th,
            mode=modes,
            node_temps_c=node_hist,
            sensor_temps_c={nid: node_hist[nid].copy() for nid in self.storage.node_ids},
        )
