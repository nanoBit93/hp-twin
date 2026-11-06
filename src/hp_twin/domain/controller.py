"""Regelbasierter Zwilling des Wärmepumpenmanagers (WPM). Keine ML, nur Logik.

Zustandsautomat: IDLE -> (HEAT_DHW | HEAT_BUFFER) -> PAUSE -> IDLE.
- Start, wenn ein Fühler unter (Soll - Hysterese) fällt. Warmwasservorrang bei Gleichstand.
- Stopp, wenn Ladeziel (Soll + Overshoot) erreicht und Mindestlaufzeit erfüllt.
- Nach Stopp mindestens min_off_s Pause (PAUSE), bevor neu gestartet werden darf.
- Sollwerte sind Zeitreihen-Eingang (SetpointInput) -> SG3 später nur weiterer Input.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from hp_twin.domain.topology import ControllerConfig


class Mode(StrEnum):
    IDLE = "idle"
    HEAT_BUFFER = "heat"
    HEAT_DHW = "dhw"
    PAUSE = "pause"


@dataclass(frozen=True)
class Setpoints:
    """Aktuelle Sollwerte (aus Zeitreihe) und Ladeziel-Overshoot je Zone."""

    dhw_c: float
    buffer_c: float
    dhw_hysteresis_k: float = 5.0
    buffer_hysteresis_k: float = 5.0
    dhw_overshoot_k: float = 2.0
    buffer_overshoot_k: float = 10.0


@dataclass
class ControllerState:
    mode: Mode = Mode.IDLE
    time_in_mode_s: float = 0.0
    time_since_stop_s: float = 1e9  # groß -> darf sofort starten
    active_node: str | None = None


@dataclass(frozen=True)
class ControlDecision:
    mode: Mode
    charging: bool
    target_node: str | None  # Speicherknoten, der geladen wird
    flow_setpoint_c: float


@dataclass(frozen=True)
class Controller:
    """WPM-Logik. dhw_node/buffer_node benennen die geladenen Speicherknoten."""

    config: ControllerConfig
    dhw_node: str
    buffer_node: str
    buffer_charge_offset_k: float = 10.0  # Puffer wird über VL-Soll hinaus geladen
    dhw_charge_offset_k: float = 5.0

    def _wants(self, t: float, setpoint: float, hysteresis: float) -> bool:
        return t < setpoint - hysteresis

    def _reached(self, t: float, setpoint: float, overshoot: float) -> bool:
        return t >= setpoint + overshoot

    def decide(
        self, state: ControllerState, temps: dict[str, float], sp: Setpoints, dt_s: float
    ) -> tuple[ControllerState, ControlDecision]:
        state.time_in_mode_s += dt_s
        if state.mode in (Mode.IDLE, Mode.PAUSE):
            state.time_since_stop_s += dt_s

        t_dhw = temps[self.dhw_node]
        t_buf = temps[self.buffer_node]
        min_off = self.config.min_off_s

        def _transition(new: Mode, node: str | None) -> ControllerState:
            if new != state.mode:
                state.mode = new
                state.time_in_mode_s = 0.0
            state.active_node = node
            return state

        # --- laufende Ladung: Stopp prüfen ---
        if state.mode == Mode.HEAT_DHW:
            done = self._reached(t_dhw, sp.dhw_c, sp.dhw_overshoot_k)
            if done and state.time_in_mode_s >= self.config.min_on_s:
                state = _transition(Mode.PAUSE, None)
                state.time_since_stop_s = 0.0
            else:
                return state, self._charge_decision(Mode.HEAT_DHW, self.dhw_node, sp)
        elif state.mode == Mode.HEAT_BUFFER:
            done = self._reached(t_buf, sp.buffer_c, sp.buffer_overshoot_k)
            if done and state.time_in_mode_s >= self.config.min_on_s:
                state = _transition(Mode.PAUSE, None)
                state.time_since_stop_s = 0.0
            else:
                return state, self._charge_decision(Mode.HEAT_BUFFER, self.buffer_node, sp)

        # --- PAUSE: warten bis min_off ---
        if state.mode == Mode.PAUSE:
            if state.time_since_stop_s < min_off:
                return state, ControlDecision(Mode.PAUSE, False, None, 0.0)
            state = _transition(Mode.IDLE, None)

        # --- IDLE: Startbedarf prüfen (Warmwasservorrang) ---
        want_dhw = self._wants(t_dhw, sp.dhw_c, sp.dhw_hysteresis_k)
        want_buf = self._wants(t_buf, sp.buffer_c, sp.buffer_hysteresis_k)
        can_start = state.time_since_stop_s >= min_off
        if can_start and want_dhw:
            state = _transition(Mode.HEAT_DHW, self.dhw_node)
            return state, self._charge_decision(Mode.HEAT_DHW, self.dhw_node, sp)
        if can_start and want_buf:
            state = _transition(Mode.HEAT_BUFFER, self.buffer_node)
            return state, self._charge_decision(Mode.HEAT_BUFFER, self.buffer_node, sp)

        state = _transition(Mode.IDLE, None)
        return state, ControlDecision(Mode.IDLE, False, None, 0.0)

    def _charge_decision(self, mode: Mode, node: str, sp: Setpoints) -> ControlDecision:
        if mode == Mode.HEAT_DHW:
            flow = sp.dhw_c + self.dhw_charge_offset_k
        else:
            flow = sp.buffer_c + self.buffer_charge_offset_k
        return ControlDecision(mode, True, node, flow)
