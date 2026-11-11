"""Regler-Logik: Warmwasservorrang, Mindeststillstand, Mindestlaufzeit."""

from __future__ import annotations

from hp_twin.domain.controller import Controller, ControllerState, Mode, Setpoints
from hp_twin.domain.topology import ControllerConfig


def _ctrl(min_off: float = 1800, min_on: float = 300) -> Controller:
    return Controller(
        config=ControllerConfig(min_off_s=min_off, min_on_s=min_on),
        dhw_node="dhw",
        buffer_node="buffer",
    )


def test_dhw_priority_over_buffer() -> None:
    c = _ctrl()
    st = ControllerState()
    sp = Setpoints(dhw_c=45, buffer_c=32)
    temps = {"dhw": 30.0, "buffer": 20.0}  # beide unter Schwelle
    st, d = c.decide(st, temps, sp, 60.0)
    assert d.mode is Mode.HEAT_DHW and d.target_node == "dhw"


def test_min_off_blocks_restart() -> None:
    c = _ctrl(min_off=1800)
    st = ControllerState(mode=Mode.PAUSE, time_since_stop_s=0.0)
    sp = Setpoints(dhw_c=60, buffer_c=32)  # dhw hoch -> nur Puffer könnte wollen
    temps = {"dhw": 60.0, "buffer": 20.0}
    # direkt nach Stopp: Pause, kein Start
    st, d = c.decide(st, temps, sp, 60.0)
    assert d.charging is False and d.mode is Mode.PAUSE
    # Zeit über min_off vorspulen
    st.time_since_stop_s = 1800.0
    st, d = c.decide(st, temps, sp, 60.0)
    assert d.charging is True and d.mode is Mode.HEAT_BUFFER


def test_sg2_stop_does_not_activate_the_sg3_completion_latch() -> None:
    c = _ctrl(min_off=0)
    st = ControllerState(mode=Mode.HEAT_BUFFER, time_in_mode_s=600.0, active_node="buffer")
    sp = Setpoints(dhw_c=60.0, buffer_c=32.0, buffer_overshoot_k=2.0)
    st, d = c.decide(st, {"dhw": 60.0, "buffer": 34.0}, sp, 60.0)
    assert not d.charging and not st.sg3_target_reached
    st, d = c.decide(st, {"dhw": 60.0, "buffer": 20.0}, sp, 60.0)
    assert d.charging and d.mode is Mode.HEAT_BUFFER


def test_min_on_prevents_early_stop() -> None:
    c = _ctrl(min_on=600)
    st = ControllerState(mode=Mode.HEAT_BUFFER, time_in_mode_s=60.0, active_node="buffer")
    sp = Setpoints(dhw_c=60, buffer_c=32, buffer_overshoot_k=10)
    temps = {"dhw": 60.0, "buffer": 50.0}  # Ziel längst erreicht
    st, d = c.decide(st, temps, sp, 60.0)
    assert d.charging is True  # min_on noch nicht erfüllt -> weiter laden
    st.time_in_mode_s = 600.0
    st, d = c.decide(st, temps, sp, 60.0)
    assert d.charging is False  # jetzt Stopp
