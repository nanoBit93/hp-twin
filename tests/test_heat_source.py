"""WP-Kennfeld: Temperaturabhängigkeit innerhalb eines Takts (P_el steigt, COP/Q_th fallen)."""

from __future__ import annotations

from hp_twin.domain.heat_source import HeatPumpAirWater


def test_cop_falls_with_flow_temp() -> None:
    wp = HeatPumpAirWater()
    assert wp.cop(5.0, 30.0) > wp.cop(5.0, 45.0)  # höherer Hub -> kleinerer COP


def test_power_rises_with_flow_temp() -> None:
    wp = HeatPumpAirWater()
    p_low = wp.output(5.0, 32.0, active=True).p_el_w
    p_high = wp.output(5.0, 45.0, active=True).p_el_w
    assert p_high > p_low  # Inverter moduliert mit steigender Vorlauftemp hoch


def test_qth_consistent_over_cycle() -> None:
    """Über den Takt: P_el steigt, COP fällt -> Q_th bleibt ~konstant (keine Überschätzung).

    Genau dieser Ausgleich verhindert, dass bei konstantem P_el × COP>1 zum Taktende zu viel
    thermische Wärme angesetzt wird.
    """
    wp = HeatPumpAirWater()
    q_start = wp.output(3.0, 32.0, active=True).q_th_w
    q_end = wp.output(3.0, 46.0, active=True).q_th_w
    assert abs(q_end - q_start) / q_start < 0.2  # im Band, nicht entkoppelt
    # und Q_th folgt P_comp*COP exakt
    assert q_end == wp.compressor_power_w(3.0, 46.0) * wp.cop(3.0, 46.0)


def test_standby_when_off() -> None:
    wp = HeatPumpAirWater(standby_w=80.0)
    out = wp.output(5.0, 40.0, active=False)
    assert out.p_el_w == 80.0 and out.q_th_w == 0.0


def test_power_bounds() -> None:
    wp = HeatPumpAirWater(p_comp_min_w=800, p_comp_max_w=5500)
    # extrem hoher Vorlauf -> Deckelung
    assert wp.compressor_power_w(5.0, 200.0) == 5500.0
    assert wp.compressor_power_w(40.0, 10.0) == 800.0
