import pandas as pd

from hp_twin.adapters.ha_history import HaMapping, build_frame


def _mapping() -> HaMapping:
    return HaMapping(
        sensors={
            "t_outdoor_c": "sensor.out",
            "ghi_wm2": "sensor.ghi",
            "p_el_w": "sensor.pel",
            "t_flow_c": "sensor.vl",
            "t_return_c": "sensor.rl",
            "t_dhw_c": "sensor.dhw",
            "t_hk1_c": "sensor.puf",
            "t_hk2_c": "sensor.hk2",
            "setpoint_dhw_c": "sensor.sdhw",
            "setpoint_hk1_c": "sensor.shk1",
            "setpoint_hk2_c": "sensor.shk2",
            "flow_rate_lpm": "sensor.flow",
        },
        rooms={"wohnen": "sensor.room"},
        derived={
            "compressor_on": "binary_sensor.comp",
            "mode": {"heat": "binary_sensor.heat", "dhw": "binary_sensor.dhw"},
            "circuit_active": {"hk1": "binary_sensor.puf_pump", "dhw": "binary_sensor.dhw_pump"},
        },
    )


def _const(entity_value: dict[str, object], grid: pd.DatetimeIndex) -> dict[str, pd.Series]:
    return {e: pd.Series([v], index=grid[:1]) for e, v in entity_value.items()}


def test_build_frame_modes_and_qth() -> None:
    grid = pd.date_range("2026-01-10", periods=4, freq="1min", tz="UTC")
    hist = _const(
        {
            "sensor.out": 5.0,
            "sensor.ghi": 100,
            "sensor.pel": 3500,
            "sensor.vl": 40.0,
            "sensor.rl": 35.0,
            "sensor.dhw": 50,
            "sensor.puf": 32,
            "sensor.hk2": 26,
            "sensor.sdhw": 45,
            "sensor.shk1": 31,
            "sensor.shk2": 28,
            "sensor.flow": 12.0,
            "sensor.room": 21,
            "binary_sensor.comp": "on",
            "binary_sensor.heat": "on",
            "binary_sensor.dhw": "off",
            "binary_sensor.puf_pump": "on",
            "binary_sensor.dhw_pump": "off",
        },
        grid,
    )
    df = build_frame(hist, _mapping(), grid)
    assert (df["mode"] == "heat").all()
    assert (df["circuit_active"] == "hk1").all()
    # Q_th = (12/60 L/s) * 4186 J/(L*K) * 5K = 4186 W; über 60 s = 69.8 Wh
    assert abs(df["q_heat_wh"].iloc[0] - 69.8) < 1.0
    assert (df["compressor_on"] == 1).all()


def test_qth_zero_when_off() -> None:
    grid = pd.date_range("2026-01-10", periods=2, freq="1min", tz="UTC")
    hist = _const(
        {
            k: 0
            for k in [
                "sensor.out",
                "sensor.ghi",
                "sensor.pel",
                "sensor.dhw",
                "sensor.puf",
                "sensor.hk2",
                "sensor.sdhw",
                "sensor.shk1",
                "sensor.shk2",
                "sensor.room",
            ]
        }
        | {"sensor.vl": 40.0, "sensor.rl": 30.0, "sensor.flow": 12.0, "binary_sensor.comp": "off"},
        grid,
    )
    df = build_frame(hist, _mapping(), grid)
    assert (df["q_heat_wh"] == 0).all()
    assert (df["mode"] == "idle").all()


def test_scales_applied() -> None:
    grid = pd.date_range("2026-01-10", periods=2, freq="1min", tz="UTC")
    m = _mapping()
    m = HaMapping(sensors=m.sensors, rooms=m.rooms, derived=m.derived, scales={"p_el_w": 1000.0})
    hist = _const(
        {
            k: 0
            for k in [
                "sensor.out",
                "sensor.ghi",
                "sensor.vl",
                "sensor.rl",
                "sensor.dhw",
                "sensor.puf",
                "sensor.hk2",
                "sensor.sdhw",
                "sensor.shk1",
                "sensor.shk2",
                "sensor.flow",
                "sensor.room",
            ]
        }
        | {"sensor.pel": 2.5, "binary_sensor.comp": "off"},
        grid,
    )
    df = build_frame(hist, m, grid)
    assert (df["p_el_w"] == 2500.0).all()
