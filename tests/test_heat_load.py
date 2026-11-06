from hp_twin.domain.heat_load import BuildingHeatLoad, RoomHeatLoad


def test_ua_from_design_load() -> None:
    r = RoomHeatLoad(name="wz", design_heat_load_w=1200, design_room_temp_c=20)
    assert abs(r.ua_w_per_k(-10.0) - 1200 / 30) < 1e-9  # 40 W/K


def test_total_and_active_ua() -> None:
    b = BuildingHeatLoad(
        design_outdoor_c=-10,
        rooms=[
            RoomHeatLoad(name="wz", design_heat_load_w=1200, design_room_temp_c=20),
            RoomHeatLoad(name="sz", design_heat_load_w=600, design_room_temp_c=20),
        ],
    )
    assert abs(b.summed_room_load_w - 1800) < 1e-9
    assert abs(b.total_ua_w_per_k() - (1200 + 600) / 30) < 1e-9
    assert abs(b.total_ua_w_per_k(only={"wz"}) - 1200 / 30) < 1e-9  # nur aktive Räume
