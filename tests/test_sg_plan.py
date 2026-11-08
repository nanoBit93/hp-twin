"""SG-Plan-Domain: Masken, Blocks, Strip zur SG2-Basis, Injection, Kandidaten-Fenster."""

from __future__ import annotations

import numpy as np
import pytest

from hp_twin.domain.sg_plan import (
    blocks,
    evaluate_candidate,
    inject_sg3_start,
    sg3_candidates,
    sg3_mask,
    strip_sg3_setpoints,
)


def test_sg3_mask_prefers_status_sensor() -> None:
    hk1 = np.full(10, 32.0)
    stat = np.zeros(10)
    stat[4:7] = 3.0
    m = sg3_mask(stat, hk1, dilate=0)
    assert m[[4, 5, 6]].all() and not m[:4].any() and not m[7:].any()


def test_sg3_mask_falls_back_to_setpoint_threshold() -> None:
    hk1 = np.full(8, 32.0)
    hk1[2] = 45.0
    hk1[3] = 44.0
    m = sg3_mask(None, hk1, sg3_min_setpoint_c=38.0, dilate=0)
    assert m[2] and m[3] and not m[0] and not m[5]


def test_sg3_mask_unions_and_dilates() -> None:
    """Sensorhänger am Block-Ende (Status 2, Sollwert noch hoch) wird mitgenommen."""
    hk1 = np.full(10, 32.0)
    hk1[4:8] = 48.0
    stat = np.full(10, 2.0)
    # Status schaltet 1 Schritt zu früh zurück: Soll ist noch hoch, Status schon 2
    stat[7] = 2.0
    m = sg3_mask(stat, hk1, dilate=0)
    assert m[4:8].all()  # Soll-Trigger sichert das Blockende
    m2 = sg3_mask(stat, hk1, dilate=2)
    assert m2[2] and m2[9] and not m2[0]  # Dilatation greift, Frame-Rand bleibt sauber


def test_blocks_matches_contiguous_ranges() -> None:
    m = np.array([0, 0, 1, 1, 1, 0, 0, 1, 0], dtype=bool)
    assert blocks(m) == [(2, 4), (7, 7)]
    assert blocks(np.zeros(4, dtype=bool)) == []


def test_strip_restores_sg2_via_interpolation() -> None:
    # Base SG2 = 32 °C; SG3-Block 30..43 auf 44
    hk1 = np.full(96, 32.0)
    hk1[30:44] = 44.0
    dhw = np.full(96, 40.0)
    dhw[30:44] = 45.0
    room = np.full(96, 20.5)
    room[30:44] = 22.5
    m = sg3_mask(None, hk1)
    out2 = strip_sg3_setpoints(
        {"setpoint_hk1_c": hk1, "setpoint_dhw_c": dhw, "setpoint_room_wz_c": room}, mask=m
    )
    # Alle drei Spuren flach auf SG2-Niveau nach dem Strip
    for key, level in (
        ("setpoint_hk1_c", 32.0),
        ("setpoint_dhw_c", 40.0),
        ("setpoint_room_wz_c", 20.5),
    ):
        assert out2[key].max() == pytest.approx(level, abs=1e-6), key
        assert out2[key].min() == pytest.approx(level, abs=1e-6), key


def test_strip_block_at_frame_edge_fills_constant() -> None:
    hk1 = np.full(24, 32.0)
    hk1[:6] = 46.0  # Block am Anfang: kein linker Rand
    out = strip_sg3_setpoints({"setpoint_hk1_c": hk1}, mask=sg3_mask(None, hk1))
    assert out["setpoint_hk1_c"].max() == pytest.approx(32.0)


def test_strip_keeps_unpowered_rooms_off() -> None:
    room = np.full(24, 5.0)  # nie beheizt
    room[8:12] = 7.0  # kleiner Boost über floor
    hk1 = np.full(24, 32.0)
    hk1[8:12] = 44.0
    out = strip_sg3_setpoints(
        {"setpoint_room_keller_c": room, "setpoint_hk1_c": hk1}, mask=sg3_mask(None, hk1)
    )
    assert out["setpoint_room_keller_c"][8:12].max() == pytest.approx(5.0)


def test_inject_ramps_to_top_and_returns_to_base() -> None:
    base = np.full(60, 32.0)
    inj = inject_sg3_start(base, 20, 40, sg3_top_c=46.0)
    # Anstieg im ersten Drittel des Fensters, Plateau danach
    assert inj[19] == pytest.approx(32.0)
    assert inj[20] > 32.0  # erster Schritt schon über Basis
    assert inj[-21] == pytest.approx(46.0)  # Plateau ab ~2/3
    assert inj[40] == pytest.approx(32.0)  # nach Fenstende wieder Basis
    assert inj.max() == pytest.approx(46.0)


def test_inject_respects_bounds() -> None:
    base = np.full(10, 30.0)
    inj = inject_sg3_start(base, 8, 25, sg3_top_c=42.0)  # end über n -> geclippt
    assert inj.max() == pytest.approx(42.0)
    assert inj[9] == pytest.approx(42.0) and inj.shape == (10,)


def test_sg3_candidates_basic_window() -> None:
    n = 96  # 96 min
    dt = 60.0
    running = np.zeros(n)
    t_buf = np.full(n, 28.0)
    sp_buf = np.full(n, 31.0)
    cands = sg3_candidates(
        t_buf,
        sp_buf,
        buffer_overshoot_k=7.0,
        running_w=running,
        running_threshold_w=200.0,
        min_off_s=1800.0,
        dt_s=dt,
        allowed_start_s=8 * 3600.0,  # außerhalb -> kein Kandidat
        allowed_end_s=15 * 3600.0,
        sg3_top_c=46.0,
    )
    assert cands == []  # Fenster 8-15h liegt hinter dem 96-min-Datensatz -> leer

    # Innerhalb: Fenster 10-50 min und Puffer weit unter Top -> ein Kandidat-Abschnitt
    cands2 = sg3_candidates(
        t_buf,
        sp_buf,
        buffer_overshoot_k=7.0,
        running_w=running,
        running_threshold_w=200.0,
        min_off_s=1800.0,
        dt_s=dt,
        allowed_start_s=10 * 60.0,
        allowed_end_s=50 * 60.0,
        sg3_top_c=46.0,
    )
    assert cands2 == [(10 * 60.0, 50 * 60.0)]


def test_sg3_candidates_respects_min_off_lock() -> None:
    n = 120
    dt = 60.0
    running = np.zeros(n, dtype=float)
    running[30:40] = 3000.0  # Lauf 30..39, Stopp bei 40 -> Lock bis 70 (min_off 30 min)
    t_buf = np.full(n, 28.0)
    cands = sg3_candidates(
        t_buf,
        np.full(n, 31.0),
        buffer_overshoot_k=7.0,
        running_w=running,
        running_threshold_w=200.0,
        min_off_s=30 * 60.0,
        dt_s=dt,
        allowed_start_s=0.0,
        allowed_end_s=(n - 1) * dt,
        sg3_top_c=46.0,
    )
    # Vor dem Lauf (0..30 min) ist thermisch ok; nach dem Stopp (min 40) lockt min_off
    # bis min 70. Kein Kandidat darf die Lock-Zone 40..70 schneiden.
    assert [c for c in cands if c[0] == 0.0] == [(0.0, 30.0 * 60.0)]
    assert all(lo >= 70 * 60.0 for lo, _ in cands if lo > 0)


def test_evaluate_candidate_pv_threshold() -> None:
    slot = 900.0
    solar = np.full(16, 100.0)
    solar[2] = 2600.0  # Spitze im Fenster 0..4
    c = evaluate_candidate(0.0, 4 * slot, 6500.0, 4200.0, slot, solar, min_pv_surplus_w=2500.0)
    assert c is not None and "pv_surplus" in c.reason and c.deadline_hours() == 1.0


def test_evaluate_candidate_below_threshold_is_none() -> None:
    slot = 900.0
    solar = np.full(8, 500.0)
    price = np.full(8, 0.45)
    c = evaluate_candidate(
        0.0, 7200.0, 6500.0, 4200.0, slot, solar, 2500.0, price, max_price_eur=0.30
    )
    assert c is None  # weder PV noch Preis erfüllt


def test_evaluate_candidate_price_fallback() -> None:
    slot = 900.0
    solar = np.full(4, 100.0)
    price = np.array([0.42, 0.11, 0.12, 0.13])
    c = evaluate_candidate(
        0.0, 3600.0, 6500.0, 4200.0, slot, solar, 2500.0, price, max_price_eur=0.30
    )
    assert c is not None and "price" in c.reason


def test_evaluate_candidate_without_market_signal_is_none() -> None:
    c = evaluate_candidate(0.0, 3600.0, 6500.0, 4200.0, 900.0, None, 2500.0)
    assert c is None  # kein blindes SG3 ohne Markt-Signal
