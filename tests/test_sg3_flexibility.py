"""Contract tests for twin-derived SG3 feasibility, value and placement."""

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from hp_twin.adapters.eos_connect import EosContingent, planned_windows_from_status
from hp_twin.adapters.sg_orchestrator import SgPlanConfig, withdraw_contingent
from hp_twin.domain.controller import ControllerState, Mode
from hp_twin.domain.heat_source import HeatPumpAirWater
from hp_twin.domain.sg_plan import (
    SgCandidate,
    evaluate_sg3_economics,
    sg3_start_feasible_slots,
)


def _mask(
    *,
    running: np.ndarray | None = None,
    outdoor: np.ndarray | None = None,
    state: ControllerState | None = None,
    n: int = 96,
    min_on_s: float = 1800,
):
    return sg3_start_feasible_slots(
        t_buffer_c=np.full(n, 28.0),
        running_w=np.zeros(n) if running is None else running,
        outdoor_c=np.full(n, 10.0) if outdoor is None else outdoor,
        initial_controller_state=state or ControllerState(),
        heat_pump=HeatPumpAirWater(cop_min=1.0),
        controller_config=SimpleNamespace(min_off_s=1800.0, min_on_s=min_on_s),
        sg3_top_c=46.0,
        flow_offset_k=10.0,
        buffer_overshoot_k=5.0,
        running_threshold_w=200.0,
        dt_s=300.0,
        slot_s=900.0,
        allowed_start_s=0.0,
        allowed_end_s=n * 300.0,
        max_duration_s=3 * 3600.0,
    )


def test_current_min_off_lock_expires_on_a_quarter_hour_boundary():
    mask, reasons = _mask(state=ControllerState(mode=Mode.PAUSE, time_since_stop_s=600.0))
    assert mask[:3] == [False, False, True]
    assert reasons[:2] == ["min_off", "min_off"]


def test_current_run_does_not_lock_the_entire_horizon_after_predicted_stop():
    running = np.zeros(96)
    running[:6] = 3000.0  # currently running, stops at minute 30
    mask, reasons = _mask(
        running=running,
        state=ControllerState(mode=Mode.HEAT_BUFFER, time_since_stop_s=0.0),
    )
    assert reasons[0] == "baseline_running"
    assert reasons[2] == "min_off"
    assert reasons[4] == "ok"
    assert mask[4]


def test_current_run_without_predicted_stop_keeps_horizon_closed():
    mask, reasons = _mask(
        running=np.full(96, 3000.0),
        state=ControllerState(mode=Mode.HEAT_BUFFER, time_since_stop_s=0.0),
    )
    assert not any(mask)
    assert set(reasons) <= {
        "baseline_running",
        "baseline_running_during_min_on",
        "min_on_beyond_horizon",
    }


def test_min_on_and_horizon_end_are_hard_constraints():
    mask, reasons = _mask(n=6)
    assert not mask[-1]
    assert reasons[-1] == "min_on_beyond_horizon"


def test_baseline_run_blocks_start_and_required_min_on():
    running = np.zeros(96)
    running[6:12] = 3000.0  # 30..60 min
    mask, reasons = _mask(running=running)
    assert reasons[1] == "baseline_running_during_min_on"
    assert reasons[4] == "min_off"


def test_bivalence_cop_limit_blocks_cold_start():
    cold = np.full(96, 10.0)
    cold[3:9] = -30.0  # second start slot; COP model is at its lower bound
    mask, reasons = _mask(outdoor=cold)
    assert not mask[1]
    assert reasons[1] == "cop_bivalence"


def test_pure_load_shift_is_a_valid_flexibility_contingent():
    """No energy is saved, but moving the window to cheap power still has value."""
    baseline = np.array([1000.0, 1000.0, 4000.0, 4000.0])
    sg3 = np.array([4000.0, 4000.0, 1000.0, 1000.0])
    result = evaluate_sg3_economics(
        baseline,
        sg3,
        np.array([0.10, 0.10, 0.50, 0.50]),
        dt_s=900.0,
        slot_s=900.0,
        outdoor_c=np.full(4, 10.0),
        offered_wh=2000.0,
    )
    assert result["total_extra_wh"] == pytest.approx(0.0)
    assert result["offered_wh"] == pytest.approx(2000.0)
    # Both displaced 1 kWh baseline slots cost 0.50 EUR/kWh; the same energy
    # later costs 0.10 EUR/kWh. Unchanged consumption is not part of the reward.
    assert result["saving_eur"] == pytest.approx(0.6)
    assert result["avoided_eur"] == pytest.approx(0.75)
    assert result["value_eur_per_kwh"] == pytest.approx(0.375)


def test_gross_value_does_not_subtract_supply_cost_twice():
    baseline = np.array([0.0, 0.0, 4000.0, 0.0])
    sg3 = np.array([4000.0, 4000.0, 0.0, 0.0])
    result = evaluate_sg3_economics(
        baseline,
        sg3,
        np.array([0.10, 0.10, 0.50, 0.50]),
        dt_s=900.0,
        slot_s=900.0,
        outdoor_c=np.full(4, 10.0),
    )
    assert result["total_extra_wh"] == pytest.approx(1000.0)
    assert result["avoided_eur"] == pytest.approx(0.5)
    assert result["value_eur_per_kwh"] == pytest.approx(0.5)
    assert result["saving_eur"] == pytest.approx(0.3)


def test_displaced_later_baseline_run_counts_in_whole_scenario():
    baseline = np.array([0.0, 0.0, 4000.0, 0.0, 4000.0, 0.0])
    sg3 = np.array([4000.0, 4000.0, 0.0, 0.0, 0.0, 0.0])
    result = evaluate_sg3_economics(
        baseline,
        sg3,
        np.array([0.10, 0.10, 0.50, 0.50, 0.30, 0.30]),
        dt_s=900.0,
        slot_s=900.0,
        outdoor_c=np.full(6, 10.0),
        offered_wh=2000.0,
    )
    assert result["total_extra_wh"] == pytest.approx(0.0)
    assert result["offered_wh"] == pytest.approx(2000.0)
    assert result["avoided_eur"] == pytest.approx(0.8)
    assert result["saving_eur"] == pytest.approx(0.6)


def test_missing_late_prices_do_not_reject_priced_sg3_window():
    baseline = np.array([0.0, 0.0, 4000.0, 0.0])
    sg3 = np.array([4000.0, 4000.0, 0.0, 0.0])
    result = evaluate_sg3_economics(
        baseline,
        sg3,
        np.array([0.10, 0.10, 0.50, np.nan]),
        dt_s=900.0,
        slot_s=900.0,
        outdoor_c=np.full(4, 10.0),
    )
    assert result["total_extra_wh"] == pytest.approx(1000.0)
    assert result["priced_slots"] == 3
    assert result["total_slots"] == 4
    assert result["priced_baseline_fraction"] == pytest.approx(1.0)


def test_unpriced_slot_discards_its_energy_instead_of_inventing_tariff():
    baseline = np.array([0.0, 0.0, 4000.0, 0.0])
    sg3 = np.array([4000.0, 4000.0, 0.0, 0.0])
    with pytest.raises(ValueError, match="no fully priced energy"):
        evaluate_sg3_economics(
            baseline,
            sg3,
            np.array([0.10, 0.10, np.nan, np.nan]),
            dt_s=900.0,
            slot_s=900.0,
            outdoor_c=np.full(4, 10.0),
        )


def test_contingent_serializes_explicit_contract():
    payload = EosContingent(
        3200,
        2.5,
        feasible=[True] * 192,
        value_eur_per_kwh=0.25,
        max_power_w=3600,
        start="2026-01-13T00:00:00+00:00",
        ttl_minutes=45,
        min_runtime_slots=10,
        contiguous=True,
    ).to_payload()
    assert payload["start"] == "2026-01-13T00:00:00+00:00"
    assert payload["resolution"] == "15min"
    assert len(payload["feasible"]) == 192
    assert payload["value_eur_per_kwh"] == 0.25
    assert payload["max_power_w"] == 3600
    assert payload["ttl_minutes"] == 45
    assert payload["min_runtime_slots"] == 10
    assert payload["contiguous"] is True


def test_two_hour_contingent_keeps_its_full_operating_window():
    contingent = EosContingent(
        3200,
        12.0,
        min_runtime_slots=8,
        contiguous=True,
    )
    assert contingent.to_payload()["min_runtime_slots"] == 8


def test_sg3_deadline_keeps_the_last_full_slot_available():
    candidate = SgCandidate(
        start_s=1800.0,
        end_s=9000.0,
        energy_wh=2400.0,
        max_power_w=3600.0,
        reason="test",
    )
    assert candidate.deadline_hours() == pytest.approx(2.5)
    contingent = EosContingent(
        candidate.energy_wh,
        candidate.deadline_hours(),
        start="2026-01-13T10:30:00+00:00",
        deadline="2026-01-13T13:00:00+00:00",
        min_runtime_slots=8,
        contiguous=True,
    )
    assert contingent.to_payload()["deadline"] == "2026-01-13T13:00:00+00:00"
    assert "deadline_hours" not in contingent.to_payload()


def test_multiple_placement_blocks_use_midnight_anchor():
    status = {
        "plan_anchor": "2026-01-13T00:00:00+00:00",
        "time_frame_base": 900,
        "fallback_active": False,
        "plan": [100.0, 100.0, 0.0, 200.0] + [0.0] * 188,
    }
    assert planned_windows_from_status(
        status, pd.Timestamp("2026-01-13T00:00:00Z"), 48 * 3600.0
    ) == [(0.0, 1800.0), (2700.0, 3600.0)]


def test_minute_aligned_forecast_uses_containing_eos_slot():
    """A forecast beginning at 19:36 owns EOS slot 19:30, not slot 19:45."""
    status = {
        "plan_anchor": "2026-01-13T00:00:00+00:00",
        "time_frame_base": 900,
        "fallback_active": False,
        "plan": [0.0] * 80 + [400.0, 400.0] + [0.0] * 110,
    }
    assert planned_windows_from_status(
        status, pd.Timestamp("2026-01-13T19:36:00Z"), 48 * 3600.0
    ) == [(1800.0, 3600.0)]


def test_placement_without_anchor_is_refused():
    with pytest.raises(ValueError, match="plan_anchor"):
        planned_windows_from_status(
            {"plan": [100.0], "time_frame_base": 900},
            pd.Timestamp("2026-01-13T00:00:00Z"),
            48 * 3600.0,
        )


def test_withdrawal_zeroes_contingent(monkeypatch: pytest.MonkeyPatch):
    calls = []

    def _push(base_url, load_id, contingent, token=None):
        calls.append(contingent)
        return {"accepted": True}

    monkeypatch.setattr("hp_twin.adapters.sg_orchestrator.push_contingent", _push)
    withdraw_contingent("http://eos", SgPlanConfig(), None, pd.Timestamp("2026-01-13T00:00:00Z"))
    assert calls[0].total_wh == 0.0
    assert calls[0].ttl_minutes == 1.0
