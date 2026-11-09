"""SG-Orchestrator: Basis->Kandidat->Contingent(->EOS-Fake)->Placement->Injektion."""

from __future__ import annotations

import dataclasses
from types import SimpleNamespace
from typing import Any

import numpy as np
import pandas as pd
import pytest

from hp_twin.adapters.eos_connect import EosContingent
from hp_twin.adapters.sg_orchestrator import (
    Sg3CycleState,
    SgPlanConfig,
    SgPlanResult,
    _adopt_placement,
    allowed_windows_s,
    push_and_collect_placement,
    run_sg_cycle,
    simulate_sg3_run,
)
from scripts_helper import synthetic_frame


class FakeForecastResult:
    def __init__(self, n: int = 576, p_on: int = 1000) -> None:
        self.deterministic_p_el_w = np.zeros(n)
        self.deterministic_p_el_w[400:450] = 2800.0  # Basis-Lauf morgens
        self.buffer_sim_c = np.full(n, 28.0)
        self.buffer_sim_c[400:450] = 38.0  # ladend
        self.controller_mode = ["idle"] * n
        self.controller_mode[400:450] = ["heat"] * 50
        # Event im 600..7200-s-Fenster (für simulate_sg3_run): 600..6600 s @ 60 s
        self.events: list[Any] = [_Event(600.0, 6600.0, "heat", 3200.0)]
        self.outdoor_c = np.full(n, 8.0)
        from hp_twin.domain.heat_source import HeatPumpAirWater

        self.heat_pump = HeatPumpAirWater()
        self.dhw_sim_c = np.full(n, 45.0)
        self.frame_start = pd.Timestamp("2026-01-13T00:00:00Z")
        self.eos = SimpleNamespace(
            thermal={
                "start": self.frame_start.isoformat(),
                "resolution": "15min",
                "series": {},
                "plan": {},
                "run_probability": [0.0, 0.2, 1.0],
            }
        )


class _Event:
    def __init__(self, start_s: float, end_s: float, mode: str, mean_w: float) -> None:
        self.start_s = start_s
        self.end_s = end_s
        self.mode = mode
        self.mean_p_el_w = mean_w


class FakeForecastFn:
    """Zählt Aufrufe; simuliert Injektion: sg3-kwargs -> Lauf am Frame-Raster (dt aus df)."""

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def __call__(self, df: Any, topology: Any, calib: Any, **kw: Any) -> FakeForecastResult:
        self.calls.append(kw)
        windows = kw.get("sg3_windows")
        if windows is None and "sg3_start_s" in kw:
            windows = [(kw["sg3_start_s"], kw["sg3_end_s"])]
        if windows is not None:
            dt_s = float(df.index.to_series().diff().dropna().iloc[0].total_seconds())
            res = FakeForecastResult()
            for start_s, end_s in windows:
                i0 = int(start_s / dt_s)
                i1 = min(len(res.deterministic_p_el_w), int(end_s / dt_s))
                res.deterministic_p_el_w[i0:i1] = 3200.0
                res.controller_mode[i0:i1] = ["heat"] * (i1 - i0)
                stop = min(i0 + 30, i1)
                target = float(kw.get("sg3_top_c") or 46.0) + 1.6
                res.buffer_sim_c[i0:i1] = np.where(
                    np.arange(i1 - i0) < stop - i0,
                    28.0 + (target - 28.0) * np.arange(i1 - i0) / max(stop - i0 - 1, 1),
                    target,
                )
                # Target reached in 30 minutes; the setpoint window may still
                # be longer and must not change the actual compressor runtime.
                res.deterministic_p_el_w[stop:] = 0.0
                res.controller_mode[stop:] = ["idle"] * len(res.controller_mode[stop:])
                res.buffer_sim_c[stop:] = 47.0
                if stop == i1:
                    res.buffer_sim_c[i1 - 1] = target
                    res.controller_mode[i1 - 1] = "heat"
            return res
        return FakeForecastResult()


class FakeEos:
    def __init__(self) -> None:
        self.pushed: list[EosContingent] = []
        self.plan = [0.0] * 192
        self.anchor = "2026-01-13T00:00:00+00:00"
        # One uninterrupted one-hour SG3 window; no later isolated slots.
        self.plan[0:4] = [3500.0] * 4

    def push_contingent(
        self, base_url: str, load_id: str, c: EosContingent, token=None, timeout_s=15.0
    ):
        self.pushed.append(c)
        return {
            "kind": "contingent",
            "total_wh": c.total_wh,
            "deadline_slot": None,
            "valid_until": "2026-11-09T00:00:00+00:00",
            "accepted": True,
        }

    def read_load_status(self, base_url: str, load_id: str, token=None, timeout_s=15.0):
        chosen = self.pushed[-1] if self.pushed else None
        withdrawn = chosen is not None and chosen.total_wh == 0.0
        slots = max(1, int(round(chosen.min_runtime_slots or 1))) if chosen else 4
        plan = [0.0] * 192
        if not withdrawn:
            plan[0:slots] = [3500.0] * slots
        return {
            "id": load_id,
            "kind": "contingent",
            "planned_wh": float(sum(plan)),
            "plan": plan,
            "release": {"state": "blocked"},
            "fallback_active": False,
            "plan_anchor": self.anchor,
            "time_frame_base": 900,
        }


def _df() -> pd.DataFrame:
    from hp_twin.adapters.io_csv import validate_frame

    return validate_frame(synthetic_frame(days=3))


@pytest.fixture()
def monkey_eos(monkeypatch: pytest.MonkeyPatch) -> FakeEos:
    fake = FakeEos()
    import hp_twin.adapters.sg_orchestrator as orch

    monkeypatch.setattr(orch, "push_contingent", fake.push_contingent)
    monkeypatch.setattr(orch, "read_load_status", fake.read_load_status)
    return fake


def _market(solar: list[float] | None, price: list[float] | None = None) -> Any:
    from hp_twin.adapters.evcc import MarketSeries

    n = len(solar) if solar else 192
    idx = pd.date_range("2026-01-13T00:00:00Z", periods=max(n, 192), freq="15min", tz="UTC")
    return MarketSeries(
        index=idx,
        solar_w=np.asarray(solar if solar is not None else [0.0] * n, dtype=float),
        price=np.asarray(price if price is not None else [0.4] * n, dtype=float),
        feed_in=None,
        battery_soc_kwh=None,
        battery_max_power_w=None,
    )


def test_all_future_windows_are_returned_in_order() -> None:
    cfg = SgPlanConfig(windows_utc_h=[(7, 8), (10, 14)])
    windows = allowed_windows_s(pd.Timestamp("2026-11-08 20:00:00+00:00"), cfg, 48 * 3600.0)
    assert windows == [
        (11 * 3600.0, 12 * 3600.0),
        (14 * 3600.0, 18 * 3600.0),
        (35 * 3600.0, 36 * 3600.0),
        (38 * 3600.0, 42 * 3600.0),
    ]


def test_push_and_collect_with_fake_eos(monkey_eos: FakeEos) -> None:
    from hp_twin.domain.sg_plan import SgCandidate

    cand = SgCandidate(0.0, 7200.0, 6500.0, 4200.0, "pv_surplus>=2500W")
    cfg = SgPlanConfig(enabled=True, dry_run=False)
    pushed, slots, status, notes = push_and_collect_placement(
        cand, cfg, "http://eos:8080", None, last_total_wh=None
    )
    assert pushed and len(monkey_eos.pushed) == 1
    assert monkey_eos.pushed[0].total_wh == pytest.approx(6500.0)
    assert slots == [(i, v) for i, v in enumerate(status["plan"]) if v > 0]
    assert status.get("planned_wh") == pytest.approx(3500.0 * 8)
    assert any("akzeptiert" in n for n in notes)


def test_push_hysteresis_skips_second_push(monkey_eos: FakeEos) -> None:
    from hp_twin.domain.sg_plan import SgCandidate

    cand = SgCandidate(0.0, 7200.0, 6500.0, 4200.0, "pv_surplus>=2500W")
    cfg = SgPlanConfig(enabled=True, dry_run=False)
    state = Sg3CycleState()
    push_and_collect_placement(
        cand,
        cfg,
        "http://eos:8080",
        None,
        last_total_wh=None,
        frame_start=pd.Timestamp("2026-01-13T00:00:00Z"),
        state=state,
    )
    pushed2, _, _, notes = push_and_collect_placement(
        cand,
        cfg,
        "http://eos:8080",
        None,
        last_total_wh=6400.0,
        frame_start=pd.Timestamp("2026-01-13T00:00:00Z"),
        state=state,
    )
    assert not pushed2
    assert any("Hysterese" in n for n in notes)
    assert len(monkey_eos.pushed) == 1


def test_hysteresis_still_reads_existing_placement(monkey_eos: FakeEos) -> None:
    from hp_twin.domain.sg_plan import SgCandidate

    cand = SgCandidate(0.0, 7200.0, 6500.0, 4200.0, "unchanged offer")
    cfg = SgPlanConfig(enabled=True, dry_run=False)
    state = Sg3CycleState(offered_frame_start_iso="2026-01-13T00:00:00+00:00")
    pushed, slots, status, notes = push_and_collect_placement(
        cand,
        cfg,
        "http://eos:8080",
        None,
        last_total_wh=6400.0,
        frame_start=pd.Timestamp("2026-01-13T00:00:00Z"),
        state=state,
    )

    assert not pushed
    assert slots
    assert status["planned_wh"] > 0, (pushed, slots, status, notes)
    assert not monkey_eos.pushed


def test_unchanged_offer_adopts_placement_without_another_push(
    monkey_eos: FakeEos,
) -> None:
    """Hysterese verhindert nur den Push, nicht die Adoption des fertigen Plans."""
    from hp_twin.adapters.evcc import MarketSeries
    from hp_twin.adapters.sg_orchestrator import run_sg_cycle

    idx = pd.date_range("2026-01-13T00:00:00Z", periods=192, freq="15min", tz="UTC")
    market = MarketSeries(idx, np.zeros(192), np.full(192, 0.3), None, None, None)
    state = Sg3CycleState(
        pending=True,
        offered_total_wh=800.0,
        offered_duration_s=3600.0,
        offered_frame_start_iso="2026-01-13T00:00:00+00:00",
    )
    _, res = run_sg_cycle(
        _df(),
        _topo(),
        _calib(),
        _model(),
        FakeForecastFn(),
        market=market,
        cfg=SgPlanConfig(enabled=True, dry_run=False, windows_utc_h=[(0, 24)]),
        eos_base_url="http://eos",
        state=state,
    )

    assert res.sg3_injected, res.notes
    assert res.contingency_withdrawn
    assert not res.contingency_pushed
    assert all(contingent.total_wh == 0.0 for contingent in monkey_eos.pushed)


def test_push_dry_run_never_pushes(monkey_eos: FakeEos) -> None:
    from hp_twin.domain.sg_plan import SgCandidate

    cand = SgCandidate(0.0, 7200.0, 6500.0, 4200.0, "pv")
    cfg = SgPlanConfig(enabled=True, dry_run=True)
    pushed, slots, _, notes = push_and_collect_placement(cand, cfg, "http://eos:8080", None)
    assert not pushed and not monkey_eos.pushed
    assert any("Dry-Run" in n for n in notes)


def test_run_sg_cycle_full_flow(monkey_eos: FakeEos) -> None:
    df = _df()
    fn = FakeForecastFn()
    # The stored three-day frame ends at 23:55 and aligns to the next day.
    fn.base_anchor = "2026-01-13T00:00:00+00:00"
    monkey_eos.anchor = fn.base_anchor
    cfg = SgPlanConfig(enabled=True, dry_run=False, windows_utc_h=[(0, 24)])
    # Marktzeit = Frame-Zeit: PV-Spitze ab Slot 12 (3 h) — im 8-h-Horizon sichtbar
    solar = [0.0] * 12 + [3000.0] * 16 + [0.0] * (192 - 28)
    market = _market(solar)
    state = Sg3CycleState()
    final, res = run_sg_cycle(
        df,
        _topo(),
        _calib(),
        _model(),
        fn,
        market=market,
        cfg=cfg,
        eos_base_url="http://eos:8080",
        state=state,
    )
    # Contingent gepusht, Placement gelesen, Endprognose mit Injektion gerufen
    assert len(monkey_eos.pushed) == 2, {
        "notes": res.notes,
        "candidates": res.to_dict()["candidates"],
        "feasible_reasons": sorted(set(res.feasible_reasons)),
    }
    assert res.sg3_injected, {"notes": res.notes, "chosen": res.to_dict()["chosen"]}
    injected_calls = [c for c in fn.calls if "sg3_windows" in c]
    assert len(injected_calls) >= 2
    assert injected_calls[-1].get("sg3_windows") is not None
    assert res.placement_slots
    assert res.chosen is not None
    assert len(res.feasible) == 192
    payload = monkey_eos.pushed[0].to_payload()
    assert payload["max_power_w"] > 0
    assert payload["value_eur_per_kwh"] > 0
    assert len(payload["feasible"]) == 192
    assert payload["contiguous"] is True
    assert payload["min_runtime_slots"] > 0
    assert "deadline" in payload
    assert "deadline_hours" not in payload
    assert monkey_eos.pushed[-1].total_wh == 0.0

    _, next_result = run_sg_cycle(
        df,
        _topo(),
        _calib(),
        _model(),
        fn,
        market=market,
        cfg=cfg,
        eos_base_url="http://eos:8080",
        state=state,
    )
    assert monkey_eos.pushed[-1].total_wh == 0.0
    assert not next_result.contingency_withdrawn, next_result.notes


def test_split_placement_is_not_adopted_as_sg3() -> None:
    from hp_twin.domain.sg_plan import SgCandidate

    status = {
        "plan_anchor": "2026-01-13T00:00:00+00:00",
        "time_frame_base": 900,
        "fallback_active": False,
        "plan": [400.0, 0.0, 400.0] + [0.0] * 189,
    }
    res = SgPlanResult()
    state = Sg3CycleState(pending=True)
    final = _adopt_placement(
        df=_df(),
        topology=_topo(),
        calib=_calib(),
        model=_model(),
        forecast_fn=FakeForecastFn(),
        weather=None,
        chosen=SgCandidate(0.0, 1800.0, 800.0, 3600.0, "test"),
        cfg=SgPlanConfig(),
        status=status,
        frame_start=pd.Timestamp("2026-01-13T00:00:00Z"),
        res=res,
        state=state,
    )
    assert final is None
    assert not res.sg3_injected
    assert state.pending
    assert any("ununterbrochenen" in note for note in res.notes)


def test_pending_offer_adopts_one_complete_block_in_the_next_cycle() -> None:
    from hp_twin.domain.sg_plan import SgCandidate

    status = {
        "plan_anchor": "2026-01-13T00:00:00+00:00",
        "time_frame_base": 900,
        "fallback_active": False,
        "plan": [400.0, 400.0] + [0.0] * 190,
    }
    res = SgPlanResult()
    state = Sg3CycleState(pending=True)
    final = _adopt_placement(
        df=_df(),
        topology=_topo(),
        calib=_calib(),
        model=_model(),
        forecast_fn=FakeForecastFn(),
        weather=None,
        chosen=SgCandidate(0.0, 1800.0, 800.0, 3600.0, "previous offer"),
        cfg=SgPlanConfig(),
        status=status,
        frame_start=pd.Timestamp("2026-01-13T00:00:00Z"),
        res=res,
        state=state,
    )
    assert final is not None
    assert res.sg3_injected
    assert state.adopted
    assert not state.pending


def test_placement_must_match_the_simulated_runtime_exactly() -> None:
    from hp_twin.domain.sg_plan import SgCandidate

    status = {
        "plan_anchor": "2026-01-13T00:00:00+00:00",
        "time_frame_base": 900,
        "fallback_active": False,
        "plan": [400.0, 400.0, 400.0] + [0.0] * 189,
    }
    res = SgPlanResult()
    state = Sg3CycleState(pending=True)
    final = _adopt_placement(
        df=_df(),
        topology=_topo(),
        calib=_calib(),
        model=_model(),
        forecast_fn=FakeForecastFn(),
        weather=None,
        chosen=SgCandidate(0.0, 1800.0, 800.0, 3600.0, "previous offer"),
        cfg=SgPlanConfig(),
        status=status,
        frame_start=pd.Timestamp("2026-01-13T00:00:00Z"),
        res=res,
        state=state,
    )
    assert final is None
    assert not res.sg3_injected
    assert res.placement_rejected
    assert state.pending


def test_pending_offer_preserves_the_simulated_setpoint_deadline() -> None:
    from hp_twin.domain.sg_plan import SgCandidate

    status = {
        "plan_anchor": "2026-01-13T00:00:00+00:00",
        "time_frame_base": 900,
        "fallback_active": False,
        "plan": [400.0, 400.0] + [0.0] * 190,
    }
    chosen = SgCandidate(
        0.0,
        1800.0,
        800.0,
        3600.0,
        "previous offer",
        thermal={"setpoint_end_s": 7200.0},
    )
    res = SgPlanResult()
    final = _adopt_placement(
        df=_df(),
        topology=_topo(),
        calib=_calib(),
        model=_model(),
        forecast_fn=FakeForecastFn(),
        weather=None,
        chosen=chosen,
        cfg=SgPlanConfig(),
        status=status,
        frame_start=pd.Timestamp("2026-01-13T00:00:00Z"),
        res=res,
        state=Sg3CycleState(pending=True),
    )
    assert final is not None
    assert not res.placement_rejected


def test_adopted_shifted_run_must_reach_its_thermal_target() -> None:
    from hp_twin.domain.sg_plan import SgCandidate

    status = {
        "plan_anchor": "2026-01-13T00:00:00+00:00",
        "time_frame_base": 900,
        "fallback_active": False,
        "plan": [400.0, 400.0] + [0.0] * 190,
    }
    res = SgPlanResult()
    state = Sg3CycleState(pending=True)
    chosen = SgCandidate(0.0, 1800.0, 800.0, 3600.0, "previous offer", thermal={})
    final = _adopt_placement(
        df=_df(),
        topology=_topo(),
        calib=_calib(),
        model=_model(),
        forecast_fn=FakeForecastFn(),
        weather=None,
        chosen=chosen,
        cfg=SgPlanConfig(),
        status=status,
        frame_start=pd.Timestamp("2026-01-13T00:00:00Z"),
        res=res,
        state=state,
    )
    assert final is not None, res.notes
    assert chosen.thermal["adopted_start_s"] == 0.0
    assert chosen.thermal["adopted_completion_s"] == 1800.0
    assert res.sg3_injected
    assert final.eos.thermal["plan"]["contingent_id"] == "sg3_req"
    assert final.eos.thermal["run_probability"] == [0.0, 0.2, 1.0]


def test_failed_withdrawal_does_not_publish_the_final_profile(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from hp_twin.domain.sg_plan import SgCandidate

    status = {
        "plan_anchor": "2026-01-13T00:00:00+00:00",
        "time_frame_base": 900,
        "fallback_active": False,
        "plan": [400.0, 400.0] + [0.0] * 190,
    }
    monkeypatch.setattr(
        "hp_twin.adapters.sg_orchestrator.withdraw_contingent",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("offline")),
    )
    state = Sg3CycleState(pending=True)
    res = SgPlanResult()
    final = _adopt_placement(
        df=_df(),
        topology=_topo(),
        calib=_calib(),
        model=_model(),
        forecast_fn=FakeForecastFn(),
        weather=None,
        chosen=SgCandidate(0.0, 1800.0, 800.0, 3600.0, "offer", thermal={}),
        cfg=SgPlanConfig(),
        status=status,
        frame_start=pd.Timestamp("2026-01-13T00:00:00Z"),
        res=res,
        state=state,
        eos_base_url="http://eos:8080",
    )
    assert final is None
    assert not res.sg3_injected
    assert state.pending
    assert not state.adopted


def test_pending_zero_plan_is_replanned_after_eos_restart(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from hp_twin.adapters.evcc import MarketSeries
    from hp_twin.adapters.sg_orchestrator import run_sg_cycle

    idx = pd.date_range("2026-01-13T00:00:00Z", periods=192, freq="15min", tz="UTC")
    market = MarketSeries(idx, np.zeros(192), np.full(192, 0.3), None, None, None)
    monkeypatch.setattr(
        "hp_twin.adapters.sg_orchestrator.read_load_status",
        lambda *args, **kwargs: {
            "plan": [0.0] * 192,
            "plan_anchor": str(idx[0]),
            "time_frame_base": 900,
        },
    )
    monkeypatch.setattr(
        "hp_twin.adapters.sg_orchestrator.push_contingent",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("stop after replanning")),
    )
    _, res = run_sg_cycle(
        _df(),
        _topo(),
        _calib(),
        _model(),
        FakeForecastFn(),
        market=market,
        cfg=SgPlanConfig(enabled=True, dry_run=False, windows_utc_h=[(0, 24)]),
        eos_base_url="http://eos",
        state=Sg3CycleState(pending=True, offered_total_wh=800.0),
    )
    assert res.candidates
    assert any("neu geplant" in note for note in res.notes)


def test_pending_offer_keeps_reading_placement_when_candidate_is_unchanged(
    monkey_eos: FakeEos,
) -> None:
    """Die Energie-Hysterese darf ein ausstehendes Placement nicht verstecken."""
    from hp_twin.adapters.evcc import MarketSeries
    from hp_twin.adapters.sg_orchestrator import run_sg_cycle

    idx = pd.date_range("2026-01-13T00:00:00Z", periods=192, freq="15min", tz="UTC")
    market = MarketSeries(idx, np.zeros(192), np.full(192, 0.3), None, None, None)
    state = Sg3CycleState(
        pending=True,
        offered_total_wh=800.0,
        offered_duration_s=3600.0,
        offered_frame_start_iso="2026-01-13T00:00:00+00:00",
    )
    _, res = run_sg_cycle(
        _df(),
        _topo(),
        _calib(),
        _model(),
        FakeForecastFn(),
        market=market,
        cfg=SgPlanConfig(enabled=True, dry_run=False),
        eos_base_url="http://eos",
        state=state,
    )

    assert res.sg3_injected
    assert res.contingency_withdrawn
    assert not any("Hysterese" in note for note in res.notes)


def test_pending_expired_offer_still_adopts_existing_eos_placement(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Die Push-TTL blockiert neue Platzierungen, nicht eine bereits fertige."""
    from hp_twin.adapters.sg_orchestrator import run_sg_cycle

    monkeypatch.setattr(
        "hp_twin.adapters.sg_orchestrator.read_load_status",
        lambda *args, **kwargs: {
            "plan": [400.0, 400.0] + [0.0] * 190,
            "plan_anchor": "2026-01-13T00:00:00+00:00",
            "time_frame_base": 900,
            "reason": "pushed data expired",
            "fallback_active": False,
        },
    )
    monkeypatch.setattr(
        "hp_twin.adapters.sg_orchestrator.withdraw_contingent",
        lambda *args, **kwargs: {"accepted": True},
    )
    _, res = run_sg_cycle(
        _df(),
        _topo(),
        _calib(),
        _model(),
        FakeForecastFn(),
        cfg=SgPlanConfig(enabled=True, dry_run=False),
        eos_base_url="http://eos",
        state=Sg3CycleState(pending=True, offered_total_wh=800.0, offered_duration_s=1800.0),
    )

    assert res.placement_slots == [(0, 400.0), (1, 400.0)]
    assert res.sg3_injected, res.notes
    assert res.contingency_withdrawn
    assert not any("neu geplant" in note for note in res.notes)


def test_sg3_offer_uses_window_delta_not_later_displaced_runs() -> None:
    df = _df()
    fn = FakeForecastFn()
    basis = fn(df, None, None, model=_model())
    # At minute 10 the baseline uses 1 kW and SG3 3.2 kW. The later baseline
    # run is displaced and may not erase this positive flexible window offer.
    # Fourteen full 3.2 kW minutes plus one minute at 2.2 kW give 800 Wh.
    basis.deterministic_p_el_w[10] = 1000.0
    energy_wh, _, _, _, diagnostics = simulate_sg3_run(
        fn,
        df,
        _topo(),
        _calib(),
        _model(),
        600.0,
        1500.0,
        46.0,
        basis=basis,
    )
    assert energy_wh == pytest.approx(800.0)
    assert diagnostics["target_reached"] is True


def test_target_completion_uses_stop_not_full_setpoint_window() -> None:
    df = _df()
    fn = FakeForecastFn()
    basis = fn(df, None, None, model=_model())
    _, _, run_end_s, _, diagnostics = simulate_sg3_run(
        fn,
        df,
        _topo(),
        _calib(),
        _model(),
        600.0,
        600.0 + 3 * 3600.0,
        46.0,
        basis=basis,
    )
    assert run_end_s == 9600.0
    assert diagnostics["target_reached"] is True
    assert diagnostics["setpoint_end_s"] == 11400.0


def test_real_simulation_ends_offer_at_verified_target_completion() -> None:
    from hp_twin.adapters.forecast import run_live_forecast
    from hp_twin.domain.calibration import ThresholdEstimate

    df = _df()
    topo = _topo()
    calib = _calib()
    calib.sg3_buffer_threshold = ThresholdEstimate("sg3_buffer", 1, 1, -5.0, 1.6)
    calib.discharge = dataclasses.replace(calib.discharge, ua_w_per_k=100.0, base_load_w=0.0)
    df = df.copy()
    df["t_hk1_c"] = 29.4
    df["t_dhw_c"] = 47.0
    model = _model()
    model.montecarlo.n_runs = 0
    basis = run_live_forecast(df, topo, calib, model=model)
    _, _, run_end_s, final, diagnostics = simulate_sg3_run(
        run_live_forecast,
        df,
        topo,
        calib,
        model,
        0.0,
        3 * 3600.0,
        49.8,
        basis=basis,
    )
    assert diagnostics["target_reached"] is True
    assert diagnostics["completion_s"] is not None
    assert run_end_s >= diagnostics["completion_s"]
    assert run_end_s < 3 * 3600.0
    assert diagnostics["following_off_s"] is not None
    assert diagnostics["following_off_s"] >= topo.controller.min_off_s
    assert diagnostics["start_buffer_c"] == pytest.approx(float(df["t_hk1_c"].iloc[-1]))
    assert diagnostics["end_buffer_c"] >= diagnostics["target_buffer_c"]
    assert diagnostics["offer_end_s"] >= run_end_s


def _topo():  # type: ignore[no-untyped-def]
    from hp_twin.adapters.io_yaml import load_topology

    return load_topology("config/example_topology.yaml")


def _calib():  # type: ignore[no-untyped-def]
    from hp_twin.cli import _calibrate_report
    from hp_twin.domain.model_config import ModelConfig

    frame = synthetic_frame(days=3)
    frame["flow_rate_lpm"] = np.where(frame["compressor_on"] == 1, 20.0, 0.0)
    # Two elevated-target stop observations calibrate SG3 separately from SG2.
    frame["setpoint_hk1_c"] = np.where(frame["compressor_on"] == 1, 49.8, 31.5)
    frame["t_hk1_c"] = np.where(frame["compressor_on"] == 1, 51.4, 25.95)
    frame.to_csv("/tmp/sg_calib.csv", index=False)
    _, _, calib = _calibrate_report(
        "config/example_topology.yaml", "/tmp/sg_calib.csv", model=ModelConfig()
    )
    assert calib.sg3_buffer_threshold is not None
    assert calib.sg3_buffer_threshold.stop_offset_k == pytest.approx(1.6)
    return calib


def _model():  # type: ignore[no-untyped-def]
    from hp_twin.domain.model_config import ModelConfig

    m = ModelConfig()
    m.sim.horizon_h = 48.0
    return m
