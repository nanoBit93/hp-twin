"""SG-Orchestrator: Basis->Kandidat->Contingent(->EOS-Fake)->Placement->Injektion."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import pytest

from hp_twin.adapters.eos_connect import EosContingent
from hp_twin.adapters.sg_orchestrator import (
    SgPlanConfig,
    allowed_window_s,
    push_and_collect_placement,
    run_sg_cycle,
    sg3_energy_vs_basis,
)
from scripts_helper import synthetic_frame


class FakeForecastResult:
    def __init__(self, n: int = 2880, p_on: int = 1000) -> None:
        self.deterministic_p_el_w = np.zeros(n)
        self.deterministic_p_el_w[400:450] = 2800.0  # Basis-Lauf morgens
        self.buffer_sim_c = np.full(n, 28.0)
        self.buffer_sim_c[400:450] = 38.0  # ladend
        # Event im 600..7200-s-Fenster (für simulate_sg3_run): 600..6600 s @ 60 s
        self.events: list[Any] = [_Event(600.0, 6600.0, "heat", 3200.0)]


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
        if "sg3_start_s" in kw:
            dt_s = float(df.index.to_series().diff().dropna().iloc[0].total_seconds())
            res = FakeForecastResult()
            i0 = int(kw["sg3_start_s"] / dt_s)
            i1 = min(
                len(res.deterministic_p_el_w),
                i0 + int((kw["sg3_end_s"] - kw["sg3_start_s"]) / dt_s),
            )
            res.deterministic_p_el_w[i0:i1] = 3200.0
            return res
        return FakeForecastResult()


class FakeEos:
    def __init__(self) -> None:
        self.pushed: list[EosContingent] = []
        self.plan = [0.0] * 192
        self.plan[44:56] = [3500.0] * 12  # Placement: Slots 44-55 (11:00-12:00 nach Start)

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
        return {
            "id": load_id,
            "kind": "contingent",
            "planned_wh": float(sum(self.plan)),
            "plan": self.plan,
            "release": {"state": "blocked"},
            "fallback_active": False,
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


def test_allowed_window_utc() -> None:
    cfg = SgPlanConfig(windows_utc_h=[(7, 8), (10, 14)])
    inside = pd.Timestamp("2026-11-08 10:30:00+00:00")
    w = allowed_window_s(inside, cfg, 48 * 3600.0)
    assert w == (0.0, 3.5 * 3600.0)
    outside = pd.Timestamp("2026-11-08 20:00:00+00:00")
    assert allowed_window_s(outside, cfg, 48 * 3600.0) is None


def test_sg3_energy_vs_basis_counts_only_running() -> None:
    fr = FakeForecastResult(n=2880)  # 48 h à 60 s
    # Fenster 60..600 s liegt bei idx 1..10; Basis läuft erst ab idx 24000 (400 min)
    wh = sg3_energy_vs_basis(fr, 60.0, 600.0, running_threshold_w=200.0, rated_w=3500.0)
    assert wh == pytest.approx(3500.0 * 540.0 / 3600.0, rel=1e-2)


def test_sg3_energy_vs_basis_deducts_running_basis() -> None:
    fr = FakeForecastResult(n=2880)
    fr.deterministic_p_el_w[:] = 0.0
    # Basis läuft im Fenster 60..600 s konstant auf 3000 W (idx 1..10)
    fr.deterministic_p_el_w[1:10] = 3000.0
    wh = sg3_energy_vs_basis(fr, 60.0, 600.0, running_threshold_w=200.0, rated_w=3500.0)
    # SG3 ersetzt die Basis-Ladung (540 s) -> nur die Differenz ist Contingent-Energie
    assert wh == pytest.approx((3500.0 - 3000.0) * 540.0 / 3600.0, rel=1e-2)


def test_push_and_collect_with_fake_eos(monkey_eos: FakeEos) -> None:
    from hp_twin.domain.sg_plan import SgCandidate

    cand = SgCandidate(0.0, 7200.0, 6500.0, 4200.0, "pv_surplus>=2500W")
    cfg = SgPlanConfig(enabled=True, dry_run=False)
    pushed, slots, status, notes = push_and_collect_placement(
        cand, cfg, "http://eos:8080", None, last_total_wh=None
    )
    assert pushed and len(monkey_eos.pushed) == 1
    assert monkey_eos.pushed[0].total_wh == pytest.approx(6500.0)
    assert slots == [(i, v) for i, v in enumerate(monkey_eos.plan) if v > 0]
    assert status.get("planned_wh") == pytest.approx(3500.0 * 12)
    assert any("akzeptiert" in n for n in notes)


def test_push_hysteresis_skips_second_push(monkey_eos: FakeEos) -> None:
    from hp_twin.domain.sg_plan import SgCandidate

    cand = SgCandidate(0.0, 7200.0, 6500.0, 4200.0, "pv_surplus>=2500W")
    cfg = SgPlanConfig(enabled=True, dry_run=False)
    push_and_collect_placement(cand, cfg, "http://eos:8080", None, last_total_wh=None)
    pushed2, _, _, notes = push_and_collect_placement(
        cand, cfg, "http://eos:8080", None, last_total_wh=6400.0
    )
    assert not pushed2
    assert any("Hysterese" in n for n in notes)
    assert len(monkey_eos.pushed) == 1


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
    cfg = SgPlanConfig(enabled=True, dry_run=False, windows_utc_h=[(0, 24)])
    # Marktzeit = Frame-Zeit: PV-Spitze ab Slot 12 (3 h) — im 8-h-Horizon sichtbar
    solar = [0.0] * 12 + [3000.0] * 16 + [0.0] * (192 - 28)
    market = _market(solar)
    final, res = run_sg_cycle(
        df, _topo(), _calib(), _model(), fn, market=market, cfg=cfg, eos_base_url="http://eos:8080"
    )
    # Contingent gepusht, Placement gelesen, Endprognose mit Injektion gerufen
    assert len(monkey_eos.pushed) == 1
    assert res.sg3_injected
    injected_calls = [c for c in fn.calls if "sg3_start_s" in c]
    # simulate_sg3_run (Kandidaten je Kandidat-Fenster) + 1x Endprognose
    assert len(injected_calls) >= 2
    assert injected_calls[-1].get("sg3_start_s") is not None
    assert res.placement_slots
    assert res.chosen is not None


def _topo():  # type: ignore[no-untyped-def]
    from hp_twin.adapters.io_yaml import load_topology

    return load_topology("config/example_topology.yaml")


def _calib():  # type: ignore[no-untyped-def]
    from hp_twin.cli import _calibrate_report
    from hp_twin.domain.model_config import ModelConfig

    frame = synthetic_frame(days=3)
    frame["flow_rate_lpm"] = np.where(frame["compressor_on"] == 1, 20.0, 0.0)
    frame.to_csv("/tmp/sg_calib.csv", index=False)
    _, _, calib = _calibrate_report(
        "config/example_topology.yaml", "/tmp/sg_calib.csv", model=ModelConfig()
    )
    return calib


def _model():  # type: ignore[no-untyped-def]
    from hp_twin.domain.model_config import ModelConfig

    m = ModelConfig()
    m.sim.horizon_h = 8.0
    return m
