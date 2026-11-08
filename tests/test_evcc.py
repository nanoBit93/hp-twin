"""EVCC/Energie-Markt-Adapter: Series-Parsing, Alignment, fehlende Quellen."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import pytest

from hp_twin.adapters.evcc import DEFAULT_ENTITIES, MarketSeries, fetch_market_series
from hp_twin.adapters.home_assistant import HaError


class FakeStates:
    def __init__(self, states: dict[str, dict[str, Any]]) -> None:
        self.states = states
        self.base_url = "http://ha:8123"
        self.token = "t"
        self.timeout_s = 5.0

    def state(self, entity_id: str) -> dict[str, Any]:
        if entity_id not in self.states:
            raise HaError(f"nicht vorhanden: {entity_id}")
        return self.states[entity_id]


def _ts_entity(
    values: list[float], start: str = "2026-11-08T00:00:00+00:00", key: str = "timeseries"
) -> dict[str, Any]:
    return {
        "attributes": {
            key: {
                "start_utc": start,
                "deltas_in_minutes": [15] * len(values),
                "values": values,
            }
        }
    }


def _state(value: Any) -> dict[str, Any]:
    return {"state": value, "attributes": {}}


def test_fetch_market_series_parses_and_aligns() -> None:
    solar_vals = [0.0] * 40 + [2500.0, 3000.0, 2800.0] + [0.0] * 5
    price_vals = [0.40, 0.39] * 12
    client = FakeStates(
        {
            DEFAULT_ENTITIES["solar"]: _ts_entity(solar_vals),
            DEFAULT_ENTITIES["price"]: _ts_entity(price_vals, key="rates"),
            DEFAULT_ENTITIES["feed_in_price"]: _state(0.07),
            DEFAULT_ENTITIES["battery_soc"]: _state(5.46),
        }
    )
    ms = fetch_market_series(client)
    assert len(ms.solar_surplus_w) == len(solar_vals)
    assert max(ms.solar_surplus_w) == pytest.approx(3000.0)
    assert len(ms.price_eur_per_kwh) == len(price_vals)
    assert ms.price_eur_per_kwh[0] == pytest.approx(0.40, abs=1e-6)
    assert ms.feed_in_price_eur_per_kwh == [0.07]
    assert ms.battery_soc_kwh == pytest.approx(5.46)


def test_fetch_market_series_missing_price_yields_empty_series() -> None:
    solar_vals = [100.0, 200.0]
    client = FakeStates({DEFAULT_ENTITIES["solar"]: _ts_entity(solar_vals)})
    ms = fetch_market_series(client)
    assert ms.price_eur_per_kwh == []
    assert ms.solar_surplus_w == [100.0, 200.0]
    assert ms.battery_soc_kwh is None


def test_fetch_market_series_no_solar_raises() -> None:
    client = FakeStates({})
    with pytest.raises(HaError):
        fetch_market_series(client)


def test_price_at_outside_coverage_is_nan() -> None:
    idx = pd.date_range("2026-11-08T00:00:00Z", periods=6, freq="15min", tz="UTC")
    ms = MarketSeries(
        index=idx,
        solar_w=np.zeros(6),
        price=np.array([0.4, 0.4, 0.4, np.nan, np.nan, np.nan]),
        feed_in=None,
        battery_soc_kwh=None,
        battery_max_power_w=None,
    )
    out = ms.price_at(idx)
    assert np.isfinite(out[:3]).all() and np.isnan(out[3:]).all()
