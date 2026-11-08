"""Wetterprognose-Adapter über HA (DWD): Fetch-Parser, Merge in den Forecast-Frame, Settings."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import pytest

from hp_twin.adapters.home_assistant import HaError
from hp_twin.adapters.io_csv import validate_frame
from hp_twin.adapters.replay import build_forecast_frame
from hp_twin.adapters.settings import RuntimeSettings
from hp_twin.adapters.weather_forecast import (
    fetch_weather_forecast,
    forecast_horizon_hours,
    merge_forecast_columns,
)
from scripts_helper import synthetic_frame


def _df(days: int = 3) -> pd.DataFrame:
    return validate_frame(synthetic_frame(days=days))


class FakeClient:
    """HaClient-Lookalike ohne Netz: füttert fixe Service-Antworten."""

    def __init__(self, response: Any) -> None:
        self.base_url = "http://fake:8123"
        self.token = "t"
        self.timeout_s = 5.0
        self.response = response

    def post_response(self, path: str, payload: dict[str, Any]) -> Any:
        return self.response


def _forecast_rows() -> list[dict[str, float | str]]:
    return [
        {"datetime": "2026-01-10T00:00:00Z", "temperature": 5.0, "sun_irradiance": 0.0},
        {"datetime": "2026-01-10T01:00:00Z", "temperature": 4.5, "sun_irradiance": 10.0},
        {"datetime": "2026-01-10T02:00:00Z", "temperature": 4.0, "sun_irradiance": 50.0},
    ]


def test_fetch_weather_forecast_parses_service_response() -> None:
    client = FakeClient({"weather.foo": {"forecast": _forecast_rows()}})
    fc = fetch_weather_forecast(client, "weather.foo")
    assert list(fc.columns) == ["t_outdoor_c", "ghi_wm2"]
    assert fc.index.tz is not None
    assert float(fc["t_outdoor_c"].iloc[2]) == pytest.approx(4.0)
    assert float(fc["ghi_wm2"].iloc[2]) == pytest.approx(50.0)
    assert forecast_horizon_hours(fc) == pytest.approx(2.0)


def test_fetch_weather_forecast_list_form_and_empty_error() -> None:
    fc = fetch_weather_forecast(FakeClient([{"forecast": _forecast_rows()}]), "weather.foo")
    assert len(fc) == 3
    with pytest.raises(HaError):
        fetch_weather_forecast(FakeClient({"weather.foo": {"forecast": []}}), "weather.foo")


def _forecast_frame() -> pd.DataFrame:
    # Prognose muss nach "jetzt" (letzte Messung) liegen
    start = _df(3).index[-1]
    return pd.DataFrame(
        {"t_outdoor_c": [10.0, 12.0], "ghi_wm2": [200.0, 400.0]},
        index=pd.DatetimeIndex([start, start + pd.Timedelta(hours=4)], tz="UTC"),
    )


def test_merge_replaces_within_and_keeps_fallback_after_forecast() -> None:
    df = _df(3)
    fut = build_forecast_frame(df, horizon_h=48.0)
    base = fut["ghi_wm2"]
    fc = _forecast_frame()
    merged = merge_forecast_columns(fut, fc)
    lo, hi = fc.index.min(), fc.index.max()
    within = merged.loc[lo:hi]
    assert within["ghi_wm2"].max() == pytest.approx(400.0)  # Prognosewert statt Fallback
    after = merged.loc[hi + pd.Timedelta(minutes=5) :]
    assert (after["ghi_wm2"] == base.loc[after.index]).all()

    # t_outdoor 2 h nach Start = interpoliert (10->12 über 4 h)
    assert merged["t_outdoor_c"].loc[lo + pd.Timedelta(hours=2)] == pytest.approx(11.0, rel=1e-3)


def test_merge_no_forecast_or_missing_columns_is_noop() -> None:
    df = _df(2)
    fut = build_forecast_frame(df, horizon_h=24.0)
    assert merge_forecast_columns(fut, pd.DataFrame()) is not fut
    assert np.allclose(
        merge_forecast_columns(fut, pd.DataFrame())["t_outdoor_c"], fut["t_outdoor_c"]
    )
    empty_cols = pd.DataFrame({"unrelated": [1.0]}, index=fut.index[:1])
    m = merge_forecast_columns(fut, empty_cols)
    assert np.allclose(m["t_outdoor_c"], fut["t_outdoor_c"])


def test_build_forecast_frame_with_weather() -> None:
    df = _df(3)
    start = df.index[-1]
    fc = pd.DataFrame(
        {"t_outdoor_c": [9.0, 8.0], "ghi_wm2": [123.0, 60.0]},
        index=pd.DatetimeIndex([start, df.index[-1] + pd.Timedelta(hours=1)], tz="UTC"),
    )
    fut = build_forecast_frame(df, horizon_h=24.0, weather=fc)
    # 5 min nach Vorhersagestart (Frame beginnt t0+5min): zwischen den Punkten interpoliert
    assert float(fut["t_outdoor_c"].iloc[0]) == pytest.approx(9.0 + (8.0 - 9.0) * 5 / 60, rel=1e-2)
    assert float(fut["ghi_wm2"].iloc[0]) == pytest.approx(123.0 + (60.0 - 123.0) * 5 / 60, rel=1e-2)
    # Ohne Prognose unverändert (Fallback)
    fut0 = build_forecast_frame(df, horizon_h=24.0)
    assert float(fut0["t_outdoor_c"].iloc[0]) != pytest.approx(fut["t_outdoor_c"].iloc[0])


def test_settings_weather_fields_roundtrip(tmp_path: Any) -> None:
    p = tmp_path / "runtime.json"
    RuntimeSettings(weather_entity="weather.duesseldorf", weather_required=True).save(str(p))
    s = RuntimeSettings.load(str(p))
    assert s.weather_entity == "weather.duesseldorf"
    assert s.weather_required is True
    d = s.public_dict()
    assert d["weather_entity"] == "weather.duesseldorf"


def test_ha_client_post_response_unwraps_service_response(monkeypatch: Any) -> None:
    """post_response entpackt {changed_states, service_response} -> service_response."""
    import json
    import urllib.error

    from hp_twin.adapters.home_assistant import HaClient

    body = json.dumps({"changed_states": [], "service_response": {"a": 1}}).encode()

    class FakeResp:
        def __enter__(self):  # type: ignore[no-untyped-def]
            return self

        def __exit__(self, *args: object) -> None:
            pass

        def read(self) -> bytes:
            return body

    monkeypatch.setattr(urllib.request, "urlopen", lambda req, timeout=None: FakeResp())
    client = HaClient(base_url="http://ha:8123", token="t", timeout_s=2.0)
    assert client.post_response("/api/services/x", {}) == {"a": 1}
