"""EVCC/Energie-Markt-Adapter: Solarprognose, Tarife, Batterie aus Home-Assistant-Sensoren.

Quellen (Entities per Settings konfigurierbar, Defaults wie in der Anlage vorhanden):
- solar: `sensor.evcc_forecast_solar` (Attribut timeseries, 15-min-Raster, UTC-Start,
  W-Skala mit evcc-internem Faktor `scale` bereits eingerechnet)
- price: `sensor.evcc_tariff_api_planner` (Attribut rates, 15-min-Raster, €/kWh)
- feed-in: `sensor.evcc_forecast_feedin` (aktueller Einspeisepreis, €/kWh -> konstant)
- battery: `sensor.evcc_batterie_restladung` (kWh), Leistung über EOS/EVCC (optional)

Trennung: fetch_market_series (IO) vs. reine Transformation in der Dataclass.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from contextlib import suppress
from typing import Any

import numpy as np
import pandas as pd

from hp_twin.adapters.home_assistant import HaClient, HaError

DEFAULT_ENTITIES: dict[str, str] = {
    "solar": "sensor.evcc_forecast_solar",
    "price": "sensor.evcc_tariff_api_planner",
    "feed_in_price": "sensor.evcc_forecast_feedin",
    "battery_soc": "sensor.evcc_batterie_restladung",
}


class MarketSeries:
    """Gespeicherte Reihen im 15-min-Raster (UTC-Index) als MarketSeries-Implementierung."""

    def __init__(
        self,
        index: pd.DatetimeIndex,
        solar_w: np.ndarray,
        price: np.ndarray,
        feed_in: float | None,
        battery_soc_kwh: float | None,
        battery_max_power_w: float | None,
    ) -> None:
        self.index = pd.DatetimeIndex(index)
        self._solar = np.asarray(solar_w, dtype=float)
        self._price = np.asarray(price, dtype=float)  # NaN außerhalb des Tarif-Zeitraums
        self._feed_in = feed_in
        self._battery_soc = battery_soc_kwh
        self._battery_power = battery_max_power_w

    def price_at(self, dt_index: pd.DatetimeIndex) -> np.ndarray:
        """Preise für beliebige Zeitpunkte; außerhalb der Tarifdeckung = NaN.

        Die Simulation beginnt eine Minute nach dem letzten Messpunkt, also nicht
        zwingend auf einer Viertelstunde. Der Preis gehört zum bereits laufenden
        15-min-Slot; exaktes Reindex würde sonst alle Tarife verlieren.
        """
        slot_index = pd.DatetimeIndex(dt_index).floor("15min")
        return pd.Series(self._price, index=self.index).reindex(slot_index).to_numpy(dtype=float)

    @property
    def solar_surplus_w(self) -> list[float]:
        return [round(float(v), 1) for v in self._solar]

    @property
    def price_eur_per_kwh(self) -> list[float]:
        """Nur belegte Slots (ohne NaN) — leer, wenn der Tarif fehlt."""
        return [round(float(v), 5) for v in self._price if np.isfinite(v)]

    @property
    def feed_in_price_eur_per_kwh(self) -> list[float]:
        return [] if self._feed_in is None else [round(float(self._feed_in), 5)]

    @property
    def battery_soc_kwh(self) -> float | None:
        return self._battery_soc

    @property
    def battery_max_power_w(self) -> float | None:
        return self._battery_power


def _attr_series(entity: dict[str, Any]) -> tuple[pd.Timestamp, list[float], int]:
    """Aus {start_utc, deltas_in_minutes, values} -> (Start, Werte, Raster in Minuten)."""
    key = "timeseries" if "timeseries" in entity.get("attributes", {}) else "rates"
    s = entity["attributes"][key]
    vals = [float(v) if v is not None else float("nan") for v in s.get("values", [])]
    deltas = s.get("deltas_in_minutes", [])
    step_min = int(deltas[0]) if deltas else 15
    start = pd.Timestamp(s["start_utc"])
    return start, vals, step_min


def _read_series(client: HaClient, entity_id: str, expected_key: str) -> pd.Series:
    st = client.state(entity_id)
    if not st or "attributes" not in st:
        raise HaError(f"Entity '{entity_id}' fehlt oder hat keine Attribute")
    start, vals, step = _attr_series(st)
    idx = pd.date_range(start, periods=len(vals), freq=f"{step}min", tz="UTC")
    return pd.Series(vals, index=idx, name=entity_id)


def fetch_market_series(client: HaClient, entities: dict[str, str] | None = None) -> MarketSeries:
    """Holt Solar-/Preis-/Batterie-Werte aus HA und normalisiert aufs 15-min-Raster.

    Fehlt eine Quelle (Entity nicht vorhanden), bleibt die entsprechende Reihe leer bzw.
    None — der SG-Orchestrator behandelt fehlende Markt-Signale als "kein SG3".
    """
    ents = {**DEFAULT_ENTITIES, **(entities or {})}

    def _try(fn: Callable[[], Any]) -> Any | None:
        try:
            return fn()
        except HaError:
            return None

    solar = price = None
    feed_in = None
    battery_soc = None
    battery_power = None
    if ents.get("solar"):
        solar = _try(lambda: _read_series(client, ents["solar"], "timeseries"))
    if ents.get("price"):
        price = _try(lambda: _read_series(client, ents["price"], "rates"))
    if ents.get("feed_in_price"):
        st = _try(lambda: client.state(ents["feed_in_price"]))
        if st and st.get("state") not in (None, "unknown", "unavailable"):
            with suppress(ValueError):
                feed_in = float(st["state"])
    if ents.get("battery_soc"):
        st = _try(lambda: client.state(ents["battery_soc"]))
        if st and st.get("state") not in (None, "unknown", "unavailable"):
            with suppress(ValueError):
                battery_soc = float(st["state"])
    if solar is None or solar.empty:
        raise HaError("Keine Solarprognose ('sensor.evcc_forecast_solar' o. ä.) verfügbar")
    base_index = solar.index
    # Preis-Reihe deckt nur 24 h ab: außerhalb NaN markieren statt 0 (0 würde "gratis")
    price_aligned = (
        price.reindex(base_index).to_numpy(dtype=float)
        if price is not None and not price.empty
        else np.full(len(base_index), np.nan)
    )
    return MarketSeries(
        index=base_index,
        solar_w=np.nan_to_num(solar.to_numpy(dtype=float), nan=0.0),
        price=price_aligned,
        feed_in=feed_in,
        battery_soc_kwh=battery_soc,
        battery_max_power_w=battery_power,
    )


def fetch_market_series_safe(settings: Any) -> MarketSeries | None:
    """fetch_market_series mit Fehler-Toleranz: None statt Exception bei fehlendem EVCC.

    Der SG-Orchestrator behandelt None als "keine Markt-Signale" (kein SG3) — die
    reguläre Prognose läuft ohne SG3 weiter.
    """
    try:
        client = HaClient.from_env(settings.env_path)
        return fetch_market_series(client, entities=settings.evcc_entities or None)
    except HaError as e:
        logging.getLogger(__name__).warning("Markt-Signale nicht verfügbar (%s)", e)
        return None
