"""Protocols für Energie-Markt-Eingänge (EVCC/EOS): Solar, Preise, Batterie.

Nur die Datenschittstelle, keine Implementierung: Der Adapter (adapters/evcc.py) liefert
15-min-Reihen im Schema des Forecast-Frames. Die Domain konsumiert die Daten über diese
Protocols, ohne HA/EVCC zu kennen.
"""

from __future__ import annotations

from typing import Protocol


class MarketSeries(Protocol):
    """Zeitreihe in 15-min-Slots (UTC-Index als Minuten-Slot ab Frame-Start)."""

    @property
    def solar_surplus_w(self) -> list[float]:
        """PV-Überschuss pro Slot (W), 0 = keine Sonne / unbekannt."""
        ...

    @property
    def price_eur_per_kwh(self) -> list[float]:
        """Bezugspreis pro Slot (€/kWh); leer, wenn kein Tarif vorhanden."""
        ...

    @property
    def feed_in_price_eur_per_kwh(self) -> list[float]:
        """Einspeisepreis pro Slot (€/kWh); leer, wenn unbekannt."""
        ...

    @property
    def battery_soc_kwh(self) -> float | None:
        """Batterie-Restladung jetzt (kWh); None, wenn keine Batterie bekannt."""
        ...

    @property
    def battery_max_power_w(self) -> float | None:
        """Symmetrische Batterie-Lade-/Entladeleistung (W); None wenn unbekannt."""
        ...
