"""Warmwasserzapfungen als Tagesprofil (Erwartungswert).

Deterministisch als Erwartungswert je Stunde; die stochastische Streuung kommt in der
Monte-Carlo-Schicht (M6). Das Profil kann aus der Historie geschätzt werden (DHW-Temperatur-
abfälle, die nicht durch Verluste/Kopplung erklärt sind).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

RHO_CP_J_PER_L_K = 4186.0


@dataclass(frozen=True)
class DhwDrawProfile:
    """Erwartete Zapfmenge je Stunde des Tages (Liter/Stunde)."""

    liters_by_hour: np.ndarray  # shape (24,)
    start_hour: float = 0.0  # Tagesstunde bei Simulationszeit t=0

    def __post_init__(self) -> None:
        if self.liters_by_hour.shape != (24,):
            raise ValueError("liters_by_hour muss Länge 24 haben")

    def liters_per_s(self, t_s: float) -> float:
        hour = int((self.start_hour + t_s / 3600.0) % 24)
        return float(self.liters_by_hour[hour]) / 3600.0

    @classmethod
    def flat(cls, liters_per_day: float, start_hour: float = 0.0) -> DhwDrawProfile:
        return cls(np.full(24, liters_per_day / 24.0), start_hour)

    @classmethod
    def two_peak(cls, liters_per_day: float = 150.0, start_hour: float = 0.0) -> DhwDrawProfile:
        """Typisches Haushaltsprofil: Morgen- und Abendspitze."""
        shape = np.zeros(24)
        for h in range(24):
            shape[h] = np.exp(-((h - 7) ** 2) / 4.0) + 0.8 * np.exp(-((h - 19) ** 2) / 6.0) + 0.05
        shape = shape / shape.sum() * liters_per_day
        return cls(shape, start_hour)


@dataclass
class DhwDraw:
    """Lastmodell: kühlt den DHW-Knoten entsprechend der gezapften Menge und T_kaltwasser."""

    node: str
    profile: DhwDrawProfile
    cold_water_c: float = 12.0

    def draw(self, t_s: float, dt_s: float, storage_temps: dict[str, float]) -> dict[str, float]:
        t_dhw = storage_temps[self.node]
        l_per_s = self.profile.liters_per_s(t_s)
        q = l_per_s * RHO_CP_J_PER_L_K * max(t_dhw - self.cold_water_c, 0.0)
        return {self.node: q}


def estimate_profile_from_history(
    timestamps: np.ndarray,
    t_dhw_c: np.ndarray,
    charging_dhw: np.ndarray,
    volume_l: float,
    cold_water_c: float = 12.0,
) -> DhwDrawProfile:
    """Grobschätzung des Zapfprofils aus DHW-Temperaturabfällen (nicht beim Laden).

    Zapf-Liter ~ Volumen * (-dT) / (T_dhw - T_kalt), stündlich gemittelt über die Tage.
    Feinschliff (Verluste/Kopplung abziehen) folgt in der Kalibrierung (M4).
    """
    ts = np.asarray(timestamps, dtype="datetime64[s]")
    t = np.asarray(t_dhw_c, dtype=float)
    charging = np.asarray(charging_dhw, dtype=bool)
    hours = ts.astype("datetime64[h]").astype(int) % 24

    liters_per_hour_bucket: list[list[float]] = [[] for _ in range(24)]
    daily_bucket: dict[int, np.ndarray] = {}
    for i in range(1, len(t)):
        if charging[i] or charging[i - 1]:
            continue
        dtemp = t[i] - t[i - 1]
        if dtemp >= 0:
            continue  # nur Abfälle = Zapfung
        denom = max(t[i] - cold_water_c, 5.0)
        liters = volume_l * (-dtemp) / denom
        h = int(hours[i])
        day = int(ts[i].astype("datetime64[D]").astype(int))
        daily_bucket.setdefault(day, np.zeros(24))
        daily_bucket[day][h] += liters
        liters_per_hour_bucket[h].append(liters)

    if daily_bucket:
        profile = np.mean(np.stack(list(daily_bucket.values())), axis=0)
    else:
        profile = np.zeros(24)
    return DhwDrawProfile(profile)
