"""Kalibrierung der Modellparameter aus historischen Messdaten.

Getrennte, testbare Fitter:
- fit_heatpump_map: WP-Leistungskennfeld (P_el) und COP aus Betriebspunkten.
- estimate_thresholds: Start-/Stoppschwellen relativ zum Sollwert empirisch aus Taktflanken.
- fit_discharge: effektiver Emissions-/Entladekoeffizient aus der gemessenen Puffer-Entladerate.

Alle Fitter liefern zusätzlich Gütemetriken (R², RMSE) zur Beurteilung des Fits.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np

from hp_twin.domain.heat_source import KELVIN, HeatPumpAirWater

RHO_CP_J_PER_L_K = 4186.0


def _r2(meas: np.ndarray, pred: np.ndarray) -> float:
    meas = np.asarray(meas, float)
    denom = np.sum((meas - meas.mean()) ** 2)
    if denom <= 0:
        return float("nan")
    return float(1.0 - np.sum((pred - meas) ** 2) / denom)


def _rmse(meas: np.ndarray, pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((np.asarray(pred, float) - np.asarray(meas, float)) ** 2)))


# --------------------------------------------------------------- WP-Kennfeld
@dataclass(frozen=True)
class HeatPumpFit:
    heat_pump: HeatPumpAirWater
    n_points: int
    p_el_r2: float
    p_el_rmse_w: float
    q_th_r2: float
    q_th_rmse_w: float
    cop_median_measured: float


def fit_heatpump_map(
    t_outdoor_c: np.ndarray,
    t_flow_c: np.ndarray,
    p_el_w: np.ndarray,
    q_th_w: np.ndarray,
    standby_w: float = 80.0,
    t_flow_ref_c: float = 35.0,
    t_out_ref_c: float = 5.0,
) -> HeatPumpFit:
    """Fittet P_comp = a + b*T_vl + c*T_aus (linear) und eta_carnot für den COP."""
    to = np.asarray(t_outdoor_c, float)
    tf = np.asarray(t_flow_c, float)
    pe = np.asarray(p_el_w, float)
    qt = np.asarray(q_th_w, float)

    p_comp = pe - standby_w
    x = np.column_stack([np.ones_like(tf), tf, to])
    coef, *_ = np.linalg.lstsq(x, p_comp, rcond=None)
    a, b, c = (float(v) for v in coef)
    p_el_ref = a + b * t_flow_ref_c + c * t_out_ref_c

    # COP aus Q_th / P_comp (nur positive, sinnvolle Punkte)
    cop_meas = qt / np.clip(p_comp, 1.0, None)
    t_cond = tf + 5.0 + KELVIN
    t_evap = to - 6.0 + KELVIN
    carnot = t_cond / np.clip(t_cond - t_evap, 1.0, None)
    eta = float(np.sum(carnot * cop_meas) / np.sum(carnot**2))

    hp = HeatPumpAirWater(
        p_el_ref_w=p_el_ref,
        standby_w=standby_w,
        dp_dt_flow_w_per_k=b,
        dp_dt_out_w_per_k=c,
        t_flow_ref_c=t_flow_ref_c,
        t_out_ref_c=t_out_ref_c,
        eta_carnot=eta,
    )
    p_pred = np.array(
        [hp.compressor_power_w(o, f) + standby_w for o, f in zip(to, tf, strict=True)]
    )
    q_pred = np.array([hp.output(o, f, True).q_th_w for o, f in zip(to, tf, strict=True)])
    return HeatPumpFit(
        heat_pump=hp,
        n_points=len(tf),
        p_el_r2=_r2(pe, p_pred),
        p_el_rmse_w=_rmse(pe, p_pred),
        q_th_r2=_r2(qt, q_pred),
        q_th_rmse_w=_rmse(qt, q_pred),
        cop_median_measured=float(np.median(cop_meas)),
    )


# --------------------------------------------------------- Regler-Schwellen
@dataclass(frozen=True)
class ThresholdEstimate:
    circuit: str
    n_starts: int
    n_stops: int
    start_offset_k: float  # (T_sensor - Soll) beim Start (typ. negativ)
    stop_offset_k: float  # (T_sensor - Soll) beim Stopp (typ. positiv)


def estimate_thresholds(
    compressor_on: np.ndarray,
    t_sensor_c: np.ndarray,
    setpoint_c: np.ndarray,
    circuit: str = "buffer",
    min_setpoint_c: float | None = None,
    target_tolerance_k: float | None = None,
) -> ThresholdEstimate:
    """Schätzt Start-/Stoppschwelle relativ zum Soll aus den Verdichter-Flanken.

    Bei einem Mindestsollwert werden ausschließlich Stopps verwendet, deren letzter
    Betriebsschritt diesen Wert erreicht hat. Mit einer Zieltoleranz muss die
    Temperatur diesen Sollwert auch tatsächlich erreicht haben: Ein vorzeitiger
    Abbruch unter dem angehobenen SG3-Ziel ist kein thermischer Ladeabschluss.
    """
    on = np.asarray(compressor_on, float).astype(int)
    ts = np.asarray(t_sensor_c, float)
    sp = np.asarray(setpoint_c, float)
    d = np.diff(on)
    starts = np.where(d == 1)[0]  # Index VOR dem Einschalten
    stops = np.where(d == -1)[0]  # Index VOR dem Ausschalten (letzter An-Schritt)
    if min_setpoint_c is not None:
        starts = starts[sp[starts] >= min_setpoint_c]
        stops = stops[sp[stops] >= min_setpoint_c]
    if target_tolerance_k is not None:
        stops = stops[ts[stops] >= sp[stops] - target_tolerance_k]
    start_off = (ts[starts] - sp[starts]) if len(starts) else np.array([np.nan])
    stop_off = (ts[stops] - sp[stops]) if len(stops) else np.array([np.nan])
    return ThresholdEstimate(
        circuit=circuit,
        n_starts=int(len(starts)),
        n_stops=int(len(stops)),
        start_offset_k=(
            float(np.nanmedian(start_off)) if np.isfinite(start_off).any() else float("nan")
        ),
        stop_offset_k=(
            float(np.nanmedian(stop_off)) if np.isfinite(stop_off).any() else float("nan")
        ),
    )


# --------------------------------------------------- Gebäude/Entladung
@dataclass(frozen=True)
class DischargeFit:
    ua_w_per_k: float  # Entladeleistung je K des Treibers (z. B. T_raum - T_außen) = UA_Gebäude
    base_load_w: float  # Achsenabschnitt (mittlere Grundlast/Zapfung)
    solar_w_per_wm2: float  # Solarentlastung je W/m² GHI (0 ohne Solar-Regressor)
    n_points: int
    r2: float


def fit_discharge(
    buffer_temp_c: np.ndarray,
    driver_dt_c: np.ndarray,
    compressor_on: np.ndarray,
    dt_s: float,
    volume_l: float,
    ghi_w_m2: np.ndarray | None = None,
    window_s: float = 900.0,
) -> DischargeFit:
    """Heizlast aus der Puffer-Entladerate bei abgeschalteter WP.

    Der Pufferfühler ist grob quantisiert (0,1 K) -> eine 1-min-Ableitung ist Rauschen. Deshalb
    wird die Entladeleistung über FENSTER (Default 15 min) ohne WP-Betrieb gemittelt:
    P_entlade = C_puffer * (T_start - T_ende)/window, regressiert auf das mittlere `driver_dt_c`
    (z. B. T_raum - T_außen) und optional die Globalstrahlung (Solargewinn entlastet die Heizung):
    P = base + ua*driver - solar*GHI. Steigung ua = effektives UA_Gebäude.
    """
    tb = np.asarray(buffer_temp_c, float)
    drv = np.asarray(driver_dt_c, float)
    on = np.asarray(compressor_on, float).astype(int)
    ghi = np.asarray(ghi_w_m2, float) if ghi_w_m2 is not None else None
    c_buffer = volume_l * RHO_CP_J_PER_L_K
    w = max(int(round(window_s / dt_s)), 2)

    drv_pts: list[float] = []
    ghi_pts: list[float] = []
    disch_pts: list[float] = []
    for s in range(0, len(tb) - w, w):
        seg = slice(s, s + w)
        if on[seg].any():  # WP lief im Fenster -> überspringen
            continue
        discharge_w = c_buffer * (tb[s] - tb[s + w]) / window_s
        if discharge_w > 0 and np.isfinite(discharge_w):
            d_mean = float(np.mean(drv[seg]))
            g_mean = float(np.mean(ghi[seg])) if ghi is not None else 0.0
            if not (np.isfinite(d_mean) and np.isfinite(g_mean)):
                continue  # Fenster mit Sensorlücke nicht verwenden
            disch_pts.append(discharge_w)
            drv_pts.append(d_mean)
            ghi_pts.append(g_mean)

    if len(disch_pts) < 10:
        return DischargeFit(0.0, 0.0, 0.0, len(disch_pts), float("nan"))
    drv_a = np.array(drv_pts)
    y = np.array(disch_pts)
    cols = [np.ones(len(drv_a)), drv_a]
    if ghi is not None:
        cols.append(-np.array(ghi_pts))  # negatives Vorzeichen: Solar entlastet
    x = np.column_stack(cols)
    coef, *_ = np.linalg.lstsq(x, y, rcond=None)
    solar = float(coef[2]) if ghi is not None else 0.0
    return DischargeFit(
        ua_w_per_k=float(coef[1]),
        base_load_w=float(coef[0]),
        solar_w_per_wm2=solar,
        n_points=len(drv_a),
        r2=_r2(y, x @ coef),
    )


# ------------------------------------------------------------- Gesamtbericht
@dataclass
class CalibrationReport:
    heat_pump: HeatPumpFit
    buffer_threshold: ThresholdEstimate
    discharge: DischargeFit
    notes: list[str] = field(default_factory=list)
    sg3_buffer_threshold: ThresholdEstimate | None = None

    def to_dict(self) -> dict[str, Any]:
        hp = self.heat_pump
        return {
            "heat_pump": {
                "p_el_ref_w": round(hp.heat_pump.p_el_ref_w, 1),
                "dp_dt_flow_w_per_k": round(hp.heat_pump.dp_dt_flow_w_per_k, 2),
                "dp_dt_out_w_per_k": round(hp.heat_pump.dp_dt_out_w_per_k, 2),
                "eta_carnot": round(hp.heat_pump.eta_carnot, 3),
                "n_points": hp.n_points,
                "p_el_r2": round(hp.p_el_r2, 3),
                "p_el_rmse_w": round(hp.p_el_rmse_w, 1),
                "q_th_r2": round(hp.q_th_r2, 3),
                "q_th_rmse_w": round(hp.q_th_rmse_w, 1),
                "cop_median_measured": round(hp.cop_median_measured, 2),
            },
            "buffer_threshold": {
                "n_starts": self.buffer_threshold.n_starts,
                "n_stops": self.buffer_threshold.n_stops,
                "start_offset_k": round(self.buffer_threshold.start_offset_k, 2),
                "stop_offset_k": round(self.buffer_threshold.stop_offset_k, 2),
            },
            "sg3_buffer_threshold": asdict(self.sg3_buffer_threshold)
            if self.sg3_buffer_threshold is not None
            else None,
            "discharge": {
                "ua_w_per_k": round(self.discharge.ua_w_per_k, 1),
                "base_load_w": round(self.discharge.base_load_w, 1),
                "solar_w_per_wm2": round(self.discharge.solar_w_per_wm2, 2),
                "n_points": self.discharge.n_points,
                "r2": round(self.discharge.r2, 3),
            },
            "notes": self.notes,
        }


# ----------------------------------------------- Serialisierung (Bundle-Export/Import)
_HP_METRIC_KEYS = (
    "n_points",
    "p_el_r2",
    "p_el_rmse_w",
    "q_th_r2",
    "q_th_rmse_w",
    "cop_median_measured",
)


def calibration_to_json(report: CalibrationReport) -> dict[str, Any]:
    """Vollständige, verlustfreie Serialisierung einer Kalibrierung (für portable Bundles)."""
    return {
        "heat_pump": asdict(report.heat_pump.heat_pump),
        "heat_pump_metrics": {k: getattr(report.heat_pump, k) for k in _HP_METRIC_KEYS},
        "buffer_threshold": asdict(report.buffer_threshold),
        "sg3_buffer_threshold": asdict(report.sg3_buffer_threshold)
        if report.sg3_buffer_threshold is not None
        else None,
        "discharge": asdict(report.discharge),
        "notes": list(report.notes),
    }


def calibration_from_json(d: dict[str, Any]) -> CalibrationReport:
    """Rekonstruiert eine CalibrationReport aus calibration_to_json()."""
    hp = HeatPumpAirWater(**d["heat_pump"])
    hp_fit = HeatPumpFit(heat_pump=hp, **d["heat_pump_metrics"])
    return CalibrationReport(
        heat_pump=hp_fit,
        buffer_threshold=ThresholdEstimate(**d["buffer_threshold"]),
        discharge=DischargeFit(**d["discharge"]),
        notes=list(d.get("notes", [])),
        sg3_buffer_threshold=ThresholdEstimate(**d["sg3_buffer_threshold"])
        if d.get("sg3_buffer_threshold") is not None
        else None,
    )
