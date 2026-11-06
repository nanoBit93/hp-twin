"""CLI-Einstieg: validate | export (weitere Befehle folgen je Meilenstein)."""

from __future__ import annotations

import argparse
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pandas as pd

from hp_twin.adapters.io_yaml import load_topology


def _cmd_validate(args: argparse.Namespace) -> int:
    topo = load_topology(args.topology)
    print(f"OK: {topo.name}  ({len(topo.storage.nodes)} Knoten, {len(topo.consumers)} Consumer)")
    return 0


def _cmd_export(args: argparse.Namespace) -> int:
    from hp_twin.adapters.ha_history import HaMapping, fetch_measurements
    from hp_twin.adapters.home_assistant import HaClient

    client = HaClient.from_env(args.env, timeout_s=args.timeout)
    mapping = HaMapping.load(args.mapping)
    end = datetime.now(UTC)
    start = end - timedelta(days=args.days)
    print(
        f"Ziehe {args.days} Tage ({start:%Y-%m-%d %H:%M} .. {end:%Y-%m-%d %H:%M} UTC) für "
        f"{len(mapping.all_entities())} Entities ..."
    )
    df = fetch_measurements(
        client, mapping, start, end, chunk_hours=args.chunk_hours, progress=True
    )

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    if args.append and out.exists():
        old = pd.read_csv(out)
        old["timestamp"] = pd.to_datetime(old["timestamp"], utc=True)
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
        df = (
            pd.concat([old, df], ignore_index=True)
            .drop_duplicates(subset="timestamp", keep="last")
            .sort_values("timestamp")
        )
    df.to_csv(out, index=False)
    print(f"{len(df)} Zeilen -> {out}")
    return 0


def _calibrate_report(topology: str, csv: str, model: Any = None):  # type: ignore[no-untyped-def]
    """Kalibriert WP-Kennfeld, Regler-Schwellen und Gebäude-UA aus Messdaten (parametrisiert)."""
    import numpy as np

    from hp_twin.adapters.io_csv import read_measurements, room_columns
    from hp_twin.domain.calibration import (
        CalibrationReport,
        estimate_thresholds,
        fit_discharge,
        fit_heatpump_map,
    )
    from hp_twin.domain.model_config import ModelConfig

    model = model or ModelConfig()
    cal = model.calibration
    topo = load_topology(topology)
    df = read_measurements(csv)
    dt_s = float(df.index.to_series().diff().dropna().iloc[0].total_seconds())
    m = (
        (df["compressor_on"] == 1)
        & (df["mode"] == "heat")
        & (df["flow_rate_lpm"] > cal.op_min_flow_lpm)
        & (df["p_el_w"] > cal.op_min_p_el_w)
        & (df["t_flow_c"] > df["t_return_c"])
    )
    d = df.loc[m]
    idle_p = df.loc[df["compressor_on"] == 0, "p_el_w"]
    standby = float(idle_p.median()) if len(idle_p) else cal.standby_fallback_w
    hp_fit = fit_heatpump_map(
        d["t_outdoor_c"].to_numpy(),
        d["t_flow_c"].to_numpy(),
        d["p_el_w"].to_numpy(),
        (d["q_heat_wh"] * 60.0).to_numpy(),
        standby_w=standby,
        t_flow_ref_c=cal.t_flow_ref_c,
        t_out_ref_c=cal.t_out_ref_c,
    )
    thr = estimate_thresholds(
        df["compressor_on"].to_numpy(),
        df["t_hk1_c"].to_numpy(),
        df["setpoint_hk1_c"].to_numpy(),
        circuit="buffer",
    )
    rooms = room_columns(df.reset_index())
    driver = df[rooms].mean(axis=1).to_numpy() - df["t_outdoor_c"].to_numpy()
    vol = next((n.volume_l for n in topo.storage.nodes if n.id == "buffer"), 300.0)
    dis = fit_discharge(
        df["t_hk1_c"].to_numpy(),
        driver,
        df["compressor_on"].to_numpy(),
        dt_s=dt_s,
        volume_l=vol,
        ghi_w_m2=df["ghi_wm2"].to_numpy(),
        window_s=cal.discharge_window_s,
    )
    notes = [
        f"COP real median {hp_fit.cop_median_measured:.2f}; Zeitschritt {dt_s:.0f} s",
        f"Gebäude UA~{dis.ua_w_per_k:.0f} W/K, Solar~{dis.solar_w_per_wm2:.2f} W/(W/m²), "
        f"r²={dis.r2:.2f}",
    ]
    if not np.isnan(thr.start_offset_k):
        notes.append(
            f"Start-Hysterese {abs(thr.start_offset_k):.1f} K, Stopp +{thr.stop_offset_k:.1f} K"
        )
    return (
        topo,
        df,
        CalibrationReport(heat_pump=hp_fit, buffer_threshold=thr, discharge=dis, notes=notes),
    )


def resolve_calibration(settings: Any, model: Any):  # type: ignore[no-untyped-def]
    """Liefert (topo, df, calib): gespeicherte Kalibrierung ODER frisch aus CSV gefittet.

    Bei Fit aus CSV wird die Kalibrierung zusätzlich nach settings.calibration_path gespeichert
    (damit sie ins Bundle exportiert werden kann).
    """
    import json

    from hp_twin.adapters.io_csv import read_measurements
    from hp_twin.domain.calibration import calibration_from_json, calibration_to_json

    topo = load_topology(settings.topology_path)
    if settings.use_stored_calibration and Path(settings.calibration_path).exists():
        calib = calibration_from_json(
            json.loads(Path(settings.calibration_path).read_text("utf-8"))
        )
        df = read_measurements(settings.csv_path) if Path(settings.csv_path).exists() else None
        return topo, df, calib
    topo, df, calib = _calibrate_report(settings.topology_path, settings.csv_path, model=model)
    out = Path(settings.calibration_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(calibration_to_json(calib), indent=2, ensure_ascii=False), "utf-8")
    return topo, df, calib


def _cmd_calibrate(args: argparse.Namespace) -> int:
    import json

    from hp_twin.domain.model_config import ModelConfig

    model = ModelConfig.from_path(args.model)
    _topo, _df, report = _calibrate_report(args.topology, args.csv, model=model)
    out = json.dumps(report.to_dict(), indent=2, ensure_ascii=False)
    if args.out:
        Path(args.out).write_text(out, encoding="utf-8")
        print(f"Kalibrierbericht -> {args.out}")
    print(out)
    return 0


def _cmd_backtest(args: argparse.Namespace) -> int:
    import numpy as np

    from hp_twin.adapters.replay import build_replay
    from hp_twin.domain.backtest import compute_metrics
    from hp_twin.domain.model_config import ModelConfig

    model = ModelConfig.from_path(args.model)
    topo, df, calib = _calibrate_report(args.topology, args.csv, model=model)
    dt_s = float(df.index.to_series().diff().dropna().iloc[0].total_seconds())
    step = max(int(round(args.step_min * 60 / dt_s)), 1)
    horizon_n = int(round(args.horizon_h * 3600 / dt_s))
    starts = list(range(0, len(df) - horizon_n, step))
    if args.max_runs:
        starts = starts[: args.max_runs]

    next_errs, e_mae, cyc_sim, cyc_meas, buf_rmse = [], [], [], [], []
    first = None
    for si in starts:
        rep = build_replay(df, topo, calib, si, args.horizon_h, model=model)
        res, sl = rep.result, rep.measured
        n = len(res.time_s)
        sim_on = np.array([1 if mo in ("heat", "dhw") else 0 for mo in res.mode])
        meas_on = sl["compressor_on"].to_numpy()[:n].astype(int)
        met = compute_metrics(
            res.time_s,
            sim_on,
            meas_on,
            res.p_el_w,
            sl["p_el_w"].to_numpy()[:n],
            res.node_temps_c[rep.buffer_node],
            sl["t_hk1_c"].to_numpy()[:n],
        )
        if not np.isnan(met.next_start_error_min):
            next_errs.append(met.next_start_error_min)
        e_mae.append(met.energy_6h_mae_wh)
        cyc_sim.append(met.cycles_per_day_sim)
        cyc_meas.append(met.cycles_per_day_meas)
        buf_rmse.append(met.buffer_rmse_c)
        if first is None:
            first = (rep, met)

    print(
        f"Rollierender Backtest: {len(starts)} Läufe à {args.horizon_h} h "
        f"(Schritt {args.step_min} min)\n"
    )
    if next_errs:
        a = np.abs(next_errs)
        print(
            f"  Nächster Start |Fehler|: median {np.median(a):.0f} min, "
            f"p90 {np.percentile(a, 90):.0f} min (Bias {np.median(next_errs):+.0f})"
        )
    print(f"  Energie 6h MAE: median {np.median(e_mae):.0f} Wh")
    print(f"  Zyklen/Tag: sim {np.median(cyc_sim):.1f} vs real {np.median(cyc_meas):.1f}")
    print(f"  Puffer-RMSE: median {np.median(buf_rmse):.1f} K")

    if args.plot and first is not None:
        _plot_replay(first[0], first[1], args.plot)
        print(f"\n  Plot (erster Lauf) -> {args.plot}")
    return 0


def _plot_replay(rep: Any, met: Any, out: str) -> None:
    try:
        import matplotlib
    except ImportError:
        print('  (Plot übersprungen: pip install "hp-twin[plots]" für matplotlib)')
        return

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    res, sl = rep.result, rep.measured
    n = len(res.time_s)
    h = res.time_s / 3600
    fig, ax = plt.subplots(2, 1, figsize=(13, 7), sharex=True)
    ax[0].plot(h, sl["t_hk1_c"].to_numpy()[:n], color="tab:blue", label="Puffer gemessen")
    ax[0].plot(
        h, res.node_temps_c[rep.buffer_node], color="tab:orange", ls="--", label="Puffer simuliert"
    )
    ax[0].set_ylabel("Puffer °C")
    ax[0].legend(fontsize=8)
    ax[0].grid(alpha=0.3)
    ax[0].set_title(met.summary(), fontsize=9)
    sim_heat = [1 if mo == "heat" else 0 for mo in res.mode]
    sim_dhw = [1 if mo == "dhw" else 0 for mo in res.mode]
    ax[1].step(
        h, sl["compressor_on"].to_numpy()[:n], where="post", color="tab:blue", label="WP gemessen"
    )
    ax[1].step(h, [0.9 * x for x in sim_heat], where="post", color="tab:orange", label="Sim Heizen")
    ax[1].step(h, [0.85 * x for x in sim_dhw], where="post", color="tab:red", label="Sim DHW")
    ax[1].set_ylabel("WP an")
    ax[1].set_xlabel("Stunden")
    ax[1].legend(fontsize=8)
    ax[1].grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=110)


def _cmd_forecast(args: argparse.Namespace) -> int:
    import json

    from hp_twin.adapters.forecast import run_forecast
    from hp_twin.domain.model_config import ModelConfig

    model = ModelConfig.from_path(args.model)
    # CLI-Overrides auf die zentrale Config anwenden
    model.montecarlo.n_runs = args.runs
    model.forecast.dhw_setpoint_cap_c = args.dhw_cap
    model.forecast.eos_ttl_minutes = args.ttl_minutes
    topo, df, calib = _calibrate_report(args.topology, args.csv, model=model)
    start_idx = len(df) - int(round(args.horizon_h * 60)) - 1 if args.from_end else 0
    start_idx = max(start_idx, 0)
    fc = run_forecast(df, topo, calib, start_idx, horizon_h=args.horizon_h, model=model)
    out = {
        "n_events": len(fc.events),
        "expected_energy_kwh": round(float(fc.slots.energy_wh.sum()) / 1000, 2),
        **fc.to_dict(),
    }
    text = json.dumps(out, indent=2, ensure_ascii=False)
    if args.out:
        Path(args.out).write_text(text, encoding="utf-8")
        print(f"Prognose -> {args.out}  ({len(fc.events)} Events, {fc.slots.n_slots} Slots)")
    else:
        print(text)
    return 0


def _cmd_push(args: argparse.Namespace) -> int:
    import os

    from hp_twin.adapters.eos_connect import push_rest
    from hp_twin.adapters.forecast import run_forecast
    from hp_twin.domain.model_config import ModelConfig

    model = ModelConfig.from_path(args.model)
    model.montecarlo.n_runs = args.runs
    model.forecast.dhw_setpoint_cap_c = args.dhw_cap
    model.forecast.eos_ttl_minutes = args.ttl_minutes
    topo, df, calib = _calibrate_report(args.topology, args.csv, model=model)
    start_idx = max(len(df) - int(round(args.horizon_h * 60)) - 1, 0)
    fc = run_forecast(df, topo, calib, start_idx, horizon_h=args.horizon_h, model=model)
    token = os.environ.get("EOS_TOKEN")
    status = push_rest(args.eos_url, args.managed_load_id, fc.eos, token=token)
    print(
        f"EOS-Push an {args.eos_url} (Load {args.managed_load_id}): HTTP {status}, "
        f"{len(fc.eos.values_w)} Werte"
    )
    return 0


def _cmd_bundle(args: argparse.Namespace) -> int:
    import json

    from hp_twin.adapters.bundle import apply_bundle, build_bundle
    from hp_twin.adapters.settings import RuntimeSettings

    settings = RuntimeSettings.load()
    if args.action == "export":
        b = build_bundle(settings, args.include_secrets, not args.no_calibration)
        Path(args.file).write_text(json.dumps(b, indent=2, ensure_ascii=False), encoding="utf-8")
        print(
            f"Config-Bundle -> {args.file} (Kalibrierung: {'calibration' in b}, "
            f"Secrets: {args.include_secrets})"
        )
    else:
        bundle = json.loads(Path(args.file).read_text(encoding="utf-8"))
        res = apply_bundle(bundle, settings)
        print(f"Bundle angewendet. Geschrieben: {', '.join(res['written'])}")
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="hp-twin")
    sub = p.add_subparsers(dest="cmd", required=True)

    v = sub.add_parser("validate", help="Topologie-YAML validieren")
    v.add_argument("topology")
    v.set_defaults(func=_cmd_validate)

    e = sub.add_parser("export", help="Messdaten aus Home Assistant als CSV exportieren")
    e.add_argument("--mapping", default="config/ha_mapping.yaml")
    e.add_argument("--env", default=".env")
    e.add_argument("--days", type=float, default=10.0)
    e.add_argument("--chunk-hours", type=float, default=6.0)
    e.add_argument("--timeout", type=float, default=120.0)
    e.add_argument("--out", default="data/ha_export/measurements.csv")
    e.add_argument(
        "--append", action="store_true", help="an bestehende CSV anhängen (dedupliziert)"
    )
    e.set_defaults(func=_cmd_export)

    cal = sub.add_parser("calibrate", help="Modellparameter aus Messdaten-CSV fitten")
    cal.add_argument("topology")
    cal.add_argument("--csv", default="data/ha_export/measurements.csv")
    cal.add_argument("--out", default="data/ha_export/calibration.json")
    cal.add_argument("--model", default="config/model.yaml")
    cal.set_defaults(func=_cmd_calibrate)

    bt = sub.add_parser("backtest", help="Rollierender Backtest gegen Messdaten")
    bt.add_argument("topology")
    bt.add_argument("--csv", default="data/ha_export/measurements.csv")
    bt.add_argument("--horizon-h", type=float, default=24.0)
    bt.add_argument("--step-min", type=float, default=180.0)
    bt.add_argument("--max-runs", type=int, default=0)
    bt.add_argument("--plot", default="data/ha_export/backtest.png")
    bt.add_argument("--model", default="config/model.yaml")
    bt.set_defaults(func=_cmd_backtest)

    fc = sub.add_parser("forecast", help="48-h-Lastprognose (Monte-Carlo) + EOS-Payload")
    fc.add_argument("topology")
    fc.add_argument("--csv", default="data/ha_export/measurements.csv")
    fc.add_argument("--horizon-h", type=float, default=48.0)
    fc.add_argument("--runs", type=int, default=100)
    fc.add_argument("--dhw-cap", type=float, default=45.0)
    fc.add_argument("--ttl-minutes", type=int, default=None)
    fc.add_argument("--from-end", action="store_true", help="Startzustand = Ende der CSV")
    fc.add_argument("--out", default=None)
    fc.add_argument("--model", default="config/model.yaml")
    fc.set_defaults(func=_cmd_forecast)

    pu = sub.add_parser("push", help="Prognose rechnen und an EOS Connect senden (REST)")
    pu.add_argument("topology")
    pu.add_argument("eos_url", help="Basis-URL der EOS-Connect-Instanz")
    pu.add_argument("managed_load_id", help="Managed-Load-ID in EOS Connect")
    pu.add_argument("--csv", default="data/ha_export/measurements.csv")
    pu.add_argument("--horizon-h", type=float, default=48.0)
    pu.add_argument("--runs", type=int, default=100)
    pu.add_argument("--dhw-cap", type=float, default=45.0)
    pu.add_argument("--ttl-minutes", type=int, default=None)
    pu.add_argument("--model", default="config/model.yaml")
    pu.set_defaults(func=_cmd_push)

    bd = sub.add_parser("bundle", help="Konfig+Kalibrierung als eine Datei exportieren/importieren")
    bd.add_argument("action", choices=["export", "import"])
    bd.add_argument("file")
    bd.add_argument("--include-secrets", action="store_true")
    bd.add_argument("--no-calibration", action="store_true")
    bd.set_defaults(func=_cmd_bundle)

    args = p.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    sys.exit(main())
