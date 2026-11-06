"""REST-API (dünne Schicht über die Domain). POST /simulate -> 48-h-Lastprognose, GET /health."""

from __future__ import annotations

import asyncio
import contextlib
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from hp_twin.adapters.service import ServiceStatus, data_age_minutes, run_cycle
from hp_twin.adapters.settings import RuntimeSettings

_STATIC = Path(__file__).parent / "static"
_STATUS = ServiceStatus()


async def _scheduler(app: FastAPI) -> None:
    """Hintergrundschleife: fährt bei aktivierter Automatik zyklisch den Push-Zyklus."""
    while True:
        settings = RuntimeSettings.load()
        interval = max(settings.schedule_interval_minutes, 1.0) * 60.0
        if settings.auto_push_enabled:
            try:
                st = await asyncio.get_event_loop().run_in_executor(None, run_cycle, settings, True)
                globals()["_STATUS"] = st
            except Exception as e:  # pragma: no cover - defensive
                _STATUS.error = str(e)
        await asyncio.sleep(interval)


@contextlib.asynccontextmanager
async def _lifespan(app: FastAPI) -> AsyncIterator[None]:
    task = asyncio.create_task(_scheduler(app))
    yield
    task.cancel()
    with contextlib.suppress(asyncio.CancelledError):
        await task


app = FastAPI(title="hp-twin", version="0.1.0", lifespan=_lifespan)


@app.get("/")
def dashboard() -> FileResponse:
    return FileResponse(_STATIC / "dashboard.html")


@app.get("/api/analysis")
def analysis(
    topology_path: str = "config/example_topology.yaml",
    csv_path: str = "data/ha_export/measurements.csv",
    model_path: str = "config/model.yaml",
    runs: int = 50,
    horizon_h: float = 48.0,
    from_end: bool = True,
) -> dict[str, Any]:
    from hp_twin.adapters.analysis import build_analysis
    from hp_twin.cli import _calibrate_report
    from hp_twin.domain.model_config import ModelConfig

    model = ModelConfig.from_path(model_path)
    model.montecarlo.n_runs = runs
    model.sim.horizon_h = horizon_h
    try:
        topo, df, calib = _calibrate_report(topology_path, csv_path, model=model)
    except FileNotFoundError as e:  # pragma: no cover - IO-abhängig
        raise HTTPException(status_code=404, detail=str(e)) from e
    start_idx = max(len(df) - int(round(horizon_h * 60)) - 1, 0) if from_end else 0
    return build_analysis(df, topo, calib, start_idx, model)


class SimulateRequest(BaseModel):
    topology_path: str = "config/example_topology.yaml"
    csv_path: str = "data/ha_export/measurements.csv"
    model_path: str = "config/model.yaml"
    horizon_h: float = Field(default=48.0, gt=0)
    runs: int = Field(default=100, ge=1)
    dhw_setpoint_cap_c: float | None = 45.0
    ttl_minutes: int | None = None
    from_end: bool = True


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


# ------------------------------------------------------------ Settings / Automatik
@app.get("/api/settings")
def get_settings() -> dict[str, Any]:
    return RuntimeSettings.load().public_dict()


@app.post("/api/settings")
def post_settings(patch: dict[str, Any]) -> dict[str, Any]:
    """Teil-Update der Einstellungen. Leeres eos_token lässt das bestehende Token unangetastet."""
    cur = RuntimeSettings.load()
    data = cur.model_dump()
    for k, v in patch.items():
        if k in data:
            data[k] = v
    if not patch.get("eos_token"):  # leeres Feld -> Token behalten
        data["eos_token"] = cur.eos_token
    if not patch.get("ha_token"):
        data["ha_token"] = cur.ha_token
    new = RuntimeSettings.model_validate(data)
    new.save()
    return new.public_dict()


@app.get("/api/model")
def get_model() -> dict[str, Any]:
    from hp_twin.domain.model_config import ModelConfig

    s = RuntimeSettings.load()
    return ModelConfig.from_path(s.model_path).model_dump()


@app.post("/api/model")
def post_model(patch: dict[str, Any]) -> dict[str, Any]:
    """Speichert Modell-Parameter (verschachtelt) nach model_path (config/model.yaml)."""
    import yaml

    from hp_twin.domain.model_config import ModelConfig

    s = RuntimeSettings.load()
    cur = ModelConfig.from_path(s.model_path).model_dump()
    for section, vals in patch.items():
        if section in cur and isinstance(vals, dict):
            cur[section].update(vals)
    model = ModelConfig.model_validate(cur)  # validiert
    Path(s.model_path).write_text(
        yaml.safe_dump(model.model_dump(), sort_keys=False, allow_unicode=True), encoding="utf-8"
    )
    return model.model_dump()


@app.get("/api/status")
def status() -> dict[str, Any]:
    s = RuntimeSettings.load()
    out = _STATUS.to_dict()
    out["data_age_minutes_live"] = data_age_minutes(s.csv_path)
    out["auto_push_enabled"] = s.auto_push_enabled
    out["schedule_interval_minutes"] = s.schedule_interval_minutes
    out["eos_configured"] = bool(s.eos_base_url and s.eos_managed_load_id)
    return out


@app.post("/api/push_now")
async def push_now(body: dict[str, Any] | None = None) -> dict[str, Any]:
    """Führt sofort einen Zyklus aus (Export->Forecast->Push). body={"push":false} = Dry-Run."""
    settings = RuntimeSettings.load()
    do_push = True if body is None else bool(body.get("push", True))
    st = await asyncio.get_event_loop().run_in_executor(None, run_cycle, settings, do_push)
    globals()["_STATUS"] = st
    return st.to_dict()


@app.post("/simulate")
def simulate(req: SimulateRequest) -> dict[str, Any]:
    from hp_twin.adapters.forecast import run_forecast
    from hp_twin.cli import _calibrate_report
    from hp_twin.domain.model_config import ModelConfig

    model = ModelConfig.from_path(req.model_path)
    model.montecarlo.n_runs = req.runs
    model.forecast.dhw_setpoint_cap_c = req.dhw_setpoint_cap_c
    model.forecast.eos_ttl_minutes = req.ttl_minutes
    try:
        topo, df, calib = _calibrate_report(req.topology_path, req.csv_path, model=model)
    except FileNotFoundError as e:  # pragma: no cover - IO-abhängig
        raise HTTPException(status_code=404, detail=str(e)) from e
    start_idx = max(len(df) - int(round(req.horizon_h * 60)) - 1, 0) if req.from_end else 0
    fc = run_forecast(df, topo, calib, start_idx, horizon_h=req.horizon_h, model=model)
    return {"n_events": len(fc.events), **fc.to_dict()}
