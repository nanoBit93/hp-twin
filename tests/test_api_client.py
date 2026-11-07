"""Tests für den Dev-API-Client (`adapters.api_client`) und die `remote`-CLI-Befehle."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest


class _FakeServer:
    """Kleiner Fake der Dev-API (Antworten je Pfad; Token-Prüfung wie in security.py)."""

    token: str | None = "t"

    def __init__(self) -> None:
        self.calls: list[tuple[str, str]] = []

    def handle(self, method: str, path: str, headers: dict[str, str]) -> tuple[int, Any]:
        self.calls.append((method, path))
        auth = headers.get("Authorization", "")
        wanted = auth.startswith("Bearer ") and auth[7:] == self.token
        if self.token and not wanted:
            return 401, {"detail": "Token ungültig/fehlt."}
        if path == "/health":
            return 200, {"status": "ok"}
        if path == "/api/version":
            return 200, {"version": "9.9.9"}
        if path == "/api/dev/state":
            return 200, {
                "nodes_c": {"dhw": 40.0, "buffer": 30.0},
                "controller_state": {"mode": "idle"},
            }
        if path.startswith("/api/dev/data"):
            return 200, {"n_rows": 1, "window": None}
        if path.startswith("/api/dev/trace"):
            return 200, {"trace": [{"index": 0}], "summary": {}, "compare": {}}
        if path == "/api/dev/scenario":
            return 200, {"summary": {"energy_kwh": 1.5, "n_cycles": 2}}
        if path == "/api/dev/scenarios":
            return 200, {
                "n": 2,
                "results": [
                    {"index": 0, "summary": {"energy_kwh": 1.5, "n_cycles": 2}},
                    {"index": 1, "summary": {"energy_kwh": 2.0, "n_cycles": 3}},
                ],
                "energy_kwh": [1.5, 2.0],
            }
        return 404, {"detail": "not found"}


@pytest.fixture()
def fake(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    """Überschreibt die HTTP-Methoden des Clients mit dem Fake-Server."""
    from hp_twin.adapters import api_client

    server = _FakeServer()
    monkeypatch.delenv("HP_TWIN_API_URL", raising=False)
    monkeypatch.delenv("HP_TWIN_API_TOKEN", raising=False)

    def fake_get(self: api_client.DevApiClient, path: str) -> Any:
        code, body = server.handle("GET", path, self._headers())
        if code == 401:
            raise RuntimeError(f"HTTP 401 bei GET {path}: Token ungültig/fehlt.")
        return body

    def fake_post(self: api_client.DevApiClient, path: str, body: Any) -> Any:
        _ = body
        code, resp = server.handle("POST", path, self._headers())
        if code == 401:
            raise RuntimeError(f"HTTP 401 bei POST {path}: Token ungültig/fehlt.")
        return resp

    monkeypatch.setattr(api_client.DevApiClient, "_get_json", fake_get)
    monkeypatch.setattr(api_client.DevApiClient, "_post_json", fake_post)

    env = tmp_path / ".env"
    env.write_text("HP_TWIN_API_URL=http://fake:8000\nHP_TWIN_API_TOKEN=t\n", encoding="utf-8")
    return server, str(env)


def test_client_from_env_and_reads(fake) -> None:
    server, env = fake
    from hp_twin.adapters.api_client import DevApiClient

    c = DevApiClient.from_env(env)
    assert c.state()["nodes_c"]["buffer"] == 30.0
    assert c.version() == {"version": "9.9.9"}
    assert c.health() == {"status": "ok"}
    assert ("GET", "/api/dev/state") in server.calls


def test_client_query_encoding(fake) -> None:
    from hp_twin.adapters import api_client

    server, env = fake
    c = api_client.DevApiClient.from_env(env)
    c.data(start="2026-11-06T00:00:00+00:00", end="2026-11-06T01:00:00+00:00")
    c.trace(horizon_h=6.0, trace_step_s=120.0)
    paths = [p for _m, p in server.calls]
    assert any("start=2026-11-06T00%3A00%3A00%2B00%3A00" in p for p in paths)
    assert any("trace_step_s=120.0" in p for p in paths)


def test_client_scenario_and_batch(fake) -> None:
    _server, env = fake
    from hp_twin.adapters.api_client import DevApiClient

    c = DevApiClient.from_env(env)
    s = c.scenario(horizon_h=6, overrides={"t_outdoor_offset_k": -5})
    assert s["summary"]["energy_kwh"] == 1.5
    b = c.scenarios([{"horizon_h": 6}, {"horizon_h": 6, "overrides": {}}])
    assert b["energy_kwh"] == [1.5, 2.0]
    cmp_ = c.compare_scenarios([{"horizon_h": 6}])
    assert cmp_["0"]["energy_kwh"] == 1.5


def test_client_missing_url_error(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("HP_TWIN_API_URL", raising=False)
    monkeypatch.delenv("HP_TWIN_API_TOKEN", raising=False)
    from hp_twin.adapters.api_client import DevApiClient

    with pytest.raises(ValueError, match="HP_TWIN_API_URL"):
        DevApiClient.from_env(str(tmp_path / "leer.env"))


def test_client_rejects_non_object(fake) -> None:
    _server, env = fake
    from hp_twin.adapters import api_client

    def bad_get(self: api_client.DevApiClient, path: str) -> Any:
        return [1, 2, 3]  # kein Objekt

    api_client.DevApiClient._get_json = bad_get
    c = api_client.DevApiClient.from_env(env)
    with pytest.raises(RuntimeError, match="kein JSON-Objekt"):
        c.state()


def test_cli_remote_state(
    fake, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    _server, env = fake
    from hp_twin import cli

    argv = ["remote", "state", "--env", env]
    assert cli.main(argv) == 0
    out = json.loads(capsys.readouterr().out)
    assert out["nodes_c"]["buffer"] == 30.0


def test_cli_remote_scenario_with_overrides_file(
    fake, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    _server, env = fake
    from hp_twin import cli

    ov = tmp_path / "ov.json"
    ov.write_text(json.dumps({"t_outdoor_offset_k": -5}), encoding="utf-8")
    argv = ["remote-scenario", "--horizon-h", "6", "--overrides", str(ov), "--env", env]
    assert cli.main(argv) == 0
    out = json.loads(capsys.readouterr().out)
    assert out["summary"]["energy_kwh"] == 1.5


def test_cli_remote_error_exits_1(
    fake, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    from hp_twin import cli

    env = tmp_path / "no_token.env"
    env.write_text("HP_TWIN_API_URL=http://fake:8000\n", encoding="utf-8")  # Token fehlt -> 401
    assert cli.main(["remote", "state", "--env", str(env)]) == 1
    assert "HTTP 401" in capsys.readouterr().err


def test_cli_requires_url(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.delenv("HP_TWIN_API_URL", raising=False)
    monkeypatch.delenv("HP_TWIN_API_TOKEN", raising=False)
    from hp_twin import cli

    assert cli.main(["remote", "state", "--env", str(tmp_path / "none.env")]) == 1
    assert "HP_TWIN_API_URL" in capsys.readouterr().err
