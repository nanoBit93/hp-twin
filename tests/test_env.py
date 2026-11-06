from pathlib import Path

from hp_twin.adapters.env import load_dotenv


def test_load_dotenv(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.delenv("HA_TOKEN", raising=False)
    f = tmp_path / ".env"
    f.write_text('HA_BASE_URL="http://x:8123"\n# comment\nHA_TOKEN=abc\n')
    vals = load_dotenv(f)
    assert vals["HA_BASE_URL"] == "http://x:8123"
    assert vals["HA_TOKEN"] == "abc"


def test_missing_env_returns_empty(tmp_path: Path) -> None:
    assert load_dotenv(tmp_path / "nope.env") == {}
