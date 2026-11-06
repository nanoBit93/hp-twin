# hp-twin – Projektkonventionen

Digitaler Zwilling einer Wärmepumpen-Anlage (Stiebel Eltron WPL-A 25 + WPM) → 48-h-Lastprognose für EOS Connect.
Zustandsbasierte Vorwärtssimulation (kein Profil-Mitteln). Sprache: Doku/Kommentare Deutsch, Code-Bezeichner Englisch.

## Architektur
- `src/hp_twin/domain/`: reine Simulationslogik. **Keine** Imports von FastAPI, Dateisystem-IO, Netzwerk. mypy strict.
- `ports/`: Protocols (StateProvider, WeatherProvider, SetpointProvider, LoadPublisher).
- `adapters/`: CSV/YAML-IO, EOS-Connect-Format; später HA/Modbus/MQTT/Wetter.
- `api/`, `cli.py`: dünne Schicht über domain. Abhängigkeit nur nach innen: api/cli/adapters → ports ← domain.
- Anlage wird komplett per YAML beschrieben (`config/*.yaml`), validiert mit pydantic v2 (`domain/topology.py`).
- Neue Anlagen-Bausteine = neue Implementierung eines Protocols, kein Umbau des Simulators.

## Konventionen
- Python 3.12+, `src/`-Layout, SI-Einheiten intern (K/°C, W, J, s); Einheit im Namen (`t_dhw_c`, `p_el_w`).
- Zeit intern UTC (tz-aware), Anzeige Europe/Berlin. Feste Zeitschritte (Default 60 s).
- Simulation deterministisch; Zufall nur in `montecarlo.py` mit explizitem `numpy.random.Generator`/Seed.
- Sollwerte/Modi sind Zeitreihen-Eingänge (`ControlInputs`) – SG-Ready später nur ein weiterer Input.
- Testwerte aus dem Anlagen-Tagesverlauf sind Plausibilitätsreferenzen (Toleranzen), nie hart codiert.
- Energieerhaltung im Speicher ist eine Test-Invariante.

## Anlage (Phase 1)
1 Verdichter, 1 Umschaltventil, 600-l-Kombispeicher (2 Knoten 50/50, DHW oben), Kreise DHW/HK1/HK2,
Warmwasservorrang, Parallelbetrieb, min_off 30 min, Heizstab nur Notbetrieb (Phase 1 aus).

## Befehle
- Setup: `uv venv --python 3.12 && uv pip install -e ".[dev]"`
- Tests: `pytest` · Lint: `ruff check . && ruff format --check .` · Typen: `mypy src/hp_twin`
