# hp-twin

Digitaler Zwilling einer Wärmepumpen-Anlage: **zustandsbasierte 48-h-Lastprognose** (Monte-Carlo)
für das Energiemanagement **EOS Connect**, mit Troubleshooting-Dashboard. Statt historische
Lastprofile zu mitteln, wird vorwärts simuliert: gemessener Speicherzustand + Wetter + Gebäude-
modell + Reglerlogik → wann/wie lange/in welchem Modus/mit welcher Leistung die WP läuft.

Siehe `CLAUDE.md` (Konventionen), `data/schema.md` (CSV-Schema), `config/*.example.yaml` (Vorlagen).

## Architektur (kurz)
- `src/hp_twin/domain/` – reine Simulationslogik (Speicher, WP-Kennfeld, Regler, Gebäude, Monte-Carlo,
  Kalibrierung, Backtest), ohne IO. Als Bibliothek in EOS Connect einbindbar.
- `src/hp_twin/adapters/` – Home-Assistant-Export, Replay/Backtest, Prognose, EOS-Connect-Output.
- `src/hp_twin/api/` – FastAPI (Dashboard + REST). `cli.py` – Kommandozeile.

## Konfiguration (alles konfigurierbar, nichts hartcodiert)
| Datei | Inhalt |
|---|---|
| `.env` | Secrets: `HA_BASE_URL`, `HA_TOKEN` (gitignored; Vorlage `.env.example`) |
| `config/example_topology.yaml` | Anlage: Speicher/Knoten/Kopplung, Erzeuger, Consumer, Regler |
| `config/model.yaml` | **alle** Modell-/Sim-/Monte-Carlo-/Kalibrier-Parameter (Vorlage `model.example.yaml`) |
| `config/ha_mapping.yaml` | HA-Entity → CSV-Schema (Vorlage `ha_mapping.example.yaml`) |
| `config/heizlast.yaml` | Raumweise Norm-Heizlast + Thermostat-Zuordnung (Vorlage `heizlast.example.yaml`) |

Installationsspezifische Configs und alle Mess-/Ergebnisdaten sind **gitignored** – ins Repo
kommen nur Code und Vorlagen. Kopiere die `*.example.yaml` auf den Namen ohne `.example` und passe sie an.

## Setup (lokal)
```bash
uv venv --python 3.12 && uv pip install -e ".[dev]"
cp .env.example .env        # HA_BASE_URL + HA_TOKEN eintragen
cp config/ha_mapping.example.yaml config/ha_mapping.yaml   # Entities anpassen
cp config/heizlast.example.yaml   config/heizlast.yaml     # Heizlast eintragen
cp config/model.example.yaml      config/model.yaml        # Parameter (optional) anpassen
```

## Workflow
1. **Daten exportieren** (HA-History hält nur ~10–14 Tage → regelmäßig anhängen):
   `hp-twin export --days 10 --out data/ha_export/measurements.csv`
   Periodisch (Cron, täglich 03:00): `hp-twin export --days 2 --append --out data/ha_export/measurements.csv`
2. **Kalibrieren** (WP-Kennfeld, Regler-Schwellen, Gebäude-UA): `hp-twin calibrate config/example_topology.yaml`
3. **Backtest** gegen die Realität: `hp-twin backtest config/example_topology.yaml --plot data/ha_export/backtest.png`
4. **Prognose** (48 h, Monte-Carlo): `hp-twin forecast config/example_topology.yaml --from-end --runs 100`
5. **An EOS Connect senden**: `hp-twin push config/example_topology.yaml http://<eos-host>:8080 <managed_load_id>`
   (Token optional über `EOS_TOKEN`.)

## Dashboard (Troubleshooting)
```bash
uvicorn hp_twin.api.app:app --host 0.0.0.0 --port 8000
```
- `http://localhost:8000/` – interaktives Dashboard: Prognoseband (P10/P50/P90), Laufwahrscheinlichkeit,
  Sim-vs-Mess-Temperaturen, Einflussfaktoren (Wetter/Sollwerte/SG/aktive Räume), Kalibrierparameter
  („wie die Berechnung zustande kommt"), Ereignisliste, EOS-Payload, Konfiguration.
- REST: `POST /simulate` (Prognose-JSON), `GET /api/analysis` (Analyse-Bundle fürs Dashboard), `GET /health`.

## Automatik (vollautomatischer Push)
Im Dashboard unter **„⚙️ Automatik & EOS-Connect-Verbindung"** einstellbar und persistent
(`config/runtime.json`, gitignored):
- EOS Basis-URL, Managed-Load-ID, optional Token; MC-Läufe, Horizont, DHW-Cap, Export-Fenster, Intervall.
- **Automatik aktiv** → ein Hintergrund-Scheduler fährt zyklisch: frischer HA-Export → Kalibrierung →
  Prognose → EOS-Push. **„Push jetzt"** / **„Dry-Run"** lösen sofort einen Zyklus aus.
- **Start-Abgleich**: Vor jedem Push wird ein kurzes Fenster frisch exportiert, sodass der simulierte
  Startzustand ≈ „jetzt" ist. Das **Datenalter** wird im Status angezeigt (grün = synchron < 30 min);
  der EOS-Payload wird mit `"start":"now"` gesendet → beide Systeme teilen dieselbe Zeitachse.
- Endpunkte: `GET/POST /api/settings`, `POST /api/push_now` (`{"push":false}` = Dry-Run), `GET /api/status`.

## Portable Konfiguration (Export/Import)
Gesamte Konfiguration (+ optional Kalibrierung) als **eine Datei** sichern und in einer neuen
Installation wiederherstellen – im Dashboard („Portable Konfiguration": ⬇ exportieren / ⬆ importieren)
oder per CLI:
```bash
hp-twin bundle export hp-twin-config.json            # ohne Secrets
hp-twin bundle export hp-twin-config.json --include-secrets
hp-twin bundle import hp-twin-config.json            # in neuer Installation anwenden
```
Das Bundle enthält Topologie, Modellparameter, HA-Mapping, Heizlast, Laufzeit-Einstellungen und
die Kalibrierung. **Tokens sind standardmäßig nicht enthalten**; beim Import bleiben vorhandene
Secrets erhalten. Mit `use_stored_calibration` nutzt die neue Installation die mitgelieferte
Kalibrierung sofort (statt erst aus Messdaten zu fitten). Endpunkte: `GET/POST /api/bundle`.

## EOS-Connect-Anbindung
- Format („External load profile", Managed Load, forecast only):
  `{"values": [192×W], "unit": "W", "start": "now", "ttl_minutes"?}` – **gesamte** WP-Leistung
  (in EOS Connect „Replaces sensor" auf den WP-Leistungssensor setzen, damit nichts doppelt zählt).
- Push: REST `POST <eos>/api/managed_loads/<id>/push` (`hp-twin push ...`) **oder** MQTT retained
  auf `eos_connect/managed_load/<id>/set` (Topic via `hp_twin.adapters.eos_connect.mqtt_topic`).
- DHW: neue Konfig (SG3-Soll = Tagestemperatur) → DHW passiv über HK1; `--dhw-cap 45`.

## Installation in Home Assistant (Add-on)
EOS Connect läuft als HA-Add-on; hp-twin genauso. Siehe `hp-twin-addon/DOCS.md`. Kurz:
1. **Einstellungen → Add-ons → Add-on-Store → ⋮ → Repositories** → URL dieses GitHub-Repos hinzufügen.
   (In `hp-twin-addon/build.yaml` den eigenen Fork als `REPO_URL` eintragen.)
2. Add-on **hp-twin Wärmepumpen-Zwilling** installieren.
3. `config/*.yaml` nach `/config` im HA-Dateisystem kopieren (Samba-/VS-Code-Add-on) und in den
   Add-on-Optionen Pfade/Parameter setzen (`topology`, `model`, `forecast_runs`, `eos_base_url`, …).
4. Starten – das Dashboard erscheint per **Ingress** in der HA-Seitenleiste. Die HA-Sensoren liest
   das Add-on über den **Supervisor-Proxy** (`http://supervisor/core`), ein Long-Lived-Token ist
   nicht nötig.

## Tests / Qualität
`pytest` · `ruff check . && ruff format --check .` · `mypy src/hp_twin` · Docker: `docker compose up`.
