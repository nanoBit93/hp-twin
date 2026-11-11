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
uv venv --python 3.12 && uv pip install -e ".[dev]"   # [plots] = matplotlib für CLI-Plots
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
4. **Prognose** (48 h, Monte-Carlo, ab letztem Messpunkt): `hp-twin forecast config/example_topology.yaml --runs 100`
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
- **Echte Vorwärtsprognose ab „jetzt"**: Vor jedem Push wird frisch exportiert. Der Startzustand
  (Puffer-/DHW-Temperatur, Reglerzustand inkl. Mindeststillstand) kommt aus dem **letzten Messpunkt**,
  die Simulation läuft von dort 48 h in die Zukunft. Die zukünftigen **Eingänge** (Außentemperatur,
  Strahlung, Heizkurven-/Raumsollwerte, SG) sind eine **Näherung: die letzten 24 h, täglich wiederholt**
  („gestern wie heute") – ein Wetterprognose-Adapter kann sie später ersetzen. Das Datenalter steht im Status.
- **Erster Start / Lücken**: Fehlt Historie (< `min_history_days`), lädt der Zyklus automatisch
  `backfill_days` (Default 10) aus HA nach – im Hintergrund, Fortschritt im Status. Größere Lücken
  starten die Historie neu.
- **Kalibrierung**: Bei < `min_fit_days` (Default 5) Historie wird die mitgelieferte/gespeicherte
  Kalibrierung benutzt (Bundle-Import) statt aus wenigen Stunden zu fitten.
- Endpunkte: `GET/POST /api/settings`, `POST /api/push_now` (startet im Hintergrund; `{"push":false}` = Dry-Run), `GET /api/status`.
Kommt ein weiterer Refresh während eines laufenden Zyklus an, wird genau ein Folgezyklus mit den
neuesten Einstellungen nachgeholt, statt die Anforderung zu verwerfen.

## Analyse-/Dev-API (Fernanschluss für die Entwicklung)
Die laufende Instanz (HA Add-on) ist über die **Analyse-API** von der Entwickler-Maschine aus
diagnostizierbar: Daten-/Startzustandsabfrage, Kalibrierung, **Zwischenschritt-Trace** (Regler-
entscheid je Zeitschritt) und **Was-wäre-wen-Szenarien** mit Overrides auf Eingänge/Modell.
Die verbindliche Referenz (Endpunkte, Auth, Python-Client, CLI, Debug-Rezepte) steht in
**[docs/dev-api.md](docs/dev-api.md)**. Kurz:
- Konfig auf der Entwickler-Maschine: `HP_TWIN_API_URL` + `HP_TWIN_API_TOKEN` in `.env`
  (Add-on-Option `api_token` in HA setzen, der Port 8000 ist seit v0.1.4 im LAN offen).
- Python: `from hp_twin.adapters.api_client import DevApiClient` · CLI: `hp-twin remote state`.
- Schreib-Endpunkte (`POST /api/settings`, `/api/model`, `/api/bundle`, `/api/push_now`) sind
  gleich mitgeschützt; über den Dashboard-Ingress entfällt der Token.

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

### SG3-Flexibilitätszyklus (mit EOS Connect)

Wenn **SG3-Planung aktiv** ist, liefert der Zwilling ein verschiebbares Kontingent an die Load-ID
`sg_load_id` (Standard `sg3_req`). EOS Connect platziert es mit dem eingebauten `local_evopt` gemeinsam
mit Batterie, PV und Netzbezug. Das finale Profil geht an `profile_load_id` (Standard `hp_ext`).
**Replaces sensor** dieses Profils muss auf den WP-Leistungssensor zeigen; das Kontingent selbst wird
nach Adoption zurückgezogen. So ist die SG3-Energie nur einmal im System.

- **Horizont:** 48 h, 15-min-Slots. Die Wirtschaftlichkeit wird nur für vollständig bepreiste
  EOS-Slots bewertet. Fehlen Preise am Horizontende, wird diese Energie nicht geschätzt oder
  verworfen; die Preisabdeckung steht in den SG3-Notizen. Ganz ohne bepreiste Energie bleibt
  die SG2-Prognose aktiv (Degraded statt blindes SG3). Beginnt die Simulation innerhalb eines
  Viertelstunden-Slots, verwendet sie den Preis dieses bereits laufenden Slots; der tatsächliche
  Messbeginn muss nicht auf :00/:15/:30/:45 liegen. Dasselbe Slot-Mapping gilt beim Zurücklesen
  des EOS-Placements und bei einer offenen Adoption nach Ablauf der Push-TTL: Die TTL verhindert
  nur neue Platzierungen, nicht die Übernahme eines bereits geplanten Blocks. Die Hysterese
  verhindert ebenfalls nur einen erneuten Push, nicht das Lesen und Adoptieren dieses Plans.
- **Feasibility:** erlaubte Startslots, keine bloße Temperaturschwelle. Berücksichtigt werden laufende
  Basisladung, aktuelle und prognostizierte `min_off`-Sperren, `min_on`, Horizontende, COP und die
  konfigurierbare Bivalenz-Außentemperatur `sg_bivalence_outdoor_c` (Default 0 °C). Eine aktuell
  laufende Basisladung sperrt nur bis zu ihrem prognostizierten Stopp (plus `min_off`), nicht den
  gesamten Horizont. Der aktuelle
  Simulationspfad modelliert keinen parallel zugeschalteten Heizstab; deshalb wird unterhalb dieser
  Grenze kein SG3 angeboten. `sg_top_setpoint_c` steuert den SG3-Sollwert (Default 49,8 °C,
  Median der gemessenen Referenzanlage). Das HK1-Ladeziel wird aus vollständigen historischen
  SG3-Läufen kalibriert: Nur ein Stopp am erhöhten Sollwert zählt; ein früher Abbruch unter dem
  Ziel ist kein abgeschlossener Lauf. Damit bleibt der normale SG2-Stopp separat und verzerrt das
  SG3-Ziel nicht. In der Referenzhistorie liegt der SG3-Stopp etwa 1,6 K über dem Sollwert.

  Jeder freie Startbereich wird zunächst über bis zu drei Stunden simuliert. Erst der tatsächliche
  Stopp am HK1-Ziel bestimmt Laufzeit und zusätzliche elektrische Energie des Angebots. Ein kurzer
  freier Bereich oder ein einzelner EOS-Slot ist daher **kein** abgeschlossener SG3-Lauf. Kann das
  Ziel in dieser Zeit nicht erreicht werden, wird der Kandidat verworfen. Nach dem Zielstopp bleibt
  der Verdichter für den Rest des SG3-Sollfensters aus; die folgende Pause bis zum nächsten normalen
  Lauf wird separat ausgewiesen. In der lokalen Referenzhistorie dauern vollständige Läufe typischerweise
  2–3 Stunden, gefolgt von mehrstündigen Pausen. Eine reine Lastverschiebung spart keine Energie, kann
  aber Stromkosten senken. EOS erhält die zusätzliche elektrische Energie **des nachgewiesenen Laufs**
  gegenüber der Basisprognose, nicht die Netto-Differenz über den gesamten Horizont. Später entfallene
  SG2-Läufe mindern die Kosten, ohne dieses flexible Angebot zu löschen. Der normale WP-Grundbetrieb
  wird nicht doppelt gezählt.
- **Wert:** `value_eur_per_kwh` ist der Brutto-Vermeidungswert der Szenario-Differenz vor den Kosten
  der SG3-Zusatzenergie. EOS vergleicht ihn mit den tatsächlichen Versorgungskosten.
  `saving_eur = Kosten(SG2) − Kosten(SG3)` ist nur eine Dashboard-Kennzahl und wird nicht nochmals als
  Solver-Reward verwendet. `total_wh` ist die zusätzliche elektrische Energie des SG3-Fensters.
  Auch bei null oder negativem Netto-Mehrverbrauch über den Horizont kann ein Angebot entstehen:
  Entscheidend sind geringere Versorgungskosten durch die Verschiebung, nicht Energieeinsparung.
- **Zeitvertrag:** Contingent und Maske tragen `start` (ISO), `resolution: 15min` und eine
  absolute ISO-`deadline` für das Ende des simulierten Sollfensters. `min_runtime_slots`
  beschreibt die tatsächlich simulierte, auf EOS-Slots aufgerundete Kompressorlaufzeit;
  `contiguous: true` verhindert Unterbrechungen und Nullenergie-Slots. Der Lauf wird nicht
  künstlich auf einen vollen Sollfensterzeitraum verlängert. EOS Connect verteilt die
  Szenario-Zusatzenergie über diesen Block; das ist keine Forderung nach Nennleistung in jedem Slot.
  Ein früher Abbruch durch reale Anlagenbedingungen, etwa eine nicht ausreichende PV-Versorgung,
  wird dadurch nicht als gewünschter Kurzlauf geplant. Das EOS-Placement wird mit `plan_anchor` und
  `time_frame_base` zurückübersetzt und nur als **genau ein Block der angebotenen Dauer** übernommen.
  Vor der Endprognose muss der tatsächlich platzierte Start erneut den thermischen Zielstopp erreichen.
  `plan[]` enthält Wh, nicht W.
- **Betrieb:** `sg_contingent_ttl_minutes` begrenzt die Gültigkeit. Es gibt **keine feste
  PV-Schwelle und kein Preisdach mehr**. PV-Prognose und Tarife dienen nur der Diagnose;
  EOS Connect entscheidet anhand Kontingent, Machbarkeit und Brutto-Vermeidungswert, wie
  die Energie mit Netz, PV und Batterie platziert wird. Bestehende JSON-Konfigurationen
  mit den alten Schlüsseln `sg_min_pv_surplus_w`, `sg_max_price_eur` und
  `sg_emergency_gate` werden beim Laden ignoriert. `sg_dry_run` pusht nichts. Außerhalb
  eines Startfensters werden alle konfigurierten Fenster im 48-h-Horizont geprüft, einschließlich
  des Folgetags. Die reale SG-Ready-Ausführung am WPM (Phase 3) ist nicht Bestandteil dieses Zyklus.

**Einstellungen im Dashboard:** `SG3-Load-ID`, `Profil-Load-ID`, `SG3-Bivalenz`,
`Contingent-TTL`, `SG3-Planung aktiv` und `SG3 Dry-Run`. Alle Felder werden über
`/api/settings` gespeichert und im portablen Konfigurations-Bundle mitgeführt. Für die erste Inbetriebnahme
`sg_dry_run = true` lassen und Kandidaten/Maske prüfen. Der Produktivbetrieb erfordert `sg_enabled = true`,
48 h Horizont, erreichbare EOS-URL und Preise für alle 48 h. Mindestlaufzeit und Max-Preis werden am
`sg3_req`-Load in EOS Connect gesetzt, nicht im hp-twin-Dashboard.

Der Status und das Dashboard zeigen Kandidaten, zulässige Startslots, `saving_eur`, Bruttowert,
Placement und Degraded-Notizen. Die reine Analyseansicht verwendet dieselbe Thermik und Wetterdaten,
schreibt aber keine EOS-Payloads. Das finale Profil enthält zusätzlich einen optionalen
  `thermal`-Block mit dem tatsächlich übernommenen Verlauf: Puffer, DHW, Außentemperatur und
  Betriebsmodus je 15 Minuten, beginnend beim echten Simulationsstart. Damit ist in EOS Connect
  die Abkühlung nach SG3 bis zum nächsten SG2-Lauf sichtbar, nicht nur die elektrische Leistung.
  `thermal.plan` bindet diesen Sidecar an das SG3-Kontingent und den tatsächlich adoptierten EOS-Block
  samt `plan_anchor`; dadurch kann EOS eine veraltete Temperaturkurve nach einer Planverschiebung
  verwerfen. Die Analyseansicht bleibt davon getrennt; das Dashboard zeigt auch Zielstopp und Folgepause.
  Für den lokalen Integrations-Test kann EOS Connect über `EOS_HP_TWIN_LOCAL_URL` ausschließlich eine
  Loopback-Adresse (`127.0.0.1` oder `localhost`) für `POST /api/push_now` verwenden. Das ist keine
  produktive Konfiguration und beinhaltet keine Steuerung der Wärmepumpe oder des Batteriesystems.

**SG3-Diagnose im Dashboard:** Unter der Lastprognose zeigt das Panel „SG3-Diagnose“ die
Feasibility-Maske als Zeitachse, konfigurierte Startfenster mit UTC-Zeitpunkten, Sperrgründe,
Kandidaten mit Energie/Leistung/Wirtschaftlichkeit, EOS-Placement samt `plan_anchor` sowie
Dry-Run-/Push-/Adopt-Status. Für den gewählten Lauf zeigt das Panel Start-/Ziel-/Endtemperatur,
tatsächliche Kompressorlaufzeit, den Zielstopp, das Ende des Sollfensters und die folgende Pause.
Der Abschnitt „Verworfene Kandidaten und Gründe“ unterscheidet
thermisch unzulässige Starts, fehlende Zusatzenergie und Preis-/Bewertungsfehler. So ist sichtbar,
ob ein Lauf an Thermik, Marktdaten oder Wirtschaftlichkeit scheitert.

## Installation in Home Assistant (Add-on)
EOS Connect läuft als HA-Add-on; hp-twin genauso. Siehe `hp-twin-addon/DOCS.md`. Kurz:
1. **Einstellungen → Add-ons → Add-on-Store → ⋮ → Repositories** → URL dieses GitHub-Repos hinzufügen.
   (Eigener Fork: `REPO_URL` im `hp-twin-addon/Dockerfile` anpassen.)
2. Add-on **hp-twin Wärmepumpen-Zwilling** installieren.
3. `config/*.yaml` nach `/config` im HA-Dateisystem kopieren (Samba-/VS-Code-Add-on) und in den
   Add-on-Optionen Pfade/Parameter setzen (`topology`, `model`, `forecast_runs`, `eos_base_url`, …).
4. Starten – das Dashboard erscheint per **Ingress** in der HA-Seitenleiste. Die HA-Sensoren liest
   das Add-on über den **Supervisor-Proxy** (`http://supervisor/core`), ein Long-Lived-Token ist
   nicht nötig.

## Release (Add-on aktualisieren)
Das Add-on installiert beim Build `hp-twin` aus Git per **Tag `v<version>`** (`version` in
`hp-twin-addon/config.yaml`). Für ein Update: Version in `pyproject.toml` **und** `config.yaml`
erhöhen, committen, Tag setzen und pushen:
```bash
git tag v0.1.4 && git push origin main v0.1.4
```
Danach in HA im Store nach Updates suchen. Die laufende Version steht im Dashboard-Header.

## Tests / Qualität
`pytest` · `ruff check . && ruff format --check .` · `mypy src/hp_twin` · Docker: `docker compose up`.
