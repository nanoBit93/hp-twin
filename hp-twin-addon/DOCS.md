# hp-twin – Home-Assistant-App (Add-on)

Zustandsbasierte 48-h-Lastprognose der Wärmepumpe mit Dashboard und EOS-Connect-Ausgabe.

## Installation
1. **Einstellungen → Apps/Add-ons → Store → ⋮ → Repositories**: `https://github.com/nanoBit93/hp-twin` hinzufügen.
2. **hp-twin Wärmepumpen-Zwilling** installieren und starten (der erste Build dauert einige Minuten).
3. Dashboard per **Ingress** in der Seitenleiste öffnen.

## Erste Einrichtung
Das Add-on startet ohne Anlagenkonfiguration. Zwei Wege:
- **Bundle importieren (empfohlen):** In deiner bestehenden Installation `hp-twin bundle export`
  (oder im Dashboard „⬇ Config exportieren") und im Add-on-Dashboard unter
  **„Portable Konfiguration" → „⬆ Config importieren"** einspielen. Topologie, HA-Mapping, Heizlast,
  Modellparameter und Kalibrierung sind danach da.
- **Von Hand:** Dateien `config/example_topology.yaml`, `config/ha_mapping.yaml`,
  `config/heizlast.yaml` (Vorlagen: `*.example.yaml` im Repo) in den Add-on-Konfigurationsordner legen
  (Samba-Freigabe **addon_configs → …hp_twin**).

Alle weiteren Einstellungen (EOS-Verbindung, Automatik, Modellparameter) im Dashboard.

## Add-on-Optionen (nur Startwerte)
Werden beim **ersten Start** in `config/runtime.json` übernommen; danach gilt, was im Dashboard steht.

| Option | Bedeutung |
|---|---|
| `eos_base_url` / `eos_managed_load_id` | EOS Connect (z. B. `http://<host>:8081`, Load-ID z. B. `hp_ext`) |
| `forecast_runs` | Monte-Carlo-Läufe |
| `horizon_h` | Prognosehorizont (max. 48) |
| `dhw_setpoint_cap_c` | DHW-Soll deckeln (passive DHW); leer = Messwert |
| `auto_push_enabled` | Automatik (Export → Prognose → Push) beim Start aktivieren |
| `api_token` | API-Token für direkte Zugriffe auf die Instanz (siehe unten); leer = keiner |

## Analyse-/Dev-API (Fern-Zugriff von der Entwickler-Maschine)

Die laufende App kann von außen analysiert werden (Troubleshooting im Produktivsystem ohne
lokale Daten). Alle Endpunkte unter `/api/dev/*` sind **lesend/rechnend**:

| Endpunkt | Zweck |
|---|---|
| `GET /api/dev/data?start=…&end=…` | Datenbestand, Messreihen-Ausschnitt (JSON, gerundet) |
| `GET /api/dev/state` | Startzustand „jetzt": Speicher-/Reglerzustand aus der Messung |
| `GET /api/dev/calibration` | Aktive Kalibrierung + Modellparameter |
| `GET /api/dev/trace?start=…&horizon_h=…&trace_step_s=…` | Zwischenschritte je Zeitschritt (Modus, Sollwerte, Vorlauf, Q_th/P_el, Knotentemperaturen, Reglerzustand) |
| `POST /api/dev/scenario` | Was-wäre-wen: Overrides auf Eingänge (Außentemp, Sollwerte, Raumtemp, SG) und Modell (min_off/min_on, WP-Kennfeld, Gebäudemasse, DHW-Cap) |
| `POST /api/dev/scenarios` | Batch: bis zu 20 Szenarien in einem Aufruf (kann direkt vergleichen) |

Zugriff aus dem HA-Netz (Beispiel):

```bash
TOKEN=mein-token
HA=http://homeassistant.local:8123
# Add-on-Port nicht direkt freigeben? Dann über Port 8010 des Add-on-Hosts (Ingress-Port 8000):
BASE=http://homeassistant.local:8123  # Dev-API wird am Add-on-Port 8000 bedient
curl -s -H "Authorization: Bearer mein-token" "http://<host-ip>:8000/api/dev/state"
curl -s -H "Authorization: Bearer mein-token" "http://<host-ip>:8000/api/dev/trace?horizon_h=6&trace_step_s=300"
curl -s -X POST -H "Authorization: Bearer mein-token" -H "Content-Type: application/json" \
  -d '{"horizon_h":12,"overrides":{"t_outdoor_offset_k": -5}}' \
  "http://<host-ip>:8000/api/dev/scenario"
```

- Über den **Ingress** (Dashboard in der Seitenleiste) ist die API ohne Token nutzbar – der
  Supervisor-Proxy authentifiziert bereits.
- Direkte Aufrufe (z. B. von der Entwickler-Maschine) verlangen **`Authorization: Bearer <api_token>`**
  sobald `api_token` gesetzt ist (auch `?token=…` für schnelle curl-Tests). Ohne Token gelten
  direkte Zugriffe als lokal.
- Schreib-Endpunkte (`POST /api/settings`, `/api/model`, `/api/bundle`, `/api/push_now`) sind
  gleichermaßen token-geschützt; die Prognose-Automatik selbst läuft wie bisher.

Wichtig: Seit v0.1.4 ist der Port 8000 des Add-ons im HA-Netz direkt erreichbar. Setze
`api_token`, sobald das Add-on aus einem anderen Netzsegment erreichbar sein soll; im
Dashboard-Ingress ändert sich nichts. **Vollständige Referenz: [`docs/dev-api.md`](../docs/dev-api.md)**

## Datenzugriff
HA-Sensoren werden über den **Supervisor-Proxy** (`http://supervisor/core`) gelesen – kein
Long-Lived-Token nötig. Das Entity-Mapping steht in `config/ha_mapping.yaml`.
