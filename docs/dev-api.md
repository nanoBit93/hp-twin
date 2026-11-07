# hp-twin Analyse-/Dev-API – Referenz für Modelle & Fern-Entwicklung

> Diese Datei ist die verbindliche Schnittstellen-Doku für Assistenten/Modelle, die eine laufende
> hp-twin-Instanz (Home-Assistant-Add-on) aus der Ferne analysieren. Alle Endpunkte sind
> lesend/rechnend; es wird nichts an Konfiguration oder Produktivdaten verändert.

## Zweck & Lage

hp-twin läuft als HA-Add-on im Wohnnetz und hält die **Messdaten, Kalibrierung und Automatik**
vor Ort. Die Dev-API macht diesen Zustand über HTTP abfragbar, damit z. B. ein Modell auf der
Entwickler-Maschine fragen kann:

- *Wie steht die Anlage da?* → `GET /api/dev/state`
- *Welche Parameter benutzt die Prognose?* → `GET /api/dev/calibration`
- *Warum startet die WP um 6:14?* → `GET /api/dev/trace` (Zwischenschritte je Zeitschritt)
- *Was passiert bei −5 °C Outdoor oder min_off=15 min?* → `POST /api/dev/scenario`
- *Welches Szenario braucht am meisten Energie?* → `POST /api/dev/scenarios` (Batch)

Die Antwort-Dokumente sind JSON. Temperaturen °C, Leistungen W, Arbeit Wh, Zeiten s/min/h
(Feldnamen verraten die Einheit: `_c`, `_w`, `_wh`, `_k`, `_s`, `_min`, `_h`). Zeitstempel UTC-ISO.

## Zugang & Konfiguration (einmalig)

1. **Add-on aktualisieren** auf ≥ v0.1.4 (siehe `hp-twin-addon/DOCS.md`). Seither ist der
   API-Port im LAN freigegeben (`ports: 8000/tcp`).
2. **In den Add-on-Optionen** (HA-GUI → Einstellungen → Apps → hp-twin → Konfiguration):
   `api_token` auf einen langen Zufallswert setzen (z. B. `openssl rand -hex 24`), dann
   Add-on neu starten. Der Token wird per Umgebungsvariable `HP_TWIN_API_TOKEN` an die App
   gereicht und dort geprüft.
3. **Auf der Entwickler-Maschine** die gitignored `.env` befüllen (Vorlage `.env.example`):

```dotenv
HP_TWIN_API_URL=http://homeassistant.local:8000
HP_TWIN_API_TOKEN=<der Token aus den Add-on-Optionen>
```

Alternativ als Prozess-Umgebungsvariablen. Ohne Token in den Optionen (`api_token` leer) ist der
Direktzugriff ungeschützt – in Heimnetzen typisch, ausdrücklich erlaubt.

### Auth-Regeln im Detail

| Pfad | GET `/api/dev/*` | POST `/api/dev/scenario(s)` | POST Settings/Model/Push |
|---|---|---|---|
| HA-Dashboard (Ingress, Cookie `ingress_session`) | ok | ok | ok |
| Direkt mit `Authorization: Bearer <token>` | ok | ok | ok |
| Direkt mit `?token=<token>` | ok | ok | ok |
| Direkt ohne Token & kein `api_token` gesetzt | ok | ok | ok |
| Direkt ohne Token & `api_token` gesetzt | **401** | **401** | **401** |

Nicht betroffen: `/` (Dashboard), `/health`, `/api/version` (öffentlich lesbar).

## Endpunkte

### `GET /api/dev/state`

Startzustand „jetzt" aus dem letzten Messpunkt.

```json
{
  "meta": {"via_ingress": false, "token_required": true, "root_path": ""},
  "last_measurement": "2026-11-06T02:13:00+00:00",
  "data_age_minutes": 2684.9,
  "nodes_c": {"dhw": 53.5, "buffer": 36.0},
  "t_outdoor_c": 4.1,
  "compressor_on": true,
  "measured_mode": "heat",
  "controller_state": {"mode": "heat", "time_in_mode_s": 1560.0,
                        "time_since_stop_s": 0.0, "active_node": "buffer"}
}
```

`controller_state.mode` ∈ `idle | pause | heat | dhw` (aus der Historie abgeleitet; `pause`
bedeutet: WP gestoppt, min_off noch nicht erfüllt → nächster Start frühestens danach).

### `GET /api/dev/data[?start=<UTC-ISO>&end=<UTC-ISO>]`

Ohne Query: Datenbestand (Zeitbereich, Raster, Spalten, letzter Zustand, aktive Kalibrierung).
Mit Query: Reihen-Ausschnitt. `window.series` enthält nur vorhandene Spalten; Werte sind gerundet
(`t_outdoor_c` 2 Dezimalen, Leistungen 0). Mehr als 5000 Punkte werden gleichmäßig verkleinert
(`window.downsampled_by`).

```json
{
  "n_rows": 14528, "first_iso": "...", "last_iso": "...", "span_days": 10.09, "step_s": 60.0,
  "state": {"buffer_c": 36.0, "dhw_c": 53.5, "compressor_on": 1, "mode": "heat"},
  "calibration": {"heat_pump": {}, "buffer_threshold": {}, "discharge": {}, "notes": []},
  "window": {"start_iso": "...", "end_iso": "...", "n": 360, "downsampled_by": 1,
             "timestamps": ["..."], "series": {"t_outdoor_c": [], "p_el_w": []}}
}
```

### `GET /api/dev/calibration`

Aktive Kalibrierung + Modell-Defaults (`model`), gleiche Struktur wie `data()` – ohne Messdaten.

### `GET /api/dev/trace?start=<UTC-ISO>&horizon_h=<h>&trace_step_s=<s>`

**Zwischenschritte des unveränderten Laufs** (Modell vs. Messung im Fenster). `trace` ist eine
Liste je Zeitschritt (Standardauflösung 300 s, max. 4000 Punkte; fehlendes `start` = Historienanfang):

```json
{"t_s": 3660.0, "index": 61, "mode": "heat", "charging": true, "target_node": "buffer",
 "flow_setpoint_c": 42.2, "t_flow_eff_c": 34.22, "p_el_w": 1898.2, "q_th_w": 8492.4, "cop_est": 4.47,
 "setpoints": {"dhw_c": 40.0, "buffer_c": 32.2, "dhw_hysteresis_k": 5.0, "buffer_hysteresis_k": 4.95,
               "dhw_overshoot_k": 2.0, "buffer_overshoot_k": 8.4},
 "controller": {"mode": "heat", "time_in_mode_s": 60.0, "time_since_stop_s": 0.0,
                 "active_node": "buffer"},
 "nodes_before_c": {"dhw": 49.2, "buffer": 27.7},
 "nodes_after_c": {"dhw": 49.19, "buffer": 28.0},
 "q_in_w": {"buffer": 8492.4}, "q_out_w": {"buffer": 1659.3}}
```

Lesen dieser Liste beantwortet typische Debug-Fragen: Warum startet/stoppt der Regler? Welcher
Sollwert war wirksam? Wie groß ist der Wärmestrom je Kreis? `compare` daneben zeigt
Sim-vs-Messung (Puffer/DHW/P_el) und `summary.metrics` den Fehler des Laufs gegen die Realität.

### `POST /api/dev/scenario`

Ein Was-wäre-wen-Lauf. Body:

```json
{
  "start": "2026-11-06T00:00:00+00:00",
  "horizon_h": 24.0,
  "trace": false,
  "trace_step_s": 300.0,
  "overrides": {
    "t_outdoor_c": null,
    "t_outdoor_offset_k": -5.0,
    "ghi_wm2": null,
    "room_setpoint_c": null,
    "setpoint_hk1_offset_k": 0.0,
    "setpoint_dhw_offset_k": 0.0,
    "controller": {"min_off_s": 900, "min_on_s": 240},
    "heat_pump": {"p_el_ref_w": 2000, "eta_carnot": 0.42, "cop_min": 1.6, "cop_max": 5.0},
    "c_building_j_per_k": 1.2e7,
    "dhw_setpoint_cap_c": 45.0
  }
}
```

Alle Felder optional (`start` fehlt → Historienanfang). Overrides greifen nur für diesen Lauf:
- **Eingänge** werden aufs Mess-/Prognosefenster gelegt (Offset addiert, absolut gesetzt).
- **`controller`** merged auf die Topologie-Controller-Config (`min_off_s`, `min_on_s`,
  `dhw_hysteresis_k`, `buffer_hysteresis_k`, `buffer_overshoot_k`, `dhw_overshoot_k`,
  `mode_switch_pause_s`).
- **`heat_pump`** merged auf das **kalibrierte** WP-Kennfeld (alle Felder von
  `HeatPumpAirWater`, z. B. `p_el_ref_w`, `dp_dt_flow_w_per_k`, `dp_dt_out_w_per_k`,
  `eta_carnot`, `p_comp_min_w`, `p_comp_max_w`, `cop_min`, `cop_max`, `standby_w`).
- **`c_building_j_per_k`** = wirksame Gebäudemasse.
- **`dhw_setpoint_cap_c`** deckelt den DHW-Soll (passive DHW, wie in Produktion).

Antwort: `summary` (u. a. `energy_kwh`, `n_cycles`, `controller_effective`,
`heat_pump_effective`, `metrics` mit Startzeit-/Energie-/Zyklen-Vergleich gegen die Messung
im Fenster), `compare` (Zeitreihen Sim vs. Messung), optional `trace` (= wie /trace).

### `POST /api/dev/scenarios`

Batch mit 1–20 Szenarien (Liste von Body-Objekten wie oben). Antwort:

```json
{
  "n": 4,
  "results": [{"index": 0, "summary": {}, "compare": {}, "trace": []}],
  "energy_kwh": [5.83, 8.63, 5.83, 8.62],
  "n_cycles": [4, 5, 4, 4]
}
```

`energy_kwh`/`n_cycles` sind die Schnellvergleichsspalten; `results[i]` enthält die volle Antwort.

## Status-/Prozess-Endpunkte (bestanden bereits)

- `GET /api/status` – letzter Automatik-Zyklus (Fehler, Phase, Energie), Datenalter, Konfigfehler.
- `POST /api/push_now` – Automatik-Zyklus sofort starten (Body `{"push": false}` = Probe). Token.
- `GET /api/analysis?runs=&horizon_h=&from_end=` – Dashboard-Analyse-Bundle.
- `POST /simulate` – Prognose-JSON, gleiche Parameter wie CLI `hp-twin forecast`.

## Python-Client (empfohlen für Modelle)

```python
from hp_twin.adapters.api_client import DevApiClient, overview

client = DevApiClient.from_env()  # liest .env: HP_TWIN_API_URL, HP_TWIN_API_TOKEN

# 1) Wie steht die Anlage?
overview(client).summary()
# 'hp-twin 0.1.4 | letzte Messung ... | Puffer 36.0 °C, DHW 53.5 °C | regler: heat'

# 2) Zwischenschritte im Fenster (6 h, Auflösung 5 min):
t = client.trace(horizon_h=6.0, trace_step_s=300.0)
starts = [x for x in t["trace"] if x["charging"] and x["controller"]["time_in_mode_s"] <= 60]

# 3) Was-wäre-wen-Vergleich (Batch):
res = client.scenarios(
    [
        {"horizon_h": 12},
        {"horizon_h": 12, "overrides": {"t_outdoor_offset_k": -5}},
        {"horizon_h": 12, "overrides": {"controller": {"min_off_s": 900}}},
    ]
)
res["energy_kwh"]  # [5.83, 9.07, 5.83]
```

## CLI (schnelle Abfragen auf der Konsole)

```bash
hp-twin remote state
hp-twin remote calibration
hp-twin remote data --start 2026-11-06T00:00:00Z --end 2026-11-06T06:00:00Z
hp-twin remote trace --horizon-h 6 --step 300
hp-twin remote-scenario --horizon-h 12 --overrides cold.json
hp-twin remote-scenario --batch base.json cold.json fast.json
```

(`--env` überschreibt `.env`; Verbindungs-/HTTP-Fehler kommen als `RuntimeError` mit Status und
Server-Detail.)

## Debug-Rezepte (typische Fragen → Vorgehen)

| Frage | Vorgehen |
|---|---|
| „Warum läuft die WP um 6:14 real, die Prognose sagt 7:05?" | `trace` ab ≈ 5:00 mit `trace_step_s=60`: `nodes_before_c.buffer` vs. `setpoints.buffer_c - buffer_hysteresis_k` → Startregel verletzt? `summary.metrics.next_start_error_min` zeigt den Bias |
| „Prognose-Energie zu hoch?" | `scenario` ohne Overrides (24 h): `compare.metrics.energy_6h_mae_wh` → Modell- oder Eingabefehler? Dann Overrides isoliert testen |
| „Was bringen kältere Tage?" | Batch: Basis vs. `{"t_outdoor_offset_k": -5}` vs. `-10` → `energy_kwh`-Kurve |
| „Einfluss der Gebäudemasse?" | `c_building_j_per_k` ×0.75 / ×1.25 |
| „WP-Kennfeld unscharf?" | `heat_pump.eta_carnot` ±0.05, `p_el_ref_w` ±10 % |
| „Ist das Add-on online?" | `client.health()` → `{"status": "ok"}`, `client.version()` |

## Grenzen

- Szenarien/Traces laufen synchron im Add-on-Prozess (Zeitschritt 60 s, 48 h ≈ Sekunden);
  Batches > 20 werden abgelehnt.
- Die API ändert nichts an der Konfiguration; Szenario-Overrides sind flüchtig.
- Prognosen ab „jetzt" benutzen die letzten 24 h als Wetter/Sollwert-Proxy (Details: README).
