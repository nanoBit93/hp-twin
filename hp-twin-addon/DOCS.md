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

## Datenzugriff
HA-Sensoren werden über den **Supervisor-Proxy** (`http://supervisor/core`) gelesen – kein
Long-Lived-Token nötig. Das Entity-Mapping steht in `config/ha_mapping.yaml`.
