# hp-twin – Home-Assistant-Add-on

Zustandsbasierte 48-h-Lastprognose der Wärmepumpe mit Dashboard und EOS-Connect-Ausgabe.

## Installation
1. In HA: **Einstellungen → Add-ons → Add-on-Store → ⋮ → Repositories** und die URL dieses
   GitHub-Repos hinzufügen.
2. Add-on **hp-twin Wärmepumpen-Zwilling** installieren.
3. Vor dem Start die Konfiguration anlegen (siehe unten), dann starten.
4. Das Dashboard erscheint per **Ingress** in der HA-Seitenleiste.

## Konfiguration (Add-on-Optionen)
| Option | Bedeutung |
|---|---|
| `topology` | Pfad (unter `/config`) zur Anlagen-Topologie, z. B. `config/example_topology.yaml` |
| `model` | Pfad zur Modellparameter-Datei `config/model.yaml` |
| `export_days` | Tage Historie, die aus HA exportiert werden |
| `forecast_runs` | Monte-Carlo-Läufe |
| `horizon_h` | Prognosehorizont (max. 48) |
| `dhw_setpoint_cap_c` | DHW-Soll deckeln (passive DHW); leer = Messwert nutzen |
| `eos_base_url` | Basis-URL der EOS-Connect-Instanz für den Push (optional) |
| `eos_managed_load_id` | Managed-Load-ID in EOS Connect |

Die Dateien `config/*.yaml` (Topologie, Modell, ha_mapping, heizlast) nach `/config` im HA-
Dateisystem kopieren (z. B. via Samba/VS-Code-Add-on). Vorlagen liegen als `*.example.yaml` im Repo.

## Datenzugriff
Das Add-on liest die HA-Sensoren über den **Supervisor-Proxy** (`http://supervisor/core`),
ein Long-Lived-Token ist nicht nötig. Das Entity-Mapping steht in `config/ha_mapping.yaml`.
