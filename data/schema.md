# CSV-Schema für Messdaten (Kalibrierung & Backtest)

- **Zeitzone**: `timestamp` ISO-8601, **UTC** (`2026-01-10T00:05:00Z`). Anzeige später Europe/Berlin.
- **Auflösung**: gleichmäßiges Raster, 1–5 min empfohlen (keine Lücken, keine Duplikate). Intervallwerte (`q_heat_wh`) beziehen sich auf das Intervall *bis* zum Zeitstempel.
- **Trennzeichen** `,`, Dezimalpunkt `.`, Header-Zeile Pflicht.

| Spalte | Einheit | Pflicht | Beschreibung |
|---|---|---|---|
| timestamp | UTC | ja | Zeitstempel |
| t_outdoor_c | °C | ja | Außentemperatur gemessen |
| t_outdoor_fc_c | °C | nein | Außentemperatur-Prognose |
| ghi_wm2 | W/m² | ja | Globalstrahlung |
| t_room_<name>_c | °C | ≥1 | Raumtemperatur je Raum |
| compressor_on | 0/1 | ja | Verdichter an |
| mode | – | ja | `idle`, `heat`, `dhw`, `defrost` |
| circuit_active | – | nein | `hk1`, `hk2`, `dhw`, `none` |
| p_el_w | W | ja | elektrische Leistung WP gesamt |
| q_heat_wh | Wh | ja | Wärmemenge im Intervall |
| t_flow_c / t_return_c | °C | ja | Vorlauf / Rücklauf WP |
| t_dhw_c / t_hk1_c / t_hk2_c | °C | ja | Speicherfühler DHW / HK1, HK2-Ist |
| setpoint_dhw_c / setpoint_hk1_c / setpoint_hk2_c | °C | ja | Sollwerte (WPM) |

Validierung: `hp_twin.adapters.io_csv.validate_frame`. Vorlage: `data/template.csv`; Synthetik: `scripts/make_synthetic_data.py`.
