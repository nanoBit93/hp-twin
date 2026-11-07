#!/usr/bin/with-contenv bashio
set -e

# HA-Core-API über den Supervisor-Proxy (kein Long-Lived-Token nötig)
export HA_BASE_URL="http://supervisor/core"
export HA_TOKEN="${SUPERVISOR_TOKEN}"

# API-Token für direkte (nicht-Ingress) Zugriffe auf die Analyse-/Dev-API und Schreib-Endpunkte.
# Leer = kein Token: Ingress und lokale Aufrufe erlaubt.
API_TOKEN="$(bashio::config 'api_token')"
if [ -n "${API_TOKEN}" ]; then
  export HP_TWIN_API_TOKEN="${API_TOKEN}"
fi

cd /config
mkdir -p config data/ha_export

# Beim ersten Start die Add-on-Optionen als Startwerte in runtime.json übernehmen.
if [ ! -f config/runtime.json ]; then
  bashio::log.info "Erster Start: schreibe Startwerte nach /config/config/runtime.json"
  EOS_URL="$(bashio::config 'eos_base_url')" \
  EOS_ID="$(bashio::config 'eos_managed_load_id')" \
  RUNS="$(bashio::config 'forecast_runs')" \
  HORIZON="$(bashio::config 'horizon_h')" \
  DHW_CAP="$(bashio::config 'dhw_setpoint_cap_c')" \
  AUTO="$(bashio::config 'auto_push_enabled')" \
  python - <<'PY'
import os
from hp_twin.adapters.settings import RuntimeSettings
cap = os.environ.get("DHW_CAP", "")
RuntimeSettings(
    eos_base_url=os.environ.get("EOS_URL", ""),
    eos_managed_load_id=os.environ.get("EOS_ID", ""),
    forecast_runs=int(float(os.environ["RUNS"])),
    horizon_h=float(os.environ["HORIZON"]),
    dhw_setpoint_cap_c=float(cap) if cap not in ("", "null") else None,
    auto_push_enabled=os.environ.get("AUTO") == "true",
).save("config/runtime.json")
PY
fi

if [ ! -f config/ha_mapping.yaml ]; then
  bashio::log.warning "Noch keine Konfiguration (ha_mapping/topology/heizlast) vorhanden."
  bashio::log.warning "Im Dashboard unter 'Portable Konfiguration' ein Config-Bundle importieren"
  bashio::log.warning "(hp-twin-config.json aus deiner bestehenden Installation)."
fi

INGRESS_ENTRY="$(bashio::app.ingress_entry)"
bashio::log.info "hp-twin startet (Ingress ${INGRESS_ENTRY})"
exec uvicorn hp_twin.api.app:app --host 0.0.0.0 --port 8000 --root-path "${INGRESS_ENTRY}"
