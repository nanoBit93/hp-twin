#!/usr/bin/with-contenv bashio
set -e

TOPOLOGY=$(bashio::config 'topology')
MODEL=$(bashio::config 'model')
RUNS=$(bashio::config 'forecast_runs')

# HA-Core-API über den Supervisor-Proxy (kein Long-Lived-Token nötig)
export HA_BASE_URL="http://supervisor/core"
export HA_TOKEN="${SUPERVISOR_TOKEN}"
export HP_TWIN_TOPOLOGY="/config/${TOPOLOGY}"
export HP_TWIN_MODEL="/config/${MODEL}"
export HP_TWIN_RUNS="${RUNS}"

# Beim ersten Start Vorlagen nach /config legen, falls noch keine da sind
if [ ! -f "/config/${TOPOLOGY}" ]; then
  mkdir -p "$(dirname "/config/${TOPOLOGY}")"
  python -c "import importlib.resources" 2>/dev/null || true
  bashio::log.warning "Keine Konfiguration unter /config/${TOPOLOGY} gefunden."
  bashio::log.warning "Bitte config/ (example_topology.yaml, model.yaml, ha_mapping.yaml, heizlast.yaml) nach /config kopieren."
fi

cd /config
INGRESS_ENTRY=$(bashio::addon.ingress_entry)
bashio::log.info "hp-twin startet auf Ingress ${INGRESS_ENTRY} (Topologie ${TOPOLOGY}, ${RUNS} MC-Läufe)"
exec uvicorn hp_twin.api.app:app --host 0.0.0.0 --port 8000 --root-path "${INGRESS_ENTRY}"
