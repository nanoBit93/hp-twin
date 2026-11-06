# Multi-Arch-Base von Home Assistant (Alpine + s6 + bashio). Die früheren *-base-python-Images
# gibt es nicht mehr -> Python wird per apk installiert.
ARG BUILD_FROM=ghcr.io/home-assistant/base:3.22
FROM ${BUILD_FROM}

# Quelle des Pakets (eigenen Fork/Branch/Tag hier eintragen oder per --build-arg überschreiben)
ARG REPO_URL="git+https://github.com/nanoBit93/hp-twin@main"

RUN apk add --no-cache python3 py3-pip git \
    && python3 -m venv /opt/venv \
    && /opt/venv/bin/pip install --no-cache-dir --upgrade pip \
    && /opt/venv/bin/pip install --no-cache-dir "hp-twin[api] @ ${REPO_URL}" \
    && apk del git

ENV PATH="/opt/venv/bin:${PATH}"

COPY run.sh /run.sh
RUN chmod +x /run.sh
CMD ["/run.sh"]
