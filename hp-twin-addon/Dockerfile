# Multi-Arch-Base von Home Assistant (Alpine + s6 + bashio). Die früheren *-base-python-Images
# gibt es nicht mehr -> Python wird per apk installiert.
ARG BUILD_FROM=ghcr.io/home-assistant/base:3.22
FROM ${BUILD_FROM}

# BUILD_VERSION übergibt der Supervisor (= version aus config.yaml). Sie wählt den Git-Tag
# "v<version>" und ändert damit den RUN-Befehl -> kein veralteter Docker-Cache. Pro Release muss
# daher ein Tag v<version> existieren (siehe README "Release").
ARG BUILD_VERSION
ARG REPO_URL="git+https://github.com/nanoBit93/hp-twin"

RUN test -n "${BUILD_VERSION}" \
    && apk add --no-cache python3 py3-pip git \
    && python3 -m venv /opt/venv \
    && /opt/venv/bin/pip install --no-cache-dir --upgrade pip \
    && /opt/venv/bin/pip install --no-cache-dir "hp-twin[api] @ ${REPO_URL}@v${BUILD_VERSION}" \
    && apk del git

ENV PATH="/opt/venv/bin:${PATH}"

COPY run.sh /run.sh
RUN chmod +x /run.sh
CMD ["/run.sh"]
