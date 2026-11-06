ARG BUILD_FROM=ghcr.io/home-assistant/aarch64-base-python:3.12
FROM ${BUILD_FROM}

# hp-twin aus dem GitHub-Repo installieren. REPO_URL beim Add-on-Build setzen
# (build.yaml -> args) oder hier den eigenen Fork eintragen.
ARG REPO_URL="git+https://github.com/nanoBit93/hp-twin@main"
RUN pip install --no-cache-dir "hp-twin[api] @ ${REPO_URL}"

COPY run.sh /run.sh
RUN chmod +x /run.sh
CMD ["/run.sh"]
