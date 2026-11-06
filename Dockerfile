FROM python:3.12-slim AS build
WORKDIR /build
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --no-cache-dir build && pip wheel --wheel-dir /wheels ".[api]"

FROM python:3.12-slim
RUN useradd --create-home --uid 10001 app
COPY --from=build /wheels /wheels
RUN pip install --no-cache-dir /wheels/*.whl && rm -rf /wheels
USER app
WORKDIR /app
ENV HP_TWIN_CONFIG=/config/topology.yaml HP_TWIN_DATA=/data
EXPOSE 8000
CMD ["uvicorn", "hp_twin.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
