FROM python:3.12.13-slim-bookworm@sha256:d50fb7611f86d04a3b0471b46d7557818d88983fc3136726336b2a4c657aa30b

ARG APP_VERSION=0.9.0

LABEL org.opencontainers.image.title="Quant Agent" \
      org.opencontainers.image.version="${APP_VERSION}" \
      org.opencontainers.image.source="https://github.com/wx1127/Quant-Agent"

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app/packages:/app \
    QUANT_AGENT_APP_ENV=production \
    QUANT_AGENT_RUNTIME_MODE=LIVE_ASSISTED

WORKDIR /app

COPY requirements.lock /app/requirements.lock
RUN python -m pip install --no-cache-dir --requirement /app/requirements.lock

RUN groupadd --system --gid 10001 quant-agent \
    && useradd --system --uid 10001 --gid quant-agent --home-dir /nonexistent quant-agent \
    && mkdir -p /var/lib/quant-agent/data /var/lib/quant-agent/audit \
        /var/lib/quant-agent/shadow /var/lib/quant-agent/reports \
    && chown -R quant-agent:quant-agent /var/lib/quant-agent

COPY --chown=quant-agent:quant-agent apps /app/apps
COPY --chown=quant-agent:quant-agent packages /app/packages
COPY --chown=quant-agent:quant-agent configs /app/configs
COPY --chown=quant-agent:quant-agent migrations /app/migrations
COPY --chown=quant-agent:quant-agent scripts/deploy /app/scripts/deploy
COPY --chown=quant-agent:quant-agent scripts/shadow /app/scripts/shadow
COPY --chown=quant-agent:quant-agent alembic.ini pyproject.toml /app/

USER 10001:10001

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
  CMD ["python", "/app/scripts/deploy/healthcheck.py"]

CMD ["python", "-m", "uvicorn", "apps.api.app:app", "--host", "0.0.0.0", "--port", "8000", "--no-access-log"]
