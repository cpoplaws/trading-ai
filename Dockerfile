# Multi-stage Docker build for Quantlytics / Itera engine
FROM python:3.11-slim as base

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    curl \
    postgresql-client \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements-core.txt requirements-ml.txt requirements-secure.txt ./
RUN pip install --upgrade pip && pip install -r requirements-secure.txt

COPY pyproject.toml README.md start.py ./
COPY src ./src
COPY examples ./examples
COPY scripts ./scripts

RUN pip install .

RUN useradd -m -u 1000 trader && chown -R trader:trader /app
USER trader

EXPOSE 8000 8765 8501

HEALTHCHECK --interval=30s --timeout=10s --start-period=20s --retries=3 \
  CMD curl -fsS http://localhost:8000/health || exit 1

CMD ["quantlytics-api"]
