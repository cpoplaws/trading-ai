"""Configuration for Quantlytics API runtime security controls."""

from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass
class APISettings:
    environment: str
    api_keys: list[str]
    cors_origins: list[str]
    rate_limit_per_minute: int


def _split_csv(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def load_api_settings() -> APISettings:
    environment = os.getenv("ENVIRONMENT", "development").lower()
    api_keys = _split_csv(os.getenv("QUANTLYTICS_API_KEYS", ""))

    cors = os.getenv("QUANTLYTICS_CORS_ORIGINS", "http://localhost:3000,http://localhost:8501")
    cors_origins = _split_csv(cors)

    raw_limit = os.getenv("QUANTLYTICS_RATE_LIMIT_PER_MINUTE", "120")
    rate_limit_per_minute = max(1, int(raw_limit))

    return APISettings(
        environment=environment,
        api_keys=api_keys,
        cors_origins=cors_origins,
        rate_limit_per_minute=rate_limit_per_minute,
    )
