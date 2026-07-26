# Security Notes

## Scope and maturity

This repository is a hardened prototype. Security controls are improving, but this is **not** a certified production trading system.

## Reporting

If you discover a vulnerability:

1. Do not open a public issue with exploit details.
2. Use GitHub private advisories for this repository.

## Implemented controls

- API key auth via `X-API-Key` matched against `QUANTLYTICS_API_KEYS`.
- Configurable CORS allowlist via `QUANTLYTICS_CORS_ORIGINS`.
- Basic in-memory per-client rate limiting (configurable with `QUANTLYTICS_RATE_LIMIT_PER_MINUTE`).
- Docker Compose now requires explicit secrets instead of weak defaults.
- Non-root Docker runtime user remains enabled.

## Known limitations

- Rate limiting is in-memory and per-process; use Redis or gateway enforcement for multi-instance deployments.
- API key storage is env-based; use a managed secret store for higher assurance.
- No centralized authn/authz service yet.

## Minimum deployment checklist

- [ ] Set strong `POSTGRES_PASSWORD`.
- [ ] Set non-test values in `QUANTLYTICS_API_KEYS`.
- [ ] Restrict `QUANTLYTICS_CORS_ORIGINS` to trusted domains.
- [ ] Keep live broker keys disabled for demo-only environments.
- [ ] Run test and lint checks before deployment.

