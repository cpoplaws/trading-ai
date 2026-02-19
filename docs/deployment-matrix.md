# Deployment Matrix

## Service Matrix

| Service | Canonical path | Platform | Build/start command | Health endpoints | Smoke check |
|---|---|---|---|---|---|
| Backend API | `apps/api` | Railway | `uvicorn main:app --host 0.0.0.0 --port $PORT` | `GET /healthz`, `GET /readyz` | `python scripts/deployment_smoke_check.py --base-url <url>` |
| Frontend dashboard | `apps/dashboard` | Vercel | `npm install`, `npm run build` | frontend depends on backend health | validates via backend contract endpoints |

## Environment Variable Reconciliation

### Backend (Railway) required for stable runtime
| Variable | Purpose | Source file references | Required |
|---|---|---|---|
| `PORT` | Runtime port for uvicorn | `apps/api/.env.example:20`, Railway runtime | yes |
| `CORS_ALLOW_ORIGINS` | Explicit frontend allowlist | `apps/api/.env.example:17`, `apps/api/main.py:61` | yes (production) |
| `ALPACA_API_KEY` | Legacy/optional CEX data path | `apps/api/.env.example:2`, `.env.example:5` | optional |
| `ALPACA_SECRET_KEY` | Legacy/optional CEX data path | `apps/api/.env.example:3`, `.env.example:6` | optional |
| `BINANCE_API_KEY` | CEX connector auth | `apps/api/.env.example:6`, `.env.template:18`, `apps/api/main.py:167` | optional |
| `BINANCE_API_SECRET` | CEX connector auth | `apps/api/.env.example:7`, `apps/api/main.py:168` | optional |
| `WALLET_MASTER_PASSWORD` | Wallet manager encryption | `apps/api/.env.example:10`, `apps/api/main.py:176` | optional (required for real wallet ops) |
| `DATABASE_URL` | Optional DB readiness probe | `apps/api/.env.example:13`, `.env.template:111`, `apps/api/main.py:139` | optional |
| `REDIS_URL` | Optional cache/readiness probe | `apps/api/.env.example:14`, `.env.template:112`, `apps/api/main.py:150` | optional |

### Frontend (Vercel) required
| Variable | Purpose | Source file references | Required |
|---|---|---|---|
| `NEXT_PUBLIC_API_URL` | Backend API base URL | `apps/dashboard/.env.example:2`, `apps/dashboard/next.config.js:5` | yes |
| `NEXT_PUBLIC_WS_URL` | Realtime channel URL | `apps/dashboard/.env.example:3`, `apps/dashboard/next.config.js:6`, `apps/dashboard/hooks/useWebSocket.ts:4` | yes |

## Platform Config Notes
1. Railway start command is consistent across `apps/api/railway.toml`, `apps/api/railway.json`, and `apps/api/Procfile`.
2. Root `vercel.json` routes all requests to `apps/dashboard`; `apps/dashboard/vercel.json` also defines framework commands. Keep one owner model per Vercel project to avoid route/build drift.
3. Backend dependencies are now pinned in `apps/api/requirements.txt` for deterministic deployment.

## CI Smoke Workflow
- Workflow file: `.github/workflows/deployment-smoke.yml`
- Trigger: `push` to `main`/`develop` and manual dispatch
- Required input: `DEPLOY_SMOKE_BASE_URL` as workflow input, repo variable, or secret
- Failure behavior: workflow fails when smoke endpoint contract breaks

## Deployment Guardrails
1. Run smoke checks immediately post-deploy against deployed backend URL.
2. Block promotion if `/readyz` fails required checks.
3. Keep CORS origin list explicit in production via `CORS_ALLOW_ORIGINS`.
4. Track one canonical backend URL and propagate to Vercel `NEXT_PUBLIC_API_URL`.
