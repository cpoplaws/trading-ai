# Repository Topology

## Canonical Runtime Decisions
1. Backend production runtime is `apps/api` on Railway.
2. Frontend production runtime is `apps/dashboard` on Vercel.
3. `src/api` and `api` are non-canonical parallel API stacks and should be treated as migration/reference surfaces unless explicitly promoted.
4. Environment source of truth is now split by service: backend (`apps/api/.env.example`) and frontend (`apps/dashboard/.env.example`) with extended optional keys in `.env.template`.

## Topology Summary

| Surface | Path | Files | Python files | TS/TSX files | Markdown files | Notes |
|---|---|---:|---:|---:|---:|---|
| Backend app | `apps/api` | 25 | 18 | 0 | 0 | Railway service entrypoint and runtime code |
| Frontend app | `apps/dashboard` | 31 | 0 | 17 | 1 | Vercel Next.js app |
| Platform engine | `src` | 200 | 178 | 12 | 1 | Strategies, execution, risk, data, infra modules |
| Alt API stack A | `src/api` | included in `src` | included | included | included | FastAPI with `/api/v1` routers |
| Alt API stack B | `api` | 12 | 11 | 0 | 1 | FastAPI with route modules under `api/routes` |
| Infra config | `config`, `docker`, `k8s`, `infrastructure` | 16 | 0 | 0 | 0 | Deployment and operational templates |
| Validation | `tests` | 26 | 24 | 0 | 2 | Unit/integration plus Phase 7 launch tests |
| Documentation | `docs` | 65 | 0 | 0 | 63 | Architecture and implementation reports |

## Runtime Entrypoints

| Service | Entrypoint | Start command | Platform config |
|---|---|---|---|
| Backend (canonical) | `apps/api/main.py` | `uvicorn main:app --host 0.0.0.0 --port $PORT` | `apps/api/railway.toml`, `apps/api/railway.json`, `apps/api/Procfile` |
| Frontend (canonical) | `apps/dashboard/app/page.tsx` | `npm run build` / `npm run dev` | `apps/dashboard/vercel.json`, root `vercel.json` |
| Alt backend A | `src/api/main.py` | `uvicorn main:app` (local pattern) | no canonical deployment binding found |
| Alt backend B | `api/main.py` | `uvicorn api.main:app` (local pattern) | no canonical deployment binding found |

## Backend Route Inventory (Canonical `apps/api`)
- Route count: 24 total (23 HTTP + 1 WebSocket)
- Health and readiness: `GET /healthz`, `GET /readyz`, root `GET /`
- Core API groups:
1. Portfolio and strategies: `/api/portfolio`, `/api/portfolio/history`, `/api/strategies`, strategy toggle/set endpoints
2. Trading and execution: `/api/trades/recent`, `/api/base/balance/{address}`, `/api/base/trade`
3. Agent and intelligence: `/api/agents/*`, `/api/intelligence*`
4. Supervisor: `/api/supervisor/*`
5. Realtime: `WS /ws`

## Frontend API Surfaces (Canonical `apps/dashboard`)
1. Axios client path: `apps/dashboard/lib/api-client.ts` (primary usage in pages/components)
2. Fetch client path: `apps/dashboard/lib/api.ts` (legacy helper; partially overlapping)
3. Component direct fetches: `apps/dashboard/components/dashboard/*`
4. Realtime hook: `apps/dashboard/hooks/useWebSocket.ts`

## Duplicate Surface Risk

| Risk | Evidence | Impact | Recommendation |
|---|---|---|---|
| Multiple API stacks | `apps/api`, `src/api`, `api` each expose FastAPI apps | Route drift and client confusion | Keep `apps/api` canonical for deploy; document others as non-prod |
| Mixed frontend API clients | `api-client.ts` and `api.ts` both active in repo | Divergent method/URL contracts | Consolidate on `api-client.ts` and deprecate `api.ts` |
| WebSocket protocol mismatch | Frontend uses `socket.io-client`; backend serves raw FastAPI WebSocket at `/ws` | Realtime disconnect/failed subscriptions | Align on one protocol (raw WS or socket.io gateway) |

## Immediate Stabilization Status
- Canonical backend path locked to `apps/api`.
- Canonical frontend path locked to `apps/dashboard`.
- Backend now includes liveness/readiness endpoints and CORS allowlist strategy (`apps/api/main.py`).
- Deployment smoke workflow added (`.github/workflows/deployment-smoke.yml`).
