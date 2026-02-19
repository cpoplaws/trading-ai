# API Contract Matrix

Canonical contract comparison between frontend callers (`apps/dashboard`) and backend routes (`apps/api/main.py`).

## Backend Route Inventory (`apps/api/main.py`)

| Method | Path | Line |
|---|---|---:|
| GET | `/` | 269 |
| GET | `/healthz` | 280 |
| GET | `/readyz` | 291 |
| GET | `/api/portfolio` | 312 |
| GET | `/api/portfolio/history` | 355 |
| GET | `/api/strategies` | 379 |
| POST | `/api/strategies/{strategy_id}/toggle` | 417 |
| PATCH | `/api/strategies/{strategy_id}` | 452 |
| GET | `/api/base/balance/{address}` | 469 |
| POST | `/api/base/trade` | 481 |
| GET | `/api/trades/recent` | 492 |
| GET | `/api/agents/status` | 530 |
| POST | `/api/agents/enable` | 541 |
| POST | `/api/agents/disable` | 553 |
| POST | `/api/agents/{agent_name}/toggle` | 565 |
| GET | `/api/agents/decisions` | 588 |
| GET | `/api/intelligence` | 599 |
| POST | `/api/intelligence/analyze` | 624 |
| GET | `/api/supervisor/status` | 649 |
| GET | `/api/supervisor/instances` | 680 |
| POST | `/api/supervisor/instances/{instance_id}/toggle` | 703 |
| GET | `/api/supervisor/arbitrage` | 733 |
| WEBSOCKET | `/ws` | 751 |

## Frontend-to-Backend Contract Coverage

| Frontend caller | Request | Backend route match | Status | Notes |
|---|---|---|---|---|
| `apps/dashboard/lib/api-client.ts:14` | `GET /api/portfolio` | yes | matched | primary dashboard portfolio call |
| `apps/dashboard/lib/api-client.ts:19` | `GET /api/portfolio/history` | yes | matched | query parameter supported |
| `apps/dashboard/lib/api-client.ts:25` | `GET /api/strategies` | yes | matched | strategy list contract available |
| `apps/dashboard/lib/api-client.ts:30` | `POST /api/strategies/{id}/toggle` | yes | matched | strategy toggle path |
| `apps/dashboard/lib/api-client.ts:36` | `GET /api/trades/recent` | yes | matched | recent trades path |
| `apps/dashboard/lib/api-client.ts:42` | `GET /api/base/balance/{address}` | yes | matched | compatibility endpoint added |
| `apps/dashboard/lib/api-client.ts:52` | `POST /api/base/trade` | yes | matched | compatibility endpoint added |
| `apps/dashboard/lib/api.ts:59` | `PATCH /api/strategies/{name}` | yes | matched | compatibility endpoint added |
| `apps/dashboard/components/dashboard/AgentSwarm.tsx` | agents status/decisions/toggle | yes | matched | all used routes exist |
| `apps/dashboard/components/dashboard/MarketIntelligence.tsx:65` | `GET /api/intelligence` | yes | matched | live intelligence endpoint |
| `apps/dashboard/components/dashboard/RecentTrades.tsx:27` | `GET /api/trades/recent` | yes | matched | trades table source |

## Contract Drift Findings

### P0
1. Realtime protocol drift: frontend uses `socket.io-client` (`apps/dashboard/hooks/useWebSocket.ts`) while backend exposes raw FastAPI websocket at `/ws` (`apps/api/main.py:751`).

### P1
1. Frontend has two API client modules with different defaults: `apps/dashboard/lib/api.ts` defaults to `http://localhost:8080` while `apps/dashboard/lib/api-client.ts` defaults to `http://localhost:8000`.
2. Non-canonical API stacks (`src/api`, `api`) expose different contracts (`/api/v1/*`) from the canonical `apps/api` surface.

## Contract Smoke Coverage
- Implemented by `scripts/deployment_smoke_check.py`.
- Checks: `/healthz`, `/readyz`, `/api/portfolio`, `/api/strategies`, `/api/trades/recent`.
