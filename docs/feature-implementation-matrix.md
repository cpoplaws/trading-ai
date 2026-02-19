# Feature Implementation Matrix

This matrix links major production features to code, tests, deployment surfaces, and current state.

## Core Features

| Feature | Market scope | Code path(s) | Test/validation path(s) | Deployment surface | Status | Priority |
|---|---|---|---|---|---|---|
| API liveness/readiness | crypto | `apps/api/main.py:280`, `apps/api/main.py:291` | `scripts/deployment_smoke_check.py` | Railway (`apps/api/railway.toml`) | implemented | P0 |
| Portfolio API | cross-market | `apps/api/main.py:312` | `apps/dashboard/lib/api-client.ts:14` | Railway + Vercel contract | implemented | P0 |
| Strategy lifecycle APIs | crypto | `apps/api/main.py:379`, `apps/api/main.py:417`, `apps/api/main.py:452` | `apps/dashboard/lib/api-client.ts:25`, `apps/dashboard/lib/api-client.ts:30` | Railway + Vercel contract | implemented | P0 |
| Recent trades API | crypto | `apps/api/main.py:492` | `apps/dashboard/components/dashboard/RecentTrades.tsx:27` | Railway + Vercel contract | implemented | P0 |
| Base chain adapter endpoints | crypto | `apps/api/main.py:469`, `apps/api/main.py:481` | `apps/dashboard/lib/api-client.ts:42`, `apps/dashboard/lib/api-client.ts:52` | Railway + Vercel contract | partial (mock backend response) | P1 |
| Agent swarm status/control | crypto | `apps/api/main.py:530`, `apps/api/main.py:541`, `apps/api/main.py:553`, `apps/api/main.py:565` | `apps/dashboard/components/dashboard/AgentSwarm.tsx` | Railway + Vercel contract | implemented | P1 |
| Market intelligence service | cross-market | `apps/api/main.py:599`, `apps/api/main.py:624` | `apps/dashboard/components/dashboard/MarketIntelligence.tsx:65` | Railway + Vercel contract | partial (uses generated sample data path) | P1 |
| Supervisor overview and instance control | crypto | `apps/api/main.py:649`, `apps/api/main.py:680`, `apps/api/main.py:703`, `apps/api/main.py:733` | API contract verification | Railway | implemented | P1 |
| Realtime dashboard stream | crypto | `apps/api/main.py:751` | `apps/dashboard/hooks/useWebSocket.ts` | Railway + Vercel runtime | partial (protocol mismatch: socket.io vs raw ws) | P0 |
| Prelaunch validator | crypto | `src/deployment/prelaunch_validator.py` | `tests/test_phase7_mainnet_launch.py` | deployment process gate | implemented | P1 |
| Rollout manager phases | crypto | `src/deployment/rollout_manager.py` | `tests/test_phase7_mainnet_launch.py` | deployment process gate | implemented | P1 |
| Feature evidence register generation | crypto-first policy | `scripts/repo_discovery/generate_feature_register.py` | generated artifact `docs/feature-evidence-register.csv` | discovery governance | implemented | P1 |

## Platform and Delivery Features

| Capability | Evidence | Status | Risk |
|---|---|---|---|
| Railway startup command consistency | `apps/api/railway.toml`, `apps/api/railway.json`, `apps/api/Procfile` | implemented | low |
| Deterministic backend dependency pinning | `apps/api/requirements.txt` | implemented | low |
| CORS allowlist strategy with Vercel regex | `apps/api/main.py:202`, `apps/api/main.py:203` | implemented | medium |
| Post-deploy smoke checks in CI | `.github/workflows/deployment-smoke.yml` | implemented | medium (requires base URL secret/var) |
| Frontend required API env variable | `apps/dashboard/.env.example`, `apps/dashboard/next.config.js` | implemented | low |

## Known Gaps To Close
1. Eliminate `apps/dashboard/lib/api.ts` or migrate all callers to `apps/dashboard/lib/api-client.ts`.
2. Resolve realtime transport mismatch (`socket.io` client vs backend raw websocket endpoint).
3. Remove or formally deprecate non-canonical API trees (`src/api`, `api`) from deployment path.
4. Replace mock Base chain execution responses with real connector-backed execution when readiness criteria are met.
