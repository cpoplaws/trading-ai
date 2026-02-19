# Reliability Runbook

## Scope
Operational runbook for canonical production stack:
- Backend: `apps/api` on Railway
- Frontend: `apps/dashboard` on Vercel

## SLO Baseline
1. API availability (monthly): >= 99.5%
2. `/readyz` success rate: >= 99.0%
3. Dashboard API fetch success: >= 99.0%
4. Realtime reconnect success after backend restart: >= 95.0%

## Alert Conditions

| Alert | Trigger | Severity | First action |
|---|---|---|---|
| Backend down | `/healthz` non-200 for 2 consecutive checks | critical | verify Railway deploy status and recent release |
| Readiness degraded | `/readyz` returns 503 | high | inspect required failures payload |
| API contract break | smoke workflow failure | high | inspect failing endpoint and rollback if regression |
| Frontend degraded mode | `NEXT_PUBLIC_API_URL` unreachable from dashboard | high | validate Vercel env and backend URL |
| Realtime failure | websocket reconnect success below threshold | medium | verify protocol compatibility and backend ws path |

## Triage Procedure

### 1) Backend outage (Railway)
1. Check Railway startup command and logs for `uvicorn main:app --host 0.0.0.0 --port $PORT`.
2. Hit `/healthz` and `/readyz` manually.
3. If `/readyz` fails, inspect `required_failures` in response.
4. Validate env presence requirements from startup banner logs (`PORT`, `CORS_ALLOW_ORIGINS`, key service vars).

### 2) Frontend drift (Vercel)
1. Confirm `NEXT_PUBLIC_API_URL` and `NEXT_PUBLIC_WS_URL` values.
2. Verify root `vercel.json` and `apps/dashboard/vercel.json` are consistent for the project root.
3. Check browser network traces for failing endpoints in `docs/api-contract-matrix.md`.

### 3) Contract regressions
1. Run smoke checks against deployed backend:
```bash
python scripts/deployment_smoke_check.py --base-url https://<backend-url>
```
2. Compare failures against `docs/api-contract-matrix.md`.
3. If regression introduced in last deploy, rollback backend first, then frontend if needed.

## Rollback Procedure
1. Backend rollback
- Re-deploy previous known-good commit/railway release for `apps/api`.
- Re-run smoke checks against rolled back URL.

2. Frontend rollback
- Promote prior Vercel deployment for `apps/dashboard`.
- Validate dashboard load + portfolio endpoint contract.

3. Post-rollback verification
- `/healthz` returns 200
- `/readyz` returns 200 with no required failures
- smoke workflow passes

## Daily Reliability Checklist
1. Confirm smoke workflow status in GitHub Actions (`deployment-smoke.yml`).
2. Verify latest deploy still serves `/healthz` and `/readyz`.
3. Confirm dashboard can fetch `/api/portfolio` and `/api/strategies`.
4. Track open P0 contract issue: websocket protocol mismatch.

## Incident Template
- Start time (UTC):
- Detection source:
- User impact:
- Failed endpoints:
- Root cause:
- Rollback applied (yes/no):
- Recovery time:
- Follow-up tasks:
