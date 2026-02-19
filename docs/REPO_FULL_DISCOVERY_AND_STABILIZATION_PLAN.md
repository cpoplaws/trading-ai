# Repository-Wide Discovery & Stabilization Plan

## 1) Outcomes this plan guarantees

1. A complete, searchable feature inventory that captures **every feature mention** (including brief mentions in docs/comments/configs).
2. A repo map that links each feature to code ownership, runtime entrypoints, tests, and deployment environment.
3. A repeatable triage process to stop Railway backend breaks and Vercel frontend drift in crypto production.
4. A prioritized execution roadmap (stabilize first, then complete missing capabilities).

---


## 1.1) Crypto-first pivot guardrails (mandatory)

- Treat the platform as **crypto-native** across code, docs, monitoring, and deployment language.
- Remove or de-prioritize stock/equity terminology in active plans; retain only where a strategy concept transfers to crypto.
- Keep transferable strategy logic (momentum, mean reversion, RL/ML models), but map each to crypto pairs/venues/liquidity constraints.
- Ensure all runtime validation and smoke tests use crypto endpoints, crypto symbols/pairs, and crypto execution paths.
- Require every new feature entry in the register to include a `market_scope` value (`crypto`, `cross-market`, `legacy-non-crypto`).

---

## 2) Scope map (what to learn, in order)

### Tier A — Runtime-critical surfaces (Day 1–3)
- `apps/api` (Railway backend runtime, FastAPI entrypoint, strategy/intelligence/swarm services).
- `apps/dashboard` (Vercel Next.js runtime, API client/env wiring, websocket hooks).
- Root deployment/config files (`vercel.json`, Railway files, CI workflow files, env examples).

### Tier B — Core platform engine (Day 3–7)
- `src/` modules for strategies, execution, risk, exchanges, dashboards, realtime, API routes, monitoring.
- `api/` + `src/api` duplicates/overlaps to identify canonical API path.
- `config/`, `docker/`, `k8s/`, `infrastructure/` operational setup.

### Tier C — Verification and long-tail capabilities (Week 2)
- `tests/`, `examples/`, `research/`, `notebooks/`, `docs/`, `backtests/`.
- Any archived/backup folders that still influence active deployment behavior.

---

## 3) Feature-capture methodology (ensures no mention is missed)

Create a **Feature Evidence Register** (`docs/feature-evidence-register.csv`) with columns:

- `feature_id`
- `feature_name`
- `market_scope` (crypto/cross-market/legacy-non-crypto)
- `source_type` (code/doc/config/test/example)
- `source_path`
- `source_line`
- `mention_excerpt`
- `implemented_state` (implemented/partial/planned/unclear)
- `runtime_surface` (api/dashboard/worker/infra)
- `dependency_risk`
- `deployment_risk`
- `owner`
- `verification_method`

### Collection passes (mandatory)

1. **Docs pass**: parse all markdown files for explicit and implied features.
2. **Code pass**: parse strategy names, route names, CLI flags, service classes, task schedulers, feature flags.
3. **Config pass**: parse env vars, deployment manifests, compose/k8s resources, feature toggles.
4. **Test/examples pass**: parse described capabilities even if not production enabled.
5. **Cross-check pass**: verify every unique feature noun appears at least once in register.
6. **Crypto conformance pass**: flag and triage stock/equity-only mentions; either map to crypto usage or mark as legacy.

### Suggested extraction commands

```bash
# A) Capture high-signal feature mentions in docs/config
rg -n --glob '*.md' --glob '*.yaml' --glob '*.yml' --glob '*.toml' --glob '*.json' \
  -e 'strategy|agent|risk|dashboard|api|websocket|backtest|arbitrage|portfolio|deployment|railway|vercel|feature|mode|trading'

# B) Capture API/runtime capabilities
rg -n --glob '*.py' --glob '*.ts' --glob '*.tsx' \
  -e '@app\.|APIRouter|router\.|FastAPI\(|uvicorn|websocket|strategy|agent|risk|scheduler|cron|mode|feature'

# C) Capture env/deployment dependencies
rg -n --glob '.env*' --glob '*.example' --glob '*.json' --glob '*.toml' --glob '*.yml' \
  -e 'API_KEY|SECRET|DATABASE|REDIS|PORT|NEXT_PUBLIC|RAILWAY|VERCEL|CORS|ORIGIN'
```

---

## 4) Canonical architecture decisions to lock early

1. **One production backend path**: designate `apps/api` as canonical Railway service unless a migration to `src/api` is intentional and documented.
2. **One production frontend path**: designate `apps/dashboard` as canonical Vercel app.
3. **One source of truth for env vars**: consolidate `.env.example` + deployment platform vars + runtime loader.
4. **One compatibility matrix**: backend routes/response contracts tied to frontend API client expectations.

---

## 5) Railway + Vercel stabilization plan

## Phase 1 — Stop breakages (first 48 hours)

### Backend (Railway)
- Add `/healthz` and `/readyz` checks with dependency checks (DB, Redis, exchange/CEX/DEX connectors optional).
- Confirm startup command resolves from deployed working directory (`uvicorn main:app --host 0.0.0.0 --port $PORT`).
- Enforce deterministic dependency resolution (pin working versions in `apps/api/requirements.txt`).
- Add startup log banner that prints critical env presence (without secrets).
- Add explicit CORS allowlist strategy for Vercel domains.

### Frontend (Vercel)
- Validate `NEXT_PUBLIC_API_URL` is required at build/runtime.
- Add runtime API connectivity banner (degraded mode if backend unavailable).
- Verify root `vercel.json` routing does not conflict with app-level `apps/dashboard/vercel.json`.
- Ensure websocket fallback/reconnect logic is resilient to backend restarts.

### Cross-platform hardening
- Add contract smoke test: frontend can fetch backend health + one key data endpoint.
- Add deployment smoke script run post-deploy from CI.

## Phase 2 — Remove deployment drift (Week 1)
- Add a deployment matrix doc mapping each service to platform/project/env vars/start command.
- Create a single command for local parity (backend + frontend + optional dependencies).
- Add branch-safe preview workflows for both services.
- Add rollback checklist and version tagging strategy.

## Phase 3 — Reliability baseline (Week 2)
- SLOs: API availability, p95 latency, websocket reconnect success, dashboard error rate.
- Error budget policy + incident template.
- Alerting for health failure, startup crash loops, and API contract failures.

---

## 6) Work breakdown structure (WBS)

1. **Inventory & Taxonomy**
   - Build directory map and module catalog.
   - Produce feature evidence register with line-level citations.
2. **Runtime Validation**
   - Verify each entrypoint boots locally.
   - Identify dead/duplicated entrypoints.
3. **Contract Validation**
   - Map frontend API calls to backend routes.
   - Flag mismatches and missing fields.
4. **Deployment Validation**
   - Reproduce Railway and Vercel builds locally (where possible).
   - Validate env var completeness and secrets handling.
5. **Gap Closure Roadmap**
   - Prioritize features as P0 (breaking), P1 (core), P2 (enhancement).

---

## 7) Deliverables checklist

- [ ] `docs/repo-topology.md` (canonical map of modules and runtimes)
- [ ] `docs/feature-evidence-register.csv` (all features + evidence)
- [ ] `docs/feature-implementation-matrix.md` (feature -> code -> test -> deployment)
- [ ] `docs/deployment-matrix.md` (Railway/Vercel vars, commands, healthchecks)
- [ ] `docs/api-contract-matrix.md` (frontend hooks/pages -> backend endpoints)
- [ ] `docs/reliability-runbook.md` (triage + rollback + alerts)

---

## 8) 14-day execution timeline

### Days 1–2
- Build inventory and feature register skeleton.
- Lock canonical runtime paths (`apps/api`, `apps/dashboard`).
- Patch immediate deployment blockers.

### Days 3–5
- Complete feature evidence harvesting across all folders.
- Complete API contract mapping and drift report.
- Add health endpoints + smoke checks.

### Days 6–8
- Resolve top P0/P1 mismatches.
- Add deployment matrix and env validation.
- Set up reliability alerts and dashboards.

### Days 9–11
- Validate all documented features have status (implemented/partial/planned).
- Update docs so no feature exists only in tribal knowledge.

### Days 12–14
- Final repo-wide review.
- Publish prioritized backlog for remaining feature completion.
- Freeze baseline with release tag and runbook handoff.

---

## 9) Definition of done (for “learned in entirety”)

This initiative is complete when:

1. Every feature mention in repo text/code/config is captured in the feature evidence register.
2. Every captured feature has implementation status and owner.
3. Every production route/UI dependency is contract-verified.
4. Railway and Vercel deployments have automated smoke checks and rollback steps.
5. Unknown/duplicate modules are either deprecated or assigned a migration plan.

---

## 10) Immediate next 5 actions (start now)

1. Generate initial feature evidence register from docs + runtime code.
2. Produce backend route inventory from `apps/api/main.py` and related routers.
3. Produce frontend API usage inventory from `apps/dashboard/lib` + hooks + pages.
4. Reconcile env vars across `.env.example`, Railway, and Vercel configs.
5. Implement health/smoke checks and run them on every deploy.
