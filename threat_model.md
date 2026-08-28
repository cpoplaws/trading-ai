# Threat Model

## Project Overview

Quantlytics is a crypto AI trading dashboard. The production-facing web artifact is a Vite/React client in `artifacts/trading-dashboard` that reads portfolio, strategy, agent, and market data from an external Python FastAPI trading backend configured at build time through `VITE_API_URL` and `VITE_WS_URL`. A separately packaged Expo static/mobile artifact is served by the built-in Node server in `artifacts/trading-dashboard-mobile/server/serve.js`. `artifacts/api-server` is an Express/PostgreSQL/Drizzle scaffold currently documented as unused by the trading frontend. `.migration-backup` contains an imported legacy application and is reference/development-only unless deployment configuration proves otherwise.

## Assets

- **Trading and portfolio data** -- balances, PnL, positions, signals, strategies, and recent trades can reveal financial activity and influence trading decisions.
- **Trading credentials and wallet/exchange secrets** -- backend API keys, exchange credentials, signing keys, and session/API tokens could enable unauthorized trades or asset theft.
- **Application configuration** -- build-time API URLs and any server-side environment secrets determine which service the client trusts.
- **Static build and server files** -- manifests, JavaScript bundles, source maps, and server-readable files may disclose implementation details or embedded secrets.
- **User/admin authorization state** -- any authenticated trading actions, account settings, or privileged controls must remain bound to the correct account and role.

## Trust Boundaries

- **Browser/mobile client to external trading API** -- all client-controlled identifiers, headers, URLs, and WebSocket messages are untrusted; the backend must authenticate and authorize every sensitive request.
- **Static server to filesystem** -- request paths and headers cross into filesystem reads and must remain confined to the intended static-build directory.
- **Build configuration to browser bundle** -- all `VITE_*` values and bundled code are public; secrets must not be placed there.
- **Trading API to exchanges/blockchain providers** -- backend credentials and privileged operations must never be exposed to the client and outbound calls must validate attacker-controlled inputs.
- **API/scaffold to database** -- future or reachable Express routes must use parameterized queries and object/tenant authorization.
- **Public/unauthenticated to authenticated/admin surfaces** -- server-side identity and role checks are required even if the frontend hides routes or controls.
- **Legacy/reference to production** -- `.migration-backup` code is excluded unless deployment or an active entry point demonstrates reachability.

## Scan Anchors

- Production web entry: `artifacts/trading-dashboard/src/main.tsx`, `src/pages/Dashboard.tsx`, `src/lib/api-client.ts`, `src/hooks/`.
- Production mobile entry: `artifacts/trading-dashboard-mobile/server/serve.js`; generated static content under `static-build/` when present.
- Scaffold entry: `artifacts/api-server/src/index.ts`, `src/app.ts`, `src/routes/`.
- Highest-risk areas: client API/WebSocket configuration, browser bundle/source maps, mobile filesystem serving, external FastAPI authentication/authorization, and any trade/credential endpoints.
- `.migration-backup` and `artifacts/mockup-sandbox` are dev/reference surfaces unless production reachability is established.

## Threat Categories

### Spoofing and Elevation of Privilege

The dashboard itself is a public client and cannot enforce identity or roles. The external trading backend must authenticate every private request and bind portfolio, trade, credential, and account identifiers to the authenticated subject. Role checks must be server-side, and WebSocket connections must have equivalent authentication and tenant isolation.

### Tampering

Trading quantities, destinations, strategy settings, and any account or credential identifiers supplied by a client are attacker-controlled. The backend must recalculate and validate sensitive values server-side, authorize the exact object/action, and require integrity-protected provider callbacks where applicable.

### Information Disclosure

Portfolio and trade data are financial information. APIs, static files, manifests, bundles, source maps, logs, and errors must not disclose other users' data, credentials, signing material, or internal filesystem contents. Browser-delivered configuration is public by design and must contain only non-secret values.

### Injection and Unsafe File Access

Any route that maps a request path to a filesystem file must canonicalize and constrain it to the static root. External API inputs must not reach command, query, template, or unsafe deserialization sinks. Client-rendered user or provider data must be safely encoded before HTML insertion.

### Denial of Service

Public static and API surfaces should bound request size, filesystem work, expensive market/trading operations, and WebSocket connection/resource usage. External calls need bounded timeouts and failure handling.

### Misconfiguration and Cryptographic Failures

Production deployments must not expose debugging interfaces or source secrets. API keys, exchange credentials, and private keys belong in server-side secret storage, use strong transport/session protections, and must not be embedded in Vite bundles or logs. CORS and proxy trust must be narrowly configured.
