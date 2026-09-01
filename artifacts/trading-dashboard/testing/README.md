# Staging-backed checks for the dashboard trading controls

The web dashboard talks straight to the external Python trading engine — REST
via `VITE_API_URL` (`src/lib/api-client.ts` plus direct `fetch` calls in
`src/components/dashboard/`) and live updates via `VITE_WS_URL`
(`src/hooks/useWebSocket.ts`). Running release checks against that engine would
place real orders, so these checks run against the same **safe staging mirror**
the mobile checks use.

## The staging mirror

`artifacts/api-server/src/routes/staging-trading.ts` reproduces every endpoint
the dashboard calls, with the same paths, methods, payloads and response shapes,
backed by deterministic in-memory fixtures.

- Base URL: `https://$REPLIT_DEV_DOMAIN/api/staging-trading`
  (locally: `http://localhost:8080/api/staging-trading`)
- Client paths are unchanged underneath it — e.g. `/api/staging-trading/api/portfolio`
- `GET /api/staging-trading/health` reports `{ demo_mode: true }`
- `POST /api/staging-trading/reset` restores the fixtures
- Portfolio always reports `demo_mode: true`; there is no brokerage, database or
  credential access anywhere in the mirror
- Enabled in development; in production it requires `ENABLE_STAGING_TRADING_API=1`

Endpoints the dashboard uses, on top of the ones the mobile app shares:

| Dashboard caller | Endpoint |
| --- | --- |
| `MarketIntelligence` | `GET /api/intelligence` |
| `useWebSocket` | Socket.IO — `portfolio_update`, `trade_update`, `strategy_update` |

The rest (`/api/portfolio`, `/api/portfolio/history`, `/api/strategies`,
`/api/strategies/:id/toggle`, `/api/agents/status`, `/api/agents/decisions`,
`/api/agents/enable`, `/api/agents/disable`, `/api/agents/:name/toggle`,
`/api/trades/recent`) are the mirror's existing mobile coverage, asserted here
against the fields the *dashboard* reads.

### The live-update channel

Socket.IO always connects to `<origin>` plus its `path` option and treats any
path in the URL it is handed as a *namespace*, so the mirror cannot serve it at
the default root `/socket.io` — only `/api/*` reaches the API server through the
workspace proxy. It is served at `/api/staging-trading/socket.io` instead, and
`useWebSocket` derives that path from the prefix in `VITE_WS_URL`. Against the
root-mounted Python engine (`ws://localhost:8000`) this resolves to the default
`/socket.io`, so live behaviour is unchanged.

Payloads are slices of the same fixtures the REST mirror serves — a snapshot on
connect, then a tick every 2s.

## 1. Contract check (fast, run on every change)

Catches endpoint and response-shape regressions before they reach dashboard
users.

```bash
# API Server workflow must be running
pnpm --filter @workspace/trading-dashboard run check:trading-api
```

It refuses to run unless the backend reports `demo_mode`, then exercises every
REST endpoint — including the strategy enable/disable and swarm/agent toggle
round-trips — plus the WebSocket channel, and asserts each field the dashboard
components read.

Registered as the `dashboard-trading-api` validation command, alongside
`mobile-trading-api`.

Point it at another (non-live) backend with `STAGING_API_URL` (and
`STAGING_WS_URL`, which defaults to `STAGING_API_URL`).

Not covered: `getBaseBalance` / `executeBaseTrade` in `src/lib/api-client.ts`.
No dashboard component calls them, so there is no rendered shape to assert and
the mirror deliberately does not reproduce a trade-execution endpoint.

## 2. Desktop-width browser check

Run the dashboard against the staging mirror and drive it at desktop width
(1440x900):

```bash
DASHBOARD_API_URL="https://$REPLIT_DEV_DOMAIN/api/staging-trading" \
DASHBOARD_WS_URL="https://$REPLIT_DEV_DOMAIN/api/staging-trading" \
  pnpm --filter @workspace/trading-dashboard run dev
```

`DASHBOARD_API_URL` / `DASHBOARD_WS_URL` take precedence over the shared
`VITE_API_URL` / `VITE_WS_URL`, so this override never touches the mobile app's
configuration.

What the browser pass covers:

- **Portfolio stats** — the four cards (Total Portfolio, Sharpe Ratio, Win Rate,
  Cash Available) render values from the polled `/api/portfolio` payload, and
  the header shows the `(Demo Mode)` marker.
- **Market Intelligence** — signal badge, composite-score bar, regime and
  sentiment tiles, the three technical indicators, alerts and recommendations.
- **Strategy grid** — one card per strategy with P&L, trades and win rate, and
  the Enable/Disable button flipping a card's state, surviving the 10s refetch.
- **Agent swarm** — swarm status badge, the swarm Pause/Resume control, one
  specialist's Enable/Disable button, and the Recent Decisions list.
- **Recent trades** — rows with side badge, symbol, quantity, strategy, time,
  price and P&L; the 5s poll keeps the list stable.

Note: `src/components/dashboard/PortfolioChart.tsx` is a static placeholder that
the dashboard does not render, so there is nothing chart-shaped to assert in the
browser. The check covers the portfolio *data* path instead — the stats cards
plus the `/api/portfolio/history` contract that a real chart would plot.

To verify the error states instead, restart the dashboard with
`DASHBOARD_API_URL="http://localhost:1"` — every section should show its error
message rather than crashing.
