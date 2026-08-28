# Staging-backed checks for the mobile trading controls

The mobile app talks straight to the external Python trading engine
(`lib/trading-api.ts`, base URL from `EXPO_PUBLIC_API_URL`). Running release
checks against that engine would place real orders, so these checks run against
a **safe staging mirror** instead.

## The staging mirror

`artifacts/api-server/src/routes/staging-trading.ts` reproduces every endpoint
the mobile client calls, with the same paths, methods, payloads and response
shapes, backed by deterministic in-memory fixtures.

- Base URL: `https://$REPLIT_DEV_DOMAIN/api/staging-trading`
  (locally: `http://localhost:8080/api/staging-trading`)
- Client paths are unchanged underneath it — e.g. `/api/staging-trading/api/portfolio`
- `GET /api/staging-trading/health` reports `{ demo_mode: true }`
- `POST /api/staging-trading/reset` restores the fixtures
- Portfolio always reports `demo_mode: true`; there is no brokerage, database or
  credential access anywhere in the mirror
- Enabled in development; in production it requires `ENABLE_STAGING_TRADING_API=1`

Endpoints mirrored:

| Mobile call | Endpoint |
| --- | --- |
| `getPortfolio` | `GET /api/portfolio` |
| `getPortfolioHistory` | `GET /api/portfolio/history?days=7` |
| `getStrategies` | `GET /api/strategies` |
| `setStrategyEnabled` | `POST /api/strategies/:id/toggle` |
| `getSwarmStatus` | `GET /api/agents/status` |
| `getAgentDecisions` | `GET /api/agents/decisions?limit=5` |
| `setSwarmEnabled` | `POST /api/agents/enable` · `POST /api/agents/disable` |
| `toggleAgent` | `POST /api/agents/:name/toggle` |
| `getRecentTrades` | `GET /api/trades/recent?limit=20` |

## 1. Contract check (fast, run on every change)

Catches endpoint and response-shape regressions before they reach iOS/Android.

```bash
# API Server workflow must be running
pnpm --filter @workspace/trading-dashboard-mobile run check:trading-api
```

It refuses to run unless the backend reports `demo_mode`, then exercises all
nine endpoints — including the strategy enable/disable and swarm/agent toggle
round-trips — and asserts each field the mobile screens read.

Registered as the `mobile-trading-api` validation command.

Point it at another (non-live) backend with `STAGING_API_URL`.

## 2. Mobile-width browser check (all four tabs)

Run the Expo app against the staging mirror and drive it at phone width
(402x874):

```bash
EXPO_PUBLIC_API_URL="https://$REPLIT_DEV_DOMAIN/api/staging-trading" \
  pnpm --filter @workspace/trading-dashboard-mobile run dev
```

`EXPO_PUBLIC_API_URL` takes precedence over `VITE_API_URL`, so this override
never touches the web dashboard's configuration.

What the browser pass covers:

- **Overview** — portfolio card renders values from the polled payload, shows
  the `DEMO MODE` pill, sparkline renders from `/portfolio/history`, and the
  four metric tiles (cash, Sharpe, win rate, daily P&L) are populated.
- **Strategies** — grid loads, the active count matches, and toggling a card
  (`strategy-toggle-<id>`) flips it, with the state surviving the refetch.
- **Agents** — swarm status pill, `swarm-toggle` pausing/resuming the swarm,
  `agent-toggle-<key>` flipping one specialist, and the decisions list.
- **Trades** — recent-trade rows with symbol, quantity, strategy, time, price
  and P&L; polling keeps the list stable.

Useful `testID`s: `strategy-toggle-<strategyId>`, `swarm-toggle`,
`agent-toggle-<agentKey>`.

To verify the error/empty states instead, restart the app with
`EXPO_PUBLIC_API_URL=""` (every screen should show its `ErrorState` with an
"API is not configured" message rather than crashing).
