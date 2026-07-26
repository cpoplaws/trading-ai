# Quantlytics — Crypto AI Trading Dashboard

A multi-chain AI-driven trading platform dashboard. Monitors portfolio performance, AI agent swarm activity, market intelligence signals, and trading strategies across Base, Solana, and L2 chains.

## Run & Operate

- `pnpm --filter @workspace/trading-dashboard run dev` — run the dashboard frontend
- `pnpm --filter @workspace/api-server run dev` — run the API server
- `pnpm run typecheck` — full typecheck across all packages
- `pnpm --filter @workspace/api-spec run codegen` — regenerate API hooks and Zod schemas from the OpenAPI spec
- Required env: `VITE_API_URL` — URL of the Python trading backend (defaults to `http://localhost:8000`)
- Required env: `VITE_WS_URL` — WebSocket URL for live updates (defaults to `ws://localhost:8000`)

## Stack

- pnpm workspaces, Node.js 24, TypeScript 5.9
- Frontend: Vite + React 18, Tailwind CSS v4, wouter for routing
- API: Express 5 (scaffold, not yet used by the trading frontend)
- DB: PostgreSQL + Drizzle ORM (scaffold)
- Backend trading engine: Python FastAPI (external service in `.migration-backup/`)

## Where things live

- `artifacts/trading-dashboard/` — the main dashboard web app
  - `src/pages/Dashboard.tsx` — main dashboard page
  - `src/components/dashboard/` — AgentSwarm, MarketIntelligence, StrategyGrid, RecentTrades, PortfolioChart
  - `src/hooks/` — usePortfolio, useStrategies, useWebSocket
  - `src/lib/api-client.ts` — axios client pointing at the Python backend
- `artifacts/api-server/` — Express API scaffold (unused by the trading app; available for future features)
- `.migration-backup/` — original Vercel/Next.js import (read-only reference)
- `lib/api-spec/openapi.yaml` — OpenAPI spec (scaffold, extend for new features)

## Architecture decisions

- The trading frontend connects directly to the external Python FastAPI backend via `VITE_API_URL`. It does NOT use the workspace Express api-server. The api-server is available for future Replit-native features.
- App is dark-mode only — `dark` class is added to `<html>` in `main.tsx`. All theme CSS vars are set for dark in both `:root` and `.dark`.
- `process.env.NEXT_PUBLIC_*` vars were converted to `import.meta.env.VITE_*` during migration.
- No Next.js API routes existed in the original — the app is purely a frontend dashboard consuming an external backend.

## Product

A real-time AI trading dashboard showing: live portfolio value/PnL, AI agent swarm status and decisions, market intelligence signals (regime, sentiment, technical indicators), trading strategy performance, and recent trades feed.

## User preferences

_Populate as you build — explicit user instructions worth remembering across sessions._

## Gotchas

- The Python trading backend must be deployed separately and `VITE_API_URL` pointed at it. Without it, all sections show "Failed to fetch" or empty states — this is expected.
- Do NOT run `pnpm dev` at the workspace root — use `--filter @workspace/trading-dashboard` instead.
- Tailwind v4 is used (via `@tailwindcss/vite`), not v3. The config is in `vite.config.ts`, not `tailwind.config.js`.

## Pointers

- See the `pnpm-workspace` skill for workspace structure, TypeScript setup, and package details
