#!/usr/bin/env node
/**
 * Staging-backed contract check for the Quantlytics web dashboard.
 *
 * Exercises every endpoint the dashboard calls — `src/lib/api-client.ts` plus
 * the direct `fetch` calls in `src/components/dashboard/` — using the exact
 * paths, methods and payloads the dashboard sends, and asserts the response
 * shapes its components read. It also connects to the live-update channel the
 * `useWebSocket` hook subscribes to and asserts the event payloads.
 *
 * A response-shape or endpoint regression in the trading backend fails this
 * check before it can reach dashboard users.
 *
 * Runs against the safe staging mirror (`/api/staging-trading`) only: the check
 * refuses to run unless the backend reports `demo_mode`, so it can never place
 * orders or read live trading data.
 *
 * Usage:
 *   node artifacts/trading-dashboard/testing/dashboard-api-contract.mjs
 *   STAGING_API_URL=https://<host>/api/staging-trading node .../dashboard-api-contract.mjs
 */

import { io } from 'socket.io-client'

const BASE_URL = (process.env.STAGING_API_URL ?? 'http://localhost:8080/api/staging-trading').replace(/\/$/, '')

let failures = 0
let checks = 0

function check(label, condition, detail) {
  checks += 1
  if (condition) {
    console.log(`  ✓ ${label}`)
  } else {
    failures += 1
    console.log(`  ✗ ${label}${detail ? ` — ${detail}` : ''}`)
  }
}

function section(title) {
  console.log(`\n${title}`)
}

/** Mirrors the dashboard's axios client / component `fetch` calls. */
async function request(path, options) {
  const response = await fetch(`${BASE_URL}${path}`, {
    ...options,
    headers: { 'Content-Type': 'application/json', ...(options?.headers ?? {}) },
  })

  if (!response.ok) {
    throw new Error(`API request failed (${response.status}) for ${options?.method ?? 'GET'} ${path}`)
  }

  return response.json()
}

function isFiniteNumber(value) {
  return typeof value === 'number' && Number.isFinite(value)
}

function isNonEmptyString(value) {
  return typeof value === 'string' && value.length > 0
}

function missingNumberFields(object, fields) {
  return fields.filter((field) => !isFiniteNumber(object?.[field]))
}

/** Same resolution `src/hooks/useWebSocket.ts` applies to `VITE_WS_URL`. */
function resolveSocketTarget(url) {
  try {
    const parsed = new URL(url)
    const prefix = parsed.pathname.replace(/\/+$/, '')
    return { origin: `${parsed.protocol}//${parsed.host}`, path: `${prefix}/socket.io` }
  } catch {
    return { origin: url, path: '/socket.io' }
  }
}

/**
 * Connects exactly as `useWebSocket` does and collects the first payload of
 * each event the hook subscribes to.
 */
function collectLiveUpdates(wsUrl, timeoutMs = 12000) {
  const { origin, path } = resolveSocketTarget(wsUrl)
  const events = ['portfolio_update', 'trade_update', 'strategy_update']

  return new Promise((resolve) => {
    const received = {}
    let connected = false

    const socket = io(origin, {
      path,
      transports: ['websocket'],
      reconnection: true,
      reconnectionDelay: 1000,
      reconnectionAttempts: 5,
    })

    const finish = (error) => {
      clearTimeout(timer)
      socket.close()
      resolve({ connected, received, error })
    }

    const timer = setTimeout(() => finish(connected ? 'timed out waiting for events' : 'never connected'), timeoutMs)

    socket.on('connect', () => {
      connected = true
    })
    socket.on('connect_error', (error) => finish(`connect_error: ${error?.message ?? error}`))

    for (const event of events) {
      socket.on(event, (payload) => {
        received[event] ??= payload
        if (events.every((name) => name in received)) finish(null)
      })
    }
  })
}

async function main() {
  console.log(`Dashboard trading API contract check\nTarget: ${BASE_URL}\n`)

  // --- Safety gate: never run against a live trading backend ---------------
  const health = await request('/health')
  if (health?.demo_mode !== true) {
    console.error('ABORT: target backend did not report demo_mode. Refusing to exercise trading mutations.')
    process.exit(2)
  }
  console.log('Safety gate: backend reports demo_mode — no live trading data will be touched.')

  // Deterministic starting point so repeated runs assert the same states.
  await request('/reset', { method: 'POST' })

  // --- Header + stats grid (Dashboard.tsx) ---------------------------------
  section('Stats grid — portfolio polling')
  const portfolio = await request('/api/portfolio')
  const portfolioNumbers = [
    'total_value',
    'cash',
    'buying_power',
    'daily_pnl',
    'daily_pnl_percent',
    'positions_count',
    'sharpe_ratio',
    'win_rate',
  ]
  const missingPortfolio = missingNumberFields(portfolio, portfolioNumbers)
  check('GET /api/portfolio returns every numeric field the stats grid renders', missingPortfolio.length === 0, `missing/non-numeric: ${missingPortfolio.join(', ')}`)
  check('portfolio.demo_mode is true (staging, not live trading)', portfolio.demo_mode === true, `got ${JSON.stringify(portfolio.demo_mode)}`)
  check('win_rate is a 0–1 ratio (the dashboard multiplies by 100)', portfolio.win_rate >= 0 && portfolio.win_rate <= 1, `got ${portfolio.win_rate}`)

  const historyResponse = await request('/api/portfolio/history?days=30')
  const history = Array.isArray(historyResponse) ? historyResponse : (historyResponse?.history ?? [])
  check('GET /api/portfolio/history?days=30 returns an array (bare or under `history`)', Array.isArray(history) && history.length > 0, `got ${JSON.stringify(historyResponse)?.slice(0, 120)}`)
  const chartValues = history.map((point) => point?.value ?? point?.total_value ?? 0).filter((value) => value > 0)
  check('history points expose `value` or `total_value` so a chart can plot them', chartValues.length > 1, `${chartValues.length} usable point(s)`)
  check('history points carry a parseable `timestamp`', history.every((point) => !Number.isNaN(new Date(point?.timestamp).getTime())))

  // --- Market Intelligence (MarketIntelligence.tsx) -------------------------
  section('Market Intelligence — signals panel')
  const intelligence = await request('/api/intelligence')
  check('GET /api/intelligence returns a `signal` string', isNonEmptyString(intelligence?.signal), `got ${JSON.stringify(intelligence?.signal)}`)
  check('composite_score is a -1..1 number (rendered as a bar width)', isFiniteNumber(intelligence?.composite_score) && Math.abs(intelligence.composite_score) <= 1, `got ${intelligence?.composite_score}`)
  check('confidence is numeric', isFiniteNumber(intelligence?.confidence), `got ${intelligence?.confidence}`)

  const missingRegime = missingNumberFields(intelligence?.regime, ['confidence', 'momentum', 'volatility'])
  check('regime has a `regime` label plus confidence/momentum/volatility', isNonEmptyString(intelligence?.regime?.regime) && missingRegime.length === 0, `missing: ${missingRegime.join(', ')}`)
  check('sentiment has a label and a numeric score', isNonEmptyString(intelligence?.sentiment?.sentiment) && isFiniteNumber(intelligence?.sentiment?.score), `got ${JSON.stringify(intelligence?.sentiment)}`)

  const missingTechnical = missingNumberFields(intelligence?.technical, ['rsi', 'macd', 'bb_position'])
  check('technical exposes rsi/macd/bb_position as numbers (all are `.toFixed()`-ed)', missingTechnical.length === 0, `missing/non-numeric: ${missingTechnical.join(', ')}`)
  check('macro.trend is present', isNonEmptyString(intelligence?.macro?.trend), `got ${JSON.stringify(intelligence?.macro)}`)

  check('recommendations is an array of strings', Array.isArray(intelligence?.recommendations) && intelligence.recommendations.every(isNonEmptyString), `got ${JSON.stringify(intelligence?.recommendations)?.slice(0, 120)}`)
  const badAlerts = (intelligence?.alerts ?? []).filter(
    (alert) => !isNonEmptyString(alert?.type) || !isNonEmptyString(alert?.message) || !isNonEmptyString(alert?.severity),
  )
  check('alerts is an array of {type, message, severity}', Array.isArray(intelligence?.alerts) && badAlerts.length === 0, `${badAlerts.length} malformed alert(s)`)
  check('timestamp parses as a date (rendered as "Last updated")', !Number.isNaN(new Date(intelligence?.timestamp).getTime()), `got ${JSON.stringify(intelligence?.timestamp)}`)

  // --- Strategy grid (StrategyGrid.tsx / useStrategies.ts) ------------------
  section('Strategy grid — list and enable/disable')
  const strategiesResponse = await request('/api/strategies')
  const strategies = strategiesResponse?.strategies ?? []
  check('GET /api/strategies returns `{ strategies: [...] }`', Array.isArray(strategiesResponse?.strategies) && strategies.length > 0, `got ${JSON.stringify(strategiesResponse)?.slice(0, 120)}`)

  const badStrategies = strategies.filter(
    (strategy) =>
      !isNonEmptyString(strategy?.id) ||
      !isNonEmptyString(strategy?.name) ||
      typeof strategy?.enabled !== 'boolean' ||
      !isFiniteNumber(strategy?.pnl) ||
      !isFiniteNumber(strategy?.trades) ||
      !isFiniteNumber(strategy?.win_rate),
  )
  check('every strategy has id/name/enabled/pnl/trades/win_rate', badStrategies.length === 0, `${badStrategies.length} malformed entr(ies)`)

  const target = strategies.find((strategy) => strategy.enabled) ?? strategies[0]
  if (target) {
    await request(`/api/strategies/${target.id}/toggle`, { method: 'POST', body: JSON.stringify({ enabled: false }) })
    const afterDisable = (await request('/api/strategies')).strategies.find((item) => item.id === target.id)
    check(`POST /api/strategies/${target.id}/toggle {enabled:false} disables it`, afterDisable?.enabled === false, `enabled=${afterDisable?.enabled}`)

    await request(`/api/strategies/${target.id}/toggle`, { method: 'POST', body: JSON.stringify({ enabled: true }) })
    const afterEnable = (await request('/api/strategies')).strategies.find((item) => item.id === target.id)
    check(`POST /api/strategies/${target.id}/toggle {enabled:true} re-enables it`, afterEnable?.enabled === true, `enabled=${afterEnable?.enabled}`)
  } else {
    check('a strategy was available to toggle', false, 'strategy list was empty')
  }

  // --- Agent swarm (AgentSwarm.tsx) ----------------------------------------
  section('Agent swarm — status, toggles and decisions')
  const swarm = await request('/api/agents/status')
  check('GET /api/agents/status returns `enabled`', typeof swarm?.enabled === 'boolean', `got ${JSON.stringify(swarm?.enabled)}`)
  check('swarm exposes `coordination_mode`', isNonEmptyString(swarm?.coordination_mode), `got ${JSON.stringify(swarm?.coordination_mode)}`)

  const agentEntries = Object.entries(swarm?.agents ?? {})
  check('swarm.agents is a keyed record of agents', agentEntries.length > 0, `${agentEntries.length} agent(s)`)
  const badAgents = agentEntries.filter(
    ([, agent]) =>
      typeof agent?.enabled !== 'boolean' ||
      !isFiniteNumber(agent?.performance?.total_decisions) ||
      !isFiniteNumber(agent?.performance?.successful_decisions) ||
      !isFiniteNumber(agent?.performance?.failed_decisions) ||
      !isFiniteNumber(agent?.performance?.accuracy) ||
      !isFiniteNumber(agent?.recent_decisions),
  )
  check('every agent has enabled + full performance block + recent_decisions', badAgents.length === 0, `malformed: ${badAgents.map(([key]) => key).join(', ')}`)

  await request('/api/agents/disable', { method: 'POST' })
  check('POST /api/agents/disable pauses the swarm', (await request('/api/agents/status')).enabled === false)
  await request('/api/agents/enable', { method: 'POST' })
  check('POST /api/agents/enable resumes the swarm', (await request('/api/agents/status')).enabled === true)

  const [agentKey, agentBefore] = agentEntries[0] ?? []
  if (agentKey) {
    await request(`/api/agents/${agentKey}/toggle`, { method: 'POST' })
    const flipped = (await request('/api/agents/status')).agents?.[agentKey]
    check(`POST /api/agents/${agentKey}/toggle flips that agent`, flipped?.enabled === !agentBefore.enabled, `${agentBefore.enabled} -> ${flipped?.enabled}`)

    await request(`/api/agents/${agentKey}/toggle`, { method: 'POST' })
    const restored = (await request('/api/agents/status')).agents?.[agentKey]
    check(`POST /api/agents/${agentKey}/toggle restores it`, restored?.enabled === agentBefore.enabled, `enabled=${restored?.enabled}`)
  } else {
    check('an agent was available to toggle', false, 'agent record was empty')
  }

  const decisionsResponse = await request('/api/agents/decisions?limit=10')
  const decisions = decisionsResponse?.decisions ?? []
  check('GET /api/agents/decisions?limit=10 returns `{ decisions: [...] }`', Array.isArray(decisionsResponse?.decisions), `got ${JSON.stringify(decisionsResponse)?.slice(0, 120)}`)
  check('decisions honours the limit', decisions.length <= 10, `got ${decisions.length}`)
  const badDecisions = decisions.filter(
    (decision) =>
      !isNonEmptyString(decision?.agent) ||
      !isNonEmptyString(decision?.action) ||
      !isFiniteNumber(decision?.confidence) ||
      !isNonEmptyString(decision?.reason) ||
      Number.isNaN(new Date(decision?.timestamp).getTime()),
  )
  check('every decision has agent/action/confidence/reason/parseable timestamp', badDecisions.length === 0, `${badDecisions.length} malformed entr(ies)`)

  // --- Recent trades (RecentTrades.tsx) -------------------------------------
  section('Recent trades — polling')
  const tradesResponse = await request('/api/trades/recent?limit=20')
  const trades = tradesResponse?.trades ?? []
  check('GET /api/trades/recent?limit=20 returns `{ trades: [...] }`', Array.isArray(tradesResponse?.trades), `got ${JSON.stringify(tradesResponse)?.slice(0, 120)}`)
  check('trades honours the limit', trades.length <= 20, `got ${trades.length}`)

  const badTrades = trades.filter(
    (trade) =>
      !isNonEmptyString(trade?.id) ||
      !isNonEmptyString(trade?.symbol) ||
      !isNonEmptyString(trade?.side) ||
      !isFiniteNumber(trade?.quantity) ||
      !isFiniteNumber(trade?.price) ||
      !isNonEmptyString(trade?.timestamp) ||
      !isNonEmptyString(trade?.strategy),
  )
  check('every trade has id/symbol/side/quantity/price/timestamp/strategy', badTrades.length === 0, `${badTrades.length} malformed entr(ies)`)
  check('trade ids are unique (the dashboard keys rows by id)', new Set(trades.map((trade) => trade.id)).size === trades.length)
  check('trade timestamps parse as dates (rendered with toLocaleString)', trades.every((trade) => !Number.isNaN(new Date(trade.timestamp).getTime())))
  check('trade.side is buy/sell (the dashboard branches on it)', trades.every((trade) => ['buy', 'sell'].includes(String(trade.side).toLowerCase())))
  check('trade.pnl, when present, is numeric (`.toFixed(2)`)', trades.every((trade) => trade.pnl === undefined || isFiniteNumber(trade.pnl)))

  // --- Live-update channel (useWebSocket.ts / VITE_WS_URL) ------------------
  section('Live updates — WebSocket channel')
  const wsUrl = process.env.STAGING_WS_URL ?? BASE_URL
  const live = await collectLiveUpdates(wsUrl)
  check(`socket.io connects to ${resolveSocketTarget(wsUrl).origin} (path ${resolveSocketTarget(wsUrl).path})`, live.connected, live.error ?? undefined)

  const livePortfolio = live.received.portfolio_update
  const missingLivePortfolio = missingNumberFields(livePortfolio, portfolioNumbers)
  check('`portfolio_update` carries the same numeric fields as GET /api/portfolio', livePortfolio !== undefined && missingLivePortfolio.length === 0, livePortfolio === undefined ? 'event never arrived' : `missing/non-numeric: ${missingLivePortfolio.join(', ')}`)

  const liveTrade = live.received.trade_update
  check(
    '`trade_update` carries a full trade (id/symbol/side/quantity/price/timestamp/strategy)',
    liveTrade !== undefined &&
      isNonEmptyString(liveTrade.id) &&
      isNonEmptyString(liveTrade.symbol) &&
      isNonEmptyString(liveTrade.side) &&
      isFiniteNumber(liveTrade.quantity) &&
      isFiniteNumber(liveTrade.price) &&
      isNonEmptyString(liveTrade.timestamp) &&
      isNonEmptyString(liveTrade.strategy),
    liveTrade === undefined ? 'event never arrived' : `got ${JSON.stringify(liveTrade)?.slice(0, 120)}`,
  )

  const liveStrategy = live.received.strategy_update
  check(
    '`strategy_update` carries a full strategy (id/name/enabled/pnl/trades/win_rate)',
    liveStrategy !== undefined &&
      isNonEmptyString(liveStrategy.id) &&
      isNonEmptyString(liveStrategy.name) &&
      typeof liveStrategy.enabled === 'boolean' &&
      isFiniteNumber(liveStrategy.pnl) &&
      isFiniteNumber(liveStrategy.trades) &&
      isFiniteNumber(liveStrategy.win_rate),
    liveStrategy === undefined ? 'event never arrived' : `got ${JSON.stringify(liveStrategy)?.slice(0, 120)}`,
  )

  // --- Restore the staging fixtures ----------------------------------------
  await request('/reset', { method: 'POST' })

  console.log(`\n${checks - failures}/${checks} checks passed.`)
  if (failures > 0) {
    console.error(`${failures} contract check(s) FAILED.`)
    process.exit(1)
  }
  console.log('Dashboard trading API contract is intact.')
}

main().catch((error) => {
  console.error(`\nContract check could not complete: ${error instanceof Error ? error.message : String(error)}`)
  console.error('Is the API Server workflow running? Expected staging mirror at ' + BASE_URL)
  process.exit(1)
})
