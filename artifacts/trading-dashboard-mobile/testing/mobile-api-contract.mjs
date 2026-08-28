#!/usr/bin/env node
/**
 * Staging-backed contract check for the Quantlytics mobile trading controls.
 *
 * Exercises every endpoint `lib/trading-api.ts` calls, using the exact paths,
 * methods and payloads the mobile client sends, and asserts the response shapes
 * the mobile TypeScript interfaces depend on. A response-shape or endpoint
 * regression in the trading backend fails this check before it can ship to iOS
 * and Android users.
 *
 * Runs against the safe staging mirror (`/api/staging-trading`) only: the check
 * refuses to run unless the backend reports `demo_mode`, so it can never place
 * orders or read live trading data.
 *
 * Usage:
 *   node artifacts/trading-dashboard-mobile/testing/mobile-api-contract.mjs
 *   STAGING_API_URL=https://<host>/api/staging-trading node .../mobile-api-contract.mjs
 */

const BASE_URL = (process.env.STAGING_API_URL ?? 'http://localhost:8080/api/staging-trading').replace(/\/$/, '');

let failures = 0;
let checks = 0;

function check(label, condition, detail) {
  checks += 1;
  if (condition) {
    console.log(`  ✓ ${label}`);
  } else {
    failures += 1;
    console.log(`  ✗ ${label}${detail ? ` — ${detail}` : ''}`);
  }
}

function section(title) {
  console.log(`\n${title}`);
}

/** Mirrors the mobile client's `request()` helper. */
async function request(path, options) {
  const response = await fetch(`${BASE_URL}${path}`, {
    ...options,
    headers: { 'Content-Type': 'application/json', ...(options?.headers ?? {}) },
  });

  if (!response.ok) {
    throw new Error(`API request failed (${response.status}) for ${options?.method ?? 'GET'} ${path}`);
  }

  return response.json();
}

function isFiniteNumber(value) {
  return typeof value === 'number' && Number.isFinite(value);
}

function isNonEmptyString(value) {
  return typeof value === 'string' && value.length > 0;
}

function missingNumberFields(object, fields) {
  return fields.filter((field) => !isFiniteNumber(object?.[field]));
}

async function main() {
  console.log(`Mobile trading API contract check\nTarget: ${BASE_URL}\n`);

  // --- Safety gate: never run against a live trading backend ---------------
  const health = await request('/health');
  if (health?.demo_mode !== true) {
    console.error('ABORT: target backend did not report demo_mode. Refusing to exercise trading mutations.');
    process.exit(2);
  }
  console.log('Safety gate: backend reports demo_mode — no live trading data will be touched.');

  // Deterministic starting point so repeated runs assert the same states.
  await request('/reset', { method: 'POST' });

  // --- Overview tab: portfolio polling -------------------------------------
  section('Overview tab — portfolio polling');
  const portfolio = await request('/api/portfolio');
  const portfolioNumbers = [
    'total_value',
    'cash',
    'buying_power',
    'daily_pnl',
    'daily_pnl_percent',
    'positions_count',
    'sharpe_ratio',
    'win_rate',
  ];
  const missingPortfolio = missingNumberFields(portfolio, portfolioNumbers);
  check('GET /api/portfolio returns every numeric field the overview renders', missingPortfolio.length === 0, `missing/non-numeric: ${missingPortfolio.join(', ')}`);
  check('portfolio.demo_mode is true (staging, not live trading)', portfolio.demo_mode === true, `got ${JSON.stringify(portfolio.demo_mode)}`);
  check('win_rate is a 0–1 ratio (mobile multiplies by 100)', portfolio.win_rate >= 0 && portfolio.win_rate <= 1, `got ${portfolio.win_rate}`);

  const historyResponse = await request('/api/portfolio/history?days=7');
  const history = Array.isArray(historyResponse) ? historyResponse : (historyResponse?.history ?? []);
  check('GET /api/portfolio/history?days=7 returns an array (bare or under `history`)', Array.isArray(history) && history.length > 0, `got ${JSON.stringify(historyResponse)?.slice(0, 120)}`);
  const chartValues = history.map((point) => point?.value ?? point?.total_value ?? 0).filter((value) => value > 0);
  check('history points expose `value` or `total_value` so the sparkline renders', chartValues.length > 1, `${chartValues.length} usable point(s)`);

  // --- Strategies tab: list + enable/disable --------------------------------
  section('Strategies tab — list and enable/disable');
  const strategiesResponse = await request('/api/strategies');
  const strategies = strategiesResponse?.strategies ?? [];
  check('GET /api/strategies returns `{ strategies: [...] }`', Array.isArray(strategiesResponse?.strategies) && strategies.length > 0, `got ${JSON.stringify(strategiesResponse)?.slice(0, 120)}`);

  const badStrategies = strategies.filter(
    (strategy) =>
      !isNonEmptyString(strategy?.id) ||
      !isNonEmptyString(strategy?.name) ||
      typeof strategy?.enabled !== 'boolean' ||
      !isFiniteNumber(strategy?.pnl) ||
      !isFiniteNumber(strategy?.trades) ||
      !isFiniteNumber(strategy?.win_rate),
  );
  check('every strategy has id/name/enabled/pnl/trades/win_rate', badStrategies.length === 0, `${badStrategies.length} malformed entr(ies)`);

  const target = strategies.find((strategy) => strategy.enabled) ?? strategies[0];
  if (target) {
    await request(`/api/strategies/${target.id}/toggle`, { method: 'POST', body: JSON.stringify({ enabled: false }) });
    const afterDisable = (await request('/api/strategies')).strategies.find((item) => item.id === target.id);
    check(`POST /api/strategies/${target.id}/toggle {enabled:false} disables it`, afterDisable?.enabled === false, `enabled=${afterDisable?.enabled}`);

    await request(`/api/strategies/${target.id}/toggle`, { method: 'POST', body: JSON.stringify({ enabled: true }) });
    const afterEnable = (await request('/api/strategies')).strategies.find((item) => item.id === target.id);
    check(`POST /api/strategies/${target.id}/toggle {enabled:true} re-enables it`, afterEnable?.enabled === true, `enabled=${afterEnable?.enabled}`);
  } else {
    check('a strategy was available to toggle', false, 'strategy list was empty');
  }

  // --- Agents tab: swarm status, swarm toggle, agent toggle, decisions ------
  section('Agents tab — swarm status, toggles and decisions');
  const swarm = await request('/api/agents/status');
  check('GET /api/agents/status returns `enabled`', typeof swarm?.enabled === 'boolean', `got ${JSON.stringify(swarm?.enabled)}`);
  check('swarm exposes `coordination_mode`', isNonEmptyString(swarm?.coordination_mode), `got ${JSON.stringify(swarm?.coordination_mode)}`);

  const agentEntries = Object.entries(swarm?.agents ?? {});
  check('swarm.agents is a keyed record of agents', agentEntries.length > 0, `${agentEntries.length} agent(s)`);
  const badAgents = agentEntries.filter(
    ([, agent]) =>
      typeof agent?.enabled !== 'boolean' ||
      !isFiniteNumber(agent?.performance?.total_decisions) ||
      !isFiniteNumber(agent?.performance?.successful_decisions) ||
      !isFiniteNumber(agent?.performance?.failed_decisions) ||
      !isFiniteNumber(agent?.performance?.accuracy),
  );
  check('every agent has enabled + full performance block', badAgents.length === 0, `malformed: ${badAgents.map(([key]) => key).join(', ')}`);

  await request('/api/agents/disable', { method: 'POST' });
  check('POST /api/agents/disable pauses the swarm', (await request('/api/agents/status')).enabled === false);
  await request('/api/agents/enable', { method: 'POST' });
  check('POST /api/agents/enable resumes the swarm', (await request('/api/agents/status')).enabled === true);

  const [agentKey, agentBefore] = agentEntries[0] ?? [];
  if (agentKey) {
    await request(`/api/agents/${agentKey}/toggle`, { method: 'POST' });
    const flipped = (await request('/api/agents/status')).agents?.[agentKey];
    check(`POST /api/agents/${agentKey}/toggle flips that agent`, flipped?.enabled === !agentBefore.enabled, `${agentBefore.enabled} -> ${flipped?.enabled}`);

    await request(`/api/agents/${agentKey}/toggle`, { method: 'POST' });
    const restored = (await request('/api/agents/status')).agents?.[agentKey];
    check(`POST /api/agents/${agentKey}/toggle restores it`, restored?.enabled === agentBefore.enabled, `enabled=${restored?.enabled}`);
  } else {
    check('an agent was available to toggle', false, 'agent record was empty');
  }

  const decisionsResponse = await request('/api/agents/decisions?limit=5');
  const decisions = decisionsResponse?.decisions ?? [];
  check('GET /api/agents/decisions?limit=5 returns `{ decisions: [...] }`', Array.isArray(decisionsResponse?.decisions), `got ${JSON.stringify(decisionsResponse)?.slice(0, 120)}`);
  check('decisions honours the limit', decisions.length <= 5, `got ${decisions.length}`);
  const badDecisions = decisions.filter(
    (decision) =>
      !isNonEmptyString(decision?.agent) ||
      !isNonEmptyString(decision?.action) ||
      !isFiniteNumber(decision?.confidence) ||
      !isNonEmptyString(decision?.reason) ||
      !isNonEmptyString(decision?.timestamp),
  );
  check('every decision has agent/action/confidence/reason/timestamp', badDecisions.length === 0, `${badDecisions.length} malformed entr(ies)`);

  // --- Trades tab: recent-trade polling -------------------------------------
  section('Trades tab — recent-trade polling');
  const tradesResponse = await request('/api/trades/recent?limit=20');
  const trades = tradesResponse?.trades ?? [];
  check('GET /api/trades/recent?limit=20 returns `{ trades: [...] }`', Array.isArray(tradesResponse?.trades), `got ${JSON.stringify(tradesResponse)?.slice(0, 120)}`);
  check('trades honours the limit', trades.length <= 20, `got ${trades.length}`);

  const badTrades = trades.filter(
    (trade) =>
      !isNonEmptyString(trade?.id) ||
      !isNonEmptyString(trade?.symbol) ||
      !isNonEmptyString(trade?.side) ||
      !isFiniteNumber(trade?.quantity) ||
      !isFiniteNumber(trade?.price) ||
      !isNonEmptyString(trade?.timestamp) ||
      !isNonEmptyString(trade?.strategy),
  );
  check('every trade has id/symbol/side/quantity/price/timestamp/strategy', badTrades.length === 0, `${badTrades.length} malformed entr(ies)`);
  check('trade ids are unique (mobile keys rows by id)', new Set(trades.map((trade) => trade.id)).size === trades.length);
  check('trade timestamps parse as dates (mobile calls toLocaleTimeString)', trades.every((trade) => !Number.isNaN(new Date(trade.timestamp).getTime())));
  check('trade.side is buy/sell (mobile branches on it)', trades.every((trade) => ['buy', 'sell'].includes(String(trade.side).toLowerCase())));

  // --- Restore the staging fixtures ----------------------------------------
  await request('/reset', { method: 'POST' });

  console.log(`\n${checks - failures}/${checks} checks passed.`);
  if (failures > 0) {
    console.error(`${failures} contract check(s) FAILED.`);
    process.exit(1);
  }
  console.log('Mobile trading API contract is intact.');
}

main().catch((error) => {
  console.error(`\nContract check could not complete: ${error instanceof Error ? error.message : String(error)}`);
  console.error('Is the API Server workflow running? Expected staging mirror at ' + BASE_URL);
  process.exit(1);
});
