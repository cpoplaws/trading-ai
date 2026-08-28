import { Router, type IRouter } from "express";

/**
 * Safe staging mirror of the external Python trading API.
 *
 * The Quantlytics mobile app talks directly to the Python trading engine via
 * `EXPO_PUBLIC_API_URL`. Pointing automated checks at that engine would place
 * real orders, so this router reproduces the exact endpoints and response
 * shapes the mobile client consumes, backed by deterministic in-memory
 * fixtures. Nothing here touches live trading, brokerage credentials, or the
 * database.
 *
 * Mounted under `/api/staging-trading`, so the mobile client can be pointed at
 * `<origin>/api/staging-trading` and keep using its real request paths
 * (`/api/portfolio`, `/api/strategies/:id/toggle`, ...) unchanged. That is what
 * makes the checks able to catch endpoint or response-shape regressions.
 */

interface Portfolio {
  total_value: number;
  cash: number;
  buying_power: number;
  daily_pnl: number;
  daily_pnl_percent: number;
  positions_count: number;
  sharpe_ratio: number;
  win_rate: number;
  demo_mode: boolean;
}

interface Strategy {
  id: string;
  name: string;
  enabled: boolean;
  pnl: number;
  trades: number;
  win_rate: number;
}

interface Agent {
  enabled: boolean;
  performance: {
    total_decisions: number;
    successful_decisions: number;
    failed_decisions: number;
    accuracy: number;
  };
  recent_decisions: number;
}

interface SwarmStatus {
  enabled: boolean;
  coordination_mode: string;
  agents: Record<string, Agent>;
}

interface Decision {
  agent: string;
  action: string;
  confidence: number;
  reason: string;
  timestamp: string;
}

interface Trade {
  id: string;
  symbol: string;
  side: string;
  quantity: number;
  price: number;
  timestamp: string;
  strategy: string;
  pnl?: number;
}

interface StagingState {
  portfolio: Portfolio;
  strategies: Strategy[];
  swarm: SwarmStatus;
  decisions: Decision[];
  trades: Trade[];
}

const TRADE_SEEDS: Array<{
  symbol: string;
  side: string;
  quantity: number;
  price: number;
  strategy: string;
  pnl: number;
  minutesAgo: number;
}> = [
  { symbol: "ETH/USDC", side: "buy", quantity: 4.25, price: 3412.88, strategy: "Momentum", pnl: 184.32, minutesAgo: 3 },
  { symbol: "SOL/USDC", side: "sell", quantity: 120, price: 178.4, strategy: "Mean reversion", pnl: -62.15, minutesAgo: 11 },
  { symbol: "WBTC/USDC", side: "buy", quantity: 0.32, price: 67420.5, strategy: "Trend following", pnl: 512.7, minutesAgo: 24 },
  { symbol: "ARB/USDC", side: "sell", quantity: 1850, price: 1.14, strategy: "Arbitrage", pnl: 97.44, minutesAgo: 38 },
  { symbol: "OP/USDC", side: "buy", quantity: 940, price: 2.36, strategy: "Momentum", pnl: -18.9, minutesAgo: 52 },
  { symbol: "BASE/USDC", side: "buy", quantity: 3100, price: 0.84, strategy: "Market making", pnl: 43.06, minutesAgo: 67 },
  { symbol: "ETH/USDC", side: "sell", quantity: 2.1, price: 3398.02, strategy: "Mean reversion", pnl: 128.55, minutesAgo: 83 },
  { symbol: "SOL/USDC", side: "buy", quantity: 65, price: 174.9, strategy: "Trend following", pnl: 226.71, minutesAgo: 96 },
  { symbol: "JUP/USDC", side: "sell", quantity: 2400, price: 0.92, strategy: "Arbitrage", pnl: -35.28, minutesAgo: 114 },
  { symbol: "WBTC/USDC", side: "buy", quantity: 0.14, price: 67110.25, strategy: "Momentum", pnl: 76.4, minutesAgo: 131 },
  { symbol: "ARB/USDC", side: "buy", quantity: 1200, price: 1.09, strategy: "Market making", pnl: 21.87, minutesAgo: 148 },
  { symbol: "ETH/USDC", side: "sell", quantity: 1.75, price: 3380.11, strategy: "Trend following", pnl: -91.33, minutesAgo: 166 },
];

function buildInitialState(): StagingState {
  const now = Date.now();

  return {
    portfolio: {
      total_value: 128450.72,
      cash: 41230.18,
      buying_power: 82460.36,
      daily_pnl: 2384.55,
      daily_pnl_percent: 1.89,
      positions_count: 7,
      sharpe_ratio: 1.84,
      win_rate: 0.643,
      // Always true: the staging backend never represents a live brokerage account.
      demo_mode: true,
    },
    strategies: [
      { id: "momentum", name: "Momentum breakout", enabled: true, pnl: 4820.44, trades: 186, win_rate: 0.671 },
      { id: "mean-reversion", name: "Mean reversion", enabled: true, pnl: 1735.9, trades: 142, win_rate: 0.598 },
      { id: "arbitrage", name: "Cross-chain arbitrage", enabled: false, pnl: -412.36, trades: 74, win_rate: 0.446 },
      { id: "market-making", name: "Market making", enabled: false, pnl: 968.12, trades: 311, win_rate: 0.552 },
    ],
    swarm: {
      enabled: true,
      coordination_mode: "Consensus voting across 4 specialists",
      agents: {
        execution: {
          enabled: true,
          performance: { total_decisions: 1284, successful_decisions: 902, failed_decisions: 382, accuracy: 0.702 },
          recent_decisions: 12,
        },
        risk: {
          enabled: true,
          performance: { total_decisions: 964, successful_decisions: 728, failed_decisions: 236, accuracy: 0.755 },
          recent_decisions: 8,
        },
        arbitrage: {
          enabled: false,
          performance: { total_decisions: 512, successful_decisions: 289, failed_decisions: 223, accuracy: 0.564 },
          recent_decisions: 0,
        },
        market_making: {
          enabled: true,
          performance: { total_decisions: 2140, successful_decisions: 1305, failed_decisions: 835, accuracy: 0.61 },
          recent_decisions: 21,
        },
      },
    },
    decisions: [
      { agent: "execution", action: "BUY ETH/USDC", confidence: 0.82, reason: "Breakout confirmed above 4h resistance with rising volume", timestamp: new Date(now - 60_000).toISOString() },
      { agent: "risk", action: "REDUCE SOL exposure", confidence: 0.74, reason: "Position concentration exceeded 18% of portfolio", timestamp: new Date(now - 240_000).toISOString() },
      { agent: "market_making", action: "WIDEN spread", confidence: 0.66, reason: "Realized volatility up 2.3x over the trailing hour", timestamp: new Date(now - 480_000).toISOString() },
      { agent: "execution", action: "SELL ARB/USDC", confidence: 0.71, reason: "Momentum decay and negative funding drift", timestamp: new Date(now - 900_000).toISOString() },
      { agent: "risk", action: "HOLD", confidence: 0.58, reason: "Drawdown within tolerance, no rebalance required", timestamp: new Date(now - 1_500_000).toISOString() },
    ],
    trades: TRADE_SEEDS.map((seed, index) => ({
      id: `staging-trade-${index + 1}`,
      symbol: seed.symbol,
      side: seed.side,
      quantity: seed.quantity,
      price: seed.price,
      timestamp: new Date(now - seed.minutesAgo * 60_000).toISOString(),
      strategy: seed.strategy,
      pnl: seed.pnl,
    })),
  };
}

let state: StagingState = buildInitialState();

function portfolioHistory() {
  const now = Date.now();
  const values = [121_480.11, 122_905.4, 120_744.86, 124_318.72, 125_902.55, 126_071.9, state.portfolio.total_value];

  return values.map((value, index) => ({
    timestamp: new Date(now - (values.length - 1 - index) * 24 * 60 * 60 * 1000).toISOString(),
    value: Number(value.toFixed(2)),
    total_value: Number(value.toFixed(2)),
  }));
}

const router: IRouter = Router();

// --- Control surface for automated checks (not part of the Python API) -----

router.post("/reset", (_req, res) => {
  state = buildInitialState();
  res.json({ status: "reset", demo_mode: true });
});

router.get("/health", (_req, res) => {
  res.json({ status: "ok", staging: true, demo_mode: true });
});

// --- Mirror of the Python trading API ------------------------------------
// Mounted at `/api` beneath the staging root so client paths stay identical.

const api: IRouter = Router();

api.get("/portfolio", (_req, res) => {
  res.json(state.portfolio);
});

api.get("/portfolio/history", (_req, res) => {
  res.json({ history: portfolioHistory() });
});

api.get("/strategies", (_req, res) => {
  res.json({ strategies: state.strategies });
});

api.post("/strategies/:strategyId/toggle", (req, res) => {
  const strategy = state.strategies.find((item) => item.id === req.params["strategyId"]);

  if (!strategy) {
    res.status(404).json({ detail: `Unknown strategy: ${req.params["strategyId"]}` });
    return;
  }

  const requested = (req.body as { enabled?: unknown } | undefined)?.enabled;
  strategy.enabled = typeof requested === "boolean" ? requested : !strategy.enabled;

  res.json({ status: "ok", strategy });
});

api.get("/agents/status", (_req, res) => {
  res.json(state.swarm);
});

api.get("/agents/decisions", (req, res) => {
  const rawLimit = Number(req.query["limit"]);
  const limit = Number.isFinite(rawLimit) && rawLimit > 0 ? Math.floor(rawLimit) : state.decisions.length;

  res.json({ decisions: state.decisions.slice(0, limit) });
});

api.post("/agents/enable", (_req, res) => {
  state.swarm.enabled = true;
  res.json({ status: "ok", enabled: true });
});

api.post("/agents/disable", (_req, res) => {
  state.swarm.enabled = false;
  res.json({ status: "ok", enabled: false });
});

api.post("/agents/:agentName/toggle", (req, res) => {
  const agentName = req.params["agentName"] ?? "";
  const agent = state.swarm.agents[agentName];

  if (!agent) {
    res.status(404).json({ detail: `Unknown agent: ${agentName}` });
    return;
  }

  agent.enabled = !agent.enabled;
  res.json({ status: "ok", agent: agentName, enabled: agent.enabled });
});

api.get("/trades/recent", (req, res) => {
  const rawLimit = Number(req.query["limit"]);
  const limit = Number.isFinite(rawLimit) && rawLimit > 0 ? Math.floor(rawLimit) : state.trades.length;

  res.json({ trades: state.trades.slice(0, limit) });
});

router.use("/api", api);

export default router;
