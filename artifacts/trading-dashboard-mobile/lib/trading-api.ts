import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query';

const API_URL = process.env.EXPO_PUBLIC_API_URL || process.env.VITE_API_URL || '';

export interface Portfolio {
  total_value: number;
  cash: number;
  buying_power: number;
  daily_pnl: number;
  daily_pnl_percent: number;
  positions_count: number;
  sharpe_ratio: number;
  win_rate: number;
  demo_mode?: boolean;
}

export interface Strategy {
  id: string;
  name: string;
  enabled: boolean;
  pnl: number;
  trades: number;
  win_rate: number;
}

export interface Agent {
  enabled: boolean;
  performance: {
    total_decisions: number;
    successful_decisions: number;
    failed_decisions: number;
    accuracy: number;
  };
  recent_decisions: number;
}

export interface SwarmStatus {
  enabled: boolean;
  coordination_mode: string;
  agents: Record<string, Agent>;
}

export interface Decision {
  agent: string;
  action: string;
  confidence: number;
  reason: string;
  timestamp: string;
}

export interface Trade {
  id: string;
  symbol: string;
  side: string;
  quantity: number;
  price: number;
  timestamp: string;
  strategy: string;
  pnl?: number;
}

export interface PortfolioHistoryPoint {
  timestamp?: string;
  date?: string;
  value?: number;
  total_value?: number;
}

async function request<T>(path: string, options?: RequestInit): Promise<T> {
  if (!API_URL) {
    throw new Error('API is not configured. Set EXPO_PUBLIC_API_URL to the same backend URL used by the web dashboard.');
  }

  const response = await fetch(`${API_URL.replace(/\/$/, '')}${path}`, {
    ...options,
    headers: { 'Content-Type': 'application/json', ...(options?.headers ?? {}) },
  });

  if (!response.ok) {
    throw new Error(`API request failed (${response.status})`);
  }

  return response.json() as Promise<T>;
}

export const getPortfolio = () => request<Portfolio>('/api/portfolio');

export const getPortfolioHistory = () =>
  request<{ history?: PortfolioHistoryPoint[] } | PortfolioHistoryPoint[]>('/api/portfolio/history?days=7');

export const getStrategies = async () => {
  const response = await request<{ strategies?: Strategy[] }>('/api/strategies');
  return response.strategies ?? [];
};

export const setStrategyEnabled = (strategyId: string, enabled: boolean) =>
  request(`/api/strategies/${strategyId}/toggle`, {
    method: 'POST',
    body: JSON.stringify({ enabled }),
  });

export const getSwarmStatus = () => request<SwarmStatus>('/api/agents/status');

export const getAgentDecisions = async () => {
  const response = await request<{ decisions?: Decision[] }>('/api/agents/decisions?limit=5');
  return response.decisions ?? [];
};

export const setSwarmEnabled = (enabled: boolean) =>
  request(`/api/agents/${enabled ? 'enable' : 'disable'}`, { method: 'POST' });

export const toggleAgent = (agentName: string) =>
  request(`/api/agents/${agentName}/toggle`, { method: 'POST' });

export const getRecentTrades = async () => {
  const response = await request<{ trades?: Trade[] }>('/api/trades/recent?limit=20');
  return response.trades ?? [];
};

export function usePortfolioQuery() {
  return useQuery({ queryKey: ['portfolio'], queryFn: getPortfolio, refetchInterval: 10000 });
}

export function usePortfolioHistoryQuery() {
  return useQuery({ queryKey: ['portfolio-history'], queryFn: getPortfolioHistory, refetchInterval: 30000 });
}

export function useStrategiesQuery() {
  return useQuery({ queryKey: ['strategies'], queryFn: getStrategies, refetchInterval: 10000 });
}

export function useStrategyToggleMutation() {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: ({ strategyId, enabled }: { strategyId: string; enabled: boolean }) =>
      setStrategyEnabled(strategyId, enabled),
    onSuccess: () => queryClient.invalidateQueries({ queryKey: ['strategies'] }),
  });
}

export function useSwarmQuery() {
  return useQuery({ queryKey: ['swarm'], queryFn: getSwarmStatus, refetchInterval: 10000 });
}

export function useDecisionsQuery() {
  return useQuery({ queryKey: ['decisions'], queryFn: getAgentDecisions, refetchInterval: 5000 });
}

export function useSwarmToggleMutation() {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: setSwarmEnabled,
    onSuccess: () => queryClient.invalidateQueries({ queryKey: ['swarm'] }),
  });
}

export function useAgentToggleMutation() {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: toggleAgent,
    onSuccess: () => queryClient.invalidateQueries({ queryKey: ['swarm'] }),
  });
}

export function useTradesQuery() {
  return useQuery({ queryKey: ['trades'], queryFn: getRecentTrades, refetchInterval: 5000 });
}