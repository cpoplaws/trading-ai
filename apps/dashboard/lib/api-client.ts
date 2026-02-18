import axios from 'axios'

const API_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000'

export const apiClient = axios.create({
  baseURL: API_URL,
  headers: {
    'Content-Type': 'application/json',
  },
})

// Portfolio
export const getPortfolio = async () => {
  const { data } = await apiClient.get('/api/portfolio')
  return data
}

export const getPortfolioHistory = async (days: number = 30) => {
  const { data } = await apiClient.get(`/api/portfolio/history?days=${days}`)
  return data
}

// Strategies
export const getStrategies = async () => {
  const { data } = await apiClient.get('/api/strategies')
  return data
}

export const toggleStrategy = async (strategyId: string, enabled: boolean) => {
  const { data } = await apiClient.post(`/api/strategies/${strategyId}/toggle`, { enabled })
  return data
}

// Trades
export const getRecentTrades = async (limit: number = 20) => {
  const { data } = await apiClient.get(`/api/trades/recent?limit=${limit}`)
  return data
}

// Base blockchain
export const getBaseBalance = async (address: string) => {
  const { data } = await apiClient.get(`/api/base/balance/${address}`)
  return data
}

export const executeBaseTrade = async (trade: {
  tokenIn: string
  tokenOut: string
  amountIn: number
  slippage: number
}) => {
  const { data } = await apiClient.post('/api/base/trade', trade)
  return data
}
