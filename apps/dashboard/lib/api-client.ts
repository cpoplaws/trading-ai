import axios, { AxiosError, AxiosRequestConfig } from 'axios'

const API_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000'
const API_KEY = process.env.NEXT_PUBLIC_API_KEY

const DEFAULT_TIMEOUT_MS = 15000
const MAX_RETRIES = 2
const RETRY_BASE_DELAY_MS = 300

type RetryableConfig = AxiosRequestConfig & { __retryCount?: number }

export const apiClient = axios.create({
  baseURL: API_URL,
  timeout: DEFAULT_TIMEOUT_MS,
  headers: {
    'Content-Type': 'application/json',
    ...(API_KEY ? { 'X-API-Key': API_KEY } : {}),
  },
})

apiClient.interceptors.response.use(
  (response) => response,
  async (error: AxiosError) => {
    const config = error.config as RetryableConfig | undefined
    if (!config) return Promise.reject(error)

    const status = error.response?.status
    const isRetryable =
      status === undefined || status >= 500 || status === 429 || error.code === 'ECONNABORTED'
    if (!isRetryable) return Promise.reject(error)

    config.__retryCount = (config.__retryCount ?? 0) + 1
    if (config.__retryCount > MAX_RETRIES) return Promise.reject(error)

    const delay = RETRY_BASE_DELAY_MS * 2 ** (config.__retryCount - 1)
    await new Promise((resolve) => setTimeout(resolve, delay))
    return apiClient(config)
  },
)

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
