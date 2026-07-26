const API_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8080'

export interface Portfolio {
  total_value: number
  cash: number
  buying_power: number
  daily_pnl: number
  daily_pnl_percent: number
  positions_count: number
  sharpe_ratio: number
  win_rate: number
  timestamp: string
  demo_mode: boolean
}

export interface Strategy {
  name: string
  pnl: number
  win_rate: number
  sharpe_ratio: number
  enabled: boolean
}

export interface Trade {
  symbol: string
  side: 'buy' | 'sell'
  qty: number
  price: number
  timestamp: string
  strategy: string
}

export const api = {
  async getPortfolio(): Promise<Portfolio> {
    const response = await fetch(`${API_URL}/api/portfolio`)
    if (!response.ok) {
      throw new Error('Failed to fetch portfolio')
    }
    return response.json()
  },

  async getStrategies(): Promise<Strategy[]> {
    const response = await fetch(`${API_URL}/api/strategies`)
    if (!response.ok) {
      throw new Error('Failed to fetch strategies')
    }
    return response.json()
  },

  async getRecentTrades(): Promise<Trade[]> {
    const response = await fetch(`${API_URL}/api/trades/recent`)
    if (!response.ok) {
      throw new Error('Failed to fetch trades')
    }
    return response.json()
  },

  async toggleStrategy(name: string, enabled: boolean): Promise<void> {
    const response = await fetch(`${API_URL}/api/strategies/${name}`, {
      method: 'PATCH',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ enabled }),
    })
    if (!response.ok) {
      throw new Error('Failed to toggle strategy')
    }
  },
}
