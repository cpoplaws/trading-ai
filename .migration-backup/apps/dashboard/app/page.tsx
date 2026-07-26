'use client'

import { useEffect, useState } from 'react'
import { StrategyGrid } from '@/components/dashboard/StrategyGrid'
import { AgentSwarm } from '@/components/dashboard/AgentSwarm'
import { MarketIntelligence } from '@/components/dashboard/MarketIntelligence'
import { RecentTrades } from '@/components/dashboard/RecentTrades'
import { getPortfolio } from '@/lib/api-client'

interface Portfolio {
  total_value: number
  cash: number
  buying_power: number
  daily_pnl: number
  daily_pnl_percent: number
  positions_count: number
  sharpe_ratio: number
  win_rate: number
  demo_mode?: boolean
}

export default function Dashboard() {
  const [portfolio, setPortfolio] = useState<Portfolio | null>(null)
  const [activeStrategiesCount, setActiveStrategiesCount] = useState(0)

  useEffect(() => {
    // Fetch portfolio data
    const fetchPortfolio = async () => {
      try {
        const data = await getPortfolio()
        setPortfolio(data)
      } catch (error) {
        console.error('Error fetching portfolio:', error)
      }
    }

    fetchPortfolio()
    // Refresh every 10 seconds
    const interval = setInterval(fetchPortfolio, 10000)
    return () => clearInterval(interval)
  }, [])

  return (
    <div className="min-h-screen bg-gray-950 text-white p-8">
      {/* Header */}
      <header className="mb-8">
        <h1 className="text-4xl font-bold bg-gradient-to-r from-blue-500 to-purple-500 bg-clip-text text-transparent">
          Crypto AI Trading Dashboard
        </h1>
        <p className="text-gray-400 mt-2">
          Multi-Chain Trading • Base • Solana • L2s {portfolio?.demo_mode && '• (Demo Mode)'}
        </p>
      </header>

      {/* Stats Grid */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-4 mb-8">
        <div className="bg-gray-900 p-6 rounded-lg border border-gray-800">
          <p className="text-gray-400 text-sm">Total Portfolio</p>
          <p className="text-3xl font-bold mt-2">
            ${portfolio?.total_value.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 }) || '0.00'}
          </p>
          <p className={`text-sm mt-1 ${(portfolio?.daily_pnl || 0) >= 0 ? 'text-green-500' : 'text-red-500'}`}>
            {(portfolio?.daily_pnl || 0) >= 0 ? '+' : ''}${portfolio?.daily_pnl.toFixed(2) || '0.00'}
            ({(portfolio?.daily_pnl_percent || 0).toFixed(2)}%)
          </p>
        </div>

        <div className="bg-gray-900 p-6 rounded-lg border border-gray-800">
          <p className="text-gray-400 text-sm">Sharpe Ratio</p>
          <p className="text-3xl font-bold mt-2">{portfolio?.sharpe_ratio.toFixed(2) || '0.00'}</p>
          <p className="text-gray-400 text-sm mt-1">Risk-adjusted returns</p>
        </div>

        <div className="bg-gray-900 p-6 rounded-lg border border-gray-800">
          <p className="text-gray-400 text-sm">Win Rate</p>
          <p className="text-3xl font-bold mt-2">
            {((portfolio?.win_rate || 0) * 100).toFixed(1)}%
          </p>
          <p className="text-gray-400 text-sm mt-1">Successful trades</p>
        </div>

        <div className="bg-gray-900 p-6 rounded-lg border border-gray-800">
          <p className="text-gray-400 text-sm">Cash Available</p>
          <p className="text-3xl font-bold mt-2">
            ${portfolio?.cash.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 }) || '0.00'}
          </p>
          <p className="text-gray-400 text-sm mt-1">Buying power: ${portfolio?.buying_power.toFixed(0) || '0'}</p>
        </div>
      </div>

      {/* Market Intelligence */}
      <div className="mb-8">
        <h2 className="text-2xl font-bold mb-4">🧠 Market Intelligence</h2>
        <MarketIntelligence />
      </div>

      {/* Strategies */}
      <div className="mb-8">
        <h2 className="text-2xl font-bold mb-4">Trading Strategies</h2>
        <StrategyGrid />
      </div>

      {/* Agent Swarm */}
      <div className="mb-8">
        <h2 className="text-2xl font-bold mb-4">AI Agent Swarm</h2>
        <AgentSwarm />
      </div>

      {/* Recent Trades */}
      <div className="mb-8">
        <h2 className="text-2xl font-bold mb-4">Recent Trades</h2>
        <RecentTrades />
      </div>

      {/* Status */}
      <div className="bg-gray-900 p-6 rounded-lg border border-gray-800">
        <h2 className="text-xl font-bold mb-4">System Status</h2>
        <div className="space-y-2">
          <div className="flex items-center gap-2">
            <div className="w-2 h-2 bg-green-500 rounded-full"></div>
            <span className="text-gray-300">Dashboard Online</span>
          </div>
          <div className="flex items-center gap-2">
            <div className={`w-2 h-2 ${portfolio?.demo_mode ? 'bg-yellow-500' : 'bg-green-500'} rounded-full`}></div>
            <span className="text-gray-300">
              Backend: {portfolio?.demo_mode ? 'Demo Mode' : 'Connected to Alpaca'}
            </span>
          </div>
          <div className="flex items-center gap-2">
            <div className="w-2 h-2 bg-green-500 rounded-full"></div>
            <span className="text-gray-300">API: Connected</span>
          </div>
        </div>
      </div>

      {/* Footer */}
      <footer className="mt-8 text-center text-gray-500 text-sm">
        <p>Trading AI © 2026 • Built with Next.js</p>
        <p className="mt-1">🚀 Frontend on Vercel • Backend on Railway</p>
      </footer>
    </div>
  )
}
