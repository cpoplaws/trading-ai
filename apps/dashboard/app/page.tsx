'use client'

import { useEffect, useState } from 'react'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { Badge } from '@/components/ui/badge'
import { Button } from '@/components/ui/button'
import { ArrowUpRight, ArrowDownRight, TrendingUp, DollarSign, Activity, Zap } from 'lucide-react'
import { PortfolioChart } from '@/components/dashboard/PortfolioChart'
import { StrategyGrid } from '@/components/dashboard/StrategyGrid'
import { RecentTrades } from '@/components/dashboard/RecentTrades'
import { BaseIntegration } from '@/components/base/BaseIntegration'
import { usePortfolio } from '@/hooks/usePortfolio'
import { useWebSocket } from '@/hooks/useWebSocket'

export default function Dashboard() {
  const { portfolio, loading } = usePortfolio()
  const { connected, data: liveData } = useWebSocket()

  const [stats, setStats] = useState({
    totalValue: 0,
    dailyChange: 0,
    dailyChangePercent: 0,
    sharpeRatio: 0,
    winRate: 0,
  })

  useEffect(() => {
    if (portfolio) {
      setStats({
        totalValue: portfolio.total_value,
        dailyChange: portfolio.daily_pnl,
        dailyChangePercent: portfolio.daily_pnl_percent,
        sharpeRatio: portfolio.sharpe_ratio,
        winRate: portfolio.win_rate,
      })
    }
  }, [portfolio])

  // Update with live data from WebSocket
  useEffect(() => {
    if (liveData?.portfolio) {
      setStats(prev => ({
        ...prev,
        totalValue: liveData.portfolio.value,
        dailyChange: liveData.portfolio.daily_change,
      }))
    }
  }, [liveData])

  return (
    <div className="min-h-screen bg-gradient-to-br from-background via-background to-secondary/20">
      {/* Header */}
      <header className="border-b border-border/40 backdrop-blur-xl sticky top-0 z-50">
        <div className="container mx-auto px-4 py-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-2">
              <Zap className="w-8 h-8 text-primary" />
              <h1 className="text-2xl font-bold bg-gradient-to-r from-primary to-blue-400 bg-clip-text text-transparent">
                Trading AI
              </h1>
            </div>

            <div className="flex items-center gap-4">
              <Badge variant={connected ? "default" : "secondary"} className="gap-1">
                <div className={`w-2 h-2 rounded-full ${connected ? 'bg-green-500 animate-pulse' : 'bg-gray-500'}`} />
                {connected ? 'Live' : 'Connecting...'}
              </Badge>

              <Button variant="outline" size="sm">
                Paper Trading
              </Button>
            </div>
          </div>
        </div>
      </header>

      {/* Main Content */}
      <main className="container mx-auto px-4 py-8">
        {/* Stats Grid */}
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4 mb-8">
          {/* Total Portfolio */}
          <Card className="relative overflow-hidden group hover:border-primary/50 transition-all">
            <div className="absolute inset-0 bg-gradient-to-br from-primary/10 to-transparent opacity-0 group-hover:opacity-100 transition-opacity" />
            <CardHeader className="pb-2">
              <CardDescription className="flex items-center justify-between">
                <span>Total Portfolio</span>
                <DollarSign className="w-4 h-4 text-muted-foreground" />
              </CardDescription>
            </CardHeader>
            <CardContent>
              <div className="text-3xl font-bold">
                ${loading ? '...' : stats.totalValue.toLocaleString('en-US', { minimumFractionDigits: 2, maximumFractionDigits: 2 })}
              </div>
              <div className={`flex items-center gap-1 text-sm mt-1 ${stats.dailyChange >= 0 ? 'text-green-500' : 'text-red-500'}`}>
                {stats.dailyChange >= 0 ? <ArrowUpRight className="w-4 h-4" /> : <ArrowDownRight className="w-4 h-4" />}
                <span className="font-medium">
                  ${Math.abs(stats.dailyChange).toLocaleString('en-US', { minimumFractionDigits: 2, maximumFractionDigits: 2 })}
                </span>
                <span className="text-muted-foreground">({stats.dailyChangePercent.toFixed(2)}%)</span>
              </div>
            </CardContent>
          </Card>

          {/* Sharpe Ratio */}
          <Card className="relative overflow-hidden group hover:border-primary/50 transition-all">
            <div className="absolute inset-0 bg-gradient-to-br from-blue-500/10 to-transparent opacity-0 group-hover:opacity-100 transition-opacity" />
            <CardHeader className="pb-2">
              <CardDescription className="flex items-center justify-between">
                <span>Sharpe Ratio</span>
                <TrendingUp className="w-4 h-4 text-muted-foreground" />
              </CardDescription>
            </CardHeader>
            <CardContent>
              <div className="text-3xl font-bold">{loading ? '...' : stats.sharpeRatio.toFixed(2)}</div>
              <p className="text-sm text-muted-foreground mt-1">Risk-adjusted returns</p>
            </CardContent>
          </Card>

          {/* Win Rate */}
          <Card className="relative overflow-hidden group hover:border-primary/50 transition-all">
            <div className="absolute inset-0 bg-gradient-to-br from-green-500/10 to-transparent opacity-0 group-hover:opacity-100 transition-opacity" />
            <CardHeader className="pb-2">
              <CardDescription className="flex items-center justify-between">
                <span>Win Rate</span>
                <Activity className="w-4 h-4 text-muted-foreground" />
              </CardDescription>
            </CardHeader>
            <CardContent>
              <div className="text-3xl font-bold">{loading ? '...' : (stats.winRate * 100).toFixed(1)}%</div>
              <p className="text-sm text-muted-foreground mt-1">Successful trades</p>
            </CardContent>
          </Card>

          {/* Active Strategies */}
          <Card className="relative overflow-hidden group hover:border-primary/50 transition-all">
            <div className="absolute inset-0 bg-gradient-to-br from-purple-500/10 to-transparent opacity-0 group-hover:opacity-100 transition-opacity" />
            <CardHeader className="pb-2">
              <CardDescription>Active Strategies</CardDescription>
            </CardHeader>
            <CardContent>
              <div className="text-3xl font-bold">11</div>
              <p className="text-sm text-green-500 mt-1">All systems operational</p>
            </CardContent>
          </Card>
        </div>

        {/* Portfolio Chart */}
        <Card className="mb-8">
          <CardHeader>
            <CardTitle>Portfolio Performance</CardTitle>
            <CardDescription>Last 30 days</CardDescription>
          </CardHeader>
          <CardContent>
            <PortfolioChart />
          </CardContent>
        </Card>

        {/* Strategy Grid */}
        <div className="mb-8">
          <h2 className="text-2xl font-bold mb-4">Trading Strategies</h2>
          <StrategyGrid />
        </div>

        {/* Recent Trades & Base Integration */}
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-8 mb-8">
          <RecentTrades />
          <BaseIntegration />
        </div>
      </main>

      {/* Footer */}
      <footer className="border-t border-border/40 py-6 mt-12">
        <div className="container mx-auto px-4 text-center text-sm text-muted-foreground">
          <p>Trading AI © 2026 • Built with Next.js + Base</p>
        </div>
      </footer>
    </div>
  )
}
