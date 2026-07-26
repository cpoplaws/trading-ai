'use client'

import { useState, useEffect } from 'react'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { Badge } from '@/components/ui/badge'

interface Trade {
  id: string
  symbol: string
  side: string
  quantity: number
  price: number
  timestamp: string
  strategy: string
  pnl?: number
}

const API_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000'

export function RecentTrades() {
  const [trades, setTrades] = useState<Trade[]>([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)

  const fetchTrades = async () => {
    try {
      const response = await fetch(`${API_URL}/api/trades/recent?limit=20`)
      if (!response.ok) throw new Error('Failed to fetch trades')
      const data = await response.json()
      setTrades(data.trades || [])
      setError(null)
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to fetch trades')
      console.error('Error fetching trades:', err)
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => {
    fetchTrades()

    // Refresh every 5 seconds
    const interval = setInterval(fetchTrades, 5000)
    return () => clearInterval(interval)
  }, [])

  if (loading) {
    return (
      <Card className="bg-gray-900 border-gray-800">
        <CardHeader>
          <CardTitle className="text-white">Recent Trades</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="text-center text-gray-400 py-8">
            Loading trades...
          </div>
        </CardContent>
      </Card>
    )
  }

  if (error) {
    return (
      <Card className="bg-gray-900 border-gray-800">
        <CardHeader>
          <CardTitle className="text-white">Recent Trades</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="text-center text-red-500 py-8">
            Error: {error}
          </div>
        </CardContent>
      </Card>
    )
  }

  return (
    <Card className="bg-gray-900 border-gray-800">
      <CardHeader>
        <CardTitle className="text-white">Recent Trades</CardTitle>
      </CardHeader>
      <CardContent>
        {trades.length === 0 ? (
          <div className="text-center text-gray-400 py-8">
            No trades yet. Enable strategies to start trading.
          </div>
        ) : (
          <div className="space-y-2">
            {trades.map((trade) => (
              <div
                key={trade.id}
                className="flex items-center justify-between p-3 bg-gray-800 rounded-lg hover:bg-gray-750 transition-colors"
              >
                <div className="flex items-center gap-3">
                  <Badge
                    variant={trade.side === 'buy' ? 'default' : 'destructive'}
                    className={trade.side === 'buy' ? 'bg-green-600' : 'bg-red-600'}
                  >
                    {trade.side.toUpperCase()}
                  </Badge>
                  <div>
                    <div className="flex items-center gap-2">
                      <span className="text-white font-semibold">{trade.symbol}</span>
                      <span className="text-gray-400 text-sm">×{trade.quantity}</span>
                    </div>
                    <div className="text-xs text-gray-500">
                      {trade.strategy} • {new Date(trade.timestamp).toLocaleString()}
                    </div>
                  </div>
                </div>
                <div className="text-right">
                  <div className="text-white font-semibold">
                    ${trade.price.toFixed(2)}
                  </div>
                  {trade.pnl !== undefined && (
                    <div className={`text-sm ${trade.pnl >= 0 ? 'text-green-500' : 'text-red-500'}`}>
                      {trade.pnl >= 0 ? '+' : ''}${trade.pnl.toFixed(2)}
                    </div>
                  )}
                </div>
              </div>
            ))}
          </div>
        )}
      </CardContent>
    </Card>
  )
}
