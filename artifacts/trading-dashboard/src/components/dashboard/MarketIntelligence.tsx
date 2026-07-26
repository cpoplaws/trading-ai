import { useState, useEffect } from 'react'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { Badge } from '@/components/ui/badge'

interface Intelligence {
  signal: string
  composite_score: number
  confidence: number
  regime: {
    regime: string
    confidence: number
    momentum: number
    volatility: number
  }
  sentiment: {
    sentiment: string
    score: number
  }
  technical: {
    rsi: number
    macd: number
    bb_position: number
  }
  macro: {
    trend: string
  }
  recommendations: string[]
  alerts: Array<{
    type: string
    message: string
    severity: string
  }>
  timestamp: string
}

const API_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000'

const regimeEmojis: Record<string, string> = {
  bull_trend: '🐂',
  bear_trend: '🐻',
  high_volatility: '⚡',
  low_volatility: '😴',
  sideways: '↔️',
  unknown: '❓',
}

const regimeColors: Record<string, string> = {
  bull_trend: 'text-green-500',
  bear_trend: 'text-red-500',
  high_volatility: 'text-yellow-500',
  low_volatility: 'text-blue-500',
  sideways: 'text-gray-400',
  unknown: 'text-gray-500',
}

export function MarketIntelligence() {
  const [intelligence, setIntelligence] = useState<Intelligence | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)

  const fetchIntelligence = async () => {
    try {
      const response = await fetch(`${API_URL}/api/intelligence`)
      if (!response.ok) throw new Error('Failed to fetch intelligence')
      const data = await response.json()
      setIntelligence(data)
      setError(null)
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to fetch intelligence')
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => {
    fetchIntelligence()
    const interval = setInterval(fetchIntelligence, 30000)
    return () => clearInterval(interval)
  }, [])

  if (loading) return <div className="text-center text-gray-400 py-8">Analyzing market intelligence...</div>
  if (error) return <div className="text-center text-red-500 py-8">Error: {error}</div>
  if (!intelligence) return null

  const getSignalColor = (signal: string) => {
    if (signal.includes('buy')) return 'bg-green-600'
    if (signal.includes('sell')) return 'bg-red-600'
    return 'bg-gray-600'
  }

  const getSentimentEmoji = (sentiment: string) => {
    if (sentiment === 'bullish') return '😃'
    if (sentiment === 'bearish') return '😟'
    return '😐'
  }

  return (
    <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
      {/* Main Intelligence Card */}
      <Card className="bg-gray-900 border-gray-800 lg:col-span-2">
        <CardHeader>
          <CardTitle className="text-white flex items-center justify-between">
            <span>Market Intelligence</span>
            <Badge className={getSignalColor(intelligence.signal)}>
              {intelligence.signal.replace('_', ' ').toUpperCase()}
            </Badge>
          </CardTitle>
        </CardHeader>
        <CardContent className="space-y-4">
          {/* Composite Score */}
          <div>
            <div className="flex justify-between items-center mb-2">
              <span className="text-gray-400">Composite Score</span>
              <span className="text-white font-bold">
                {(intelligence.composite_score * 100).toFixed(0)}%
              </span>
            </div>
            <div className="w-full bg-gray-800 rounded-full h-2">
              <div
                className={`h-2 rounded-full ${intelligence.composite_score > 0 ? 'bg-green-500' : 'bg-red-500'}`}
                style={{ width: `${Math.abs(intelligence.composite_score) * 100}%` }}
              />
            </div>
          </div>

          {/* Regime */}
          <div className="grid grid-cols-2 gap-4">
            <div className="bg-gray-800 p-3 rounded-lg">
              <p className="text-gray-400 text-xs mb-1">Market Regime</p>
              <p className={`font-bold ${regimeColors[intelligence.regime.regime] || 'text-white'}`}>
                {regimeEmojis[intelligence.regime.regime] || '❓'} {intelligence.regime.regime.replace('_', ' ')}
              </p>
            </div>
            <div className="bg-gray-800 p-3 rounded-lg">
              <p className="text-gray-400 text-xs mb-1">Sentiment</p>
              <p className="text-white font-bold">
                {getSentimentEmoji(intelligence.sentiment.sentiment)} {intelligence.sentiment.sentiment}
              </p>
            </div>
          </div>

          {/* Technical Indicators */}
          <div>
            <p className="text-gray-400 text-xs mb-2">Technical Indicators</p>
            <div className="grid grid-cols-3 gap-2">
              <div className="bg-gray-800 p-3 rounded-lg text-center">
                <p className="text-xs text-gray-400">RSI</p>
                <p className="text-white font-bold">{intelligence.technical.rsi.toFixed(1)}</p>
              </div>
              <div className="bg-gray-800 p-3 rounded-lg text-center">
                <p className="text-xs text-gray-400">MACD</p>
                <p className={`font-bold ${intelligence.technical.macd > 0 ? 'text-green-500' : 'text-red-500'}`}>
                  {intelligence.technical.macd.toFixed(3)}
                </p>
              </div>
              <div className="bg-gray-800 p-3 rounded-lg text-center">
                <p className="text-xs text-gray-400">BB Position</p>
                <p className="text-white font-bold">{(intelligence.technical.bb_position * 100).toFixed(0)}%</p>
              </div>
            </div>
          </div>
        </CardContent>
      </Card>

      {/* Recommendations & Alerts */}
      <div className="space-y-6">
        {intelligence.alerts.length > 0 && (
          <Card className="bg-gray-900 border-gray-800">
            <CardHeader>
              <CardTitle className="text-white text-base">Alerts</CardTitle>
            </CardHeader>
            <CardContent className="space-y-2">
              {intelligence.alerts.map((alert, index) => (
                <div
                  key={index}
                  className={`p-2 rounded-lg text-sm ${
                    alert.type === 'warning' ? 'bg-yellow-900/30 text-yellow-200' : 'bg-blue-900/30 text-blue-200'
                  }`}
                >
                  {alert.message}
                </div>
              ))}
            </CardContent>
          </Card>
        )}

        <Card className="bg-gray-900 border-gray-800">
          <CardHeader>
            <CardTitle className="text-white text-base">Recommendations</CardTitle>
          </CardHeader>
          <CardContent className="space-y-2">
            {intelligence.recommendations.map((rec, index) => (
              <div key={index} className="text-sm text-gray-300 p-2 bg-gray-800 rounded">
                {rec}
              </div>
            ))}
          </CardContent>
        </Card>

        <div className="text-xs text-gray-500 text-center">
          Last updated: {new Date(intelligence.timestamp).toLocaleTimeString()}
        </div>
      </div>
    </div>
  )
}
