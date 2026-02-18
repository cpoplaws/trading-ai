'use client'

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

const API_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000'

const regimeEmojis: Record<string, string> = {
  bull_trend: '🐂',
  bear_trend: '🐻',
  high_volatility: '⚡',
  low_volatility: '😴',
  sideways: '↔️',
  unknown: '❓'
}

const regimeColors: Record<string, string> = {
  bull_trend: 'text-green-500',
  bear_trend: 'text-red-500',
  high_volatility: 'text-yellow-500',
  low_volatility: 'text-blue-500',
  sideways: 'text-gray-400',
  unknown: 'text-gray-500'
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
      console.error('Error fetching intelligence:', err)
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => {
    fetchIntelligence()

    // Refresh every 30 seconds
    const interval = setInterval(fetchIntelligence, 30000)
    return () => clearInterval(interval)
  }, [])

  if (loading) {
    return (
      <div className="text-center text-gray-400 py-8">
        Analyzing market intelligence...
      </div>
    )
  }

  if (error) {
    return (
      <div className="text-center text-red-500 py-8">
        Error: {error}
      </div>
    )
  }

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
                className={`h-2 rounded-full ${
                  intelligence.composite_score > 0 ? 'bg-green-500' : 'bg-red-500'
                }`}
                style={{
                  width: `${Math.abs(intelligence.composite_score) * 100}%`
                }}
              />
            </div>
          </div>

          {/* Confidence */}
          <div>
            <div className="flex justify-between items-center mb-2">
              <span className="text-gray-400">Confidence</span>
              <span className="text-white font-bold">
                {(intelligence.confidence * 100).toFixed(0)}%
              </span>
            </div>
            <div className="w-full bg-gray-800 rounded-full h-2">
              <div
                className="h-2 rounded-full bg-blue-500"
                style={{ width: `${intelligence.confidence * 100}%` }}
              />
            </div>
          </div>

          {/* Component Scores */}
          <div className="grid grid-cols-2 gap-4 pt-4">
            <div className="bg-gray-800 p-3 rounded-lg">
              <div className="flex items-center gap-2 mb-2">
                <span className="text-xl">{regimeEmojis[intelligence.regime.regime]}</span>
                <span className="text-gray-400 text-sm">Market Regime</span>
              </div>
              <div className={`font-semibold ${regimeColors[intelligence.regime.regime]}`}>
                {intelligence.regime.regime.replace('_', ' ').toUpperCase()}
              </div>
              <div className="text-xs text-gray-400 mt-1">
                {(intelligence.regime.confidence * 100).toFixed(0)}% confidence
              </div>
            </div>

            <div className="bg-gray-800 p-3 rounded-lg">
              <div className="flex items-center gap-2 mb-2">
                <span className="text-xl">{getSentimentEmoji(intelligence.sentiment.sentiment)}</span>
                <span className="text-gray-400 text-sm">Sentiment</span>
              </div>
              <div className={`font-semibold ${
                intelligence.sentiment.sentiment === 'bullish' ? 'text-green-500' :
                intelligence.sentiment.sentiment === 'bearish' ? 'text-red-500' :
                'text-gray-400'
              }`}>
                {intelligence.sentiment.sentiment.toUpperCase()}
              </div>
              <div className="text-xs text-gray-400 mt-1">
                Score: {(intelligence.sentiment.score * 100).toFixed(0)}%
              </div>
            </div>

            <div className="bg-gray-800 p-3 rounded-lg">
              <div className="flex items-center gap-2 mb-2">
                <span className="text-xl">📊</span>
                <span className="text-gray-400 text-sm">Technical</span>
              </div>
              <div className="text-white font-semibold">
                RSI: {intelligence.technical.rsi.toFixed(0)}
              </div>
              <div className="text-xs text-gray-400 mt-1">
                {intelligence.technical.rsi > 70 ? 'Overbought' :
                 intelligence.technical.rsi < 30 ? 'Oversold' :
                 'Neutral'}
              </div>
            </div>

            <div className="bg-gray-800 p-3 rounded-lg">
              <div className="flex items-center gap-2 mb-2">
                <span className="text-xl">🌍</span>
                <span className="text-gray-400 text-sm">Macro</span>
              </div>
              <div className="text-white font-semibold">
                {intelligence.macro.trend.toUpperCase()}
              </div>
              <div className="text-xs text-gray-400 mt-1">
                Long-term trend
              </div>
            </div>
          </div>
        </CardContent>
      </Card>

      {/* Recommendations & Alerts */}
      <div className="space-y-6">
        {/* Alerts */}
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
                    alert.type === 'warning' ? 'bg-yellow-900/30 text-yellow-200' :
                    'bg-blue-900/30 text-blue-200'
                  }`}
                >
                  {alert.message}
                </div>
              ))}
            </CardContent>
          </Card>
        )}

        {/* Recommendations */}
        <Card className="bg-gray-900 border-gray-800">
          <CardHeader>
            <CardTitle className="text-white text-base">Recommendations</CardTitle>
          </CardHeader>
          <CardContent className="space-y-2">
            {intelligence.recommendations.map((rec, index) => (
              <div
                key={index}
                className="text-sm text-gray-300 p-2 bg-gray-800 rounded"
              >
                {rec}
              </div>
            ))}
          </CardContent>
        </Card>

        {/* Last Updated */}
        <div className="text-xs text-gray-500 text-center">
          Last updated: {new Date(intelligence.timestamp).toLocaleTimeString()}
        </div>
      </div>
    </div>
  )
}
