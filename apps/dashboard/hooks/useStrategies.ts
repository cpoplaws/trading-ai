'use client'

import { useState, useEffect } from 'react'
import { getStrategies, toggleStrategy } from '@/lib/api-client'

export interface Strategy {
  id: string
  name: string
  enabled: boolean
  pnl: number
  trades: number
  win_rate: number
}

export function useStrategies() {
  const [strategies, setStrategies] = useState<Strategy[]>([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)

  const fetchStrategies = async () => {
    try {
      setLoading(true)
      setError(null)
      const response = await getStrategies()
      setStrategies(response.strategies || [])
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to fetch strategies')
      console.error('Error fetching strategies:', err)
    } finally {
      setLoading(false)
    }
  }

  const handleToggleStrategy = async (strategyId: string, enabled: boolean) => {
    try {
      await toggleStrategy(strategyId, enabled)
      // Optimistically update local state
      setStrategies(prev =>
        prev.map(s =>
          s.id === strategyId ? { ...s, enabled } : s
        )
      )
    } catch (err) {
      console.error('Error toggling strategy:', err)
      // Revert on error
      await fetchStrategies()
    }
  }

  useEffect(() => {
    fetchStrategies()
    // Refresh every 10 seconds
    const interval = setInterval(fetchStrategies, 10000)
    return () => clearInterval(interval)
  }, [])

  return {
    strategies,
    loading,
    error,
    toggleStrategy: handleToggleStrategy,
    refresh: fetchStrategies,
  }
}
