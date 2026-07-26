import { useState, useEffect } from 'react'
import { getPortfolio } from '@/lib/api-client'

export function usePortfolio() {
  const [portfolio, setPortfolio] = useState<any>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    fetchPortfolio()
    // Refresh every 30 seconds
    const interval = setInterval(fetchPortfolio, 30000)
    return () => clearInterval(interval)
  }, [])

  const fetchPortfolio = async () => {
    try {
      const data = await getPortfolio()
      setPortfolio(data)
      setError(null)
    } catch (err) {
      console.error('Failed to fetch portfolio:', err)
      setError('Failed to load portfolio')
    } finally {
      setLoading(false)
    }
  }

  return { portfolio, loading, error, refetch: fetchPortfolio }
}
