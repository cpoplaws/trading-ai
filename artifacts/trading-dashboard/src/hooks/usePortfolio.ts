import { useState, useEffect } from 'react'
import { getPortfolio } from '@/lib/api-client'

export function usePortfolio() {
  const [portfolio, setPortfolio] = useState<any>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)

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

  useEffect(() => {
    fetchPortfolio()
    const interval = setInterval(fetchPortfolio, 30000)
    return () => clearInterval(interval)
  }, [])

  return { portfolio, loading, error, refetch: fetchPortfolio }
}
