'use client'

import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { Badge } from '@/components/ui/badge'
import { Button } from '@/components/ui/button'
import { useStrategies } from '@/hooks/useStrategies'

export function StrategyGrid() {
  const { strategies, loading, error, toggleStrategy } = useStrategies()

  if (loading) {
    return (
      <div className="text-center text-gray-400 py-8">
        Loading strategies...
      </div>
    )
  }

  if (error) {
    return (
      <div className="text-center text-red-500 py-8">
        Error loading strategies: {error}
      </div>
    )
  }

  return (
    <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
      {strategies.map((strategy) => (
        <Card key={strategy.id} className="bg-gray-900 border-gray-800">
          <CardHeader className="pb-3">
            <CardTitle className="text-lg text-white flex items-center justify-between">
              <span>{strategy.name}</span>
              <Badge
                variant={strategy.enabled ? "default" : "secondary"}
                className={strategy.enabled ? "bg-green-600" : "bg-gray-600"}
              >
                {strategy.enabled ? 'Active' : 'Paused'}
              </Badge>
            </CardTitle>
          </CardHeader>
          <CardContent className="space-y-3">
            <div className={`text-2xl font-bold ${strategy.pnl >= 0 ? 'text-green-500' : 'text-red-500'}`}>
              {strategy.pnl >= 0 ? '+' : ''}${strategy.pnl.toFixed(2)}
            </div>
            <div className="text-sm text-gray-400 space-y-1">
              <div className="flex justify-between">
                <span>Trades:</span>
                <span className="text-white">{strategy.trades}</span>
              </div>
              <div className="flex justify-between">
                <span>Win Rate:</span>
                <span className="text-white">{(strategy.win_rate * 100).toFixed(1)}%</span>
              </div>
            </div>
            <Button
              onClick={() => toggleStrategy(strategy.id, !strategy.enabled)}
              variant={strategy.enabled ? "destructive" : "default"}
              className="w-full"
              size="sm"
            >
              {strategy.enabled ? 'Disable' : 'Enable'}
            </Button>
          </CardContent>
        </Card>
      ))}
    </div>
  )
}
