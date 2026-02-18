'use client'

import { useState, useEffect } from 'react'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { Badge } from '@/components/ui/badge'
import { Button } from '@/components/ui/button'

interface Agent {
  enabled: boolean
  performance: {
    total_decisions: number
    successful_decisions: number
    failed_decisions: number
    accuracy: number
  }
  recent_decisions: number
}

interface SwarmStatus {
  enabled: boolean
  coordination_mode: string
  agents: {
    [key: string]: Agent
  }
}

interface Decision {
  agent: string
  action: string
  confidence: number
  reason: string
  timestamp: string
}

const API_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000'

const agentInfo = {
  execution: {
    name: 'Execution Agent',
    description: 'Optimizes trade timing and sizing',
    icon: '⚡'
  },
  risk: {
    name: 'Risk Agent',
    description: 'Monitors portfolio risk and enforces limits',
    icon: '🛡️'
  },
  arbitrage: {
    name: 'Arbitrage Agent',
    description: 'Finds price discrepancies and arbitrage',
    icon: '🔄'
  },
  market_making: {
    name: 'Market Making Agent',
    description: 'Provides liquidity and captures spread',
    icon: '📊'
  }
}

export function AgentSwarm() {
  const [swarmStatus, setSwarmStatus] = useState<SwarmStatus | null>(null)
  const [decisions, setDecisions] = useState<Decision[]>([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)

  const fetchSwarmStatus = async () => {
    try {
      const response = await fetch(`${API_URL}/api/agents/status`)
      if (!response.ok) throw new Error('Failed to fetch swarm status')
      const data = await response.json()
      setSwarmStatus(data)
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to fetch swarm status')
      console.error('Error fetching swarm status:', err)
    }
  }

  const fetchDecisions = async () => {
    try {
      const response = await fetch(`${API_URL}/api/agents/decisions?limit=10`)
      if (!response.ok) throw new Error('Failed to fetch decisions')
      const data = await response.json()
      setDecisions(data.decisions || [])
    } catch (err) {
      console.error('Error fetching decisions:', err)
    }
  }

  const toggleSwarm = async () => {
    try {
      const endpoint = swarmStatus?.enabled ? '/api/agents/disable' : '/api/agents/enable'
      const response = await fetch(`${API_URL}${endpoint}`, { method: 'POST' })
      if (!response.ok) throw new Error('Failed to toggle swarm')
      await fetchSwarmStatus()
    } catch (err) {
      console.error('Error toggling swarm:', err)
    }
  }

  const toggleAgent = async (agentName: string) => {
    try {
      const response = await fetch(`${API_URL}/api/agents/${agentName}/toggle`, {
        method: 'POST'
      })
      if (!response.ok) throw new Error('Failed to toggle agent')
      await fetchSwarmStatus()
    } catch (err) {
      console.error('Error toggling agent:', err)
    }
  }

  useEffect(() => {
    const fetchData = async () => {
      setLoading(true)
      await Promise.all([fetchSwarmStatus(), fetchDecisions()])
      setLoading(false)
    }

    fetchData()

    // Refresh every 5 seconds
    const interval = setInterval(() => {
      fetchSwarmStatus()
      fetchDecisions()
    }, 5000)

    return () => clearInterval(interval)
  }, [])

  if (loading) {
    return (
      <div className="text-center text-gray-400 py-8">
        Loading agent swarm...
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

  return (
    <div className="space-y-6">
      {/* Swarm Control */}
      <Card className="bg-gray-900 border-gray-800">
        <CardHeader>
          <div className="flex items-center justify-between">
            <div>
              <CardTitle className="text-xl text-white">Agent Swarm Control</CardTitle>
              <p className="text-gray-400 text-sm mt-1">
                Multi-agent coordination • {swarmStatus?.coordination_mode}
              </p>
            </div>
            <Button
              onClick={toggleSwarm}
              variant={swarmStatus?.enabled ? "destructive" : "default"}
              size="lg"
            >
              {swarmStatus?.enabled ? '⏸️ Disable Swarm' : '▶️ Enable Swarm'}
            </Button>
          </div>
        </CardHeader>
      </Card>

      {/* Agent Cards */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
        {Object.entries(swarmStatus?.agents || {}).map(([agentKey, agent]) => {
          const info = agentInfo[agentKey as keyof typeof agentInfo]
          if (!info) return null

          return (
            <Card key={agentKey} className="bg-gray-900 border-gray-800">
              <CardHeader className="pb-3">
                <div className="flex items-center justify-between">
                  <div className="flex items-center gap-2">
                    <span className="text-2xl">{info.icon}</span>
                    <div>
                      <CardTitle className="text-base text-white">{info.name}</CardTitle>
                      <Badge
                        variant={agent.enabled ? "default" : "secondary"}
                        className={`mt-1 ${agent.enabled ? 'bg-green-600' : 'bg-gray-600'}`}
                      >
                        {agent.enabled ? 'Active' : 'Paused'}
                      </Badge>
                    </div>
                  </div>
                </div>
              </CardHeader>
              <CardContent className="space-y-3">
                <p className="text-xs text-gray-400">{info.description}</p>

                <div className="text-sm text-gray-400 space-y-1">
                  <div className="flex justify-between">
                    <span>Decisions:</span>
                    <span className="text-white">{agent.performance.total_decisions}</span>
                  </div>
                  <div className="flex justify-between">
                    <span>Accuracy:</span>
                    <span className="text-white">
                      {(agent.performance.accuracy * 100).toFixed(1)}%
                    </span>
                  </div>
                </div>

                <Button
                  onClick={() => toggleAgent(agentKey)}
                  variant={agent.enabled ? "outline" : "default"}
                  className="w-full"
                  size="sm"
                  disabled={!swarmStatus?.enabled}
                >
                  {agent.enabled ? 'Disable' : 'Enable'}
                </Button>
              </CardContent>
            </Card>
          )
        })}
      </div>

      {/* Recent Decisions */}
      <Card className="bg-gray-900 border-gray-800">
        <CardHeader>
          <CardTitle className="text-white">Recent Agent Decisions</CardTitle>
        </CardHeader>
        <CardContent>
          {decisions.length === 0 ? (
            <p className="text-gray-400 text-center py-4">No decisions yet</p>
          ) : (
            <div className="space-y-2">
              {decisions.map((decision, index) => (
                <div
                  key={index}
                  className="flex items-center justify-between p-3 bg-gray-800 rounded-lg"
                >
                  <div className="flex items-center gap-3">
                    <span className="text-xl">
                      {agentInfo[decision.agent as keyof typeof agentInfo]?.icon || '🤖'}
                    </span>
                    <div>
                      <div className="flex items-center gap-2">
                        <span className="text-white font-medium">
                          {agentInfo[decision.agent as keyof typeof agentInfo]?.name || decision.agent}
                        </span>
                        <Badge
                          variant={
                            decision.action === 'BUY' ? 'default' :
                            decision.action === 'SELL' ? 'destructive' :
                            'secondary'
                          }
                        >
                          {decision.action}
                        </Badge>
                      </div>
                      <p className="text-xs text-gray-400 mt-1">{decision.reason}</p>
                    </div>
                  </div>
                  <div className="text-right">
                    <div className="text-sm text-white">
                      {(decision.confidence * 100).toFixed(0)}% confidence
                    </div>
                    <div className="text-xs text-gray-400">
                      {new Date(decision.timestamp).toLocaleTimeString()}
                    </div>
                  </div>
                </div>
              ))}
            </div>
          )}
        </CardContent>
      </Card>
    </div>
  )
}
