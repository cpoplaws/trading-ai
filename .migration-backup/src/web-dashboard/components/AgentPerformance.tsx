/**
 * Agent Performance Component
 * Displays RL agent performance metrics and recent decisions
 */
'use client';

import React, { useEffect, useState } from 'react';
import { apiClient } from '../lib/api-client';
import { wsClient } from '../lib/websocket-client';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Legend } from 'recharts';

export default function AgentPerformance() {
  const [agents, setAgents] = useState<any[]>([]);
  const [selectedAgent, setSelectedAgent] = useState<string>('');
  const [performance, setPerformance] = useState<any>(null);
  const [decisions, setDecisions] = useState<any[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    loadAgents();
  }, []);

  useEffect(() => {
    if (selectedAgent) {
      loadAgentData();

      // Subscribe to agent decisions
      const unsubscribe = wsClient.subscribeAgentDecisions(selectedAgent, (decision) => {
        setDecisions(prev => [decision, ...prev].slice(0, 10));
      });

      return () => unsubscribe();
    }
  }, [selectedAgent]);

  const loadAgents = async () => {
    try {
      const agentsData = await apiClient.getAgents();
      setAgents(agentsData);
      if (agentsData.length > 0) {
        setSelectedAgent(agentsData[0].agent_id);
      }
      setLoading(false);
    } catch (error) {
      console.error('Failed to load agents:', error);
    }
  };

  const loadAgentData = async () => {
    try {
      const [perfData, decisionsData] = await Promise.all([
        apiClient.getAgentPerformance(selectedAgent),
        apiClient.getAgentDecisions(selectedAgent, 10)
      ]);
      setPerformance(perfData);
      setDecisions(decisionsData);
    } catch (error) {
      console.error('Failed to load agent data:', error);
    }
  };

  if (loading) {
    return (
      <div className="bg-gray-800 rounded-lg p-6 shadow-lg animate-pulse">
        <div className="h-64 bg-gray-700 rounded"></div>
      </div>
    );
  }

  return (
    <div className="bg-gray-800 rounded-lg p-6 shadow-lg">
      <div className="flex justify-between items-center mb-4">
        <h2 className="text-2xl font-bold">RL Agent Performance</h2>

        {/* Agent Selector */}
        {agents.length > 0 && (
          <select
            value={selectedAgent}
            onChange={(e) => setSelectedAgent(e.target.value)}
            className="bg-gray-700 border border-gray-600 rounded px-3 py-1 text-sm text-white"
          >
            {agents.map((agent) => (
              <option key={agent.agent_id} value={agent.agent_id}>
                {agent.name || agent.agent_id}
              </option>
            ))}
          </select>
        )}
      </div>

      {performance && (
        <>
          {/* Performance Metrics */}
          <div className="grid grid-cols-3 gap-4 mb-6">
            <div className="bg-gray-700 rounded-lg p-3">
              <div className="text-gray-400 text-xs mb-1">Total Reward</div>
              <div className={`text-xl font-bold ${
                performance.total_reward >= 0 ? 'text-green-400' : 'text-red-400'
              }`}>
                {performance.total_reward?.toFixed(2) || 'N/A'}
              </div>
            </div>

            <div className="bg-gray-700 rounded-lg p-3">
              <div className="text-gray-400 text-xs mb-1">Win Rate</div>
              <div className="text-xl font-bold text-white">
                {((performance.win_rate || 0) * 100).toFixed(1)}%
              </div>
            </div>

            <div className="bg-gray-700 rounded-lg p-3">
              <div className="text-gray-400 text-xs mb-1">Sharpe Ratio</div>
              <div className="text-xl font-bold text-white">
                {performance.sharpe_ratio?.toFixed(2) || 'N/A'}
              </div>
            </div>
          </div>

          {/* Cumulative Reward Chart */}
          <div className="mb-6">
            <h3 className="text-sm font-semibold mb-3 text-gray-400">Cumulative Reward</h3>
            <ResponsiveContainer width="100%" height={150}>
              <LineChart data={performance.reward_history || []}>
                <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
                <XAxis
                  dataKey="episode"
                  stroke="#9CA3AF"
                  fontSize={10}
                />
                <YAxis stroke="#9CA3AF" fontSize={10} />
                <Tooltip
                  contentStyle={{ backgroundColor: '#1F2937', border: 'none', borderRadius: '8px' }}
                  labelStyle={{ color: '#F3F4F6' }}
                />
                <Line
                  type="monotone"
                  dataKey="reward"
                  stroke="#10B981"
                  strokeWidth={2}
                  dot={false}
                />
              </LineChart>
            </ResponsiveContainer>
          </div>
        </>
      )}

      {/* Recent Decisions */}
      <div>
        <h3 className="text-lg font-semibold mb-3">Recent Decisions</h3>
        <div className="space-y-2 max-h-64 overflow-y-auto">
          {decisions.map((decision, i) => {
            const action = decision.action?.toUpperCase() || 'UNKNOWN';
            const isBuy = action === 'BUY';
            const isSell = action === 'SELL';

            return (
              <div key={i} className="bg-gray-700 p-3 rounded text-xs">
                <div className="flex justify-between items-center mb-1">
                  <span className={`px-2 py-1 rounded font-bold ${
                    isBuy ? 'bg-green-600' : isSell ? 'bg-red-600' : 'bg-gray-600'
                  }`}>
                    {action}
                  </span>
                  <span className="text-gray-400">
                    {decision.symbol || 'N/A'}
                  </span>
                </div>

                <div className="flex justify-between text-gray-400 mt-2">
                  <span>Reward: <span className={decision.reward >= 0 ? 'text-green-400' : 'text-red-400'}>
                    {decision.reward?.toFixed(4)}
                  </span></span>
                  <span>{new Date(decision.timestamp).toLocaleTimeString()}</span>
                </div>

                {decision.confidence && (
                  <div className="mt-2">
                    <div className="flex justify-between text-xs mb-1">
                      <span className="text-gray-400">Confidence</span>
                      <span className="text-white">{(decision.confidence * 100).toFixed(0)}%</span>
                    </div>
                    <div className="h-1 bg-gray-600 rounded-full overflow-hidden">
                      <div
                        className="h-full bg-blue-500"
                        style={{ width: `${decision.confidence * 100}%` }}
                      />
                    </div>
                  </div>
                )}
              </div>
            );
          })}

          {decisions.length === 0 && (
            <div className="text-center text-gray-400 py-4">
              No recent decisions
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
