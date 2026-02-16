/**
 * Risk Metrics Component
 * Displays VaR, CVaR, position sizing, and risk limits
 */
'use client';

import React, { useEffect, useState } from 'react';
import { apiClient } from '../lib/api-client';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from 'recharts';

export default function RiskMetrics() {
  const [metrics, setMetrics] = useState<any>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    loadRiskMetrics();
    const interval = setInterval(loadRiskMetrics, 30000); // Update every 30s
    return () => clearInterval(interval);
  }, []);

  const loadRiskMetrics = async () => {
    try {
      const data = await apiClient.getRiskMetrics();
      setMetrics(data);
      setLoading(false);
    } catch (error) {
      console.error('Failed to load risk metrics:', error);
    }
  };

  if (loading) {
    return (
      <div className="bg-gray-800 rounded-lg p-6 shadow-lg animate-pulse">
        <div className="h-64 bg-gray-700 rounded"></div>
      </div>
    );
  }

  const var95 = metrics?.var_95 || 0;
  const cvar95 = metrics?.cvar_95 || 0;
  const portfolioValue = metrics?.portfolio_value || 0;
  const varPercent = (var95 / portfolioValue) * 100;

  return (
    <div className="bg-gray-800 rounded-lg p-6 shadow-lg">
      <h2 className="text-2xl font-bold mb-6">Risk Metrics</h2>

      {/* VaR and CVaR */}
      <div className="grid grid-cols-2 gap-4 mb-6">
        <div className="bg-gray-700 rounded-lg p-4">
          <div className="text-gray-400 text-sm mb-1">VaR (95%)</div>
          <div className="text-2xl font-bold text-red-400">
            ${var95.toLocaleString()}
          </div>
          <div className="text-sm text-gray-400 mt-1">
            {varPercent.toFixed(2)}% of portfolio
          </div>
        </div>

        <div className="bg-gray-700 rounded-lg p-4">
          <div className="text-gray-400 text-sm mb-1">CVaR (95%)</div>
          <div className="text-2xl font-bold text-red-400">
            ${cvar95.toLocaleString()}
          </div>
          <div className="text-sm text-gray-400 mt-1">
            Tail risk estimate
          </div>
        </div>
      </div>

      {/* Risk Limits */}
      <div className="mb-6">
        <h3 className="text-lg font-semibold mb-3">Risk Limits</h3>
        <div className="space-y-3">
          {/* Max Position Size */}
          <div>
            <div className="flex justify-between text-sm mb-1">
              <span className="text-gray-400">Max Position Size</span>
              <span className="text-white">
                {metrics?.position_usage || 0}% / 100%
              </span>
            </div>
            <div className="h-2 bg-gray-600 rounded-full overflow-hidden">
              <div
                className={`h-full ${
                  metrics?.position_usage > 80 ? 'bg-red-500' :
                  metrics?.position_usage > 60 ? 'bg-yellow-500' : 'bg-green-500'
                }`}
                style={{ width: `${metrics?.position_usage || 0}%` }}
              />
            </div>
          </div>

          {/* Daily Loss Limit */}
          <div>
            <div className="flex justify-between text-sm mb-1">
              <span className="text-gray-400">Daily Loss Limit</span>
              <span className="text-white">
                {metrics?.daily_loss_usage || 0}% / 100%
              </span>
            </div>
            <div className="h-2 bg-gray-600 rounded-full overflow-hidden">
              <div
                className={`h-full ${
                  metrics?.daily_loss_usage > 80 ? 'bg-red-500' :
                  metrics?.daily_loss_usage > 60 ? 'bg-yellow-500' : 'bg-green-500'
                }`}
                style={{ width: `${metrics?.daily_loss_usage || 0}%` }}
              />
            </div>
          </div>

          {/* Leverage */}
          <div>
            <div className="flex justify-between text-sm mb-1">
              <span className="text-gray-400">Leverage</span>
              <span className="text-white">
                {metrics?.leverage?.toFixed(2)}x / {metrics?.max_leverage || 3}x
              </span>
            </div>
            <div className="h-2 bg-gray-600 rounded-full overflow-hidden">
              <div
                className={`h-full ${
                  (metrics?.leverage / metrics?.max_leverage) > 0.8 ? 'bg-red-500' :
                  (metrics?.leverage / metrics?.max_leverage) > 0.6 ? 'bg-yellow-500' : 'bg-green-500'
                }`}
                style={{ width: `${((metrics?.leverage / metrics?.max_leverage) * 100) || 0}%` }}
              />
            </div>
          </div>
        </div>
      </div>

      {/* VaR by Asset */}
      {metrics?.var_by_asset && (
        <div>
          <h3 className="text-lg font-semibold mb-3">VaR by Asset</h3>
          <ResponsiveContainer width="100%" height={200}>
            <BarChart data={metrics.var_by_asset}>
              <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
              <XAxis dataKey="symbol" stroke="#9CA3AF" />
              <YAxis stroke="#9CA3AF" />
              <Tooltip
                contentStyle={{ backgroundColor: '#1F2937', border: 'none', borderRadius: '8px' }}
                labelStyle={{ color: '#F3F4F6' }}
              />
              <Bar dataKey="var" fill="#EF4444" />
            </BarChart>
          </ResponsiveContainer>
        </div>
      )}

      {/* Risk Status */}
      <div className="mt-6 p-4 rounded-lg bg-gray-700">
        <div className="flex items-center justify-between">
          <span className="text-gray-400">Overall Risk Status</span>
          <span className={`px-3 py-1 rounded-full text-sm font-bold ${
            metrics?.risk_status === 'low' ? 'bg-green-600' :
            metrics?.risk_status === 'medium' ? 'bg-yellow-600' : 'bg-red-600'
          }`}>
            {(metrics?.risk_status || 'unknown').toUpperCase()}
          </span>
        </div>
      </div>
    </div>
  );
}
