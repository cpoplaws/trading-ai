/**
 * Portfolio Overview Component
 * Displays total portfolio value, P&L, and key metrics
 */
'use client';

import React, { useEffect, useState } from 'react';
import { apiClient } from '../lib/api-client';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from 'recharts';

export default function PortfolioOverview() {
  const [portfolio, setPortfolio] = useState<any>(null);
  const [stats, setStats] = useState<any>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    loadPortfolioData();
    const interval = setInterval(loadPortfolioData, 10000); // Update every 10s
    return () => clearInterval(interval);
  }, []);

  const loadPortfolioData = async () => {
    try {
      const [portfolioData, statsData] = await Promise.all([
        apiClient.getPortfolioValue(),
        apiClient.getPortfolioStats()
      ]);
      setPortfolio(portfolioData);
      setStats(statsData);
      setLoading(false);
    } catch (error) {
      console.error('Failed to load portfolio:', error);
    }
  };

  if (loading) {
    return (
      <div className="bg-gray-800 rounded-lg p-6 shadow-lg animate-pulse">
        <div className="h-32 bg-gray-700 rounded"></div>
      </div>
    );
  }

  const totalValue = portfolio?.total_value || 0;
  const dailyPnL = portfolio?.daily_pnl || 0;
  const dailyPnLPercent = portfolio?.daily_pnl_percent || 0;

  return (
    <div className="bg-gray-800 rounded-lg p-6 shadow-lg">
      <h2 className="text-2xl font-bold mb-6">Portfolio Overview</h2>

      <div className="grid grid-cols-1 md:grid-cols-4 gap-6 mb-8">
        {/* Total Value */}
        <div className="bg-gray-700 rounded-lg p-4">
          <div className="text-gray-400 text-sm mb-1">Total Value</div>
          <div className="text-3xl font-bold text-white">
            ${totalValue.toLocaleString()}
          </div>
        </div>

        {/* Daily P&L */}
        <div className="bg-gray-700 rounded-lg p-4">
          <div className="text-gray-400 text-sm mb-1">Daily P&L</div>
          <div className={`text-3xl font-bold ${dailyPnL >= 0 ? 'text-green-500' : 'text-red-500'}`}>
            {dailyPnL >= 0 ? '+' : ''}{dailyPnL.toLocaleString()}
          </div>
          <div className={`text-sm ${dailyPnLPercent >= 0 ? 'text-green-400' : 'text-red-400'}`}>
            {dailyPnLPercent >= 0 ? '+' : ''}{dailyPnLPercent.toFixed(2)}%
          </div>
        </div>

        {/* Sharpe Ratio */}
        <div className="bg-gray-700 rounded-lg p-4">
          <div className="text-gray-400 text-sm mb-1">Sharpe Ratio</div>
          <div className="text-3xl font-bold text-white">
            {stats?.sharpe_ratio?.toFixed(2) || 'N/A'}
          </div>
          <div className="text-sm text-gray-400">
            {stats?.sharpe_ratio > 2 ? 'Excellent' : stats?.sharpe_ratio > 1 ? 'Good' : 'Fair'}
          </div>
        </div>

        {/* Win Rate */}
        <div className="bg-gray-700 rounded-lg p-4">
          <div className="text-gray-400 text-sm mb-1">Win Rate</div>
          <div className="text-3xl font-bold text-white">
            {((stats?.win_rate || 0) * 100).toFixed(1)}%
          </div>
          <div className="text-sm text-gray-400">
            {stats?.total_trades || 0} trades
          </div>
        </div>
      </div>

      {/* Portfolio Value Chart */}
      <div className="bg-gray-700 rounded-lg p-4">
        <h3 className="text-lg font-semibold mb-4">Portfolio Value (24h)</h3>
        <ResponsiveContainer width="100%" height={200}>
          <LineChart data={portfolio?.history || []}>
            <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
            <XAxis
              dataKey="timestamp"
              stroke="#9CA3AF"
              tickFormatter={(ts) => new Date(ts).toLocaleTimeString()}
            />
            <YAxis stroke="#9CA3AF" />
            <Tooltip
              contentStyle={{ backgroundColor: '#1F2937', border: 'none', borderRadius: '8px' }}
              labelStyle={{ color: '#F3F4F6' }}
            />
            <Line
              type="monotone"
              dataKey="value"
              stroke="#10B981"
              strokeWidth={2}
              dot={false}
            />
          </LineChart>
        </ResponsiveContainer>
      </div>

      {/* Additional Stats */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mt-6">
        <div>
          <div className="text-gray-400 text-xs mb-1">Max Drawdown</div>
          <div className="text-red-400 font-semibold">
            {((stats?.max_drawdown || 0) * 100).toFixed(2)}%
          </div>
        </div>
        <div>
          <div className="text-gray-400 text-xs mb-1">Profit Factor</div>
          <div className="text-white font-semibold">
            {stats?.profit_factor?.toFixed(2) || 'N/A'}
          </div>
        </div>
        <div>
          <div className="text-gray-400 text-xs mb-1">Avg Win</div>
          <div className="text-green-400 font-semibold">
            ${(stats?.avg_win || 0).toLocaleString()}
          </div>
        </div>
        <div>
          <div className="text-gray-400 text-xs mb-1">Avg Loss</div>
          <div className="text-red-400 font-semibold">
            -${Math.abs(stats?.avg_loss || 0).toLocaleString()}
          </div>
        </div>
      </div>
    </div>
  );
}
