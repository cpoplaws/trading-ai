/**
 * Main Dashboard Component
 * Real-time trading system monitoring
 */
'use client';

import React, { useEffect, useState } from 'react';
import { apiClient } from '../lib/api-client';
import { wsClient } from '../lib/websocket-client';
import PortfolioOverview from './PortfolioOverview';
import MarketData from './MarketData';
import TradingSignals from './TradingSignals';
import RiskMetrics from './RiskMetrics';
import AgentPerformance from './AgentPerformance';
import toast, { Toaster } from 'react-hot-toast';

export default function Dashboard() {
  const [isConnected, setIsConnected] = useState(false);
  const [healthStatus, setHealthStatus] = useState<any>(null);
  const [selectedSymbol, setSelectedSymbol] = useState('BTCUSDT');

  useEffect(() => {
    // Check health on mount
    checkHealth();

    // Connect WebSocket
    const apiKey = localStorage.getItem('api_key');
    if (apiKey) {
      connectWebSocket(apiKey);
    }

    // Refresh health every 30 seconds
    const healthInterval = setInterval(checkHealth, 30000);

    return () => {
      clearInterval(healthInterval);
      wsClient.disconnect();
    };
  }, []);

  const checkHealth = async () => {
    try {
      const health = await apiClient.getHealthDetailed();
      setHealthStatus(health);
    } catch (error) {
      console.error('Health check failed:', error);
      toast.error('System health check failed');
    }
  };

  const connectWebSocket = async (apiKey: string) => {
    try {
      await wsClient.connect(apiKey);
      setIsConnected(true);
      toast.success('Connected to real-time data');
    } catch (error) {
      console.error('WebSocket connection failed:', error);
      toast.error('Failed to connect to real-time data');
      setIsConnected(false);
    }
  };

  return (
    <div className="min-h-screen bg-gray-900 text-white">
      <Toaster position="top-right" />

      {/* Header */}
      <header className="bg-gray-800 shadow-lg">
        <div className="container mx-auto px-4 py-4">
          <div className="flex items-center justify-between">
            <div>
              <h1 className="text-3xl font-bold text-white">
                🤖 Trading AI Dashboard
              </h1>
              <p className="text-gray-400 text-sm mt-1">
                Real-time monitoring and control
              </p>
            </div>

            <div className="flex items-center space-x-4">
              {/* Connection Status */}
              <div className="flex items-center space-x-2">
                <div className={`w-3 h-3 rounded-full ${isConnected ? 'bg-green-500' : 'bg-red-500'}`} />
                <span className="text-sm">
                  {isConnected ? 'Connected' : 'Disconnected'}
                </span>
              </div>

              {/* System Health */}
              {healthStatus && (
                <div className={`px-3 py-1 rounded text-sm ${
                  healthStatus.status === 'healthy' ? 'bg-green-600' : 'bg-red-600'
                }`}>
                  {healthStatus.status === 'healthy' ? '✓ Healthy' : '✗ Unhealthy'}
                </div>
              )}
            </div>
          </div>
        </div>
      </header>

      {/* Main Content */}
      <main className="container mx-auto px-4 py-6">
        {/* Symbol Selector */}
        <div className="mb-6">
          <label className="block text-sm font-medium mb-2">
            Trading Pair
          </label>
          <select
            value={selectedSymbol}
            onChange={(e) => setSelectedSymbol(e.target.value)}
            className="bg-gray-800 border border-gray-700 rounded-lg px-4 py-2 text-white focus:ring-2 focus:ring-blue-500"
          >
            <option value="BTCUSDT">BTC/USDT</option>
            <option value="ETHUSDT">ETH/USDT</option>
            <option value="SOLUSDT">SOL/USDT</option>
            <option value="BNBUSDT">BNB/USDT</option>
            <option value="ADAUSDT">ADA/USDT</option>
          </select>
        </div>

        {/* Portfolio Overview - Full Width */}
        <div className="mb-6">
          <PortfolioOverview />
        </div>

        {/* Main Grid */}
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 mb-6">
          {/* Market Data */}
          <MarketData symbol={selectedSymbol} />

          {/* Trading Signals */}
          <TradingSignals symbol={selectedSymbol} />
        </div>

        {/* Risk Metrics and Agent Performance */}
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          {/* Risk Metrics */}
          <RiskMetrics />

          {/* Agent Performance */}
          <AgentPerformance />
        </div>
      </main>

      {/* Footer */}
      <footer className="bg-gray-800 mt-12 py-6">
        <div className="container mx-auto px-4 text-center text-gray-400 text-sm">
          <p>Trading AI System v1.0.0 | Last update: {new Date().toLocaleString()}</p>
          <p className="mt-2">
            {healthStatus && (
              <>
                Uptime: {healthStatus.uptime || 'N/A'} |
                Database: {healthStatus.database?.status || 'Unknown'} |
                Redis: {healthStatus.redis?.status || 'Unknown'}
              </>
            )}
          </p>
        </div>
      </footer>
    </div>
  );
}
