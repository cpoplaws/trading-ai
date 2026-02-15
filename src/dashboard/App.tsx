/**
 * Main Trading Dashboard Application
 * Real-time trading data visualization with WebSocket updates
 */
import React, { useState, useEffect } from 'react';
import { LivePriceChart } from './components/LivePriceChart';
import { PortfolioSummary } from './components/PortfolioSummary';
import { TradeHistory } from './components/TradeHistory';
import { MLPredictions } from './components/MLPredictions';
import { PatternDetector } from './components/PatternDetector';
import { AlertCenter } from './components/AlertCenter';
import { useWebSocket } from './hooks/useWebSocket';

interface DashboardState {
  isConnected: boolean;
  lastUpdate: Date | null;
  activeTab: string;
}

export const App: React.FC = () => {
  const [state, setState] = useState<DashboardState>({
    isConnected: false,
    lastUpdate: null,
    activeTab: 'overview'
  });

  const { isConnected, lastMessage } = useWebSocket('ws://localhost:8765');

  useEffect(() => {
    setState(prev => ({
      ...prev,
      isConnected,
      lastUpdate: lastMessage ? new Date() : prev.lastUpdate
    }));
  }, [isConnected, lastMessage]);

  return (
    <div className="min-h-screen bg-gray-100">
      {/* Header */}
      <header className="bg-white shadow">
        <div className="max-w-7xl mx-auto px-4 py-6">
          <div className="flex justify-between items-center">
            <h1 className="text-3xl font-bold text-gray-900">
              AI Trading Dashboard
            </h1>

            <div className="flex items-center gap-4">
              {/* Connection Status */}
              <div className="flex items-center gap-2">
                <div className={`w-3 h-3 rounded-full ${
                  state.isConnected ? 'bg-green-500' : 'bg-red-500'
                }`} />
                <span className="text-sm text-gray-600">
                  {state.isConnected ? 'Connected' : 'Disconnected'}
                </span>
              </div>

              {/* Last Update */}
              {state.lastUpdate && (
                <span className="text-sm text-gray-500">
                  Last update: {state.lastUpdate.toLocaleTimeString()}
                </span>
              )}
            </div>
          </div>
        </div>
      </header>

      {/* Main Content */}
      <main className="max-w-7xl mx-auto px-4 py-8">
        {/* Top Row - Portfolio Summary */}
        <div className="mb-8">
          <PortfolioSummary />
        </div>

        {/* Second Row - Price Chart and ML Predictions */}
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6 mb-8">
          <div className="lg:col-span-2">
            <LivePriceChart symbol="ETH-USDC" />
          </div>
          <div>
            <MLPredictions />
          </div>
        </div>

        {/* Third Row - Pattern Detector and Alerts */}
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 mb-8">
          <PatternDetector />
          <AlertCenter />
        </div>

        {/* Bottom Row - Trade History */}
        <div>
          <TradeHistory />
        </div>
      </main>
    </div>
  );
};

export default App;
