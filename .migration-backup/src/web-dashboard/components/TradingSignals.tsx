/**
 * Trading Signals Component
 * Displays recent trading signals and their performance
 */
'use client';

import React, { useEffect, useState } from 'react';
import { apiClient } from '../lib/api-client';
import { wsClient } from '../lib/websocket-client';

interface TradingSignalsProps {
  symbol: string;
}

export default function TradingSignals({ symbol }: TradingSignalsProps) {
  const [signals, setSignals] = useState<any[]>([]);
  const [performance, setPerformance] = useState<any>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    loadSignals();

    // Subscribe to new signals
    const unsubscribe = wsClient.subscribeSignals((signal) => {
      if (signal.symbol === symbol) {
        setSignals(prev => [signal, ...prev].slice(0, 10));
      }
    });

    return () => unsubscribe();
  }, [symbol]);

  const loadSignals = async () => {
    try {
      const [signalsData, perfData] = await Promise.all([
        apiClient.getSignals(symbol, 10),
        apiClient.getSignalPerformance('24h')
      ]);
      setSignals(signalsData);
      setPerformance(perfData);
      setLoading(false);
    } catch (error) {
      console.error('Failed to load signals:', error);
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
        <h2 className="text-2xl font-bold">Trading Signals</h2>
        {performance && (
          <div className="text-sm">
            <span className="text-gray-400">Win Rate: </span>
            <span className={performance.win_rate >= 0.6 ? 'text-green-400' : 'text-red-400'}>
              {(performance.win_rate * 100).toFixed(1)}%
            </span>
          </div>
        )}
      </div>

      {/* Performance Summary */}
      {performance && (
        <div className="grid grid-cols-3 gap-4 mb-6 p-4 bg-gray-700 rounded-lg">
          <div>
            <div className="text-gray-400 text-xs mb-1">Total Signals</div>
            <div className="text-white font-semibold">{performance.total_signals}</div>
          </div>
          <div>
            <div className="text-gray-400 text-xs mb-1">Accuracy</div>
            <div className="text-green-400 font-semibold">
              {(performance.accuracy * 100).toFixed(1)}%
            </div>
          </div>
          <div>
            <div className="text-gray-400 text-xs mb-1">Avg Return</div>
            <div className={`font-semibold ${performance.avg_return >= 0 ? 'text-green-400' : 'text-red-400'}`}>
              {performance.avg_return >= 0 ? '+' : ''}{(performance.avg_return * 100).toFixed(2)}%
            </div>
          </div>
        </div>
      )}

      {/* Recent Signals */}
      <div className="space-y-3 max-h-96 overflow-y-auto">
        {signals.map((signal, i) => {
          const isBuy = signal.signal === 'BUY';
          const isHold = signal.signal === 'HOLD';
          const confidence = signal.confidence || 0;

          return (
            <div key={i} className="bg-gray-700 p-4 rounded-lg">
              <div className="flex justify-between items-start mb-2">
                <div>
                  <span className={`px-3 py-1 rounded-full text-xs font-bold ${
                    isBuy ? 'bg-green-600' : isHold ? 'bg-yellow-600' : 'bg-red-600'
                  }`}>
                    {signal.signal}
                  </span>
                  <span className="ml-3 text-white font-semibold">
                    ${signal.price?.toFixed(2)}
                  </span>
                </div>
                <div className="text-right">
                  <div className="text-gray-400 text-xs">Confidence</div>
                  <div className="text-white font-semibold">
                    {(confidence * 100).toFixed(0)}%
                  </div>
                </div>
              </div>

              <div className="flex justify-between text-xs text-gray-400 mt-2">
                <span>{signal.agent_id || 'Unknown Agent'}</span>
                <span>{new Date(signal.timestamp).toLocaleString()}</span>
              </div>

              {/* Confidence Bar */}
              <div className="mt-2 h-2 bg-gray-600 rounded-full overflow-hidden">
                <div
                  className={`h-full ${
                    confidence >= 0.7 ? 'bg-green-500' :
                    confidence >= 0.5 ? 'bg-yellow-500' : 'bg-red-500'
                  }`}
                  style={{ width: `${confidence * 100}%` }}
                />
              </div>

              {/* Reasoning (if available) */}
              {signal.reasoning && (
                <div className="mt-2 text-xs text-gray-400 italic">
                  {signal.reasoning}
                </div>
              )}
            </div>
          );
        })}

        {signals.length === 0 && (
          <div className="text-center text-gray-400 py-8">
            No signals available for {symbol}
          </div>
        )}
      </div>
    </div>
  );
}
