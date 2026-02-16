/**
 * Market Data Component
 * Real-time price, orderbook, and trades
 */
'use client';

import React, { useEffect, useState } from 'react';
import { wsClient } from '../lib/websocket-client';
import { apiClient } from '../lib/api-client';

interface MarketDataProps {
  symbol: string;
}

export default function MarketData({ symbol }: MarketDataProps) {
  const [ticker, setTicker] = useState<any>(null);
  const [trades, setTrades] = useState<any[]>([]);
  const [orderbook, setOrderbook] = useState<any>(null);

  useEffect(() => {
    // Subscribe to real-time data
    const unsubTicker = wsClient.subscribeTicker(symbol, (data) => {
      setTicker(data);
    });

    const unsubTrades = wsClient.subscribeTrades(symbol, (data) => {
      setTrades(prev => [data, ...prev].slice(0, 20)); // Keep last 20 trades
    });

    const unsubOrderbook = wsClient.subscribeOrderBook(symbol, (data) => {
      setOrderbook(data);
    });

    // Initial load from API
    loadInitialData();

    return () => {
      unsubTicker();
      unsubTrades();
      unsubOrderbook();
    };
  }, [symbol]);

  const loadInitialData = async () => {
    try {
      const [tickerData, tradesData, orderbookData] = await Promise.all([
        apiClient.getTicker(symbol),
        apiClient.getTrades(symbol, 20),
        apiClient.getOrderBook(symbol, 10)
      ]);
      setTicker(tickerData);
      setTrades(tradesData);
      setOrderbook(orderbookData);
    } catch (error) {
      console.error('Failed to load market data:', error);
    }
  };

  const priceChange = ticker?.price_change_percent || 0;
  const isPriceUp = priceChange >= 0;

  return (
    <div className="bg-gray-800 rounded-lg p-6 shadow-lg">
      <h2 className="text-2xl font-bold mb-4">{symbol}</h2>

      {/* Current Price */}
      {ticker && (
        <div className="mb-6">
          <div className={`text-4xl font-bold ${isPriceUp ? 'text-green-500' : 'text-red-500'}`}>
            ${ticker.last_price?.toLocaleString()}
          </div>
          <div className={`text-sm mt-1 ${isPriceUp ? 'text-green-400' : 'text-red-400'}`}>
            {isPriceUp ? '+' : ''}{priceChange.toFixed(2)}% (24h)
          </div>
          <div className="grid grid-cols-3 gap-4 mt-4 text-sm">
            <div>
              <div className="text-gray-400">24h High</div>
              <div className="text-white font-semibold">${ticker.high_24h?.toLocaleString()}</div>
            </div>
            <div>
              <div className="text-gray-400">24h Low</div>
              <div className="text-white font-semibold">${ticker.low_24h?.toLocaleString()}</div>
            </div>
            <div>
              <div className="text-gray-400">24h Volume</div>
              <div className="text-white font-semibold">{ticker.volume_24h?.toLocaleString()}</div>
            </div>
          </div>
        </div>
      )}

      {/* Order Book */}
      <div className="mb-6">
        <h3 className="text-lg font-semibold mb-3">Order Book</h3>
        <div className="grid grid-cols-2 gap-4">
          {/* Asks (Sell Orders) */}
          <div>
            <div className="text-red-400 text-xs mb-2 font-semibold">ASKS</div>
            <div className="space-y-1">
              {orderbook?.asks?.slice(0, 5).map((ask: any, i: number) => (
                <div key={i} className="flex justify-between text-xs">
                  <span className="text-red-400">${ask.price.toFixed(2)}</span>
                  <span className="text-gray-400">{ask.quantity.toFixed(4)}</span>
                </div>
              ))}
            </div>
          </div>

          {/* Bids (Buy Orders) */}
          <div>
            <div className="text-green-400 text-xs mb-2 font-semibold">BIDS</div>
            <div className="space-y-1">
              {orderbook?.bids?.slice(0, 5).map((bid: any, i: number) => (
                <div key={i} className="flex justify-between text-xs">
                  <span className="text-green-400">${bid.price.toFixed(2)}</span>
                  <span className="text-gray-400">{bid.quantity.toFixed(4)}</span>
                </div>
              ))}
            </div>
          </div>
        </div>
      </div>

      {/* Recent Trades */}
      <div>
        <h3 className="text-lg font-semibold mb-3">Recent Trades</h3>
        <div className="space-y-2 max-h-64 overflow-y-auto">
          {trades.map((trade, i) => {
            const isBuy = trade.side === 'buy';
            return (
              <div key={i} className="flex justify-between text-xs bg-gray-700 p-2 rounded">
                <span className={isBuy ? 'text-green-400' : 'text-red-400'}>
                  {isBuy ? 'BUY' : 'SELL'}
                </span>
                <span className="text-white font-semibold">
                  ${trade.price?.toFixed(2)}
                </span>
                <span className="text-gray-400">
                  {trade.quantity?.toFixed(4)}
                </span>
                <span className="text-gray-500">
                  {new Date(trade.timestamp).toLocaleTimeString()}
                </span>
              </div>
            );
          })}
        </div>
      </div>
    </div>
  );
}
