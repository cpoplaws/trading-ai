/**
 * WebSocket Client for Real-Time Data
 * Connects to backend WebSocket for live market data and trading signals
 */
import { io, Socket } from 'socket.io-client';

const WS_URL = process.env.NEXT_PUBLIC_WS_URL || 'ws://localhost:8000';

export type MessageHandler = (data: any) => void;

export interface WebSocketSubscription {
  channel: string;
  handler: MessageHandler;
}

class WebSocketClient {
  private socket: Socket | null = null;
  private subscriptions: Map<string, Set<MessageHandler>> = new Map();
  private reconnectAttempts = 0;
  private maxReconnectAttempts = 5;
  private reconnectDelay = 1000;
  private isConnecting = false;

  connect(apiKey: string): Promise<void> {
    return new Promise((resolve, reject) => {
      if (this.isConnecting) {
        return;
      }

      this.isConnecting = true;

      this.socket = io(WS_URL, {
        auth: {
          api_key: apiKey
        },
        transports: ['websocket'],
        reconnection: true,
        reconnectionAttempts: this.maxReconnectAttempts,
        reconnectionDelay: this.reconnectDelay,
      });

      this.socket.on('connect', () => {
        console.log('WebSocket connected');
        this.reconnectAttempts = 0;
        this.isConnecting = false;

        // Resubscribe to all channels
        this.subscriptions.forEach((_, channel) => {
          this.socket?.emit('subscribe', { channel });
        });

        resolve();
      });

      this.socket.on('disconnect', (reason) => {
        console.log('WebSocket disconnected:', reason);
        this.isConnecting = false;
      });

      this.socket.on('connect_error', (error) => {
        console.error('WebSocket connection error:', error);
        this.isConnecting = false;
        this.reconnectAttempts++;

        if (this.reconnectAttempts >= this.maxReconnectAttempts) {
          reject(new Error('Max reconnection attempts reached'));
        }
      });

      // Handle incoming messages
      this.socket.on('message', (data: any) => {
        const { channel, payload } = data;
        const handlers = this.subscriptions.get(channel);

        if (handlers) {
          handlers.forEach(handler => handler(payload));
        }
      });

      // Specific event handlers
      this.socket.on('ticker', (data) => this.handleMessage('ticker', data));
      this.socket.on('trade', (data) => this.handleMessage('trade', data));
      this.socket.on('orderbook', (data) => this.handleMessage('orderbook', data));
      this.socket.on('signal', (data) => this.handleMessage('signal', data));
      this.socket.on('agent_decision', (data) => this.handleMessage('agent_decision', data));
    });
  }

  disconnect() {
    if (this.socket) {
      this.socket.disconnect();
      this.socket = null;
    }
    this.subscriptions.clear();
  }

  subscribe(channel: string, handler: MessageHandler) {
    if (!this.subscriptions.has(channel)) {
      this.subscriptions.set(channel, new Set());

      // Send subscribe message to server
      if (this.socket?.connected) {
        this.socket.emit('subscribe', { channel });
      }
    }

    this.subscriptions.get(channel)!.add(handler);

    // Return unsubscribe function
    return () => this.unsubscribe(channel, handler);
  }

  unsubscribe(channel: string, handler: MessageHandler) {
    const handlers = this.subscriptions.get(channel);
    if (handlers) {
      handlers.delete(handler);

      if (handlers.size === 0) {
        this.subscriptions.delete(channel);

        // Send unsubscribe message to server
        if (this.socket?.connected) {
          this.socket.emit('unsubscribe', { channel });
        }
      }
    }
  }

  private handleMessage(channel: string, data: any) {
    const handlers = this.subscriptions.get(channel);
    if (handlers) {
      handlers.forEach(handler => handler(data));
    }
  }

  isConnected(): boolean {
    return this.socket?.connected || false;
  }

  // Convenience methods for common subscriptions
  subscribeTicker(symbol: string, handler: MessageHandler) {
    return this.subscribe(`ticker:${symbol}`, handler);
  }

  subscribeTrades(symbol: string, handler: MessageHandler) {
    return this.subscribe(`trades:${symbol}`, handler);
  }

  subscribeOrderBook(symbol: string, handler: MessageHandler) {
    return this.subscribe(`orderbook:${symbol}`, handler);
  }

  subscribeSignals(handler: MessageHandler) {
    return this.subscribe('signals', handler);
  }

  subscribeAgentDecisions(agentId: string, handler: MessageHandler) {
    return this.subscribe(`agent:${agentId}:decisions`, handler);
  }
}

// Export singleton instance
export const wsClient = new WebSocketClient();
export default wsClient;
