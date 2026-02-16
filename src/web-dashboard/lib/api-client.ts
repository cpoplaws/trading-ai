/**
 * API Client for Trading AI Backend
 * Handles authentication and requests to FastAPI backend
 */
import axios, { AxiosInstance, AxiosRequestConfig } from 'axios';

const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';

class APIClient {
  private client: AxiosInstance;
  private apiKey: string | null = null;

  constructor() {
    this.client = axios.create({
      baseURL: API_BASE_URL,
      timeout: 30000,
      headers: {
        'Content-Type': 'application/json',
      },
    });

    // Request interceptor to add API key
    this.client.interceptors.request.use(
      (config) => {
        if (this.apiKey) {
          config.headers['X-API-Key'] = this.apiKey;
        }
        return config;
      },
      (error) => Promise.reject(error)
    );

    // Response interceptor for error handling
    this.client.interceptors.response.use(
      (response) => response,
      (error) => {
        if (error.response?.status === 401) {
          console.error('Unauthorized - Invalid API key');
          // Clear stored API key
          this.apiKey = null;
          if (typeof window !== 'undefined') {
            localStorage.removeItem('api_key');
          }
        }
        return Promise.reject(error);
      }
    );

    // Load API key from localStorage
    if (typeof window !== 'undefined') {
      this.apiKey = localStorage.getItem('api_key');
    }
  }

  setApiKey(key: string) {
    this.apiKey = key;
    if (typeof window !== 'undefined') {
      localStorage.setItem('api_key', key);
    }
  }

  clearApiKey() {
    this.apiKey = null;
    if (typeof window !== 'undefined') {
      localStorage.removeItem('api_key');
    }
  }

  // ==========================================
  // Market Data Endpoints
  // ==========================================

  async getOHLCV(symbol: string, interval: string, limit: number = 100) {
    const response = await this.client.get('/api/v1/market/ohlcv', {
      params: { symbol, interval, limit }
    });
    return response.data;
  }

  async getTicker(symbol: string) {
    const response = await this.client.get(`/api/v1/market/ticker/${symbol}`);
    return response.data;
  }

  async getOrderBook(symbol: string, limit: number = 20) {
    const response = await this.client.get('/api/v1/market/orderbook', {
      params: { symbol, limit }
    });
    return response.data;
  }

  async getTrades(symbol: string, limit: number = 50) {
    const response = await this.client.get('/api/v1/market/trades', {
      params: { symbol, limit }
    });
    return response.data;
  }

  // ==========================================
  // Risk Management Endpoints
  // ==========================================

  async calculateVaR(returns: number[], method: string = 'historical', portfolioValue: number = 100000) {
    const response = await this.client.post('/api/v1/risk/var', {
      returns,
      method,
      portfolio_value: portfolioValue
    });
    return response.data;
  }

  async getRiskMetrics() {
    const response = await this.client.get('/api/v1/risk/metrics');
    return response.data;
  }

  async getPositionSize(symbol: string, entryPrice: number, stopLoss: number, method: string = 'fixed_risk') {
    const response = await this.client.post('/api/v1/risk/position-size', {
      symbol,
      entry_price: entryPrice,
      stop_loss: stopLoss,
      method
    });
    return response.data;
  }

  // ==========================================
  // Trading Signals Endpoints
  // ==========================================

  async getSignals(symbol?: string, limit: number = 20) {
    const response = await this.client.get('/api/v1/signals', {
      params: { symbol, limit }
    });
    return response.data;
  }

  async getLatestSignal(symbol: string) {
    const response = await this.client.get(`/api/v1/signals/${symbol}/latest`);
    return response.data;
  }

  async getSignalPerformance(timeframe: string = '24h') {
    const response = await this.client.get('/api/v1/signals/performance', {
      params: { timeframe }
    });
    return response.data;
  }

  // ==========================================
  // Portfolio Endpoints
  // ==========================================

  async getPortfolio() {
    const response = await this.client.get('/api/v1/portfolio');
    return response.data;
  }

  async getPortfolioValue() {
    const response = await this.client.get('/api/v1/portfolio/value');
    return response.data;
  }

  async getPortfolioStats() {
    const response = await this.client.get('/api/v1/portfolio/stats');
    return response.data;
  }

  async getPositions() {
    const response = await this.client.get('/api/v1/portfolio/positions');
    return response.data;
  }

  // ==========================================
  // RL Agents Endpoints
  // ==========================================

  async getAgents() {
    const response = await this.client.get('/api/v1/agents');
    return response.data;
  }

  async getAgentDecisions(agentId: string, limit: number = 20) {
    const response = await this.client.get(`/api/v1/agents/${agentId}/decisions`, {
      params: { limit }
    });
    return response.data;
  }

  async getAgentPerformance(agentId: string) {
    const response = await this.client.get(`/api/v1/agents/${agentId}/performance`);
    return response.data;
  }

  // ==========================================
  // Health Check
  // ==========================================

  async getHealth() {
    const response = await this.client.get('/health');
    return response.data;
  }

  async getHealthDetailed() {
    const response = await this.client.get('/health/detailed');
    return response.data;
  }
}

// Export singleton instance
export const apiClient = new APIClient();
export default apiClient;
