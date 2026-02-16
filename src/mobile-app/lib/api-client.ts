/**
 * Mobile API Client
 * Simplified API client for React Native
 */
import axios, { AxiosInstance } from 'axios';

const API_BASE_URL = 'http://localhost:8000'; // Update for production

class MobileAPIClient {
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

    this.client.interceptors.request.use(
      (config) => {
        if (this.apiKey) {
          config.headers['X-API-Key'] = this.apiKey;
        }
        return config;
      },
      (error) => Promise.reject(error)
    );
  }

  setApiKey(key: string) {
    this.apiKey = key;
  }

  clearApiKey() {
    this.apiKey = null;
  }

  // Market Data
  async getTicker(symbol: string) {
    const response = await this.client.get(`/api/v1/market/ticker/${symbol}`);
    return response.data;
  }

  async getOHLCV(symbol: string, interval: string, limit: number = 100) {
    const response = await this.client.get('/api/v1/market/ohlcv', {
      params: { symbol, interval, limit }
    });
    return response.data;
  }

  // Portfolio
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

  // Signals
  async getSignals(symbol?: string, limit: number = 20) {
    const response = await this.client.get('/api/v1/signals', {
      params: { symbol, limit }
    });
    return response.data;
  }

  async getSignalPerformance(timeframe: string = '24h') {
    const response = await this.client.get('/api/v1/signals/performance', {
      params: { timeframe }
    });
    return response.data;
  }

  // Risk
  async getRiskMetrics() {
    const response = await this.client.get('/api/v1/risk/metrics');
    return response.data;
  }

  // Health
  async getHealth() {
    const response = await this.client.get('/health');
    return response.data;
  }
}

export const apiClient = new MobileAPIClient();
export default apiClient;
