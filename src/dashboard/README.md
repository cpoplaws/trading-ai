# 📊 Real-time Trading Dashboard

React-based dashboard for real-time trading data visualization.

## Features

- **Live Price Updates**: Real-time price charts via WebSocket
- **Portfolio Tracking**: Live P&L and balance updates
- **Trade Notifications**: Instant trade execution alerts
- **ML Predictions**: Live ML model predictions
- **Pattern Alerts**: Real-time pattern detection notifications
- **Performance Analytics**: Charts and metrics

## Tech Stack

- **React 18**: UI framework
- **WebSocket**: Real-time data
- **Chart.js / Recharts**: Data visualization
- **TailwindCSS**: Styling
- **TypeScript**: Type safety

## Setup

```bash
# Install dependencies
cd src/dashboard
npm install

# Start development server
npm start

# Build for production
npm run build
```

## Dashboard Components

### 1. LivePriceChart
Real-time price chart with WebSocket updates.

### 2. PortfolioSummary
Live portfolio value, P&L, and balances.

### 3. TradeHistory
Recent trades with execution details.

### 4. MLPredictions
Live ML model predictions and confidence.

### 5. PatternDetector
Real-time pattern detection alerts.

### 6. AlertCenter
Notification center for all alerts.

## WebSocket Connection

```typescript
// Connect to WebSocket server
const ws = new WebSocket('ws://localhost:8765');

// Handle messages
ws.onmessage = (event) => {
  const message = JSON.parse(event.data);

  switch (message.type) {
    case 'price_update':
      updatePriceChart(message.data);
      break;
    case 'portfolio_update':
      updatePortfolio(message.data);
      break;
    case 'trade_executed':
      showTradeNotification(message.data);
      break;
    case 'ml_prediction':
      updatePredictions(message.data);
      break;
  }
};
```

## File Structure

```
src/dashboard/
├── public/
│   └── index.html
├── src/
│   ├── components/
│   │   ├── LivePriceChart.tsx
│   │   ├── PortfolioSummary.tsx
│   │   ├── TradeHistory.tsx
│   │   ├── MLPredictions.tsx
│   │   ├── PatternDetector.tsx
│   │   └── AlertCenter.tsx
│   ├── hooks/
│   │   ├── useWebSocket.ts
│   │   └── useTradingData.ts
│   ├── services/
│   │   └── api.ts
│   ├── types/
│   │   └── index.ts
│   ├── App.tsx
│   └── index.tsx
├── package.json
└── tsconfig.json
```

## Example Component

```typescript
// LivePriceChart.tsx
import React, { useEffect, useState } from 'react';
import { Line } from 'react-chartjs-2';
import { useWebSocket } from '../hooks/useWebSocket';

export const LivePriceChart: React.FC = () => {
  const [prices, setPrices] = useState<number[]>([]);
  const { message } = useWebSocket();

  useEffect(() => {
    if (message?.type === 'price_update') {
      setPrices(prev => [...prev.slice(-100), message.data.price]);
    }
  }, [message]);

  const data = {
    labels: prices.map((_, i) => i),
    datasets: [{
      label: 'Price',
      data: prices,
      borderColor: 'rgb(75, 192, 192)',
      tension: 0.1
    }]
  };

  return (
    <div className="bg-white p-4 rounded-lg shadow">
      <h2 className="text-xl font-bold mb-4">Live Price</h2>
      <Line data={data} />
    </div>
  );
};
```

## API Endpoints

The dashboard connects to:
- WebSocket: `ws://localhost:8765`
- REST API: `http://localhost:8000/api`

## Environment Variables

Create `.env` file:

```env
REACT_APP_WS_URL=ws://localhost:8765
REACT_APP_API_URL=http://localhost:8000/api
```

## Development

```bash
# Start WebSocket server
python3 -m src.websocket.server

# Start REST API
cd api && uvicorn main:app --reload

# Start dashboard
cd src/dashboard && npm start
```

## Production Build

```bash
# Build dashboard
npm run build

# Serve with nginx/apache
# Or deploy to Vercel/Netlify
```

## Screenshots

(Add screenshots here)

## TODO

- [ ] Add authentication
- [ ] Implement dark mode
- [ ] Add more chart types
- [ ] Mobile responsive design
- [ ] Export data functionality
- [ ] Customizable dashboard layouts
