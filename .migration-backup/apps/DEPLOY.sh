#!/bin/bash
# Crypto AI Trading System - Deployment Script

echo "🚀 Crypto AI Trading System - Deployment"
echo "========================================"
echo ""

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Check if Railway CLI is installed
if ! command -v railway &> /dev/null; then
    echo -e "${YELLOW}Railway CLI not found. Installing...${NC}"
    npm install -g @railway/cli
fi

# Check if Vercel CLI is installed
if ! command -v vercel &> /dev/null; then
    echo -e "${YELLOW}Vercel CLI not found. Installing...${NC}"
    npm install -g vercel
fi

echo ""
echo -e "${BLUE}Step 1: Backend Deployment (Railway)${NC}"
echo "========================================"
cd api

# Check for Alpaca keys
if [ -z "$ALPACA_API_KEY" ] || [ -z "$ALPACA_SECRET_KEY" ]; then
    echo -e "${RED}⚠️  Alpaca API keys not found!${NC}"
    echo ""
    echo "Please set your Alpaca API keys:"
    echo "  export ALPACA_API_KEY='your_key'"
    echo "  export ALPACA_SECRET_KEY='your_secret'"
    echo ""
    read -p "Press Enter after setting keys, or Ctrl+C to exit..."
fi

echo -e "${GREEN}✅ Alpaca keys configured${NC}"
echo ""

# Initialize Railway project
echo "Initializing Railway project..."
railway login
railway init

# Set environment variables
echo "Setting environment variables..."
railway variables set ALPACA_API_KEY="$ALPACA_API_KEY"
railway variables set ALPACA_SECRET_KEY="$ALPACA_SECRET_KEY"

# Deploy to Railway
echo "Deploying to Railway..."
railway up

# Get Railway URL
RAILWAY_URL=$(railway status --json | jq -r '.url')
echo -e "${GREEN}✅ Backend deployed to: $RAILWAY_URL${NC}"
echo ""

# Step 2: Frontend Deployment
echo ""
echo -e "${BLUE}Step 2: Frontend Deployment (Vercel)${NC}"
echo "========================================"
cd ../dashboard

# Set API URL
echo "NEXT_PUBLIC_API_URL=$RAILWAY_URL" > .env.production

echo "Deploying to Vercel..."
vercel --prod

echo ""
echo -e "${GREEN}✅✅✅ DEPLOYMENT COMPLETE! ✅✅✅${NC}"
echo ""
echo "🎉 Your Crypto AI Trading System is now live!"
echo ""
echo "📊 Backend API: $RAILWAY_URL"
echo "🌐 Frontend Dashboard: (check Vercel output above)"
echo ""
echo "Next steps:"
echo "  1. Open your dashboard URL"
echo "  2. Enable strategies from the UI"
echo "  3. Watch your AI agents trade crypto!"
echo ""
