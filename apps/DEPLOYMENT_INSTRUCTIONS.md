# 🚀 Deployment Instructions - Crypto AI Trading System

## Quick Deploy (Automated)

```bash
cd /Users/silasmarkowicz/trading-ai-working/apps

# Set your Alpaca API keys first
export ALPACA_API_KEY="your_alpaca_paper_key"
export ALPACA_SECRET_KEY="your_alpaca_paper_secret"

# Run deployment script
./DEPLOY.sh
```

---

## Manual Deployment (Step-by-Step)

### Prerequisites

1. **Get Alpaca API Keys** (Paper Trading)
   - Sign up at https://alpaca.markets
   - Go to Paper Trading section
   - Generate API keys
   - Save both API Key and Secret Key

2. **Install CLIs**
   ```bash
   npm install -g @railway/cli vercel
   ```

---

## Part 1: Deploy Backend to Railway

### Step 1: Prepare Backend

```bash
cd /Users/silasmarkowicz/trading-ai-working/apps/api

# Verify files exist
ls -la requirements.txt main.py Procfile railway.toml
```

### Step 2: Login to Railway

```bash
railway login
```

This will open your browser to login with GitHub.

### Step 3: Create Railway Project

```bash
# Initialize new project
railway init

# Or link to existing project
railway link
```

### Step 4: Set Environment Variables

```bash
# Set Alpaca keys
railway variables set ALPACA_API_KEY="your_key_here"
railway variables set ALPACA_SECRET_KEY="your_secret_here"

# Verify variables
railway variables
```

### Step 5: Deploy

```bash
railway up
```

### Step 6: Get Your Backend URL

```bash
# Check deployment status
railway status

# Get your URL (something like: https://xxx.railway.app)
railway domain
```

**Copy this URL - you'll need it for frontend!**

---

## Part 2: Deploy Frontend to Vercel

### Step 1: Prepare Frontend

```bash
cd /Users/silasmarkowicz/trading-ai-working/apps/dashboard

# Install dependencies if needed
npm install

# Create production environment file
echo "NEXT_PUBLIC_API_URL=https://YOUR_RAILWAY_URL.railway.app" > .env.production
```

**Replace `YOUR_RAILWAY_URL` with your actual Railway URL from Part 1!**

### Step 2: Login to Vercel

```bash
vercel login
```

### Step 3: Deploy

```bash
# Deploy to production
vercel --prod
```

Answer the prompts:
- Set up and deploy? **Y**
- Which scope? Choose your account
- Link to existing project? **N** (first time)
- Project name? `crypto-ai-dashboard` (or your choice)
- Directory? **./apps/dashboard** or **.** (if already in dashboard dir)
- Override settings? **N**

### Step 4: Get Your Frontend URL

After deployment completes, you'll see:
```
✅ Production: https://crypto-ai-dashboard.vercel.app
```

---

## Part 3: Verify Deployment

### Test Backend

```bash
# Test health endpoint
curl https://YOUR_RAILWAY_URL.railway.app/

# Expected response:
# {"status":"online","service":"Trading AI API","version":"1.0.0","alpaca_connected":true}
```

### Test Frontend

1. Open your Vercel URL in browser
2. You should see the dashboard load
3. Check that all sections appear:
   - Portfolio Stats
   - Market Intelligence
   - Trading Strategies
   - Agent Swarm
   - Recent Trades

### Enable Strategies

1. Scroll to "Trading Strategies" section
2. Click "Enable" on 1-2 strategies (start with Mean Reversion)
3. Watch the backend logs: `railway logs`
4. You should see:
   ```
   Strategy 'mean_reversion' toggled to: ENABLED
   ```

---

## Troubleshooting

### Backend Issues

**Problem**: `alpaca_connected: false`

**Solution**:
```bash
# Check environment variables
railway variables

# Re-set if missing
railway variables set ALPACA_API_KEY="your_key"
railway variables set ALPACA_SECRET_KEY="your_secret"

# Redeploy
railway up
```

**Problem**: Build fails with dependency errors

**Solution**:
```bash
cd /Users/silasmarkowicz/trading-ai-working/apps/api

# Update requirements
pip freeze > requirements-frozen.txt

# Use minimal requirements
cat > requirements.txt << EOF
fastapi>=0.110.0
uvicorn[standard]>=0.27.0
alpaca-py>=0.21.0
numpy>=1.24.0
scikit-learn>=1.3.0
pandas>=2.0.0
python-dotenv>=1.0.0
EOF

# Redeploy
railway up
```

### Frontend Issues

**Problem**: API calls failing (CORS or connection errors)

**Solution**: Update CORS in backend
```python
# In apps/api/main.py, update CORS origins:
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://your-vercel-url.vercel.app"],  # Your actual domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

**Problem**: Environment variable not set

**Solution**:
```bash
# In Vercel dashboard: Settings → Environment Variables
# Add: NEXT_PUBLIC_API_URL = https://your-railway-url.railway.app

# Or via CLI:
vercel env add NEXT_PUBLIC_API_URL production
# Enter value: https://your-railway-url.railway.app

# Redeploy
vercel --prod
```

---

## Monitoring

### Watch Backend Logs

```bash
railway logs
```

You should see:
```
🚀 Trading AI API starting up...
✅ Connected to Alpaca API
Initializing Strategy Runner...
✅ Registered 5 CRYPTO strategies
   🔗 Primary: Base Network
   🔗 Secondary: Solana
```

### Monitor Strategies

```bash
# Watch strategy execution
railway logs --follow | grep "Strategy"
```

### Check Trades

```bash
# View trade executions
railway logs --follow | grep "Trade"
```

---

## Production Checklist

Before going live with real money:

- [ ] Backend deployed and accessible
- [ ] Frontend deployed and connected to backend
- [ ] Alpaca API keys working (paper trading)
- [ ] Strategies can be enabled/disabled
- [ ] Logs showing strategy execution
- [ ] Agent swarm responding
- [ ] Market intelligence updating
- [ ] **Test with paper trading for at least 1 week**
- [ ] Monitor win rate and P&L
- [ ] Verify risk management working (stop loss, position limits)
- [ ] Only then consider real money (start small!)

---

## Updating After Deployment

### Update Backend

```bash
cd /Users/silasmarkowicz/trading-ai-working/apps/api

# Make your changes to code
# Then redeploy:
railway up
```

### Update Frontend

```bash
cd /Users/silasmarkowicz/trading-ai-working/apps/dashboard

# Make your changes
# Then redeploy:
vercel --prod
```

---

## Environment Variables Reference

### Backend (Railway)
```
ALPACA_API_KEY=pk_xxxxxxxxxxxxx
ALPACA_SECRET_KEY=xxxxxxxxxxxxx
PORT=8000 (auto-set by Railway)
```

### Frontend (Vercel)
```
NEXT_PUBLIC_API_URL=https://your-backend.railway.app
```

---

## Cost Estimates

### Railway (Backend)
- **Free Tier**: $5 credit/month
- **Hobby Plan**: $5/month (recommended)
- Includes: 500 hours, 512MB RAM, 1GB disk

### Vercel (Frontend)
- **Free Tier**: Perfect for this project
- Unlimited bandwidth
- 100GB bandwidth included

**Total Cost**: ~$5/month for hobby use

---

## Security Best Practices

1. **Never commit API keys** to git
2. **Use paper trading** first (Alpaca paper keys)
3. **Enable Railway IP restrictions** (Settings → Networking)
4. **Add Vercel deployment protection** (Settings → Deployment Protection)
5. **Monitor logs regularly** for suspicious activity
6. **Set up alerts** for large losses (add to strategy_runner.py)

---

## Support

If you encounter issues:

1. **Check logs**: `railway logs` or Vercel dashboard
2. **Test locally first**: Run both backend and frontend locally
3. **Verify environment variables**: Railway and Vercel dashboards
4. **Check API docs**: `https://your-backend.railway.app/docs`

---

## Success! 🎉

Once deployed, you should have:

✅ Backend running on Railway
✅ Frontend running on Vercel
✅ 5 crypto strategies ready
✅ 4 AI agents active
✅ Market intelligence updating
✅ Real-time dashboard

**Start trading crypto with AI!** 🚀

---

## Quick Links

- Railway Dashboard: https://railway.app/dashboard
- Vercel Dashboard: https://vercel.com/dashboard
- Alpaca Paper Trading: https://app.alpaca.markets/paper/dashboard/overview
- API Documentation: https://your-backend.railway.app/docs
- Frontend: https://your-frontend.vercel.app

---

**Remember**: Start with paper trading, monitor performance, and only use real money when confident!
