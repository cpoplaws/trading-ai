# Deploy NOW - Step-by-Step Guide

**Time**: 30-45 minutes
**Cost**: $0 to start (free tiers), ~$50/month when scaling

---

## 🎯 What We're Deploying:

1. **Backend API** (FastAPI) → Railway
   - Connects to Alpaca for real trading data
   - WebSocket for live updates
   - Manages strategies

2. **Frontend Dashboard** (Next.js) → Vercel
   - Beautiful professional UI
   - Real-time updates
   - Your custom domain

---

## ✅ Prerequisites (5 min):

**1. Get Alpaca API Keys** (FREE paper trading):
```
1. Go to: https://alpaca.markets/
2. Sign up (takes 2 minutes)
3. Click "Generate API Keys"
4. Select "Paper Trading" (not live!)
5. Copy both keys
```

**2. Install Required Tools**:
```bash
# Check if you have npm
npm --version

# If not installed:
# Mac: brew install node
# Or download from: https://nodejs.org/
```

---

## 🚀 PART 1: Deploy Backend (15 min)

### Step 1.1: Sign Up for Railway

```
1. Go to: https://railway.app
2. Click "Login with GitHub"
3. Authorize Railway
4. Done! ✅
```

### Step 1.2: Deploy Backend

```bash
# Open terminal
cd /Users/silasmarkowicz/trading-ai-working/apps/api

# Install Railway CLI
npm install -g @railway/cli

# Login to Railway
railway login

# Initialize project
railway init

# Follow prompts:
# - Project name: trading-ai-api
# - Select: Create new project

# Deploy!
railway up

# Wait ~2-3 minutes for deployment
```

### Step 1.3: Add Environment Variables

```bash
# Set Alpaca keys (use your paper trading keys!)
railway variables set ALPACA_API_KEY="PKxxx..."
railway variables set ALPACA_SECRET_KEY="xxx..."

# Backend will auto-restart with new variables
```

### Step 1.4: Get Your API URL

```bash
# Generate public URL
railway domain

# You'll get something like:
# https://trading-ai-api-production-xxxx.up.railway.app

# Copy this URL! You'll need it.
```

### Step 1.5: Test Backend

```bash
# Visit in browser (replace with your URL):
https://your-api.up.railway.app

# Should see:
{
  "status": "online",
  "service": "Trading AI API",
  "version": "1.0.0",
  "alpaca_connected": true
}

# ✅ Backend is live!
```

---

## 🎨 PART 2: Deploy Frontend (15 min)

### Step 2.1: Install Dependencies

```bash
cd /Users/silasmarkowicz/trading-ai-working/apps/dashboard

# Install packages
npm install

# This will take 2-3 minutes
```

### Step 2.2: Sign Up for Vercel

```
1. Go to: https://vercel.com
2. Click "Sign Up"
3. Choose "Continue with GitHub"
4. Authorize Vercel
5. Done! ✅
```

### Step 2.3: Deploy to Vercel

```bash
# Install Vercel CLI
npm install -g vercel

# Login
vercel login

# Deploy!
vercel

# Follow prompts:
# Q: Set up and deploy? YES
# Q: Which scope? (your account)
# Q: Link to existing project? NO
# Q: Project name? trading-ai-dashboard
# Q: Directory? ./ (just press enter)
# Q: Override settings? NO

# Wait ~1-2 minutes for deployment

# You'll get a URL like:
# https://trading-ai-dashboard-xxxx.vercel.app
```

### Step 2.4: Add Environment Variables

```bash
# Add your Railway API URL
vercel env add NEXT_PUBLIC_API_URL

# When prompted, enter:
https://your-api.up.railway.app

# Add WebSocket URL
vercel env add NEXT_PUBLIC_WS_URL

# When prompted, enter:
wss://your-api.up.railway.app

# Deploy production with env vars
vercel --prod
```

### Step 2.5: Test Frontend

```bash
# Visit your Vercel URL:
https://trading-ai-dashboard-xxxx.vercel.app

# You should see:
# ✅ Beautiful dashboard
# ✅ "Live" badge (if backend is running)
# ✅ Portfolio data from Alpaca
```

---

## 🌐 PART 3: Add Your Domain (10 min)

### Step 3.1: Configure Frontend Domain

**In Vercel Dashboard**:
```
1. Go to: https://vercel.com/dashboard
2. Click your project
3. Go to "Settings" → "Domains"
4. Click "Add"
5. Enter: trading.yourdomain.com
6. Vercel will show you DNS instructions
```

### Step 3.2: Configure Backend Domain

**In Railway Dashboard**:
```
1. Go to: https://railway.app/dashboard
2. Click your project
3. Click "Settings"
4. Scroll to "Domains"
5. Click "Custom Domain"
6. Enter: api.yourdomain.com
7. Railway will show you DNS instructions
```

### Step 3.3: Update DNS

**In Your Domain Provider** (GoDaddy/Namecheap/Cloudflare):
```
Add these CNAME records:

Name: trading
Type: CNAME
Value: cname.vercel-dns.com
TTL: Auto

Name: api
Type: CNAME
Value: xxxxx.up.railway.app (from Railway)
TTL: Auto
```

### Step 3.4: Wait for DNS (5-30 min)

```bash
# Check if propagated
dig trading.yourdomain.com
dig api.yourdomain.com

# When ready, visit:
https://trading.yourdomain.com

# ✅ Your custom domain is live!
```

---

## ✅ Verification Checklist

**[ ] Backend Deployed**
```bash
curl https://api.yourdomain.com
# Should return: {"status": "online"}
```

**[ ] Frontend Deployed**
```bash
# Visit: https://trading.yourdomain.com
# Should see dashboard
```

**[ ] Real Data Loading**
```bash
# Dashboard should show:
# - Your Alpaca paper trading balance
# - Real portfolio value
# - "Live" badge (green dot)
```

**[ ] WebSocket Connected**
```bash
# Dashboard should show:
# - Green "Live" badge
# - Real-time updates every 5 seconds
```

---

## 🎉 You're Live!

**Your Trading AI is now deployed at:**
- Dashboard: https://trading.yourdomain.com
- API: https://api.yourdomain.com

**What's Running**:
- ✅ Backend on Railway (always on)
- ✅ Frontend on Vercel (global CDN)
- ✅ Connected to Alpaca (paper trading)
- ✅ Real-time WebSocket updates
- ✅ Your custom domain with SSL

---

## 📊 What To Do Next:

### This Week:
1. **Monitor Paper Trading**
   - Check dashboard daily
   - Watch strategies perform
   - Look for any errors in logs

2. **Get Familiar**
   - Explore all dashboard features
   - Test different strategies
   - Understand the metrics

3. **Document Performance**
   - Take screenshots
   - Track P&L
   - Note what works

### Next Week:
1. **Go Live with Small Amount**
   - Start with $100-500
   - Use real Alpaca keys (not paper)
   - Set strict risk limits

2. **Start Building in Public**
   - Tweet your progress
   - Share screenshots
   - Build audience

### Next Month:
1. **Scale Your Trading**
   - Increase to $1k-5k
   - Optimize strategies
   - Track everything

2. **Plan SaaS Launch**
   - Create landing page
   - Set pricing
   - Get beta users

---

## 🐛 Troubleshooting:

**Problem: "Backend not connecting"**
```bash
# Check Railway logs:
railway logs

# Common issues:
# - Alpaca keys not set
# - Port not configured
# - Module not installed

# Fix: Set environment variables:
railway variables set ALPACA_API_KEY="your_key"
```

**Problem: "Frontend shows errors"**
```bash
# Check Vercel logs:
vercel logs

# Common issues:
# - API URL not set
# - Wrong API URL format
# - CORS error

# Fix: Update environment variables:
vercel env add NEXT_PUBLIC_API_URL
```

**Problem: "No real data showing"**
```
# Causes:
# - Alpaca keys are wrong
# - Using wrong Alpaca endpoint
# - Paper trading account not funded

# Fix:
# 1. Verify keys in Railway dashboard
# 2. Check Railway logs for errors
# 3. Fund paper trading account on Alpaca
```

**Problem: "Domain not working"**
```
# DNS takes time to propagate (5-60 min)
# Check status:
dig trading.yourdomain.com

# If still not working after 1 hour:
# - Verify CNAME records are correct
# - Check for conflicting A records
# - Try clearing DNS cache
```

---

## 💰 Current Costs:

**Free Tier** (Testing):
```
Railway:  $5 free credit/month
Vercel:   Generous free tier
Total:    $0 for first month!
```

**Production** (When Scaling):
```
Railway:  ~$20/month (backend + DB)
Vercel:   $20/month (Pro features)
Total:    ~$40/month
```

---

## 🎯 Success Metrics:

**Week 1**:
- [ ] Deployed successfully
- [ ] Shows real Alpaca data
- [ ] No errors for 24 hours
- [ ] WebSocket stays connected

**Week 2**:
- [ ] Paper trading running smoothly
- [ ] Dashboard shows positive P&L
- [ ] Understand all metrics
- [ ] Comfortable with platform

**Week 3-4**:
- [ ] Ready for real money
- [ ] Risk limits configured
- [ ] Monitoring set up
- [ ] First $100 deployed

---

## 🚀 You Did It!

Your professional trading AI is now LIVE with:
- ✅ Real-time data from Alpaca
- ✅ Beautiful dashboard on your domain
- ✅ WebSocket live updates
- ✅ Production-ready infrastructure
- ✅ Scalable architecture

**Next**: Watch it trade, document results, start building in public!

---

**Questions?** Check the logs:
- Railway logs: `railway logs`
- Vercel logs: `vercel logs`

**Need help?** All the code is in `/apps/api` and `/apps/dashboard`
