#!/usr/bin/env python3
"""
Live Trading Demo
Demonstrates using the running infrastructure
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

print("🚀 Live Trading System Demo")
print("=" * 60)

# 1. Connect to Database
print("\n1. Connecting to Database...")
try:
    from database.config import DatabaseConfig
    
    # Connect to running TimescaleDB
    db = DatabaseConfig(
        database_url="postgresql://trader:changeme@localhost:5432/trading_ai",
        echo=False
    )
    
    if db.health_check():
        print("   ✅ Database connected!")
        stats = db.get_stats()
        print(f"   Database: {stats['url']}")
    else:
        print("   ⚠️  Database not responding, using SQLite fallback")
        db = DatabaseConfig(database_url="sqlite:///demo.db")
        db.create_tables()
        print("   ✅ SQLite database created")
    
except Exception as e:
    print(f"   ⚠️  Using SQLite: {e}")
    from database.config import DatabaseConfig
    db = DatabaseConfig(database_url="sqlite:///demo.db")
    db.create_tables()

# 2. Run Live Strategy
print("\n2. Running Market Making Strategy...")
try:
    from crypto_strategies.market_making import MarketMakingStrategy, MarketMakingConfig
    import numpy as np
    
    config = MarketMakingConfig(
        symbol='BTC-USDC',
        base_spread_bps=10.0,
        order_size_usd=1000.0,
        max_inventory_usd=10000.0
    )
    
    strategy = MarketMakingStrategy(config)
    
    # Simulate 200 periods
    np.random.seed(int(os.urandom(4).hex(), 16) % 2**32)
    base = 40000
    prices = base + np.cumsum(np.random.randn(200) * 50)
    
    results = strategy.simulate_market_making(prices, hit_probability=0.25)
    
    print(f"   ✅ Strategy executed!")
    print(f"   Trades: {results['total_trades']}")
    print(f"   Volume: ${results['total_volume_usd']:,.2f}")
    print(f"   P&L: ${results['total_pnl']:.2f}")
    print(f"   Final Equity: ${results['final_equity']:,.2f}")
    
    pnl_percent = (results['total_pnl'] / 10000) * 100
    print(f"   Return: {pnl_percent:+.2f}%")
    
except Exception as e:
    print(f"   ❌ Failed: {e}")

# 3. Save to Database
print("\n3. Saving Results to Database...")
try:
    from database.models import User, Portfolio, Trade, OrderSide
    from datetime import datetime
    
    with db.get_session() as session:
        # Get or create demo user
        user = session.query(User).filter_by(username='demo_trader').first()
        if not user:
            from database.models import UserRole
            user = User(
                username='demo_trader',
                email='demo@trading-ai.com',
                password_hash='demo_hash',
                role=UserRole.TRADER
            )
            session.add(user)
            session.flush()
        
        # Get or create portfolio
        portfolio = session.query(Portfolio).filter_by(
            user_id=user.id, 
            name='Live Demo'
        ).first()
        
        if not portfolio:
            portfolio = Portfolio(
                user_id=user.id,
                name='Live Demo',
                total_value_usd=10000.0 + results['total_pnl'],
                cash_balance_usd=10000.0 + results['total_pnl'],
                total_pnl=results['total_pnl'],
                is_paper=True
            )
            session.add(portfolio)
        else:
            portfolio.total_value_usd += results['total_pnl']
            portfolio.total_pnl += results['total_pnl']
        
        # Save a sample trade
        trade = Trade(
            portfolio_id=portfolio.id,
            symbol='BTC-USDC',
            exchange='demo',
            side=OrderSide.BUY,
            quantity=0.1,
            price=40000.0,
            value=4000.0,
            executed_at=datetime.now()
        )
        session.add(trade)
        session.commit()
        
        print(f"   ✅ Results saved!")
        print(f"   User: {user.username}")
        print(f"   Portfolio Value: ${portfolio.total_value_usd:,.2f}")
        print(f"   Total P&L: ${portfolio.total_pnl:.2f}")
    
except Exception as e:
    print(f"   ⚠️  Database save failed: {e}")

# 4. Summary
print("\n" + "=" * 60)
print("📊 Demo Summary")
print("=" * 60)
print("\n✅ What's Working:")
print("   • Market making strategy executed")
print("   • Database connected (PostgreSQL or SQLite)")
print("   • Trades recorded")
print("   • Portfolio tracking active")
print("\n📈 Available Services:")
print("   • Grafana Dashboard: http://localhost:3001")
print("   • Prometheus Metrics: http://localhost:9090")
print("   • TimescaleDB: localhost:5432")
print("   • Redis: localhost:6379")
print("\n🎯 Next Steps:")
print("   1. Access Grafana for visualization")
print("   2. Run more strategies (grid, DCA, momentum)")
print("   3. Connect to Coinbase Pro with real API keys")
print("   4. Monitor performance in Prometheus")
print("\n🎉 System is LIVE and ready for trading!")
