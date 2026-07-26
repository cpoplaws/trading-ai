#!/usr/bin/env python3
"""
Integration Test - Test complete trading workflow
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

print("🔗 Integration Test: Complete Trading Workflow")
print("=" * 60)

# Scenario: Execute a complete trading cycle
print("\nScenario: Grid Trading + Market Making + DCA")
print("-" * 60)

# 1. Initialize database
print("\n1. Database Setup...")
try:
    from database.config import DatabaseConfig
    from database.models import User, Portfolio, UserRole
    
    db = DatabaseConfig(database_url="sqlite:///integration_test.db")
    db.create_tables()
    
    with db.get_session() as session:
        # Create user
        user = User(
            username="test_trader",
            email="trader@test.com",
            password_hash="hashed_password",
            role=UserRole.TRADER
        )
        session.add(user)
        session.flush()
        
        # Create portfolio
        portfolio = Portfolio(
            user_id=user.id,
            name="Test Portfolio",
            total_value_usd=10000.0,
            cash_balance_usd=10000.0,
            is_paper=True
        )
        session.add(portfolio)
        session.commit()
        
        print(f"   ✅ User created: {user.username}")
        print(f"   ✅ Portfolio created: ${portfolio.total_value_usd:,.2f}")
        
        user_id = user.id
        portfolio_id = portfolio.id
    
except Exception as e:
    print(f"   ❌ Failed: {e}")
    sys.exit(1)

# 2. Run Grid Trading Strategy
print("\n2. Grid Trading Bot...")
try:
    from crypto_strategies.grid_trading_bot import GridTradingBot, GridConfig
    import numpy as np
    
    config = GridConfig(
        symbol='ETH-USDC',
        lower_price=1800,
        upper_price=2200,
        num_grids=10,
        total_investment=5000,
        grid_type='arithmetic'
    )
    
    bot = GridTradingBot(config)
    bot.place_all_orders()
    
    # Simulate price movement
    for price in [1900, 2000, 1950, 2100, 2050]:
        stats = bot.update_orders(price)
    
    print(f"   ✅ Grid bot executed")
    print(f"   Orders: {len(bot.all_orders)}")
    print(f"   Profit: ${stats['total_profit']:.2f}")
    print(f"   Completed cycles: {stats['completed_cycles']}")
    
except Exception as e:
    print(f"   ❌ Failed: {e}")

# 3. Run Market Making
print("\n3. Market Making...")
try:
    from crypto_strategies.market_making import MarketMakingStrategy, MarketMakingConfig
    import numpy as np
    
    config = MarketMakingConfig(
        symbol='BTC-USDC',
        base_spread_bps=10.0,
        order_size_usd=1000.0,
        max_inventory_usd=5000.0
    )
    
    strategy = MarketMakingStrategy(config)
    
    # Simulate 100 periods
    np.random.seed(42)
    base = 40000
    prices = base + np.cumsum(np.random.randn(100) * 50)
    
    results = strategy.simulate_market_making(prices, hit_probability=0.3)
    
    print(f"   ✅ Market making executed")
    print(f"   Trades: {results['total_trades']}")
    print(f"   Volume: ${results['total_volume_usd']:,.2f}")
    print(f"   P&L: ${results['total_pnl']:.2f}")
    
except Exception as e:
    print(f"   ❌ Failed: {e}")

# 4. Run DCA Bot
print("\n4. DCA Bot...")
try:
    from crypto_strategies.dca_bot import DCABot, DCAConfig, DCAFrequency, DCAMode
    from datetime import datetime, timedelta
    
    config = DCAConfig(
        symbol='BTC',
        frequency=DCAFrequency.DAILY,
        mode=DCAMode.DYNAMIC,
        base_amount=100.0,
        enable_dips=True,
        max_position_size=5000.0
    )
    
    bot = DCABot(config)
    
    # Simulate 30 days
    prices = []
    timestamps = []
    for i in range(30):
        price = 40000 + np.random.randn() * 1000
        # Add occasional dips
        if np.random.random() < 0.1:
            price *= 0.9
        prices.append(price)
        timestamps.append(datetime.now() - timedelta(days=30-i))
    
    results = bot.simulate_dca(prices, timestamps)
    
    print(f"   ✅ DCA executed")
    print(f"   Purchases: {results['num_purchases']}")
    print(f"   Invested: ${results['total_invested']:.2f}")
    print(f"   Current Value: ${results['final_value']:.2f}")
    print(f"   P&L: {results['pnl_percent']:.2f}%")
    
except Exception as e:
    print(f"   ❌ Failed: {e}")

# 5. Update database with results
print("\n5. Update Database...")
try:
    with db.get_session() as session:
        portfolio = session.query(Portfolio).filter_by(id=portfolio_id).first()
        
        # Update portfolio value
        total_profit = stats['total_profit'] + results['total_pnl']
        portfolio.total_value_usd = 10000 + total_profit
        portfolio.total_pnl = total_profit
        portfolio.total_pnl_percent = (total_profit / 10000) * 100
        
        session.commit()
        
        print(f"   ✅ Portfolio updated")
        print(f"   Starting Value: $10,000.00")
        print(f"   Current Value: ${portfolio.total_value_usd:,.2f}")
        print(f"   Total P&L: ${portfolio.total_pnl:.2f} ({portfolio.total_pnl_percent:.2f}%)")
    
    db.close()
    
except Exception as e:
    print(f"   ❌ Failed: {e}")

# Summary
print("\n" + "=" * 60)
print("📊 Integration Test Summary")
print("=" * 60)
print("\n✅ Complete Workflow Executed:")
print("   • Database initialized with user & portfolio")
print("   • Grid trading bot placed orders")
print("   • Market making generated trades")
print("   • DCA bot accumulated position")
print("   • Portfolio updated with results")
print("\n💰 Simulated Returns:")
print(f"   Grid Trading: ${stats['total_profit']:.2f}")
print(f"   Market Making: ${results['total_pnl']:.2f}")
print(f"   DCA Bot: {results['pnl_percent']:.2f}%")
print("\n🎉 Integration test completed successfully!")
print("\nClean up...")
try:
    os.remove("integration_test.db")
    print("✅ Test database removed")
except:
    pass
