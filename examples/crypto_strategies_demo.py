"""
🚀 Comprehensive Crypto Strategies Demo
Demonstrates all advanced crypto trading strategies.
"""
import sys
from pathlib import Path
import numpy as np
from datetime import datetime, timedelta

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.crypto_strategies.grid_trading_bot import GridTradingBot, GridConfig, GridBacktester
from src.crypto_strategies.liquidation_hunter import LiquidationHunter
from src.crypto_strategies.whale_follower import WhaleFollower, WhaleWallet, WhaleTransaction
from src.crypto_strategies.yield_optimizer import YieldOptimizer, YieldOpportunity


def demo_grid_trading():
    """Demo grid trading strategy."""
    print("\n" + "="*70)
    print("📊 DEMO 1: Grid Trading Bot")
    print("="*70)

    # Create sample price data
    base_price = 45000
    volatility = 2000
    num_points = 500

    price_data = []
    for i in range(num_points):
        noise = np.random.normal(0, volatility * 0.3)
        oscillation = volatility * np.sin(i / 50)
        price = base_price + oscillation + noise
        price_data.append(price)

    print(f"\n📈 Simulating {num_points} BTC price points")
    print(f"   Range: ${min(price_data):,.0f} - ${max(price_data):,.0f}")

    # Configure grid
    config = GridConfig(
        symbol='BTC/USDT',
        lower_price=43000,
        upper_price=47000,
        num_grids=10,
        total_investment=10000,
        grid_type='arithmetic'
    )

    print(f"\n⚙️  Grid Configuration:")
    print(f"   Range: ${config.lower_price:,.0f} - ${config.upper_price:,.0f}")
    print(f"   Grids: {config.num_grids}")
    print(f"   Investment: ${config.total_investment:,.0f}")

    # Run backtest
    backtester = GridBacktester(price_data, config)
    results = backtester.run()

    print(f"\n✅ Results:")
    print(f"   Total Profit: ${results['total_profit']:,.2f}")
    print(f"   ROI: {results['profit_percentage']:.2f}%")
    print(f"   Trades: {results['trades_count']}")
    print(f"   Avg Profit/Trade: ${results['avg_profit_per_trade']:.2f}")

    return results


def demo_liquidation_hunter():
    """Demo liquidation hunting strategy."""
    print("\n" + "="*70)
    print("🎯 DEMO 2: Liquidation Hunter")
    print("="*70)

    hunter = LiquidationHunter(
        min_liquidation_size=1_000_000,
        min_confidence=0.6
    )

    # Simulate market data
    market_data = {
        'timestamp': datetime.now().timestamp(),
        'price': 45000.0,
        'open_interest': 2_000_000_000,
        'long_short_ratio': 1.5,
        'funding_rate': 0.0002,
        'recent_liquidations': [
            {'timestamp': datetime.now().timestamp() - i, 'amount_usd': 100000 * (1 + i/10), 'side': 'long'}
            for i in range(20)
        ]
    }

    print(f"\n📊 Market Data:")
    print(f"   Price: ${market_data['price']:,.0f}")
    print(f"   Open Interest: ${market_data['open_interest']:,.0f}")
    print(f"   Long/Short: {market_data['long_short_ratio']:.2f}")
    print(f"   Funding Rate: {market_data['funding_rate']:.4f}%")

    # Analyze
    alert = hunter.monitor_and_alert('BTC/USDT', market_data)

    print(f"\n✅ Analysis:")
    print(f"   Clusters Found: {alert['clusters_found']}")

    if alert['signal']:
        signal = alert['signal']
        print(f"\n🚨 SIGNAL:")
        print(f"   Action: {signal.action.upper()}")
        print(f"   Entry: ${signal.entry_price:,.2f}")
        print(f"   Target: ${signal.take_profit:,.2f}")
        print(f"   Est. Profit: {signal.estimated_profit_percent:.2f}%")
        print(f"   Confidence: {signal.confidence:.0%}")

    return alert


def demo_whale_follower():
    """Demo whale following strategy."""
    print("\n" + "="*70)
    print("🐋 DEMO 3: Whale Follower")
    print("="*70)

    follower = WhaleFollower(min_whale_count=3, min_confidence=0.65)

    # Add whale wallets
    whales = [
        WhaleWallet(
            address="0x1234",
            label="Jump Trading",
            total_value_usd=500_000_000,
            success_rate=0.75,
            avg_hold_time_hours=120,
            tokens_held={'ETH': 50000}
        ),
        WhaleWallet(
            address="0xabcd",
            label="3AC",
            total_value_usd=200_000_000,
            success_rate=0.68,
            avg_hold_time_hours=72,
            tokens_held={'ETH': 30000}
        ),
        WhaleWallet(
            address="0x9876",
            label="DWF Labs",
            total_value_usd=150_000_000,
            success_rate=0.82,
            avg_hold_time_hours=48,
            tokens_held={'ETH': 25000}
        ),
        WhaleWallet(
            address="0xfed",
            label="Alameda",
            total_value_usd=1_000_000_000,
            success_rate=0.71,
            avg_hold_time_hours=96,
            tokens_held={'ETH': 100000}
        ),
    ]

    for whale in whales:
        follower.add_whale_wallet(whale)

    # Simulate transactions
    for i, whale in enumerate(whales):
        tx = WhaleTransaction(
            tx_hash=f"0x{i}",
            timestamp=datetime.now() - timedelta(hours=i*2),
            whale_address=whale.address,
            token="ETH",
            action="buy",
            amount=300 + i*100,
            usd_value=(300 + i*100) * 2000,
            exchange="Binance"
        )
        follower.record_transaction(tx)

    print(f"\n👥 Tracking {len(whales)} whale wallets")

    # Generate signal
    signal = follower.generate_signal("ETH", 2000.0)

    if signal:
        print(f"\n✅ WHALE SIGNAL:")
        print(f"   Action: {signal.action.upper()}")
        print(f"   Confidence: {signal.confidence:.0%}")
        print(f"   Whales: {signal.num_whales}")
        print(f"   Volume: ${signal.total_volume_usd:,.0f}")
        print(f"   Hold Time: {signal.suggested_hold_time_hours:.0f}h")
        print(f"   Reasoning: {signal.reasoning}")

    return signal


def demo_yield_optimizer():
    """Demo yield optimization."""
    print("\n" + "="*70)
    print("🌾 DEMO 4: Yield Optimizer")
    print("="*70)

    optimizer = YieldOptimizer(min_apy=5.0)

    # Add opportunities
    opportunities = [
        YieldOpportunity(
            protocol='aave', pool_name='USDC', token='USDC',
            apy=3.5, tvl=500_000_000, risk_score=0.15,
            il_risk=False, lockup_period=0, auto_compound=True,
            rewards_token='AAVE', gas_cost_usd=25
        ),
        YieldOpportunity(
            protocol='curve', pool_name='3pool', token='3CRV',
            apy=8.2, tvl=1_200_000_000, risk_score=0.2,
            il_risk=False, lockup_period=0, auto_compound=False,
            rewards_token='CRV', gas_cost_usd=30
        ),
        YieldOpportunity(
            protocol='convex', pool_name='cvxCRV', token='cvxCRV',
            apy=12.5, tvl=800_000_000, risk_score=0.3,
            il_risk=False, lockup_period=0, auto_compound=True,
            rewards_token='CVX', gas_cost_usd=35
        ),
    ]

    for opp in opportunities:
        optimizer.add_opportunity(opp)

    capital = 10000
    print(f"\n💰 Optimizing ${capital:,.0f} capital")

    portfolio = optimizer.create_portfolio(capital)

    print(f"\n✅ Optimized Portfolio:")
    print(f"   Expected APY: {portfolio.total_apy:.2f}%")
    print(f"   Monthly Return: ${portfolio.total_monthly_return:.2f}")
    print(f"   Risk Score: {portfolio.avg_risk_score:.2f}")
    print(f"\n📊 Top Allocations:")

    for i, allocation in enumerate(portfolio.allocations[:3], 1):
        opp = allocation.opportunity
        amount = capital * (allocation.allocation_percent / 100)
        print(f"   {i}. {opp.protocol} - {opp.pool_name}: {allocation.allocation_percent:.1f}% (${amount:,.0f})")

    return portfolio


def comparison_summary():
    """Compare all strategies."""
    print("\n" + "="*70)
    print("⚖️  STRATEGY COMPARISON")
    print("="*70)

    strategies = {
        'Grid Trading': {
            'best_for': 'Range-bound markets',
            'risk': 'Low-Medium',
            'time_horizon': 'Days-Weeks',
            'capital_required': '$1k-$10k',
            'expected_return': '5-20% monthly',
            'complexity': 'Medium'
        },
        'Liquidation Hunter': {
            'best_for': 'High volatility events',
            'risk': 'High',
            'time_horizon': 'Minutes-Hours',
            'capital_required': '$5k-$50k',
            'expected_return': '2-10% per trade',
            'complexity': 'High'
        },
        'Whale Follower': {
            'best_for': 'Smart money tracking',
            'risk': 'Medium',
            'time_horizon': 'Days-Weeks',
            'capital_required': '$1k-$100k',
            'expected_return': '10-30% per signal',
            'complexity': 'Medium'
        },
        'Yield Optimizer': {
            'best_for': 'Passive income',
            'risk': 'Low-Medium',
            'time_horizon': 'Weeks-Months',
            'capital_required': '$1k-$1M',
            'expected_return': '5-50% APY',
            'complexity': 'Low'
        },
    }

    print(f"\n{'Strategy':<20} {'Best For':<25} {'Risk':<12} {'Return':<20}")
    print("-" * 90)

    for name, info in strategies.items():
        print(f"{name:<20} {info['best_for']:<25} {info['risk']:<12} {info['expected_return']:<20}")

    print(f"\n💡 Recommendation:")
    print(f"   Beginners: Start with Yield Optimizer (lowest risk)")
    print(f"   Intermediate: Grid Trading + Whale Follower")
    print(f"   Advanced: Liquidation Hunter (requires fast execution)")
    print(f"   Best: Combine all strategies for diversification!")


def main():
    """Run all demos."""
    print("\n" + "="*70)
    print("🚀 ADVANCED CRYPTO STRATEGIES - COMPLETE DEMO")
    print("="*70)

    results = {}

    try:
        # Run all demos
        results['grid'] = demo_grid_trading()
        results['liquidation'] = demo_liquidation_hunter()
        results['whale'] = demo_whale_follower()
        results['yield'] = demo_yield_optimizer()

        # Comparison
        comparison_summary()

        print("\n" + "="*70)
        print("✅ ALL DEMOS COMPLETE!")
        print("="*70)

        print("\n📚 Summary:")
        print(f"   Grid Trading ROI: {results['grid']['profit_percentage']:.2f}%")
        print(f"   Liquidation Signals: {1 if results['liquidation']['signal'] else 0}")
        print(f"   Whale Signals: {1 if results['whale'] else 0}")
        print(f"   Yield APY: {results['yield'].total_apy:.2f}%")

        print("\n🚀 Next Steps:")
        print("   1. Integrate with live data feeds")
        print("   2. Connect to exchange APIs")
        print("   3. Implement automated execution")
        print("   4. Add real-time monitoring dashboard")
        print("   5. Build risk management system")

    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()
