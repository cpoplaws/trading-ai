"""
🦄 DEX Aggregation & Arbitrage Demo
Complete demonstration of DEX aggregation, arbitrage detection, and MEV protection.
"""
import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.defi.uniswap_v3 import UniswapV3Client, WETH, USDC, DAI, USDT
from src.defi.curve_finance import CurveFinanceClient
from src.defi.arbitrage_detector import ArbitrageDetector
from src.defi.mev_protection import MEVProtector, MEVProtectionConfig


def demo_uniswap_v3():
    """Demo Uniswap V3 integration."""
    print("\n" + "="*70)
    print("🦄 DEMO 1: Uniswap V3 Integration")
    print("="*70)

    rpc_url = os.getenv('ETHEREUM_RPC_URL', 'https://eth-mainnet.g.alchemy.com/v2/demo')
    client = UniswapV3Client(rpc_url)

    # Get best quote across all fee tiers
    print("\n📊 Getting best quote for 1 ETH → USDC...")
    quote = client.get_best_quote(WETH, USDC, 1.0)

    if quote:
        print(f"✅ Best Quote:")
        print(f"   DEX: {quote['dex']}")
        print(f"   Amount In: {quote['amount_in']} ETH")
        print(f"   Amount Out: {quote['amount_out']:.2f} USDC")
        print(f"   Price: ${quote['price']:.2f} per ETH")
        print(f"   Fee Tier: {quote['fee_percent']}%")
        print(f"   Price Impact: {quote['price_impact']:.4f}%")
        print(f"   Gas Estimate: {quote['gas_estimate']} units")

        print(f"\n📊 All Fee Tiers Comparison:")
        for q in quote.get('all_quotes', []):
            print(f"   {q['fee_tier']:>5} bp ({q['fee_percent']:>5}%): {q['amount_out']:>10,.2f} USDC")

        # Calculate potential profit by choosing best tier
        worst_output = min(q['amount_out'] for q in quote['all_quotes'])
        best_output = quote['amount_out']
        savings = best_output - worst_output

        print(f"\n💰 Savings by using best fee tier: ${savings:.2f} ({(savings/best_output)*100:.3f}%)")
    else:
        print("❌ Could not get quote")


def demo_curve_finance():
    """Demo Curve Finance integration."""
    print("\n" + "="*70)
    print("🌊 DEMO 2: Curve Finance (Stablecoin Swaps)")
    print("="*70)

    rpc_url = os.getenv('ETHEREUM_RPC_URL', 'https://eth-mainnet.g.alchemy.com/v2/demo')
    client = CurveFinanceClient(rpc_url)

    # Stablecoin swap with minimal slippage
    print("\n📊 Getting quote for 10,000 USDC → DAI...")
    quote = client.quote_price(USDC, DAI, 10000.0)

    if quote:
        print(f"✅ Curve Quote:")
        print(f"   DEX: {quote['dex']}")
        print(f"   Pool: {quote['pool_name']}")
        print(f"   Amount In: {quote['amount_in']:,.2f} USDC")
        print(f"   Amount Out: {quote['amount_out']:,.2f} DAI")
        print(f"   Exchange Rate: {quote['exchange_rate']:.6f}")
        print(f"   Fee: {quote['fee_percent']}%")
        print(f"   Price Impact: {quote['price_impact']:.4f}% (very low!)")
        print(f"   Gas Estimate: {quote['gas_estimate']} units (efficient!)")

        # Compare to ideal 1:1
        comparison = client.compare_to_1_1_rate(USDC, DAI, 10000.0)
        print(f"\n📊 Compared to ideal 1:1 rate:")
        print(f"   Ideal: {comparison['ideal_rate']:.6f}")
        print(f"   Actual: {comparison['actual_rate']:.6f}")
        print(f"   Deviation: {comparison['deviation_percent']:.4f}%")
        print(f"   Loss: ${comparison['loss_to_ideal']:.4f}")
        print(f"   ✅ This is EXCELLENT for a $10k swap!")
    else:
        print("❌ Could not get quote")

    # Show pool liquidity
    print(f"\n💰 Curve 3Pool Liquidity:")
    balances = client.get_pool_balances(client.POOLS['3pool']['address'])
    for coin, balance in balances.items():
        print(f"   {coin[:6]}...{coin[-4:]}: ${balance:>15,.2f}")


def demo_cross_dex_comparison():
    """Demo comparing prices across DEXs."""
    print("\n" + "="*70)
    print("⚖️  DEMO 3: Cross-DEX Price Comparison")
    print("="*70)

    rpc_url = os.getenv('ETHEREUM_RPC_URL', 'https://eth-mainnet.g.alchemy.com/v2/demo')

    uniswap = UniswapV3Client(rpc_url)
    curve = CurveFinanceClient(rpc_url)

    # Compare USDC -> DAI on both DEXs
    amount = 5000.0
    print(f"\n📊 Comparing {amount:,.0f} USDC → DAI across DEXs...")

    uni_quote = uniswap.get_best_quote(USDC, DAI, amount)
    curve_quote = curve.quote_price(USDC, DAI, amount)

    quotes = []
    if uni_quote:
        quotes.append(uni_quote)
    if curve_quote:
        quotes.append(curve_quote)

    if quotes:
        print(f"\n{'DEX':<20} {'Output':<15} {'Fee':<10} {'Impact':<10} {'Gas':<10}")
        print("-" * 70)
        for q in quotes:
            print(f"{q['dex']:<20} {q['amount_out']:>10,.2f} DAI  {q['fee_percent']:>6}%  {q.get('price_impact', 0):>6.3f}%  {q['gas_estimate']:>6,}")

        # Find best
        best = max(quotes, key=lambda q: q['amount_out'])
        worst = min(quotes, key=lambda q: q['amount_out'])
        difference = best['amount_out'] - worst['amount_out']

        print(f"\n💰 Best DEX: {best['dex']}")
        print(f"   Saves: ${difference:.2f} ({(difference/best['amount_out'])*100:.3f}%)")
        print(f"\n✅ For stablecoins, Curve usually wins due to low fees and slippage!")
    else:
        print("❌ Could not get quotes")


def demo_arbitrage_detection():
    """Demo arbitrage opportunity detection."""
    print("\n" + "="*70)
    print("🔍 DEMO 4: Arbitrage Opportunity Detection")
    print("="*70)

    rpc_url = os.getenv('ETHEREUM_RPC_URL', 'https://eth-mainnet.g.alchemy.com/v2/demo')

    detector = ArbitrageDetector(
        rpc_url=rpc_url,
        min_profit_percent=0.3,  # 0.3% minimum profit
        max_gas_cost_usd=50.0
    )

    # Scan for opportunities
    print("\n🔍 Scanning for arbitrage opportunities...")
    print("   (Using test amount of $1,000)")

    opportunities = detector.scan_all_pairs(amount_in=1000.0)

    if opportunities:
        print(f"\n✅ Found {len(opportunities)} opportunities!")

        # Show top 3
        for i, opp in enumerate(opportunities[:3], 1):
            print(f"\n{'='*70}")
            print(f"Opportunity #{i}:")
            print(detector.format_opportunity(opp))

        # Summary
        total_profit = sum(o.profit_after_gas for o in opportunities)
        avg_profit = total_profit / len(opportunities) if opportunities else 0

        print(f"\n📊 Summary:")
        print(f"   Total Opportunities: {len(opportunities)}")
        print(f"   Total Potential Profit: ${total_profit:.2f}")
        print(f"   Average Profit: ${avg_profit:.2f}")
        print(f"   Best Profit: ${opportunities[0].profit_after_gas:.2f} ({opportunities[0].profit_percent:.2f}%)")
    else:
        print("\n❌ No profitable arbitrage opportunities found")
        print("   Reasons:")
        print("   - Market is efficient")
        print("   - Gas costs too high")
        print("   - Profit margins below threshold (0.3%)")


def demo_mev_protection():
    """Demo MEV protection strategies."""
    print("\n" + "="*70)
    print("🛡️  DEMO 5: MEV Protection")
    print("="*70)

    # Configure protection
    config = MEVProtectionConfig(
        use_flashbots=True,
        max_slippage=0.5,
        use_splitting=True,
        split_count=4,
        randomize_timing=True,
        min_delay_ms=100,
        max_delay_ms=500
    )

    protector = MEVProtector(config)

    # Protect a large swap
    print("\n🔒 Protecting a $100,000 ETH → USDC swap...")

    plan = protector.protect_swap(
        token_in="WETH",
        token_out="USDC",
        amount_in=100000.0,
        expected_output=100000.0,
        dex="uniswap_v3"
    )

    print(f"\n✅ Protection Plan Generated:")
    print(f"   Strategies: {', '.join(plan['strategies_applied'])}")
    print(f"   Order Splits: {len(plan['splits'])}")

    print(f"\n📦 Split Breakdown:")
    for i, (split, delay) in enumerate(zip(plan['splits'], plan['delays_ms'] + [0]), 1):
        print(f"   Split {i}: ${split:>12,.2f}  (wait {delay:>3}ms before next)")

    print(f"\n⚙️  Protection Settings:")
    print(f"   Flashbots: {plan['use_flashbots']}")
    print(f"   Max Slippage: {plan['max_slippage_percent']}%")
    print(f"   Expected MEV Loss: ${plan['expected_mev_loss']:.2f}")

    # Calculate savings
    unprotected_mev_loss = 100000.0 * 0.003  # 0.3% typical MEV loss
    protected_mev_loss = plan['expected_mev_loss']
    savings = unprotected_mev_loss - protected_mev_loss

    print(f"\n💰 MEV Protection Savings:")
    print(f"   Without Protection: ${unprotected_mev_loss:,.2f} loss")
    print(f"   With Protection: ${protected_mev_loss:,.2f} loss")
    print(f"   Savings: ${savings:,.2f}")
    print(f"   ✅ {(savings/unprotected_mev_loss)*100:.1f}% reduction in MEV loss!")


def demo_execution_strategy():
    """Demo optimal execution strategy."""
    print("\n" + "="*70)
    print("🎯 DEMO 6: Optimal Execution Strategy")
    print("="*70)

    print("\n📋 Step-by-Step Execution Plan:")

    print("\n1️⃣  QUOTE AGGREGATION")
    print("   - Query Uniswap V3 (all fee tiers)")
    print("   - Query Curve Finance")
    print("   - Query 1inch aggregator")
    print("   - Compare quotes")
    print("   ✅ Select best quote")

    print("\n2️⃣  ARBITRAGE CHECK")
    print("   - Scan for arbitrage opportunities")
    print("   - Check triangular arbitrage")
    print("   - Calculate net profit after gas")
    print("   ✅ Execute if profitable")

    print("\n3️⃣  MEV PROTECTION")
    print("   - Split large orders")
    print("   - Randomize timing")
    print("   - Use Flashbots relay")
    print("   - Set strict slippage limits")
    print("   ✅ Protected execution")

    print("\n4️⃣  EXECUTION")
    print("   - Approve tokens")
    print("   - Submit transactions (split if needed)")
    print("   - Monitor mempool for attacks")
    print("   - Wait for confirmations")
    print("   ✅ Transaction complete")

    print("\n5️⃣  POST-EXECUTION")
    print("   - Verify actual slippage")
    print("   - Calculate realized profit")
    print("   - Analyze MEV exposure")
    print("   - Log for future optimization")
    print("   ✅ Performance tracked")


def main():
    """Run all demos."""
    print("\n" + "="*70)
    print("🚀 DEX AGGREGATION & ARBITRAGE - COMPLETE DEMO")
    print("="*70)
    print("\nThis demo showcases:")
    print("  1. Uniswap V3 integration with fee tier optimization")
    print("  2. Curve Finance for low-slippage stablecoin swaps")
    print("  3. Cross-DEX price comparison")
    print("  4. Arbitrage opportunity detection")
    print("  5. MEV protection strategies")
    print("  6. Optimal execution workflow")

    # Run demos
    try:
        demo_uniswap_v3()
        demo_curve_finance()
        demo_cross_dex_comparison()
        demo_arbitrage_detection()
        demo_mev_protection()
        demo_execution_strategy()

        print("\n" + "="*70)
        print("✅ ALL DEMOS COMPLETE!")
        print("="*70)

        print("\n📚 Key Takeaways:")
        print("   ✅ Uniswap V3 offers multiple fee tiers - always compare!")
        print("   ✅ Curve is best for stablecoins (minimal slippage)")
        print("   ✅ Cross-DEX arbitrage exists but consider gas costs")
        print("   ✅ MEV protection is crucial for large orders")
        print("   ✅ Order splitting and timing randomization help")
        print("   ✅ Flashbots eliminates mempool exposure")

        print("\n🚀 Next Steps:")
        print("   1. Integrate with live trading system")
        print("   2. Add more DEXs (Balancer, 1inch)")
        print("   3. Implement flash loan arbitrage")
        print("   4. Build real-time monitoring dashboard")
        print("   5. Add cross-chain arbitrage")

    except Exception as e:
        print(f"\n❌ Error running demos: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()
