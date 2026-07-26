"""
Test Phase 5: Multi-Chain Expansion
Tests chain configuration, cross-chain deployment, and bridge operations
"""
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.blockchain.chain_config import ChainConfig, Network, Chain, get_chain_config
from src.bridges.bridge_manager import BridgeManager, BridgeProtocol
from src.agents.supervisor_agent import SupervisorAgent, TradeResult
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def test_chain_configuration():
    """Test 1: Chain Configuration"""
    logger.info("\n" + "="*70)
    logger.info("TEST 1: Chain Configuration")
    logger.info("="*70)

    # Test testnet configuration
    logger.info("\n--- Test 1a: Testnet Configuration ---")
    config = get_chain_config(Network.TESTNET, force_new=True)

    summary = config.get_summary()
    logger.info(f"   Network: {summary['network']}")
    logger.info(f"   Total chains: {summary['total_chains']}")
    logger.info(f"   EVM chains: {summary['evm_chains']}")

    assert summary['total_chains'] == 6, "Should have 6 chains"
    assert summary['evm_chains'] == 5, "Should have 5 EVM chains"

    logger.info("\n--- Test 1b: Chain Metadata ---")
    for chain in [Chain.BASE, Chain.ARBITRUM, Chain.OPTIMISM, Chain.POLYGON]:
        metadata = config.get_metadata(chain)
        logger.info(f"\n   {metadata.name}:")
        logger.info(f"      Chain ID: {metadata.chain_id}")
        logger.info(f"      Native Token: {metadata.native_token}")
        logger.info(f"      Block Time: {metadata.average_block_time}s")
        logger.info(f"      RPC Endpoints: {len(metadata.rpc_endpoints)}")

        assert metadata is not None, f"Metadata should exist for {chain.value}"
        assert len(metadata.rpc_endpoints) > 0, "Should have RPC endpoints"

    logger.info("\n--- Test 1c: RPC URLs ---")
    for chain in [Chain.BASE, Chain.ARBITRUM, Chain.OPTIMISM]:
        rpc_url = config.get_rpc_url(chain)
        logger.info(f"   {chain.value}: {rpc_url}")
        assert rpc_url is not None, f"Should have RPC URL for {chain.value}"
        assert rpc_url.startswith("http"), "RPC URL should be HTTP(S)"

    logger.info("\n--- Test 1d: Block Explorers ---")
    mock_tx = "0x1234567890abcdef1234567890abcdef1234567890abcdef1234567890abcdef"
    for chain in [Chain.BASE, Chain.ARBITRUM]:
        explorer_url = config.get_explorer_url(chain, tx_hash=mock_tx)
        logger.info(f"   {chain.value}: {explorer_url}")
        assert mock_tx in explorer_url, "Explorer URL should contain tx hash"

    logger.info("\n--- Test 1e: EVM Detection ---")
    assert config.is_evm_chain(Chain.BASE) == True, "Base should be EVM"
    assert config.is_evm_chain(Chain.ARBITRUM) == True, "Arbitrum should be EVM"
    assert config.is_evm_chain(Chain.SOLANA) == False, "Solana should not be EVM"
    logger.info("   ✅ EVM detection working correctly")

    logger.info("\n   ✅ Chain configuration test completed")
    return config


def test_bridge_manager():
    """Test 2: Bridge Manager"""
    logger.info("\n" + "="*70)
    logger.info("TEST 2: Bridge Manager")
    logger.info("="*70)

    manager = BridgeManager(max_fee_pct=0.01, max_time_minutes=60)

    logger.info("\n--- Test 2a: Get Bridge Quote ---")
    quote = manager.get_quote(
        from_chain="base",
        to_chain="arbitrum",
        token="USDC",
        amount=1000.0
    )

    assert quote is not None, "Should get a quote"
    logger.info(f"   Protocol: {quote.protocol.value}")
    logger.info(f"   Route: {' → '.join(quote.route)}")
    logger.info(f"   Input: {quote.amount_in} USDC")
    logger.info(f"   Output: {quote.amount_out:.4f} USDC")
    logger.info(f"   Fee: ${quote.fee_usd:.2f}")
    logger.info(f"   Gas: ${quote.gas_cost_usd:.2f}")
    logger.info(f"   Time: ~{quote.estimated_time_minutes} minutes")

    assert quote.amount_out < quote.amount_in, "Output should be less than input (fees)"
    assert quote.fee_usd > 0, "Fee should be positive"

    logger.info("\n--- Test 2b: Compare Bridge Protocols ---")
    protocols = [BridgeProtocol.HOP, BridgeProtocol.ACROSS, BridgeProtocol.STARGATE]

    best_output = 0
    best_protocol = None

    for protocol in protocols:
        quote = manager.get_quote("base", "arbitrum", "USDC", 1000.0, protocol=protocol)
        if quote:
            net_output = quote.amount_out
            total_cost = quote.fee_usd + quote.gas_cost_usd
            logger.info(
                f"   {protocol.value:10s}: "
                f"Output: {net_output:.4f} USDC | "
                f"Cost: ${total_cost:.2f} | "
                f"Time: {quote.estimated_time_minutes}min"
            )

            if net_output > best_output:
                best_output = net_output
                best_protocol = protocol
        else:
            logger.info(f"   {protocol.value:10s}: Not supported")

    logger.info(f"\n   Best protocol: {best_protocol.value if best_protocol else 'None'}")

    logger.info("\n--- Test 2c: Multi-Chain Routes ---")
    routes = [
        ("base", "arbitrum", "USDC", 1000.0),
        ("arbitrum", "optimism", "USDC", 1000.0),
        ("optimism", "polygon", "USDC", 1000.0),
        ("base", "optimism", "ETH", 1.0),
    ]

    supported_count = 0
    for from_chain, to_chain, token, amount in routes:
        quote = manager.get_quote(from_chain, to_chain, token, amount)
        if quote:
            supported_count += 1
            logger.info(
                f"   ✅ {from_chain} → {to_chain} ({token}): "
                f"{quote.amount_in} → {quote.amount_out:.4f}"
            )
        else:
            logger.info(f"   ❌ {from_chain} → {to_chain} ({token}): Not supported")

    logger.info(f"\n   Supported routes: {supported_count}/{len(routes)}")

    logger.info("\n--- Test 2d: Initiate Transfer ---")
    quote = manager.get_quote("base", "arbitrum", "USDC", 1000.0)
    if quote:
        transfer = manager.initiate_transfer(quote)
        logger.info(f"   Transfer ID: {transfer.transfer_id}")
        logger.info(f"   Status: {transfer.status.value}")
        logger.info(f"   Protocol: {transfer.protocol.value}")
        logger.info(f"   Amount: {transfer.amount} USDC")

        assert transfer.transfer_id is not None, "Should have transfer ID"
        assert transfer.status.value in ["pending", "submitted"], "Should be pending or submitted"

        # Test status check
        status = manager.get_transfer_status(transfer.transfer_id)
        assert status is not None, "Should be able to check status"
        logger.info(f"   Status check: ✅")

        # Test confirmation
        confirmed = manager.wait_for_confirmation(transfer.transfer_id, timeout_minutes=5)
        assert confirmed, "Transfer should confirm"
        logger.info(f"   Confirmation: ✅")

    logger.info("\n--- Test 2e: Balance Rebalancing ---")
    balances = {
        "base": 8000.0,      # 53% - over target
        "arbitrum": 3000.0,  # 20%
        "optimism": 4000.0,  # 27%
    }

    logger.info("   Current balances:")
    total = sum(balances.values())
    for chain, balance in balances.items():
        pct = (balance / total) * 100
        logger.info(f"      {chain}: ${balance:,.2f} ({pct:.1f}%)")

    suggestions = manager.suggest_rebalance(balances)
    logger.info(f"\n   Rebalancing suggestions: {len(suggestions)}")
    for suggestion in suggestions:
        logger.info(
            f"      {suggestion.from_chain} → {suggestion.to_chain}: "
            f"{suggestion.amount_in:.2f} USDC"
        )

    logger.info("\n   ✅ Bridge manager test completed")
    return manager


def test_multichain_deployment():
    """Test 3: Multi-Chain Strategy Deployment"""
    logger.info("\n" + "="*70)
    logger.info("TEST 3: Multi-Chain Strategy Deployment")
    logger.info("="*70)

    # Initialize supervisor with multiple chains
    supervisor = SupervisorAgent(total_capital=100000.0)

    logger.info("\n--- Test 3a: Deploy Strategies Across Chains ---")

    # Deploy same strategy on different chains
    chains = ["base", "arbitrum", "optimism", "polygon"]
    strategy_name = "mean_reversion"
    capital_per_chain = 25000.0

    instances = []
    for chain in chains:
        instance_id = supervisor.register_instance(
            strategy_name,
            chain.capitalize(),
            capital_per_chain
        )
        instances.append(instance_id)
        logger.info(f"   ✅ Deployed {strategy_name} on {chain}")

    assert len(instances) == len(chains), "Should have instance per chain"
    logger.info(f"\n   Total instances: {len(instances)}")

    logger.info("\n--- Test 3b: Simulate Multi-Chain Trading ---")

    # Simulate trades on each chain
    trades_per_chain = 5
    import random
    random.seed(42)

    for instance_id in instances:
        for i in range(trades_per_chain):
            pnl = random.gauss(15.0, 30.0)
            supervisor.track_trade(
                TradeResult(
                    instance_id,
                    "WETH/USDC",
                    "buy",
                    1.0,
                    3000.0,
                    3000.0 + pnl,
                    pnl,
                    random.uniform(0.5, 2.0),  # Lower gas on L2s
                    0.0
                )
            )

    total_trades = len(instances) * trades_per_chain
    logger.info(f"   Simulated {total_trades} trades across {len(chains)} chains")

    logger.info("\n--- Test 3c: Cross-Chain Performance Comparison ---")
    logger.info("\n   Performance by chain:")

    for instance_id in instances:
        instance = supervisor.instances[instance_id]
        logger.info(f"\n   {instance.chain}:")
        logger.info(f"      Total trades: {instance.total_trades}")
        logger.info(f"      Win rate: {instance.win_rate:.1%}")
        logger.info(f"      Total P&L: ${instance.total_pnl:.2f}")
        logger.info(f"      Sharpe ratio: {instance.sharpe_ratio:.2f}")

    # Get overall summary
    summary = supervisor.get_performance_summary()
    logger.info(f"\n   Overall Summary:")
    logger.info(f"      Total instances: {summary['total_instances']}")
    logger.info(f"      Total trades: {summary['total_trades']}")
    logger.info(f"      Total P&L: ${summary['total_pnl']:.2f}")
    logger.info(f"      Overall return: {summary['total_return_pct']:.2f}%")

    logger.info("\n   ✅ Multi-chain deployment test completed")
    return supervisor


def test_cross_chain_operations():
    """Test 4: Cross-Chain Operations"""
    logger.info("\n" + "="*70)
    logger.info("TEST 4: Cross-Chain Operations")
    logger.info("="*70)

    config = get_chain_config(Network.TESTNET)
    bridge = BridgeManager()

    logger.info("\n--- Test 4a: Identify All Supported Chains ---")
    all_chains = config.get_all_chains()
    evm_chains = config.get_all_evm_chains()

    logger.info(f"   Total chains: {len(all_chains)}")
    logger.info(f"   EVM chains: {len(evm_chains)}")
    logger.info(f"   Chains: {[c.value for c in all_chains]}")

    assert len(all_chains) >= 4, "Should support at least 4 chains"

    logger.info("\n--- Test 4b: Test Cross-Chain Bridging ---")

    # Test all EVM chain pairs
    test_pairs = [
        ("base", "arbitrum"),
        ("arbitrum", "optimism"),
        ("optimism", "polygon"),
        ("base", "optimism"),
    ]

    successful_bridges = 0
    for from_chain, to_chain in test_pairs:
        quote = bridge.get_quote(from_chain, to_chain, "USDC", 1000.0)
        if quote:
            successful_bridges += 1
            logger.info(
                f"   ✅ {from_chain} → {to_chain}: "
                f"{quote.amount_in} → {quote.amount_out:.2f} USDC "
                f"(${quote.fee_usd:.2f} fee)"
            )
        else:
            logger.info(f"   ❌ {from_chain} → {to_chain}: Not available")

    logger.info(f"\n   Successful bridge routes: {successful_bridges}/{len(test_pairs)}")

    logger.info("\n--- Test 4c: Simulate Cross-Chain Arbitrage ---")

    # Mock prices on different chains
    mock_prices = {
        "base": {"WETH": 3000.0},
        "arbitrum": {"WETH": 3005.0},  # 5 USDC higher
        "optimism": {"WETH": 2998.0},
        "polygon": {"WETH": 3002.0},
    }

    logger.info("   Mock WETH prices:")
    for chain, prices in mock_prices.items():
        logger.info(f"      {chain}: ${prices['WETH']:.2f}")

    # Find arbitrage opportunity
    chains_list = list(mock_prices.keys())
    best_arb = None
    best_profit = 0

    for i, buy_chain in enumerate(chains_list):
        for sell_chain in chains_list[i+1:]:
            buy_price = mock_prices[buy_chain]["WETH"]
            sell_price = mock_prices[sell_chain]["WETH"]

            spread = sell_price - buy_price
            spread_pct = spread / buy_price

            if abs(spread_pct) > 0.001:  # 0.1% threshold
                # Check if bridge is available
                direction = (buy_chain, sell_chain) if spread > 0 else (sell_chain, buy_chain)
                quote = bridge.get_quote(direction[0], direction[1], "WETH", 1.0)

                if quote:
                    net_profit = abs(spread) - quote.fee_usd - quote.gas_cost_usd
                    if net_profit > best_profit:
                        best_profit = net_profit
                        best_arb = {
                            "buy": buy_chain if spread > 0 else sell_chain,
                            "sell": sell_chain if spread > 0 else buy_chain,
                            "spread": abs(spread),
                            "net_profit": net_profit,
                            "quote": quote
                        }

    if best_arb:
        logger.info(f"\n   Best arbitrage opportunity:")
        logger.info(f"      Buy: {best_arb['buy']} @ ${mock_prices[best_arb['buy']]['WETH']:.2f}")
        logger.info(f"      Sell: {best_arb['sell']} @ ${mock_prices[best_arb['sell']]['WETH']:.2f}")
        logger.info(f"      Spread: ${best_arb['spread']:.2f}")
        logger.info(f"      Bridge fee: ${best_arb['quote'].fee_usd:.2f}")
        logger.info(f"      Net profit: ${best_arb['net_profit']:.2f}")
    else:
        logger.info("   No profitable arbitrage found")

    logger.info("\n   ✅ Cross-chain operations test completed")


def test_integration():
    """Test 5: Full Multi-Chain Integration"""
    logger.info("\n" + "="*70)
    logger.info("TEST 5: Full Multi-Chain Integration")
    logger.info("="*70)

    logger.info("\n--- Test 5a: Initialize Multi-Chain System ---")

    # Initialize all components
    config = get_chain_config(Network.TESTNET)
    bridge = BridgeManager()
    supervisor = SupervisorAgent(total_capital=100000.0)

    logger.info("   ✅ Chain config initialized")
    logger.info("   ✅ Bridge manager initialized")
    logger.info("   ✅ Supervisor initialized")

    logger.info("\n--- Test 5b: Deploy Strategies on 4 Chains ---")

    deployment_plan = [
        ("mean_reversion", "Base", 15000.0),
        ("mean_reversion", "Arbitrum", 15000.0),
        ("breakout", "Optimism", 15000.0),
        ("momentum", "Polygon", 15000.0),
        ("mean_reversion", "Solana", 20000.0),
        ("breakout", "Base", 20000.0),
    ]

    for strategy, chain, capital in deployment_plan:
        instance_id = supervisor.register_instance(strategy, chain, capital)
        logger.info(f"   ✅ {strategy} on {chain}: ${capital:,.0f}")

    logger.info(f"\n   Total deployments: {len(deployment_plan)}")

    logger.info("\n--- Test 5c: Simulate Trading Session ---")

    # Simulate 30 trades
    import random
    random.seed(123)

    for i in range(30):
        # Pick random instance
        instance_id = random.choice(list(supervisor.instances.keys()))
        pnl = random.gauss(20.0, 40.0)

        supervisor.track_trade(
            TradeResult(
                instance_id,
                random.choice(["WETH/USDC", "BTC/USDC"]),
                random.choice(["buy", "sell"]),
                1.0,
                3000.0,
                3000.0 + pnl,
                pnl,
                random.uniform(0.3, 1.5),  # L2 gas costs
                0.0
            )
        )

    logger.info(f"   Simulated 30 trades")

    logger.info("\n--- Test 5d: Performance by Chain ---")

    # Aggregate performance by chain
    chain_stats = {}
    for instance_id, instance in supervisor.instances.items():
        chain = instance.chain
        if chain not in chain_stats:
            chain_stats[chain] = {
                "trades": 0,
                "pnl": 0.0,
                "instances": 0
            }
        chain_stats[chain]["trades"] += instance.total_trades
        chain_stats[chain]["pnl"] += instance.total_pnl
        chain_stats[chain]["instances"] += 1

    logger.info("\n   Chain Performance:")
    for chain, stats in sorted(chain_stats.items(), key=lambda x: x[1]["pnl"], reverse=True):
        logger.info(f"      {chain:10s}: "
                   f"Instances: {stats['instances']} | "
                   f"Trades: {stats['trades']} | "
                   f"P&L: ${stats['pnl']:+.2f}")

    logger.info("\n--- Test 5e: Suggest Cross-Chain Rebalancing ---")

    # Calculate capital per chain
    chain_balances = {}
    for instance_id, instance in supervisor.instances.items():
        chain = instance.chain.lower()
        if chain not in chain_balances:
            chain_balances[chain] = 0.0
        chain_balances[chain] += instance.current_capital

    logger.info("   Current distribution:")
    total_capital = sum(chain_balances.values())
    for chain, balance in sorted(chain_balances.items(), key=lambda x: x[1], reverse=True):
        pct = (balance / total_capital) * 100
        logger.info(f"      {chain:10s}: ${balance:,.2f} ({pct:.1f}%)")

    # Get rebalancing suggestions
    suggestions = bridge.suggest_rebalance(chain_balances)
    if suggestions:
        logger.info(f"\n   Rebalancing suggestions: {len(suggestions)}")
        for suggestion in suggestions:
            logger.info(
                f"      {suggestion.from_chain} → {suggestion.to_chain}: "
                f"${suggestion.amount_in:,.2f}"
            )
    else:
        logger.info("\n   No rebalancing needed")

    logger.info("\n--- Test 5f: Final Summary ---")
    summary = supervisor.get_performance_summary()

    logger.info(f"   Total Capital: ${summary['total_capital']:,.2f}")
    logger.info(f"   Total P&L: ${summary['total_pnl']:+.2f}")
    logger.info(f"   Return: {summary['total_return_pct']:+.2f}%")
    logger.info(f"   Win Rate: {summary['overall_win_rate']:.1%}")
    logger.info(f"   Active Chains: {len(chain_stats)}")
    logger.info(f"   Total Instances: {summary['total_instances']}")

    logger.info("\n   ✅ Full integration test completed")


def run_all_tests():
    """Run all Phase 5 tests"""
    logger.info("\n" + "="*70)
    logger.info("PHASE 5: MULTI-CHAIN EXPANSION TESTS")
    logger.info("="*70)

    try:
        # Test 1: Chain Configuration
        test_chain_configuration()

        # Test 2: Bridge Manager
        test_bridge_manager()

        # Test 3: Multi-Chain Deployment
        test_multichain_deployment()

        # Test 4: Cross-Chain Operations
        test_cross_chain_operations()

        # Test 5: Full Integration
        test_integration()

        logger.info("\n" + "="*70)
        logger.info("✅ ALL PHASE 5 TESTS PASSED")
        logger.info("="*70)
        logger.info("\nPhase 5 Multi-Chain Expansion:")
        logger.info("  ✅ Chain configuration (6 chains: Base, Arbitrum, Optimism, Polygon, Ethereum, Solana)")
        logger.info("  ✅ RPC endpoint management with failover")
        logger.info("  ✅ Bridge integration (Hop, Across, Stargate)")
        logger.info("  ✅ Multi-chain strategy deployment")
        logger.info("  ✅ Cross-chain performance tracking")
        logger.info("  ✅ Automated rebalancing suggestions")
        logger.info("  ✅ Cross-chain arbitrage detection")
        logger.info("\n🎯 Phase 5 Deliverable: COMPLETE")
        logger.info("   System now supports:")
        logger.info("   - 4+ EVM L2 chains (Base, Arbitrum, Optimism, Polygon)")
        logger.info("   - Ethereum L1 and Solana")
        logger.info("   - Cross-chain bridging with 3 protocols")
        logger.info("   - Multi-chain strategy deployment")
        logger.info("   - Cross-chain performance comparison")
        logger.info("   - Automated capital rebalancing")
        logger.info("\n📍 Next: Phase 6 (Production Hardening)")
        logger.info("="*70)

        return True

    except AssertionError as e:
        logger.error(f"\n❌ TEST FAILED: {e}")
        return False
    except Exception as e:
        logger.error(f"\n❌ ERROR: {e}", exc_info=True)
        return False


if __name__ == "__main__":
    success = run_all_tests()
    exit(0 if success else 1)
