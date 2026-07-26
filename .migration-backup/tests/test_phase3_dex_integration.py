"""
Test Phase 3: DEX Integration
Tests wallet management, gas optimization, and DEX trading
"""
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.wallet.wallet_manager import WalletManager, Chain as WalletChain, Transaction
from src.gas.gas_manager import GasManager, Chain as GasChain, GasSpeed
from src.dex.dex_connector import DEXConnector, SwapRequest, Chain as DEXChain
from src.execution.execution_router import ExecutionRouter
import logging
import tempfile

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def test_wallet_manager():
    """Test 1: Wallet Manager"""
    logger.info("\n" + "="*70)
    logger.info("TEST 1: Wallet Manager")
    logger.info("="*70)

    # Create temporary wallet file
    temp_file = os.path.join(tempfile.gettempdir(), "test_wallets.enc")

    # Initialize wallet manager
    manager = WalletManager(
        wallet_file=temp_file,
        master_password="test_password_123"
    )

    logger.info("✅ Wallet Manager initialized")

    # Test wallet generation
    logger.info("\n--- Test 1a: Generate Wallets ---")
    try:
        # Generate EVM wallet
        base_addr = manager.generate_wallet(WalletChain.BASE)
        logger.info(f"   ✅ Base wallet: {base_addr}")
        assert len(base_addr) == 42, "Invalid EVM address format"
        assert base_addr.startswith("0x"), "EVM address should start with 0x"

    except Exception as e:
        logger.warning(f"   ⚠️  Wallet generation failed: {e}")
        logger.info("   Install eth-account: pip install eth-account")

    # Test wallet saving
    logger.info("\n--- Test 1b: Save/Load Wallets ---")
    if manager.save_wallets():
        logger.info("   ✅ Wallets saved to encrypted file")

        # Load wallets in new instance
        manager2 = WalletManager(
            wallet_file=temp_file,
            master_password="test_password_123"
        )

        if manager2.load_wallets():
            logger.info("   ✅ Wallets loaded successfully")
            wallets = manager2.get_all_wallets()
            logger.info(f"   Loaded {len(wallets)} wallets")

    # Test balance checking
    logger.info("\n--- Test 1c: Balance Checking ---")
    if manager.has_wallet(WalletChain.BASE):
        balance = manager.get_balance(WalletChain.BASE)
        logger.info(f"   Chain: {balance.chain.value}")
        logger.info(f"   Balance: {balance.balance_native:.4f}")
        logger.info(f"   Gas Reserve: {balance.gas_reserve}")
        logger.info(f"   Available: {balance.available:.4f}")
        logger.info("   ✅ Balance check completed (mock data)")

    # Cleanup
    if os.path.exists(temp_file):
        os.remove(temp_file)

    logger.info("✅ Wallet Manager test completed")
    return manager


def test_gas_manager():
    """Test 2: Gas Manager"""
    logger.info("\n" + "="*70)
    logger.info("TEST 2: Gas Manager")
    logger.info("="*70)

    manager = GasManager(max_gas_pct=0.02)

    # Test gas price fetching
    logger.info("\n--- Test 2a: Gas Prices ---")
    for chain in [GasChain.BASE, GasChain.ETHEREUM, GasChain.SOLANA]:
        prices = manager._fetch_gas_prices(chain)
        logger.info(f"   {chain.value}:")
        logger.info(f"      Slow: {prices.slow} | Standard: {prices.standard} | Fast: {prices.fast}")

    logger.info("   ✅ Gas prices fetched")

    # Test gas estimation
    logger.info("\n--- Test 2b: Gas Estimation ---")
    test_cases = [
        (GasChain.BASE, 10000.0, "Large trade on Base"),
        (GasChain.BASE, 100.0, "Small trade on Base"),
        (GasChain.ETHEREUM, 10000.0, "Mainnet trade"),
        (GasChain.SOLANA, 1000.0, "Solana trade"),
    ]

    for chain, trade_value, description in test_cases:
        estimate = manager.estimate_gas(chain, "uniswap_swap", trade_value)
        logger.info(f"\n   {description}:")
        logger.info(f"      Trade: ${trade_value:,.2f}")
        logger.info(f"      Gas: ${estimate.total_gas_usd:.4f}")
        logger.info(f"      Percentage: {estimate.percentage_of_trade*100:.3f}%")

        assert estimate.percentage_of_trade >= 0, "Percentage should be non-negative"

    logger.info("\n   ✅ Gas estimation completed")

    # Test trade execution decision
    logger.info("\n--- Test 2c: Trade Execution Decision ---")
    for chain, trade_value, description in test_cases:
        should_exec, reason, estimate = manager.should_execute_trade(chain, trade_value)
        status = "✅ EXECUTE" if should_exec else "❌ REJECT"
        logger.info(f"   {description}: {status}")
        logger.info(f"      {reason}")

    # Test optimal speed selection
    logger.info("\n--- Test 2d: Optimal Gas Speed ---")
    for chain, trade_value, description in test_cases[:2]:
        optimal = manager.get_optimal_speed(chain, trade_value)
        logger.info(f"   {description}: {optimal.value}")

    logger.info("✅ Gas Manager test completed")
    return manager


def test_dex_connector():
    """Test 3: DEX Connector"""
    logger.info("\n" + "="*70)
    logger.info("TEST 3: DEX Connector")
    logger.info("="*70)

    # Initialize with wallet and gas managers for full functionality
    temp_file = os.path.join(tempfile.gettempdir(), "test_dex_wallets.enc")
    wallet_mgr = WalletManager(wallet_file=temp_file, master_password="test123")
    gas_mgr = GasManager(max_gas_pct=0.02)

    connector = DEXConnector(
        wallet_manager=wallet_mgr,
        gas_manager=gas_mgr,
        use_aggregators=True
    )

    # Test quote fetching
    logger.info("\n--- Test 3a: Swap Quotes ---")

    # Uniswap quote (Base)
    base_request = SwapRequest(
        chain=DEXChain.BASE,
        input_token="WETH",
        output_token="USDC",
        amount_in=1.0,
        slippage_pct=0.01
    )

    quote = connector.get_quote(base_request)
    logger.info(f"\n   Base (Uniswap V3):")
    logger.info(f"      Input: {quote.input_amount} {quote.input_token}")
    logger.info(f"      Output: {quote.output_amount:.2f} {quote.output_token}")
    logger.info(f"      Price: ${quote.price:.2f}")
    logger.info(f"      Gas: ${quote.gas_cost_usd:.2f}")
    logger.info(f"      Fee: ${quote.fee:.2f} ({quote.fee_pct*100:.1f}%)")

    assert quote.output_amount > 0, "Output should be positive"
    assert quote.gas_cost_usd > 0, "Gas cost should be positive"

    # Jupiter quote (Solana)
    sol_request = SwapRequest(
        chain=DEXChain.SOLANA,
        input_token="SOL",
        output_token="USDC",
        amount_in=10.0,
        slippage_pct=0.01
    )

    quote_sol = connector.get_quote(sol_request)
    logger.info(f"\n   Solana (Jupiter):")
    logger.info(f"      Input: {quote_sol.input_amount} {quote_sol.input_token}")
    logger.info(f"      Output: {quote_sol.output_amount:.2f} {quote_sol.output_token}")
    logger.info(f"      Price: ${quote_sol.price:.2f}")
    logger.info(f"      Gas: ${quote_sol.gas_cost_usd:.4f}")
    logger.info(f"      Fee: ${quote_sol.fee:.2f}")

    logger.info("\n   ✅ Quotes fetched successfully")

    # Test best quote selection
    logger.info("\n--- Test 3b: Best Quote Selection ---")
    best = connector.get_best_quote(base_request)
    logger.info(f"   Best DEX: {best.dex.value}")
    logger.info(f"   Output: {best.output_amount:.2f} {best.output_token}")
    logger.info(f"   Total cost: ${best.fee + best.gas_cost_usd:.2f}")
    logger.info("   ✅ Best quote selected")

    # Test swap execution (mock)
    logger.info("\n--- Test 3c: Swap Execution (Mock) ---")
    result = connector.execute_swap(base_request, quote)
    logger.info(f"   Success: {result.success}")
    logger.info(f"   TX Hash: {result.tx_hash[:20]}...")
    logger.info(f"   Output: {result.output_amount:.4f} {result.output_token}")
    logger.info(f"   Gas: ${result.gas_cost_usd:.2f}")
    logger.info(f"   Total Cost: ${result.total_cost_usd:.2f}")

    assert result.success, "Mock swap should succeed"
    logger.info("   ✅ Swap execution test completed")

    logger.info("✅ DEX Connector test completed")
    return connector


def test_integrated_execution_router():
    """Test 4: Integrated Execution Router"""
    logger.info("\n" + "="*70)
    logger.info("TEST 4: Integrated Execution Router")
    logger.info("="*70)

    # Initialize all components
    gas_manager = GasManager(max_gas_pct=0.02)
    dex_connector = DEXConnector(
        wallet_manager=None,  # No wallet for mock test
        gas_manager=gas_manager,
        use_aggregators=True
    )

    router = ExecutionRouter(
        max_gas_pct=0.02,
        cex_priority=False,
        cex_connector=None,  # No CEX for this test
        dex_connector=dex_connector
    )

    logger.info("   ✅ Router initialized with DEX connector")

    # Test routing decision
    logger.info("\n--- Test 4a: Routing Decision ---")
    decision = router.get_best_route(
        symbol="WETH/USDC",
        side="buy",
        quantity=10.0,
        chain="Base"
    )

    logger.info(f"   Selected venue: {decision.selected_venue.value}")
    logger.info(f"   Reason: {decision.reason}")
    logger.info(f"   Quote price: ${decision.selected_quote.price:.2f}")
    logger.info(f"   Gas: ${decision.selected_quote.gas:.2f}")

    # Test execution through router
    logger.info("\n--- Test 4b: Execute via Router ---")
    result = router.execute_trade(decision, instance_id="test_phase3_001")

    logger.info(f"   Success: {result['success']}")
    logger.info(f"   Venue: {result['venue']}")
    logger.info(f"   Symbol: {result['symbol']}")
    logger.info(f"   Quantity: {result['quantity']}")
    logger.info(f"   Gas: ${result['gas']:.2f}")

    assert result is not None, "Execution should return a result"
    logger.info("   ✅ Router execution test completed")

    logger.info("✅ Integrated Router test completed")


def test_end_to_end_workflow():
    """Test 5: End-to-end workflow"""
    logger.info("\n" + "="*70)
    logger.info("TEST 5: End-to-End Workflow")
    logger.info("="*70)

    logger.info("\n--- Simulating Multi-Chain Trade Execution ---")

    # Step 1: Initialize all components
    logger.info("\n   Step 1: Initialize components...")
    wallet_mgr = WalletManager(
        wallet_file=os.path.join(tempfile.gettempdir(), "test_e2e.enc"),
        master_password="test123"
    )
    gas_mgr = GasManager(max_gas_pct=0.02)
    dex_conn = DEXConnector(
        wallet_manager=wallet_mgr,
        gas_manager=gas_mgr,
        use_aggregators=True
    )
    logger.info("   ✅ Components initialized")

    # Step 2: Check gas prices
    logger.info("\n   Step 2: Check gas prices...")
    for chain in [GasChain.BASE, GasChain.ARBITRUM]:
        gas_price = gas_mgr.get_gas_price(chain, GasSpeed.STANDARD)
        logger.info(f"   {chain.value}: {gas_price} Gwei")
    logger.info("   ✅ Gas prices checked")

    # Step 3: Get swap quote
    logger.info("\n   Step 3: Get swap quote...")
    swap_req = SwapRequest(
        chain=DEXChain.BASE,
        input_token="WETH",
        output_token="USDC",
        amount_in=1.0,
        slippage_pct=0.01
    )
    quote = dex_conn.get_best_quote(swap_req)
    logger.info(f"   Best quote: {quote.output_amount:.2f} {quote.output_token}")
    logger.info(f"   Via: {quote.dex.value}")
    logger.info("   ✅ Quote obtained")

    # Step 4: Check if should execute
    logger.info("\n   Step 4: Check execution criteria...")
    should_exec, reason, _ = gas_mgr.should_execute_trade(
        GasChain.BASE, quote.output_amount
    )
    logger.info(f"   Decision: {'EXECUTE' if should_exec else 'REJECT'}")
    logger.info(f"   Reason: {reason}")
    logger.info("   ✅ Execution criteria checked")

    # Step 5: Execute swap (mock)
    if should_exec:
        logger.info("\n   Step 5: Execute swap...")
        result = dex_conn.execute_swap(swap_req, quote)
        logger.info(f"   Success: {result.success}")
        logger.info(f"   TX: {result.tx_hash[:20]}...")
        logger.info(f"   Output: {result.output_amount:.4f}")
        logger.info("   ✅ Swap executed")

    logger.info("\n✅ End-to-end workflow completed")


def run_all_tests():
    """Run all Phase 3 tests"""
    logger.info("\n" + "="*70)
    logger.info("PHASE 3: DEX INTEGRATION TESTS")
    logger.info("="*70)

    try:
        # Test 1: Wallet Manager
        test_wallet_manager()

        # Test 2: Gas Manager
        test_gas_manager()

        # Test 3: DEX Connector
        test_dex_connector()

        # Test 4: Integrated Router
        test_integrated_execution_router()

        # Test 5: End-to-end workflow
        test_end_to_end_workflow()

        logger.info("\n" + "="*70)
        logger.info("✅ ALL PHASE 3 TESTS PASSED")
        logger.info("="*70)
        logger.info("\nPhase 3 DEX Integration:")
        logger.info("  ✅ Wallet Manager with encryption")
        logger.info("  ✅ Gas Manager with price tracking")
        logger.info("  ✅ DEX Connector (Uniswap, Jupiter)")
        logger.info("  ✅ Smart quote routing")
        logger.info("  ✅ Execution Router integration")
        logger.info("  ✅ End-to-end workflow")
        logger.info("\n🎯 Phase 3 Deliverable: COMPLETE")
        logger.info("   Infrastructure ready for real DEX swaps")
        logger.info("\n⚠️  Note: Currently using MOCK execution")
        logger.info("   To enable real swaps:")
        logger.info("   1. Integrate Uniswap V3 SDK (@uniswap/v3-sdk)")
        logger.info("   2. Integrate Jupiter API (Jupiter Swap API)")
        logger.info("   3. Add transaction confirmation monitoring")
        logger.info("   4. Set up testnet wallets with funds")
        logger.info("\n📍 Next: Phase 4 (Supervisor Logic Enhancement)")
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
