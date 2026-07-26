"""
Test Phase 1: Multi-Chain Supervisor and Execution Infrastructure
Tests the integration of SupervisorAgent, StrategyInstanceManager, and ExecutionRouter
"""
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.agents.supervisor_agent import SupervisorAgent, TradeResult
from src.execution.strategy_instance_manager import StrategyInstanceManager, Chain
from src.execution.execution_router import ExecutionRouter, VenueType
import logging
from datetime import datetime, timedelta

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def test_supervisor_initialization():
    """Test 1: Supervisor Agent initialization"""
    logger.info("\n" + "="*70)
    logger.info("TEST 1: Supervisor Agent Initialization")
    logger.info("="*70)

    supervisor = SupervisorAgent(
        total_capital=30000.0,
        reallocation_interval_hours=6,
        max_asset_pct=0.50,
        max_position_pct=0.10,
        circuit_breaker_daily_loss=0.15,
        arbitrage_threshold=0.01
    )

    assert supervisor.total_capital == 30000.0
    assert supervisor.max_asset_pct == 0.50
    assert supervisor.max_position_pct == 0.10

    logger.info("✅ Supervisor initialized successfully")
    logger.info(f"   Capital: ${supervisor.total_capital:,.2f}")
    logger.info(f"   Risk Limits: {supervisor.max_asset_pct*100}% per asset, {supervisor.max_position_pct*100}% per position")

    return supervisor


def test_instance_manager():
    """Test 2: Strategy Instance Manager"""
    logger.info("\n" + "="*70)
    logger.info("TEST 2: Strategy Instance Manager")
    logger.info("="*70)

    manager = StrategyInstanceManager()

    # Mock strategy class
    class MockStrategy:
        def __init__(self, symbols):
            self.symbols = symbols

    # Spawn instances across chains
    ids = manager.spawn_strategy_instances(
        strategy_name="test_strategy",
        strategy_class=MockStrategy,
        chains=[Chain.BASE, Chain.SOLANA, Chain.CEX_BINANCE],
        capital_per_instance=1000.0
    )

    assert len(ids) == 3
    assert manager.get_instance_summary()["total_instances"] == 3
    assert manager.get_instance_summary()["total_capital"] == 3000.0

    logger.info(f"✅ Spawned {len(ids)} instances successfully")

    # Test enabling instances
    for instance_id in ids:
        success = manager.enable_instance(instance_id)
        assert success

    enabled = manager.get_enabled_instances()
    assert len(enabled) == 3

    logger.info(f"✅ Enabled {len(enabled)} instances")

    # Get summary
    summary = manager.get_instance_summary()
    logger.info(f"\nInstance Summary:")
    logger.info(f"   Total: {summary['total_instances']}")
    logger.info(f"   Enabled: {summary['enabled_instances']}")
    logger.info(f"   Capital: ${summary['total_capital']:,.2f}")
    for chain, data in summary['by_chain'].items():
        logger.info(f"   {chain}: {data['count']} instances")

    return manager


def test_execution_router():
    """Test 3: Execution Router"""
    logger.info("\n" + "="*70)
    logger.info("TEST 3: Execution Router")
    logger.info("="*70)

    router = ExecutionRouter(max_gas_pct=0.02)

    # Test routing decision for small trade (high gas %)
    logger.info("\nTest 3a: Small trade (should use CEX due to high gas %)")
    decision = router.get_best_route(
        symbol="WETH/USDC",
        side="buy",
        quantity=0.01,  # Very small
        chain="Base"
    )

    logger.info(f"   Selected: {decision.selected_venue.value}")
    logger.info(f"   Reason: {decision.reason}")
    assert decision.selected_venue == VenueType.CEX

    # Test routing decision for large trade (low gas %)
    logger.info("\nTest 3b: Large trade (compare CEX vs DEX)")
    decision = router.get_best_route(
        symbol="WETH/USDC",
        side="buy",
        quantity=10.0,  # Large amount
        chain="Base"
    )

    logger.info(f"   Selected: {decision.selected_venue.value}")
    logger.info(f"   Reason: {decision.reason}")
    logger.info(f"   Savings: ${decision.savings:.2f}")

    # Test execution
    logger.info("\nTest 3c: Execute trade via selected venue")
    result = router.execute_trade(decision, instance_id="test_001")

    assert result["success"] == True
    assert result["instance_id"] == "test_001"
    logger.info(f"   ✅ Trade executed successfully")
    logger.info(f"   Venue: {result['venue']}")
    logger.info(f"   Symbol: {result['symbol']} {result['side']}")
    logger.info(f"   Net output: ${result['net_output']:.2f}")

    return router


def test_supervisor_performance_tracking():
    """Test 4: Supervisor performance tracking"""
    logger.info("\n" + "="*70)
    logger.info("TEST 4: Supervisor Performance Tracking")
    logger.info("="*70)

    supervisor = SupervisorAgent(total_capital=30000.0)

    # Register instances first - capture the IDs
    instance_id_1 = supervisor.register_instance("mean_reversion", "Base", 1000.0)
    instance_id_2 = supervisor.register_instance("momentum", "Solana", 1000.0)

    # Simulate some trades using the actual instance IDs
    trades = [
        TradeResult(
            instance_id=instance_id_1,
            symbol="WETH/USDC",
            side="buy",
            quantity=1.0,
            entry_price=3000.0,
            exit_price=3050.0,
            pnl=50.0,
            fees=3.0,
            gas=8.0
        ),
        TradeResult(
            instance_id=instance_id_1,
            symbol="WETH/USDC",
            side="sell",
            quantity=1.0,
            entry_price=3050.0,
            exit_price=3020.0,
            pnl=-30.0,
            fees=3.0,
            gas=8.0
        ),
        TradeResult(
            instance_id=instance_id_2,
            symbol="SOL/USDC",
            side="buy",
            quantity=10.0,
            entry_price=120.0,
            exit_price=125.0,
            pnl=50.0,
            fees=1.5,
            gas=0.5
        )
    ]

    for trade in trades:
        supervisor.track_trade(trade)

    # Get summary
    summary = supervisor.get_performance_summary()

    logger.info(f"✅ Tracked {summary['total_trades']} trades")
    logger.info(f"   Total P&L: ${summary['total_pnl']:.2f}")
    logger.info(f"   Win Rate: {summary['overall_win_rate']*100:.1f}%")

    logger.info(f"\nTop Performers:")
    for perf in summary['top_performers'][:3]:
        logger.info(f"   {perf['instance_id']}: ${perf['pnl']:.2f} (Win rate: {perf['win_rate']*100:.1f}%)")

    assert summary['total_trades'] == 3

    return supervisor


def test_capital_allocation():
    """Test 5: Capital allocation algorithm"""
    logger.info("\n" + "="*70)
    logger.info("TEST 5: Capital Allocation Algorithm")
    logger.info("="*70)

    supervisor = SupervisorAgent(total_capital=10000.0)

    # Register 4 instances with different performance
    instances = [
        supervisor.register_instance("mean_reversion", "Base", 2500.0),
        supervisor.register_instance("momentum", "Solana", 2500.0),
        supervisor.register_instance("rsi", "Arbitrum", 2500.0),
        supervisor.register_instance("ml_ensemble", "Base", 2500.0)
    ]

    # Simulate trades with different performance
    # Instance 1: Best performer (Sharpe 2.5)
    for i in range(10):
        pnl = 50.0 if i < 8 else -30.0
        supervisor.track_trade(TradeResult(
            instance_id=instances[0], symbol="WETH/USDC", side="buy",
            quantity=1.0, entry_price=3000.0, exit_price=3050.0 if pnl > 0 else 2970.0,
            pnl=pnl, fees=3.0, gas=8.0
        ))

    # Instance 2: Good performer
    for i in range(10):
        pnl = 25.0 if i < 6 else -20.0
        supervisor.track_trade(TradeResult(
            instance_id=instances[1], symbol="SOL/USDC", side="buy",
            quantity=10.0, entry_price=120.0, exit_price=122.5 if pnl > 0 else 118.0,
            pnl=pnl, fees=1.5, gas=0.5
        ))

    # Instance 3: Average performer
    for i in range(10):
        pnl = 15.0 if i < 5 else -15.0
        supervisor.track_trade(TradeResult(
            instance_id=instances[2], symbol="ARB/USDC", side="buy",
            quantity=10.0, entry_price=1.0, exit_price=1.015 if pnl > 0 else 0.985,
            pnl=pnl, fees=0.5, gas=2.0
        ))

    # Instance 4: Poor performer
    for i in range(10):
        pnl = 10.0 if i < 4 else -10.0
        supervisor.track_trade(TradeResult(
            instance_id=instances[3], symbol="ETH/USDC", side="buy",
            quantity=0.1, entry_price=3000.0, exit_price=3100.0 if pnl > 0 else 2900.0,
            pnl=pnl, fees=3.0, gas=8.0
        ))

    # Force reallocation by setting last_reallocation to the past
    supervisor.last_reallocation = datetime.now() - timedelta(hours=7)

    # Calculate allocations
    allocations = supervisor.calculate_allocations()

    logger.info(f"✅ Calculated allocations for {len(allocations)} instances")
    logger.info(f"\nAllocation Distribution:")

    sorted_allocs = sorted(allocations.items(), key=lambda x: x[1], reverse=True)
    for instance_id, amount in sorted_allocs:
        pct = (amount / supervisor.total_capital) * 100
        inst = supervisor.instances[instance_id]
        logger.info(f"   {instance_id}: ${amount:,.2f} ({pct:.1f}%) - Sharpe: {inst.sharpe_ratio:.2f}, P&L: ${inst.total_pnl:.2f}")

    # Top performer should get approximately 40%
    top_allocation = sorted_allocs[0][1]
    top_pct = (top_allocation / supervisor.total_capital) * 100

    logger.info(f"\n✅ Top performer gets {top_pct:.1f}% of capital")
    assert top_pct >= 35.0  # Should be close to 40%

    return supervisor


def test_risk_management():
    """Test 6: Risk management checks"""
    logger.info("\n" + "="*70)
    logger.info("TEST 6: Risk Management")
    logger.info("="*70)

    supervisor = SupervisorAgent(
        total_capital=10000.0,
        max_asset_pct=0.50,
        max_position_pct=0.10,
        circuit_breaker_daily_loss=0.15
    )

    # Register an instance for testing
    instance_id = supervisor.register_instance("test_strategy", "Base", 10000.0)

    # Test position size limits
    logger.info("\nTest 6a: Position size limits")

    # Should pass: 5% of portfolio
    allowed, reason = supervisor.check_risk_limits(instance_id, 500.0, "WETH")
    logger.info(f"   $500 trade (5%): {'✅ ALLOWED' if allowed else '❌ BLOCKED'} - {reason}")
    assert allowed == True

    # Should fail: 15% of portfolio (max is 10%)
    allowed, reason = supervisor.check_risk_limits(instance_id, 1500.0, "WETH")
    logger.info(f"   $1500 trade (15%): {'✅ ALLOWED' if allowed else '❌ BLOCKED'} - {reason}")
    assert allowed == False

    # Test daily loss limit
    logger.info("\nTest 6b: Daily loss circuit breaker")

    # Simulate large loss by manipulating capital
    for inst in supervisor.instances.values():
        inst.current_capital = 8400.0  # 16% loss from $10,000

    # Try to make a trade - should trigger circuit breaker
    allowed, reason = supervisor.check_risk_limits(instance_id, 100.0, "WETH")
    daily_loss_pct = (8400.0 - supervisor.daily_start_value) / supervisor.daily_start_value * 100

    logger.info(f"   Current value: $8,400 (Daily loss: {daily_loss_pct:.1f}%)")
    logger.info(f"   Trade attempt: {'✅ ALLOWED' if allowed else '🛑 BLOCKED'} - {reason}")
    logger.info(f"   Circuit Breaker: {'🛑 TRIGGERED' if supervisor.circuit_breaker_triggered else '✅ OK'}")

    assert supervisor.circuit_breaker_triggered == True
    assert allowed == False

    logger.info(f"\n✅ Risk management working correctly")

    return supervisor


def run_all_tests():
    """Run all Phase 1 tests"""
    logger.info("\n" + "="*70)
    logger.info("PHASE 1 INTEGRATION TESTS")
    logger.info("="*70)

    try:
        test_supervisor_initialization()
        test_instance_manager()
        test_execution_router()
        test_supervisor_performance_tracking()
        test_capital_allocation()
        test_risk_management()

        logger.info("\n" + "="*70)
        logger.info("✅ ALL TESTS PASSED")
        logger.info("="*70)
        logger.info("\nPhase 1 Foundation:")
        logger.info("  ✅ Supervisor Agent working")
        logger.info("  ✅ Strategy Instance Manager working")
        logger.info("  ✅ Execution Router working")
        logger.info("  ✅ Performance tracking working")
        logger.info("  ✅ Capital allocation working")
        logger.info("  ✅ Risk management working")
        logger.info("\n🎯 Phase 1 Deliverable: COMPLETE")
        logger.info("   System spawns instances, tracks mock performance, routes to mock executor")
        logger.info("\n📍 Next: Phase 2 (CEX Integration)")
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
