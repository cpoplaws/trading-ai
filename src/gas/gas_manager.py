"""
Gas Manager - Gas price tracking and optimization

Responsibilities:
1. Track gas prices across chains (real-time)
2. Reject trades if gas > threshold% of trade value
3. Provide gas estimates for swaps
4. Suggest optimal gas prices (fast/standard/slow)
5. Historical gas price tracking

Optimization:
- Wait for low gas if trade is not urgent
- Batch transactions when possible
- Use EIP-1559 on supported chains
"""

import logging
import time
from typing import Dict, Optional, Literal
from dataclasses import dataclass
from enum import Enum
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


class Chain(Enum):
    """Supported chains"""
    BASE = "base"
    ARBITRUM = "arbitrum"
    OPTIMISM = "optimism"
    POLYGON = "polygon"
    ETHEREUM = "ethereum"
    SOLANA = "solana"


class GasSpeed(Enum):
    """Gas speed options"""
    SLOW = "slow"       # ~10 mins
    STANDARD = "standard"  # ~3 mins
    FAST = "fast"       # ~30 secs
    INSTANT = "instant"    # Next block


@dataclass
class GasPrice:
    """Gas price information"""
    chain: Chain
    slow: float      # In Gwei (or lamports for Solana)
    standard: float
    fast: float
    instant: float
    timestamp: datetime
    base_fee: Optional[float] = None  # EIP-1559
    priority_fee: Optional[float] = None  # EIP-1559


@dataclass
class GasEstimate:
    """Gas estimate for transaction"""
    chain: Chain
    gas_limit: int
    gas_price: float  # In Gwei
    total_gas_eth: float  # Total in native token
    total_gas_usd: float
    percentage_of_trade: float
    speed: GasSpeed
    timestamp: datetime


class GasManager:
    """
    Gas price manager across multiple chains.

    Features:
    - Real-time gas price tracking
    - Gas estimation for swaps
    - Trade rejection based on gas threshold
    - Historical gas tracking
    - Optimal timing recommendations

    Thresholds:
    - Max gas as % of trade: 2% (default)
    - High gas warning: 1%
    """

    # Default gas limits for operations
    DEFAULT_GAS_LIMITS = {
        "simple_transfer": 21000,
        "erc20_transfer": 65000,
        "uniswap_swap": 180000,
        "complex_defi": 350000,
    }

    # Chain native token prices (USD) - will be fetched from oracle
    NATIVE_TOKEN_PRICES = {
        Chain.BASE: 3000.0,      # ETH
        Chain.ARBITRUM: 3000.0,
        Chain.OPTIMISM: 3000.0,
        Chain.ETHEREUM: 3000.0,
        Chain.POLYGON: 0.80,     # MATIC
        Chain.SOLANA: 120.0,     # SOL
    }

    def __init__(
        self,
        max_gas_pct: float = 0.02,  # 2% max
        cache_duration: int = 30,    # Cache gas prices for 30 seconds
    ):
        """
        Initialize gas manager.

        Args:
            max_gas_pct: Max gas as % of trade value
            cache_duration: Cache duration in seconds
        """
        self.max_gas_pct = max_gas_pct
        self.cache_duration = cache_duration

        # Gas price cache
        self._gas_cache: Dict[Chain, GasPrice] = {}
        self._last_update: Dict[Chain, datetime] = {}

        logger.info(f"Gas Manager initialized | Max gas: {max_gas_pct*100:.1f}% of trade")

    def get_gas_price(self, chain: Chain, speed: GasSpeed = GasSpeed.STANDARD) -> float:
        """
        Get current gas price for chain.

        Args:
            chain: Chain to query
            speed: Desired speed (slow/standard/fast/instant)

        Returns:
            Gas price in Gwei (or lamports for Solana)
        """
        # Check cache
        if chain in self._gas_cache:
            last_update = self._last_update.get(chain)
            if last_update and (datetime.now() - last_update).seconds < self.cache_duration:
                prices = self._gas_cache[chain]
                return getattr(prices, speed.value)

        # Fetch fresh gas prices
        prices = self._fetch_gas_prices(chain)
        self._gas_cache[chain] = prices
        self._last_update[chain] = datetime.now()

        return getattr(prices, speed.value)

    def _fetch_gas_prices(self, chain: Chain) -> GasPrice:
        """
        Fetch current gas prices from chain.

        Args:
            chain: Chain to query

        Returns:
            Gas prices
        """
        # TODO: Implement real gas price fetching
        # For now, return mock data based on chain

        if chain == Chain.SOLANA:
            # Solana uses lamports per signature
            return GasPrice(
                chain=chain,
                slow=5000,      # 5000 lamports (~$0.0006)
                standard=5000,
                fast=10000,
                instant=20000,
                timestamp=datetime.now()
            )

        elif chain == Chain.ETHEREUM:
            # Ethereum mainnet (expensive)
            return GasPrice(
                chain=chain,
                slow=20.0,      # 20 Gwei
                standard=30.0,
                fast=50.0,
                instant=80.0,
                timestamp=datetime.now(),
                base_fee=25.0,
                priority_fee=2.0
            )

        elif chain == Chain.BASE:
            # Base L2 (cheap)
            return GasPrice(
                chain=chain,
                slow=0.1,       # 0.1 Gwei (very cheap)
                standard=0.2,
                fast=0.5,
                instant=1.0,
                timestamp=datetime.now(),
                base_fee=0.15,
                priority_fee=0.05
            )

        elif chain == Chain.ARBITRUM:
            # Arbitrum (cheap)
            return GasPrice(
                chain=chain,
                slow=0.2,
                standard=0.5,
                fast=1.0,
                instant=2.0,
                timestamp=datetime.now()
            )

        elif chain == Chain.OPTIMISM:
            # Optimism (cheap)
            return GasPrice(
                chain=chain,
                slow=0.3,
                standard=0.6,
                fast=1.2,
                instant=2.5,
                timestamp=datetime.now()
            )

        elif chain == Chain.POLYGON:
            # Polygon (cheap)
            return GasPrice(
                chain=chain,
                slow=30.0,      # Higher Gwei but MATIC is cheap
                standard=50.0,
                fast=80.0,
                instant=150.0,
                timestamp=datetime.now()
            )

        else:
            # Default
            return GasPrice(
                chain=chain,
                slow=1.0,
                standard=2.0,
                fast=5.0,
                instant=10.0,
                timestamp=datetime.now()
            )

    def estimate_gas(
        self,
        chain: Chain,
        operation: Literal["simple_transfer", "erc20_transfer", "uniswap_swap", "complex_defi"],
        trade_value_usd: float,
        speed: GasSpeed = GasSpeed.STANDARD
    ) -> GasEstimate:
        """
        Estimate gas cost for operation.

        Args:
            chain: Chain
            operation: Type of operation
            trade_value_usd: Trade value in USD
            speed: Gas speed

        Returns:
            Gas estimate
        """
        # Get gas limit for operation
        gas_limit = self.DEFAULT_GAS_LIMITS.get(operation, 200000)

        # Get gas price
        gas_price_gwei = self.get_gas_price(chain, speed)

        # Calculate total gas cost
        if chain == Chain.SOLANA:
            # Solana: lamports per signature
            total_gas_native = gas_price_gwei / 1e9  # lamports to SOL
        else:
            # EVM: gas_limit * gas_price
            gas_price_eth = gas_price_gwei / 1e9  # Gwei to ETH
            total_gas_native = gas_limit * gas_price_eth

        # Convert to USD
        native_price = self.NATIVE_TOKEN_PRICES.get(chain, 1.0)
        total_gas_usd = total_gas_native * native_price

        # Calculate percentage of trade
        percentage = (total_gas_usd / trade_value_usd) * 100 if trade_value_usd > 0 else 0

        return GasEstimate(
            chain=chain,
            gas_limit=gas_limit,
            gas_price=gas_price_gwei,
            total_gas_eth=total_gas_native,
            total_gas_usd=total_gas_usd,
            percentage_of_trade=percentage / 100,  # As decimal
            speed=speed,
            timestamp=datetime.now()
        )

    def should_execute_trade(
        self,
        chain: Chain,
        trade_value_usd: float,
        operation: str = "uniswap_swap",
        speed: GasSpeed = GasSpeed.STANDARD
    ) -> tuple[bool, str, GasEstimate]:
        """
        Check if trade should be executed based on gas costs.

        Args:
            chain: Chain
            trade_value_usd: Trade value in USD
            operation: Operation type
            speed: Gas speed

        Returns:
            (should_execute, reason, gas_estimate)
        """
        estimate = self.estimate_gas(chain, operation, trade_value_usd, speed)

        if estimate.percentage_of_trade > self.max_gas_pct:
            return (
                False,
                f"Gas too high: {estimate.percentage_of_trade*100:.2f}% > {self.max_gas_pct*100:.1f}%",
                estimate
            )

        if estimate.percentage_of_trade > self.max_gas_pct * 0.5:
            return (
                True,
                f"⚠️  High gas: {estimate.percentage_of_trade*100:.2f}% of trade value",
                estimate
            )

        return (
            True,
            f"✅ Gas acceptable: {estimate.percentage_of_trade*100:.2f}% of trade value",
            estimate
        )

    def get_optimal_speed(
        self,
        chain: Chain,
        trade_value_usd: float,
        urgent: bool = False
    ) -> GasSpeed:
        """
        Get optimal gas speed for trade.

        Args:
            chain: Chain
            trade_value_usd: Trade value in USD
            urgent: Whether trade is urgent

        Returns:
            Recommended gas speed
        """
        if urgent:
            return GasSpeed.FAST

        # Try each speed from slow to fast
        for speed in [GasSpeed.SLOW, GasSpeed.STANDARD, GasSpeed.FAST]:
            should_execute, _, estimate = self.should_execute_trade(
                chain, trade_value_usd, speed=speed
            )
            if should_execute and estimate.percentage_of_trade < self.max_gas_pct * 0.75:
                return speed

        # Default to standard
        return GasSpeed.STANDARD

    def wait_for_low_gas(
        self,
        chain: Chain,
        target_gwei: float,
        timeout: int = 300,  # 5 minutes
        check_interval: int = 10
    ) -> bool:
        """
        Wait for gas price to drop below target.

        Args:
            chain: Chain
            target_gwei: Target gas price in Gwei
            timeout: Max wait time in seconds
            check_interval: Check interval in seconds

        Returns:
            True if gas dropped below target, False if timeout
        """
        start_time = time.time()

        logger.info(f"Waiting for gas on {chain.value} to drop below {target_gwei} Gwei...")

        while time.time() - start_time < timeout:
            current_gas = self.get_gas_price(chain, GasSpeed.STANDARD)

            if current_gas <= target_gwei:
                logger.info(f"✅ Gas dropped to {current_gas} Gwei (target: {target_gwei})")
                return True

            elapsed = int(time.time() - start_time)
            logger.debug(f"Gas: {current_gas} Gwei | Waiting... ({elapsed}/{timeout}s)")

            time.sleep(check_interval)

        logger.warning(f"⏱️  Timeout waiting for low gas on {chain.value}")
        return False

    def get_all_gas_prices(self) -> Dict[Chain, GasPrice]:
        """Get current gas prices for all chains."""
        return {
            chain: self._fetch_gas_prices(chain)
            for chain in Chain
        }


# Example usage
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    print("\n" + "="*70)
    print("GAS MANAGER TEST")
    print("="*70)

    manager = GasManager(max_gas_pct=0.02)

    # Test 1: Get gas prices
    print("\n--- Test 1: Gas Prices ---")
    for chain in [Chain.BASE, Chain.ETHEREUM, Chain.SOLANA]:
        prices = manager._fetch_gas_prices(chain)
        print(f"{chain.value}:")
        print(f"  Slow: {prices.slow} | Standard: {prices.standard} | Fast: {prices.fast}")

    # Test 2: Estimate gas for trade
    print("\n--- Test 2: Gas Estimation ---")
    test_cases = [
        (Chain.BASE, 10000.0, "Large trade"),
        (Chain.BASE, 100.0, "Small trade"),
        (Chain.ETHEREUM, 10000.0, "Mainnet trade"),
    ]

    for chain, trade_value, description in test_cases:
        estimate = manager.estimate_gas(chain, "uniswap_swap", trade_value)
        print(f"\n{description} ({chain.value}):")
        print(f"  Trade value: ${trade_value:,.2f}")
        print(f"  Gas cost: ${estimate.total_gas_usd:.2f}")
        print(f"  Percentage: {estimate.percentage_of_trade*100:.2f}%")

    # Test 3: Should execute trade?
    print("\n--- Test 3: Trade Execution Decision ---")
    for chain, trade_value, description in test_cases:
        should_exec, reason, estimate = manager.should_execute_trade(
            chain, trade_value
        )
        status = "✅ EXECUTE" if should_exec else "❌ REJECT"
        print(f"\n{description}: {status}")
        print(f"  {reason}")

    # Test 4: Optimal speed
    print("\n--- Test 4: Optimal Gas Speed ---")
    for chain, trade_value, description in test_cases:
        optimal = manager.get_optimal_speed(chain, trade_value)
        print(f"{description}: {optimal.value}")

    print("\n" + "="*70)
    print("✅ Gas Manager ready!")
    print("="*70)
