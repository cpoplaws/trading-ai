"""
StrategyInstanceManager - Spawns and manages multiple strategy instances

Responsibilities:
1. Create N instances of each strategy for M chains/exchanges
2. Map strategies to appropriate trading pairs per chain
3. Distribute capital across instances
4. Manage instance lifecycle (start, stop, restart)
"""

from typing import Dict, List, Any, Optional
from dataclasses import dataclass
import logging
from enum import Enum

logger = logging.getLogger(__name__)


class Chain(Enum):
    """Supported chains/exchanges"""
    BASE = "Base"
    SOLANA = "Solana"
    ARBITRUM = "Arbitrum"
    OPTIMISM = "Optimism"
    POLYGON = "Polygon"
    CEX_BINANCE = "CEX:Binance"
    CEX_COINBASE = "CEX:Coinbase"
    CEX_KRAKEN = "CEX:Kraken"


@dataclass
class StrategyInstance:
    """A strategy instance running on a specific chain"""
    instance_id: str
    strategy_name: str
    strategy_class: Any  # The strategy class (MeanReversionStrategy, etc.)
    chain: Chain
    symbols: List[str]
    allocated_capital: float
    enabled: bool = False
    strategy_object: Optional[Any] = None  # Instantiated strategy


class StrategyInstanceManager:
    """
    Manages multiple instances of strategies across chains/exchanges.

    Example usage:
        manager = StrategyInstanceManager()

        # Spawn Mean Reversion on 3 venues
        manager.spawn_strategy_instances(
            strategy_name="mean_reversion",
            strategy_class=MeanReversionStrategy,
            chains=[Chain.BASE, Chain.SOLANA, Chain.CEX_BINANCE],
            capital_per_instance=1000.0
        )

        # This creates:
        # - mean_reversion_Base_001 trading WETH/USDC on Uniswap Base
        # - mean_reversion_Solana_002 trading SOL/USDC on Jupiter
        # - mean_reversion_CEX:Binance_003 trading BTC/USDT on Binance
    """

    # Trading pair mappings per chain
    CHAIN_SYMBOLS = {
        Chain.BASE: [
            "WETH/USDC",
            "WETH/USDbC",
            "cbETH/WETH",
        ],
        Chain.SOLANA: [
            "SOL/USDC",
            "SOL/USDT",
            "JUP/USDC",
        ],
        Chain.ARBITRUM: [
            "WETH/USDC",
            "ARB/USDC",
            "GMX/USDC",
        ],
        Chain.OPTIMISM: [
            "WETH/USDC",
            "OP/USDC",
        ],
        Chain.POLYGON: [
            "WMATIC/USDC",
            "WETH/USDC",
        ],
        Chain.CEX_BINANCE: [
            "BTC/USDT",
            "ETH/USDT",
            "SOL/USDT",
            "BNB/USDT",
        ],
        Chain.CEX_COINBASE: [
            "BTC/USD",
            "ETH/USD",
            "SOL/USD",
        ],
        Chain.CEX_KRAKEN: [
            "BTC/USD",
            "ETH/USD",
        ],
    }

    def __init__(self):
        self.instances: Dict[str, StrategyInstance] = {}
        self.instance_counter = 0
        logger.info("StrategyInstanceManager initialized")

    def spawn_strategy_instances(
        self,
        strategy_name: str,
        strategy_class: Any,
        chains: List[Chain],
        capital_per_instance: float,
        symbols_per_chain: Optional[Dict[Chain, List[str]]] = None,
    ) -> List[str]:
        """
        Spawn multiple instances of a strategy across different chains.

        Args:
            strategy_name: Name of the strategy (e.g., "mean_reversion")
            strategy_class: The strategy class to instantiate
            chains: List of chains/exchanges to deploy on
            capital_per_instance: Capital allocated to each instance
            symbols_per_chain: Optional custom symbol mapping

        Returns:
            List of instance IDs created
        """
        created_ids = []

        for chain in chains:
            # Get symbols for this chain
            if symbols_per_chain and chain in symbols_per_chain:
                symbols = symbols_per_chain[chain]
            else:
                symbols = self.CHAIN_SYMBOLS.get(chain, ["BTC/USDT"])

            # Create instance
            self.instance_counter += 1
            instance_id = f"{strategy_name}_{chain.value}_{self.instance_counter:03d}"

            instance = StrategyInstance(
                instance_id=instance_id,
                strategy_name=strategy_name,
                strategy_class=strategy_class,
                chain=chain,
                symbols=symbols,
                allocated_capital=capital_per_instance,
                enabled=False,
            )

            self.instances[instance_id] = instance
            created_ids.append(instance_id)

            logger.info(
                f"Spawned instance: {instance_id} | "
                f"Chain: {chain.value} | "
                f"Symbols: {symbols} | "
                f"Capital: ${capital_per_instance:,.2f}"
            )

        return created_ids

    def spawn_single_strategy_everywhere(
        self,
        strategy_name: str,
        strategy_class: Any,
        total_capital: float,
        chains: Optional[List[Chain]] = None,
    ) -> List[str]:
        """
        Spawn a strategy on ALL chains with equal capital allocation.

        Args:
            strategy_name: Strategy name
            strategy_class: Strategy class
            total_capital: Total capital to split across all chains
            chains: Optional list of chains (defaults to all)

        Returns:
            List of instance IDs
        """
        if chains is None:
            chains = [
                Chain.BASE,
                Chain.SOLANA,
                Chain.ARBITRUM,
                Chain.OPTIMISM,
                Chain.CEX_BINANCE,
                Chain.CEX_COINBASE,
            ]

        capital_per_instance = total_capital / len(chains)

        logger.info(
            f"Spawning {strategy_name} everywhere: "
            f"{len(chains)} chains × ${capital_per_instance:,.2f} = ${total_capital:,.2f}"
        )

        return self.spawn_strategy_instances(
            strategy_name=strategy_name,
            strategy_class=strategy_class,
            chains=chains,
            capital_per_instance=capital_per_instance,
        )

    def instantiate_strategy(self, instance_id: str) -> bool:
        """
        Actually instantiate the strategy object for an instance.

        This separates registration (spawn) from instantiation.
        """
        if instance_id not in self.instances:
            logger.error(f"Instance not found: {instance_id}")
            return False

        instance = self.instances[instance_id]

        try:
            # Instantiate the strategy with its symbols
            instance.strategy_object = instance.strategy_class(
                symbols=instance.symbols
            )
            logger.info(f"Instantiated strategy object for {instance_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to instantiate {instance_id}: {e}")
            return False

    def enable_instance(self, instance_id: str) -> bool:
        """Enable a strategy instance to start trading"""
        if instance_id not in self.instances:
            logger.error(f"Instance not found: {instance_id}")
            return False

        instance = self.instances[instance_id]

        # Instantiate if not already done
        if instance.strategy_object is None:
            if not self.instantiate_strategy(instance_id):
                return False

        instance.enabled = True
        logger.info(f"Enabled instance: {instance_id}")
        return True

    def disable_instance(self, instance_id: str) -> bool:
        """Disable a strategy instance to stop trading"""
        if instance_id not in self.instances:
            logger.error(f"Instance not found: {instance_id}")
            return False

        self.instances[instance_id].enabled = False
        logger.info(f"Disabled instance: {instance_id}")
        return True

    def get_enabled_instances(self) -> List[StrategyInstance]:
        """Get all enabled instances"""
        return [
            instance for instance in self.instances.values()
            if instance.enabled
        ]

    def get_instances_by_chain(self, chain: Chain) -> List[StrategyInstance]:
        """Get all instances for a specific chain"""
        return [
            instance for instance in self.instances.values()
            if instance.chain == chain
        ]

    def get_instances_by_strategy(self, strategy_name: str) -> List[StrategyInstance]:
        """Get all instances of a specific strategy"""
        return [
            instance for instance in self.instances.values()
            if instance.strategy_name == strategy_name
        ]

    def update_capital(self, instance_id: str, new_capital: float) -> bool:
        """Update allocated capital for an instance (rebalancing)"""
        if instance_id not in self.instances:
            logger.error(f"Instance not found: {instance_id}")
            return False

        old_capital = self.instances[instance_id].allocated_capital
        self.instances[instance_id].allocated_capital = new_capital

        logger.info(
            f"Capital updated: {instance_id} | "
            f"${old_capital:,.2f} → ${new_capital:,.2f}"
        )
        return True

    def get_instance_summary(self) -> Dict:
        """Get summary of all instances"""
        enabled_count = sum(1 for inst in self.instances.values() if inst.enabled)
        total_capital = sum(inst.allocated_capital for inst in self.instances.values())

        by_chain = {}
        for instance in self.instances.values():
            chain_name = instance.chain.value
            if chain_name not in by_chain:
                by_chain[chain_name] = {"count": 0, "capital": 0.0, "enabled": 0}
            by_chain[chain_name]["count"] += 1
            by_chain[chain_name]["capital"] += instance.allocated_capital
            if instance.enabled:
                by_chain[chain_name]["enabled"] += 1

        by_strategy = {}
        for instance in self.instances.values():
            strategy = instance.strategy_name
            if strategy not in by_strategy:
                by_strategy[strategy] = {"count": 0, "capital": 0.0, "enabled": 0}
            by_strategy[strategy]["count"] += 1
            by_strategy[strategy]["capital"] += instance.allocated_capital
            if instance.enabled:
                by_strategy[strategy]["enabled"] += 1

        return {
            "total_instances": len(self.instances),
            "enabled_instances": enabled_count,
            "total_capital": total_capital,
            "by_chain": by_chain,
            "by_strategy": by_strategy,
        }

    def get_instance(self, instance_id: str) -> Optional[StrategyInstance]:
        """Get a specific instance"""
        return self.instances.get(instance_id)

    def generate_signal(self, instance_id: str, market_data: Dict) -> Optional[str]:
        """
        Generate trading signal for an instance.

        Args:
            instance_id: Instance ID
            market_data: Current market data

        Returns:
            Signal: "BUY", "SELL", "HOLD", or None if error
        """
        if instance_id not in self.instances:
            logger.error(f"Instance not found: {instance_id}")
            return None

        instance = self.instances[instance_id]

        if not instance.enabled:
            return "HOLD"

        if instance.strategy_object is None:
            logger.error(f"Strategy not instantiated: {instance_id}")
            return None

        try:
            signal = instance.strategy_object.generate_signal(market_data)
            return signal.value if hasattr(signal, 'value') else str(signal)
        except Exception as e:
            logger.error(f"Error generating signal for {instance_id}: {e}")
            return None


# Example usage patterns
if __name__ == "__main__":
    # This demonstrates how the manager works
    logging.basicConfig(level=logging.INFO)

    manager = StrategyInstanceManager()

    # Scenario 1: Spawn Mean Reversion on select chains
    print("\n=== Scenario 1: Spawn strategy on specific chains ===")
    from apps.api.strategies.mean_reversion import MeanReversionStrategy

    ids = manager.spawn_strategy_instances(
        strategy_name="mean_reversion",
        strategy_class=MeanReversionStrategy,
        chains=[Chain.BASE, Chain.SOLANA, Chain.CEX_BINANCE],
        capital_per_instance=1000.0,
    )
    print(f"Created instances: {ids}")

    # Scenario 2: Spawn Momentum everywhere
    print("\n=== Scenario 2: Spawn strategy everywhere ===")
    # (Would need MomentumStrategy import)
    # ids = manager.spawn_single_strategy_everywhere(
    #     strategy_name="momentum",
    #     strategy_class=MomentumStrategy,
    #     total_capital=5000.0,
    # )

    # Enable instances
    print("\n=== Enabling instances ===")
    for instance_id in ids:
        manager.enable_instance(instance_id)

    # Get summary
    print("\n=== Instance Summary ===")
    summary = manager.get_instance_summary()
    print(f"Total instances: {summary['total_instances']}")
    print(f"Enabled: {summary['enabled_instances']}")
    print(f"Total capital: ${summary['total_capital']:,.2f}")
    print(f"\nBy chain:")
    for chain, data in summary['by_chain'].items():
        print(f"  {chain}: {data['count']} instances, ${data['capital']:,.2f}")
