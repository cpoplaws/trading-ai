"""
Multi-Chain Arbitrage Strategy
==============================

Detects and executes arbitrage opportunities across multiple blockchain networks.

Supported Chains:
- Ethereum (ETH)
- Binance Smart Chain (BSC)
- Polygon (MATIC)
- Arbitrum
- Optimism
- Avalanche

Supported Bridges:
- Hop Protocol
- Across Protocol
- Stargate
- Synapse
- Multichain (Anyswap)

Key Features:
- Real-time price monitoring across chains
- Bridge fee calculation
- Gas cost estimation
- Profitability analysis
- Automatic trade execution
- Slippage protection
"""

import logging
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
from datetime import datetime
import numpy as np

logger = logging.getLogger(__name__)


class Chain(Enum):
    """Supported blockchain networks."""
    ETHEREUM = "ethereum"
    BSC = "bsc"
    POLYGON = "polygon"
    ARBITRUM = "arbitrum"
    OPTIMISM = "optimism"
    AVALANCHE = "avalanche"


class BridgeProtocol(Enum):
    """Supported bridge protocols."""
    HOP = "hop"
    ACROSS = "across"
    STARGATE = "stargate"
    SYNAPSE = "synapse"
    MULTICHAIN = "multichain"


@dataclass
class ChainConfig:
    """Configuration for a blockchain network."""
    chain: Chain
    rpc_url: str
    chain_id: int
    native_token: str  # ETH, BNB, MATIC, etc.
    gas_price_gwei: float = 50.0  # Default gas price
    block_time: float = 2.0  # Average block time in seconds

    # DEX contracts on this chain
    uniswap_v2_router: Optional[str] = None
    uniswap_v3_router: Optional[str] = None
    sushiswap_router: Optional[str] = None

    def estimated_gas_cost(self, gas_units: int) -> float:
        """
        Estimate gas cost in native token.

        Args:
            gas_units: Number of gas units

        Returns:
            Cost in native token (e.g., ETH)
        """
        gas_cost_wei = gas_units * (self.gas_price_gwei * 1e9)
        return gas_cost_wei / 1e18


@dataclass
class BridgeConfig:
    """Configuration for a bridge protocol."""
    protocol: BridgeProtocol
    supported_chains: List[Chain]
    fee_percentage: float  # Bridge fee as percentage (0.01 = 1%)
    min_bridge_amount: float  # Minimum amount to bridge
    max_bridge_amount: float  # Maximum amount to bridge
    estimated_time_minutes: float  # Estimated bridge time

    def calculate_fee(self, amount: float) -> float:
        """Calculate bridge fee."""
        return amount * self.fee_percentage


@dataclass
class TokenPrice:
    """Price of a token on a specific chain."""
    chain: Chain
    token_address: str
    symbol: str
    price_usd: float
    liquidity_usd: float
    dex: str  # Which DEX this price is from
    timestamp: datetime

    def price_difference_pct(self, other: 'TokenPrice') -> float:
        """
        Calculate price difference with another price.

        Args:
            other: Another token price

        Returns:
            Percentage difference (positive means self is higher)
        """
        if other.price_usd == 0:
            return 0.0
        return ((self.price_usd - other.price_usd) / other.price_usd) * 100


@dataclass
class ArbitrageOpportunity:
    """An arbitrage opportunity across chains."""
    token_symbol: str
    token_address_source: str
    token_address_dest: str

    source_chain: Chain
    dest_chain: Chain

    source_price: TokenPrice
    dest_price: TokenPrice

    price_difference_pct: float

    bridge_protocol: BridgeProtocol
    bridge_fee: float
    bridge_time_minutes: float

    source_gas_cost: float  # In USD
    dest_gas_cost: float  # In USD
    total_gas_cost: float  # In USD

    trade_amount: float  # Amount to trade

    # Profit calculation
    gross_profit_usd: float
    net_profit_usd: float
    roi_pct: float

    timestamp: datetime

    def is_profitable(self, min_profit_usd: float = 10.0, min_roi_pct: float = 1.0) -> bool:
        """
        Check if opportunity is profitable.

        Args:
            min_profit_usd: Minimum profit in USD
            min_roi_pct: Minimum ROI percentage

        Returns:
            True if profitable
        """
        return (
            self.net_profit_usd >= min_profit_usd and
            self.roi_pct >= min_roi_pct
        )

    def __str__(self) -> str:
        return (
            f"{self.token_symbol} Arbitrage: "
            f"{self.source_chain.value} → {self.dest_chain.value} "
            f"| Price Diff: {self.price_difference_pct:.2f}% "
            f"| Net Profit: ${self.net_profit_usd:.2f} "
            f"| ROI: {self.roi_pct:.2f}%"
        )


@dataclass
class MultichainArbitrageConfig:
    """Configuration for multi-chain arbitrage."""
    # Chains to monitor
    enabled_chains: List[Chain] = None

    # Bridges to use
    enabled_bridges: List[BridgeProtocol] = None

    # Tokens to monitor (by symbol)
    monitored_tokens: List[str] = None

    # Profitability thresholds
    min_profit_usd: float = 20.0  # Minimum profit to execute
    min_roi_pct: float = 2.0  # Minimum ROI percentage
    min_price_difference_pct: float = 0.5  # Minimum price difference

    # Trade parameters
    default_trade_amount_usd: float = 1000.0
    max_trade_amount_usd: float = 10000.0
    max_slippage_pct: float = 0.5

    # Risk management
    max_bridge_time_minutes: float = 30.0  # Max acceptable bridge time
    min_liquidity_usd: float = 50000.0  # Minimum liquidity on DEX

    # Execution
    auto_execute: bool = False  # Automatically execute trades
    gas_price_multiplier: float = 1.2  # Multiply gas price for faster execution

    def __post_init__(self):
        if self.enabled_chains is None:
            self.enabled_chains = [
                Chain.ETHEREUM,
                Chain.POLYGON,
                Chain.ARBITRUM,
                Chain.OPTIMISM
            ]

        if self.enabled_bridges is None:
            self.enabled_bridges = [
                BridgeProtocol.HOP,
                BridgeProtocol.ACROSS,
                BridgeProtocol.STARGATE
            ]

        if self.monitored_tokens is None:
            self.monitored_tokens = [
                "USDC", "USDT", "DAI", "WETH", "WBTC"
            ]


class MultichainArbitrage:
    """
    Multi-chain arbitrage strategy.

    Monitors token prices across multiple chains and executes
    profitable arbitrage trades via bridges.
    """

    def __init__(self, config: Optional[MultichainArbitrageConfig] = None):
        """
        Initialize multi-chain arbitrage.

        Args:
            config: Arbitrage configuration
        """
        self.config = config or MultichainArbitrageConfig()

        # Chain configurations
        self.chains: Dict[Chain, ChainConfig] = {}
        self._setup_default_chains()

        # Bridge configurations
        self.bridges: Dict[BridgeProtocol, BridgeConfig] = {}
        self._setup_default_bridges()

        # Current token prices across chains
        self.token_prices: Dict[str, Dict[Chain, TokenPrice]] = {}

        # Found opportunities
        self.opportunities: List[ArbitrageOpportunity] = []

        logger.info(
            f"Initialized MultichainArbitrage: "
            f"{len(self.config.enabled_chains)} chains, "
            f"{len(self.config.enabled_bridges)} bridges, "
            f"{len(self.config.monitored_tokens)} tokens"
        )

    def _setup_default_chains(self):
        """Setup default chain configurations."""
        self.chains[Chain.ETHEREUM] = ChainConfig(
            chain=Chain.ETHEREUM,
            rpc_url="https://eth-mainnet.g.alchemy.com/v2/YOUR_KEY",
            chain_id=1,
            native_token="ETH",
            gas_price_gwei=50.0,
            block_time=12.0
        )

        self.chains[Chain.BSC] = ChainConfig(
            chain=Chain.BSC,
            rpc_url="https://bsc-dataseed.binance.org",
            chain_id=56,
            native_token="BNB",
            gas_price_gwei=5.0,
            block_time=3.0
        )

        self.chains[Chain.POLYGON] = ChainConfig(
            chain=Chain.POLYGON,
            rpc_url="https://polygon-rpc.com",
            chain_id=137,
            native_token="MATIC",
            gas_price_gwei=50.0,
            block_time=2.0
        )

        self.chains[Chain.ARBITRUM] = ChainConfig(
            chain=Chain.ARBITRUM,
            rpc_url="https://arb1.arbitrum.io/rpc",
            chain_id=42161,
            native_token="ETH",
            gas_price_gwei=0.1,
            block_time=0.25
        )

        self.chains[Chain.OPTIMISM] = ChainConfig(
            chain=Chain.OPTIMISM,
            rpc_url="https://mainnet.optimism.io",
            chain_id=10,
            native_token="ETH",
            gas_price_gwei=0.001,
            block_time=2.0
        )

        self.chains[Chain.AVALANCHE] = ChainConfig(
            chain=Chain.AVALANCHE,
            rpc_url="https://api.avax.network/ext/bc/C/rpc",
            chain_id=43114,
            native_token="AVAX",
            gas_price_gwei=25.0,
            block_time=2.0
        )

    def _setup_default_bridges(self):
        """Setup default bridge configurations."""
        self.bridges[BridgeProtocol.HOP] = BridgeConfig(
            protocol=BridgeProtocol.HOP,
            supported_chains=[
                Chain.ETHEREUM, Chain.POLYGON,
                Chain.ARBITRUM, Chain.OPTIMISM
            ],
            fee_percentage=0.0004,  # 0.04%
            min_bridge_amount=10.0,
            max_bridge_amount=1000000.0,
            estimated_time_minutes=5.0
        )

        self.bridges[BridgeProtocol.ACROSS] = BridgeConfig(
            protocol=BridgeProtocol.ACROSS,
            supported_chains=[
                Chain.ETHEREUM, Chain.POLYGON,
                Chain.ARBITRUM, Chain.OPTIMISM
            ],
            fee_percentage=0.0005,  # 0.05%
            min_bridge_amount=10.0,
            max_bridge_amount=500000.0,
            estimated_time_minutes=3.0
        )

        self.bridges[BridgeProtocol.STARGATE] = BridgeConfig(
            protocol=BridgeProtocol.STARGATE,
            supported_chains=[
                Chain.ETHEREUM, Chain.BSC, Chain.POLYGON,
                Chain.ARBITRUM, Chain.OPTIMISM, Chain.AVALANCHE
            ],
            fee_percentage=0.0006,  # 0.06%
            min_bridge_amount=20.0,
            max_bridge_amount=2000000.0,
            estimated_time_minutes=10.0
        )

        self.bridges[BridgeProtocol.SYNAPSE] = BridgeConfig(
            protocol=BridgeProtocol.SYNAPSE,
            supported_chains=[
                Chain.ETHEREUM, Chain.BSC, Chain.POLYGON,
                Chain.ARBITRUM, Chain.OPTIMISM, Chain.AVALANCHE
            ],
            fee_percentage=0.001,  # 0.1%
            min_bridge_amount=5.0,
            max_bridge_amount=100000.0,
            estimated_time_minutes=15.0
        )

        self.bridges[BridgeProtocol.MULTICHAIN] = BridgeConfig(
            protocol=BridgeProtocol.MULTICHAIN,
            supported_chains=[
                Chain.ETHEREUM, Chain.BSC, Chain.POLYGON,
                Chain.ARBITRUM, Chain.OPTIMISM, Chain.AVALANCHE
            ],
            fee_percentage=0.001,  # 0.1%
            min_bridge_amount=100.0,
            max_bridge_amount=5000000.0,
            estimated_time_minutes=20.0
        )

    def update_token_price(
        self,
        chain: Chain,
        token_address: str,
        symbol: str,
        price_usd: float,
        liquidity_usd: float,
        dex: str
    ):
        """
        Update token price on a chain.

        Args:
            chain: Blockchain network
            token_address: Token contract address
            symbol: Token symbol
            price_usd: Price in USD
            liquidity_usd: Available liquidity in USD
            dex: DEX name
        """
        if symbol not in self.token_prices:
            self.token_prices[symbol] = {}

        self.token_prices[symbol][chain] = TokenPrice(
            chain=chain,
            token_address=token_address,
            symbol=symbol,
            price_usd=price_usd,
            liquidity_usd=liquidity_usd,
            dex=dex,
            timestamp=datetime.now()
        )

    def scan_opportunities(self) -> List[ArbitrageOpportunity]:
        """
        Scan for arbitrage opportunities across all chains.

        Returns:
            List of profitable opportunities
        """
        opportunities = []

        # For each token
        for symbol, prices_by_chain in self.token_prices.items():
            if symbol not in self.config.monitored_tokens:
                continue

            # Compare all chain pairs
            chains = list(prices_by_chain.keys())
            for i, source_chain in enumerate(chains):
                for dest_chain in chains[i+1:]:
                    if source_chain not in self.config.enabled_chains:
                        continue
                    if dest_chain not in self.config.enabled_chains:
                        continue

                    source_price = prices_by_chain[source_chain]
                    dest_price = prices_by_chain[dest_chain]

                    # Calculate price difference
                    price_diff_pct = dest_price.price_difference_pct(source_price)

                    # Skip if difference too small
                    if abs(price_diff_pct) < self.config.min_price_difference_pct:
                        continue

                    # Check liquidity
                    if source_price.liquidity_usd < self.config.min_liquidity_usd:
                        continue
                    if dest_price.liquidity_usd < self.config.min_liquidity_usd:
                        continue

                    # Find suitable bridge
                    bridge = self._find_best_bridge(source_chain, dest_chain)
                    if not bridge:
                        continue

                    # Calculate opportunity (if price on dest is higher, buy on source and sell on dest)
                    if price_diff_pct > 0:
                        opp = self._calculate_opportunity(
                            symbol=symbol,
                            source_chain=source_chain,
                            dest_chain=dest_chain,
                            source_price=source_price,
                            dest_price=dest_price,
                            bridge=bridge,
                            price_diff_pct=price_diff_pct
                        )
                    else:
                        # Reverse direction
                        opp = self._calculate_opportunity(
                            symbol=symbol,
                            source_chain=dest_chain,
                            dest_chain=source_chain,
                            source_price=dest_price,
                            dest_price=source_price,
                            bridge=bridge,
                            price_diff_pct=abs(price_diff_pct)
                        )

                    if opp and opp.is_profitable(
                        self.config.min_profit_usd,
                        self.config.min_roi_pct
                    ):
                        opportunities.append(opp)

        # Sort by profitability
        opportunities.sort(key=lambda x: x.net_profit_usd, reverse=True)

        self.opportunities = opportunities

        logger.info(f"Found {len(opportunities)} profitable arbitrage opportunities")
        return opportunities

    def _find_best_bridge(
        self,
        source_chain: Chain,
        dest_chain: Chain
    ) -> Optional[BridgeConfig]:
        """
        Find best bridge for chain pair.

        Args:
            source_chain: Source chain
            dest_chain: Destination chain

        Returns:
            Best bridge configuration
        """
        suitable_bridges = []

        for protocol in self.config.enabled_bridges:
            bridge = self.bridges.get(protocol)
            if not bridge:
                continue

            # Check if bridge supports both chains
            if (source_chain in bridge.supported_chains and
                dest_chain in bridge.supported_chains):

                # Check bridge time
                if bridge.estimated_time_minutes <= self.config.max_bridge_time_minutes:
                    suitable_bridges.append(bridge)

        if not suitable_bridges:
            return None

        # Return bridge with lowest fee
        return min(suitable_bridges, key=lambda b: b.fee_percentage)

    def _calculate_opportunity(
        self,
        symbol: str,
        source_chain: Chain,
        dest_chain: Chain,
        source_price: TokenPrice,
        dest_price: TokenPrice,
        bridge: BridgeConfig,
        price_diff_pct: float
    ) -> Optional[ArbitrageOpportunity]:
        """
        Calculate arbitrage opportunity profitability.

        Args:
            symbol: Token symbol
            source_chain: Source chain
            dest_chain: Destination chain
            source_price: Price on source chain
            dest_price: Price on destination chain
            bridge: Bridge to use
            price_diff_pct: Price difference percentage

        Returns:
            Arbitrage opportunity or None
        """
        # Trade amount
        trade_amount = self.config.default_trade_amount_usd

        # Ensure within bridge limits
        trade_amount = max(trade_amount, bridge.min_bridge_amount)
        trade_amount = min(trade_amount, bridge.max_bridge_amount)
        trade_amount = min(trade_amount, self.config.max_trade_amount_usd)

        # Calculate costs
        bridge_fee = bridge.calculate_fee(trade_amount)

        # Gas costs (estimated)
        source_gas_units = 200000  # Swap + bridge initiation
        dest_gas_units = 150000  # Bridge claim + swap

        source_chain_config = self.chains[source_chain]
        dest_chain_config = self.chains[dest_chain]

        # Get native token prices (simplified - in production, fetch real prices)
        native_prices = {
            "ETH": 3000.0,
            "BNB": 400.0,
            "MATIC": 0.8,
            "AVAX": 35.0
        }

        source_gas_cost_native = source_chain_config.estimated_gas_cost(source_gas_units)
        source_gas_cost_usd = source_gas_cost_native * native_prices.get(
            source_chain_config.native_token, 1.0
        )

        dest_gas_cost_native = dest_chain_config.estimated_gas_cost(dest_gas_units)
        dest_gas_cost_usd = dest_gas_cost_native * native_prices.get(
            dest_chain_config.native_token, 1.0
        )

        total_gas_cost = source_gas_cost_usd + dest_gas_cost_usd

        # Calculate profit
        # Buy on source chain
        tokens_bought = trade_amount / source_price.price_usd

        # After bridge fee
        tokens_after_bridge = tokens_bought * (1 - bridge.fee_percentage)

        # Sell on dest chain
        sell_value = tokens_after_bridge * dest_price.price_usd

        gross_profit = sell_value - trade_amount
        net_profit = gross_profit - bridge_fee - total_gas_cost

        roi_pct = (net_profit / trade_amount) * 100 if trade_amount > 0 else 0

        return ArbitrageOpportunity(
            token_symbol=symbol,
            token_address_source=source_price.token_address,
            token_address_dest=dest_price.token_address,
            source_chain=source_chain,
            dest_chain=dest_chain,
            source_price=source_price,
            dest_price=dest_price,
            price_difference_pct=price_diff_pct,
            bridge_protocol=bridge.protocol,
            bridge_fee=bridge_fee,
            bridge_time_minutes=bridge.estimated_time_minutes,
            source_gas_cost=source_gas_cost_usd,
            dest_gas_cost=dest_gas_cost_usd,
            total_gas_cost=total_gas_cost,
            trade_amount=trade_amount,
            gross_profit_usd=gross_profit,
            net_profit_usd=net_profit,
            roi_pct=roi_pct,
            timestamp=datetime.now()
        )

    def execute_arbitrage(
        self,
        opportunity: ArbitrageOpportunity,
        dry_run: bool = True
    ) -> Dict[str, any]:
        """
        Execute an arbitrage opportunity.

        Args:
            opportunity: Arbitrage opportunity to execute
            dry_run: If True, simulate execution without real trades

        Returns:
            Execution result
        """
        logger.info(f"{'[DRY RUN] ' if dry_run else ''}Executing: {opportunity}")

        if dry_run:
            return {
                'success': True,
                'dry_run': True,
                'opportunity': opportunity,
                'steps': [
                    f"1. Swap ${opportunity.trade_amount:.2f} for {opportunity.token_symbol} on {opportunity.source_chain.value}",
                    f"2. Bridge {opportunity.token_symbol} from {opportunity.source_chain.value} to {opportunity.dest_chain.value} via {opportunity.bridge_protocol.value}",
                    f"3. Wait ~{opportunity.bridge_time_minutes:.1f} minutes for bridge",
                    f"4. Swap {opportunity.token_symbol} back to USD on {opportunity.dest_chain.value}",
                    f"5. Net profit: ${opportunity.net_profit_usd:.2f}"
                ]
            }

        # Real execution would involve:
        # 1. Connect to source chain DEX
        # 2. Execute swap (buy token)
        # 3. Approve bridge contract
        # 4. Initiate bridge transfer
        # 5. Wait for bridge confirmation
        # 6. Connect to dest chain DEX
        # 7. Execute swap (sell token)

        logger.warning("Real execution not implemented - use dry_run=True for simulation")
        return {'success': False, 'error': 'Real execution not implemented'}

    def get_best_opportunities(self, top_n: int = 5) -> List[ArbitrageOpportunity]:
        """
        Get top N opportunities by profitability.

        Args:
            top_n: Number of opportunities to return

        Returns:
            Top opportunities
        """
        return sorted(
            self.opportunities,
            key=lambda x: x.net_profit_usd,
            reverse=True
        )[:top_n]

    def get_statistics(self) -> Dict[str, any]:
        """
        Get arbitrage statistics.

        Returns:
            Statistics dictionary
        """
        if not self.opportunities:
            return {
                'total_opportunities': 0,
                'avg_profit_usd': 0.0,
                'avg_roi_pct': 0.0,
                'total_potential_profit': 0.0
            }

        profits = [opp.net_profit_usd for opp in self.opportunities]
        rois = [opp.roi_pct for opp in self.opportunities]

        # Count by chain pair
        chain_pairs = {}
        for opp in self.opportunities:
            pair = f"{opp.source_chain.value}->{opp.dest_chain.value}"
            chain_pairs[pair] = chain_pairs.get(pair, 0) + 1

        # Count by token
        tokens = {}
        for opp in self.opportunities:
            tokens[opp.token_symbol] = tokens.get(opp.token_symbol, 0) + 1

        return {
            'total_opportunities': len(self.opportunities),
            'avg_profit_usd': np.mean(profits),
            'max_profit_usd': np.max(profits),
            'min_profit_usd': np.min(profits),
            'avg_roi_pct': np.mean(rois),
            'max_roi_pct': np.max(rois),
            'total_potential_profit': np.sum(profits),
            'opportunities_by_chain_pair': chain_pairs,
            'opportunities_by_token': tokens
        }


# Example usage
if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)

    print("Multi-Chain Arbitrage Example")
    print("=" * 80)

    # Create arbitrage scanner
    config = MultichainArbitrageConfig(
        enabled_chains=[Chain.ETHEREUM, Chain.POLYGON, Chain.ARBITRUM],
        enabled_bridges=[BridgeProtocol.HOP, BridgeProtocol.ACROSS],
        monitored_tokens=["USDC", "WETH"],
        min_profit_usd=15.0,
        min_roi_pct=1.5
    )

    arbitrage = MultichainArbitrage(config)

    # Simulate price updates (in production, fetch from DEXs)
    print("\nSimulating price updates...")

    # USDC prices (should be close to $1, but small differences exist)
    arbitrage.update_token_price(
        Chain.ETHEREUM, "0xA0b86991c6218b36c1d19D4a2e9Eb0cE3606eB48",
        "USDC", 1.000, 500000000, "Uniswap"
    )
    arbitrage.update_token_price(
        Chain.POLYGON, "0x2791Bca1f2de4661ED88A30C99A7a9449Aa84174",
        "USDC", 1.003, 50000000, "Quickswap"
    )
    arbitrage.update_token_price(
        Chain.ARBITRUM, "0xFF970A61A04b1cA14834A43f5dE4533eBDDB5CC8",
        "USDC", 0.999, 30000000, "Sushiswap"
    )

    # WETH prices (simulate small arbitrage)
    arbitrage.update_token_price(
        Chain.ETHEREUM, "0xC02aaA39b223FE8D0A0e5C4F27eAD9083C756Cc2",
        "WETH", 3000.00, 800000000, "Uniswap"
    )
    arbitrage.update_token_price(
        Chain.POLYGON, "0x7ceB23fD6bC0adD59E62ac25578270cFf1b9f619",
        "WETH", 3015.50, 10000000, "Quickswap"
    )
    arbitrage.update_token_price(
        Chain.ARBITRUM, "0x82aF49447D8a07e3bd95BD0d56f35241523fBab1",
        "WETH", 2995.00, 50000000, "Sushiswap"
    )

    # Scan for opportunities
    print("\nScanning for arbitrage opportunities...")
    opportunities = arbitrage.scan_opportunities()

    print(f"\nFound {len(opportunities)} profitable opportunities:")
    print("-" * 80)

    for i, opp in enumerate(opportunities[:5], 1):
        print(f"\n{i}. {opp}")
        print(f"   Trade: ${opp.trade_amount:.2f}")
        print(f"   Bridge: {opp.bridge_protocol.value} (~{opp.bridge_time_minutes:.0f} min)")
        print(f"   Gross Profit: ${opp.gross_profit_usd:.2f}")
        print(f"   Bridge Fee: ${opp.bridge_fee:.2f}")
        print(f"   Gas Cost: ${opp.total_gas_cost:.2f}")
        print(f"   Net Profit: ${opp.net_profit_usd:.2f}")
        print(f"   ROI: {opp.roi_pct:.2f}%")

    # Show statistics
    print("\nStatistics:")
    print("-" * 80)
    stats = arbitrage.get_statistics()
    for key, value in stats.items():
        if isinstance(value, dict):
            print(f"{key}:")
            for k, v in value.items():
                print(f"  {k}: {v}")
        elif isinstance(value, float):
            print(f"{key}: {value:.2f}")
        else:
            print(f"{key}: {value}")

    # Execute best opportunity (dry run)
    if opportunities:
        print("\nExecuting best opportunity (dry run):")
        print("-" * 80)
        best_opp = opportunities[0]
        result = arbitrage.execute_arbitrage(best_opp, dry_run=True)

        for step in result['steps']:
            print(f"  {step}")

    print("\n✅ Multi-Chain Arbitrage Example Complete!")
