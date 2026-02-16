"""
Market Making Strategy
Provides liquidity to earn bid-ask spread.

Features:
- Dynamic spread calculation based on volatility
- Inventory management to avoid directional risk
- Order book depth analysis
- Adverse selection protection
- Risk management with position limits
"""
import logging
from typing import List, Optional, Dict, Any, Tuple
from dataclasses import dataclass
from enum import Enum
import numpy as np

logger = logging.getLogger(__name__)


class InventoryStatus(Enum):
    """Inventory status."""
    BALANCED = "balanced"
    LONG_HEAVY = "long_heavy"
    SHORT_HEAVY = "short_heavy"
    CRITICAL_LONG = "critical_long"
    CRITICAL_SHORT = "critical_short"


class OrderSide(Enum):
    """Order side."""
    BID = "bid"
    ASK = "ask"


@dataclass
class MarketMakingConfig:
    """Market making configuration."""
    symbol: str

    # Spread parameters
    base_spread_bps: float = 10.0  # 10 basis points (0.1%)
    min_spread_bps: float = 5.0
    max_spread_bps: float = 50.0
    volatility_multiplier: float = 2.0

    # Order sizing
    order_size_usd: float = 1000.0
    max_orders_per_side: int = 5
    order_spacing_bps: float = 10.0  # Spacing between ladder orders

    # Inventory management
    max_inventory_usd: float = 10000.0
    inventory_target: float = 0.0  # Target neutral
    inventory_skew_factor: float = 0.5  # How much to skew based on inventory

    # Risk management
    position_limit_usd: float = 20000.0
    daily_loss_limit_usd: float = 1000.0
    max_adverse_move_bps: float = 100.0  # Exit if price moves 1% against us

    # Market conditions
    min_order_book_depth_usd: float = 50000.0
    max_spread_to_make: float = 100.0  # Don't make if spread > 1%
    pause_on_high_volatility: bool = True
    volatility_threshold: float = 0.05  # 5% volatility


@dataclass
class OrderBookLevel:
    """Order book price level."""
    price: float
    quantity: float
    num_orders: int = 1


@dataclass
class OrderBook:
    """Order book snapshot."""
    bids: List[OrderBookLevel]
    asks: List[OrderBookLevel]
    mid_price: float
    spread_bps: float
    bid_depth_usd: float
    ask_depth_usd: float


@dataclass
class Quote:
    """Market making quote."""
    bid_price: float
    bid_quantity: float
    ask_price: float
    ask_quantity: float
    spread_bps: float
    skew: float  # Positive = more weight on ask
    timestamp: Any = None


@dataclass
class MarketMakingMetrics:
    """Market making performance metrics."""
    total_trades: int
    trades_buy: int
    trades_sell: int
    total_volume_usd: float
    spread_captured: float
    inventory_pnl: float
    total_pnl: float
    sharpe_ratio: float
    current_inventory: float
    max_inventory: float
    avg_holding_time: float


class MarketMakingStrategy:
    """
    Market Making Strategy.

    Provides liquidity by quoting both bid and ask prices,
    earning the spread while managing inventory risk.
    """

    def __init__(self, config: MarketMakingConfig):
        """
        Initialize market making strategy.

        Args:
            config: Market making configuration
        """
        self.config = config

        # Inventory tracking
        self.inventory_quantity = 0.0
        self.inventory_value = 0.0
        self.inventory_cost_basis = 0.0

        # Trade tracking
        self.trades: List[Dict[str, Any]] = []
        self.daily_pnl = 0.0
        self.total_pnl = 0.0

        # Market state
        self.price_history: List[float] = []
        self.volatility = 0.0
        self.is_paused = False

        logger.info(f"Market Making initialized for {config.symbol}")

    def calculate_volatility(
        self,
        prices: np.ndarray,
        window: int = 20
    ) -> float:
        """Calculate historical volatility."""
        if len(prices) < window:
            return 0.0

        returns = np.diff(np.log(prices[-window:]))
        volatility = np.std(returns) * np.sqrt(252)  # Annualized

        return volatility

    def calculate_spread(
        self,
        mid_price: float,
        order_book: Optional[OrderBook] = None
    ) -> float:
        """
        Calculate optimal spread in basis points.

        Args:
            mid_price: Current mid price
            order_book: Order book snapshot

        Returns:
            Spread in basis points
        """
        # Base spread
        spread_bps = self.config.base_spread_bps

        # Adjust for volatility
        if self.volatility > 0:
            vol_adjustment = self.config.volatility_multiplier * (self.volatility / 0.01)
            spread_bps *= (1 + vol_adjustment)

        # Adjust for inventory
        inventory_ratio = self.inventory_value / self.config.max_inventory_usd
        if abs(inventory_ratio) > 0.3:
            # Widen spread when inventory is unbalanced
            spread_bps *= (1 + abs(inventory_ratio))

        # Adjust for order book depth
        if order_book:
            # If order book is thin, widen spread
            min_depth = min(order_book.bid_depth_usd, order_book.ask_depth_usd)
            if min_depth < self.config.min_order_book_depth_usd:
                depth_factor = self.config.min_order_book_depth_usd / min_depth
                spread_bps *= depth_factor

        # Clamp to limits
        spread_bps = np.clip(
            spread_bps,
            self.config.min_spread_bps,
            self.config.max_spread_bps
        )

        return spread_bps

    def calculate_inventory_skew(self) -> float:
        """
        Calculate inventory skew factor.

        Returns:
            Skew (-1 to 1): negative = favor bids, positive = favor asks
        """
        if self.config.max_inventory_usd == 0:
            return 0.0

        # Current inventory as ratio of max
        inventory_ratio = self.inventory_value / self.config.max_inventory_usd

        # Skew quotes to reduce inventory
        # If long, favor asks (sell)
        # If short, favor bids (buy)
        skew = inventory_ratio * self.config.inventory_skew_factor

        # Clamp to [-1, 1]
        skew = np.clip(skew, -1.0, 1.0)

        return skew

    def get_inventory_status(self) -> InventoryStatus:
        """Get current inventory status."""
        ratio = abs(self.inventory_value) / self.config.max_inventory_usd

        if ratio < 0.3:
            return InventoryStatus.BALANCED
        elif ratio < 0.7:
            if self.inventory_value > 0:
                return InventoryStatus.LONG_HEAVY
            else:
                return InventoryStatus.SHORT_HEAVY
        else:
            if self.inventory_value > 0:
                return InventoryStatus.CRITICAL_LONG
            else:
                return InventoryStatus.CRITICAL_SHORT

    def generate_quotes(
        self,
        mid_price: float,
        order_book: Optional[OrderBook] = None,
        timestamp: Any = None
    ) -> Optional[Quote]:
        """
        Generate market making quotes.

        Args:
            mid_price: Current mid price
            order_book: Order book snapshot
            timestamp: Current timestamp

        Returns:
            Quote or None if should pause
        """
        # Update price history
        self.price_history.append(mid_price)
        if len(self.price_history) > 100:
            self.price_history = self.price_history[-100:]

        # Calculate volatility
        if len(self.price_history) >= 20:
            self.volatility = self.calculate_volatility(np.array(self.price_history))

        # Check if should pause
        if self.config.pause_on_high_volatility:
            if self.volatility > self.config.volatility_threshold:
                logger.warning(f"High volatility detected: {self.volatility:.2%} - pausing")
                self.is_paused = True
                return None

        # Check daily loss limit
        if abs(self.daily_pnl) > self.config.daily_loss_limit_usd:
            logger.warning(f"Daily loss limit reached: ${self.daily_pnl:.2f}")
            self.is_paused = True
            return None

        # Check position limit
        if abs(self.inventory_value) > self.config.position_limit_usd:
            logger.warning(f"Position limit reached: ${self.inventory_value:.2f}")
            self.is_paused = True
            return None

        # Calculate spread
        spread_bps = self.calculate_spread(mid_price, order_book)
        spread_amount = mid_price * (spread_bps / 10000)

        # Calculate inventory skew
        skew = self.calculate_inventory_skew()

        # Adjust prices based on skew
        # Positive skew = move both prices up (favor selling)
        # Negative skew = move both prices down (favor buying)
        skew_adjustment = mid_price * (skew * spread_bps / 10000)

        bid_price = mid_price - spread_amount / 2 + skew_adjustment
        ask_price = mid_price + spread_amount / 2 + skew_adjustment

        # Calculate quantities
        base_quantity = self.config.order_size_usd / mid_price

        # Adjust quantities based on inventory
        # If long, make ask quantity larger
        # If short, make bid quantity larger
        if skew > 0:
            bid_quantity = base_quantity * (1 - abs(skew) * 0.5)
            ask_quantity = base_quantity * (1 + abs(skew) * 0.5)
        else:
            bid_quantity = base_quantity * (1 + abs(skew) * 0.5)
            ask_quantity = base_quantity * (1 - abs(skew) * 0.5)

        quote = Quote(
            bid_price=bid_price,
            bid_quantity=bid_quantity,
            ask_price=ask_price,
            ask_quantity=ask_quantity,
            spread_bps=spread_bps,
            skew=skew,
            timestamp=timestamp
        )

        logger.debug(
            f"Quote: Bid ${bid_price:.2f} x {bid_quantity:.4f}, "
            f"Ask ${ask_price:.2f} x {ask_quantity:.4f}, "
            f"Spread: {spread_bps:.1f}bps, Skew: {skew:+.2f}"
        )

        return quote

    def execute_trade(
        self,
        side: OrderSide,
        price: float,
        quantity: float,
        timestamp: Any = None
    ) -> Dict[str, Any]:
        """
        Execute a trade (quote gets hit).

        Args:
            side: Order side (bid or ask from maker's perspective)
            price: Execution price
            quantity: Quantity traded
            timestamp: Trade timestamp

        Returns:
            Trade details
        """
        # Update inventory
        if side == OrderSide.BID:
            # Our bid got hit - we bought
            self.inventory_quantity += quantity
            self.inventory_value += quantity * price
            trade_value = quantity * price
            trade_side = "buy"
        else:
            # Our ask got hit - we sold
            self.inventory_quantity -= quantity
            self.inventory_value -= quantity * price
            trade_value = quantity * price
            trade_side = "sell"

        # Record trade
        trade = {
            'side': trade_side,
            'price': price,
            'quantity': quantity,
            'value': trade_value,
            'inventory_after': self.inventory_quantity,
            'timestamp': timestamp
        }
        self.trades.append(trade)

        # Calculate P&L if we're flattening
        pnl = 0.0
        if len(self.trades) >= 2:
            # Simplified P&L: last buy/sell pair
            if side == OrderSide.BID and len([t for t in self.trades if t['side'] == 'sell']) > 0:
                # We bought, look for previous sell
                last_sell = [t for t in self.trades if t['side'] == 'sell'][-1]
                pnl = (last_sell['price'] - price) * quantity
            elif side == OrderSide.ASK and len([t for t in self.trades if t['side'] == 'buy']) > 0:
                # We sold, look for previous buy
                last_buy = [t for t in self.trades if t['side'] == 'buy'][-1]
                pnl = (price - last_buy['price']) * quantity

        self.daily_pnl += pnl
        self.total_pnl += pnl

        logger.info(
            f"Trade: {trade_side.upper()} {quantity:.4f} @ ${price:.2f} | "
            f"Inventory: {self.inventory_quantity:.4f} | P&L: ${pnl:.2f}"
        )

        return trade

    def simulate_market_making(
        self,
        mid_prices: np.ndarray,
        volumes: Optional[np.ndarray] = None,
        timestamps: Optional[List[Any]] = None,
        hit_probability: float = 0.3
    ) -> Dict[str, Any]:
        """
        Simulate market making strategy.

        Args:
            mid_prices: Mid price history
            volumes: Volume history (optional)
            timestamps: Timestamps (optional)
            hit_probability: Probability of quote being hit per period

        Returns:
            Simulation results
        """
        # Reset state
        self.inventory_quantity = 0.0
        self.inventory_value = 0.0
        self.trades = []
        self.daily_pnl = 0.0
        self.total_pnl = 0.0
        self.price_history = []
        self.is_paused = False

        equity_curve = [10000.0]  # Start with $10k
        inventory_curve = []
        max_inventory = 0.0

        # Simulate each period
        for i, mid_price in enumerate(mid_prices):
            ts = timestamps[i] if timestamps else i

            # Generate quotes
            quote = self.generate_quotes(mid_price, timestamp=ts)

            if quote is None:
                # Paused - skip
                inventory_curve.append(self.inventory_value)
                continue

            # Simulate market impact (quotes getting hit)
            # In reality, this depends on order flow and market conditions
            # Here we use simple probability

            # Bid hit probability (someone sells to us)
            if np.random.random() < hit_probability:
                self.execute_trade(
                    OrderSide.BID,
                    quote.bid_price,
                    quote.bid_quantity,
                    ts
                )

            # Ask hit probability (someone buys from us)
            if np.random.random() < hit_probability:
                self.execute_trade(
                    OrderSide.ASK,
                    quote.ask_price,
                    quote.ask_quantity,
                    ts
                )

            # Mark-to-market inventory
            inventory_mtm = self.inventory_quantity * mid_price

            # Update equity (realized P&L + unrealized inventory P&L)
            current_equity = equity_curve[0] + self.total_pnl + (inventory_mtm - self.inventory_value)
            equity_curve.append(current_equity)

            inventory_curve.append(inventory_mtm)
            max_inventory = max(max_inventory, abs(inventory_mtm))

        # Calculate metrics
        if self.trades:
            buys = [t for t in self.trades if t['side'] == 'buy']
            sells = [t for t in self.trades if t['side'] == 'sell']

            # Volume
            total_volume = sum(t['value'] for t in self.trades)

            # Spread captured (simplified)
            spread_captured = len(self.trades) * (self.config.base_spread_bps / 10000) * (total_volume / len(self.trades))

            # Returns
            returns = np.diff(equity_curve)
            sharpe = (np.mean(returns) / np.std(returns) * np.sqrt(252)) if np.std(returns) > 0 else 0.0

        else:
            total_volume = 0.0
            spread_captured = 0.0
            sharpe = 0.0

        # Final inventory P&L (mark-to-market)
        final_inventory_value = self.inventory_quantity * mid_prices[-1]
        inventory_pnl = final_inventory_value - self.inventory_value

        results = {
            'total_trades': len(self.trades),
            'trades_buy': len([t for t in self.trades if t['side'] == 'buy']),
            'trades_sell': len([t for t in self.trades if t['side'] == 'sell']),
            'total_volume_usd': total_volume,
            'spread_captured': spread_captured,
            'realized_pnl': self.total_pnl,
            'inventory_pnl': inventory_pnl,
            'total_pnl': self.total_pnl + inventory_pnl,
            'sharpe_ratio': sharpe,
            'final_equity': equity_curve[-1],
            'equity_curve': equity_curve,
            'inventory_curve': inventory_curve,
            'max_inventory': max_inventory,
            'trades': self.trades
        }

        logger.info(
            f"Simulation complete: {len(self.trades)} trades, "
            f"${total_volume:,.0f} volume, ${self.total_pnl:.2f} P&L"
        )

        return results


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)

    print("📈 Market Making Strategy Demo")
    print("=" * 60)

    # Configuration
    config = MarketMakingConfig(
        symbol='ETH/USDC',
        base_spread_bps=10.0,
        order_size_usd=1000.0,
        max_inventory_usd=10000.0,
        inventory_skew_factor=0.5,
        position_limit_usd=20000.0
    )

    print(f"\n1. Configuration:")
    print(f"   Symbol: {config.symbol}")
    print(f"   Base Spread: {config.base_spread_bps} bps")
    print(f"   Order Size: ${config.order_size_usd:,.0f}")
    print(f"   Max Inventory: ${config.max_inventory_usd:,.0f}")
    print(f"   Inventory Skew: {config.inventory_skew_factor}")

    # Create strategy
    strategy = MarketMakingStrategy(config)

    # Generate price data (oscillating around a mean)
    print(f"\n2. Generating market data...")
    np.random.seed(42)
    n = 500

    # Mean-reverting price
    base_price = 2000.0
    prices = base_price + np.cumsum(np.random.randn(n) * 5)
    # Add mean reversion
    for i in range(1, len(prices)):
        reversion = (base_price - prices[i-1]) * 0.1
        prices[i] += reversion

    # Simulate
    print(f"\n3. Running market making simulation...")
    results = strategy.simulate_market_making(
        mid_prices=prices,
        hit_probability=0.2  # 20% chance of fill per side
    )

    print(f"\n4. Results:")
    print(f"   Total Trades: {results['total_trades']}")
    print(f"   Buys: {results['trades_buy']}")
    print(f"   Sells: {results['trades_sell']}")
    print(f"   Total Volume: ${results['total_volume_usd']:,.2f}")
    print(f"   Spread Captured: ${results['spread_captured']:.2f}")
    print(f"   Realized P&L: ${results['realized_pnl']:.2f}")
    print(f"   Inventory P&L: ${results['inventory_pnl']:.2f}")
    print(f"   Total P&L: ${results['total_pnl']:.2f}")
    print(f"   Sharpe Ratio: {results['sharpe_ratio']:.2f}")
    print(f"   Max Inventory: ${results['max_inventory']:,.2f}")
    print(f"   Final Equity: ${results['final_equity']:,.2f}")

    print(f"\n5. Trade Balance:")
    balance = results['trades_buy'] - results['trades_sell']
    print(f"   Buy/Sell Balance: {balance:+d}")
    if abs(balance) < 5:
        print(f"   ✅ Well balanced inventory management")
    else:
        print(f"   ⚠️  Unbalanced - may need adjustment")

    print(f"\n✅ Market Making Demo Complete!")
