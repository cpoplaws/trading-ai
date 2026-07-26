"""
Grid Trading Bot
Automated grid trading strategy for range-bound markets in crypto.

Strategy:
- Place buy orders at intervals below current price
- Place sell orders at intervals above current price
- Profit from price oscillations within a range
- Ideal for sideways/ranging markets
"""
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from decimal import Decimal
import time
import logging

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

logger = logging.getLogger(__name__)


@dataclass
class GridLevel:
    """Represents a single grid level."""
    price: float
    order_type: str  # 'buy' or 'sell'
    quantity: float
    order_id: Optional[str] = None
    filled: bool = False
    filled_price: Optional[float] = None


@dataclass
class GridConfig:
    """Configuration for grid trading."""
    symbol: str
    lower_price: float  # Bottom of the grid
    upper_price: float  # Top of the grid
    num_grids: int  # Number of grid levels
    total_investment: float  # Total capital to deploy
    grid_type: str = 'arithmetic'  # 'arithmetic' or 'geometric'
    auto_rebalance: bool = True  # Rebalance when out of range
    stop_loss: Optional[float] = None  # Stop loss price
    take_profit: Optional[float] = None  # Take profit price


class GridTradingBot:
    """
    Grid Trading Bot for cryptocurrency markets.

    Features:
    - Arithmetic or geometric grid spacing
    - Automatic order placement and management
    - Profit tracking per grid
    - Auto-rebalancing when price breaks range
    - Stop-loss and take-profit support
    """

    def __init__(self, config: GridConfig, exchange_client=None):
        """
        Initialize grid trading bot.

        Args:
            config: Grid configuration
            exchange_client: Exchange API client (optional for backtesting)
        """
        self.config = config
        self.exchange = exchange_client

        # State
        self.grid_levels: List[GridLevel] = []
        self.active_orders: Dict[str, GridLevel] = {}
        self.filled_orders: List[GridLevel] = []
        self.total_profit: float = 0.0
        self.trades_count: int = 0

        # Initialize grid
        self._create_grid()

        logger.info(f"Grid trading bot initialized: {config.symbol}")
        logger.info(f"Range: ${config.lower_price:.2f} - ${config.upper_price:.2f}")
        logger.info(f"Grids: {config.num_grids}, Investment: ${config.total_investment:,.2f}")

    def _create_grid(self):
        """Create grid levels based on configuration."""
        self.grid_levels = []

        price_range = self.config.upper_price - self.config.lower_price
        mid_price = (self.config.upper_price + self.config.lower_price) / 2

        if self.config.grid_type == 'arithmetic':
            # Equal price intervals
            price_step = price_range / self.config.num_grids

            for i in range(self.config.num_grids + 1):
                price = self.config.lower_price + (i * price_step)

                if price < mid_price:
                    order_type = 'buy'
                elif price > mid_price:
                    order_type = 'sell'
                else:
                    continue  # Skip mid price

                # Calculate quantity per level
                quantity = self._calculate_quantity(price, order_type)

                level = GridLevel(
                    price=price,
                    order_type=order_type,
                    quantity=quantity
                )
                self.grid_levels.append(level)

        elif self.config.grid_type == 'geometric':
            # Percentage-based intervals (better for volatile assets)
            ratio = (self.config.upper_price / self.config.lower_price) ** (1 / self.config.num_grids)

            for i in range(self.config.num_grids + 1):
                price = self.config.lower_price * (ratio ** i)

                if price < mid_price:
                    order_type = 'buy'
                elif price > mid_price:
                    order_type = 'sell'
                else:
                    continue

                quantity = self._calculate_quantity(price, order_type)

                level = GridLevel(
                    price=price,
                    order_type=order_type,
                    quantity=quantity
                )
                self.grid_levels.append(level)

        logger.info(f"Created {len(self.grid_levels)} grid levels")

    def _calculate_quantity(self, price: float, order_type: str) -> float:
        """
        Calculate quantity for each grid level.

        Args:
            price: Grid level price
            order_type: 'buy' or 'sell'

        Returns:
            Quantity to trade at this level
        """
        # Divide capital equally among grid levels
        num_levels = self.config.num_grids // 2  # Half for buys, half for sells

        if order_type == 'buy':
            capital_per_level = self.config.total_investment / num_levels
            quantity = capital_per_level / price
        else:  # sell
            # For sells, we need to have the asset
            # Assume we start with 50% in the asset
            total_asset = (self.config.total_investment * 0.5) / ((self.config.upper_price + self.config.lower_price) / 2)
            quantity = total_asset / num_levels

        return quantity

    def place_all_orders(self) -> List[str]:
        """
        Place all grid orders on the exchange.

        Returns:
            List of order IDs
        """
        order_ids = []

        for level in self.grid_levels:
            if level.filled:
                continue

            # Place order
            order_id = self._place_order(level)

            if order_id:
                level.order_id = order_id
                self.active_orders[order_id] = level
                order_ids.append(order_id)
                logger.info(f"Placed {level.order_type} order at ${level.price:.2f}")

        logger.info(f"Placed {len(order_ids)} grid orders")
        return order_ids

    def _place_order(self, level: GridLevel) -> Optional[str]:
        """
        Place a single limit order.

        Args:
            level: Grid level to place order for

        Returns:
            Order ID if successful
        """
        if not self.exchange:
            # Simulation mode
            return f"sim_{level.order_type}_{level.price}"

        try:
            # Place limit order on exchange
            order = self.exchange.place_limit_order(
                symbol=self.config.symbol,
                side=level.order_type,
                quantity=level.quantity,
                price=level.price
            )

            return order.get('order_id') if order else None

        except Exception as e:
            logger.error(f"Error placing order at ${level.price:.2f}: {e}")
            return None

    def update_orders(self, current_price: float) -> Dict:
        """
        Update grid based on filled orders and current price.

        Args:
            current_price: Current market price

        Returns:
            Update statistics
        """
        filled_count = 0
        new_orders_placed = 0

        # Check for filled orders
        for order_id, level in list(self.active_orders.items()):
            if self._is_order_filled(order_id, current_price):
                filled_count += 1
                level.filled = True
                level.filled_price = current_price

                # Move to filled orders
                self.filled_orders.append(level)
                del self.active_orders[order_id]

                # Calculate profit if opposite order was also filled
                profit = self._calculate_grid_profit(level)
                if profit > 0:
                    self.total_profit += profit
                    self.trades_count += 1
                    logger.info(f"Grid profit: ${profit:.2f} (Total: ${self.total_profit:.2f})")

                # Place opposite order
                opposite_level = self._create_opposite_order(level, current_price)
                if opposite_level:
                    order_id = self._place_order(opposite_level)
                    if order_id:
                        opposite_level.order_id = order_id
                        self.active_orders[order_id] = opposite_level
                        new_orders_placed += 1
                        self.grid_levels.append(opposite_level)

        # Check if price is out of range
        if self.config.auto_rebalance:
            if current_price < self.config.lower_price or current_price > self.config.upper_price:
                logger.warning(f"Price ${current_price:.2f} out of range, rebalancing...")
                self._rebalance_grid(current_price)

        return {
            'filled_orders': filled_count,
            'new_orders': new_orders_placed,
            'active_orders': len(self.active_orders),
            'total_profit': self.total_profit,
            'trades_count': self.trades_count
        }

    def _is_order_filled(self, order_id: str, current_price: float) -> bool:
        """
        Check if an order has been filled.

        Args:
            order_id: Order ID to check
            current_price: Current market price

        Returns:
            True if filled
        """
        if not self.exchange:
            # Simulation: check if price crossed the level
            level = self.active_orders.get(order_id)
            if not level:
                return False

            if level.order_type == 'buy':
                return current_price <= level.price
            else:  # sell
                return current_price >= level.price

        # Real exchange: query order status
        try:
            order = self.exchange.get_order(order_id)
            return order.get('status') == 'filled'
        except Exception as e:
            logger.error(f"Error checking order {order_id}: {e}")
            return False

    def _calculate_grid_profit(self, level: GridLevel) -> float:
        """
        Calculate profit from a filled grid level.

        Args:
            level: Filled grid level

        Returns:
            Profit in USD
        """
        # Find matching opposite order
        if level.order_type == 'buy':
            # Look for sell order above this price
            for filled in self.filled_orders:
                if filled.order_type == 'sell' and filled.price > level.price:
                    # Profit = (sell_price - buy_price) * quantity
                    profit = (filled.price - level.price) * level.quantity
                    return profit

        elif level.order_type == 'sell':
            # Look for buy order below this price
            for filled in self.filled_orders:
                if filled.order_type == 'buy' and filled.price < level.price:
                    # Profit = (sell_price - buy_price) * quantity
                    profit = (level.price - filled.price) * level.quantity
                    return profit

        return 0.0

    def _create_opposite_order(self, filled_level: GridLevel, current_price: float) -> Optional[GridLevel]:
        """
        Create opposite order after a fill.

        Args:
            filled_level: The level that was just filled
            current_price: Current market price

        Returns:
            New grid level for opposite order
        """
        if filled_level.order_type == 'buy':
            # Create sell order above
            sell_price = filled_level.price * 1.01  # 1% profit target
            if sell_price <= self.config.upper_price:
                return GridLevel(
                    price=sell_price,
                    order_type='sell',
                    quantity=filled_level.quantity
                )

        elif filled_level.order_type == 'sell':
            # Create buy order below
            buy_price = filled_level.price * 0.99  # 1% below
            if buy_price >= self.config.lower_price:
                return GridLevel(
                    price=buy_price,
                    order_type='buy',
                    quantity=filled_level.quantity
                )

        return None

    def _rebalance_grid(self, current_price: float):
        """
        Rebalance grid when price moves out of range.

        Args:
            current_price: Current market price
        """
        # Cancel all active orders
        for order_id in list(self.active_orders.keys()):
            self._cancel_order(order_id)

        # Calculate new range centered on current price
        price_range = self.config.upper_price - self.config.lower_price
        self.config.lower_price = current_price - (price_range / 2)
        self.config.upper_price = current_price + (price_range / 2)

        # Recreate grid
        self._create_grid()

        # Place new orders
        self.place_all_orders()

        logger.info(f"Grid rebalanced to ${self.config.lower_price:.2f} - ${self.config.upper_price:.2f}")

    def _cancel_order(self, order_id: str) -> bool:
        """Cancel an active order."""
        if not self.exchange:
            # Simulation
            if order_id in self.active_orders:
                del self.active_orders[order_id]
            return True

        try:
            success = self.exchange.cancel_order(order_id)
            if success:
                del self.active_orders[order_id]
            return success
        except Exception as e:
            logger.error(f"Error canceling order {order_id}: {e}")
            return False

    def get_performance_stats(self) -> Dict:
        """Get current performance statistics."""
        return {
            'total_profit': self.total_profit,
            'trades_count': self.trades_count,
            'avg_profit_per_trade': self.total_profit / self.trades_count if self.trades_count > 0 else 0,
            'active_orders': len(self.active_orders),
            'filled_orders': len(self.filled_orders),
            'profit_percentage': (self.total_profit / self.config.total_investment) * 100,
            'grid_levels': len(self.grid_levels)
        }

    def stop(self):
        """Stop the bot and cancel all orders."""
        logger.info("Stopping grid trading bot...")

        for order_id in list(self.active_orders.keys()):
            self._cancel_order(order_id)

        logger.info(f"Bot stopped. Final profit: ${self.total_profit:.2f}")


# Backtesting simulation
class GridBacktester:
    """Backtest grid trading strategy on historical data."""

    def __init__(self, price_data: List[float], config: GridConfig):
        """
        Initialize backtester.

        Args:
            price_data: List of historical prices
            config: Grid configuration
        """
        self.price_data = price_data
        self.config = config
        self.bot = GridTradingBot(config)

    def run(self) -> Dict:
        """
        Run backtest.

        Returns:
            Backtest results
        """
        logger.info(f"Starting backtest with {len(self.price_data)} price points...")

        # Place initial orders
        self.bot.place_all_orders()

        # Simulate price movements
        for i, price in enumerate(self.price_data):
            stats = self.bot.update_orders(price)

            if i % 100 == 0:
                logger.info(f"Step {i}: Price=${price:.2f}, Profit=${stats['total_profit']:.2f}")

        # Final stats
        final_stats = self.bot.get_performance_stats()
        final_stats['price_range_tested'] = (min(self.price_data), max(self.price_data))

        logger.info(f"Backtest complete!")
        logger.info(f"Total profit: ${final_stats['total_profit']:.2f}")
        logger.info(f"ROI: {final_stats['profit_percentage']:.2f}%")
        logger.info(f"Trades: {final_stats['trades_count']}")

        return final_stats


if __name__ == '__main__':
    import numpy as np
    import logging
    logging.basicConfig(level=logging.INFO)

    print("📊 Grid Trading Bot Demo")
    print("=" * 60)

    # Create sample price data (oscillating market)
    base_price = 45000
    volatility = 2000
    num_points = 1000

    # Generate oscillating price data
    price_data = []
    for i in range(num_points):
        noise = np.random.normal(0, volatility * 0.3)
        oscillation = volatility * np.sin(i / 50)  # Oscillate
        price = base_price + oscillation + noise
        price_data.append(price)

    print(f"\n📈 Simulating {num_points} price points")
    print(f"   Base: ${base_price:,.0f}")
    print(f"   Range: ${min(price_data):,.0f} - ${max(price_data):,.0f}")

    # Configure grid
    config = GridConfig(
        symbol='BTC/USDT',
        lower_price=43000,
        upper_price=47000,
        num_grids=10,
        total_investment=10000,
        grid_type='arithmetic',
        auto_rebalance=True
    )

    print(f"\n⚙️  Grid Configuration:")
    print(f"   Range: ${config.lower_price:,.0f} - ${config.upper_price:,.0f}")
    print(f"   Grids: {config.num_grids}")
    print(f"   Investment: ${config.total_investment:,.0f}")
    print(f"   Type: {config.grid_type}")

    # Run backtest
    print(f"\n🧪 Running backtest...")
    backtester = GridBacktester(price_data, config)
    results = backtester.run()

    print(f"\n✅ Backtest Results:")
    print(f"   Total Profit: ${results['total_profit']:,.2f}")
    print(f"   ROI: {results['profit_percentage']:.2f}%")
    print(f"   Trades: {results['trades_count']}")
    print(f"   Avg Profit/Trade: ${results['avg_profit_per_trade']:.2f}")
    print(f"   Active Orders: {results['active_orders']}")
    print(f"   Filled Orders: {results['filled_orders']}")

    print(f"\n💡 Grid trading works best in:")
    print(f"   ✅ Range-bound markets")
    print(f"   ✅ High volatility within range")
    print(f"   ✅ Predictable oscillations")
    print(f"   ❌ Strong trending markets (use auto-rebalance)")
