"""
Dollar-Cost Averaging (DCA) Bot
Automates regular purchases of crypto assets regardless of price.

Features:
- Fixed amount or percentage-based purchases
- Multiple frequency options (daily, weekly, monthly)
- Price deviation triggers (buy more when price drops)
- Portfolio rebalancing
- Risk-adjusted position sizing
"""
import logging
from typing import List, Optional, Dict, Any
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum

logger = logging.getLogger(__name__)


class DCAFrequency(Enum):
    """DCA purchase frequency."""
    DAILY = "daily"
    WEEKLY = "weekly"
    BIWEEKLY = "biweekly"
    MONTHLY = "monthly"
    CUSTOM = "custom"


class DCAMode(Enum):
    """DCA execution mode."""
    FIXED_AMOUNT = "fixed_amount"  # Fixed USD amount
    FIXED_QUANTITY = "fixed_quantity"  # Fixed token quantity
    PERCENTAGE_PORTFOLIO = "percentage_portfolio"  # % of portfolio
    DYNAMIC = "dynamic"  # Adjust based on price


@dataclass
class DCAConfig:
    """DCA bot configuration."""
    symbol: str
    frequency: DCAFrequency
    mode: DCAMode

    # Fixed amount settings
    amount_per_purchase: float = 100.0  # USD or token amount

    # Dynamic settings
    base_amount: float = 100.0
    price_deviation_threshold: float = 0.10  # 10% price drop = buy more
    max_multiplier: float = 3.0  # Max 3x normal amount

    # Timing
    start_date: datetime = None
    end_date: datetime = None
    custom_interval_hours: int = 168  # 1 week default

    # Risk management
    max_position_size: float = 10000.0  # Max total position
    stop_on_drawdown: float = 0.30  # Stop if down 30%

    # Features
    enable_dips: bool = True  # Buy extra on dips
    enable_rebalancing: bool = False


@dataclass
class DCASchedule:
    """DCA purchase schedule."""
    next_purchase_date: datetime
    amount: float
    is_dip_purchase: bool = False
    price_at_schedule: Optional[float] = None


@dataclass
class DCAMetrics:
    """DCA performance metrics."""
    total_invested: float
    total_quantity: float
    average_cost_basis: float
    current_value: float
    unrealized_pnl: float
    pnl_percent: float
    num_purchases: int
    largest_purchase: float
    smallest_purchase: float


class DCABot:
    """
    Dollar-Cost Averaging Bot.

    Automates regular crypto purchases to build positions over time.
    """

    def __init__(self, config: DCAConfig):
        """
        Initialize DCA bot.

        Args:
            config: DCA configuration
        """
        self.config = config
        self.purchase_history: List[Dict[str, Any]] = []
        self.total_quantity = 0.0
        self.total_invested = 0.0
        self.average_cost = 0.0
        self.next_scheduled_purchase = self._calculate_next_purchase()

        # Price tracking for dips
        self.price_history: List[float] = []
        self.moving_average_20 = 0.0

        # Initialize start date
        if self.config.start_date is None:
            self.config.start_date = datetime.now()

        logger.info(f"DCA Bot initialized for {config.symbol}")
        logger.info(f"Frequency: {config.frequency.value}, Mode: {config.mode.value}")

    def _calculate_next_purchase(self) -> datetime:
        """Calculate next scheduled purchase time."""
        now = datetime.now()

        if self.config.frequency == DCAFrequency.DAILY:
            return now + timedelta(days=1)
        elif self.config.frequency == DCAFrequency.WEEKLY:
            return now + timedelta(weeks=1)
        elif self.config.frequency == DCAFrequency.BIWEEKLY:
            return now + timedelta(weeks=2)
        elif self.config.frequency == DCAFrequency.MONTHLY:
            return now + timedelta(days=30)
        elif self.config.frequency == DCAFrequency.CUSTOM:
            return now + timedelta(hours=self.config.custom_interval_hours)

        return now + timedelta(weeks=1)

    def should_execute_purchase(
        self,
        current_time: datetime,
        current_price: float
    ) -> bool:
        """
        Check if should execute purchase.

        Args:
            current_time: Current timestamp
            current_price: Current asset price

        Returns:
            True if should buy now
        """
        # Check if reached next scheduled time
        if current_time >= self.next_scheduled_purchase:
            return True

        # Check for dip opportunity
        if self.config.enable_dips and self._is_significant_dip(current_price):
            logger.info(f"Dip detected at ${current_price:.2f} - Extra purchase triggered")
            return True

        # Check max position size
        if self.total_invested >= self.config.max_position_size:
            logger.warning(f"Max position size reached: ${self.total_invested:.2f}")
            return False

        return False

    def _is_significant_dip(self, current_price: float) -> bool:
        """Check if price is significantly below recent average."""
        if len(self.price_history) < 20:
            return False

        # Update moving average
        self.moving_average_20 = sum(self.price_history[-20:]) / 20

        # Check deviation
        deviation = (current_price - self.moving_average_20) / self.moving_average_20

        return deviation < -self.config.price_deviation_threshold

    def calculate_purchase_amount(
        self,
        current_price: float,
        portfolio_value: Optional[float] = None,
        is_dip: bool = False
    ) -> float:
        """
        Calculate USD amount to purchase.

        Args:
            current_price: Current asset price
            portfolio_value: Total portfolio value (for percentage mode)
            is_dip: Whether this is a dip purchase

        Returns:
            USD amount to purchase
        """
        if self.config.mode == DCAMode.FIXED_AMOUNT:
            amount = self.config.amount_per_purchase

        elif self.config.mode == DCAMode.FIXED_QUANTITY:
            amount = self.config.amount_per_purchase * current_price

        elif self.config.mode == DCAMode.PERCENTAGE_PORTFOLIO:
            if portfolio_value is None:
                raise ValueError("Portfolio value required for percentage mode")
            percentage = self.config.amount_per_purchase / 100.0
            amount = portfolio_value * percentage

        elif self.config.mode == DCAMode.DYNAMIC:
            # Base amount
            amount = self.config.base_amount

            # Increase on dips
            if is_dip and len(self.price_history) >= 20:
                deviation = abs((current_price - self.moving_average_20) / self.moving_average_20)
                multiplier = min(1.0 + deviation * 10, self.config.max_multiplier)
                amount *= multiplier
                logger.info(f"Dynamic purchase: {multiplier:.2f}x normal amount")

        else:
            amount = self.config.amount_per_purchase

        # Respect max position size
        remaining_capacity = self.config.max_position_size - self.total_invested
        amount = min(amount, remaining_capacity)

        return amount

    def execute_purchase(
        self,
        current_price: float,
        current_time: datetime,
        portfolio_value: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Execute a DCA purchase.

        Args:
            current_price: Current asset price
            current_time: Current timestamp
            portfolio_value: Total portfolio value

        Returns:
            Purchase details
        """
        # Check if should buy
        if not self.should_execute_purchase(current_time, current_price):
            return {
                'executed': False,
                'reason': 'Not scheduled',
                'next_purchase': self.next_scheduled_purchase
            }

        # Check stop conditions
        if self._check_stop_conditions():
            return {
                'executed': False,
                'reason': 'Stop condition triggered',
                'metrics': self.get_metrics(current_price)
            }

        # Determine if dip purchase
        is_dip = self._is_significant_dip(current_price)

        # Calculate amount
        usd_amount = self.calculate_purchase_amount(
            current_price,
            portfolio_value,
            is_dip
        )

        if usd_amount <= 0:
            return {
                'executed': False,
                'reason': 'Zero amount calculated'
            }

        # Calculate quantity
        quantity = usd_amount / current_price

        # Execute purchase
        self.total_quantity += quantity
        self.total_invested += usd_amount
        self.average_cost = self.total_invested / self.total_quantity

        # Record purchase
        purchase = {
            'timestamp': current_time,
            'price': current_price,
            'quantity': quantity,
            'usd_amount': usd_amount,
            'is_dip_purchase': is_dip,
            'total_quantity': self.total_quantity,
            'total_invested': self.total_invested,
            'average_cost': self.average_cost
        }
        self.purchase_history.append(purchase)

        # Update price history
        self.price_history.append(current_price)
        if len(self.price_history) > 100:
            self.price_history = self.price_history[-100:]

        # Schedule next purchase (only if not a dip purchase)
        if not is_dip:
            self.next_scheduled_purchase = self._calculate_next_purchase()

        logger.info(
            f"DCA Purchase: {quantity:.6f} {self.config.symbol} @ ${current_price:.2f} "
            f"(${usd_amount:.2f}) {'[DIP]' if is_dip else ''}"
        )

        return {
            'executed': True,
            'purchase': purchase,
            'next_scheduled': self.next_scheduled_purchase,
            'metrics': self.get_metrics(current_price)
        }

    def _check_stop_conditions(self) -> bool:
        """Check if should stop DCA purchases."""
        # Check max position
        if self.total_invested >= self.config.max_position_size:
            logger.warning("Max position size reached - stopping DCA")
            return True

        # Check end date
        if self.config.end_date and datetime.now() >= self.config.end_date:
            logger.info("End date reached - stopping DCA")
            return True

        return False

    def get_metrics(self, current_price: float) -> DCAMetrics:
        """
        Get DCA performance metrics.

        Args:
            current_price: Current asset price

        Returns:
            Performance metrics
        """
        if self.total_quantity == 0:
            return DCAMetrics(
                total_invested=0,
                total_quantity=0,
                average_cost_basis=0,
                current_value=0,
                unrealized_pnl=0,
                pnl_percent=0,
                num_purchases=0,
                largest_purchase=0,
                smallest_purchase=0
            )

        current_value = self.total_quantity * current_price
        unrealized_pnl = current_value - self.total_invested
        pnl_percent = (unrealized_pnl / self.total_invested) * 100

        # Find largest/smallest purchases
        amounts = [p['usd_amount'] for p in self.purchase_history]
        largest = max(amounts) if amounts else 0
        smallest = min(amounts) if amounts else 0

        return DCAMetrics(
            total_invested=self.total_invested,
            total_quantity=self.total_quantity,
            average_cost_basis=self.average_cost,
            current_value=current_value,
            unrealized_pnl=unrealized_pnl,
            pnl_percent=pnl_percent,
            num_purchases=len(self.purchase_history),
            largest_purchase=largest,
            smallest_purchase=smallest
        )

    def simulate_dca(
        self,
        price_history: List[float],
        timestamps: List[datetime],
        initial_portfolio_value: float = 10000.0
    ) -> Dict[str, Any]:
        """
        Simulate DCA strategy on historical data.

        Args:
            price_history: Historical prices
            timestamps: Corresponding timestamps
            initial_portfolio_value: Starting portfolio value

        Returns:
            Simulation results
        """
        if len(price_history) != len(timestamps):
            raise ValueError("Price history and timestamps must have same length")

        # Reset state
        self.purchase_history = []
        self.total_quantity = 0.0
        self.total_invested = 0.0
        self.average_cost = 0.0
        self.price_history = []
        self.next_scheduled_purchase = timestamps[0]

        # Simulate each period
        for i, (price, timestamp) in enumerate(zip(price_history, timestamps)):
            # Update price history
            self.price_history.append(price)

            # Try to execute purchase
            result = self.execute_purchase(
                current_price=price,
                current_time=timestamp,
                portfolio_value=initial_portfolio_value
            )

            if result['executed']:
                logger.debug(f"Purchase {len(self.purchase_history)} @ ${price:.2f}")

        # Final metrics
        final_metrics = self.get_metrics(price_history[-1])

        # Calculate strategy performance
        buy_and_hold_quantity = initial_portfolio_value / price_history[0]
        buy_and_hold_value = buy_and_hold_quantity * price_history[-1]
        buy_and_hold_pnl = ((buy_and_hold_value - initial_portfolio_value) /
                            initial_portfolio_value) * 100

        results = {
            'final_metrics': final_metrics,
            'num_purchases': len(self.purchase_history),
            'total_invested': self.total_invested,
            'final_value': final_metrics.current_value,
            'pnl_percent': final_metrics.pnl_percent,
            'average_cost': self.average_cost,
            'first_price': price_history[0],
            'last_price': price_history[-1],
            'buy_and_hold_pnl': buy_and_hold_pnl,
            'outperformance': final_metrics.pnl_percent - buy_and_hold_pnl,
            'purchase_history': self.purchase_history
        }

        logger.info(f"Simulation complete: {len(self.purchase_history)} purchases")
        logger.info(f"DCA P&L: {final_metrics.pnl_percent:.2f}%")
        logger.info(f"Buy & Hold P&L: {buy_and_hold_pnl:.2f}%")
        logger.info(f"Outperformance: {results['outperformance']:.2f}%")

        return results


if __name__ == '__main__':
    import numpy as np

    logging.basicConfig(level=logging.INFO)

    print("💰 DCA Bot Demo")
    print("=" * 60)

    # Create DCA configuration
    config = DCAConfig(
        symbol='BTC',
        frequency=DCAFrequency.WEEKLY,
        mode=DCAMode.DYNAMIC,
        base_amount=100.0,
        enable_dips=True,
        price_deviation_threshold=0.10,
        max_multiplier=2.5,
        max_position_size=5000.0
    )

    print(f"\n1. Configuration:")
    print(f"   Symbol: {config.symbol}")
    print(f"   Frequency: {config.frequency.value}")
    print(f"   Mode: {config.mode.value}")
    print(f"   Amount: ${config.base_amount:.2f}")
    print(f"   Enable Dips: {config.enable_dips}")

    # Create bot
    bot = DCABot(config)

    # Generate sample price data (volatile market)
    print(f"\n2. Simulating 1 year of weekly DCA...")
    weeks = 52
    start_price = 40000.0

    # Simulate price with volatility and dips
    np.random.seed(42)
    prices = [start_price]
    for i in range(weeks - 1):
        # Random walk with occasional dips
        change = np.random.normal(0.01, 0.05)  # 1% up, 5% volatility

        # Occasional sharp dips
        if np.random.random() < 0.1:  # 10% chance of dip
            change = -0.15  # 15% dip

        new_price = prices[-1] * (1 + change)
        prices.append(max(new_price, start_price * 0.5))  # Floor at 50% of start

    # Generate timestamps
    timestamps = [
        datetime.now() - timedelta(weeks=weeks-i)
        for i in range(weeks)
    ]

    # Run simulation
    results = bot.simulate_dca(prices, timestamps, initial_portfolio_value=10000.0)

    print(f"\n3. Results:")
    print(f"   Purchases: {results['num_purchases']}")
    print(f"   Total Invested: ${results['total_invested']:.2f}")
    print(f"   Final Value: ${results['final_value']:.2f}")
    print(f"   Average Cost: ${results['average_cost']:.2f}")
    print(f"   P&L: ${results['final_value'] - results['total_invested']:.2f} "
          f"({results['pnl_percent']:.2f}%)")

    print(f"\n4. Comparison:")
    print(f"   DCA P&L: {results['pnl_percent']:.2f}%")
    print(f"   Buy & Hold P&L: {results['buy_and_hold_pnl']:.2f}%")
    print(f"   Outperformance: {results['outperformance']:.2f}%")

    # Analyze purchases
    dip_purchases = sum(1 for p in results['purchase_history'] if p['is_dip_purchase'])
    regular_purchases = results['num_purchases'] - dip_purchases

    print(f"\n5. Purchase Breakdown:")
    print(f"   Regular: {regular_purchases}")
    print(f"   Dip Buys: {dip_purchases}")
    print(f"   Largest: ${results['final_metrics'].largest_purchase:.2f}")
    print(f"   Smallest: ${results['final_metrics'].smallest_purchase:.2f}")

    print(f"\n✅ DCA Bot Demo Complete!")
