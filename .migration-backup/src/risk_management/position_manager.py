"""
Position Manager
Manages position sizing, limits, and stop losses for risk control.
"""
import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class PositionSide(Enum):
    """Position side."""
    LONG = "long"
    SHORT = "short"
    FLAT = "flat"


class StopLossType(Enum):
    """Stop loss types."""
    FIXED = "fixed"  # Fixed price
    TRAILING = "trailing"  # Trailing stop
    ATR = "atr"  # Based on ATR (Average True Range)
    PERCENT = "percent"  # Percentage from entry


@dataclass
class Position:
    """Trading position."""
    symbol: str
    side: PositionSide
    entry_price: float
    quantity: float
    entry_time: datetime
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    current_price: float = 0.0
    unrealized_pnl: float = 0.0
    unrealized_pnl_percent: float = 0.0
    max_favorable_excursion: float = 0.0  # MFE
    max_adverse_excursion: float = 0.0  # MAE
    metadata: dict = field(default_factory=dict)

    def update_price(self, current_price: float):
        """Update position with current price."""
        self.current_price = current_price

        # Calculate unrealized P&L
        if self.side == PositionSide.LONG:
            self.unrealized_pnl = (current_price - self.entry_price) * self.quantity
            self.unrealized_pnl_percent = (current_price - self.entry_price) / self.entry_price * 100

            # Update MFE and MAE
            favorable = current_price - self.entry_price
            if favorable > self.max_favorable_excursion:
                self.max_favorable_excursion = favorable

            adverse = self.entry_price - current_price
            if adverse > self.max_adverse_excursion:
                self.max_adverse_excursion = adverse

        elif self.side == PositionSide.SHORT:
            self.unrealized_pnl = (self.entry_price - current_price) * self.quantity
            self.unrealized_pnl_percent = (self.entry_price - current_price) / self.entry_price * 100

            favorable = self.entry_price - current_price
            if favorable > self.max_favorable_excursion:
                self.max_favorable_excursion = favorable

            adverse = current_price - self.entry_price
            if adverse > self.max_adverse_excursion:
                self.max_adverse_excursion = adverse

    def should_trigger_stop_loss(self) -> bool:
        """Check if stop loss should be triggered."""
        if self.stop_loss is None:
            return False

        if self.side == PositionSide.LONG:
            return self.current_price <= self.stop_loss
        elif self.side == PositionSide.SHORT:
            return self.current_price >= self.stop_loss

        return False

    def should_trigger_take_profit(self) -> bool:
        """Check if take profit should be triggered."""
        if self.take_profit is None:
            return False

        if self.side == PositionSide.LONG:
            return self.current_price >= self.take_profit
        elif self.side == PositionSide.SHORT:
            return self.current_price <= self.take_profit

        return False

    @property
    def value(self) -> float:
        """Get position value."""
        return self.quantity * self.current_price

    @property
    def duration(self) -> timedelta:
        """Get position duration."""
        return datetime.now() - self.entry_time


@dataclass
class PositionLimits:
    """Position and risk limits."""
    max_position_size: float = 100000  # Max position value
    max_positions: int = 10  # Max concurrent positions
    max_symbol_exposure: float = 50000  # Max exposure per symbol
    max_sector_exposure: float = 75000  # Max exposure per sector
    max_portfolio_leverage: float = 2.0  # Max leverage
    max_daily_loss: float = 5000  # Max loss per day
    max_drawdown: float = 20  # Max drawdown %
    position_size_percent: float = 10  # Position size as % of portfolio


class PositionManager:
    """
    Manages trading positions with risk controls.

    Features:
    - Position tracking
    - Dynamic position sizing (Kelly Criterion, Risk Parity, etc.)
    - Stop loss management (fixed, trailing, ATR-based)
    - Take profit management
    - Position limits enforcement
    - Portfolio risk aggregation
    - MFE/MAE tracking
    """

    def __init__(
        self,
        portfolio_value: float,
        limits: Optional[PositionLimits] = None
    ):
        """
        Initialize position manager.

        Args:
            portfolio_value: Initial portfolio value
            limits: Position limits configuration
        """
        self.portfolio_value = portfolio_value
        self.initial_value = portfolio_value
        self.limits = limits or PositionLimits()

        # Active positions
        self.positions: Dict[str, Position] = {}

        # Closed positions history
        self.closed_positions: List[Position] = []

        # Daily stats
        self.daily_pnl = 0.0
        self.daily_trades = 0
        self.last_reset_date = datetime.now().date()

        # Risk metrics
        self.max_drawdown_reached = 0.0
        self.peak_portfolio_value = portfolio_value

        logger.info(f"Position manager initialized: ${portfolio_value:,.2f} portfolio")

    def calculate_position_size(
        self,
        symbol: str,
        entry_price: float,
        stop_loss: float,
        method: str = "fixed_risk",
        risk_per_trade: float = 0.02,  # 2% of portfolio
        win_rate: Optional[float] = None,
        avg_win_loss_ratio: Optional[float] = None
    ) -> float:
        """
        Calculate optimal position size.

        Args:
            symbol: Trading symbol
            entry_price: Entry price
            stop_loss: Stop loss price
            method: Sizing method ('fixed_risk', 'kelly', 'risk_parity')
            risk_per_trade: Risk per trade as fraction of portfolio
            win_rate: Historical win rate (for Kelly)
            avg_win_loss_ratio: Average win/loss ratio (for Kelly)

        Returns:
            Position size in units
        """
        if method == "fixed_risk":
            # Risk fixed $ or % per trade
            risk_amount = self.portfolio_value * risk_per_trade
            risk_per_unit = abs(entry_price - stop_loss)

            if risk_per_unit == 0:
                logger.warning("Risk per unit is zero, using default position size")
                return 0

            size = risk_amount / risk_per_unit

        elif method == "kelly":
            # Kelly Criterion: f = (bp - q) / b
            # f = fraction to bet
            # b = odds (avg_win_loss_ratio)
            # p = win probability
            # q = 1 - p

            if win_rate is None or avg_win_loss_ratio is None:
                logger.warning("Kelly requires win_rate and avg_win_loss_ratio")
                return self.calculate_position_size(symbol, entry_price, stop_loss, "fixed_risk")

            b = avg_win_loss_ratio
            p = win_rate
            q = 1 - p

            kelly_fraction = (b * p - q) / b

            # Half-Kelly for safety
            kelly_fraction = max(0, kelly_fraction * 0.5)

            # Apply to portfolio
            position_value = self.portfolio_value * kelly_fraction
            size = position_value / entry_price

        elif method == "risk_parity":
            # Equal risk contribution from each position
            # Allocate based on inverse volatility
            # (This is simplified; full implementation needs historical volatility)
            target_position_value = self.portfolio_value * self.limits.position_size_percent / 100
            size = target_position_value / entry_price

        else:
            raise ValueError(f"Unknown position sizing method: {method}")

        # Apply limits
        max_size = self.limits.max_position_size / entry_price
        size = min(size, max_size)

        return size

    def open_position(
        self,
        symbol: str,
        side: PositionSide,
        entry_price: float,
        quantity: float,
        stop_loss: Optional[float] = None,
        take_profit: Optional[float] = None,
        metadata: Optional[dict] = None
    ) -> Optional[Position]:
        """
        Open a new position with risk checks.

        Args:
            symbol: Trading symbol
            side: Position side (LONG/SHORT)
            entry_price: Entry price
            quantity: Position quantity
            stop_loss: Stop loss price
            take_profit: Take profit price
            metadata: Additional metadata

        Returns:
            Position if opened, None if rejected by risk limits
        """
        # Check if we can open position
        if not self._check_can_open_position(symbol, entry_price, quantity):
            return None

        # Create position
        position = Position(
            symbol=symbol,
            side=side,
            entry_price=entry_price,
            quantity=quantity,
            entry_time=datetime.now(),
            stop_loss=stop_loss,
            take_profit=take_profit,
            current_price=entry_price,
            metadata=metadata or {}
        )

        self.positions[symbol] = position
        logger.info(f"Opened {side.value} position: {symbol} @ ${entry_price} x {quantity}")

        return position

    def close_position(
        self,
        symbol: str,
        exit_price: float,
        reason: str = "manual"
    ) -> Optional[Position]:
        """
        Close an existing position.

        Args:
            symbol: Trading symbol
            exit_price: Exit price
            reason: Close reason

        Returns:
            Closed position
        """
        if symbol not in self.positions:
            logger.warning(f"Position {symbol} not found")
            return None

        position = self.positions[symbol]
        position.update_price(exit_price)

        # Calculate realized P&L
        realized_pnl = position.unrealized_pnl

        # Update portfolio
        self.portfolio_value += realized_pnl
        self.daily_pnl += realized_pnl
        self.daily_trades += 1

        # Update peak and drawdown
        if self.portfolio_value > self.peak_portfolio_value:
            self.peak_portfolio_value = self.portfolio_value

        current_drawdown = (self.peak_portfolio_value - self.portfolio_value) / self.peak_portfolio_value * 100
        if current_drawdown > self.max_drawdown_reached:
            self.max_drawdown_reached = current_drawdown

        # Store metadata
        position.metadata['exit_price'] = exit_price
        position.metadata['exit_time'] = datetime.now()
        position.metadata['realized_pnl'] = realized_pnl
        position.metadata['close_reason'] = reason

        # Move to history
        self.closed_positions.append(position)
        del self.positions[symbol]

        logger.info(f"Closed position: {symbol} @ ${exit_price} | "
                   f"P&L: ${realized_pnl:,.2f} ({position.unrealized_pnl_percent:+.2f}%) | "
                   f"Reason: {reason}")

        return position

    def update_position_prices(self, prices: Dict[str, float]):
        """Update all positions with current prices."""
        for symbol, position in self.positions.items():
            if symbol in prices:
                position.update_price(prices[symbol])

                # Check stop loss and take profit
                if position.should_trigger_stop_loss():
                    logger.warning(f"Stop loss triggered for {symbol}")
                    self.close_position(symbol, position.current_price, "stop_loss")

                elif position.should_trigger_take_profit():
                    logger.info(f"Take profit triggered for {symbol}")
                    self.close_position(symbol, position.current_price, "take_profit")

    def update_trailing_stop(
        self,
        symbol: str,
        trailing_percent: float = 5.0
    ):
        """
        Update trailing stop loss.

        Args:
            symbol: Trading symbol
            trailing_percent: Trailing stop percentage
        """
        if symbol not in self.positions:
            return

        position = self.positions[symbol]

        if position.side == PositionSide.LONG:
            # Move stop up as price increases
            new_stop = position.current_price * (1 - trailing_percent / 100)
            if position.stop_loss is None or new_stop > position.stop_loss:
                position.stop_loss = new_stop
                logger.debug(f"Updated trailing stop for {symbol}: ${new_stop:.2f}")

        elif position.side == PositionSide.SHORT:
            # Move stop down as price decreases
            new_stop = position.current_price * (1 + trailing_percent / 100)
            if position.stop_loss is None or new_stop < position.stop_loss:
                position.stop_loss = new_stop
                logger.debug(f"Updated trailing stop for {symbol}: ${new_stop:.2f}")

    def _check_can_open_position(
        self,
        symbol: str,
        entry_price: float,
        quantity: float
    ) -> bool:
        """Check if position can be opened based on risk limits."""
        # Check max positions
        if len(self.positions) >= self.limits.max_positions:
            logger.warning(f"Cannot open position: max positions reached ({self.limits.max_positions})")
            return False

        # Check position size
        position_value = entry_price * quantity
        if position_value > self.limits.max_position_size:
            logger.warning(f"Cannot open position: exceeds max size (${position_value:,.2f} > "
                          f"${self.limits.max_position_size:,.2f})")
            return False

        # Check symbol exposure
        existing_exposure = 0
        if symbol in self.positions:
            existing_exposure = self.positions[symbol].value

        total_exposure = existing_exposure + position_value
        if total_exposure > self.limits.max_symbol_exposure:
            logger.warning(f"Cannot open position: exceeds max symbol exposure")
            return False

        # Check daily loss
        if self.daily_pnl < -self.limits.max_daily_loss:
            logger.warning(f"Cannot open position: daily loss limit reached (${self.daily_pnl:,.2f})")
            return False

        # Check drawdown
        current_drawdown = (self.peak_portfolio_value - self.portfolio_value) / self.peak_portfolio_value * 100
        if current_drawdown > self.limits.max_drawdown:
            logger.warning(f"Cannot open position: max drawdown reached ({current_drawdown:.1f}%)")
            return False

        return True

    def reset_daily_stats(self):
        """Reset daily statistics."""
        self.daily_pnl = 0.0
        self.daily_trades = 0
        self.last_reset_date = datetime.now().date()
        logger.info("Daily stats reset")

    def get_portfolio_stats(self) -> dict:
        """Get portfolio statistics."""
        total_value = self.portfolio_value
        positions_value = sum(p.value for p in self.positions.values())
        unrealized_pnl = sum(p.unrealized_pnl for p in self.positions.values())
        cash = total_value - positions_value

        return {
            'portfolio_value': total_value,
            'cash': cash,
            'positions_value': positions_value,
            'unrealized_pnl': unrealized_pnl,
            'daily_pnl': self.daily_pnl,
            'daily_trades': self.daily_trades,
            'num_positions': len(self.positions),
            'max_drawdown': self.max_drawdown_reached,
            'total_return': (total_value - self.initial_value) / self.initial_value * 100,
            'peak_value': self.peak_portfolio_value
        }

    def get_risk_metrics(self) -> dict:
        """Get risk metrics."""
        stats = self.get_portfolio_stats()

        # Calculate utilization
        position_limit_utilization = len(self.positions) / self.limits.max_positions * 100
        capital_utilization = stats['positions_value'] / self.portfolio_value * 100

        # Calculate concentration
        if self.positions:
            largest_position = max(p.value for p in self.positions.values())
            concentration = largest_position / self.portfolio_value * 100
        else:
            largest_position = 0
            concentration = 0

        return {
            'position_limit_utilization': position_limit_utilization,
            'capital_utilization': capital_utilization,
            'concentration': concentration,
            'largest_position_value': largest_position,
            'daily_loss_remaining': self.limits.max_daily_loss + self.daily_pnl,
            'drawdown_remaining': self.limits.max_drawdown - self.max_drawdown_reached
        }


if __name__ == '__main__':
    import logging
    logging.basicConfig(level=logging.INFO)

    print("🎯 Position Manager Demo")
    print("=" * 60)

    # Initialize position manager
    portfolio_value = 100000
    limits = PositionLimits(
        max_position_size=20000,
        max_positions=5,
        max_daily_loss=2000,
        max_drawdown=15
    )

    manager = PositionManager(portfolio_value, limits)

    print(f"\n💰 Portfolio: ${portfolio_value:,}")
    print(f"   Max Position Size: ${limits.max_position_size:,}")
    print(f"   Max Positions: {limits.max_positions}")
    print(f"   Max Daily Loss: ${limits.max_daily_loss:,}")

    # Example 1: Position Sizing
    print("\n" + "=" * 60)
    print("Example 1: Position Sizing")
    print("=" * 60)

    entry_price = 45000
    stop_loss = 44000
    risk_per_trade = 0.02  # 2% risk

    size = manager.calculate_position_size(
        'BTCUSD',
        entry_price,
        stop_loss,
        method='fixed_risk',
        risk_per_trade=risk_per_trade
    )

    risk_amount = portfolio_value * risk_per_trade
    print(f"\nFixed Risk Sizing:")
    print(f"   Entry: ${entry_price:,}")
    print(f"   Stop Loss: ${stop_loss:,}")
    print(f"   Risk per Unit: ${entry_price - stop_loss:,}")
    print(f"   Risk Amount: ${risk_amount:,}")
    print(f"   Position Size: {size:.4f} BTC")
    print(f"   Position Value: ${size * entry_price:,.2f}")

    # Example 2: Open and Manage Positions
    print("\n" + "=" * 60)
    print("Example 2: Open and Manage Positions")
    print("=" * 60)

    # Open long position
    position1 = manager.open_position(
        symbol='BTCUSD',
        side=PositionSide.LONG,
        entry_price=45000,
        quantity=0.5,
        stop_loss=44000,
        take_profit=47000
    )

    # Open short position
    position2 = manager.open_position(
        symbol='ETHUSD',
        side=PositionSide.SHORT,
        entry_price=2000,
        quantity=10,
        stop_loss=2100,
        take_profit=1900
    )

    print(f"\n✅ Opened 2 positions")

    # Update prices
    print(f"\n📊 Updating prices...")
    manager.update_position_prices({
        'BTCUSD': 46000,  # +$1000 profit
        'ETHUSD': 1950  # +$500 profit
    })

    # Show stats
    stats = manager.get_portfolio_stats()
    print(f"\nPortfolio Stats:")
    print(f"   Portfolio Value: ${stats['portfolio_value']:,.2f}")
    print(f"   Unrealized P&L: ${stats['unrealized_pnl']:,.2f}")
    print(f"   Positions: {stats['num_positions']}")

    # Example 3: Trailing Stop
    print("\n" + "=" * 60)
    print("Example 3: Trailing Stop Loss")
    print("=" * 60)

    print(f"\n📈 BTC moved to $47,000")
    manager.update_position_prices({'BTCUSD': 47000})
    print(f"   Current Stop: ${position1.stop_loss:,.2f}")

    manager.update_trailing_stop('BTCUSD', trailing_percent=5.0)
    print(f"   Updated Trailing Stop: ${position1.stop_loss:,.2f}")

    # Example 4: Risk Limits
    print("\n" + "=" * 60)
    print("Example 4: Risk Limits Enforcement")
    print("=" * 60)

    # Try to open too many positions
    for i in range(5):
        pos = manager.open_position(
            symbol=f'SYM{i}',
            side=PositionSide.LONG,
            entry_price=100,
            quantity=100
        )
        if pos is None:
            print(f"\n❌ Position {i+1} rejected by risk limits")
            break

    # Risk metrics
    risk = manager.get_risk_metrics()
    print(f"\nRisk Metrics:")
    print(f"   Position Limit Utilization: {risk['position_limit_utilization']:.1f}%")
    print(f"   Capital Utilization: {risk['capital_utilization']:.1f}%")
    print(f"   Concentration: {risk['concentration']:.1f}%")

    # Example 5: Close Positions
    print("\n" + "=" * 60)
    print("Example 5: Close Positions")
    print("=" * 60)

    manager.close_position('BTCUSD', 47500, reason="take_profit")
    manager.close_position('ETHUSD', 1900, reason="target_reached")

    final_stats = manager.get_portfolio_stats()
    print(f"\nFinal Portfolio Stats:")
    print(f"   Portfolio Value: ${final_stats['portfolio_value']:,.2f}")
    print(f"   Total Return: {final_stats['total_return']:+.2f}%")
    print(f"   Daily P&L: ${final_stats['daily_pnl']:,.2f}")
    print(f"   Closed Positions: {len(manager.closed_positions)}")

    print("\n✅ Position manager demo complete!")
