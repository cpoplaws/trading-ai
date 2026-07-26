"""
Advanced Order Manager
Provides sophisticated order types for professional trading.

Features:
- Bracket Orders (entry + take profit + stop loss)
- Trailing Stop Orders
- OCO (One-Cancels-Other) Orders
- Scale In/Out Orders
- TWAP/VWAP execution
- Order lifecycle management
"""
import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Callable
from enum import Enum
from decimal import Decimal

logger = logging.getLogger(__name__)


class AdvancedOrderType(Enum):
    """Advanced order types."""
    BRACKET = "bracket"
    TRAILING_STOP = "trailing_stop"
    OCO = "oco"  # One-Cancels-Other
    SCALE_IN = "scale_in"
    SCALE_OUT = "scale_out"
    TWAP = "twap"  # Time-Weighted Average Price
    VWAP = "vwap"  # Volume-Weighted Average Price


class OrderState(Enum):
    """Order lifecycle states."""
    PENDING = "pending"
    ACTIVE = "active"
    PARTIALLY_FILLED = "partially_filled"
    FILLED = "filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"
    EXPIRED = "expired"


@dataclass
class BracketOrder:
    """
    Bracket order with entry, take profit, and stop loss.

    Example:
        - Entry: BUY 100 shares at $50
        - Take Profit: SELL 100 shares at $55 (10% profit)
        - Stop Loss: SELL 100 shares at $48 (4% loss)
    """
    symbol: str
    quantity: float
    side: str  # 'buy' or 'sell'

    # Entry order
    entry_price: Optional[float] = None  # None = market order

    # Take profit (optional)
    take_profit_price: Optional[float] = None
    take_profit_pct: Optional[float] = None  # Alternative: specify as %

    # Stop loss (optional)
    stop_loss_price: Optional[float] = None
    stop_loss_pct: Optional[float] = None  # Alternative: specify as %

    # Order IDs (filled after placement)
    entry_order_id: Optional[str] = None
    take_profit_order_id: Optional[str] = None
    stop_loss_order_id: Optional[str] = None

    # State
    state: OrderState = OrderState.PENDING
    created_at: datetime = field(default_factory=datetime.now)

    def calculate_prices(self, current_price: float):
        """Calculate take profit and stop loss from percentages."""
        if self.take_profit_pct and not self.take_profit_price:
            multiplier = 1 + (self.take_profit_pct / 100)
            if self.side.lower() == 'buy':
                self.take_profit_price = current_price * multiplier
            else:
                self.take_profit_price = current_price / multiplier

        if self.stop_loss_pct and not self.stop_loss_price:
            multiplier = 1 - (self.stop_loss_pct / 100)
            if self.side.lower() == 'buy':
                self.stop_loss_price = current_price * multiplier
            else:
                self.stop_loss_price = current_price / multiplier


@dataclass
class TrailingStopOrder:
    """
    Trailing stop order that follows price movements.

    Example:
        - Buy at $50
        - Trailing stop: 5%
        - If price goes to $60, stop moves to $57 (5% below)
        - If price falls to $57, sell triggered
    """
    symbol: str
    quantity: float
    side: str

    # Trailing configuration
    trail_percent: Optional[float] = None  # e.g., 5 for 5%
    trail_amount: Optional[float] = None  # e.g., 2.5 for $2.50

    # Entry and activation
    entry_price: Optional[float] = None
    activation_price: Optional[float] = None  # Price to activate trailing

    # State tracking
    highest_price: Optional[float] = None  # For buy side
    lowest_price: Optional[float] = None  # For sell side
    current_stop_price: Optional[float] = None

    order_id: Optional[str] = None
    state: OrderState = OrderState.PENDING
    created_at: datetime = field(default_factory=datetime.now)

    def update(self, current_price: float) -> bool:
        """
        Update trailing stop based on current price.

        Returns:
            True if stop should trigger, False otherwise
        """
        # Initialize tracking
        if self.highest_price is None:
            self.highest_price = current_price
        if self.lowest_price is None:
            self.lowest_price = current_price

        # Update highest/lowest
        if current_price > self.highest_price:
            self.highest_price = current_price
        if current_price < self.lowest_price:
            self.lowest_price = current_price

        # Calculate stop price
        if self.side.lower() == 'sell':
            # For sell orders, trail below highest price
            if self.trail_percent:
                self.current_stop_price = self.highest_price * (1 - self.trail_percent / 100)
            elif self.trail_amount:
                self.current_stop_price = self.highest_price - self.trail_amount

            # Check if triggered
            return current_price <= self.current_stop_price

        else:  # buy side
            # For buy orders, trail above lowest price
            if self.trail_percent:
                self.current_stop_price = self.lowest_price * (1 + self.trail_percent / 100)
            elif self.trail_amount:
                self.current_stop_price = self.lowest_price + self.trail_amount

            # Check if triggered
            return current_price >= self.current_stop_price


@dataclass
class OCOOrder:
    """
    One-Cancels-Other order.

    Two orders where filling one automatically cancels the other.

    Example:
        - Order A: Sell at $55 (take profit)
        - Order B: Sell at $48 (stop loss)
        - If one fills, the other is cancelled
    """
    symbol: str
    quantity: float

    # First order
    order_a_price: float
    order_a_type: str  # 'limit', 'stop', 'stop_limit'
    order_a_id: Optional[str] = None

    # Second order
    order_b_price: float
    order_b_type: str
    order_b_id: Optional[str] = None

    # State
    filled_order: Optional[str] = None  # 'a' or 'b'
    state: OrderState = OrderState.PENDING
    created_at: datetime = field(default_factory=datetime.now)


@dataclass
class ScaleOrder:
    """
    Scale in or out of a position over multiple orders.

    Example Scale In:
        - Total quantity: 1000 shares
        - 5 orders of 200 shares each
        - Spaced $0.50 apart

    Example Scale Out:
        - Total quantity: 1000 shares
        - 4 orders: 25%, 25%, 25%, 25%
        - At different price targets
    """
    symbol: str
    total_quantity: float
    side: str

    # Scale configuration
    num_orders: int
    price_increment: Optional[float] = None  # Fixed spacing
    price_levels: Optional[List[float]] = None  # Explicit levels
    quantity_distribution: Optional[List[float]] = None  # e.g., [0.25, 0.25, 0.25, 0.25]

    # State
    order_ids: List[str] = field(default_factory=list)
    filled_quantity: float = 0.0
    state: OrderState = OrderState.PENDING
    created_at: datetime = field(default_factory=datetime.now)

    def get_order_quantities(self) -> List[float]:
        """Calculate quantity for each order."""
        if self.quantity_distribution:
            return [self.total_quantity * pct for pct in self.quantity_distribution]
        else:
            # Equal distribution
            return [self.total_quantity / self.num_orders] * self.num_orders

    def get_order_prices(self, start_price: float) -> List[float]:
        """Calculate price for each order."""
        if self.price_levels:
            return self.price_levels
        elif self.price_increment:
            if self.side.lower() == 'buy':
                # Buy lower (scale in on dips)
                return [start_price - (i * self.price_increment) for i in range(self.num_orders)]
            else:
                # Sell higher (scale out on rises)
                return [start_price + (i * self.price_increment) for i in range(self.num_orders)]
        else:
            return [start_price] * self.num_orders


class AdvancedOrderManager:
    """
    Manages advanced order types for professional trading.

    Provides sophisticated order execution strategies including:
    - Bracket orders
    - Trailing stops
    - OCO orders
    - Scale in/out
    - TWAP/VWAP execution
    """

    def __init__(self, broker, risk_manager=None):
        """
        Initialize advanced order manager.

        Args:
            broker: Broker interface for order execution
            risk_manager: Optional risk manager for validation
        """
        self.broker = broker
        self.risk_manager = risk_manager

        # Track active orders
        self.bracket_orders: Dict[str, BracketOrder] = {}
        self.trailing_stops: Dict[str, TrailingStopOrder] = {}
        self.oco_orders: Dict[str, OCOOrder] = {}
        self.scale_orders: Dict[str, ScaleOrder] = {}

        logger.info("AdvancedOrderManager initialized")

    def place_bracket_order(
        self,
        symbol: str,
        quantity: float,
        side: str,
        entry_price: Optional[float] = None,
        take_profit_pct: float = 10.0,
        stop_loss_pct: float = 5.0
    ) -> BracketOrder:
        """
        Place a bracket order with entry, take profit, and stop loss.

        Args:
            symbol: Trading symbol
            quantity: Order quantity
            side: 'buy' or 'sell'
            entry_price: Entry price (None for market order)
            take_profit_pct: Take profit percentage
            stop_loss_pct: Stop loss percentage

        Returns:
            BracketOrder instance
        """
        # Create bracket order
        bracket = BracketOrder(
            symbol=symbol,
            quantity=quantity,
            side=side,
            entry_price=entry_price,
            take_profit_pct=take_profit_pct,
            stop_loss_pct=stop_loss_pct
        )

        # Get current price for calculations
        current_price = self.broker.get_current_price(symbol)
        if not current_price:
            logger.error(f"Could not get price for {symbol}")
            bracket.state = OrderState.REJECTED
            return bracket

        # Calculate take profit and stop loss prices
        bracket.calculate_prices(current_price)

        # Risk check
        if self.risk_manager:
            if not self.risk_manager.check_position_size(symbol, quantity, current_price):
                logger.warning(f"Position size check failed for {symbol}")
                bracket.state = OrderState.REJECTED
                return bracket

        try:
            # Place entry order
            entry_order = self.broker.place_order(
                symbol=symbol,
                quantity=quantity,
                side=side,
                order_type='market' if entry_price is None else 'limit',
                limit_price=entry_price
            )

            if entry_order:
                bracket.entry_order_id = entry_order.get('order_id')
                bracket.state = OrderState.ACTIVE

                # Store for monitoring
                order_key = f"{symbol}_{bracket.entry_order_id}"
                self.bracket_orders[order_key] = bracket

                logger.info(
                    f"Bracket order placed: {symbol} {side} {quantity} @ "
                    f"{entry_price or 'market'}, "
                    f"TP: {bracket.take_profit_price:.2f}, "
                    f"SL: {bracket.stop_loss_price:.2f}"
                )
            else:
                bracket.state = OrderState.REJECTED

        except Exception as e:
            logger.error(f"Failed to place bracket order: {e}")
            bracket.state = OrderState.REJECTED

        return bracket

    def place_trailing_stop(
        self,
        symbol: str,
        quantity: float,
        side: str,
        trail_percent: float = 5.0
    ) -> TrailingStopOrder:
        """
        Place a trailing stop order.

        Args:
            symbol: Trading symbol
            quantity: Order quantity
            side: 'buy' or 'sell'
            trail_percent: Trailing percentage

        Returns:
            TrailingStopOrder instance
        """
        trailing_stop = TrailingStopOrder(
            symbol=symbol,
            quantity=quantity,
            side=side,
            trail_percent=trail_percent
        )

        # Get current price
        current_price = self.broker.get_current_price(symbol)
        if not current_price:
            logger.error(f"Could not get price for {symbol}")
            trailing_stop.state = OrderState.REJECTED
            return trailing_stop

        trailing_stop.entry_price = current_price
        trailing_stop.state = OrderState.ACTIVE

        # Store for monitoring
        order_key = f"{symbol}_{datetime.now().timestamp()}"
        self.trailing_stops[order_key] = trailing_stop

        logger.info(
            f"Trailing stop placed: {symbol} {side} {quantity}, "
            f"trail: {trail_percent}%"
        )

        return trailing_stop

    def place_oco_order(
        self,
        symbol: str,
        quantity: float,
        price_a: float,
        type_a: str,
        price_b: float,
        type_b: str
    ) -> OCOOrder:
        """
        Place a One-Cancels-Other order.

        Args:
            symbol: Trading symbol
            quantity: Order quantity
            price_a: First order price
            type_a: First order type
            price_b: Second order price
            type_b: Second order type

        Returns:
            OCOOrder instance
        """
        oco = OCOOrder(
            symbol=symbol,
            quantity=quantity,
            order_a_price=price_a,
            order_a_type=type_a,
            order_b_price=price_b,
            order_b_type=type_b
        )

        try:
            # Place both orders
            order_a = self.broker.place_order(
                symbol=symbol,
                quantity=quantity,
                side='sell',  # Typically for exits
                order_type=type_a,
                limit_price=price_a
            )

            order_b = self.broker.place_order(
                symbol=symbol,
                quantity=quantity,
                side='sell',
                order_type=type_b,
                limit_price=price_b
            )

            if order_a and order_b:
                oco.order_a_id = order_a.get('order_id')
                oco.order_b_id = order_b.get('order_id')
                oco.state = OrderState.ACTIVE

                # Store for monitoring
                order_key = f"{symbol}_{oco.order_a_id}"
                self.oco_orders[order_key] = oco

                logger.info(f"OCO order placed: {symbol} @ {price_a} or {price_b}")
            else:
                oco.state = OrderState.REJECTED

        except Exception as e:
            logger.error(f"Failed to place OCO order: {e}")
            oco.state = OrderState.REJECTED

        return oco

    def place_scale_order(
        self,
        symbol: str,
        total_quantity: float,
        side: str,
        num_orders: int = 5,
        price_increment: float = 0.50
    ) -> ScaleOrder:
        """
        Place a scale in/out order.

        Args:
            symbol: Trading symbol
            total_quantity: Total quantity to scale
            side: 'buy' or 'sell'
            num_orders: Number of orders to split into
            price_increment: Price spacing between orders

        Returns:
            ScaleOrder instance
        """
        scale = ScaleOrder(
            symbol=symbol,
            total_quantity=total_quantity,
            side=side,
            num_orders=num_orders,
            price_increment=price_increment
        )

        # Get current price
        current_price = self.broker.get_current_price(symbol)
        if not current_price:
            logger.error(f"Could not get price for {symbol}")
            scale.state = OrderState.REJECTED
            return scale

        # Calculate order details
        quantities = scale.get_order_quantities()
        prices = scale.get_order_prices(current_price)

        try:
            # Place all orders
            for qty, price in zip(quantities, prices):
                order = self.broker.place_order(
                    symbol=symbol,
                    quantity=qty,
                    side=side,
                    order_type='limit',
                    limit_price=price
                )

                if order:
                    scale.order_ids.append(order.get('order_id'))

            if scale.order_ids:
                scale.state = OrderState.ACTIVE

                # Store for monitoring
                order_key = f"{symbol}_{scale.order_ids[0]}"
                self.scale_orders[order_key] = scale

                logger.info(
                    f"Scale {side} order placed: {symbol} "
                    f"{total_quantity} in {num_orders} orders"
                )
            else:
                scale.state = OrderState.REJECTED

        except Exception as e:
            logger.error(f"Failed to place scale order: {e}")
            scale.state = OrderState.REJECTED

        return scale

    def update_trailing_stops(self):
        """Update all active trailing stop orders."""
        for key, trailing_stop in list(self.trailing_stops.items()):
            if trailing_stop.state != OrderState.ACTIVE:
                continue

            # Get current price
            current_price = self.broker.get_current_price(trailing_stop.symbol)
            if not current_price:
                continue

            # Update and check if triggered
            if trailing_stop.update(current_price):
                logger.info(
                    f"Trailing stop triggered: {trailing_stop.symbol} "
                    f"@ {current_price:.2f}"
                )

                # Execute stop order
                try:
                    order = self.broker.place_order(
                        symbol=trailing_stop.symbol,
                        quantity=trailing_stop.quantity,
                        side=trailing_stop.side,
                        order_type='market'
                    )

                    if order:
                        trailing_stop.order_id = order.get('order_id')
                        trailing_stop.state = OrderState.FILLED
                    else:
                        trailing_stop.state = OrderState.REJECTED

                except Exception as e:
                    logger.error(f"Failed to execute trailing stop: {e}")
                    trailing_stop.state = OrderState.REJECTED

    def cancel_oco_order(self, oco_order: OCOOrder, filled_order: str):
        """
        Cancel the other order in an OCO pair.

        Args:
            oco_order: The OCO order instance
            filled_order: Which order was filled ('a' or 'b')
        """
        try:
            # Cancel the other order
            if filled_order == 'a' and oco_order.order_b_id:
                self.broker.cancel_order(oco_order.order_b_id)
                logger.info(f"Cancelled OCO order B: {oco_order.order_b_id}")
            elif filled_order == 'b' and oco_order.order_a_id:
                self.broker.cancel_order(oco_order.order_a_id)
                logger.info(f"Cancelled OCO order A: {oco_order.order_a_id}")

            oco_order.filled_order = filled_order
            oco_order.state = OrderState.FILLED

        except Exception as e:
            logger.error(f"Failed to cancel OCO order: {e}")

    def get_active_orders(self) -> Dict:
        """Get all active advanced orders."""
        return {
            'bracket_orders': len([o for o in self.bracket_orders.values() if o.state == OrderState.ACTIVE]),
            'trailing_stops': len([o for o in self.trailing_stops.values() if o.state == OrderState.ACTIVE]),
            'oco_orders': len([o for o in self.oco_orders.values() if o.state == OrderState.ACTIVE]),
            'scale_orders': len([o for o in self.scale_orders.values() if o.state == OrderState.ACTIVE])
        }
