"""
Paper Trading Engine
Simulates realistic order execution with slippage, fees, and gas costs.
"""
import logging
from typing import Dict, List, Optional, Tuple
from datetime import datetime
from dataclasses import dataclass, field
from enum import Enum
import random

logger = logging.getLogger(__name__)


class OrderType(Enum):
    """Order types."""
    MARKET = "market"
    LIMIT = "limit"
    STOP_LOSS = "stop_loss"


class OrderSide(Enum):
    """Order side."""
    BUY = "buy"
    SELL = "sell"


class OrderStatus(Enum):
    """Order status."""
    PENDING = "pending"
    FILLED = "filled"
    PARTIALLY_FILLED = "partially_filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"


class Exchange(Enum):
    """Exchange type."""
    COINBASE = "coinbase"
    UNISWAP = "uniswap"
    SUSHISWAP = "sushiswap"


@dataclass
class Order:
    """Trading order."""
    order_id: str
    exchange: Exchange
    symbol: str
    side: OrderSide
    order_type: OrderType
    quantity: float
    price: Optional[float] = None  # Limit/stop price
    status: OrderStatus = OrderStatus.PENDING

    # Execution details
    filled_quantity: float = 0.0
    avg_fill_price: float = 0.0
    total_cost: float = 0.0
    fees: float = 0.0
    gas_cost: float = 0.0
    slippage: float = 0.0

    # Timestamps
    created_at: datetime = field(default_factory=datetime.now)
    filled_at: Optional[datetime] = None

    # Metadata
    metadata: Dict = field(default_factory=dict)


@dataclass
class Fill:
    """Order fill details."""
    order_id: str
    quantity: float
    price: float
    fees: float
    gas_cost: float
    timestamp: datetime

    @property
    def total_cost(self) -> float:
        """Total cost including fees and gas."""
        return (self.quantity * self.price) + self.fees + self.gas_cost


class PaperTradingEngine:
    """
    Paper trading engine with realistic execution simulation.

    Features:
    - Market order execution with slippage
    - Limit order matching
    - Stop loss triggers
    - CEX fees (0.5% Coinbase)
    - DEX fees (0.3% Uniswap)
    - Gas cost simulation
    - Order book impact modeling
    """

    # Fee structures
    CEX_TAKER_FEE = 0.005  # 0.5% Coinbase taker fee
    CEX_MAKER_FEE = 0.005  # 0.5% Coinbase maker fee
    DEX_SWAP_FEE = 0.003   # 0.3% Uniswap V2 fee

    # Gas costs (USD)
    GAS_COSTS = {
        Exchange.UNISWAP: 5.0,      # Average Uniswap swap
        Exchange.SUSHISWAP: 5.0,    # Average Sushi swap
        Exchange.COINBASE: 0.0       # No gas on CEX
    }

    def __init__(
        self,
        enable_slippage: bool = True,
        enable_fees: bool = True,
        enable_gas: bool = True,
        slippage_range: Tuple[float, float] = (0.0001, 0.005)  # 0.01% - 0.5%
    ):
        """
        Initialize paper trading engine.

        Args:
            enable_slippage: Enable slippage simulation
            enable_fees: Enable trading fees
            enable_gas: Enable gas costs for DEX
            slippage_range: Min/max slippage percentage
        """
        self.enable_slippage = enable_slippage
        self.enable_fees = enable_fees
        self.enable_gas = enable_gas
        self.slippage_range = slippage_range

        # Order tracking
        self.orders: Dict[str, Order] = {}
        self.order_counter = 0

        logger.info(f"Paper trading engine initialized "
                   f"(slippage={enable_slippage}, fees={enable_fees}, gas={enable_gas})")

    def _generate_order_id(self) -> str:
        """Generate unique order ID."""
        self.order_counter += 1
        return f"PAPER-{self.order_counter:06d}"

    def _calculate_slippage(
        self,
        exchange: Exchange,
        quantity: float,
        liquidity: float = 1000000.0
    ) -> float:
        """
        Calculate slippage based on order size and liquidity.

        Larger orders relative to liquidity = more slippage.

        Args:
            exchange: Exchange type
            quantity: Order quantity (USD value)
            liquidity: Available liquidity (USD)

        Returns:
            Slippage percentage
        """
        if not self.enable_slippage:
            return 0.0

        # Base slippage
        min_slip, max_slip = self.slippage_range

        # Impact based on order size vs liquidity
        impact_ratio = min(quantity / liquidity, 0.1)  # Cap at 10%

        # DEX has more slippage than CEX
        if exchange in [Exchange.UNISWAP, Exchange.SUSHISWAP]:
            slippage = min_slip + (impact_ratio * max_slip * 2)
        else:  # CEX
            slippage = min_slip + (impact_ratio * max_slip)

        return min(slippage, max_slip)

    def _calculate_fees(
        self,
        exchange: Exchange,
        order_type: OrderType,
        quantity: float,
        price: float
    ) -> float:
        """
        Calculate trading fees.

        Args:
            exchange: Exchange type
            order_type: Order type
            quantity: Quantity
            price: Price

        Returns:
            Fee amount in USD
        """
        if not self.enable_fees:
            return 0.0

        trade_value = quantity * price

        if exchange == Exchange.COINBASE:
            # CEX: taker fee for market orders, maker for limit
            fee_rate = self.CEX_TAKER_FEE if order_type == OrderType.MARKET else self.CEX_MAKER_FEE
            return trade_value * fee_rate
        else:  # DEX
            # DEX: always swap fee
            return trade_value * self.DEX_SWAP_FEE

    def _calculate_gas(
        self,
        exchange: Exchange,
        gas_price_gwei: float = 15.0
    ) -> float:
        """
        Calculate gas cost.

        Args:
            exchange: Exchange type
            gas_price_gwei: Current gas price

        Returns:
            Gas cost in USD
        """
        if not self.enable_gas:
            return 0.0

        # Scale base gas cost by current gas price
        # Base costs at 15 Gwei, scale linearly
        base_cost = self.GAS_COSTS.get(exchange, 0.0)
        scaling = gas_price_gwei / 15.0

        return base_cost * scaling

    def execute_market_order(
        self,
        exchange: Exchange,
        symbol: str,
        side: OrderSide,
        quantity: float,
        current_price: float,
        liquidity: float = 1000000.0,
        gas_price_gwei: float = 15.0
    ) -> Order:
        """
        Execute a market order with realistic simulation.

        Args:
            exchange: Exchange to trade on
            symbol: Trading pair
            side: Buy or sell
            quantity: Amount to trade
            current_price: Current market price
            liquidity: Available liquidity (for slippage)
            gas_price_gwei: Current gas price

        Returns:
            Executed order
        """
        order_id = self._generate_order_id()

        # Calculate slippage
        trade_value = quantity * current_price
        slippage_pct = self._calculate_slippage(exchange, trade_value, liquidity)

        # Apply slippage
        if side == OrderSide.BUY:
            # Buying: price goes up
            fill_price = current_price * (1 + slippage_pct)
        else:
            # Selling: price goes down
            fill_price = current_price * (1 - slippage_pct)

        # Calculate costs
        fees = self._calculate_fees(exchange, OrderType.MARKET, quantity, fill_price)
        gas_cost = self._calculate_gas(exchange, gas_price_gwei)

        # Calculate total
        if side == OrderSide.BUY:
            total_cost = (quantity * fill_price) + fees + gas_cost
        else:
            total_cost = (quantity * fill_price) - fees - gas_cost

        # Create order
        order = Order(
            order_id=order_id,
            exchange=exchange,
            symbol=symbol,
            side=side,
            order_type=OrderType.MARKET,
            quantity=quantity,
            status=OrderStatus.FILLED,
            filled_quantity=quantity,
            avg_fill_price=fill_price,
            total_cost=total_cost,
            fees=fees,
            gas_cost=gas_cost,
            slippage=slippage_pct * 100,  # Store as percentage
            filled_at=datetime.now()
        )

        self.orders[order_id] = order

        logger.info(f"Executed market order: {order_id} | "
                   f"{side.value} {quantity} {symbol} @ ${fill_price:.2f} | "
                   f"Slippage: {slippage_pct*100:.3f}% | "
                   f"Fees: ${fees:.2f} | Gas: ${gas_cost:.2f}")

        return order

    def execute_limit_order(
        self,
        exchange: Exchange,
        symbol: str,
        side: OrderSide,
        quantity: float,
        limit_price: float,
        current_price: float
    ) -> Order:
        """
        Execute a limit order (simplified - immediately checks if executable).

        Args:
            exchange: Exchange
            symbol: Trading pair
            side: Buy or sell
            quantity: Amount
            limit_price: Limit price
            current_price: Current market price

        Returns:
            Order (filled if price matches, pending otherwise)
        """
        order_id = self._generate_order_id()

        # Check if limit order would execute
        can_execute = (
            (side == OrderSide.BUY and current_price <= limit_price) or
            (side == OrderSide.SELL and current_price >= limit_price)
        )

        if can_execute:
            # Execute at limit price
            fees = self._calculate_fees(exchange, OrderType.LIMIT, quantity, limit_price)
            gas_cost = self._calculate_gas(exchange)

            if side == OrderSide.BUY:
                total_cost = (quantity * limit_price) + fees + gas_cost
            else:
                total_cost = (quantity * limit_price) - fees - gas_cost

            order = Order(
                order_id=order_id,
                exchange=exchange,
                symbol=symbol,
                side=side,
                order_type=OrderType.LIMIT,
                quantity=quantity,
                price=limit_price,
                status=OrderStatus.FILLED,
                filled_quantity=quantity,
                avg_fill_price=limit_price,
                total_cost=total_cost,
                fees=fees,
                gas_cost=gas_cost,
                slippage=0.0,  # No slippage on limit orders
                filled_at=datetime.now()
            )

            logger.info(f"Filled limit order: {order_id}")
        else:
            # Order stays pending
            order = Order(
                order_id=order_id,
                exchange=exchange,
                symbol=symbol,
                side=side,
                order_type=OrderType.LIMIT,
                quantity=quantity,
                price=limit_price,
                status=OrderStatus.PENDING
            )

            logger.info(f"Limit order pending: {order_id}")

        self.orders[order_id] = order
        return order

    def get_order(self, order_id: str) -> Optional[Order]:
        """Get order by ID."""
        return self.orders.get(order_id)

    def get_all_orders(self, status: Optional[OrderStatus] = None) -> List[Order]:
        """
        Get all orders, optionally filtered by status.

        Args:
            status: Filter by status (None = all orders)

        Returns:
            List of orders
        """
        orders = list(self.orders.values())

        if status:
            orders = [o for o in orders if o.status == status]

        return sorted(orders, key=lambda x: x.created_at, reverse=True)

    def cancel_order(self, order_id: str) -> bool:
        """
        Cancel a pending order.

        Args:
            order_id: Order ID

        Returns:
            True if cancelled, False if not found or already filled
        """
        order = self.orders.get(order_id)

        if not order:
            logger.warning(f"Order {order_id} not found")
            return False

        if order.status != OrderStatus.PENDING:
            logger.warning(f"Cannot cancel order {order_id} with status {order.status.value}")
            return False

        order.status = OrderStatus.CANCELLED
        logger.info(f"Cancelled order: {order_id}")
        return True


if __name__ == '__main__':
    import logging
    logging.basicConfig(level=logging.INFO)

    print("💰 Paper Trading Engine Demo")
    print("=" * 60)

    # Initialize engine
    engine = PaperTradingEngine(
        enable_slippage=True,
        enable_fees=True,
        enable_gas=True
    )

    print("\n1. Execute Market Buy on Coinbase")
    print("-" * 60)

    order1 = engine.execute_market_order(
        exchange=Exchange.COINBASE,
        symbol="ETH-USD",
        side=OrderSide.BUY,
        quantity=1.0,
        current_price=2150.00,
        liquidity=5000000.0,
        gas_price_gwei=15.0
    )

    print(f"Order ID: {order1.order_id}")
    print(f"Status: {order1.status.value}")
    print(f"Filled: {order1.filled_quantity} ETH @ ${order1.avg_fill_price:.2f}")
    print(f"Slippage: {order1.slippage:.3f}%")
    print(f"Fees: ${order1.fees:.2f}")
    print(f"Gas: ${order1.gas_cost:.2f}")
    print(f"Total Cost: ${order1.total_cost:.2f}")

    print("\n2. Execute Market Sell on Uniswap")
    print("-" * 60)

    order2 = engine.execute_market_order(
        exchange=Exchange.UNISWAP,
        symbol="ETH-USDC",
        side=OrderSide.SELL,
        quantity=1.0,
        current_price=2155.00,
        liquidity=2000000.0,  # Less liquidity = more slippage
        gas_price_gwei=25.0   # Higher gas
    )

    print(f"Order ID: {order2.order_id}")
    print(f"Filled: {order2.filled_quantity} ETH @ ${order2.avg_fill_price:.2f}")
    print(f"Slippage: {order2.slippage:.3f}%")
    print(f"Fees: ${order2.fees:.2f}")
    print(f"Gas: ${order2.gas_cost:.2f}")
    print(f"Total Received: ${order2.total_cost:.2f}")

    print("\n3. Limit Order Example")
    print("-" * 60)

    order3 = engine.execute_limit_order(
        exchange=Exchange.COINBASE,
        symbol="BTC-USD",
        side=OrderSide.BUY,
        quantity=0.1,
        limit_price=44000.00,
        current_price=45000.00  # Too high, order pending
    )

    print(f"Order ID: {order3.order_id}")
    print(f"Status: {order3.status.value}")
    print(f"Limit Price: ${order3.price:.2f}")
    print(f"Current Price: $45,000.00")

    print("\n4. All Orders Summary")
    print("-" * 60)

    all_orders = engine.get_all_orders()
    print(f"Total Orders: {len(all_orders)}")

    filled_orders = engine.get_all_orders(OrderStatus.FILLED)
    print(f"Filled: {len(filled_orders)}")

    pending_orders = engine.get_all_orders(OrderStatus.PENDING)
    print(f"Pending: {len(pending_orders)}")

    print("\n✅ Paper trading engine demo complete!")
