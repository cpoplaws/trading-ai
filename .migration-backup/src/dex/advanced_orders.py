"""
Advanced Order Types
TWAP, VWAP, and other sophisticated order execution strategies.
"""
import logging
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timedelta
import math

logger = logging.getLogger(__name__)


class OrderType(Enum):
    """Advanced order types."""
    TWAP = "twap"  # Time-Weighted Average Price
    VWAP = "vwap"  # Volume-Weighted Average Price
    ICEBERG = "iceberg"  # Hidden order size
    SNIPER = "sniper"  # Execute at optimal price
    ADAPTIVE = "adaptive"  # Adjust based on market conditions


@dataclass
class OrderSlice:
    """Single slice of a larger order."""
    slice_number: int
    execution_time: datetime
    quantity: float
    expected_price: float
    executed: bool = False
    actual_price: Optional[float] = None
    slippage: Optional[float] = None


@dataclass
class AdvancedOrder:
    """Advanced order configuration."""
    order_id: str
    order_type: OrderType
    token_in: str
    token_out: str
    total_quantity: float
    target_price: Optional[float] = None

    # Execution parameters
    num_slices: int = 10
    duration_minutes: int = 60
    start_time: datetime = field(default_factory=datetime.now)

    # Slices
    slices: List[OrderSlice] = field(default_factory=list)

    # Results
    total_executed: float = 0.0
    avg_execution_price: float = 0.0
    total_slippage: float = 0.0
    completion_percent: float = 0.0

    # Metadata
    metadata: Dict = field(default_factory=dict)


class TWAPExecutor:
    """
    Time-Weighted Average Price (TWAP) Executor

    Splits large orders into equal slices executed at regular intervals.

    Benefits:
    - Reduces market impact
    - Averages out price volatility
    - Predictable execution schedule
    """

    def __init__(self):
        """Initialize TWAP executor."""
        self.orders: Dict[str, AdvancedOrder] = {}
        self.order_counter = 0
        logger.info("TWAP executor initialized")

    def create_order(
        self,
        token_in: str,
        token_out: str,
        total_quantity: float,
        duration_minutes: int = 60,
        num_slices: int = 10
    ) -> AdvancedOrder:
        """
        Create TWAP order.

        Args:
            token_in: Input token
            token_out: Output token
            total_quantity: Total amount to trade
            duration_minutes: Duration to spread order over
            num_slices: Number of slices to split into

        Returns:
            AdvancedOrder
        """
        self.order_counter += 1
        order_id = f"TWAP-{self.order_counter:06d}"

        # Calculate slice size and timing
        slice_quantity = total_quantity / num_slices
        slice_interval = duration_minutes / num_slices

        # Create slices
        slices = []
        start_time = datetime.now()

        for i in range(num_slices):
            execution_time = start_time + timedelta(minutes=i * slice_interval)
            slice_obj = OrderSlice(
                slice_number=i + 1,
                execution_time=execution_time,
                quantity=slice_quantity,
                expected_price=0.0  # Will be set at execution
            )
            slices.append(slice_obj)

        order = AdvancedOrder(
            order_id=order_id,
            order_type=OrderType.TWAP,
            token_in=token_in,
            token_out=token_out,
            total_quantity=total_quantity,
            num_slices=num_slices,
            duration_minutes=duration_minutes,
            start_time=start_time,
            slices=slices,
            metadata={
                'slice_quantity': slice_quantity,
                'slice_interval_minutes': slice_interval
            }
        )

        self.orders[order_id] = order

        logger.info(
            f"Created TWAP order {order_id}: "
            f"{total_quantity} {token_in} → {token_out} "
            f"over {duration_minutes}min in {num_slices} slices"
        )

        return order

    def execute_slice(
        self,
        order: AdvancedOrder,
        slice_number: int,
        current_price: float
    ) -> OrderSlice:
        """
        Execute a single slice.

        Args:
            order: Parent order
            slice_number: Slice to execute (1-indexed)
            current_price: Current market price

        Returns:
            Executed slice
        """
        slice_obj = order.slices[slice_number - 1]

        if slice_obj.executed:
            logger.warning(f"Slice {slice_number} already executed")
            return slice_obj

        # Simulate execution
        slice_obj.executed = True
        slice_obj.actual_price = current_price
        slice_obj.expected_price = current_price

        # Calculate slippage (small for TWAP slices)
        slice_obj.slippage = 0.001  # 0.1% average slippage

        # Update order totals
        order.total_executed += slice_obj.quantity
        order.completion_percent = (order.total_executed / order.total_quantity) * 100

        # Update average price
        if order.avg_execution_price == 0:
            order.avg_execution_price = current_price
        else:
            order.avg_execution_price = (
                (order.avg_execution_price * (slice_number - 1) + current_price) / slice_number
            )

        order.total_slippage += slice_obj.slippage

        logger.info(
            f"Executed {order.order_id} slice {slice_number}/{order.num_slices}: "
            f"{slice_obj.quantity:.4f} @ ${current_price:.2f} "
            f"({order.completion_percent:.1f}% complete)"
        )

        return slice_obj

    def simulate_execution(
        self,
        order: AdvancedOrder,
        price_path: List[float]
    ) -> Dict:
        """
        Simulate full TWAP execution with price path.

        Args:
            order: Order to execute
            price_path: List of prices at each slice time

        Returns:
            Execution results
        """
        if len(price_path) != order.num_slices:
            raise ValueError(f"Price path must have {order.num_slices} prices")

        for i, price in enumerate(price_path):
            self.execute_slice(order, i + 1, price)

        # Calculate results
        total_cost = sum(
            slice_obj.quantity * slice_obj.actual_price
            for slice_obj in order.slices
            if slice_obj.executed
        )

        avg_price = order.avg_execution_price
        total_slippage_pct = order.total_slippage / order.num_slices * 100

        return {
            'order_id': order.order_id,
            'completed': order.completion_percent == 100,
            'total_executed': order.total_executed,
            'avg_price': avg_price,
            'total_cost': total_cost,
            'total_slippage_percent': total_slippage_pct,
            'num_slices_executed': sum(1 for s in order.slices if s.executed)
        }


class VWAPExecutor:
    """
    Volume-Weighted Average Price (VWAP) Executor

    Allocates order quantity based on historical volume patterns.

    Benefits:
    - Executes more during high-volume periods
    - Aims to match market average price
    - Reduces information leakage
    """

    # Typical intraday volume distribution (hourly %)
    VOLUME_PROFILE = [
        0.03,  # 00:00-01:00
        0.02,  # 01:00-02:00
        0.02,  # 02:00-03:00
        0.02,  # 03:00-04:00
        0.03,  # 04:00-05:00
        0.04,  # 05:00-06:00
        0.05,  # 06:00-07:00
        0.06,  # 07:00-08:00
        0.08,  # 08:00-09:00 (opening)
        0.10,  # 09:00-10:00
        0.09,  # 10:00-11:00
        0.08,  # 11:00-12:00
        0.07,  # 12:00-13:00
        0.06,  # 13:00-14:00
        0.07,  # 14:00-15:00
        0.09,  # 15:00-16:00 (closing)
        0.06,  # 16:00-17:00
        0.04,  # 17:00-18:00
        0.03,  # 18:00-19:00
        0.02,  # 19:00-20:00
        0.02,  # 20:00-21:00
        0.01,  # 21:00-22:00
        0.01,  # 22:00-23:00
        0.01   # 23:00-24:00
    ]

    def __init__(self):
        """Initialize VWAP executor."""
        self.orders: Dict[str, AdvancedOrder] = {}
        self.order_counter = 0
        logger.info("VWAP executor initialized")

    def create_order(
        self,
        token_in: str,
        token_out: str,
        total_quantity: float,
        duration_hours: int = 8,
        start_hour: int = 9  # 9 AM start
    ) -> AdvancedOrder:
        """
        Create VWAP order with volume-weighted slicing.

        Args:
            token_in: Input token
            token_out: Output token
            total_quantity: Total amount to trade
            duration_hours: Duration in hours
            start_hour: Starting hour (0-23)

        Returns:
            AdvancedOrder
        """
        self.order_counter += 1
        order_id = f"VWAP-{self.order_counter:06d}"

        # Get volume weights for execution hours
        hours = [(start_hour + i) % 24 for i in range(duration_hours)]
        volume_weights = [self.VOLUME_PROFILE[h] for h in hours]

        # Normalize weights
        total_weight = sum(volume_weights)
        normalized_weights = [w / total_weight for w in volume_weights]

        # Create slices based on volume
        slices = []
        start_time = datetime.now()

        for i, weight in enumerate(normalized_weights):
            slice_quantity = total_quantity * weight
            execution_time = start_time + timedelta(hours=i)

            slice_obj = OrderSlice(
                slice_number=i + 1,
                execution_time=execution_time,
                quantity=slice_quantity,
                expected_price=0.0
            )
            slices.append(slice_obj)

        order = AdvancedOrder(
            order_id=order_id,
            order_type=OrderType.VWAP,
            token_in=token_in,
            token_out=token_out,
            total_quantity=total_quantity,
            num_slices=duration_hours,
            duration_minutes=duration_hours * 60,
            start_time=start_time,
            slices=slices,
            metadata={
                'start_hour': start_hour,
                'volume_weights': normalized_weights
            }
        )

        self.orders[order_id] = order

        logger.info(
            f"Created VWAP order {order_id}: "
            f"{total_quantity} {token_in} → {token_out} "
            f"over {duration_hours}hrs starting {start_hour}:00"
        )

        return order

    def get_target_vwap(self, price_volume_pairs: List[Tuple[float, float]]) -> float:
        """
        Calculate target VWAP from price-volume pairs.

        Args:
            price_volume_pairs: List of (price, volume) tuples

        Returns:
            VWAP price
        """
        if not price_volume_pairs:
            return 0.0

        total_pv = sum(price * volume for price, volume in price_volume_pairs)
        total_volume = sum(volume for _, volume in price_volume_pairs)

        return total_pv / total_volume if total_volume > 0 else 0.0


class IcebergOrder:
    """
    Iceberg Order Executor

    Shows only small portion of total order size.

    Benefits:
    - Hides true order size
    - Prevents frontrunning
    - Reduces market impact signaling
    """

    def __init__(self, visible_percent: float = 0.1):
        """
        Initialize iceberg executor.

        Args:
            visible_percent: Percent of order to show (0.1 = 10%)
        """
        self.visible_percent = visible_percent
        self.orders: Dict[str, AdvancedOrder] = {}
        self.order_counter = 0
        logger.info(f"Iceberg order executor initialized (visible: {visible_percent*100}%)")

    def create_order(
        self,
        token_in: str,
        token_out: str,
        total_quantity: float,
        visible_quantity: Optional[float] = None
    ) -> AdvancedOrder:
        """
        Create iceberg order.

        Args:
            token_in: Input token
            token_out: Output token
            total_quantity: Total hidden amount
            visible_quantity: Visible amount (default: visible_percent of total)

        Returns:
            AdvancedOrder
        """
        self.order_counter += 1
        order_id = f"ICE-{self.order_counter:06d}"

        if visible_quantity is None:
            visible_quantity = total_quantity * self.visible_percent

        num_slices = math.ceil(total_quantity / visible_quantity)

        # Create slices
        slices = []
        remaining = total_quantity

        for i in range(num_slices):
            slice_qty = min(visible_quantity, remaining)
            slice_obj = OrderSlice(
                slice_number=i + 1,
                execution_time=datetime.now(),  # Executes as fills
                quantity=slice_qty,
                expected_price=0.0
            )
            slices.append(slice_obj)
            remaining -= slice_qty

        order = AdvancedOrder(
            order_id=order_id,
            order_type=OrderType.ICEBERG,
            token_in=token_in,
            token_out=token_out,
            total_quantity=total_quantity,
            num_slices=num_slices,
            slices=slices,
            metadata={
                'visible_quantity': visible_quantity,
                'hidden_quantity': total_quantity - visible_quantity
            }
        )

        self.orders[order_id] = order

        logger.info(
            f"Created iceberg order {order_id}: "
            f"{total_quantity} {token_in} (showing {visible_quantity})"
        )

        return order


def compare_execution_strategies(
    total_quantity: float,
    price_path: List[float]
) -> str:
    """
    Compare different execution strategies.

    Args:
        total_quantity: Total amount to trade
        price_path: Price evolution over time

    Returns:
        Comparison report
    """
    # Market order (single execution at first price)
    market_price = price_path[0]
    market_cost = total_quantity * market_price

    # TWAP execution
    twap = TWAPExecutor()
    twap_order = twap.create_order("ETH", "USDC", total_quantity, duration_minutes=len(price_path), num_slices=len(price_path))
    twap_result = twap.simulate_execution(twap_order, price_path)
    twap_cost = twap_result['total_cost']

    # Average price (for comparison)
    avg_price = sum(price_path) / len(price_path)
    avg_cost = total_quantity * avg_price

    # Generate report
    report = f"""
╔══════════════════════════════════════════════════════════════╗
║           EXECUTION STRATEGY COMPARISON                       ║
╠══════════════════════════════════════════════════════════════╣
║ Order Size: {total_quantity:>10.2f} ETH                                ║
║ Price Range: ${min(price_path):>10.2f} - ${max(price_path):>10.2f}                     ║
╠══════════════════════════════════════════════════════════════╣
║ MARKET ORDER (immediate)                                      ║
║   Price: ${market_price:>15.2f}                                   ║
║   Cost:  ${market_cost:>15,.2f}                                   ║
╠══════════════════════════════════════════════════════════════╣
║ TWAP ({len(price_path)} slices)                                         ║
║   Avg Price: ${twap_result['avg_price']:>10.2f}                                   ║
║   Cost:      ${twap_cost:>10,.2f}                                   ║
║   Savings:   ${market_cost - twap_cost:>10,.2f} ({((market_cost - twap_cost)/market_cost*100):>5.2f}%)              ║
╠══════════════════════════════════════════════════════════════╣
║ BENCHMARK (avg price)                                         ║
║   Avg Price: ${avg_price:>10.2f}                                   ║
║   Cost:      ${avg_cost:>10,.2f}                                   ║
╚══════════════════════════════════════════════════════════════╝
"""

    return report


if __name__ == '__main__':
    import logging
    logging.basicConfig(level=logging.INFO)

    print("🎯 Advanced Order Types Demo")
    print("=" * 60)

    # Test TWAP
    print("\n1. TWAP Order Execution")
    print("-" * 60)

    twap = TWAPExecutor()
    order = twap.create_order(
        token_in="ETH",
        token_out="USDC",
        total_quantity=10.0,
        duration_minutes=60,
        num_slices=10
    )

    print(f"\nOrder Details:")
    print(f"  Order ID: {order.order_id}")
    print(f"  Type: {order.order_type.value.upper()}")
    print(f"  Quantity: {order.total_quantity} {order.token_in}")
    print(f"  Slices: {order.num_slices}")
    print(f"  Duration: {order.duration_minutes} minutes")
    print(f"  Slice size: {order.metadata['slice_quantity']:.4f} {order.token_in}")

    # Simulate execution with varying prices
    prices = [2100 + i * 5 for i in range(10)]  # Prices increasing
    print(f"\nSimulating execution...")
    print(f"  Price range: ${min(prices):.2f} - ${max(prices):.2f}")

    result = twap.simulate_execution(order, prices)

    print(f"\nResults:")
    print(f"  Completed: {result['completed']}")
    print(f"  Avg Price: ${result['avg_price']:.2f}")
    print(f"  Total Cost: ${result['total_cost']:,.2f}")
    print(f"  Slippage: {result['total_slippage_percent']:.3f}%")

    # Test VWAP
    print("\n2. VWAP Order Creation")
    print("-" * 60)

    vwap = VWAPExecutor()
    vwap_order = vwap.create_order(
        token_in="ETH",
        token_out="USDC",
        total_quantity=50.0,
        duration_hours=8,
        start_hour=9
    )

    print(f"\nVWAP Slices (volume-weighted):")
    for i, slice_obj in enumerate(vwap_order.slices[:5], 1):  # Show first 5
        weight = vwap_order.metadata['volume_weights'][i-1]
        print(f"  Slice {i}: {slice_obj.quantity:.4f} ETH ({weight*100:.1f}% of total)")
    print(f"  ... ({vwap_order.num_slices - 5} more slices)")

    # Test Iceberg
    print("\n3. Iceberg Order")
    print("-" * 60)

    iceberg = IcebergOrder(visible_percent=0.1)
    ice_order = iceberg.create_order(
        token_in="ETH",
        token_out="USDC",
        total_quantity=100.0
    )

    print(f"\nIceberg Order:")
    print(f"  Total Size: {ice_order.total_quantity} {ice_order.token_in} (hidden)")
    print(f"  Visible: {ice_order.metadata['visible_quantity']} {ice_order.token_in}")
    print(f"  Hidden: {ice_order.metadata['hidden_quantity']} {ice_order.token_in}")
    print(f"  Will execute in {ice_order.num_slices} fills")

    # Strategy comparison
    print("\n4. Strategy Comparison")
    print("-" * 60)

    # Simulate price movement
    import random
    random.seed(42)
    base = 2100
    price_path = [base + random.uniform(-50, 50) for _ in range(20)]

    comparison = compare_execution_strategies(10.0, price_path)
    print(comparison)

    print("✅ Advanced order types demo complete!")
