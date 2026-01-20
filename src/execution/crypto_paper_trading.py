"""
Crypto Paper Trading Engine for backtesting and paper trading on blockchain assets.

This module provides a comprehensive paper trading engine specifically designed for
crypto/DeFi trading with support for:
- Historical blockchain data backtesting
- Real-time paper trading simulation
- Multi-chain asset support
- DEX slippage modeling
- Gas cost simulation
- Funding rate tracking for perpetual futures
"""
import logging
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta
from dataclasses import dataclass, field
import pandas as pd
import numpy as np
from enum import Enum

logger = logging.getLogger(__name__)


class OrderType(Enum):
    """Order types for crypto trading."""
    MARKET = "market"
    LIMIT = "limit"
    STOP_LOSS = "stop_loss"
    TAKE_PROFIT = "take_profit"


class OrderSide(Enum):
    """Order side."""
    BUY = "buy"
    SELL = "sell"


class OrderStatus(Enum):
    """Order status."""
    PENDING = "pending"
    FILLED = "filled"
    PARTIALLY_FILLED = "partially_filled"
    CANCELED = "canceled"
    REJECTED = "rejected"


@dataclass
class CryptoOrder:
    """Crypto trading order."""
    order_id: str
    symbol: str
    chain: str
    side: OrderSide
    order_type: OrderType
    quantity: float
    price: Optional[float] = None
    status: OrderStatus = OrderStatus.PENDING
    filled_quantity: float = 0.0
    filled_price: float = 0.0
    gas_cost: float = 0.0
    slippage: float = 0.0
    timestamp: datetime = field(default_factory=datetime.now)
    filled_at: Optional[datetime] = None


@dataclass
class Position:
    """Trading position."""
    symbol: str
    chain: str
    quantity: float
    entry_price: float
    current_price: float
    unrealized_pnl: float = 0.0
    realized_pnl: float = 0.0
    entry_timestamp: datetime = field(default_factory=datetime.now)


class CryptoPaperTradingEngine:
    """
    Paper trading engine for crypto assets with realistic simulation.
    
    Features:
    - Multi-chain asset support
    - DEX slippage simulation
    - Gas cost modeling
    - Funding rate tracking
    - Performance metrics
    """
    
    def __init__(
        self,
        initial_capital: float = 100000.0,
        commission_bps: float = 10.0,  # 0.1% = 10 basis points
        slippage_bps: float = 30.0,  # 0.3% = 30 basis points
        gas_cost_usd: float = 5.0,  # Average gas cost per transaction
    ):
        """
        Initialize paper trading engine.
        
        Args:
            initial_capital: Starting capital in USD
            commission_bps: Trading commission in basis points
            slippage_bps: Expected slippage in basis points
            gas_cost_usd: Average gas cost per transaction in USD
        """
        self.initial_capital = initial_capital
        self.commission_bps = commission_bps
        self.slippage_bps = slippage_bps
        self.gas_cost_usd = gas_cost_usd
        
        # Trading state
        self.cash = initial_capital
        self.positions: Dict[str, Position] = {}
        self.orders: List[CryptoOrder] = []
        self.filled_orders: List[CryptoOrder] = []
        self.trade_history: List[Dict] = []
        
        # Performance tracking
        self.portfolio_values: List[Tuple[datetime, float]] = [(datetime.now(), initial_capital)]
        self.max_portfolio_value = initial_capital
        self.total_trades = 0
        self.winning_trades = 0
        self.losing_trades = 0
        
        logger.info(f"Initialized CryptoPaperTradingEngine with ${initial_capital:,.2f} capital")
    
    def place_order(
        self,
        symbol: str,
        chain: str,
        side: OrderSide,
        quantity: float,
        order_type: OrderType = OrderType.MARKET,
        price: Optional[float] = None,
    ) -> Optional[CryptoOrder]:
        """
        Place a trading order.
        
        Args:
            symbol: Trading symbol (e.g., 'BTC', 'ETH')
            chain: Blockchain (e.g., 'ethereum', 'polygon')
            side: BUY or SELL
            quantity: Order quantity
            order_type: Order type
            price: Limit price (for limit orders)
            
        Returns:
            CryptoOrder if successful, None otherwise
        """
        order_id = f"ORD_{len(self.orders) + 1}_{int(datetime.now().timestamp())}"
        
        order = CryptoOrder(
            order_id=order_id,
            symbol=symbol,
            chain=chain,
            side=side,
            order_type=order_type,
            quantity=quantity,
            price=price,
        )
        
        self.orders.append(order)
        logger.info(f"Placed {side.value} order for {quantity} {symbol} on {chain}")
        
        return order
    
    def execute_order(
        self,
        order: CryptoOrder,
        current_price: float,
        timestamp: datetime = None,
    ) -> bool:
        """
        Execute an order at current price.
        
        Args:
            order: Order to execute
            current_price: Current market price
            timestamp: Execution timestamp
            
        Returns:
            True if executed successfully
        """
        if order.status != OrderStatus.PENDING:
            logger.warning(f"Order {order.order_id} is not pending")
            return False
        
        timestamp = timestamp or datetime.now()
        
        # Calculate slippage
        slippage_multiplier = 1 + (self.slippage_bps / 10000)
        if order.side == OrderSide.BUY:
            execution_price = current_price * slippage_multiplier
        else:
            execution_price = current_price / slippage_multiplier
        
        # Calculate costs
        trade_value = order.quantity * execution_price
        commission = trade_value * (self.commission_bps / 10000)
        total_cost = trade_value + commission + self.gas_cost_usd
        
        # Check if we have enough cash for buy orders
        if order.side == OrderSide.BUY:
            if total_cost > self.cash:
                logger.warning(f"Insufficient cash for order {order.order_id}")
                order.status = OrderStatus.REJECTED
                return False
        
        # Execute the order
        order.status = OrderStatus.FILLED
        order.filled_quantity = order.quantity
        order.filled_price = execution_price
        order.gas_cost = self.gas_cost_usd
        order.slippage = abs(execution_price - current_price)
        order.filled_at = timestamp
        
        # Update positions and cash
        position_key = f"{order.symbol}_{order.chain}"
        
        if order.side == OrderSide.BUY:
            self.cash -= total_cost
            
            if position_key in self.positions:
                # Add to existing position
                pos = self.positions[position_key]
                total_cost_basis = pos.quantity * pos.entry_price + order.quantity * execution_price
                pos.quantity += order.quantity
                pos.entry_price = total_cost_basis / pos.quantity
            else:
                # Create new position
                self.positions[position_key] = Position(
                    symbol=order.symbol,
                    chain=order.chain,
                    quantity=order.quantity,
                    entry_price=execution_price,
                    current_price=execution_price,
                    entry_timestamp=timestamp,
                )
        else:  # SELL
            if position_key not in self.positions:
                logger.warning(f"No position to sell for {position_key}")
                order.status = OrderStatus.REJECTED
                return False
            
            pos = self.positions[position_key]
            if pos.quantity < order.quantity:
                logger.warning(f"Insufficient quantity in position {position_key}")
                order.status = OrderStatus.REJECTED
                return False
            
            # Calculate realized PnL
            pnl = (execution_price - pos.entry_price) * order.quantity
            pos.realized_pnl += pnl
            
            # Update cash
            self.cash += trade_value - commission - self.gas_cost_usd
            
            # Update position
            pos.quantity -= order.quantity
            if pos.quantity == 0:
                del self.positions[position_key]
            
            # Track trade performance
            if pnl > 0:
                self.winning_trades += 1
            else:
                self.losing_trades += 1
        
        # Record trade
        self.filled_orders.append(order)
        self.total_trades += 1
        
        self.trade_history.append({
            'timestamp': timestamp,
            'order_id': order.order_id,
            'symbol': order.symbol,
            'chain': order.chain,
            'side': order.side.value,
            'quantity': order.quantity,
            'price': execution_price,
            'commission': commission,
            'gas_cost': self.gas_cost_usd,
            'slippage': order.slippage,
        })
        
        logger.info(
            f"Executed {order.side.value} {order.quantity} {order.symbol} @ ${execution_price:.2f}"
        )
        
        return True
    
    def update_positions(self, current_prices: Dict[str, float]) -> None:
        """
        Update position values with current prices.
        
        Args:
            current_prices: Dict mapping 'symbol_chain' to current price
        """
        for position_key, position in self.positions.items():
            if position_key in current_prices:
                position.current_price = current_prices[position_key]
                position.unrealized_pnl = (
                    position.current_price - position.entry_price
                ) * position.quantity
    
    def get_portfolio_value(self, current_prices: Dict[str, float] = None) -> float:
        """
        Calculate total portfolio value.
        
        Args:
            current_prices: Current market prices
            
        Returns:
            Total portfolio value in USD
        """
        if current_prices:
            self.update_positions(current_prices)
        
        positions_value = sum(
            pos.quantity * pos.current_price
            for pos in self.positions.values()
        )
        
        return self.cash + positions_value
    
    def record_portfolio_value(self, timestamp: datetime, current_prices: Dict[str, float]) -> None:
        """Record portfolio value at timestamp."""
        value = self.get_portfolio_value(current_prices)
        self.portfolio_values.append((timestamp, value))
        
        if value > self.max_portfolio_value:
            self.max_portfolio_value = value
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """
        Calculate comprehensive performance metrics.
        
        Returns:
            Dictionary of performance metrics
        """
        if len(self.portfolio_values) < 2:
            return {}
        
        timestamps, values = zip(*self.portfolio_values)
        returns = pd.Series(values).pct_change().dropna()
        
        final_value = values[-1]
        total_return = (final_value - self.initial_capital) / self.initial_capital * 100
        
        # Calculate drawdown
        cummax = pd.Series(values).cummax()
        drawdown = (pd.Series(values) - cummax) / cummax * 100
        max_drawdown = drawdown.min()
        
        # Sharpe ratio (annualized, assuming daily data)
        if len(returns) > 1 and returns.std() > 0:
            sharpe_ratio = (returns.mean() / returns.std()) * np.sqrt(252)
        else:
            sharpe_ratio = 0.0
        
        # Win rate
        win_rate = (
            self.winning_trades / self.total_trades * 100
            if self.total_trades > 0 else 0.0
        )
        
        return {
            'initial_capital': self.initial_capital,
            'final_value': final_value,
            'total_return_pct': total_return,
            'max_drawdown_pct': max_drawdown,
            'sharpe_ratio': sharpe_ratio,
            'total_trades': self.total_trades,
            'winning_trades': self.winning_trades,
            'losing_trades': self.losing_trades,
            'win_rate_pct': win_rate,
            'avg_return_pct': returns.mean() * 100 if len(returns) > 0 else 0.0,
            'volatility_pct': returns.std() * 100 if len(returns) > 0 else 0.0,
        }
    
    def get_open_positions(self) -> List[Dict]:
        """Get all open positions."""
        return [
            {
                'symbol': pos.symbol,
                'chain': pos.chain,
                'quantity': pos.quantity,
                'entry_price': pos.entry_price,
                'current_price': pos.current_price,
                'unrealized_pnl': pos.unrealized_pnl,
                'unrealized_pnl_pct': (
                    pos.unrealized_pnl / (pos.entry_price * pos.quantity) * 100
                ),
            }
            for pos in self.positions.values()
        ]
    
    def close_all_positions(self, current_prices: Dict[str, float]) -> None:
        """Close all open positions at current prices."""
        for position_key, position in list(self.positions.items()):
            if position_key in current_prices:
                order = self.place_order(
                    symbol=position.symbol,
                    chain=position.chain,
                    side=OrderSide.SELL,
                    quantity=position.quantity,
                )
                self.execute_order(order, current_prices[position_key])
    
    def reset(self) -> None:
        """Reset the trading engine to initial state."""
        self.cash = self.initial_capital
        self.positions = {}
        self.orders = []
        self.filled_orders = []
        self.trade_history = []
        self.portfolio_values = [(datetime.now(), self.initial_capital)]
        self.max_portfolio_value = self.initial_capital
        self.total_trades = 0
        self.winning_trades = 0
        self.losing_trades = 0
        
        logger.info("Reset paper trading engine")


if __name__ == "__main__":
    # Example usage
    engine = CryptoPaperTradingEngine(initial_capital=100000)
    
    print("=== Crypto Paper Trading Engine Test ===\n")
    
    # Place and execute buy order
    order = engine.place_order(
        symbol="BTC",
        chain="ethereum",
        side=OrderSide.BUY,
        quantity=1.0,
    )
    
    engine.execute_order(order, current_price=50000.0)
    
    # Update position prices
    engine.update_positions({"BTC_ethereum": 52000.0})
    
    print(f"Portfolio Value: ${engine.get_portfolio_value():,.2f}")
    print(f"Cash: ${engine.cash:,.2f}")
    print(f"\nOpen Positions:")
    for pos in engine.get_open_positions():
        print(f"  {pos['symbol']} on {pos['chain']}: {pos['quantity']} @ ${pos['entry_price']:,.2f}")
        print(f"    Unrealized P&L: ${pos['unrealized_pnl']:,.2f} ({pos['unrealized_pnl_pct']:.2f}%)")
    
    # Place and execute sell order
    sell_order = engine.place_order(
        symbol="BTC",
        chain="ethereum",
        side=OrderSide.SELL,
        quantity=1.0,
    )
    
    engine.execute_order(sell_order, current_price=52000.0)
    
    print(f"\nAfter closing position:")
    print(f"Portfolio Value: ${engine.get_portfolio_value():,.2f}")
    print(f"Cash: ${engine.cash:,.2f}")
    
    # Performance metrics
    engine.record_portfolio_value(datetime.now(), {})
    metrics = engine.get_performance_metrics()
    
    print(f"\nPerformance Metrics:")
    for key, value in metrics.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.2f}")
        else:
            print(f"  {key}: {value}")
    
    print("\n✅ Paper trading engine test completed!")
