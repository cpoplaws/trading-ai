"""
Paper Trading Portfolio Manager
Tracks balances, P&L, and performance metrics.
"""
import logging
from typing import Dict, List, Optional
from datetime import datetime
from dataclasses import dataclass, field

from .engine import Order, OrderSide, OrderStatus

logger = logging.getLogger(__name__)


@dataclass
class Balance:
    """Token balance."""
    symbol: str
    amount: float
    locked: float = 0.0  # Locked in pending orders

    @property
    def available(self) -> float:
        """Available balance."""
        return max(0, self.amount - self.locked)


@dataclass
class Trade:
    """Completed trade."""
    trade_id: str
    timestamp: datetime
    symbol: str
    side: str
    quantity: float
    entry_price: float
    exit_price: Optional[float] = None
    pnl: float = 0.0
    pnl_percent: float = 0.0
    fees: float = 0.0
    gas_cost: float = 0.0


class PaperPortfolio:
    """
    Paper trading portfolio manager.

    Tracks:
    - Token balances
    - Position P&L
    - Trade history
    - Performance metrics
    """

    def __init__(self, initial_usd: float = 10000.0):
        """
        Initialize portfolio.

        Args:
            initial_usd: Starting capital in USD
        """
        self.initial_value = initial_usd
        self.balances: Dict[str, Balance] = {
            'USD': Balance('USD', initial_usd)
        }
        self.trades: List[Trade] = []
        self.orders: List[Order] = []

        # Performance metrics
        self.total_trades = 0
        self.winning_trades = 0
        self.total_fees = 0.0
        self.total_gas = 0.0
        self.peak_value = initial_usd
        self.max_drawdown = 0.0

        logger.info(f"Portfolio initialized with ${initial_usd:,.2f}")

    def get_balance(self, symbol: str) -> float:
        """Get available balance for a symbol."""
        return self.balances.get(symbol, Balance(symbol, 0.0)).available

    def get_total_balance(self, symbol: str) -> float:
        """Get total balance including locked."""
        return self.balances.get(symbol, Balance(symbol, 0.0)).amount

    def process_order(self, order: Order, token_price_usd: float = 1.0):
        """
        Process a filled order and update balances.

        Args:
            order: Filled order
            token_price_usd: Token price in USD (for non-USD tokens)
        """
        if order.status != OrderStatus.FILLED:
            return

        # Extract token symbol
        token = order.symbol.split('-')[0]  # ETH from ETH-USD

        # Ensure balances exist
        if token not in self.balances:
            self.balances[token] = Balance(token, 0.0)
        if 'USD' not in self.balances:
            self.balances['USD'] = Balance('USD', 0.0)

        if order.side == OrderSide.BUY:
            # Deduct USD
            self.balances['USD'].amount -= order.total_cost

            # Add token
            self.balances[token].amount += order.filled_quantity

            logger.info(f"Buy processed: {order.filled_quantity} {token} for ${order.total_cost:.2f}")

        else:  # SELL
            # Deduct token
            self.balances[token].amount -= order.filled_quantity

            # Add USD
            self.balances['USD'].amount += order.total_cost

            logger.info(f"Sell processed: {order.filled_quantity} {token} for ${order.total_cost:.2f}")

        # Track order
        self.orders.append(order)
        self.total_fees += order.fees
        self.total_gas += order.gas_cost

    def get_portfolio_value(self, token_prices: Dict[str, float]) -> float:
        """
        Calculate total portfolio value in USD.

        Args:
            token_prices: Current token prices {symbol: price_usd}

        Returns:
            Total value in USD
        """
        total = 0.0

        for symbol, balance in self.balances.items():
            if symbol == 'USD':
                total += balance.amount
            else:
                price = token_prices.get(symbol, 0.0)
                total += balance.amount * price

        return total

    def get_pnl(self, token_prices: Dict[str, float]) -> Dict:
        """
        Get P&L statistics.

        Args:
            token_prices: Current token prices

        Returns:
            Dict with P&L metrics
        """
        current_value = self.get_portfolio_value(token_prices)
        total_pnl = current_value - self.initial_value
        total_pnl_pct = (total_pnl / self.initial_value) * 100

        # Update peak and drawdown
        if current_value > self.peak_value:
            self.peak_value = current_value

        current_drawdown = ((self.peak_value - current_value) / self.peak_value) * 100
        if current_drawdown > self.max_drawdown:
            self.max_drawdown = current_drawdown

        # Win rate
        win_rate = (self.winning_trades / self.total_trades * 100) if self.total_trades > 0 else 0

        return {
            'initial_value': self.initial_value,
            'current_value': current_value,
            'total_pnl': total_pnl,
            'total_pnl_percent': total_pnl_pct,
            'total_fees': self.total_fees,
            'total_gas': self.total_gas,
            'net_pnl': total_pnl - self.total_fees - self.total_gas,
            'total_trades': self.total_trades,
            'winning_trades': self.winning_trades,
            'win_rate': win_rate,
            'peak_value': self.peak_value,
            'max_drawdown': self.max_drawdown
        }

    def get_summary(self, token_prices: Dict[str, float]) -> str:
        """Get portfolio summary string."""
        pnl = self.get_pnl(token_prices)

        summary = f"""
╔══════════════════════════════════════════════════════════════╗
║               PAPER TRADING PORTFOLIO SUMMARY                 ║
╠══════════════════════════════════════════════════════════════╣
║ Initial Value:    ${pnl['initial_value']:>12,.2f}              ║
║ Current Value:    ${pnl['current_value']:>12,.2f}              ║
║ Total P&L:        ${pnl['total_pnl']:>12,.2f} ({pnl['total_pnl_percent']:>6.2f}%) ║
║                                                               ║
║ Total Fees:       ${pnl['total_fees']:>12,.2f}                 ║
║ Total Gas:        ${pnl['total_gas']:>12,.2f}                  ║
║ Net P&L:          ${pnl['net_pnl']:>12,.2f}                    ║
║                                                               ║
║ Total Trades:     {pnl['total_trades']:>15}                    ║
║ Win Rate:         {pnl['win_rate']:>14.1f}%                     ║
║ Max Drawdown:     {pnl['max_drawdown']:>14.2f}%                     ║
╚══════════════════════════════════════════════════════════════╝

Balances:
"""
        for symbol, balance in self.balances.items():
            if balance.amount > 0:
                if symbol == 'USD':
                    summary += f"  {symbol}: ${balance.amount:,.2f}\n"
                else:
                    price = token_prices.get(symbol, 0)
                    value = balance.amount * price
                    summary += f"  {symbol}: {balance.amount:.4f} (${value:,.2f})\n"

        return summary


if __name__ == '__main__':
    import logging
    import sys
    import os

    # Add parent directory to path for imports
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

    from src.paper_trading.engine import PaperTradingEngine, Exchange

    logging.basicConfig(level=logging.INFO)

    print("💼 Paper Trading Portfolio Demo")
    print("=" * 60)

    # Initialize
    portfolio = PaperPortfolio(initial_usd=10000.0)
    engine = PaperTradingEngine()

    print(f"\nStarting balance: ${portfolio.get_balance('USD'):,.2f}")

    # Execute buy
    print("\n1. Buy 2 ETH @ $2,150")
    order1 = engine.execute_market_order(
        exchange=Exchange.COINBASE,
        symbol="ETH-USD",
        side=OrderSide.BUY,
        quantity=2.0,
        current_price=2150.0
    )
    portfolio.process_order(order1)

    print(f"USD Balance: ${portfolio.get_balance('USD'):,.2f}")
    print(f"ETH Balance: {portfolio.get_balance('ETH'):.4f}")

    # Execute sell
    print("\n2. Sell 1 ETH @ $2,175")
    order2 = engine.execute_market_order(
        exchange=Exchange.UNISWAP,
        symbol="ETH-USD",
        side=OrderSide.SELL,
        quantity=1.0,
        current_price=2175.0
    )
    portfolio.process_order(order2)

    print(f"USD Balance: ${portfolio.get_balance('USD'):,.2f}")
    print(f"ETH Balance: {portfolio.get_balance('ETH'):.4f}")

    # Portfolio summary
    print("\n" + "=" * 60)
    print("PORTFOLIO SUMMARY")
    print("=" * 60)

    prices = {'ETH': 2180.0}
    print(portfolio.get_summary(prices))

    print("\n✅ Portfolio demo complete!")
