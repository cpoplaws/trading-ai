"""
Paper Trading Analytics
Track trade history and calculate performance metrics.
"""
import logging
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
import json
import csv

from .engine import Order, OrderSide, OrderStatus

logger = logging.getLogger(__name__)


@dataclass
class TradeRecord:
    """Completed trade record."""
    trade_id: str
    timestamp: datetime
    symbol: str
    side: str  # "buy" or "sell"
    quantity: float
    price: float
    total_cost: float
    fees: float
    gas_cost: float
    exchange: str

    # P&L (calculated later for round-trip trades)
    pnl: Optional[float] = None
    pnl_percent: Optional[float] = None
    holding_period: Optional[timedelta] = None


class TradeHistory:
    """
    Manages trade history and analytics.

    Tracks all executed orders and provides analysis tools.
    """

    def __init__(self):
        """Initialize trade history."""
        self.trades: List[TradeRecord] = []
        self.open_positions: Dict[str, List[TradeRecord]] = {}  # symbol -> [buy trades]

        logger.info("Trade history initialized")

    def add_order(self, order: Order):
        """
        Add an executed order to history.

        Args:
            order: Filled order
        """
        if order.status != OrderStatus.FILLED:
            return

        # Create trade record
        trade = TradeRecord(
            trade_id=order.order_id,
            timestamp=order.filled_at or order.created_at,
            symbol=order.symbol,
            side=order.side.value,
            quantity=order.filled_quantity,
            price=order.avg_fill_price,
            total_cost=order.total_cost,
            fees=order.fees,
            gas_cost=order.gas_cost,
            exchange=order.exchange.value
        )

        self.trades.append(trade)

        # Track open positions for P&L calculation
        token = order.symbol.split('-')[0]

        if order.side == OrderSide.BUY:
            # Add to open positions
            if token not in self.open_positions:
                self.open_positions[token] = []
            self.open_positions[token].append(trade)

        elif order.side == OrderSide.SELL:
            # Match with open positions (FIFO)
            self._calculate_realized_pnl(token, trade)

        logger.debug(f"Added trade to history: {trade.trade_id}")

    def _calculate_realized_pnl(self, token: str, sell_trade: TradeRecord):
        """
        Calculate realized P&L for a sell by matching with buy trades.

        Args:
            token: Token symbol
            sell_trade: Sell trade record
        """
        if token not in self.open_positions or not self.open_positions[token]:
            logger.warning(f"No open positions for {token} to calculate P&L")
            return

        remaining_quantity = sell_trade.quantity
        total_cost_basis = 0.0
        buy_trades_used = []

        # Match with buy trades (FIFO)
        for buy_trade in self.open_positions[token]:
            if remaining_quantity <= 0:
                break

            # How much from this buy trade?
            quantity_to_match = min(remaining_quantity, buy_trade.quantity)

            # Calculate cost basis
            cost_basis = quantity_to_match * buy_trade.price
            total_cost_basis += cost_basis

            # Track which buys were used
            buy_trades_used.append((buy_trade, quantity_to_match))

            remaining_quantity -= quantity_to_match

        # Calculate P&L
        sell_proceeds = sell_trade.quantity * sell_trade.price
        sell_fees = sell_trade.fees + sell_trade.gas_cost

        # Buy fees proportional to quantity used
        buy_fees = sum(
            (qty / bt.quantity) * (bt.fees + bt.gas_cost)
            for bt, qty in buy_trades_used
        )

        realized_pnl = sell_proceeds - total_cost_basis - sell_fees - buy_fees
        realized_pnl_pct = (realized_pnl / total_cost_basis) * 100 if total_cost_basis > 0 else 0

        # Calculate holding period
        if buy_trades_used:
            first_buy = buy_trades_used[0][0]
            holding_period = sell_trade.timestamp - first_buy.timestamp
            sell_trade.holding_period = holding_period

        # Update sell trade with P&L
        sell_trade.pnl = realized_pnl
        sell_trade.pnl_percent = realized_pnl_pct

        # Remove matched quantities from open positions
        new_positions = []
        for buy_trade in self.open_positions[token]:
            matched_qty = sum(qty for bt, qty in buy_trades_used if bt.trade_id == buy_trade.trade_id)

            if matched_qty < buy_trade.quantity:
                # Partial match - keep remaining quantity
                buy_trade.quantity -= matched_qty
                new_positions.append(buy_trade)
            # else: fully matched, don't keep

        self.open_positions[token] = new_positions

        logger.info(f"Realized P&L: ${realized_pnl:.2f} ({realized_pnl_pct:.2f}%) "
                   f"from {sell_trade.quantity} {token}")

    def get_all_trades(self, symbol: Optional[str] = None) -> List[TradeRecord]:
        """
        Get all trade records.

        Args:
            symbol: Filter by symbol (optional)

        Returns:
            List of trades
        """
        if symbol:
            return [t for t in self.trades if t.symbol == symbol]
        return self.trades

    def get_round_trip_trades(self) -> List[TradeRecord]:
        """
        Get completed round-trip trades (sells with P&L).

        Returns:
            List of completed trades with P&L
        """
        return [t for t in self.trades if t.side == "sell" and t.pnl is not None]

    def get_performance_metrics(self) -> Dict:
        """
        Calculate comprehensive performance metrics.

        Returns:
            Performance metrics dictionary
        """
        completed_trades = self.get_round_trip_trades()

        if not completed_trades:
            return {
                'total_trades': 0,
                'winning_trades': 0,
                'losing_trades': 0,
                'win_rate': 0.0,
                'total_pnl': 0.0,
                'avg_win': 0.0,
                'avg_loss': 0.0,
                'profit_factor': 0.0,
                'largest_win': 0.0,
                'largest_loss': 0.0,
                'avg_holding_period': None
            }

        # Separate wins and losses
        wins = [t for t in completed_trades if t.pnl > 0]
        losses = [t for t in completed_trades if t.pnl <= 0]

        total_trades = len(completed_trades)
        winning_trades = len(wins)
        losing_trades = len(losses)
        win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0

        # P&L metrics
        total_pnl = sum(t.pnl for t in completed_trades)
        total_wins = sum(t.pnl for t in wins) if wins else 0
        total_losses = abs(sum(t.pnl for t in losses)) if losses else 0

        avg_win = total_wins / len(wins) if wins else 0
        avg_loss = total_losses / len(losses) if losses else 0

        profit_factor = total_wins / total_losses if total_losses > 0 else float('inf')

        largest_win = max((t.pnl for t in wins), default=0)
        largest_loss = min((t.pnl for t in losses), default=0)

        # Holding period
        trades_with_holding = [t for t in completed_trades if t.holding_period]
        avg_holding = None
        if trades_with_holding:
            total_seconds = sum(t.holding_period.total_seconds() for t in trades_with_holding)
            avg_holding = timedelta(seconds=total_seconds / len(trades_with_holding))

        return {
            'total_trades': total_trades,
            'winning_trades': winning_trades,
            'losing_trades': losing_trades,
            'win_rate': win_rate,
            'total_pnl': total_pnl,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'profit_factor': profit_factor,
            'largest_win': largest_win,
            'largest_loss': largest_loss,
            'avg_holding_period': avg_holding
        }

    def get_summary(self) -> str:
        """
        Get formatted performance summary.

        Returns:
            Formatted summary string
        """
        metrics = self.get_performance_metrics()

        summary = f"""
╔══════════════════════════════════════════════════════════════╗
║                  TRADE HISTORY & ANALYTICS                    ║
╠══════════════════════════════════════════════════════════════╣
║ Total Trades:     {metrics['total_trades']:>15}                    ║
║ Winning Trades:   {metrics['winning_trades']:>15}                    ║
║ Losing Trades:    {metrics['losing_trades']:>15}                    ║
║ Win Rate:         {metrics['win_rate']:>14.1f}%                     ║
║                                                               ║
║ Total P&L:        ${metrics['total_pnl']:>12,.2f}                    ║
║ Average Win:      ${metrics['avg_win']:>12,.2f}                    ║
║ Average Loss:     ${metrics['avg_loss']:>12,.2f}                    ║
║ Profit Factor:    {metrics['profit_factor']:>15.2f}                    ║
║                                                               ║
║ Largest Win:      ${metrics['largest_win']:>12,.2f}                    ║
║ Largest Loss:     ${metrics['largest_loss']:>12,.2f}                    ║
"""

        if metrics['avg_holding_period']:
            hours = metrics['avg_holding_period'].total_seconds() / 3600
            summary += f"║ Avg Hold Time:    {hours:>14.1f}h                     ║\n"

        summary += "╚══════════════════════════════════════════════════════════════╝\n"

        return summary

    def export_to_csv(self, filepath: str):
        """
        Export trade history to CSV.

        Args:
            filepath: Output CSV file path
        """
        if not self.trades:
            logger.warning("No trades to export")
            return

        with open(filepath, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=[
                'trade_id', 'timestamp', 'symbol', 'side', 'quantity',
                'price', 'total_cost', 'fees', 'gas_cost', 'exchange',
                'pnl', 'pnl_percent', 'holding_period'
            ])

            writer.writeheader()

            for trade in self.trades:
                row = asdict(trade)
                # Format timestamp and holding_period
                row['timestamp'] = trade.timestamp.isoformat()
                row['holding_period'] = str(trade.holding_period) if trade.holding_period else ""
                writer.writerow(row)

        logger.info(f"Exported {len(self.trades)} trades to {filepath}")

    def export_to_json(self, filepath: str):
        """
        Export trade history to JSON.

        Args:
            filepath: Output JSON file path
        """
        if not self.trades:
            logger.warning("No trades to export")
            return

        data = []
        for trade in self.trades:
            trade_dict = asdict(trade)
            # Convert datetime to ISO format
            trade_dict['timestamp'] = trade.timestamp.isoformat()
            trade_dict['holding_period'] = str(trade.holding_period) if trade.holding_period else None
            data.append(trade_dict)

        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)

        logger.info(f"Exported {len(self.trades)} trades to {filepath}")


if __name__ == '__main__':
    import logging
    import sys
    import os

    # Add parent directory to path
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

    from src.paper_trading.engine import PaperTradingEngine, Exchange, OrderSide

    logging.basicConfig(level=logging.INFO)

    print("📊 Trade History & Analytics Demo")
    print("=" * 60)

    # Initialize
    engine = PaperTradingEngine()
    history = TradeHistory()

    # Execute some trades
    print("\n1. Executing sample trades...")
    print("-" * 60)

    # Buy ETH
    order1 = engine.execute_market_order(
        exchange=Exchange.COINBASE,
        symbol="ETH-USD",
        side=OrderSide.BUY,
        quantity=2.0,
        current_price=2000.0
    )
    history.add_order(order1)
    print(f"Buy: 2 ETH @ ${order1.avg_fill_price:.2f}")

    # Sell 1 ETH at profit
    order2 = engine.execute_market_order(
        exchange=Exchange.COINBASE,
        symbol="ETH-USD",
        side=OrderSide.SELL,
        quantity=1.0,
        current_price=2100.0
    )
    history.add_order(order2)
    print(f"Sell: 1 ETH @ ${order2.avg_fill_price:.2f}")

    # Sell remaining at loss
    order3 = engine.execute_market_order(
        exchange=Exchange.UNISWAP,
        symbol="ETH-USD",
        side=OrderSide.SELL,
        quantity=1.0,
        current_price=1950.0
    )
    history.add_order(order3)
    print(f"Sell: 1 ETH @ ${order3.avg_fill_price:.2f}")

    # More trades
    order4 = engine.execute_market_order(
        exchange=Exchange.COINBASE,
        symbol="ETH-USD",
        side=OrderSide.BUY,
        quantity=1.5,
        current_price=2050.0
    )
    history.add_order(order4)
    print(f"Buy: 1.5 ETH @ ${order4.avg_fill_price:.2f}")

    order5 = engine.execute_market_order(
        exchange=Exchange.COINBASE,
        symbol="ETH-USD",
        side=OrderSide.SELL,
        quantity=1.5,
        current_price=2150.0
    )
    history.add_order(order5)
    print(f"Sell: 1.5 ETH @ ${order5.avg_fill_price:.2f}")

    # Display analytics
    print("\n2. Trade History Analytics")
    print("-" * 60)

    print(history.get_summary())

    # Show individual completed trades
    print("\n3. Completed Round-Trip Trades")
    print("-" * 60)

    completed = history.get_round_trip_trades()
    for trade in completed:
        pnl_sign = "+" if trade.pnl > 0 else ""
        print(f"{trade.trade_id}: {trade.quantity} {trade.symbol.split('-')[0]} "
              f"@ ${trade.price:.2f} | "
              f"P&L: {pnl_sign}${trade.pnl:.2f} ({pnl_sign}{trade.pnl_percent:.2f}%)")

    # Export
    print("\n4. Exporting trade history...")
    print("-" * 60)

    history.export_to_csv("/tmp/trade_history.csv")
    history.export_to_json("/tmp/trade_history.json")

    print("Exported to:")
    print("  - /tmp/trade_history.csv")
    print("  - /tmp/trade_history.json")

    print("\n✅ Trade analytics demo complete!")
