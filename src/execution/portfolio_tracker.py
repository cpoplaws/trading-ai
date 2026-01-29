"""
Real-time portfolio tracking and risk management.
Monitors PnL, drawdown, exposure, and risk metrics.
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import logging
from collections import defaultdict

logger = logging.getLogger(__name__)

class PortfolioTracker:
    """
    Track portfolio performance, risk metrics, and exposure in real-time.
    """
    
    def __init__(self, broker=None, initial_capital: float = 100000.0):
        """
        Initialize portfolio tracker.
        
        Args:
            broker: Broker instance (optional)
            initial_capital: Starting portfolio value
        """
        self.broker = broker
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.positions = {}  # symbol -> position details
        self.trade_history = []
        self.equity_curve = []
        self.max_equity = initial_capital
        self.start_time = datetime.now()

    def get_portfolio_summary(self, current_prices: Dict[str, float]) -> Dict:
        """
        Return a lightweight portfolio summary (Phase 2 compatibility).
        """
        # Fetch positions from broker if provided
        positions_source = {}
        cash = self.current_capital

        if self.broker is not None:
            try:
                account_info = self.broker.get_account_info()
                if isinstance(account_info, dict):
                    cash = account_info.get("cash", cash)
                else:
                    cash = getattr(account_info, "cash", cash)
            except Exception:
                pass

            if hasattr(self.broker, "get_positions"):
                try:
                    positions_source = self.broker.get_positions()
                except Exception:
                    positions_source = {}

        if not positions_source:
            positions_source = self.positions

        position_list = []
        positions_value = 0.0

        for symbol, pos in positions_source.items():
            qty = pos.get("quantity") if isinstance(pos, dict) else getattr(pos, "qty", 0)
            avg_price = (
                pos.get("avg_price") if isinstance(pos, dict) else getattr(pos, "avg_price", 0)
            )
            qty = float(qty or 0)
            avg_price = float(avg_price or 0)
            current_price = float(current_prices.get(symbol, avg_price))
            market_value = qty * current_price
            unrealized_pnl = (current_price - avg_price) * qty
            positions_value += market_value
            denominator = avg_price * qty

            position_list.append(
                {
                    "symbol": symbol,
                    "quantity": qty,
                    "avg_price": avg_price,
                    "current_price": current_price,
                    "market_value": market_value,
                    "unrealized_pnl": unrealized_pnl,
                    "unrealized_pnl_pct": (unrealized_pnl / denominator * 100)
                    if denominator != 0
                    else 0.0,
                    "exposure": 0.0,  # filled later
                }
            )

        total_value = float(cash) + positions_value
        for pos in position_list:
            pos["exposure"] = pos["market_value"] / total_value if total_value else 0.0

        summary = {
            "positions": position_list,
            "total_value": total_value,
            "cash": float(cash),
        }
        return summary
        
    def update_position(self, symbol: str, qty: int, price: float, 
                       current_price: float = None) -> Dict:
        """
        Update or create a position.
        
        Args:
            symbol: Stock symbol
            qty: Quantity (positive for long, negative for short)
            price: Entry/average price
            current_price: Current market price
            
        Returns:
            Updated position details
        """
        if current_price is None:
            current_price = price
        
        if symbol in self.positions:
            # Update existing position
            old_pos = self.positions[symbol]
            new_qty = old_pos['qty'] + qty
            
            if new_qty == 0:
                # Position closed
                del self.positions[symbol]
                return {'symbol': symbol, 'qty': 0, 'status': 'closed'}
            else:
                # Update average price
                total_cost = (old_pos['qty'] * old_pos['avg_price']) + (qty * price)
                avg_price = total_cost / new_qty
                
                self.positions[symbol] = {
                    'symbol': symbol,
                    'qty': new_qty,
                    'avg_price': avg_price,
                    'current_price': current_price,
                    'entry_time': old_pos['entry_time'],
                    'last_update': datetime.now()
                }
        else:
            # New position
            self.positions[symbol] = {
                'symbol': symbol,
                'qty': qty,
                'avg_price': price,
                'current_price': current_price,
                'entry_time': datetime.now(),
                'last_update': datetime.now()
            }
        
        return self.positions[symbol]
    
    def record_trade(self, symbol: str, side: str, qty: int, price: float,
                    commission: float = 0.0, notes: str = "") -> None:
        """
        Record a trade in history.
        
        Args:
            symbol: Stock symbol
            side: 'BUY' or 'SELL'
            qty: Quantity traded
            price: Execution price
            commission: Trading commission
            notes: Additional notes
        """
        trade = {
            'timestamp': datetime.now(),
            'symbol': symbol,
            'side': side,
            'qty': qty,
            'price': price,
            'value': qty * price,
            'commission': commission,
            'notes': notes
        }
        
        self.trade_history.append(trade)
        
        # Update capital
        if side == 'BUY':
            self.current_capital -= (qty * price + commission)
        else:  # SELL
            self.current_capital += (qty * price - commission)
    
    def update_equity(self, timestamp: datetime = None) -> float:
        """
        Calculate and record current equity.
        
        Args:
            timestamp: Time of equity snapshot
            
        Returns:
            Current total equity
        """
        if timestamp is None:
            timestamp = datetime.now()
        
        # Calculate total position value
        position_value = sum(
            pos['qty'] * pos['current_price']
            for pos in self.positions.values()
        )
        
        total_equity = self.current_capital + position_value
        
        # Record equity point
        self.equity_curve.append({
            'timestamp': timestamp,
            'equity': total_equity,
            'cash': self.current_capital,
            'positions_value': position_value
        })
        
        # Update max equity for drawdown calculation
        if total_equity > self.max_equity:
            self.max_equity = total_equity
        
        return total_equity
    
    def get_current_pnl(self) -> Dict:
        """
        Calculate current profit/loss.
        
        Returns:
            Dictionary with PnL metrics
        """
        current_equity = self.update_equity()
        
        total_pnl = current_equity - self.initial_capital
        total_pnl_pct = (total_pnl / self.initial_capital) * 100
        
        # Calculate realized and unrealized PnL
        unrealized_pnl = 0
        for pos in self.positions.values():
            unrealized_pnl += pos['qty'] * (pos['current_price'] - pos['avg_price'])
        
        realized_pnl = total_pnl - unrealized_pnl
        
        return {
            'total_pnl': total_pnl,
            'total_pnl_pct': total_pnl_pct,
            'realized_pnl': realized_pnl,
            'unrealized_pnl': unrealized_pnl,
            'current_equity': current_equity,
            'initial_capital': self.initial_capital,
            'timestamp': datetime.now()
        }
    
    def get_drawdown(self) -> Dict:
        """
        Calculate current and maximum drawdown.
        
        Returns:
            Dictionary with drawdown metrics
        """
        current_equity = self.update_equity()
        
        current_drawdown = self.max_equity - current_equity
        current_drawdown_pct = (current_drawdown / self.max_equity) * 100 if self.max_equity > 0 else 0
        
        # Calculate max drawdown from equity curve
        if len(self.equity_curve) < 2:
            max_drawdown = 0
            max_drawdown_pct = 0
        else:
            equity_series = pd.Series([e['equity'] for e in self.equity_curve])
            running_max = equity_series.expanding().max()
            drawdown_series = (equity_series - running_max) / running_max * 100
            max_drawdown_pct = abs(drawdown_series.min())
            max_drawdown = abs((equity_series - running_max).min())
        
        return {
            'current_drawdown': current_drawdown,
            'current_drawdown_pct': current_drawdown_pct,
            'max_drawdown': max_drawdown,
            'max_drawdown_pct': max_drawdown_pct,
            'max_equity': self.max_equity,
            'timestamp': datetime.now()
        }
    
    def get_position_exposure(self) -> Dict:
        """
        Calculate portfolio exposure and concentration.
        
        Returns:
            Dictionary with exposure metrics
        """
        current_equity = self.update_equity()
        
        if current_equity == 0:
            return {
                'total_exposure': 0,
                'long_exposure': 0,
                'short_exposure': 0,
                'net_exposure': 0,
                'positions': {}
            }
        
        long_exposure = 0
        short_exposure = 0
        position_exposures = {}
        
        for symbol, pos in self.positions.items():
            position_value = pos['qty'] * pos['current_price']
            exposure_pct = (abs(position_value) / current_equity) * 100
            
            if pos['qty'] > 0:
                long_exposure += abs(position_value)
            else:
                short_exposure += abs(position_value)
            
            position_exposures[symbol] = {
                'value': position_value,
                'exposure_pct': exposure_pct,
                'qty': pos['qty'],
                'side': 'LONG' if pos['qty'] > 0 else 'SHORT'
            }
        
        total_exposure = long_exposure + short_exposure
        net_exposure = long_exposure - short_exposure
        
        return {
            'total_exposure': total_exposure,
            'total_exposure_pct': (total_exposure / current_equity) * 100,
            'long_exposure': long_exposure,
            'long_exposure_pct': (long_exposure / current_equity) * 100,
            'short_exposure': short_exposure,
            'short_exposure_pct': (short_exposure / current_equity) * 100,
            'net_exposure': net_exposure,
            'net_exposure_pct': (net_exposure / current_equity) * 100,
            'positions': position_exposures,
            'timestamp': datetime.now()
        }
    
    def get_performance_metrics(self) -> Dict:
        """
        Calculate comprehensive performance metrics.
        
        Returns:
            Dictionary with performance metrics
        """
        if len(self.equity_curve) < 2:
            return {'error': 'Insufficient data for metrics'}
        
        # Convert equity curve to DataFrame
        df = pd.DataFrame(self.equity_curve)
        df['returns'] = df['equity'].pct_change()
        
        # Calculate metrics
        total_return = ((df['equity'].iloc[-1] / self.initial_capital) - 1) * 100
        
        # Annualize based on time elapsed
        days_elapsed = (datetime.now() - self.start_time).days
        if days_elapsed > 0:
            annual_return = total_return * (365 / days_elapsed)
        else:
            annual_return = 0
        
        # Risk metrics
        volatility = df['returns'].std() * np.sqrt(252) * 100  # Annualized
        
        if volatility > 0:
            sharpe_ratio = (annual_return - 2) / volatility  # Assuming 2% risk-free rate
        else:
            sharpe_ratio = 0
        
        # Win rate from trades
        if self.trade_history:
            # Group trades by symbol to calculate PnL
            trade_pnls = []
            symbol_trades = defaultdict(list)
            
            for trade in self.trade_history:
                symbol_trades[trade['symbol']].append(trade)
            
            for symbol, trades in symbol_trades.items():
                # Simple PnL calculation
                buy_value = sum(t['value'] for t in trades if t['side'] == 'BUY')
                sell_value = sum(t['value'] for t in trades if t['side'] == 'SELL')
                if buy_value > 0 and sell_value > 0:
                    pnl = sell_value - buy_value
                    trade_pnls.append(pnl)
            
            if trade_pnls:
                win_rate = (sum(1 for pnl in trade_pnls if pnl > 0) / len(trade_pnls)) * 100
                avg_win = np.mean([pnl for pnl in trade_pnls if pnl > 0]) if any(pnl > 0 for pnl in trade_pnls) else 0
                avg_loss = abs(np.mean([pnl for pnl in trade_pnls if pnl < 0])) if any(pnl < 0 for pnl in trade_pnls) else 0
                profit_factor = sum([pnl for pnl in trade_pnls if pnl > 0]) / abs(sum([pnl for pnl in trade_pnls if pnl < 0])) if any(pnl < 0 for pnl in trade_pnls) else float('inf')
            else:
                win_rate = 0
                avg_win = 0
                avg_loss = 0
                profit_factor = 0
        else:
            win_rate = 0
            avg_win = 0
            avg_loss = 0
            profit_factor = 0
        
        pnl = self.get_current_pnl()
        drawdown = self.get_drawdown()
        
        return {
            'total_return_pct': total_return,
            'annual_return_pct': annual_return,
            'volatility_pct': volatility,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown_pct': drawdown['max_drawdown_pct'],
            'current_drawdown_pct': drawdown['current_drawdown_pct'],
            'win_rate_pct': win_rate,
            'profit_factor': profit_factor,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'total_trades': len(self.trade_history),
            'current_positions': len(self.positions),
            'days_trading': days_elapsed,
            'timestamp': datetime.now()
        }
    
    def get_dashboard(self) -> Dict:
        """
        Get comprehensive portfolio dashboard.
        
        Returns:
            Dictionary with all portfolio metrics
        """
        return {
            'pnl': self.get_current_pnl(),
            'drawdown': self.get_drawdown(),
            'exposure': self.get_position_exposure(),
            'performance': self.get_performance_metrics(),
            'positions': self.positions,
            'recent_trades': self.trade_history[-10:] if self.trade_history else [],
            'timestamp': datetime.now()
        }
    
    def get_portfolio_summary(self, current_prices: Dict[str, float]) -> Dict:
        """
        Provide a simplified portfolio summary using broker positions.

        Args:
            current_prices: Mapping from symbol to latest market price; falls back
            to avg_price when missing.

        Returns:
            Dict with positions list (symbol, quantity, avg_price, market_value,
            unrealized_pnl, unrealized_pnl_pct, exposure), total_value, and cash.
        """
        positions_data = []
        account = {}
        if self.broker and hasattr(self.broker, "get_account_info"):
            try:
                account = self.broker.get_account_info() or {}
            except Exception as exc:
                logger.exception("Error getting account info: %s", exc)
                account = {}
        cash = account.get("cash", self.current_capital) if isinstance(account, dict) else getattr(account, "cash", self.current_capital)
        portfolio_value = account.get("portfolio_value", cash) if isinstance(account, dict) else getattr(account, "portfolio_value", cash)

        raw_positions = {}
        if self.broker and hasattr(self.broker, "get_positions"):
            try:
                raw_positions = self.broker.get_positions() or {}
            except Exception as exc:
                logger.exception("Error getting positions: %s", exc)
                raw_positions = {}

        iterable_positions = (
            raw_positions.items()
            if isinstance(raw_positions, dict)
            else enumerate(raw_positions)
            if isinstance(raw_positions, (list, tuple))
            else []
        )

        for symbol, pos in iterable_positions:
            qty = pos.get("quantity", pos.get("qty", 0)) if isinstance(pos, dict) else getattr(pos, "qty", 0)
            avg_price = pos.get("avg_price", 0) if isinstance(pos, dict) else getattr(pos, "avg_entry_price", 0)
            price = current_prices.get(symbol, avg_price)
            market_value = qty * price
            unrealized = (price - avg_price) * qty
            positions_data.append({
                "symbol": symbol if isinstance(symbol, str) else getattr(pos, "symbol", str(symbol)),
                "quantity": qty,
                "avg_price": avg_price,
                "market_value": market_value,
                "unrealized_pnl": unrealized,
                "unrealized_pnl_pct": (unrealized / (avg_price * qty)) * 100 if qty and avg_price else 0,
                "exposure": (market_value / portfolio_value) if portfolio_value else 0,
            })

        total_value = cash + sum(p["market_value"] for p in positions_data)
        return {
            "positions": positions_data,
            "total_value": total_value,
            "cash": cash,
        }
    
    def check_risk_limits(self, max_drawdown_pct: float = 20.0,
                         max_position_pct: float = 30.0,
                         max_total_exposure_pct: float = 100.0) -> Dict:
        """
        Check if portfolio violates risk limits.
        
        Args:
            max_drawdown_pct: Maximum allowed drawdown percentage
            max_position_pct: Maximum position size as % of equity
            max_total_exposure_pct: Maximum total exposure as % of equity
            
        Returns:
            Dictionary with risk limit violations
        """
        violations = []
        
        drawdown = self.get_drawdown()
        if drawdown['current_drawdown_pct'] > max_drawdown_pct:
            violations.append({
                'type': 'MAX_DRAWDOWN',
                'current': drawdown['current_drawdown_pct'],
                'limit': max_drawdown_pct,
                'severity': 'HIGH'
            })
        
        exposure = self.get_position_exposure()
        if exposure['total_exposure_pct'] > max_total_exposure_pct:
            violations.append({
                'type': 'TOTAL_EXPOSURE',
                'current': exposure['total_exposure_pct'],
                'limit': max_total_exposure_pct,
                'severity': 'MEDIUM'
            })
        
        for symbol, pos_exposure in exposure['positions'].items():
            if pos_exposure['exposure_pct'] > max_position_pct:
                violations.append({
                    'type': 'POSITION_SIZE',
                    'symbol': symbol,
                    'current': pos_exposure['exposure_pct'],
                    'limit': max_position_pct,
                    'severity': 'LOW'
                })
        
        return {
            'violations': violations,
            'risk_ok': len(violations) == 0,
            'timestamp': datetime.now()
        }


if __name__ == "__main__":
    # Test the portfolio tracker
    logging.basicConfig(level=logging.INFO)
    
    tracker = PortfolioTracker(initial_capital=100000)
    
    # Simulate some trades
    tracker.update_position('AAPL', 100, 150.0, 155.0)
    tracker.record_trade('AAPL', 'BUY', 100, 150.0)
    
    tracker.update_position('MSFT', 50, 300.0, 310.0)
    tracker.record_trade('MSFT', 'BUY', 50, 300.0)
    
    # Get dashboard
    dashboard = tracker.get_dashboard()
    
    print("\n=== Portfolio Dashboard ===")
    print(f"Total PnL: ${dashboard['pnl']['total_pnl']:.2f} ({dashboard['pnl']['total_pnl_pct']:.2f}%)")
    print(f"Current Drawdown: {dashboard['drawdown']['current_drawdown_pct']:.2f}%")
    print(f"Total Exposure: {dashboard['exposure']['total_exposure_pct']:.2f}%")
    print(f"\nPositions: {len(dashboard['positions'])}")
    for symbol, pos in dashboard['positions'].items():
        print(f"  {symbol}: {pos['qty']} shares @ ${pos['avg_price']:.2f}")
