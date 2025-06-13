"""
Portfolio Tracker for real-time PnL, exposure, and performance monitoring.
Part of Phase 2: Trading System completion.
"""

import os
import json
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import logging
from dataclasses import dataclass, asdict

# Set up logging
logger = logging.getLogger(__name__)

@dataclass
class Position:
    """Represents a trading position."""
    symbol: str
    quantity: int
    avg_price: float
    current_price: float
    market_value: float
    unrealized_pnl: float
    unrealized_pnl_pct: float
    side: str  # 'long' or 'short'
    entry_date: str
    
    def to_dict(self) -> Dict:
        return asdict(self)

@dataclass
class PortfolioMetrics:
    """Portfolio performance metrics."""
    total_value: float
    cash: float
    total_equity: float
    buying_power: float
    day_pnl: float
    day_pnl_pct: float
    total_pnl: float
    total_pnl_pct: float
    exposure: float
    exposure_pct: float
    num_positions: int
    largest_position: str
    largest_position_pct: float
    
    def to_dict(self) -> Dict:
        return asdict(self)

class PortfolioTracker:
    """
    Real-time portfolio tracking with PnL calculation and risk monitoring.
    """
    
    def __init__(self, broker, initial_capital: float = 10000, save_path: str = './logs/'):
        """
        Initialize portfolio tracker.
        
        Args:
            broker: Broker interface instance
            initial_capital: Starting capital amount
            save_path: Directory to save portfolio logs
        """
        self.broker = broker
        self.initial_capital = initial_capital
        self.save_path = save_path
        os.makedirs(save_path, exist_ok=True)
        
        self.portfolio_history = []
        self.trade_history = []
        self.daily_snapshots = []
        
        logger.info(f"Portfolio tracker initialized with ${initial_capital:,.2f}")
    
    def get_current_positions(self) -> List[Position]:
        """
        Get current portfolio positions with real-time pricing.
        
        Returns:
            List of Position objects
        """
        positions = []
        
        try:
            # Get positions from broker
            broker_positions = self.broker.get_positions()
            
            for pos in broker_positions:
                if float(pos['qty']) == 0:
                    continue  # Skip closed positions
                
                symbol = pos['symbol']
                quantity = int(pos['qty'])
                avg_price = float(pos['avg_entry_price'])
                current_price = float(pos['market_value']) / quantity if quantity != 0 else avg_price
                market_value = float(pos['market_value'])
                unrealized_pnl = float(pos['unrealized_pl'])
                unrealized_pnl_pct = (unrealized_pnl / abs(avg_price * quantity)) * 100 if quantity != 0 else 0
                side = 'long' if quantity > 0 else 'short'
                
                position = Position(
                    symbol=symbol,
                    quantity=quantity,
                    avg_price=avg_price,
                    current_price=current_price,
                    market_value=market_value,
                    unrealized_pnl=unrealized_pnl,
                    unrealized_pnl_pct=unrealized_pnl_pct,
                    side=side,
                    entry_date=pos.get('created_at', datetime.now().isoformat())
                )
                
                positions.append(position)
                
        except Exception as e:
            logger.error(f"Error getting positions: {str(e)}")
        
        return positions
    
    def calculate_portfolio_metrics(self) -> PortfolioMetrics:
        """
        Calculate comprehensive portfolio metrics.
        
        Returns:
            PortfolioMetrics object
        """
        try:
            # Get account info
            account = self.broker.get_account_info()
            positions = self.get_current_positions()
            
            # Basic account metrics
            total_value = float(account.get('portfolio_value', 0))
            cash = float(account.get('cash', 0))
            total_equity = float(account.get('equity', total_value))
            buying_power = float(account.get('buying_power', 0))
            
            # PnL calculations
            day_pnl = float(account.get('todays_pl', 0))
            day_pnl_pct = (day_pnl / total_value) * 100 if total_value > 0 else 0
            total_pnl = total_value - self.initial_capital
            total_pnl_pct = (total_pnl / self.initial_capital) * 100
            
            # Position analysis
            num_positions = len(positions)
            total_exposure = sum(abs(pos.market_value) for pos in positions)
            exposure_pct = (total_exposure / total_value) * 100 if total_value > 0 else 0
            
            # Find largest position
            largest_position = ""
            largest_position_pct = 0
            if positions:
                largest = max(positions, key=lambda p: abs(p.market_value))
                largest_position = largest.symbol
                largest_position_pct = (abs(largest.market_value) / total_value) * 100
            
            metrics = PortfolioMetrics(
                total_value=total_value,
                cash=cash,
                total_equity=total_equity,
                buying_power=buying_power,
                day_pnl=day_pnl,
                day_pnl_pct=day_pnl_pct,
                total_pnl=total_pnl,
                total_pnl_pct=total_pnl_pct,
                exposure=total_exposure,
                exposure_pct=exposure_pct,
                num_positions=num_positions,
                largest_position=largest_position,
                largest_position_pct=largest_position_pct
            )
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error calculating portfolio metrics: {str(e)}")
            return PortfolioMetrics(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, "", 0)
    
    def log_trade(self, symbol: str, side: str, quantity: int, price: float, 
                  order_id: str = "", strategy: str = "") -> None:
        """
        Log a completed trade.
        
        Args:
            symbol: Stock symbol
            side: 'buy' or 'sell'
            quantity: Number of shares
            price: Execution price
            order_id: Order ID from broker
            strategy: Strategy that generated the signal
        """
        trade = {
            'timestamp': datetime.now().isoformat(),
            'symbol': symbol,
            'side': side,
            'quantity': quantity,
            'price': price,
            'value': quantity * price,
            'order_id': order_id,
            'strategy': strategy
        }
        
        self.trade_history.append(trade)
        
        # Save to file
        trades_file = os.path.join(self.save_path, 'trades.json')
        with open(trades_file, 'w') as f:
            json.dump(self.trade_history, f, indent=2)
        
        logger.info(f"Trade logged: {side.upper()} {quantity} {symbol} @ ${price:.2f}")
    
    def take_snapshot(self) -> Dict:
        """
        Take a snapshot of current portfolio state.
        
        Returns:
            Dictionary with portfolio snapshot
        """
        positions = self.get_current_positions()
        metrics = self.calculate_portfolio_metrics()
        
        snapshot = {
            'timestamp': datetime.now().isoformat(),
            'metrics': metrics.to_dict(),
            'positions': [pos.to_dict() for pos in positions]
        }
        
        self.portfolio_history.append(snapshot)
        
        # Save daily snapshot
        today = datetime.now().strftime('%Y-%m-%d')
        snapshot_file = os.path.join(self.save_path, f'portfolio_snapshot_{today}.json')
        with open(snapshot_file, 'w') as f:
            json.dump(snapshot, f, indent=2)
        
        return snapshot
    
    def get_performance_summary(self, days: int = 30) -> Dict:
        """
        Get performance summary for the last N days.
        
        Args:
            days: Number of days to analyze
            
        Returns:
            Performance summary dictionary
        """
        try:
            metrics = self.calculate_portfolio_metrics()
            
            # Calculate some basic performance metrics
            # In a real implementation, we'd use historical snapshots
            summary = {
                'current_value': metrics.total_value,
                'total_return': metrics.total_pnl,
                'total_return_pct': metrics.total_pnl_pct,
                'day_pnl': metrics.day_pnl,
                'day_pnl_pct': metrics.day_pnl_pct,
                'cash_position': metrics.cash,
                'exposure': metrics.exposure_pct,
                'num_positions': metrics.num_positions,
                'largest_position': f"{metrics.largest_position} ({metrics.largest_position_pct:.1f}%)",
                'analysis_period': f"{days} days"
            }
            
            return summary
            
        except Exception as e:
            logger.error(f"Error generating performance summary: {str(e)}")
            return {}
    
    def print_portfolio_status(self) -> None:
        """
        Print formatted portfolio status to console.
        """
        try:
            metrics = self.calculate_portfolio_metrics()
            positions = self.get_current_positions()
            
            print("\n" + "="*60)
            print("📊 PORTFOLIO STATUS")
            print("="*60)
            print(f"Total Value:    ${metrics.total_value:>10,.2f}")
            print(f"Cash:           ${metrics.cash:>10,.2f}")
            print(f"Buying Power:   ${metrics.buying_power:>10,.2f}")
            print(f"Day P&L:        ${metrics.day_pnl:>10,.2f} ({metrics.day_pnl_pct:+.2f}%)")
            print(f"Total P&L:      ${metrics.total_pnl:>10,.2f} ({metrics.total_pnl_pct:+.2f}%)")
            print(f"Exposure:       {metrics.exposure_pct:>10.1f}%")
            print(f"Positions:      {metrics.num_positions:>10}")
            
            if positions:
                print(f"\n📈 CURRENT POSITIONS:")
                print("-" * 60)
                for pos in positions:
                    pnl_sign = "+" if pos.unrealized_pnl >= 0 else ""
                    print(f"{pos.symbol:<6} {pos.quantity:>6} @ ${pos.avg_price:>8.2f} | "
                          f"${pos.market_value:>10,.2f} | {pnl_sign}{pos.unrealized_pnl_pct:>6.1f}%")
            
            print("="*60)
            
        except Exception as e:
            logger.error(f"Error printing portfolio status: {str(e)}")
    
    def export_to_csv(self, filename: Optional[str] = None) -> str:
        """
        Export portfolio data to CSV for analysis.
        
        Args:
            filename: Optional filename, defaults to timestamped name
            
        Returns:
            Path to exported file
        """
        if filename is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"portfolio_export_{timestamp}.csv"
        
        filepath = os.path.join(self.save_path, filename)
        
        try:
            positions = self.get_current_positions()
            if positions:
                df = pd.DataFrame([pos.to_dict() for pos in positions])
                df.to_csv(filepath, index=False)
                logger.info(f"Portfolio exported to {filepath}")
            else:
                logger.warning("No positions to export")
                
        except Exception as e:
            logger.error(f"Error exporting portfolio: {str(e)}")
        
        return filepath

# Convenience function for quick portfolio check
def quick_portfolio_check(broker) -> None:
    """
    Quick portfolio status check.
    
    Args:
        broker: Broker instance
    """
    tracker = PortfolioTracker(broker)
    tracker.print_portfolio_status()

if __name__ == "__main__":
    # Test portfolio tracker
    from broker_interface import create_broker
    
    print("🏦 Testing Portfolio Tracker...")
    
    # Create broker (try real, fallback to mock)
    broker = create_broker(paper_trading=True, mock=False)
    if not broker.check_connection():
        broker = create_broker(mock=True)
    
    # Create tracker and test
    tracker = PortfolioTracker(broker, initial_capital=10000)
    
    # Take snapshot and print status
    tracker.take_snapshot()
    tracker.print_portfolio_status()
    
    # Generate performance summary
    summary = tracker.get_performance_summary()
    print(f"\n📊 Performance Summary:")
    for key, value in summary.items():
        print(f"{key}: {value}")
    
    print("\n🎉 Portfolio tracker test complete!")
