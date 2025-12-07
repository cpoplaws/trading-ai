"""
Portfolio tracker for real-time P/L tracking and position management.

Monitors positions, calculates metrics, and provides portfolio analytics.
"""
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional

from execution.broker_interface import BrokerInterface, Position
from utils.logger import setup_logger

logger = setup_logger(__name__)


@dataclass
class PortfolioMetrics:
    """Portfolio performance metrics."""

    total_value: float
    cash: float
    long_value: float
    short_value: float
    total_pl: float
    total_pl_percent: float
    day_pl: float
    day_pl_percent: float
    buying_power: float
    equity: float
    num_positions: int


@dataclass
class PositionMetrics:
    """Individual position metrics."""

    symbol: str
    qty: float
    avg_entry_price: float
    current_price: float
    market_value: float
    cost_basis: float
    unrealized_pl: float
    unrealized_pl_percent: float
    portfolio_weight: float  # Percentage of total portfolio


class PortfolioTracker:
    """
    Tracks portfolio performance and positions in real-time.
    
    Provides P/L calculations, position metrics, and risk analytics.
    """

    def __init__(self, broker: BrokerInterface):
        """
        Initialize portfolio tracker.
        
        Args:
            broker: Broker interface instance
        """
        self.broker = broker
        self.initial_equity: Optional[float] = None
        logger.info("PortfolioTracker initialized")

    def get_portfolio_metrics(self) -> PortfolioMetrics:
        """
        Get comprehensive portfolio metrics.
        
        Returns:
            PortfolioMetrics object
        """
        try:
            account = self.broker.get_account_info()
            positions = self.broker.get_positions()

            # Set initial equity on first call
            if self.initial_equity is None:
                self.initial_equity = account.last_equity

            # Calculate total P/L
            total_pl = account.equity - self.initial_equity
            total_pl_percent = (total_pl / self.initial_equity) * 100 if self.initial_equity else 0

            # Calculate day P/L
            day_pl = account.equity - account.last_equity
            day_pl_percent = (day_pl / account.last_equity) * 100 if account.last_equity else 0

            return PortfolioMetrics(
                total_value=account.portfolio_value,
                cash=account.cash,
                long_value=account.long_market_value,
                short_value=account.short_market_value,
                total_pl=total_pl,
                total_pl_percent=total_pl_percent,
                day_pl=day_pl,
                day_pl_percent=day_pl_percent,
                buying_power=account.buying_power,
                equity=account.equity,
                num_positions=len(positions),
            )

        except Exception as e:
            logger.error(f"Error getting portfolio metrics: {str(e)}")
            raise

    def get_position_metrics(self, symbol: Optional[str] = None) -> List[PositionMetrics]:
        """
        Get detailed metrics for positions.
        
        Args:
            symbol: Specific symbol to get metrics for (None = all positions)
            
        Returns:
            List of PositionMetrics objects
        """
        try:
            if symbol:
                position = self.broker.get_position(symbol)
                positions = [position] if position else []
            else:
                positions = self.broker.get_positions()

            if not positions:
                return []

            account = self.broker.get_account_info()
            position_metrics = []

            for pos in positions:
                cost_basis = pos.qty * pos.avg_entry_price
                portfolio_weight = (pos.market_value / account.portfolio_value) * 100

                metrics = PositionMetrics(
                    symbol=pos.symbol,
                    qty=pos.qty,
                    avg_entry_price=pos.avg_entry_price,
                    current_price=pos.current_price,
                    market_value=pos.market_value,
                    cost_basis=cost_basis,
                    unrealized_pl=pos.unrealized_pl,
                    unrealized_pl_percent=pos.unrealized_plpc,
                    portfolio_weight=portfolio_weight,
                )

                position_metrics.append(metrics)

            return position_metrics

        except Exception as e:
            logger.error(f"Error getting position metrics: {str(e)}")
            return []

    def get_portfolio_summary(self) -> Dict:
        """
        Get a summary of portfolio status.
        
        Returns:
            Dictionary with portfolio summary
        """
        try:
            metrics = self.get_portfolio_metrics()
            position_metrics = self.get_position_metrics()

            # Calculate additional stats
            total_unrealized_pl = sum(p.unrealized_pl for p in position_metrics)
            winners = [p for p in position_metrics if p.unrealized_pl > 0]
            losers = [p for p in position_metrics if p.unrealized_pl < 0]

            return {
                "timestamp": datetime.now().isoformat(),
                "portfolio_value": metrics.total_value,
                "cash": metrics.cash,
                "equity": metrics.equity,
                "buying_power": metrics.buying_power,
                "total_pl": metrics.total_pl,
                "total_pl_percent": metrics.total_pl_percent,
                "day_pl": metrics.day_pl,
                "day_pl_percent": metrics.day_pl_percent,
                "num_positions": metrics.num_positions,
                "total_unrealized_pl": total_unrealized_pl,
                "num_winners": len(winners),
                "num_losers": len(losers),
                "win_rate": len(winners) / len(position_metrics) * 100 if position_metrics else 0,
            }

        except Exception as e:
            logger.error(f"Error getting portfolio summary: {str(e)}")
            return {}

    def print_portfolio_report(self) -> None:
        """Print formatted portfolio report to console."""
        try:
            print("\n" + "=" * 70)
            print(" " * 20 + "PORTFOLIO REPORT")
            print("=" * 70)

            metrics = self.get_portfolio_metrics()
            print(f"\n{'Portfolio Value:':<30} ${metrics.total_value:>15,.2f}")
            print(f"{'Cash:':<30} ${metrics.cash:>15,.2f}")
            print(f"{'Buying Power:':<30} ${metrics.buying_power:>15,.2f}")
            print(f"{'Equity:':<30} ${metrics.equity:>15,.2f}")

            print("\n" + "-" * 70)
            print("PERFORMANCE")
            print("-" * 70)

            pl_color = "+" if metrics.total_pl >= 0 else ""
            print(f"{'Total P/L:':<30} {pl_color}${metrics.total_pl:>14,.2f} ({pl_color}{metrics.total_pl_percent:>6.2f}%)")

            day_pl_color = "+" if metrics.day_pl >= 0 else ""
            print(f"{'Day P/L:':<30} {day_pl_color}${metrics.day_pl:>14,.2f} ({day_pl_color}{metrics.day_pl_percent:>6.2f}%)")

            print("\n" + "-" * 70)
            print("POSITIONS")
            print("-" * 70)

            position_metrics = self.get_position_metrics()

            if position_metrics:
                print(f"\n{'Symbol':<10} {'Qty':>8} {'Entry':>10} {'Current':>10} {'P/L $':>12} {'P/L %':>10} {'Weight':>8}")
                print("-" * 70)

                for pos in position_metrics:
                    pl_sign = "+" if pos.unrealized_pl >= 0 else ""
                    print(
                        f"{pos.symbol:<10} "
                        f"{pos.qty:>8.0f} "
                        f"${pos.avg_entry_price:>9.2f} "
                        f"${pos.current_price:>9.2f} "
                        f"{pl_sign}${pos.unrealized_pl:>11.2f} "
                        f"{pl_sign}{pos.unrealized_pl_percent:>9.2f}% "
                        f"{pos.portfolio_weight:>7.1f}%"
                    )

                print("-" * 70)
                print(f"{'Total Positions:':<30} {len(position_metrics):>15}")
                total_unrealized = sum(p.unrealized_pl for p in position_metrics)
                pl_sign = "+" if total_unrealized >= 0 else ""
                print(f"{'Total Unrealized P/L:':<30} {pl_sign}${total_unrealized:>14,.2f}")
            else:
                print("\nNo open positions")

            print("=" * 70 + "\n")

        except Exception as e:
            logger.error(f"Error printing portfolio report: {str(e)}")

    def get_position_concentration(self) -> Dict[str, float]:
        """
        Get position concentration (portfolio weights).
        
        Returns:
            Dictionary mapping symbol to portfolio weight percentage
        """
        position_metrics = self.get_position_metrics()
        return {pos.symbol: pos.portfolio_weight for pos in position_metrics}

    def get_risk_metrics(self) -> Dict:
        """
        Calculate basic risk metrics.
        
        Returns:
            Dictionary with risk metrics
        """
        try:
            position_metrics = self.get_position_metrics()

            if not position_metrics:
                return {
                    "max_position_weight": 0.0,
                    "concentration_risk": "LOW",
                    "num_positions": 0,
                    "diversification_score": 0.0,
                }

            weights = [p.portfolio_weight for p in position_metrics]
            max_weight = max(weights)

            # Simple concentration risk assessment
            if max_weight > 40:
                risk = "HIGH"
            elif max_weight > 25:
                risk = "MEDIUM"
            else:
                risk = "LOW"

            # Simple diversification score (0-100)
            # More positions with balanced weights = better diversification
            num_positions = len(position_metrics)
            ideal_weight = 100 / num_positions if num_positions > 0 else 0
            weight_variance = sum((w - ideal_weight) ** 2 for w in weights) / num_positions if num_positions > 0 else 0
            diversification_score = max(0, 100 - weight_variance)

            return {
                "max_position_weight": max_weight,
                "concentration_risk": risk,
                "num_positions": num_positions,
                "diversification_score": diversification_score,
                "position_weights": dict(zip([p.symbol for p in position_metrics], weights)),
            }

        except Exception as e:
            logger.error(f"Error calculating risk metrics: {str(e)}")
            return {}


if __name__ == "__main__":
    from execution.broker_interface import MockBroker

    print("📊 Testing PortfolioTracker...")

    # Use mock broker
    broker = MockBroker(initial_capital=100000)
    broker.connect()

    tracker = PortfolioTracker(broker)

    # Make some trades
    from execution.broker_interface import OrderSide, OrderType

    print("\n📈 Placing some test trades...")
    broker.place_order("AAPL", 10, OrderSide.BUY, OrderType.MARKET)
    broker.place_order("MSFT", 15, OrderSide.BUY, OrderType.MARKET)
    broker.place_order("GOOGL", 5, OrderSide.BUY, OrderType.MARKET)

    # Print portfolio report
    tracker.print_portfolio_report()

    # Get portfolio summary
    print("📋 Portfolio Summary:")
    summary = tracker.get_portfolio_summary()
    for key, value in summary.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.2f}")
        else:
            print(f"  {key}: {value}")

    # Get risk metrics
    print("\n⚠️  Risk Metrics:")
    risk = tracker.get_risk_metrics()
    print(f"  Max Position Weight: {risk.get('max_position_weight', 0):.1f}%")
    print(f"  Concentration Risk: {risk.get('concentration_risk', 'N/A')}")
    print(f"  Diversification Score: {risk.get('diversification_score', 0):.1f}/100")

    print("\n✅ PortfolioTracker test complete!")
