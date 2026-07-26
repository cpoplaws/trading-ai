"""
SupervisorAgent - The Brain of the Multi-Chain Trading System

Responsibilities:
1. Performance tracking per strategy-chain combo
2. Dynamic capital allocation based on Sharpe ratio (or multi-factor if enhanced)
3. Arbitrage detection across venues
4. Risk management and circuit breakers

Phase 4 Enhancements:
- Multi-factor allocation (Sharpe + Sortino + Calmar + consistency)
- Real-time arbitrage scanning with live prices
- Advanced risk monitoring with portfolio-wide tracking
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional
from datetime import datetime, timedelta
from enum import Enum
import logging
import numpy as np

# Phase 4: Import enhancements (optional)
try:
    from src.agents.supervisor_enhancements import (
        AllocationOptimizer,
        ArbitrageScanner,
        RiskMonitor,
        PerformanceMetrics,
        PortfolioPosition,
        RiskMetrics
    )
    ENHANCEMENTS_AVAILABLE = True
except ImportError:
    ENHANCEMENTS_AVAILABLE = False
    logger.warning("Supervisor enhancements not available - using basic mode")

logger = logging.getLogger(__name__)


class Signal(Enum):
    """Trading signals"""
    BUY = "buy"
    SELL = "sell"
    HOLD = "hold"


@dataclass
class StrategyPerformance:
    """Performance metrics for a strategy-chain instance"""
    strategy_name: str
    chain: str  # "Base", "Solana", or "CEX:Binance"
    instance_id: str

    # Performance metrics
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    total_pnl: float = 0.0
    avg_pnl_per_trade: float = 0.0
    win_rate: float = 0.0
    sharpe_ratio: float = 0.0
    max_drawdown: float = 0.0

    # Allocation
    allocated_capital: float = 0.0
    current_capital: float = 0.0
    capital_utilization: float = 0.0

    # Trade history for calculations
    trade_returns: List[float] = field(default_factory=list)
    equity_curve: List[float] = field(default_factory=list)

    # Timestamps
    last_trade: Optional[datetime] = None
    last_performance_update: datetime = field(default_factory=datetime.now)
    created_at: datetime = field(default_factory=datetime.now)


@dataclass
class ArbitrageOpportunity:
    """Cross-venue arbitrage opportunity"""
    token: str
    buy_venue: str  # "Binance" or "Base:Uniswap"
    sell_venue: str
    buy_price: float
    sell_price: float
    spread_percent: float
    estimated_fees: float
    estimated_gas: float  # 0 for CEX
    net_profit_percent: float
    trade_size: float
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class TradeResult:
    """Result of a completed trade"""
    instance_id: str
    symbol: str
    side: str  # "buy" or "sell"
    quantity: float
    entry_price: float
    exit_price: Optional[float] = None
    pnl: float = 0.0
    fees: float = 0.0
    gas: float = 0.0
    success: bool = True
    timestamp: datetime = field(default_factory=datetime.now)


class SupervisorAgent:
    """
    The brain of the multi-chain trading system.

    Manages:
    - Performance tracking across all strategy instances
    - Dynamic capital allocation
    - Arbitrage detection
    - Risk management and circuit breakers
    """

    def __init__(
        self,
        total_capital: float = 10000.0,
        reallocation_interval_hours: int = 6,
        arbitrage_threshold: float = 0.01,  # 1% minimum profit
        max_position_pct: float = 0.10,  # 10% max per position
        max_asset_pct: float = 0.50,  # 50% max per asset
        circuit_breaker_daily_loss: float = 0.15,  # 15% daily loss triggers halt
        circuit_breaker_peak_loss: float = 0.25,  # 25% from peak triggers halt
        use_enhanced_allocation: bool = False,  # Phase 4: Multi-factor allocation
        use_enhanced_arbitrage: bool = False,  # Phase 4: Real-time arb scanning
        use_enhanced_risk: bool = False,  # Phase 4: Advanced risk monitoring
        cex_connector = None,  # Phase 4: For real-time CEX prices
        dex_connector = None,  # Phase 4: For real-time DEX prices
    ):
        self.total_capital = total_capital
        self.available_capital = total_capital
        self.reallocation_interval = timedelta(hours=reallocation_interval_hours)
        self.arbitrage_threshold = arbitrage_threshold
        self.max_position_pct = max_position_pct
        self.max_asset_pct = max_asset_pct
        self.circuit_breaker_daily_loss = circuit_breaker_daily_loss
        self.circuit_breaker_peak_loss = circuit_breaker_peak_loss

        # Performance tracking
        self.instances: Dict[str, StrategyPerformance] = {}
        self.last_reallocation = datetime.now()

        # Portfolio state
        self.positions: Dict[str, float] = {}  # asset -> total quantity across all chains
        self.daily_start_value = total_capital
        self.peak_value = total_capital
        self.circuit_breaker_triggered = False

        # Phase 4: Enhancement modules
        self.use_enhanced_allocation = use_enhanced_allocation and ENHANCEMENTS_AVAILABLE
        self.use_enhanced_arbitrage = use_enhanced_arbitrage and ENHANCEMENTS_AVAILABLE
        self.use_enhanced_risk = use_enhanced_risk and ENHANCEMENTS_AVAILABLE
        self.cex_connector = cex_connector
        self.dex_connector = dex_connector

        # Initialize enhancement modules if available
        if self.use_enhanced_allocation:
            self.allocation_optimizer = AllocationOptimizer()
            logger.info("✅ Multi-factor allocation enabled")
        else:
            self.allocation_optimizer = None

        if self.use_enhanced_arbitrage:
            self.arbitrage_scanner = ArbitrageScanner(min_profit_pct=arbitrage_threshold)
            logger.info("✅ Real-time arbitrage scanning enabled")
        else:
            self.arbitrage_scanner = None

        if self.use_enhanced_risk:
            self.risk_monitor = RiskMonitor(
                max_position_pct=max_position_pct,
                max_asset_pct=max_asset_pct,
                max_daily_loss_pct=circuit_breaker_daily_loss,
                max_drawdown_pct=circuit_breaker_peak_loss
            )
            logger.info("✅ Advanced risk monitoring enabled")
        else:
            self.risk_monitor = None

        logger.info(f"SupervisorAgent initialized with ${total_capital:,.2f}")

    def register_instance(
        self,
        strategy_name: str,
        chain: str,
        initial_capital: float
    ) -> str:
        """Register a new strategy instance"""
        instance_id = f"{strategy_name}_{chain}_{len(self.instances):03d}"

        self.instances[instance_id] = StrategyPerformance(
            strategy_name=strategy_name,
            chain=chain,
            instance_id=instance_id,
            allocated_capital=initial_capital,
            current_capital=initial_capital,
        )

        self.available_capital -= initial_capital

        logger.info(
            f"Registered instance: {instance_id} with ${initial_capital:,.2f} "
            f"({chain})"
        )
        return instance_id

    def track_trade(self, trade: TradeResult) -> None:
        """Update performance metrics after a trade"""
        if trade.instance_id not in self.instances:
            logger.error(f"Unknown instance: {trade.instance_id}")
            return

        instance = self.instances[trade.instance_id]

        # Update trade counts
        instance.total_trades += 1
        if trade.pnl > 0:
            instance.winning_trades += 1
        else:
            instance.losing_trades += 1

        # Update P&L
        net_pnl = trade.pnl - trade.fees - trade.gas
        instance.total_pnl += net_pnl
        instance.current_capital += net_pnl
        instance.trade_returns.append(net_pnl / instance.allocated_capital)
        instance.equity_curve.append(instance.current_capital)

        # Recalculate metrics
        instance.win_rate = instance.winning_trades / instance.total_trades
        instance.avg_pnl_per_trade = instance.total_pnl / instance.total_trades
        instance.sharpe_ratio = self._calculate_sharpe_ratio(instance)
        instance.max_drawdown = self._calculate_max_drawdown(instance)
        instance.capital_utilization = (
            instance.current_capital / instance.allocated_capital
        )

        instance.last_trade = trade.timestamp
        instance.last_performance_update = datetime.now()

        # Update global positions
        asset = trade.symbol.split("/")[0]  # Extract base asset
        if trade.side == "buy":
            self.positions[asset] = self.positions.get(asset, 0) + trade.quantity
        else:
            self.positions[asset] = self.positions.get(asset, 0) - trade.quantity

        logger.info(
            f"Trade tracked: {trade.instance_id} | "
            f"P&L: ${net_pnl:.2f} | "
            f"Win Rate: {instance.win_rate:.1%} | "
            f"Sharpe: {instance.sharpe_ratio:.2f}"
        )

    def calculate_allocations(self) -> Dict[str, float]:
        """
        Calculate new capital allocations based on performance.

        Basic Strategy (use_enhanced_allocation=False):
        - Rank instances by Sharpe ratio
        - Top performer: 40%
        - Second: 30%
        - Third: 20%
        - Rest: 10% split equally

        Enhanced Strategy (use_enhanced_allocation=True):
        - Multi-factor scoring (Sharpe + Sortino + Calmar + consistency + trend)
        - Proportional allocation based on composite scores
        - Min 5% and max 40% per instance
        """
        if not self.instances:
            return {}

        # Check if it's time to reallocate
        if datetime.now() - self.last_reallocation < self.reallocation_interval:
            return {
                instance_id: perf.allocated_capital
                for instance_id, perf in self.instances.items()
            }

        total_available = sum(inst.current_capital for _, inst in self.instances.items())

        # Phase 4: Use enhanced allocation if enabled
        if self.use_enhanced_allocation and self.allocation_optimizer:
            # Convert StrategyPerformance to PerformanceMetrics
            metrics_dict = {}
            for instance_id, perf in self.instances.items():
                metrics_dict[instance_id] = self._to_performance_metrics(perf)

            allocations = self.allocation_optimizer.optimize_allocations(
                metrics_dict,
                total_available
            )

            # Update allocated capital
            for instance_id, new_allocation in allocations.items():
                old_allocation = self.instances[instance_id].allocated_capital
                self.instances[instance_id].allocated_capital = new_allocation
                metrics = metrics_dict[instance_id]
                logger.info(
                    f"Reallocation (Enhanced): {instance_id} | "
                    f"${old_allocation:.0f} → ${new_allocation:.0f} | "
                    f"Sharpe: {metrics.sharpe_ratio:.2f} | "
                    f"Sortino: {metrics.sortino_ratio:.2f}"
                )

            self.last_reallocation = datetime.now()
            return allocations

        # Basic allocation (original logic)
        ranked = sorted(
            self.instances.items(),
            key=lambda x: x[1].sharpe_ratio,
            reverse=True
        )

        allocations = {}

        if len(ranked) == 1:
            allocations[ranked[0][0]] = total_available
        elif len(ranked) == 2:
            allocations[ranked[0][0]] = total_available * 0.60
            allocations[ranked[1][0]] = total_available * 0.40
        elif len(ranked) == 3:
            allocations[ranked[0][0]] = total_available * 0.50
            allocations[ranked[1][0]] = total_available * 0.30
            allocations[ranked[2][0]] = total_available * 0.20
        else:
            allocations[ranked[0][0]] = total_available * 0.40
            allocations[ranked[1][0]] = total_available * 0.30
            allocations[ranked[2][0]] = total_available * 0.20

            # Split remaining 10% among the rest
            remaining = total_available * 0.10
            per_instance = remaining / (len(ranked) - 3)
            for instance_id, _ in ranked[3:]:
                allocations[instance_id] = per_instance

        # Update allocated capital
        for instance_id, new_allocation in allocations.items():
            old_allocation = self.instances[instance_id].allocated_capital
            self.instances[instance_id].allocated_capital = new_allocation
            logger.info(
                f"Reallocation (Basic): {instance_id} | "
                f"${old_allocation:.0f} → ${new_allocation:.0f} | "
                f"Sharpe: {self.instances[instance_id].sharpe_ratio:.2f}"
            )

        self.last_reallocation = datetime.now()
        return allocations

    def detect_arbitrage(
        self,
        cex_prices: Dict[str, float] = None,  # {"BTC": 64000}
        dex_prices: Dict[str, Dict[str, float]] = None,  # {"Base": {"WBTC": 64800}}
        dex_fees: Dict[str, float] = None,  # {"Base": 0.003}
        dex_gas: Dict[str, float] = None,  # {"Base": 10.0}
        tokens: List[str] = None,  # Phase 4: List of tokens to scan
    ) -> List[ArbitrageOpportunity]:
        """
        Detect arbitrage opportunities across CEX and DEX.

        Basic Mode (use_enhanced_arbitrage=False):
        - Uses provided price dictionaries
        - Manual price feeding required

        Enhanced Mode (use_enhanced_arbitrage=True):
        - Automatically fetches real-time prices from connectors
        - Continuous scanning across all supported tokens

        Example:
        - BTC on Binance: $64,000
        - WBTC on Uniswap (Base): $64,800
        - Spread: 1.25%, Fees: 0.5%, Gas: 0.016%
        - Net: 0.734% (below 1% threshold - don't execute)
        """
        # Phase 4: Use enhanced arbitrage scanner if enabled
        if self.use_enhanced_arbitrage and self.arbitrage_scanner:
            scanner_opportunities = self.arbitrage_scanner.scan_opportunities(
                self.cex_connector,
                self.dex_connector,
                tokens=tokens
            )

            # Convert to ArbitrageOpportunity format
            opportunities = []
            for opp in scanner_opportunities:
                opportunities.append(
                    ArbitrageOpportunity(
                        token=opp['token'],
                        buy_venue=opp['buy_venue'],
                        sell_venue=opp['sell_venue'],
                        buy_price=opp['buy_price'],
                        sell_price=opp['sell_price'],
                        spread_percent=opp['spread_pct'] * 100,
                        estimated_fees=0.4,  # Mock for now
                        estimated_gas=8.0,  # Mock for now
                        net_profit_percent=opp['net_profit_pct'] * 100,
                        trade_size=0.01,
                        timestamp=opp['timestamp']
                    )
                )

            logger.info(f"Enhanced arbitrage scan: Found {len(opportunities)} opportunities")
            return opportunities

        # Basic mode - use provided prices
        if cex_prices is None or dex_prices is None:
            logger.warning("No prices provided for arbitrage detection")
            return []

        dex_fees = dex_fees or {}
        dex_gas = dex_gas or {}
        opportunities = []

        for token, cex_price in cex_prices.items():
            # Check against each chain's DEX price
            for chain, chain_prices in dex_prices.items():
                # Map token names (BTC -> WBTC, ETH -> WETH, etc.)
                dex_token = self._map_token_to_dex(token, chain)
                if dex_token not in chain_prices:
                    continue

                dex_price = chain_prices[dex_token]

                # Calculate spread
                spread = (dex_price - cex_price) / cex_price

                # Estimate costs
                cex_fee_pct = 0.001  # 0.1% typical CEX fee
                dex_fee_pct = dex_fees.get(chain, 0.003)  # 0.3% default
                gas_cost = dex_gas.get(chain, 10.0)
                gas_pct = gas_cost / (cex_price * 0.01)  # Assume 0.01 BTC trade

                total_fees = cex_fee_pct + dex_fee_pct
                net_profit = spread - total_fees - gas_pct

                # Only flag if above threshold
                if abs(net_profit) > self.arbitrage_threshold:
                    direction = "buy CEX, sell DEX" if net_profit > 0 else "buy DEX, sell CEX"
                    opportunities.append(
                        ArbitrageOpportunity(
                            token=token,
                            buy_venue="CEX" if net_profit > 0 else f"{chain}:DEX",
                            sell_venue=f"{chain}:DEX" if net_profit > 0 else "CEX",
                            buy_price=cex_price if net_profit > 0 else dex_price,
                            sell_price=dex_price if net_profit > 0 else cex_price,
                            spread_percent=spread * 100,
                            estimated_fees=total_fees * 100,
                            estimated_gas=gas_pct * 100,
                            net_profit_percent=net_profit * 100,
                            trade_size=0.01,  # TODO: Calculate optimal size
                        )
                    )

                    logger.info(
                        f"Arbitrage opportunity: {token} | {direction} | "
                        f"Net: {net_profit*100:.2f}% | "
                        f"Spread: {spread*100:.2f}% | Fees: {total_fees*100:.2f}%"
                    )

        return opportunities

    def check_risk_limits(
        self,
        instance_id: str,
        trade_size: float,
        asset: str
    ) -> tuple[bool, str]:
        """
        Check if proposed trade violates risk limits.

        Basic Mode: Simple position and loss checks
        Enhanced Mode: Portfolio-wide risk aggregation and advanced metrics

        Returns: (approved, reason)
        """
        # Check circuit breaker
        if self.circuit_breaker_triggered:
            return False, "Circuit breaker triggered - all trading halted"

        # Check if instance exists
        if instance_id not in self.instances:
            return False, f"Unknown instance: {instance_id}"

        instance = self.instances[instance_id]

        # Check if instance has capital
        if instance.current_capital <= 0:
            return False, "Instance has no capital"

        current_value = sum(inst.current_capital for inst in self.instances.values())

        # Phase 4: Use enhanced risk monitoring if enabled
        if self.use_enhanced_risk and self.risk_monitor:
            # Build portfolio positions
            portfolio_positions = self._build_portfolio_positions()

            # Simulate adding this trade
            if asset in portfolio_positions:
                portfolio_positions[asset].total_quantity += trade_size
                portfolio_positions[asset].total_value_usd += trade_size
            else:
                portfolio_positions[asset] = PortfolioPosition(
                    asset=asset,
                    total_quantity=trade_size,
                    total_value_usd=trade_size,
                    chains={},
                    percentage_of_portfolio=trade_size / current_value,
                    cost_basis=trade_size,
                    unrealized_pnl=0.0
                )

            # Check enhanced risk limits
            daily_pnl = current_value - self.daily_start_value
            is_safe, violations = self.risk_monitor.check_limits(
                portfolio_positions,
                current_value,
                daily_pnl,
                self.peak_value
            )

            if not is_safe:
                self.circuit_breaker_triggered = True
                return False, "; ".join(violations)

            # Update peak
            if current_value > self.peak_value:
                self.peak_value = current_value

            return True, "Approved (Enhanced Risk Check)"

        # Basic risk checks (original logic)
        position_pct = trade_size / current_value
        if position_pct > self.max_position_pct:
            return False, f"Position too large: {position_pct:.1%} > {self.max_position_pct:.1%}"

        # Check asset concentration (50% max per asset)
        new_asset_exposure = self.positions.get(asset, 0) + trade_size
        asset_pct = new_asset_exposure / current_value
        if asset_pct > self.max_asset_pct:
            return False, f"Asset exposure too high: {asset_pct:.1%} > {self.max_asset_pct:.1%}"

        # Check daily loss limit
        daily_pnl = current_value - self.daily_start_value
        if daily_pnl / self.daily_start_value < -self.circuit_breaker_daily_loss:
            self.circuit_breaker_triggered = True
            return False, f"Circuit breaker: Daily loss exceeded {self.circuit_breaker_daily_loss:.0%}"

        # Check drawdown from peak
        if current_value < self.peak_value * (1 - self.circuit_breaker_peak_loss):
            self.circuit_breaker_triggered = True
            return False, f"Circuit breaker: Drawdown from peak exceeded {self.circuit_breaker_peak_loss:.0%}"

        # Update peak
        if current_value > self.peak_value:
            self.peak_value = current_value

        return True, "Approved"

    def get_performance_summary(self) -> Dict:
        """Get overall performance summary"""
        if not self.instances:
            return {"total_instances": 0}

        total_pnl = sum(inst.total_pnl for inst in self.instances.values())
        total_trades = sum(inst.total_trades for inst in self.instances.values())
        winning_trades = sum(inst.winning_trades for inst in self.instances.values())
        current_value = sum(inst.current_capital for inst in self.instances.values())

        summary = {
            "total_instances": len(self.instances),
            "total_capital": current_value,
            "total_pnl": total_pnl,
            "total_return_pct": (current_value - self.total_capital) / self.total_capital * 100,
            "total_trades": total_trades,
            "overall_win_rate": winning_trades / total_trades if total_trades > 0 else 0,
            "circuit_breaker": self.circuit_breaker_triggered,
            "peak_value": self.peak_value,
            "daily_start_value": self.daily_start_value,
            "positions": self.positions,
            "top_performers": [
                {
                    "instance_id": inst_id,
                    "sharpe_ratio": inst.sharpe_ratio,
                    "pnl": inst.total_pnl,
                    "win_rate": inst.win_rate,
                }
                for inst_id, inst in sorted(
                    self.instances.items(),
                    key=lambda x: x[1].sharpe_ratio,
                    reverse=True
                )[:5]
            ]
        }

        # Phase 4: Add enhanced metrics if available
        if self.use_enhanced_risk and self.risk_monitor:
            portfolio_positions = self._build_portfolio_positions()
            risk_metrics = self.risk_monitor.calculate_risk_metrics(
                portfolio_positions,
                current_value
            )
            summary["risk_metrics"] = {
                "total_exposure": risk_metrics.total_exposure,
                "max_position_size": risk_metrics.max_position_size,
                "concentration_risk": risk_metrics.concentration_risk,
                "daily_var": risk_metrics.daily_var,
                "sharpe_ratio": risk_metrics.sharpe_ratio,
                "beta": risk_metrics.beta,
                "correlation_to_btc": risk_metrics.correlation_to_btc
            }

        return summary

    def get_risk_metrics(self) -> Optional[Dict]:
        """
        Get detailed risk metrics (Phase 4 enhancement).

        Returns:
            Risk metrics dict or None if enhanced risk not enabled
        """
        if not self.use_enhanced_risk or not self.risk_monitor:
            return None

        current_value = sum(inst.current_capital for inst in self.instances.values())
        portfolio_positions = self._build_portfolio_positions()

        metrics = self.risk_monitor.calculate_risk_metrics(
            portfolio_positions,
            current_value
        )

        return {
            "total_exposure": metrics.total_exposure,
            "max_position_size": metrics.max_position_size,
            "concentration_risk": metrics.concentration_risk,
            "daily_var": metrics.daily_var,
            "sharpe_ratio": metrics.sharpe_ratio,
            "beta": metrics.beta,
            "correlation_to_btc": metrics.correlation_to_btc,
            "positions": {
                asset: {
                    "quantity": pos.total_quantity,
                    "value_usd": pos.total_value_usd,
                    "percentage": pos.percentage_of_portfolio,
                    "chains": pos.chains
                }
                for asset, pos in portfolio_positions.items()
            }
        }

    def reset_daily_tracking(self) -> None:
        """Reset daily tracking metrics (call at midnight)"""
        current_value = sum(inst.current_capital for inst in self.instances.values())
        self.daily_start_value = current_value
        logger.info(f"Daily tracking reset. Starting value: ${current_value:,.2f}")

    @staticmethod
    def _calculate_sharpe_ratio(instance: StrategyPerformance) -> float:
        """Calculate Sharpe ratio from trade returns"""
        if len(instance.trade_returns) < 2:
            return 0.0

        returns = np.array(instance.trade_returns)
        if returns.std() == 0:
            return 0.0

        # Annualize: assume ~250 trading days
        sharpe = (returns.mean() / returns.std()) * np.sqrt(250)
        return float(sharpe)

    @staticmethod
    def _calculate_max_drawdown(instance: StrategyPerformance) -> float:
        """Calculate maximum drawdown from equity curve"""
        if len(instance.equity_curve) < 2:
            return 0.0

        equity = np.array(instance.equity_curve)
        peak = np.maximum.accumulate(equity)
        drawdown = (peak - equity) / peak
        return float(drawdown.max())

    @staticmethod
    def _map_token_to_dex(token: str, chain: str) -> str:
        """Map CEX token to DEX equivalent"""
        mapping = {
            "BTC": {"Base": "WBTC", "Arbitrum": "WBTC", "Optimism": "WBTC"},
            "ETH": {"Base": "WETH", "Arbitrum": "WETH", "Optimism": "WETH", "Solana": "ETH"},
            "SOL": {"Solana": "SOL"},
        }
        return mapping.get(token, {}).get(chain, token)

    def _to_performance_metrics(self, perf: StrategyPerformance) -> 'PerformanceMetrics':
        """
        Convert StrategyPerformance to PerformanceMetrics for enhanced allocation.

        Calculates:
        - Sortino ratio (downside deviation)
        - Calmar ratio (return / max drawdown)
        - Profit factor
        - Consistency score
        """
        if not ENHANCEMENTS_AVAILABLE:
            raise RuntimeError("Performance metrics conversion requires enhancements")

        # Calculate Sortino ratio (use only negative returns for denominator)
        sortino = 0.0
        if len(perf.trade_returns) >= 2:
            returns = np.array(perf.trade_returns)
            negative_returns = returns[returns < 0]
            if len(negative_returns) > 0 and negative_returns.std() > 0:
                sortino = (returns.mean() / negative_returns.std()) * np.sqrt(250)

        # Calculate Calmar ratio (return / max drawdown)
        calmar = 0.0
        if perf.max_drawdown > 0 and perf.total_trades > 0:
            annual_return = perf.avg_pnl_per_trade * 250 / perf.allocated_capital
            calmar = annual_return / perf.max_drawdown

        # Calculate profit factor (total wins / total losses)
        profit_factor = 1.0
        if perf.winning_trades > 0 and perf.losing_trades > 0:
            avg_win = perf.total_pnl / perf.winning_trades if perf.winning_trades > 0 else 0
            avg_loss = abs(perf.total_pnl) / perf.losing_trades if perf.losing_trades > 0 else 1
            if avg_loss > 0:
                profit_factor = avg_win / avg_loss

        # Calculate consistency score (how consistent are returns?)
        consistency_score = 0.5
        if len(perf.trade_returns) >= 5:
            returns = np.array(perf.trade_returns)
            # Lower variance = higher consistency
            variance = returns.var()
            consistency_score = max(0, min(1, 1 - variance * 10))

        return PerformanceMetrics(
            sharpe_ratio=perf.sharpe_ratio,
            sortino_ratio=sortino,
            calmar_ratio=calmar,
            max_drawdown=perf.max_drawdown,
            win_rate=perf.win_rate,
            profit_factor=profit_factor,
            avg_win=perf.avg_pnl_per_trade if perf.winning_trades > 0 else 0,
            avg_loss=perf.avg_pnl_per_trade if perf.losing_trades > 0 else 0,
            consistency_score=consistency_score,
            total_pnl=perf.total_pnl,
            total_trades=perf.total_trades
        )

    def _build_portfolio_positions(self) -> Dict[str, 'PortfolioPosition']:
        """
        Build portfolio-wide positions from self.positions dict.

        Returns:
            Dict of asset -> PortfolioPosition
        """
        if not ENHANCEMENTS_AVAILABLE:
            return {}

        portfolio = {}
        current_value = sum(inst.current_capital for inst in self.instances.values())

        for asset, quantity in self.positions.items():
            if quantity > 0:
                # Mock price for now (TODO: get real prices)
                mock_price = 100.0
                if asset in ["BTC", "WBTC"]:
                    mock_price = 64000.0
                elif asset in ["ETH", "WETH"]:
                    mock_price = 3000.0
                elif asset == "SOL":
                    mock_price = 120.0

                value_usd = quantity * mock_price

                portfolio[asset] = PortfolioPosition(
                    asset=asset,
                    total_quantity=quantity,
                    total_value_usd=value_usd,
                    chains={},  # TODO: Track per-chain breakdown
                    percentage_of_portfolio=value_usd / current_value if current_value > 0 else 0,
                    cost_basis=value_usd,  # Mock for now
                    unrealized_pnl=0.0  # Mock for now
                )

        return portfolio
