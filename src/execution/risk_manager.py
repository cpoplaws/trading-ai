"""
Risk Management System for trading operations.
Part of Phase 2: Trading System completion.
"""

import logging
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass
import numpy as np

logger = logging.getLogger(__name__)

@dataclass
class RiskLimits:
    """Risk management limits and parameters."""
    max_position_size: float = 0.1  # 10% max position size
    max_portfolio_exposure: float = 0.8  # 80% max exposure
    max_daily_loss: float = 0.05  # 5% max daily loss
    max_total_loss: float = 0.20  # 20% max total portfolio loss
    min_confidence: float = 0.6  # 60% minimum signal confidence
    max_positions: int = 20  # Maximum number of open positions
    stop_loss_pct: float = 0.08  # 8% stop loss
    take_profit_pct: float = 0.15  # 15% take profit
    max_correlation: float = 0.7  # Maximum correlation between positions

@dataclass
class RiskCheck:
    """Result of a risk assessment."""
    approved: bool
    risk_score: float
    warnings: List[str]
    blocking_issues: List[str]
    recommended_size: float
    reason: str

class RiskManager:
    """
    Comprehensive risk management system for trading operations.
    """
    
    def __init__(self, portfolio_tracker, limits: Optional[RiskLimits] = None):
        """
        Initialize risk manager.
        
        Args:
            portfolio_tracker: Portfolio tracker instance
            limits: Risk limits (uses defaults if None)
        """
        self.portfolio_tracker = portfolio_tracker
        self.limits = limits or RiskLimits()
        self.risk_log = []
        
        logger.info("Risk manager initialized")
    
    def check_trade_risk(self, symbol: str, side: str, proposed_size: float, 
                        confidence: float, current_price: float) -> RiskCheck:
        """
        Comprehensive risk check for a proposed trade.
        
        Args:
            symbol: Stock symbol
            side: 'buy' or 'sell'
            proposed_size: Proposed position size (as fraction of portfolio)
            confidence: Signal confidence (0-1)
            current_price: Current stock price
            
        Returns:
            RiskCheck with approval status and recommendations
        """
        warnings = []
        blocking_issues = []
        risk_score = 0.0
        
        try:
            # Get current portfolio state
            metrics = self.portfolio_tracker.calculate_portfolio_metrics()
            positions = self.portfolio_tracker.get_current_positions()
            
            # 1. Check signal confidence
            if confidence < self.limits.min_confidence:
                blocking_issues.append(f"Signal confidence {confidence:.2f} below minimum {self.limits.min_confidence}")
                risk_score += 30
            
            # 2. Check position size limits
            if proposed_size > self.limits.max_position_size:
                blocking_issues.append(f"Position size {proposed_size:.1%} exceeds maximum {self.limits.max_position_size:.1%}")
                risk_score += 25
            elif proposed_size > self.limits.max_position_size * 0.8:
                warnings.append(f"Position size {proposed_size:.1%} approaching maximum")
                risk_score += 10
            
            # 3. Check portfolio exposure
            current_exposure = metrics.exposure_pct / 100
            new_exposure = current_exposure + proposed_size
            
            if new_exposure > self.limits.max_portfolio_exposure:
                blocking_issues.append(f"Total exposure would be {new_exposure:.1%}, exceeds maximum {self.limits.max_portfolio_exposure:.1%}")
                risk_score += 20
            elif new_exposure > self.limits.max_portfolio_exposure * 0.9:
                warnings.append(f"Total exposure approaching maximum ({new_exposure:.1%})")
                risk_score += 5
            
            # 4. Check daily loss limits
            if metrics.day_pnl_pct < -self.limits.max_daily_loss * 100:
                blocking_issues.append(f"Daily loss {metrics.day_pnl_pct:.1f}% exceeds maximum {self.limits.max_daily_loss:.1%}")
                risk_score += 35
            elif metrics.day_pnl_pct < -self.limits.max_daily_loss * 80:
                warnings.append(f"Daily loss {metrics.day_pnl_pct:.1f}% approaching limit")
                risk_score += 15
            
            # 5. Check total portfolio loss
            if metrics.total_pnl_pct < -self.limits.max_total_loss * 100:
                blocking_issues.append(f"Total loss {metrics.total_pnl_pct:.1f}% exceeds maximum {self.limits.max_total_loss:.1%}")
                risk_score += 40
            
            # 6. Check number of positions
            if len(positions) >= self.limits.max_positions:
                blocking_issues.append(f"Already at maximum positions ({len(positions)})")
                risk_score += 15
            elif len(positions) >= self.limits.max_positions * 0.9:
                warnings.append(f"Approaching maximum positions ({len(positions)}/{self.limits.max_positions})")
                risk_score += 5
            
            # 7. Check if position already exists
            existing_position = None
            for pos in positions:
                if pos.symbol == symbol:
                    existing_position = pos
                    break
            
            if existing_position and side == 'buy':
                warnings.append(f"Adding to existing {symbol} position")
                risk_score += 5
            
            # 8. Calculate recommended size
            recommended_size = self._calculate_recommended_size(
                proposed_size, confidence, current_exposure, risk_score
            )
            
            # 9. Sector concentration check (simplified)
            sector_exposure = self._check_sector_concentration(symbol, positions, proposed_size)
            if sector_exposure > 0.3:  # 30% sector limit
                warnings.append(f"High sector concentration: {sector_exposure:.1%}")
                risk_score += 10
            
            # Final approval decision
            approved = len(blocking_issues) == 0 and risk_score < 50
            
            # Create reason
            if not approved:
                reason = f"Trade rejected: {', '.join(blocking_issues)}"
            elif warnings:
                reason = f"Trade approved with warnings: {', '.join(warnings)}"
            else:
                reason = "Trade approved - all risk checks passed"
            
            risk_check = RiskCheck(
                approved=approved,
                risk_score=risk_score,
                warnings=warnings,
                blocking_issues=blocking_issues,
                recommended_size=recommended_size,
                reason=reason
            )
            
            # Log the risk check
            self._log_risk_check(symbol, side, proposed_size, risk_check)
            
            return risk_check
            
        except Exception as e:
            logger.error(f"Error in risk check: {str(e)}")
            return RiskCheck(
                approved=False,
                risk_score=100,
                warnings=[],
                blocking_issues=[f"Risk check error: {str(e)}"],
                recommended_size=0,
                reason="Risk check failed due to error"
            )
    
    def _calculate_recommended_size(self, proposed_size: float, confidence: float, 
                                  current_exposure: float, risk_score: float) -> float:
        """
        Calculate recommended position size based on risk factors.
        
        Args:
            proposed_size: Originally proposed size
            confidence: Signal confidence
            current_exposure: Current portfolio exposure
            risk_score: Calculated risk score
            
        Returns:
            Recommended position size
        """
        # Start with proposed size
        recommended = proposed_size
        
        # Adjust for confidence
        confidence_multiplier = min(confidence / self.limits.min_confidence, 1.0)
        recommended *= confidence_multiplier
        
        # Adjust for current exposure
        if current_exposure > 0.6:  # If already high exposure
            recommended *= 0.7
        
        # Adjust for risk score
        if risk_score > 30:
            recommended *= 0.5
        elif risk_score > 15:
            recommended *= 0.8
        
        # Ensure within limits
        recommended = min(recommended, self.limits.max_position_size)
        recommended = max(recommended, 0)
        
        return recommended
    
    def _check_sector_concentration(self, symbol: str, positions: List, 
                                  proposed_size: float) -> float:
        """
        Check sector concentration (simplified implementation).
        
        Args:
            symbol: New symbol
            positions: Current positions
            proposed_size: Proposed position size
            
        Returns:
            Estimated sector exposure
        """
        # Simplified sector mapping (in reality, would use proper sector data)
        tech_stocks = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'META', 'NVDA']
        financial_stocks = ['JPM', 'BAC', 'WFC', 'C', 'GS']
        
        # Calculate current sector exposure
        sector_exposure = 0
        
        if symbol in tech_stocks:
            # Add exposure from existing tech positions
            for pos in positions:
                if pos.symbol in tech_stocks:
                    sector_exposure += abs(pos.market_value) / 10000  # Assuming 10k portfolio
        
        # Add proposed position
        sector_exposure += proposed_size
        
        return sector_exposure
    
    def check_stop_loss(self, positions: List) -> List[Dict]:
        """
        Check positions for stop-loss triggers.
        
        Args:
            positions: List of current positions
            
        Returns:
            List of positions that should be closed
        """
        stop_loss_positions = []
        
        for pos in positions:
            loss_pct = pos.unrealized_pnl_pct
            
            if loss_pct <= -self.limits.stop_loss_pct * 100:
                stop_loss_positions.append({
                    'symbol': pos.symbol,
                    'reason': 'stop_loss',
                    'current_loss': loss_pct,
                    'stop_loss_limit': -self.limits.stop_loss_pct * 100,
                    'action': 'close_position'
                })
        
        return stop_loss_positions
    
    def check_take_profit(self, positions: List) -> List[Dict]:
        """
        Check positions for take-profit triggers.
        
        Args:
            positions: List of current positions
            
        Returns:
            List of positions that should be partially closed
        """
        take_profit_positions = []
        
        for pos in positions:
            profit_pct = pos.unrealized_pnl_pct
            
            if profit_pct >= self.limits.take_profit_pct * 100:
                take_profit_positions.append({
                    'symbol': pos.symbol,
                    'reason': 'take_profit',
                    'current_profit': profit_pct,
                    'take_profit_limit': self.limits.take_profit_pct * 100,
                    'action': 'partial_close',
                    'close_percentage': 0.5  # Close 50% of position
                })
        
        return take_profit_positions
    
    def _log_risk_check(self, symbol: str, side: str, size: float, check: RiskCheck) -> None:
        """
        Log risk check results.
        
        Args:
            symbol: Stock symbol
            side: Trade side
            size: Position size
            check: Risk check result
        """
        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'symbol': symbol,
            'side': side,
            'proposed_size': size,
            'approved': check.approved,
            'risk_score': check.risk_score,
            'recommended_size': check.recommended_size,
            'warnings': len(check.warnings),
            'blocking_issues': len(check.blocking_issues),
            'reason': check.reason
        }
        
        self.risk_log.append(log_entry)
        
        # Log to file
        if check.approved:
            logger.info(f"Risk check APPROVED for {symbol} {side}: {check.reason}")
        else:
            logger.warning(f"Risk check REJECTED for {symbol} {side}: {check.reason}")
    
    def get_risk_summary(self) -> Dict:
        """
        Get summary of current risk status.
        
        Returns:
            Risk summary dictionary
        """
        try:
            metrics = self.portfolio_tracker.calculate_portfolio_metrics()
            positions = self.portfolio_tracker.get_current_positions()
            
            # Check for risk violations
            violations = []
            if metrics.exposure_pct > self.limits.max_portfolio_exposure * 100:
                violations.append("Portfolio exposure limit exceeded")
            if metrics.day_pnl_pct < -self.limits.max_daily_loss * 100:
                violations.append("Daily loss limit exceeded")
            if len(positions) > self.limits.max_positions:
                violations.append("Maximum positions exceeded")
            
            # Count recent rejections
            recent_rejections = sum(1 for log in self.risk_log[-10:] if not log['approved'])
            
            summary = {
                'current_exposure': f"{metrics.exposure_pct:.1f}%",
                'exposure_limit': f"{self.limits.max_portfolio_exposure * 100:.1f}%",
                'daily_pnl': f"{metrics.day_pnl_pct:.1f}%",
                'daily_limit': f"{-self.limits.max_daily_loss * 100:.1f}%",
                'total_pnl': f"{metrics.total_pnl_pct:.1f}%",
                'total_limit': f"{-self.limits.max_total_loss * 100:.1f}%",
                'positions': f"{len(positions)}/{self.limits.max_positions}",
                'violations': violations,
                'recent_rejections': recent_rejections,
                'risk_status': 'SAFE' if not violations else 'WARNING'
            }
            
            return summary
            
        except Exception as e:
            logger.error(f"Error generating risk summary: {str(e)}")
            return {'error': str(e)}
    
    def print_risk_status(self) -> None:
        """
        Print formatted risk status to console.
        """
        try:
            summary = self.get_risk_summary()
            positions = self.portfolio_tracker.get_current_positions()
            
            print("\n" + "="*60)
            print("⚠️  RISK MANAGEMENT STATUS")
            print("="*60)
            print(f"Risk Status:      {summary['risk_status']}")
            print(f"Exposure:         {summary['current_exposure']} / {summary['exposure_limit']}")
            print(f"Daily P&L:        {summary['daily_pnl']} / {summary['daily_limit']}")
            print(f"Total P&L:        {summary['total_pnl']} / {summary['total_limit']}")
            print(f"Positions:        {summary['positions']}")
            
            if summary['violations']:
                print(f"\n🚨 VIOLATIONS:")
                for violation in summary['violations']:
                    print(f"  - {violation}")
            
            # Check stop losses and take profits
            stop_losses = self.check_stop_loss(positions)
            take_profits = self.check_take_profit(positions)
            
            if stop_losses:
                print(f"\n🛑 STOP LOSS TRIGGERS:")
                for sl in stop_losses:
                    print(f"  - {sl['symbol']}: {sl['current_loss']:.1f}%")
            
            if take_profits:
                print(f"\n💰 TAKE PROFIT TRIGGERS:")
                for tp in take_profits:
                    print(f"  - {tp['symbol']}: {tp['current_profit']:.1f}%")
            
            print("="*60)
            
        except Exception as e:
            logger.error(f"Error printing risk status: {str(e)}")

if __name__ == "__main__":
    # Test risk manager
    from portfolio_tracker import PortfolioTracker
    from broker_interface import create_broker
    
    print("⚠️  Testing Risk Manager...")
    
    # Create broker and portfolio tracker
    broker = create_broker(mock=True)
    tracker = PortfolioTracker(broker, initial_capital=10000)
    
    # Create risk manager
    risk_mgr = RiskManager(tracker)
    
    # Test risk check
    risk_check = risk_mgr.check_trade_risk('AAPL', 'buy', 0.05, 0.8, 150.0)
    
    print(f"Trade approved: {risk_check.approved}")
    print(f"Risk score: {risk_check.risk_score}")
    print(f"Reason: {risk_check.reason}")
    
    # Print risk status
    risk_mgr.print_risk_status()
    
    print("\n🎉 Risk manager test complete!")
