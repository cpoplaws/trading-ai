"""
Funding rate arbitrage strategy for perpetual futures.
Long spot, short perp when funding is positive.
"""
import logging
from typing import Dict, List, Optional, Any
from datetime import datetime
import pandas as pd

logger = logging.getLogger(__name__)


class FundingRateArbitrage:
    """
    Funding rate arbitrage strategy.
    Profit from funding rate payments by taking offsetting positions.
    """
    
    def __init__(self, min_rate_threshold: float = 0.01, position_size_pct: float = 5.0):
        """
        Initialize funding rate arbitrage strategy.
        
        Args:
            min_rate_threshold: Minimum funding rate to trigger (e.g., 0.01 = 1%)
            position_size_pct: Position size as percentage of portfolio
        """
        self.min_rate_threshold = min_rate_threshold
        self.position_size_pct = position_size_pct
        
        logger.info(f"Funding rate arbitrage initialized (threshold={min_rate_threshold})")
    
    def analyze_opportunity(self, symbol: str, funding_rate: float,
                           spot_price: float, perp_price: float) -> Dict[str, Any]:
        """
        Analyze funding rate arbitrage opportunity.
        
        Args:
            symbol: Trading symbol (e.g., 'BTC')
            funding_rate: Current 8-hour funding rate
            spot_price: Spot market price
            perp_price: Perpetual futures price
            
        Returns:
            Analysis dictionary with signal and metrics
        """
        try:
            # Calculate basis (premium/discount)
            basis = (perp_price - spot_price) / spot_price
            basis_bps = basis * 10000
            
            # Annualized funding rate (3 payments per day)
            annual_funding = funding_rate * 3 * 365 * 100
            
            # Determine signal
            signal = 'HOLD'
            confidence = 0.0
            
            if funding_rate > self.min_rate_threshold:
                # Positive funding: shorts pay longs
                # Strategy: Long spot, short perp
                signal = 'LONG_SPOT_SHORT_PERP'
                confidence = min(funding_rate / 0.05, 1.0)  # Max at 5% funding
                
            elif funding_rate < -self.min_rate_threshold:
                # Negative funding: longs pay shorts
                # Strategy: Short spot, long perp
                signal = 'SHORT_SPOT_LONG_PERP'
                confidence = min(abs(funding_rate) / 0.05, 1.0)
            
            # Calculate expected profit (excluding basis risk)
            expected_profit_8h = abs(funding_rate) * 100
            expected_profit_daily = expected_profit_8h * 3
            expected_profit_annual = expected_profit_daily * 365
            
            return {
                'symbol': symbol,
                'signal': signal,
                'confidence': confidence,
                'funding_rate': funding_rate,
                'funding_rate_percent': funding_rate * 100,
                'annual_funding_percent': annual_funding,
                'spot_price': spot_price,
                'perp_price': perp_price,
                'basis': basis,
                'basis_bps': basis_bps,
                'expected_profit_8h': expected_profit_8h,
                'expected_profit_daily': expected_profit_daily,
                'expected_profit_annual': expected_profit_annual,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error analyzing funding rate opportunity: {e}")
            return {}
    
    def scan_opportunities(self, market_data: List[Dict]) -> List[Dict]:
        """
        Scan multiple markets for funding rate opportunities.
        
        Args:
            market_data: List of market data dictionaries with funding rates
            
        Returns:
            List of opportunities sorted by expected profit
        """
        opportunities = []
        
        try:
            for data in market_data:
                analysis = self.analyze_opportunity(
                    symbol=data['symbol'],
                    funding_rate=data['funding_rate'],
                    spot_price=data['spot_price'],
                    perp_price=data['perp_price']
                )
                
                if analysis and analysis['signal'] != 'HOLD':
                    opportunities.append(analysis)
            
            # Sort by expected annual profit
            opportunities.sort(key=lambda x: x['expected_profit_annual'], reverse=True)
            
            logger.info(f"Found {len(opportunities)} funding rate opportunities")
            return opportunities
            
        except Exception as e:
            logger.error(f"Error scanning opportunities: {e}")
            return []
    
    def calculate_position_size(self, portfolio_value: float, leverage: float = 1.0) -> float:
        """
        Calculate position size for strategy.
        
        Args:
            portfolio_value: Total portfolio value in USD
            leverage: Leverage to use (default 1x)
            
        Returns:
            Position size in USD
        """
        return (portfolio_value * self.position_size_pct / 100) * leverage
    
    def estimate_risk(self, position_size: float, basis: float, volatility: float) -> Dict:
        """
        Estimate risks for funding rate arbitrage position.
        
        Args:
            position_size: Position size in USD
            basis: Current basis (perp - spot price difference)
            volatility: Asset volatility
            
        Returns:
            Risk metrics dictionary
        """
        try:
            # Basis risk: risk that basis widens unfavorably
            basis_risk = abs(basis) * position_size
            
            # Volatility risk (simplified)
            vol_risk = volatility * position_size * 0.1  # 10% of daily vol
            
            # Total risk
            total_risk = basis_risk + vol_risk
            
            # Risk-reward ratio
            expected_daily_return = 0.01  # Simplified 1% daily from funding
            risk_reward = (expected_daily_return * position_size) / total_risk if total_risk > 0 else 0
            
            return {
                'basis_risk_usd': basis_risk,
                'volatility_risk_usd': vol_risk,
                'total_risk_usd': total_risk,
                'risk_reward_ratio': risk_reward,
                'max_loss_percent': (total_risk / position_size) * 100
            }
            
        except Exception as e:
            logger.error(f"Error estimating risk: {e}")
            return {}
    
    def generate_trade_signals(self, opportunities: List[Dict],
                              portfolio_value: float, max_positions: int = 3) -> List[Dict]:
        """
        Generate trade signals from opportunities.
        
        Args:
            opportunities: List of analyzed opportunities
            portfolio_value: Total portfolio value
            max_positions: Maximum concurrent positions
            
        Returns:
            List of trade signals
        """
        signals = []
        
        try:
            # Take top opportunities up to max_positions
            for opp in opportunities[:max_positions]:
                position_size = self.calculate_position_size(portfolio_value)
                
                signals.append({
                    'symbol': opp['symbol'],
                    'strategy': 'funding_rate_arbitrage',
                    'action': opp['signal'],
                    'position_size_usd': position_size,
                    'confidence': opp['confidence'],
                    'expected_return_annual': opp['expected_profit_annual'],
                    'funding_rate': opp['funding_rate_percent'],
                    'entry_spot_price': opp['spot_price'],
                    'entry_perp_price': opp['perp_price'],
                    'timestamp': datetime.now().isoformat()
                })
            
            return signals
            
        except Exception as e:
            logger.error(f"Error generating trade signals: {e}")
            return []


if __name__ == "__main__":
    # Test funding rate arbitrage
    strategy = FundingRateArbitrage(min_rate_threshold=0.01)
    
    print("=== Funding Rate Arbitrage Test ===")
    
    # Example market data
    market_data = [
        {
            'symbol': 'BTC',
            'funding_rate': 0.015,  # 1.5% per 8 hours
            'spot_price': 50000,
            'perp_price': 50100
        },
        {
            'symbol': 'ETH',
            'funding_rate': 0.008,  # 0.8% per 8 hours
            'spot_price': 3000,
            'perp_price': 3020
        },
        {
            'symbol': 'SOL',
            'funding_rate': -0.005,  # -0.5% per 8 hours
            'spot_price': 150,
            'perp_price': 149
        }
    ]
    
    # Scan opportunities
    opportunities = strategy.scan_opportunities(market_data)
    
    print(f"\nFound {len(opportunities)} opportunities:")
    for opp in opportunities:
        print(f"\n{opp['symbol']}:")
        print(f"  Signal: {opp['signal']}")
        print(f"  Funding Rate: {opp['funding_rate_percent']:.2f}%")
        print(f"  Expected Annual Return: {opp['expected_profit_annual']:.2f}%")
        print(f"  Confidence: {opp['confidence']:.2f}")
    
    # Generate trade signals
    signals = strategy.generate_trade_signals(opportunities, portfolio_value=100000)
    print(f"\nGenerated {len(signals)} trade signals")
    
    print("\n✅ Funding rate arbitrage test completed!")
