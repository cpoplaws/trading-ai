"""
Advanced portfolio optimization with Kelly Criterion and mean reversion strategies.
"""
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Optional
import scipy.optimize as opt
import logging
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

class PortfolioOptimizer:
    """
    Advanced portfolio optimization using Kelly Criterion, mean reversion, and modern portfolio theory.
    """
    
    def __init__(self, symbols: List[str], lookback_period: int = 252):
        """
        Initialize portfolio optimizer.
        
        Args:
            symbols: List of trading symbols
            lookback_period: Days to look back for optimization
        """
        self.symbols = symbols
        self.lookback_period = lookback_period
        self.returns_data = {}
        self.current_prices = {}
        
    def calculate_kelly_criterion(self, returns: pd.Series, confidence: float = 0.6) -> float:
        """
        Calculate Kelly Criterion position size.
        
        Args:
            returns: Historical returns series
            confidence: Model confidence (0-1)
            
        Returns:
            Optimal position size (0-1)
        """
        try:
            if len(returns) < 10:
                return 0.1  # Default small position
            
            # Calculate win rate and average win/loss
            positive_returns = returns[returns > 0]
            negative_returns = returns[returns < 0]
            
            if len(positive_returns) == 0 or len(negative_returns) == 0:
                return 0.1  # Default if no wins or losses
            
            win_rate = len(positive_returns) / len(returns)
            avg_win = positive_returns.mean()
            avg_loss = abs(negative_returns.mean())
            
            if avg_loss == 0:
                return 0.1  # Avoid division by zero
            
            # Kelly formula: f = (bp - q) / b
            # where b = avg_win/avg_loss, p = win_rate, q = 1-win_rate
            b = avg_win / avg_loss
            p = win_rate * confidence  # Adjust for model confidence
            q = 1 - p
            
            kelly_fraction = (b * p - q) / b
            
            # Apply safety constraints
            kelly_fraction = max(0, min(kelly_fraction, 0.25))  # Cap at 25%
            
            logger.info(f"Kelly Criterion: {kelly_fraction:.3f} (win_rate: {win_rate:.3f}, b: {b:.3f})")
            return kelly_fraction
            
        except Exception as e:
            logger.error(f"Error calculating Kelly Criterion: {e}")
            return 0.1  # Conservative default
    
    def detect_mean_reversion_opportunities(self, prices: pd.Series, 
                                          lookback: int = 20, 
                                          threshold: float = 2.0) -> Dict:
        """
        Detect mean reversion trading opportunities.
        
        Args:
            prices: Price series
            lookback: Lookback period for mean calculation
            threshold: Z-score threshold for signals
            
        Returns:
            Mean reversion analysis results
        """
        try:
            if len(prices) < lookback + 10:
                return {'signal': 'HOLD', 'z_score': 0, 'confidence': 0}
            
            # Calculate rolling mean and std
            rolling_mean = prices.rolling(lookback).mean()
            rolling_std = prices.rolling(lookback).std()
            
            # Calculate Z-score
            current_price = prices.iloc[-1]
            recent_mean = rolling_mean.iloc[-1]
            recent_std = rolling_std.iloc[-1]
            
            if recent_std == 0:
                z_score = 0
            else:
                z_score = (current_price - recent_mean) / recent_std
            
            # Generate signals
            if z_score > threshold:
                signal = 'SELL'  # Price too high, expect reversion
                confidence = min(abs(z_score) / threshold, 1.0) * 0.8
            elif z_score < -threshold:
                signal = 'BUY'   # Price too low, expect reversion
                confidence = min(abs(z_score) / threshold, 1.0) * 0.8
            else:
                signal = 'HOLD'
                confidence = 0.3
            
            # Calculate additional metrics
            volatility = rolling_std.iloc[-10:].mean() / recent_mean  # Relative volatility
            momentum = (prices.iloc[-1] / prices.iloc[-5] - 1) * 100  # 5-day momentum
            
            return {
                'signal': signal,
                'z_score': z_score,
                'confidence': confidence,
                'current_price': current_price,
                'mean_price': recent_mean,
                'volatility': volatility,
                'momentum': momentum,
                'threshold_used': threshold
            }
            
        except Exception as e:
            logger.error(f"Error in mean reversion analysis: {e}")
            return {'signal': 'HOLD', 'z_score': 0, 'confidence': 0}
    
    def optimize_portfolio_weights(self, returns_matrix: pd.DataFrame, 
                                 target_return: Optional[float] = None) -> Dict:
        """
        Optimize portfolio weights using Modern Portfolio Theory.
        
        Args:
            returns_matrix: Matrix of asset returns
            target_return: Target portfolio return (optional)
            
        Returns:
            Optimization results with weights and metrics
        """
        try:
            if returns_matrix.empty or len(returns_matrix.columns) < 2:
                return {'weights': {}, 'expected_return': 0, 'volatility': 0}
            
            n_assets = len(returns_matrix.columns)
            
            # Calculate expected returns and covariance matrix
            expected_returns = returns_matrix.mean() * 252  # Annualized
            cov_matrix = returns_matrix.cov() * 252  # Annualized
            
            # Define optimization functions
            def portfolio_return(weights):
                return np.sum(expected_returns * weights)
            
            def portfolio_volatility(weights):
                return np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
            
            def sharpe_ratio(weights):
                ret = portfolio_return(weights)
                vol = portfolio_volatility(weights)
                return -ret / vol if vol > 0 else 0  # Negative for minimization
            
            # Constraints
            constraints = [
                {'type': 'eq', 'fun': lambda x: np.sum(x) - 1}  # Weights sum to 1
            ]
            
            if target_return is not None:
                constraints.append({
                    'type': 'eq', 
                    'fun': lambda x: portfolio_return(x) - target_return
                })
            
            # Bounds (0 to 40% per asset)
            bounds = tuple((0, 0.4) for _ in range(n_assets))
            
            # Initial guess (equal weights)
            initial_guess = np.array([1.0 / n_assets] * n_assets)
            
            # Optimize for maximum Sharpe ratio
            result = opt.minimize(
                sharpe_ratio,
                initial_guess,
                method='SLSQP',
                bounds=bounds,
                constraints=constraints
            )
            
            if result.success:
                optimal_weights = result.x
                opt_return = portfolio_return(optimal_weights)
                opt_volatility = portfolio_volatility(optimal_weights)
                opt_sharpe = -result.fun if result.fun != 0 else 0
                
                weights_dict = {
                    symbol: weight 
                    for symbol, weight in zip(returns_matrix.columns, optimal_weights)
                    if weight > 0.01  # Only include meaningful weights
                }
                
                return {
                    'weights': weights_dict,
                    'expected_return': opt_return,
                    'volatility': opt_volatility,
                    'sharpe_ratio': opt_sharpe,
                    'optimization_success': True
                }
            else:
                logger.warning("Portfolio optimization failed, using equal weights")
                equal_weights = {symbol: 1.0/n_assets for symbol in returns_matrix.columns}
                return {
                    'weights': equal_weights,
                    'expected_return': expected_returns.mean(),
                    'volatility': cov_matrix.values.diagonal().mean() ** 0.5,
                    'sharpe_ratio': 0,
                    'optimization_success': False
                }
                
        except Exception as e:
            logger.error(f"Error in portfolio optimization: {e}")
            return {'weights': {}, 'expected_return': 0, 'volatility': 0}
    
    def calculate_risk_parity_weights(self, returns_matrix: pd.DataFrame) -> Dict:
        """
        Calculate risk parity portfolio weights.
        
        Args:
            returns_matrix: Matrix of asset returns
            
        Returns:
            Risk parity weights and metrics
        """
        try:
            if returns_matrix.empty:
                return {'weights': {}}
            
            # Calculate covariance matrix
            cov_matrix = returns_matrix.cov() * 252  # Annualized
            
            n_assets = len(returns_matrix.columns)
            
            # Risk parity optimization
            def risk_budget_objective(weights):
                weights = np.array(weights)
                portfolio_vol = np.sqrt(np.dot(weights, np.dot(cov_matrix, weights)))
                
                # Marginal risk contribution
                marginal_contrib = np.dot(cov_matrix, weights) / portfolio_vol
                contrib = weights * marginal_contrib
                
                # We want equal risk contribution (1/n each)
                target_contrib = portfolio_vol / n_assets
                return np.sum((contrib - target_contrib) ** 2)
            
            # Constraints
            constraints = [
                {'type': 'eq', 'fun': lambda x: np.sum(x) - 1}
            ]
            bounds = tuple((0.01, 0.8) for _ in range(n_assets))
            initial_guess = np.array([1.0 / n_assets] * n_assets)
            
            result = opt.minimize(
                risk_budget_objective,
                initial_guess,
                method='SLSQP',
                bounds=bounds,
                constraints=constraints
            )
            
            if result.success:
                rp_weights = result.x
                weights_dict = {
                    symbol: weight 
                    for symbol, weight in zip(returns_matrix.columns, rp_weights)
                }
                
                return {
                    'weights': weights_dict,
                    'type': 'risk_parity',
                    'success': True
                }
            else:
                # Fallback to inverse volatility weights
                volatilities = np.sqrt(np.diag(cov_matrix))
                inv_vol_weights = (1 / volatilities) / np.sum(1 / volatilities)
                
                weights_dict = {
                    symbol: weight 
                    for symbol, weight in zip(returns_matrix.columns, inv_vol_weights)
                }
                
                return {
                    'weights': weights_dict,
                    'type': 'inverse_volatility',
                    'success': False
                }
                
        except Exception as e:
            logger.error(f"Error calculating risk parity weights: {e}")
            return {'weights': {}}
    
    def generate_portfolio_recommendations(self, prices_data: Dict[str, pd.Series],
                                         signals_data: Dict[str, Dict]) -> Dict:
        """
        Generate comprehensive portfolio recommendations.
        
        Args:
            prices_data: Price data for each symbol
            signals_data: Trading signals for each symbol
            
        Returns:
            Portfolio recommendations with position sizes and rationale
        """
        try:
            logger.info("Generating portfolio recommendations")
            
            # Prepare returns data
            returns_data = {}
            for symbol, prices in prices_data.items():
                if len(prices) > 1:
                    returns_data[symbol] = prices.pct_change().dropna()
            
            if not returns_data:
                return {'error': 'No valid price data'}
            
            returns_matrix = pd.DataFrame(returns_data)
            
            # 1. Portfolio Optimization
            optimal_weights = self.optimize_portfolio_weights(returns_matrix)
            risk_parity_weights = self.calculate_risk_parity_weights(returns_matrix)
            
            # 2. Individual Asset Analysis
            recommendations = {}
            
            for symbol in self.symbols:
                if symbol not in prices_data or symbol not in signals_data:
                    continue
                
                prices = prices_data[symbol]
                signals = signals_data[symbol]
                returns = returns_data.get(symbol, pd.Series())
                
                # Kelly Criterion position sizing
                model_confidence = signals.get('confidence', 0.5)
                kelly_size = self.calculate_kelly_criterion(returns, model_confidence)
                
                # Mean reversion analysis
                mean_reversion = self.detect_mean_reversion_opportunities(prices)
                
                # Combine signals
                ml_signal = signals.get('signal', 'HOLD')
                mr_signal = mean_reversion.get('signal', 'HOLD')
                
                # Signal consensus
                if ml_signal == mr_signal:
                    final_signal = ml_signal
                    consensus_confidence = 0.8
                elif ml_signal == 'HOLD' or mr_signal == 'HOLD':
                    final_signal = ml_signal if ml_signal != 'HOLD' else mr_signal
                    consensus_confidence = 0.5
                else:
                    final_signal = 'HOLD'  # Conflicting signals
                    consensus_confidence = 0.3
                
                # Position sizing
                base_weight = optimal_weights['weights'].get(symbol, 0.1)
                kelly_weight = kelly_size
                
                if final_signal == 'BUY':
                    recommended_weight = min(base_weight * 1.5, kelly_weight * 2, 0.3)
                elif final_signal == 'SELL':
                    recommended_weight = max(base_weight * 0.5, 0.05)
                else:
                    recommended_weight = base_weight
                
                recommendations[symbol] = {
                    'signal': final_signal,
                    'confidence': consensus_confidence,
                    'recommended_weight': recommended_weight,
                    'kelly_size': kelly_size,
                    'optimal_weight': base_weight,
                    'ml_signal': ml_signal,
                    'mean_reversion_signal': mr_signal,
                    'z_score': mean_reversion.get('z_score', 0),
                    'current_price': prices.iloc[-1] if len(prices) > 0 else 0,
                    'volatility': returns.std() * np.sqrt(252) if len(returns) > 0 else 0
                }
            
            # 3. Portfolio-level metrics
            total_weight = sum(rec['recommended_weight'] for rec in recommendations.values())
            if total_weight > 0:
                # Normalize weights
                for symbol in recommendations:
                    recommendations[symbol]['normalized_weight'] = (
                        recommendations[symbol]['recommended_weight'] / total_weight
                    )
            
            portfolio_summary = {
                'total_symbols': len(recommendations),
                'buy_signals': len([r for r in recommendations.values() if r['signal'] == 'BUY']),
                'sell_signals': len([r for r in recommendations.values() if r['signal'] == 'SELL']),
                'hold_signals': len([r for r in recommendations.values() if r['signal'] == 'HOLD']),
                'avg_confidence': np.mean([r['confidence'] for r in recommendations.values()]),
                'portfolio_expected_return': optimal_weights.get('expected_return', 0),
                'portfolio_volatility': optimal_weights.get('volatility', 0),
                'portfolio_sharpe': optimal_weights.get('sharpe_ratio', 0)
            }
            
            return {
                'timestamp': datetime.now(),
                'recommendations': recommendations,
                'portfolio_summary': portfolio_summary,
                'optimal_weights': optimal_weights,
                'risk_parity_weights': risk_parity_weights
            }
            
        except Exception as e:
            logger.error(f"Error generating portfolio recommendations: {e}")
            return {'error': str(e)}

# Example usage
if __name__ == "__main__":
    import yfinance as yf
    
    # Demo portfolio optimization
    print("🎯 Portfolio Optimization Demo")
    print("=" * 40)
    
    symbols = ['AAPL', 'MSFT', 'GOOGL', 'TSLA']
    optimizer = PortfolioOptimizer(symbols)
    
    # Fetch sample data
    prices_data = {}
    signals_data = {}
    
    for symbol in symbols:
        try:
            ticker = yf.Ticker(symbol)
            hist = ticker.history(period="3mo")
            if not hist.empty:
                prices_data[symbol] = hist['Close']
                # Simulate signals
                signals_data[symbol] = {
                    'signal': np.random.choice(['BUY', 'SELL', 'HOLD']),
                    'confidence': np.random.uniform(0.5, 0.9)
                }
        except:
            continue
    
    if prices_data:
        recommendations = optimizer.generate_portfolio_recommendations(prices_data, signals_data)
        
        if 'error' not in recommendations:
            print(f"\n📊 Portfolio Recommendations:")
            
            for symbol, rec in recommendations['recommendations'].items():
                print(f"\n{symbol}:")
                print(f"  Signal: {rec['signal']} (confidence: {rec['confidence']:.3f})")
                print(f"  Recommended Weight: {rec['normalized_weight']:.3f}")
                print(f"  Kelly Size: {rec['kelly_size']:.3f}")
                print(f"  Mean Reversion Z-Score: {rec['z_score']:.2f}")
            
            summary = recommendations['portfolio_summary']
            print(f"\n📈 Portfolio Summary:")
            print(f"  Buy Signals: {summary['buy_signals']}")
            print(f"  Sell Signals: {summary['sell_signals']}")
            print(f"  Expected Return: {summary['portfolio_expected_return']:.2%}")
            print(f"  Volatility: {summary['portfolio_volatility']:.2%}")
            print(f"  Sharpe Ratio: {summary['portfolio_sharpe']:.3f}")
        else:
            print(f"❌ Error: {recommendations['error']}")
    else:
        print("❌ No data available for demo")
    
    print("\n✅ Portfolio optimization demo completed!")