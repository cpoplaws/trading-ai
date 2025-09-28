"""
Advanced trading strategies integration module combining all advanced features.
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import logging
import warnings
warnings.filterwarnings('ignore')

# Import all advanced strategy modules
from .portfolio_optimizer import PortfolioOptimizer
from .sentiment_analyzer import SentimentAnalyzer
from .options_strategies import OptionsStrategy
from .enhanced_ml_models import EnhancedMLModels
from .multi_timeframe import MultiTimeframeAnalyzer

logger = logging.getLogger(__name__)

class AdvancedTradingStrategies:
    """
    Comprehensive advanced trading strategies integrating all components.
    """
    
    def __init__(self, symbols: List[str], api_keys: Optional[Dict[str, str]] = None):
        """
        Initialize advanced trading strategies.
        
        Args:
            symbols: List of trading symbols
            api_keys: API keys for various services
        """
        self.symbols = symbols
        self.api_keys = api_keys or {}
        
        # Initialize all strategy components
        self.portfolio_optimizer = PortfolioOptimizer(symbols)
        self.sentiment_analyzer = SentimentAnalyzer(api_keys)
        self.options_strategy = OptionsStrategy()
        self.ml_models = EnhancedMLModels()
        # Multi-timeframe analyzer will be initialized per symbol when needed
        self.multi_timeframe_cache = {}
        
        # Strategy weights for signal aggregation
        self.strategy_weights = {
            'ml_models': 0.30,
            'sentiment': 0.20,
            'multi_timeframe': 0.25,
            'portfolio_optimization': 0.15,
            'options': 0.10
        }
    
    def get_comprehensive_signals(self, symbol: str, market_data: Dict[str, pd.DataFrame],
                                current_price: float, market_outlook: str = 'neutral') -> Dict[str, Any]:
        """
        Generate comprehensive trading signals from all strategy components.
        
        Args:
            symbol: Trading symbol
            market_data: Market data for different timeframes
            current_price: Current market price
            market_outlook: Market outlook ('bullish', 'bearish', 'neutral', 'volatile')
            
        Returns:
            Comprehensive trading signals and analysis
        """
        try:
            logger.info(f"Generating comprehensive signals for {symbol}")
            
            signals = {}
            recommendations = {}
            
            # Get daily data for analysis
            daily_data = market_data.get('1d', market_data.get('daily', pd.DataFrame()))
            
            # 1. Multi-timeframe Analysis
            try:
                if market_data:
                    # Initialize multi-timeframe analyzer for this symbol if not cached
                    if symbol not in self.multi_timeframe_cache:
                        self.multi_timeframe_cache[symbol] = MultiTimeframeAnalyzer(symbol)
                    
                    mtf_analyzer = self.multi_timeframe_cache[symbol]
                    mtf_signals = mtf_analyzer.generate_multi_timeframe_signals(symbol, market_data)
                    if 'error' not in mtf_signals:
                        signals['multi_timeframe'] = mtf_signals
                        logger.info(f"Multi-timeframe signal: {mtf_signals.get('overall_signal', 'N/A')}")
            except Exception as e:
                logger.warning(f"Multi-timeframe analysis failed: {e}")
            
            # 2. Enhanced ML Models
            try:
                # Use daily data for ML models
                if not daily_data.empty:
                    ml_signals = self.ml_models.generate_ml_signals(daily_data)
                    if 'error' not in ml_signals:
                        signals['ml_models'] = ml_signals
                        logger.info(f"ML signal: {ml_signals.get('final_signal', 'N/A')}")
            except Exception as e:
                logger.warning(f"ML models analysis failed: {e}")
            
            # 3. Sentiment Analysis
            try:
                sentiment_signals = self.sentiment_analyzer.aggregate_sentiment_signals(symbol)
                if 'error' not in sentiment_signals:
                    signals['sentiment'] = sentiment_signals
                    logger.info(f"Sentiment signal: {sentiment_signals.get('signal', 'N/A')}")
            except Exception as e:
                logger.warning(f"Sentiment analysis failed: {e}")
            
            # 4. Portfolio Optimization
            try:
                # Create mock signals for portfolio optimization
                mock_signals = {}
                for s in self.symbols:
                    signal_data = signals.get('ml_models', {}) if s == symbol else {'signal': 'HOLD', 'confidence': 0.5}
                    mock_signals[s] = {
                        'signal': signal_data.get('final_signal', signal_data.get('signal', 'HOLD')),
                        'confidence': signal_data.get('confidence', 0.5)
                    }
                
                # Create mock price data (would use real data in production)
                prices_data = {}
                for s in self.symbols:
                    if s == symbol and not daily_data.empty:
                        prices_data[s] = daily_data['close']
                    else:
                        # Mock data
                        dates = pd.date_range(end=datetime.now(), periods=100, freq='D')
                        mock_prices = pd.Series(
                            np.random.normal(current_price, current_price * 0.02, 100),
                            index=dates
                        )
                        prices_data[s] = mock_prices
                
                portfolio_rec = self.portfolio_optimizer.generate_portfolio_recommendations(
                    prices_data, mock_signals
                )
                if 'error' not in portfolio_rec:
                    signals['portfolio_optimization'] = portfolio_rec
                    symbol_rec = portfolio_rec.get('recommendations', {}).get(symbol, {})
                    logger.info(f"Portfolio signal: {symbol_rec.get('signal', 'N/A')}")
            except Exception as e:
                logger.warning(f"Portfolio optimization failed: {e}")
            
            # 5. Options Strategies
            try:
                # Estimate volatility from daily data
                if not daily_data.empty:
                    returns = daily_data['close'].pct_change().dropna()
                    volatility = returns.std() * np.sqrt(252)  # Annualized volatility
                else:
                    volatility = 0.25  # Default volatility
                
                options_opportunities = self.options_strategy.screen_options_opportunities(
                    symbol, current_price, volatility, market_outlook
                )
                if options_opportunities:
                    signals['options'] = {
                        'opportunities': options_opportunities,
                        'best_strategy': options_opportunities[0] if options_opportunities else None,
                        'volatility': volatility
                    }
                    logger.info(f"Options opportunities found: {len(options_opportunities)}")
            except Exception as e:
                logger.warning(f"Options analysis failed: {e}")
            
            # Aggregate all signals
            aggregated_signal = self._aggregate_signals(signals)
            
            # Generate final recommendations
            final_recommendations = self._generate_final_recommendations(
                symbol, signals, aggregated_signal, current_price, market_outlook
            )
            
            return {
                'symbol': symbol,
                'timestamp': datetime.now(),
                'current_price': current_price,
                'market_outlook': market_outlook,
                'individual_signals': signals,
                'aggregated_signal': aggregated_signal,
                'final_recommendations': final_recommendations,
                'strategy_coverage': list(signals.keys()),
                'total_strategies': len(signals)
            }
            
        except Exception as e:
            logger.error(f"Error generating comprehensive signals: {e}")
            return {'error': str(e), 'symbol': symbol}
    
    def _aggregate_signals(self, signals: Dict[str, Any]) -> Dict[str, Any]:
        """Aggregate signals from different strategies using weighted voting."""
        try:
            signal_votes = {'BUY': 0, 'SELL': 0, 'HOLD': 0}
            confidence_sum = 0
            expected_return_sum = 0
            total_weight = 0
            
            strategy_contributions = {}
            
            for strategy, signal_data in signals.items():
                weight = self.strategy_weights.get(strategy, 0.1)
                
                # Extract signal and confidence
                if strategy == 'multi_timeframe':
                    signal = signal_data.get('overall_signal', 'HOLD')
                    confidence = signal_data.get('overall_confidence', 0.5)
                    expected_return = signal_data.get('expected_return', 0)
                elif strategy == 'ml_models':
                    signal = signal_data.get('final_signal', 'HOLD')
                    confidence = signal_data.get('confidence', 0.5)
                    expected_return = signal_data.get('expected_return', 0)
                elif strategy == 'sentiment':
                    signal = signal_data.get('signal', 'HOLD')
                    confidence = signal_data.get('confidence', 0.5)
                    expected_return = 0  # Sentiment doesn't predict returns directly
                elif strategy == 'portfolio_optimization':
                    # Extract signal from recommendations
                    recs = signal_data.get('recommendations', {})
                    symbol_rec = next(iter(recs.values()), {}) if recs else {}
                    signal = symbol_rec.get('signal', 'HOLD')
                    confidence = symbol_rec.get('confidence', 0.5)
                    expected_return = 0
                else:  # options or other
                    continue  # Skip for aggregation
                
                # Weight the vote
                weighted_confidence = confidence * weight
                signal_votes[signal] += weighted_confidence
                confidence_sum += weighted_confidence
                expected_return_sum += expected_return * weight
                total_weight += weight
                
                strategy_contributions[strategy] = {
                    'signal': signal,
                    'confidence': confidence,
                    'weight': weight,
                    'contribution': weighted_confidence
                }
            
            # Determine final signal
            if signal_votes['BUY'] > signal_votes['SELL'] and signal_votes['BUY'] > signal_votes['HOLD']:
                final_signal = 'BUY'
            elif signal_votes['SELL'] > signal_votes['BUY'] and signal_votes['SELL'] > signal_votes['HOLD']:
                final_signal = 'SELL'
            else:
                final_signal = 'HOLD'
            
            # Calculate aggregated confidence
            overall_confidence = confidence_sum / total_weight if total_weight > 0 else 0
            
            # Calculate consensus (how much strategies agree)
            max_votes = max(signal_votes.values())
            total_votes = sum(signal_votes.values())
            consensus = max_votes / total_votes if total_votes > 0 else 0
            
            return {
                'signal': final_signal,
                'confidence': overall_confidence,
                'expected_return': expected_return_sum,
                'consensus': consensus,
                'signal_votes': signal_votes,
                'strategy_contributions': strategy_contributions,
                'strategies_used': len(strategy_contributions)
            }
            
        except Exception as e:
            logger.error(f"Error aggregating signals: {e}")
            return {
                'signal': 'HOLD',
                'confidence': 0,
                'expected_return': 0,
                'consensus': 0,
                'error': str(e)
            }
    
    def _generate_final_recommendations(self, symbol: str, signals: Dict[str, Any], 
                                      aggregated_signal: Dict[str, Any], current_price: float,
                                      market_outlook: str) -> Dict[str, Any]:
        """Generate final trading recommendations."""
        try:
            recommendations = {
                'primary_action': aggregated_signal.get('signal', 'HOLD'),
                'confidence_level': aggregated_signal.get('confidence', 0),
                'expected_return': aggregated_signal.get('expected_return', 0),
                'risk_assessment': 'MEDIUM',  # Default
                'time_horizon': 'SHORT_TERM',  # Default
                'position_sizing': 0.1,  # Default 10%
                'stop_loss': None,
                'take_profit': None,
                'rationale': []
            }
            
            # Adjust position sizing based on confidence and Kelly criterion
            if 'portfolio_optimization' in signals:
                portfolio_data = signals['portfolio_optimization']
                symbol_rec = portfolio_data.get('recommendations', {}).get(symbol, {})
                kelly_size = symbol_rec.get('kelly_size', 0.1)
                recommendations['position_sizing'] = min(kelly_size * 2, 0.25)  # Cap at 25%
            
            # Set stop loss and take profit based on expected return and volatility
            expected_return = aggregated_signal.get('expected_return', 0)
            if abs(expected_return) > 0.01:  # If significant expected return
                if expected_return > 0:
                    recommendations['take_profit'] = current_price * (1 + expected_return * 2)
                    recommendations['stop_loss'] = current_price * (1 - expected_return * 0.5)
                else:
                    recommendations['take_profit'] = current_price * (1 + expected_return * 0.5)
                    recommendations['stop_loss'] = current_price * (1 - expected_return * 2)
            
            # Risk assessment based on consensus and volatility
            consensus = aggregated_signal.get('consensus', 0)
            if consensus > 0.8:
                recommendations['risk_assessment'] = 'LOW'
            elif consensus < 0.4:
                recommendations['risk_assessment'] = 'HIGH'
            
            # Generate rationale
            rationale = []
            for strategy, contrib in aggregated_signal.get('strategy_contributions', {}).items():
                signal = contrib['signal']
                confidence = contrib['confidence']
                weight = contrib['weight']
                rationale.append(
                    f"{strategy.replace('_', ' ').title()}: {signal} "
                    f"(confidence: {confidence:.2f}, weight: {weight:.2f})"
                )
            
            recommendations['rationale'] = rationale
            
            # Add options recommendations if available
            if 'options' in signals:
                options_data = signals['options']
                best_strategy = options_data.get('best_strategy')
                if best_strategy:
                    recommendations['options_opportunity'] = {
                        'strategy': best_strategy.get('strategy', ''),
                        'score': best_strategy.get('recommendation_score', 0),
                        'return_risk_ratio': best_strategy.get('return_on_risk', 0)
                    }
            
            return recommendations
            
        except Exception as e:
            logger.error(f"Error generating final recommendations: {e}")
            return {
                'primary_action': 'HOLD',
                'confidence_level': 0,
                'error': str(e)
            }
    
    def get_portfolio_dashboard(self, market_data: Dict[str, Dict[str, pd.DataFrame]],
                              current_prices: Dict[str, float]) -> Dict[str, Any]:
        """
        Generate comprehensive portfolio dashboard with signals for all symbols.
        
        Args:
            market_data: Market data for all symbols and timeframes
            current_prices: Current prices for all symbols
            
        Returns:
            Portfolio dashboard with signals and recommendations
        """
        try:
            logger.info("Generating portfolio dashboard")
            
            dashboard = {
                'timestamp': datetime.now(),
                'symbols': self.symbols,
                'symbol_signals': {},
                'portfolio_summary': {},
                'market_overview': {}
            }
            
            # Get signals for each symbol
            for symbol in self.symbols:
                try:
                    symbol_data = market_data.get(symbol, {})
                    current_price = current_prices.get(symbol, 100)
                    
                    signals = self.get_comprehensive_signals(
                        symbol, symbol_data, current_price
                    )
                    dashboard['symbol_signals'][symbol] = signals
                    
                except Exception as e:
                    logger.warning(f"Failed to get signals for {symbol}: {e}")
                    dashboard['symbol_signals'][symbol] = {'error': str(e)}
            
            # Generate portfolio summary
            successful_signals = {k: v for k, v in dashboard['symbol_signals'].items() 
                                if 'error' not in v}
            
            if successful_signals:
                # Aggregate portfolio metrics
                buy_signals = sum(1 for s in successful_signals.values() 
                                if s.get('aggregated_signal', {}).get('signal') == 'BUY')
                sell_signals = sum(1 for s in successful_signals.values() 
                                 if s.get('aggregated_signal', {}).get('signal') == 'SELL')
                hold_signals = len(successful_signals) - buy_signals - sell_signals
                
                avg_confidence = np.mean([
                    s.get('aggregated_signal', {}).get('confidence', 0) 
                    for s in successful_signals.values()
                ])
                
                dashboard['portfolio_summary'] = {
                    'total_symbols': len(self.symbols),
                    'signals_generated': len(successful_signals),
                    'buy_signals': buy_signals,
                    'sell_signals': sell_signals,
                    'hold_signals': hold_signals,
                    'average_confidence': avg_confidence,
                    'top_opportunities': self._get_top_opportunities(successful_signals),
                    'risk_assessment': self._assess_portfolio_risk(successful_signals)
                }
            
            return dashboard
            
        except Exception as e:
            logger.error(f"Error generating portfolio dashboard: {e}")
            return {'error': str(e)}
    
    def _get_top_opportunities(self, signals: Dict[str, Any], top_n: int = 5) -> List[Dict]:
        """Get top trading opportunities ranked by confidence and expected return."""
        try:
            opportunities = []
            
            for symbol, signal_data in signals.items():
                agg_signal = signal_data.get('aggregated_signal', {})
                final_rec = signal_data.get('final_recommendations', {})
                
                if agg_signal.get('signal') in ['BUY', 'SELL']:
                    score = (
                        agg_signal.get('confidence', 0) * 0.6 +
                        abs(agg_signal.get('expected_return', 0)) * 10 * 0.4
                    )
                    
                    opportunities.append({
                        'symbol': symbol,
                        'signal': agg_signal.get('signal'),
                        'confidence': agg_signal.get('confidence', 0),
                        'expected_return': agg_signal.get('expected_return', 0),
                        'position_size': final_rec.get('position_sizing', 0.1),
                        'score': score
                    })
            
            # Sort by score and return top N
            opportunities.sort(key=lambda x: x['score'], reverse=True)
            return opportunities[:top_n]
            
        except Exception as e:
            logger.error(f"Error getting top opportunities: {e}")
            return []
    
    def _assess_portfolio_risk(self, signals: Dict[str, Any]) -> str:
        """Assess overall portfolio risk level."""
        try:
            risk_scores = []
            
            for signal_data in signals.values():
                final_rec = signal_data.get('final_recommendations', {})
                risk_assessment = final_rec.get('risk_assessment', 'MEDIUM')
                
                if risk_assessment == 'LOW':
                    risk_scores.append(1)
                elif risk_assessment == 'MEDIUM':
                    risk_scores.append(2)
                else:  # HIGH
                    risk_scores.append(3)
            
            if not risk_scores:
                return 'MEDIUM'
            
            avg_risk = np.mean(risk_scores)
            
            if avg_risk <= 1.5:
                return 'LOW'
            elif avg_risk <= 2.5:
                return 'MEDIUM'
            else:
                return 'HIGH'
            
        except Exception:
            return 'MEDIUM'

# Example usage and demo
if __name__ == "__main__":
    print("🚀 Advanced Trading Strategies Demo")
    print("=" * 50)
    
    # Initialize with demo symbols
    symbols = ['AAPL', 'MSFT', 'GOOGL']
    advanced_strategies = AdvancedTradingStrategies(symbols)
    
    # Mock market data
    dates = pd.date_range(start='2023-01-01', periods=100, freq='D')
    mock_data = {}
    current_prices = {}
    
    for symbol in symbols:
        np.random.seed(hash(symbol) % 2**32)  # Reproducible random data per symbol
        
        # Generate mock price data
        returns = np.random.normal(0.001, 0.02, len(dates))
        price = 150
        prices = [price]
        
        for ret in returns[1:]:
            price *= (1 + ret)
            prices.append(price)
        
        # Create mock OHLCV data
        symbol_data = pd.DataFrame({
            'open': np.array(prices) * (1 + np.random.normal(0, 0.005, len(prices))),
            'high': np.array(prices) * (1 + np.abs(np.random.normal(0, 0.01, len(prices)))),
            'low': np.array(prices) * (1 - np.abs(np.random.normal(0, 0.01, len(prices)))),
            'close': prices,
            'volume': np.random.randint(1000000, 5000000, len(prices))
        }, index=dates)
        
        mock_data[symbol] = {'1d': symbol_data}
        current_prices[symbol] = prices[-1]
    
    # Test comprehensive signals for one symbol
    print(f"🔍 Generating comprehensive signals for AAPL")
    aapl_signals = advanced_strategies.get_comprehensive_signals(
        'AAPL', mock_data['AAPL'], current_prices['AAPL'], 'bullish'
    )
    
    if 'error' not in aapl_signals:
        agg = aapl_signals['aggregated_signal']
        rec = aapl_signals['final_recommendations']
        
        print(f"\n📊 AAPL Analysis Results:")
        print(f"Final Signal: {agg['signal']} (confidence: {agg['confidence']:.3f})")
        print(f"Expected Return: {agg['expected_return']:.3%}")
        print(f"Consensus: {agg['consensus']:.3f}")
        print(f"Strategies Used: {aapl_signals['total_strategies']}")
        print(f"Position Size: {rec['position_sizing']:.1%}")
        print(f"Risk Assessment: {rec['risk_assessment']}")
        
        print(f"\n🎯 Strategy Breakdown:")
        for strategy, contrib in agg.get('strategy_contributions', {}).items():
            print(f"  {strategy.replace('_', ' ').title()}: {contrib['signal']} "
                  f"(conf: {contrib['confidence']:.3f})")
    else:
        print(f"❌ Error: {aapl_signals['error']}")
    
    # Test portfolio dashboard
    print(f"\n📈 Generating Portfolio Dashboard")
    dashboard = advanced_strategies.get_portfolio_dashboard(mock_data, current_prices)
    
    if 'error' not in dashboard:
        summary = dashboard['portfolio_summary']
        print(f"\nPortfolio Summary:")
        print(f"  Total Symbols: {summary['total_symbols']}")
        print(f"  Signals Generated: {summary['signals_generated']}")
        print(f"  Buy/Sell/Hold: {summary['buy_signals']}/{summary['sell_signals']}/{summary['hold_signals']}")
        print(f"  Average Confidence: {summary['average_confidence']:.3f}")
        print(f"  Portfolio Risk: {summary['risk_assessment']}")
        
        print(f"\n🏆 Top Opportunities:")
        for i, opp in enumerate(summary['top_opportunities'][:3], 1):
            print(f"  {i}. {opp['symbol']}: {opp['signal']} "
                  f"(score: {opp['score']:.3f}, return: {opp['expected_return']:.2%})")
    else:
        print(f"❌ Dashboard Error: {dashboard['error']}")
    
    print("\n✅ Advanced trading strategies demo completed!")