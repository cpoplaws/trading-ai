"""
Multi-timeframe trading analysis system.
Analyzes 1min, 5min, 1hr, and daily signals for comprehensive trading decisions.
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import yfinance as yf
import logging
from concurrent.futures import ThreadPoolExecutor
import sys
import os

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from ..feature_engineering.feature_generator import FeatureGenerator
from ..modeling.train_model import train_model
from ..strategy.simple_strategy import generate_signals

logger = logging.getLogger(__name__)

class MultiTimeframeAnalyzer:
    """
    Multi-timeframe analysis for comprehensive trading signals.
    """
    
    TIMEFRAMES = {
        '1min': '1m',
        '5min': '5m', 
        '1hour': '1h',
        '1day': '1d'
    }
    
    def __init__(self, symbol: str):
        """
        Initialize multi-timeframe analyzer.
        
        Args:
            symbol: Trading symbol (e.g., 'AAPL', 'BTC-USD')
        """
        self.symbol = symbol
        self.data = {}
        self.signals = {}
        self.models = {}
        
    def fetch_multi_timeframe_data(self, period: str = "30d") -> Dict[str, pd.DataFrame]:
        """
        Fetch data for all timeframes.
        
        Args:
            period: Data period to fetch
            
        Returns:
            Dictionary with timeframe data
        """
        logger.info(f"Fetching multi-timeframe data for {self.symbol}")
        
        def fetch_timeframe(timeframe_key: str, interval: str) -> Tuple[str, pd.DataFrame]:
            try:
                ticker = yf.Ticker(self.symbol)
                data = ticker.history(period=period, interval=interval)
                
                if not data.empty:
                    # Standardize column names
                    data.columns = [col.lower() for col in data.columns]
                    logger.info(f"Fetched {len(data)} {timeframe_key} candles")
                    return timeframe_key, data
                else:
                    logger.warning(f"No data for {timeframe_key}")
                    return timeframe_key, pd.DataFrame()
                    
            except Exception as e:
                logger.error(f"Error fetching {timeframe_key} data: {e}")
                return timeframe_key, pd.DataFrame()
        
        # Fetch all timeframes in parallel
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = [
                executor.submit(fetch_timeframe, tf_key, interval)
                for tf_key, interval in self.TIMEFRAMES.items()
            ]
            
            for future in futures:
                tf_key, data = future.result()
                self.data[tf_key] = data
        
        return self.data
    
    def generate_timeframe_features(self) -> Dict[str, pd.DataFrame]:
        """
        Generate features for each timeframe.
        
        Returns:
            Dictionary with engineered features for each timeframe
        """
        logger.info("Generating features for all timeframes")
        
        features = {}
        
        for tf_key, data in self.data.items():
            if data.empty:
                continue
                
            try:
                # Generate features for this timeframe
                fg = FeatureGenerator(data)
                features_df = fg.generate_features()
                
                # Add timeframe-specific features
                features_df = self._add_timeframe_specific_features(features_df, tf_key)
                
                features[tf_key] = features_df
                logger.info(f"Generated {len(features_df.columns)} features for {tf_key}")
                
            except Exception as e:
                logger.error(f"Error generating features for {tf_key}: {e}")
                features[tf_key] = pd.DataFrame()
        
        return features
    
    def _add_timeframe_specific_features(self, df: pd.DataFrame, timeframe: str) -> pd.DataFrame:
        """
        Add timeframe-specific technical indicators.
        
        Args:
            df: Base dataframe with features
            timeframe: Current timeframe
            
        Returns:
            Enhanced dataframe with timeframe-specific features
        """
        try:
            # Timeframe-specific parameters
            tf_params = {
                '1min': {'fast_sma': 5, 'slow_sma': 20, 'rsi_period': 7},
                '5min': {'fast_sma': 10, 'slow_sma': 30, 'rsi_period': 14},
                '1hour': {'fast_sma': 20, 'slow_sma': 50, 'rsi_period': 14},
                '1day': {'fast_sma': 50, 'slow_sma': 200, 'rsi_period': 21}
            }
            
            params = tf_params.get(timeframe, tf_params['1day'])
            
            # Add momentum indicators
            if 'close' in df.columns:
                # Price momentum
                df[f'momentum_5_{timeframe}'] = df['close'].pct_change(5)
                df[f'momentum_10_{timeframe}'] = df['close'].pct_change(10)
                
                # Support/Resistance levels
                df[f'support_{timeframe}'] = df['low'].rolling(20).min()
                df[f'resistance_{timeframe}'] = df['high'].rolling(20).max()
                
                # Timeframe-specific trend
                df[f'trend_strength_{timeframe}'] = (
                    df['close'] - df['close'].rolling(params['slow_sma']).mean()
                ) / df['close'].rolling(params['slow_sma']).std()
            
            # Volume analysis (if available)
            if 'volume' in df.columns:
                df[f'volume_trend_{timeframe}'] = (
                    df['volume'] / df['volume'].rolling(20).mean()
                )
                
                # Price-Volume divergence
                if 'close' in df.columns:
                    price_change = df['close'].pct_change()
                    volume_change = df['volume'].pct_change()
                    df[f'pv_divergence_{timeframe}'] = price_change - volume_change
            
            return df
            
        except Exception as e:
            logger.error(f"Error adding timeframe features for {timeframe}: {e}")
            return df
    
    def train_timeframe_models(self, features: Dict[str, pd.DataFrame]) -> Dict[str, str]:
        """
        Train separate models for each timeframe.
        
        Args:
            features: Features for each timeframe
            
        Returns:
            Dictionary with model paths for each timeframe
        """
        logger.info("Training models for all timeframes")
        
        model_paths = {}
        
        for tf_key, features_df in features.items():
            if features_df.empty:
                continue
                
            try:
                model_path = f'./models/{self.symbol}_{tf_key}_model.joblib'
                
                success = train_model(features_df, save_path=model_path)
                
                if success:
                    model_paths[tf_key] = model_path
                    logger.info(f"Trained {tf_key} model successfully")
                else:
                    logger.warning(f"Failed to train {tf_key} model")
                    
            except Exception as e:
                logger.error(f"Error training {tf_key} model: {e}")
        
        return model_paths
    
    def generate_multi_timeframe_signals(self, model_paths: Dict[str, str], 
                                       features: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        """
        Generate signals for all timeframes.
        
        Args:
            model_paths: Model paths for each timeframe
            features: Features for each timeframe
            
        Returns:
            Signals for each timeframe
        """
        logger.info("Generating signals for all timeframes")
        
        signals = {}
        
        for tf_key in model_paths.keys():
            if tf_key not in features or features[tf_key].empty:
                continue
                
            try:
                # Save features for signal generation
                data_path = f'./data/processed/{self.symbol}_{tf_key}.csv'
                os.makedirs(os.path.dirname(data_path), exist_ok=True)
                features[tf_key].to_csv(data_path)
                
                # Generate signals
                signals_path = f'./signals/'
                success = generate_signals(
                    model_path=model_paths[tf_key],
                    data_path=data_path,
                    save_path=signals_path
                )
                
                if success:
                    # Load generated signals
                    signal_file = f'./signals/{self.symbol}_{tf_key}_signals.csv'
                    if os.path.exists(signal_file):
                        signals_df = pd.read_csv(signal_file)
                        signals[tf_key] = signals_df
                        logger.info(f"Generated {len(signals_df)} {tf_key} signals")
                    
            except Exception as e:
                logger.error(f"Error generating {tf_key} signals: {e}")
        
        return signals
    
    def aggregate_timeframe_signals(self, signals: Dict[str, Dict[str, float]]) -> Dict:
        """
        Aggregate signals across timeframes into a single decision.
        
        Args:
            signals: Mapping of timeframe -> {'signal': str, 'confidence': float}
                - signal: One of {'BUY', 'SELL', 'HOLD'}.
                - confidence: Confidence score in [0.0, 1.0], multiplied by the
                  timeframe weight.
        
        Returns:
            Dict with:
                - final_signal (str): Aggregated signal.
                - confidence (float): Weighted score of the winning signal.
                - votes (Dict[str, float]): Per-signal weighted totals.
        """
        try:
            if not signals:
                return {'final_signal': 'HOLD', 'confidence': 0.0}
            
            # Canonical timeframe weights with aliases for convenience
            weights = {
                '1min': 0.1,
                '5min': 0.2,
                '1h': 0.3,
                '1d': 0.4,
            }
            timeframe_aliases = {
                '1hour': '1h',
                '1day': '1d',
            }
            
            vote_score = {'BUY': 0.0, 'SELL': 0.0, 'HOLD': 0.0}
            for tf, data in signals.items():
                normalized_tf = timeframe_aliases.get(tf, tf)
                weight = weights.get(normalized_tf, 0.25)
                signal = data.get('signal', 'HOLD')
                confidence = float(data.get('confidence', 0.5))
                if signal not in vote_score:
                    logger.warning("Unknown signal '%s' in timeframe '%s'; skipping", signal, tf)
                    continue
                vote_score[signal] += weight * confidence
            
            if vote_score['BUY'] > vote_score['SELL'] and vote_score['BUY'] > vote_score['HOLD']:
                final = 'BUY'
                conf = vote_score['BUY']
            elif vote_score['SELL'] > vote_score['BUY'] and vote_score['SELL'] > vote_score['HOLD']:
                final = 'SELL'
                conf = vote_score['SELL']
            else:
                final = 'HOLD'
                conf = max(vote_score.values())
            
            return {
                'final_signal': final,
                'confidence': conf,
                'votes': vote_score
            }
        except Exception as e:
            logger.exception("Error aggregating timeframe signals: %s", e)
            raise
    
    def combine_timeframe_signals(self, signals: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """
        Combine signals from all timeframes into a unified decision.
        
        Args:
            signals: Signals from each timeframe
            
        Returns:
            Combined signals with timeframe weights
        """
        logger.info("Combining multi-timeframe signals")
        
        # Timeframe weights (longer timeframes get higher weight)
        weights = {
            '1min': 0.1,
            '5min': 0.2,
            '1hour': 0.3,
            '1day': 0.4
        }
        
        combined_signals = []
        
        # Get the latest signals from each timeframe
        latest_signals = {}
        for tf_key, signals_df in signals.items():
            if not signals_df.empty:
                latest = signals_df.iloc[-1]
                latest_signals[tf_key] = {
                    'signal': latest.get('Signal', 'HOLD'),
                    'confidence': latest.get('Confidence', 0.5),
                    'price': latest.get('Price', 0),
                    'timestamp': latest.get('Timestamp', datetime.now())
                }
        
        if not latest_signals:
            return pd.DataFrame()
        
        # Calculate weighted signal
        weighted_confidence = 0
        buy_weight = 0
        sell_weight = 0
        
        for tf_key, signal_data in latest_signals.items():
            weight = weights.get(tf_key, 0.25)
            confidence = signal_data['confidence']
            
            if signal_data['signal'] == 'BUY':
                buy_weight += weight * confidence
            elif signal_data['signal'] == 'SELL':
                sell_weight += weight * confidence
            
            weighted_confidence += weight * confidence
        
        # Determine final signal
        if buy_weight > sell_weight * 1.2:  # Require 20% edge for BUY
            final_signal = 'BUY'
            final_confidence = buy_weight
        elif sell_weight > buy_weight * 1.2:  # Require 20% edge for SELL
            final_signal = 'SELL'
            final_confidence = sell_weight
        else:
            final_signal = 'HOLD'
            final_confidence = max(buy_weight, sell_weight)
        
        # Create combined signal
        combined_signal = {
            'timestamp': datetime.now(),
            'symbol': self.symbol,
            'signal': final_signal,
            'confidence': final_confidence,
            'timeframe_signals': latest_signals,
            'buy_weight': buy_weight,
            'sell_weight': sell_weight,
            'signal_strength': 'STRONG' if final_confidence > 0.7 else 'MEDIUM' if final_confidence > 0.5 else 'WEAK'
        }
        
        logger.info(f"Combined signal: {final_signal} (confidence: {final_confidence:.3f})")
        
        return pd.DataFrame([combined_signal])
    
    def run_full_analysis(self, period: str = "30d") -> Dict:
        """
        Run complete multi-timeframe analysis.
        
        Args:
            period: Data period to analyze
            
        Returns:
            Complete analysis results
        """
        logger.info(f"Starting full multi-timeframe analysis for {self.symbol}")
        
        try:
            # 1. Fetch data
            data = self.fetch_multi_timeframe_data(period)
            
            # 2. Generate features
            features = self.generate_timeframe_features()
            
            # 3. Train models
            model_paths = self.train_timeframe_models(features)
            
            # 4. Generate signals
            signals = self.generate_multi_timeframe_signals(model_paths, features)
            
            # 5. Combine signals
            combined_signal = self.combine_timeframe_signals(signals)
            
            results = {
                'symbol': self.symbol,
                'analysis_time': datetime.now(),
                'timeframes_analyzed': list(self.TIMEFRAMES.keys()),
                'models_trained': len(model_paths),
                'signals_generated': {tf: len(sigs) for tf, sigs in signals.items()},
                'combined_signal': combined_signal.to_dict('records')[0] if not combined_signal.empty else {},
                'individual_signals': signals
            }
            
            logger.info("Multi-timeframe analysis completed successfully")
            return results
            
        except Exception as e:
            logger.error(f"Error in multi-timeframe analysis: {e}")
            return {'error': str(e)}

# Example usage and testing
if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.INFO)
    
    # Test multi-timeframe analysis
    print("🚀 Multi-Timeframe Trading Analysis Demo")
    print("=" * 50)
    
    # Analyze AAPL across all timeframes
    analyzer = MultiTimeframeAnalyzer('AAPL')
    results = analyzer.run_full_analysis(period="7d")  # 1 week for demo
    
    if 'error' not in results:
        print(f"\n✅ Analysis Results for {results['symbol']}:")
        print(f"   Timeframes: {', '.join(results['timeframes_analyzed'])}")
        print(f"   Models Trained: {results['models_trained']}")
        print(f"   Signals Generated: {results['signals_generated']}")
        
        if results['combined_signal']:
            signal = results['combined_signal']
            print(f"\n🎯 Combined Signal:")
            print(f"   Action: {signal.get('signal', 'N/A')}")
            print(f"   Confidence: {signal.get('confidence', 0):.3f}")
            print(f"   Strength: {signal.get('signal_strength', 'N/A')}")
            print(f"   Buy Weight: {signal.get('buy_weight', 0):.3f}")
            print(f"   Sell Weight: {signal.get('sell_weight', 0):.3f}")
    else:
        print(f"❌ Analysis failed: {results['error']}")
    
    print("\n✅ Multi-timeframe analysis demo completed!")
