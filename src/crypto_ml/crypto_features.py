"""
Crypto-specific machine learning features.
"""
import logging
import pandas as pd
import numpy as np
from typing import Dict, List, Optional
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


class CryptoFeatures:
    """
    Generate crypto-specific features for machine learning models.
    """
    
    def __init__(self):
        """Initialize crypto features generator."""
        logger.info("Crypto features generator initialized")
    
    def calculate_nvt_ratio(self, market_cap: float, transaction_volume: float) -> float:
        """
        Calculate Network Value to Transactions (NVT) Ratio.
        High NVT suggests overvaluation, low NVT suggests undervaluation.
        
        Args:
            market_cap: Market capitalization in USD
            transaction_volume: Daily on-chain transaction volume in USD
            
        Returns:
            NVT ratio
        """
        try:
            if transaction_volume == 0:
                return np.nan
            return market_cap / transaction_volume
        except Exception as e:
            logger.error(f"Error calculating NVT ratio: {e}")
            return np.nan
    
    def calculate_mvrv(self, market_cap: float, realized_cap: float) -> float:
        """
        Calculate Market Value to Realized Value (MVRV) Ratio.
        MVRV > 3.5: potential market top
        MVRV < 1.0: potential market bottom
        
        Args:
            market_cap: Current market capitalization
            realized_cap: Realized capitalization (sum of last moved values)
            
        Returns:
            MVRV ratio
        """
        try:
            if realized_cap == 0:
                return np.nan
            return market_cap / realized_cap
        except Exception as e:
            logger.error(f"Error calculating MVRV: {e}")
            return np.nan
    
    def calculate_sopr(self, spent_outputs: pd.DataFrame) -> float:
        """
        Calculate Spent Output Profit Ratio (SOPR).
        SOPR > 1: coins moving at profit
        SOPR < 1: coins moving at loss
        
        Args:
            spent_outputs: DataFrame with 'value_created' and 'value_spent' columns
            
        Returns:
            SOPR value
        """
        try:
            if len(spent_outputs) == 0:
                return np.nan
            
            total_value_spent = spent_outputs['value_spent'].sum()
            total_value_created = spent_outputs['value_created'].sum()
            
            if total_value_created == 0:
                return np.nan
            
            return total_value_spent / total_value_created
        except Exception as e:
            logger.error(f"Error calculating SOPR: {e}")
            return np.nan
    
    def calculate_funding_momentum(self, funding_rates: List[float], window: int = 24) -> float:
        """
        Calculate funding rate momentum.
        
        Args:
            funding_rates: List of recent funding rates
            window: Lookback window
            
        Returns:
            Funding momentum (positive = increasing funding)
        """
        try:
            if len(funding_rates) < 2:
                return 0.0
            
            recent = funding_rates[-window:] if len(funding_rates) >= window else funding_rates
            
            # Calculate momentum as difference between recent avg and older avg
            mid_point = len(recent) // 2
            recent_avg = np.mean(recent[mid_point:])
            older_avg = np.mean(recent[:mid_point])
            
            return recent_avg - older_avg
        except Exception as e:
            logger.error(f"Error calculating funding momentum: {e}")
            return 0.0
    
    def calculate_exchange_netflow(self, inflows: float, outflows: float) -> Dict:
        """
        Calculate exchange netflow indicators.
        Negative netflow (outflows > inflows) is bullish.
        
        Args:
            inflows: Amount deposited to exchanges
            outflows: Amount withdrawn from exchanges
            
        Returns:
            Netflow metrics
        """
        try:
            netflow = inflows - outflows
            netflow_ratio = netflow / (inflows + outflows) if (inflows + outflows) > 0 else 0
            
            return {
                'netflow': netflow,
                'netflow_ratio': netflow_ratio,
                'inflows': inflows,
                'outflows': outflows,
                'signal': 'bullish' if netflow < 0 else 'bearish'
            }
        except Exception as e:
            logger.error(f"Error calculating exchange netflow: {e}")
            return {}
    
    def calculate_whale_activity(self, large_transactions: int, avg_transactions: int) -> float:
        """
        Calculate whale activity score.
        
        Args:
            large_transactions: Number of large transactions (>$100k)
            avg_transactions: Average number of large transactions
            
        Returns:
            Whale activity score (>1 = above average)
        """
        try:
            if avg_transactions == 0:
                return 1.0
            return large_transactions / avg_transactions
        except Exception as e:
            logger.error(f"Error calculating whale activity: {e}")
            return 1.0
    
    def calculate_btc_dominance_trend(self, btc_dominance: List[float], window: int = 7) -> str:
        """
        Calculate Bitcoin dominance trend.
        
        Args:
            btc_dominance: List of BTC dominance percentages
            window: Lookback window in days
            
        Returns:
            Trend direction ('increasing', 'decreasing', 'stable')
        """
        try:
            if len(btc_dominance) < 2:
                return 'stable'
            
            recent = btc_dominance[-window:] if len(btc_dominance) >= window else btc_dominance
            
            # Calculate trend
            trend = np.polyfit(range(len(recent)), recent, 1)[0]
            
            if trend > 0.5:
                return 'increasing'
            elif trend < -0.5:
                return 'decreasing'
            else:
                return 'stable'
        except Exception as e:
            logger.error(f"Error calculating BTC dominance trend: {e}")
            return 'stable'
    
    def calculate_altcoin_season_index(self, altcoin_performance: List[float]) -> float:
        """
        Calculate altcoin season index (0-100).
        > 75: Strong altcoin season
        < 25: Bitcoin season
        
        Args:
            altcoin_performance: List of altcoin % performance vs BTC
            
        Returns:
            Altcoin season index (0-100)
        """
        try:
            if len(altcoin_performance) == 0:
                return 50.0
            
            # Count how many altcoins outperformed BTC
            outperforming = sum(1 for perf in altcoin_performance if perf > 0)
            index = (outperforming / len(altcoin_performance)) * 100
            
            return index
        except Exception as e:
            logger.error(f"Error calculating altcoin season index: {e}")
            return 50.0
    
    def generate_feature_set(self, df: pd.DataFrame, metadata: Dict) -> pd.DataFrame:
        """
        Generate complete crypto-specific feature set.
        
        Args:
            df: DataFrame with OHLCV data
            metadata: Dictionary with additional metadata
            
        Returns:
            DataFrame with added crypto features
        """
        try:
            df = df.copy()
            
            # On-chain metrics
            if 'market_cap' in metadata and 'tx_volume' in metadata:
                df['nvt_ratio'] = self.calculate_nvt_ratio(
                    metadata['market_cap'], 
                    metadata['tx_volume']
                )
            
            if 'market_cap' in metadata and 'realized_cap' in metadata:
                df['mvrv'] = self.calculate_mvrv(
                    metadata['market_cap'],
                    metadata['realized_cap']
                )
            
            # Funding rates (if available)
            if 'funding_rates' in metadata:
                df['funding_momentum'] = self.calculate_funding_momentum(
                    metadata['funding_rates']
                )
            
            # Exchange flows
            if 'exchange_inflows' in metadata and 'exchange_outflows' in metadata:
                netflow = self.calculate_exchange_netflow(
                    metadata['exchange_inflows'],
                    metadata['exchange_outflows']
                )
                df['exchange_netflow'] = netflow['netflow']
                df['exchange_netflow_ratio'] = netflow['netflow_ratio']
            
            # Whale activity
            if 'large_tx_count' in metadata and 'avg_large_tx' in metadata:
                df['whale_activity_score'] = self.calculate_whale_activity(
                    metadata['large_tx_count'],
                    metadata['avg_large_tx']
                )
            
            # Market structure
            if 'btc_dominance' in metadata:
                df['btc_dominance'] = metadata['btc_dominance']
            
            if 'fear_greed_index' in metadata:
                df['fear_greed'] = metadata['fear_greed_index']
            
            # Volatility metrics
            if 'close' in df.columns:
                df['returns'] = df['close'].pct_change()
                df['volatility_7d'] = df['returns'].rolling(7).std()
                df['volatility_30d'] = df['returns'].rolling(30).std()
            
            # Log features added
            new_features = [col for col in df.columns if col not in ['open', 'high', 'low', 'close', 'volume']]
            logger.info(f"Generated {len(new_features)} crypto-specific features")
            
            return df
            
        except Exception as e:
            logger.error(f"Error generating feature set: {e}")
            return df


if __name__ == "__main__":
    # Test crypto features
    features = CryptoFeatures()
    
    print("=== Crypto Features Test ===")
    
    # Test NVT ratio
    nvt = features.calculate_nvt_ratio(market_cap=1e12, transaction_volume=5e9)
    print(f"NVT Ratio: {nvt:.2f}")
    
    # Test MVRV
    mvrv = features.calculate_mvrv(market_cap=1e12, realized_cap=8e11)
    print(f"MVRV: {mvrv:.2f}")
    
    # Test funding momentum
    funding_rates = [0.01, 0.012, 0.015, 0.018, 0.020, 0.022]
    momentum = features.calculate_funding_momentum(funding_rates)
    print(f"Funding Momentum: {momentum:.4f}")
    
    # Test exchange netflow
    netflow = features.calculate_exchange_netflow(inflows=1000, outflows=1500)
    print(f"Exchange Netflow: {netflow['netflow']:.2f} ({netflow['signal']})")
    
    # Test altcoin season index
    altcoin_perf = [5, -2, 8, 3, -1, 10, 15, 2]
    alt_index = features.calculate_altcoin_season_index(altcoin_perf)
    print(f"Altcoin Season Index: {alt_index:.2f}")
    
    print("\n✅ Crypto features test completed!")
