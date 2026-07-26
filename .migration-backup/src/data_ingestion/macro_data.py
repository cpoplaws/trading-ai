"""
Macroeconomic data integration using FRED API.
Fetches indicators like CPI, unemployment, Fed rates, GDP, etc.
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import logging
import os

try:
    from fredapi import Fred
    FRED_AVAILABLE = True
except ImportError:
    FRED_AVAILABLE = False
    logging.warning("fredapi not installed. Install with: pip install fredapi")

logger = logging.getLogger(__name__)

class MacroDataFetcher:
    """
    Fetch and analyze macroeconomic data from FRED API.
    """
    
    # FRED series IDs for key economic indicators
    SERIES_IDS = {
        'fed_funds_rate': 'FEDFUNDS',  # Federal Funds Rate
        'cpi': 'CPIAUCSL',  # Consumer Price Index
        'unemployment': 'UNRATE',  # Unemployment Rate
        'gdp': 'GDP',  # Gross Domestic Product
        'inflation': 'T10YIE',  # 10-Year Breakeven Inflation Rate
        'vix': 'VIXCLS',  # CBOE Volatility Index
        'dxy': 'DTWEXBGS',  # US Dollar Index
        'treasury_10y': 'DGS10',  # 10-Year Treasury Rate
        'treasury_2y': 'DGS2',  # 2-Year Treasury Rate
        'industrial_production': 'INDPRO',  # Industrial Production Index
        'retail_sales': 'RSXFS',  # Retail Sales
        'housing_starts': 'HOUST',  # Housing Starts
        'pce': 'PCE',  # Personal Consumption Expenditures
        'consumer_sentiment': 'UMCSENT',  # University of Michigan Consumer Sentiment
    }
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize macro data fetcher.
        
        Args:
            api_key: FRED API key (get free at https://fred.stlouisfed.org/docs/api/api_key.html)
        """
        self.api_key = api_key or os.getenv('FRED_API_KEY')
        self.fred_client = None
        
        if FRED_AVAILABLE and self.api_key:
            try:
                self.fred_client = Fred(api_key=self.api_key)
                logger.info("FRED API client initialized successfully")
            except Exception as e:
                logger.error(f"Failed to initialize FRED API: {e}")
        else:
            if not FRED_AVAILABLE:
                logger.warning("FRED API not available - install fredapi package")
            else:
                logger.warning("FRED_API_KEY not set - using simulated data")
    
    def get_indicator(self, indicator_name: str, 
                     start_date: Optional[datetime] = None,
                     end_date: Optional[datetime] = None) -> pd.DataFrame:
        """
        Fetch a specific economic indicator.
        
        Args:
            indicator_name: Name of indicator (see SERIES_IDS)
            start_date: Start date for data
            end_date: End date for data
            
        Returns:
            DataFrame with date and value columns
        """
        try:
            if indicator_name not in self.SERIES_IDS:
                logger.error(f"Unknown indicator: {indicator_name}")
                return pd.DataFrame()
            
            series_id = self.SERIES_IDS[indicator_name]
            
            # Use FRED API if available
            if self.fred_client:
                try:
                    data = self.fred_client.get_series(
                        series_id,
                        observation_start=start_date,
                        observation_end=end_date
                    )
                    
                    df = pd.DataFrame({
                        'date': data.index,
                        'value': data.values,
                        'indicator': indicator_name
                    })
                    
                    logger.info(f"Fetched {len(df)} observations for {indicator_name}")
                    return df
                
                except Exception as e:
                    logger.warning(f"FRED API error for {indicator_name}: {e}")
                    return self._get_simulated_data(indicator_name, start_date, end_date)
            else:
                return self._get_simulated_data(indicator_name, start_date, end_date)
        
        except Exception as e:
            logger.error(f"Error fetching {indicator_name}: {e}")
            return pd.DataFrame()
    
    def _get_simulated_data(self, indicator_name: str,
                           start_date: Optional[datetime] = None,
                           end_date: Optional[datetime] = None) -> pd.DataFrame:
        """Generate simulated data for testing."""
        if start_date is None:
            start_date = datetime.now() - timedelta(days=365)
        if end_date is None:
            end_date = datetime.now()
        
        # Generate monthly dates
        dates = pd.date_range(start=start_date, end=end_date, freq='MS')
        
        # Simulate data based on indicator type
        base_values = {
            'fed_funds_rate': (5.0, 0.5),  # mean, std
            'cpi': (300.0, 5.0),
            'unemployment': (4.0, 0.3),
            'gdp': (25000.0, 500.0),
            'inflation': (2.5, 0.2),
            'vix': (15.0, 3.0),
            'dxy': (100.0, 2.0),
            'treasury_10y': (4.0, 0.3),
            'treasury_2y': (4.5, 0.3),
            'industrial_production': (105.0, 2.0),
            'retail_sales': (700000.0, 10000.0),
            'housing_starts': (1500.0, 100.0),
            'pce': (18000.0, 200.0),
            'consumer_sentiment': (70.0, 5.0),
        }
        
        mean, std = base_values.get(indicator_name, (100.0, 10.0))
        values = np.random.normal(mean, std, len(dates))
        
        # Add trend for some indicators
        if indicator_name in ['cpi', 'gdp', 'pce']:
            trend = np.linspace(0, mean * 0.1, len(dates))
            values += trend
        
        df = pd.DataFrame({
            'date': dates,
            'value': values,
            'indicator': indicator_name
        })
        
        logger.info(f"Generated simulated data for {indicator_name}")
        return df
    
    def get_all_indicators(self, start_date: Optional[datetime] = None,
                          end_date: Optional[datetime] = None) -> Dict[str, pd.DataFrame]:
        """
        Fetch all available economic indicators.
        
        Args:
            start_date: Start date for data
            end_date: End date for data
            
        Returns:
            Dictionary mapping indicator names to DataFrames
        """
        indicators = {}
        
        for indicator_name in self.SERIES_IDS.keys():
            df = self.get_indicator(indicator_name, start_date, end_date)
            if not df.empty:
                indicators[indicator_name] = df
        
        return indicators
    
    def get_economic_regime(self) -> Dict:
        """
        Determine current economic regime based on macro indicators.
        
        Returns:
            Dictionary with regime classification and confidence
        """
        try:
            # Fetch recent data for key indicators
            lookback = datetime.now() - timedelta(days=90)
            
            fed_rate = self.get_indicator('fed_funds_rate', start_date=lookback)
            unemployment = self.get_indicator('unemployment', start_date=lookback)
            inflation = self.get_indicator('inflation', start_date=lookback)
            vix = self.get_indicator('vix', start_date=lookback)
            
            # Get latest values
            latest_fed_rate = fed_rate['value'].iloc[-1] if not fed_rate.empty else 5.0
            latest_unemployment = unemployment['value'].iloc[-1] if not unemployment.empty else 4.0
            latest_inflation = inflation['value'].iloc[-1] if not inflation.empty else 2.5
            latest_vix = vix['value'].iloc[-1] if not vix.empty else 15.0
            
            # Determine regime
            regime_score = 0
            regime_factors = []
            
            # Fed policy stance
            if latest_fed_rate > 4.5:
                regime_score -= 1
                regime_factors.append("Hawkish Fed policy")
            elif latest_fed_rate < 2.0:
                regime_score += 1
                regime_factors.append("Dovish Fed policy")
            
            # Unemployment
            if latest_unemployment < 4.0:
                regime_score += 1
                regime_factors.append("Strong labor market")
            elif latest_unemployment > 6.0:
                regime_score -= 1
                regime_factors.append("Weak labor market")
            
            # Inflation
            if latest_inflation > 3.0:
                regime_score -= 1
                regime_factors.append("High inflation")
            elif latest_inflation < 2.0:
                regime_score += 0.5
                regime_factors.append("Low inflation")
            
            # Market volatility
            if latest_vix > 25:
                regime_score -= 1
                regime_factors.append("High volatility")
            elif latest_vix < 15:
                regime_score += 0.5
                regime_factors.append("Low volatility")
            
            # Classify regime
            if regime_score > 1:
                regime = "EXPANSION"
                confidence = 0.8
            elif regime_score < -1:
                regime = "CONTRACTION"
                confidence = 0.8
            else:
                regime = "TRANSITION"
                confidence = 0.6
            
            return {
                'regime': regime,
                'confidence': confidence,
                'regime_score': regime_score,
                'factors': regime_factors,
                'indicators': {
                    'fed_funds_rate': latest_fed_rate,
                    'unemployment': latest_unemployment,
                    'inflation': latest_inflation,
                    'vix': latest_vix
                },
                'timestamp': datetime.now(),
                'real_api': self.fred_client is not None
            }
        
        except Exception as e:
            logger.error(f"Error determining economic regime: {e}")
            return {
                'regime': 'UNKNOWN',
                'confidence': 0.0,
                'error': str(e)
            }
    
    def get_yield_curve(self) -> Dict:
        """
        Calculate yield curve spread (10Y - 2Y) as recession indicator.
        
        Returns:
            Dictionary with yield curve data and interpretation
        """
        try:
            lookback = datetime.now() - timedelta(days=30)
            
            treasury_10y = self.get_indicator('treasury_10y', start_date=lookback)
            treasury_2y = self.get_indicator('treasury_2y', start_date=lookback)
            
            if treasury_10y.empty or treasury_2y.empty:
                return {'spread': None, 'inverted': False, 'signal': 'UNKNOWN'}
            
            latest_10y = treasury_10y['value'].iloc[-1]
            latest_2y = treasury_2y['value'].iloc[-1]
            spread = latest_10y - latest_2y
            
            # Interpret spread
            if spread < -0.2:
                signal = "RECESSION WARNING"
                confidence = 0.8
            elif spread < 0:
                signal = "INVERSION"
                confidence = 0.6
            elif spread < 0.5:
                signal = "FLATTENING"
                confidence = 0.5
            else:
                signal = "NORMAL"
                confidence = 0.7
            
            return {
                'spread': spread,
                'inverted': spread < 0,
                'treasury_10y': latest_10y,
                'treasury_2y': latest_2y,
                'signal': signal,
                'confidence': confidence,
                'timestamp': datetime.now()
            }
        
        except Exception as e:
            logger.error(f"Error calculating yield curve: {e}")
            return {'spread': None, 'inverted': False, 'signal': 'ERROR'}
    
    def get_macro_summary(self) -> Dict:
        """
        Get comprehensive macroeconomic summary.
        
        Returns:
            Dictionary with all key macro indicators and regime analysis
        """
        try:
            regime = self.get_economic_regime()
            yield_curve = self.get_yield_curve()
            
            # Fetch recent data
            lookback = datetime.now() - timedelta(days=30)
            consumer_sentiment = self.get_indicator('consumer_sentiment', start_date=lookback)
            
            summary = {
                'regime': regime,
                'yield_curve': yield_curve,
                'consumer_sentiment': consumer_sentiment['value'].iloc[-1] if not consumer_sentiment.empty else None,
                'timestamp': datetime.now(),
                'data_source': 'FRED API' if self.fred_client else 'Simulated'
            }
            
            return summary
        
        except Exception as e:
            logger.error(f"Error generating macro summary: {e}")
            return {'error': str(e)}


if __name__ == "__main__":
    # Test the macro data fetcher
    logging.basicConfig(level=logging.INFO)
    
    fetcher = MacroDataFetcher()
    
    print("\n=== Economic Regime Analysis ===")
    regime = fetcher.get_economic_regime()
    print(f"Regime: {regime['regime']} (Confidence: {regime['confidence']:.1%})")
    print(f"Factors: {', '.join(regime['factors'])}")
    
    print("\n=== Yield Curve Analysis ===")
    yield_curve = fetcher.get_yield_curve()
    print(f"Spread: {yield_curve['spread']:.2f}%")
    print(f"Signal: {yield_curve['signal']}")
    
    print("\n=== Macro Summary ===")
    summary = fetcher.get_macro_summary()
    print(f"Data Source: {summary['data_source']}")
    print(f"Consumer Sentiment: {summary.get('consumer_sentiment', 'N/A')}")
