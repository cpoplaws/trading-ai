"""
Macro economic data ingestion module.
Fetches Federal Reserve data, economic indicators, and macro signals.
"""

import os
import logging
from typing import Dict, List, Optional
from datetime import datetime, timedelta
import pandas as pd
import requests
from utils.logger import setup_logger

logger = setup_logger(__name__)


class MacroDataFetcher:
    """
    Fetch macroeconomic data from various sources.
    
    Data sources:
    - FRED API (Federal Reserve Economic Data)
    - Treasury.gov (Bond yields)
    - BLS (Bureau of Labor Statistics)
    """
    
    def __init__(self, fred_api_key: Optional[str] = None):
        """
        Initialize macro data fetcher.
        
        Args:
            fred_api_key: FRED API key (get from https://fred.stlouisfed.org/docs/api/api_key.html)
        """
        self.fred_api_key = fred_api_key or os.getenv('FRED_API_KEY')
        self.fred_base_url = 'https://api.stlouisfed.org/fred/series/observations'
        
        # Economic indicators to track
        self.indicators = {
            'fed_funds_rate': 'DFF',  # Federal Funds Rate
            'inflation_cpi': 'CPIAUCSL',  # Consumer Price Index
            'unemployment': 'UNRATE',  # Unemployment Rate
            'gdp_growth': 'GDP',  # Gross Domestic Product
            'vix': 'VIXCLS',  # Market Volatility Index
            '10y_treasury': 'DGS10',  # 10-Year Treasury Yield
            '2y_treasury': 'DGS2',  # 2-Year Treasury Yield
            'retail_sales': 'RSXFS',  # Retail Sales
            'housing_starts': 'HOUST',  # Housing Starts
            'consumer_sentiment': 'UMCSENT'  # University of Michigan Consumer Sentiment
        }
        
        logger.info("MacroDataFetcher initialized")
        
    def fetch_fred_series(
        self,
        series_id: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Fetch a data series from FRED API.
        
        Args:
            series_id: FRED series ID
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            
        Returns:
            DataFrame with date and value columns
        """
        if not self.fred_api_key:
            logger.warning("FRED API key not set, using mock data")
            return self._generate_mock_data(series_id)
            
        params = {
            'series_id': series_id,
            'api_key': self.fred_api_key,
            'file_type': 'json'
        }
        
        if start_date:
            params['observation_start'] = start_date
        if end_date:
            params['observation_end'] = end_date
            
        try:
            response = requests.get(self.fred_base_url, params=params, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            observations = data.get('observations', [])
            
            df = pd.DataFrame(observations)
            df['date'] = pd.to_datetime(df['date'])
            df['value'] = pd.to_numeric(df['value'], errors='coerce')
            df = df[['date', 'value']].dropna()
            df = df.rename(columns={'value': series_id})
            
            logger.info(f"Fetched {len(df)} observations for {series_id}")
            return df
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Error fetching FRED data for {series_id}: {e}")
            return self._generate_mock_data(series_id)
            
    def fetch_all_indicators(
        self,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Fetch all macro economic indicators.
        
        Args:
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            
        Returns:
            DataFrame with all indicators merged by date
        """
        if not start_date:
            start_date = (datetime.now() - timedelta(days=365*2)).strftime('%Y-%m-%d')
        if not end_date:
            end_date = datetime.now().strftime('%Y-%m-%d')
            
        logger.info(f"Fetching all macro indicators from {start_date} to {end_date}")
        
        dfs = []
        for name, series_id in self.indicators.items():
            df = self.fetch_fred_series(series_id, start_date, end_date)
            if not df.empty:
                df = df.rename(columns={series_id: name})
                dfs.append(df)
                
        if not dfs:
            logger.warning("No macro data fetched")
            return pd.DataFrame()
            
        # Merge all dataframes on date
        result = dfs[0]
        for df in dfs[1:]:
            result = pd.merge(result, df, on='date', how='outer')
            
        result = result.sort_values('date').reset_index(drop=True)
        
        # Forward fill missing values (economic data is often monthly/quarterly)
        result = result.fillna(method='ffill')
        
        logger.info(f"Merged macro data: {len(result)} rows, {len(result.columns)-1} indicators")
        return result
        
    def get_latest_indicators(self) -> Dict[str, float]:
        """
        Get the most recent values for all indicators.
        
        Returns:
            Dictionary mapping indicator name to latest value
        """
        df = self.fetch_all_indicators()
        if df.empty:
            return {}
            
        latest = df.iloc[-1].to_dict()
        latest.pop('date', None)
        
        logger.info(f"Latest indicators: {latest}")
        return latest
        
    def calculate_macro_regime(self) -> str:
        """
        Determine current macro economic regime.
        
        Regimes:
        - expansion: Low unemployment, rising GDP, low VIX
        - recession: High unemployment, falling GDP, high VIX
        - stagflation: High inflation, low GDP growth
        - recovery: Improving indicators after recession
        
        Returns:
            String describing current regime
        """
        indicators = self.get_latest_indicators()
        
        if not indicators:
            return 'unknown'
            
        unemployment = indicators.get('unemployment', 5.0)
        inflation = indicators.get('inflation_cpi', 250.0)
        vix = indicators.get('vix', 20.0)
        
        # Simple heuristic-based regime detection
        if unemployment > 6.5 and vix > 30:
            return 'recession'
        elif inflation > 280 and unemployment > 5.5:
            return 'stagflation'
        elif vix < 15 and unemployment < 4.5:
            return 'expansion'
        elif unemployment < 5.5 and vix < 25:
            return 'recovery'
        else:
            return 'neutral'
            
    def _generate_mock_data(self, series_id: str) -> pd.DataFrame:
        """Generate mock data when API is unavailable."""
        dates = pd.date_range(end=datetime.now(), periods=30, freq='D')
        values = pd.Series(range(30)) + 100  # Simple increasing trend
        
        return pd.DataFrame({
            'date': dates,
            series_id: values
        })
        
    def save_to_csv(self, df: pd.DataFrame, filename: str) -> None:
        """
        Save macro data to CSV file.
        
        Args:
            df: DataFrame to save
            filename: Output filename
        """
        output_path = os.path.join('data', 'raw', filename)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        df.to_csv(output_path, index=False)
        logger.info(f"Saved macro data to {output_path}")


def main():
    """Example usage of MacroDataFetcher."""
    fetcher = MacroDataFetcher()
    
    # Fetch all indicators
    df = fetcher.fetch_all_indicators()
    print("\nMacro Economic Indicators:")
    print(df.tail(10))
    
    # Get latest values
    latest = fetcher.get_latest_indicators()
    print("\nLatest Indicator Values:")
    for name, value in latest.items():
        print(f"{name}: {value:.2f}")
        
    # Determine regime
    regime = fetcher.calculate_macro_regime()
    print(f"\nCurrent Macro Regime: {regime}")
    
    # Save to file
    fetcher.save_to_csv(df, 'macro_indicators.csv')


if __name__ == '__main__':
    main()
