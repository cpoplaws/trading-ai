"""
CoinGecko API client for crypto market data and token metadata.
"""
import os
import logging
import requests
from typing import Dict, List, Optional
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)


class CoinGeckoClient:
    """
    CoinGecko API client for comprehensive crypto data.
    """
    
    BASE_URL = "https://api.coingecko.com/api/v3"
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize CoinGecko client.
        
        Args:
            api_key: Optional API key for pro tier
        """
        self.api_key = api_key or os.getenv('COINGECKO_API_KEY')
        self.session = requests.Session()
        
        if self.api_key:
            self.session.headers.update({'X-Cg-Pro-Api-Key': self.api_key})
        
        logger.info("CoinGecko client initialized")
    
    def get_price(self, coin_ids: List[str], vs_currencies: List[str] = ['usd']) -> Optional[Dict]:
        """
        Get current price for multiple coins.
        
        Args:
            coin_ids: List of coin IDs (e.g., ['bitcoin', 'ethereum'])
            vs_currencies: List of currencies to get prices in
            
        Returns:
            Price dictionary
        """
        try:
            url = f"{self.BASE_URL}/simple/price"
            params = {
                'ids': ','.join(coin_ids),
                'vs_currencies': ','.join(vs_currencies),
                'include_24hr_change': 'true',
                'include_market_cap': 'true',
                'include_24hr_vol': 'true'
            }
            response = self.session.get(url, params=params, timeout=10)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"Error getting prices: {e}")
            return None
    
    def get_coin_data(self, coin_id: str) -> Optional[Dict]:
        """
        Get detailed coin data.
        
        Args:
            coin_id: Coin ID (e.g., 'bitcoin')
            
        Returns:
            Comprehensive coin data
        """
        try:
            url = f"{self.BASE_URL}/coins/{coin_id}"
            params = {
                'localization': 'false',
                'tickers': 'true',
                'market_data': 'true',
                'community_data': 'true',
                'developer_data': 'true'
            }
            response = self.session.get(url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            return {
                'id': data['id'],
                'symbol': data['symbol'],
                'name': data['name'],
                'market_cap_rank': data.get('market_cap_rank'),
                'current_price': data['market_data']['current_price']['usd'],
                'market_cap': data['market_data']['market_cap']['usd'],
                'total_volume': data['market_data']['total_volume']['usd'],
                'price_change_24h': data['market_data']['price_change_percentage_24h'],
                'price_change_7d': data['market_data'].get('price_change_percentage_7d'),
                'price_change_30d': data['market_data'].get('price_change_percentage_30d'),
                'ath': data['market_data']['ath']['usd'],
                'ath_date': data['market_data']['ath_date']['usd'],
                'atl': data['market_data']['atl']['usd'],
                'circulating_supply': data['market_data'].get('circulating_supply'),
                'total_supply': data['market_data'].get('total_supply'),
                'max_supply': data['market_data'].get('max_supply'),
            }
        except Exception as e:
            logger.error(f"Error getting coin data for {coin_id}: {e}")
            return None
    
    def get_market_chart(self, coin_id: str, vs_currency: str = 'usd', days: int = 30) -> Optional[Dict]:
        """
        Get historical market data (price, volume, market cap).
        
        Args:
            coin_id: Coin ID
            vs_currency: Currency to get data in
            days: Number of days (1, 7, 14, 30, 90, 180, 365, max)
            
        Returns:
            Historical data dictionary
        """
        try:
            url = f"{self.BASE_URL}/coins/{coin_id}/market_chart"
            params = {
                'vs_currency': vs_currency,
                'days': days,
                'interval': 'daily' if days > 90 else 'hourly'
            }
            response = self.session.get(url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            return {
                'prices': data.get('prices', []),
                'market_caps': data.get('market_caps', []),
                'total_volumes': data.get('total_volumes', [])
            }
        except Exception as e:
            logger.error(f"Error getting market chart for {coin_id}: {e}")
            return None
    
    def get_trending_coins(self) -> List[Dict]:
        """
        Get trending coins (top 7 trending coins on CoinGecko in the last 24 hours).
        
        Returns:
            List of trending coins
        """
        try:
            url = f"{self.BASE_URL}/search/trending"
            response = self.session.get(url, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            trending = []
            for item in data.get('coins', []):
                coin = item['item']
                trending.append({
                    'id': coin['id'],
                    'name': coin['name'],
                    'symbol': coin['symbol'],
                    'market_cap_rank': coin.get('market_cap_rank'),
                    'price_btc': coin.get('price_btc'),
                    'score': coin.get('score')
                })
            
            return trending
        except Exception as e:
            logger.error(f"Error getting trending coins: {e}")
            return []
    
    def get_global_data(self) -> Optional[Dict]:
        """
        Get global crypto market data.
        
        Returns:
            Global market statistics
        """
        try:
            url = f"{self.BASE_URL}/global"
            response = self.session.get(url, timeout=10)
            response.raise_for_status()
            data = response.json()['data']
            
            return {
                'active_cryptocurrencies': data['active_cryptocurrencies'],
                'markets': data['markets'],
                'total_market_cap': data['total_market_cap']['usd'],
                'total_volume': data['total_volume']['usd'],
                'market_cap_percentage': data['market_cap_percentage'],
                'market_cap_change_24h': data['market_cap_change_percentage_24h_usd'],
                'btc_dominance': data['market_cap_percentage'].get('btc', 0),
                'eth_dominance': data['market_cap_percentage'].get('eth', 0)
            }
        except Exception as e:
            logger.error(f"Error getting global data: {e}")
            return None
    
    def get_fear_greed_index(self) -> Optional[Dict]:
        """
        Get crypto fear & greed index.
        Note: This uses alternative.me API, not CoinGecko.
        
        Returns:
            Fear & greed index data
        """
        try:
            url = "https://api.alternative.me/fng/"
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            data = response.json()['data'][0]
            
            return {
                'value': int(data['value']),
                'value_classification': data['value_classification'],
                'timestamp': int(data['timestamp']),
                'time_until_update': data.get('time_until_update')
            }
        except Exception as e:
            logger.error(f"Error getting fear & greed index: {e}")
            return None
    
    def get_top_coins(self, vs_currency: str = 'usd', per_page: int = 100, page: int = 1) -> List[Dict]:
        """
        Get top coins by market cap.
        
        Args:
            vs_currency: Currency
            per_page: Results per page (max 250)
            page: Page number
            
        Returns:
            List of top coins
        """
        try:
            url = f"{self.BASE_URL}/coins/markets"
            params = {
                'vs_currency': vs_currency,
                'order': 'market_cap_desc',
                'per_page': min(per_page, 250),
                'page': page,
                'sparkline': 'false',
                'price_change_percentage': '1h,24h,7d'
            }
            response = self.session.get(url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            coins = []
            for coin in data:
                coins.append({
                    'id': coin['id'],
                    'symbol': coin['symbol'],
                    'name': coin['name'],
                    'rank': coin['market_cap_rank'],
                    'price': coin['current_price'],
                    'price_change_1h': coin.get('price_change_percentage_1h_in_currency'),
                    'price_change_24h': coin.get('price_change_percentage_24h'),
                    'price_change_7d': coin.get('price_change_percentage_7d_in_currency'),
                    'market_cap': coin['market_cap'],
                    'volume_24h': coin['total_volume'],
                    'circulating_supply': coin.get('circulating_supply')
                })
            
            return coins
        except Exception as e:
            logger.error(f"Error getting top coins: {e}")
            return []
    
    def search_coins(self, query: str) -> List[Dict]:
        """
        Search for coins by name or symbol.
        
        Args:
            query: Search query
            
        Returns:
            List of matching coins
        """
        try:
            url = f"{self.BASE_URL}/search"
            params = {'query': query}
            response = self.session.get(url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            return data.get('coins', [])
        except Exception as e:
            logger.error(f"Error searching coins: {e}")
            return []


if __name__ == "__main__":
    # Test CoinGecko client
    client = CoinGeckoClient()
    
    print("=== CoinGecko Client Test ===")
    
    # Get prices
    prices = client.get_price(['bitcoin', 'ethereum'], ['usd'])
    if prices:
        print(f"BTC: ${prices['bitcoin']['usd']:,.2f}")
        print(f"ETH: ${prices['ethereum']['usd']:,.2f}")
    
    # Get global data
    global_data = client.get_global_data()
    if global_data:
        print(f"\nTotal Market Cap: ${global_data['total_market_cap']:,.0f}")
        print(f"BTC Dominance: {global_data['btc_dominance']:.2f}%")
        print(f"ETH Dominance: {global_data['eth_dominance']:.2f}%")
    
    # Get fear & greed
    fg = client.get_fear_greed_index()
    if fg:
        print(f"\nFear & Greed Index: {fg['value']} ({fg['value_classification']})")
    
    # Get trending coins
    trending = client.get_trending_coins()
    if trending:
        print(f"\nTop 3 Trending:")
        for coin in trending[:3]:
            print(f"  {coin['name']} ({coin['symbol'].upper()})")
    
    print("\n✅ CoinGecko client test completed!")
