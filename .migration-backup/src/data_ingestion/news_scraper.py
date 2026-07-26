"""
News scraping and analysis module.
Fetches financial news from multiple sources and extracts relevant signals.
"""

import os
import logging
from typing import Dict, List, Optional
from datetime import datetime, timedelta
import requests
from bs4 import BeautifulSoup
import pandas as pd
from ..utils.logger import setup_logger

logger = setup_logger(__name__)


class NewsScraper:
    """
    Scrape and analyze financial news from multiple sources.
    
    Sources:
    - NewsAPI (news aggregator)
    - Finviz (stock-specific news)
    - RSS feeds
    """
    
    def __init__(self, newsapi_key: Optional[str] = None):
        """
        Initialize news scraper.
        
        Args:
            newsapi_key: NewsAPI key (get from https://newsapi.org)
        """
        self.newsapi_key = newsapi_key or os.getenv('NEWSAPI_API_KEY')
        self.newsapi_url = 'https://newsapi.org/v2/everything'
        
        # Financial news sources
        self.sources = [
            'bloomberg',
            'reuters',
            'financial-times',
            'the-wall-street-journal',
            'cnbc'
        ]
        
        # Keywords for filtering
        self.keywords = [
            'stock market',
            'federal reserve',
            'interest rate',
            'earnings',
            'recession',
            'inflation',
            'gdp',
            'unemployment'
        ]
        
        logger.info("NewsScraper initialized")
        
    def fetch_newsapi_articles(
        self,
        query: str,
        days_back: int = 7,
        language: str = 'en'
    ) -> List[Dict]:
        """
        Fetch news articles from NewsAPI.
        
        Args:
            query: Search query
            days_back: Number of days to look back
            language: Article language
            
        Returns:
            List of article dictionaries
        """
        if not self.newsapi_key:
            logger.warning("NewsAPI key not set, using mock data")
            return self._generate_mock_articles(query)
            
        from_date = (datetime.now() - timedelta(days=days_back)).strftime('%Y-%m-%d')
        
        params = {
            'q': query,
            'from': from_date,
            'language': language,
            'sortBy': 'publishedAt',
            'apiKey': self.newsapi_key
        }
        
        try:
            response = requests.get(self.newsapi_url, params=params, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            articles = data.get('articles', [])
            
            logger.info(f"Fetched {len(articles)} articles for query: {query}")
            return articles
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Error fetching NewsAPI articles: {e}")
            return self._generate_mock_articles(query)
            
    def fetch_finviz_news(self, ticker: str) -> List[Dict]:
        """
        Scrape news for a specific ticker from Finviz.
        
        Args:
            ticker: Stock ticker symbol
            
        Returns:
            List of news dictionaries
        """
        url = f'https://finviz.com/quote.ashx?t={ticker}'
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        
        try:
            response = requests.get(url, headers=headers, timeout=10)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            news_table = soup.find('table', {'class': 'fullview-news-outer'})
            
            if not news_table:
                logger.warning(f"No news table found for {ticker}")
                return []
                
            articles = []
            for row in news_table.find_all('tr'):
                link = row.find('a')
                if link:
                    articles.append({
                        'title': link.text.strip(),
                        'url': link.get('href'),
                        'source': 'finviz',
                        'ticker': ticker,
                        'timestamp': datetime.now().isoformat()
                    })
                    
            logger.info(f"Scraped {len(articles)} articles for {ticker} from Finviz")
            return articles
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Error scraping Finviz for {ticker}: {e}")
            return []
            
    def fetch_market_news(self, days_back: int = 7) -> pd.DataFrame:
        """
        Fetch general market news from all sources.
        
        Args:
            days_back: Number of days to look back
            
        Returns:
            DataFrame with news articles
        """
        all_articles = []
        
        # Fetch from NewsAPI for each keyword
        for keyword in self.keywords[:3]:  # Limit to avoid rate limits
            articles = self.fetch_newsapi_articles(keyword, days_back)
            all_articles.extend(articles)
            
        if not all_articles:
            logger.warning("No market news fetched")
            return pd.DataFrame()
            
        # Convert to DataFrame
        df = pd.DataFrame(all_articles)
        
        # Clean and standardize
        if 'publishedAt' in df.columns:
            df['published_at'] = pd.to_datetime(df['publishedAt'])
        else:
            df['published_at'] = datetime.now()
            
        df['title'] = df.get('title', '')
        df['description'] = df.get('description', '')
        df['url'] = df.get('url', '')
        df['source_name'] = df.get('source', {}).apply(lambda x: x.get('name', '') if isinstance(x, dict) else '')
        
        # Remove duplicates
        df = df.drop_duplicates(subset=['title'], keep='first')
        
        logger.info(f"Compiled {len(df)} unique market news articles")
        return df[['published_at', 'title', 'description', 'url', 'source_name']]
        
    def fetch_ticker_news(self, ticker: str, days_back: int = 7) -> pd.DataFrame:
        """
        Fetch news specific to a ticker.
        
        Args:
            ticker: Stock ticker symbol
            days_back: Number of days to look back
            
        Returns:
            DataFrame with ticker-specific news
        """
        # Fetch from multiple sources
        articles = []
        
        # NewsAPI
        newsapi_articles = self.fetch_newsapi_articles(ticker, days_back)
        articles.extend(newsapi_articles)
        
        # Finviz
        finviz_articles = self.fetch_finviz_news(ticker)
        articles.extend(finviz_articles)
        
        if not articles:
            logger.warning(f"No news found for {ticker}")
            return pd.DataFrame()
            
        df = pd.DataFrame(articles)
        
        # Standardize columns
        if 'publishedAt' in df.columns:
            df['published_at'] = pd.to_datetime(df['publishedAt'])
        elif 'timestamp' in df.columns:
            df['published_at'] = pd.to_datetime(df['timestamp'])
        else:
            df['published_at'] = datetime.now()
            
        df['ticker'] = ticker
        
        # Remove duplicates
        df = df.drop_duplicates(subset=['title'], keep='first')
        
        logger.info(f"Compiled {len(df)} unique articles for {ticker}")
        return df
        
    def calculate_news_sentiment_score(self, articles: pd.DataFrame) -> float:
        """
        Calculate aggregate sentiment from news headlines.
        
        Args:
            articles: DataFrame with news articles
            
        Returns:
            Sentiment score between -1 (negative) and 1 (positive)
        """
        if articles.empty:
            return 0.0
            
        # Simple keyword-based sentiment
        positive_words = ['surge', 'gain', 'profit', 'growth', 'bullish', 'rally', 'beat', 'outperform']
        negative_words = ['fall', 'loss', 'decline', 'bearish', 'crash', 'miss', 'underperform', 'warning']
        
        titles = articles['title'].fillna('').str.lower()
        
        positive_count = sum(titles.str.contains('|'.join(positive_words), case=False))
        negative_count = sum(titles.str.contains('|'.join(negative_words), case=False))
        
        total = len(articles)
        if total == 0:
            return 0.0
            
        # Calculate net sentiment
        sentiment = (positive_count - negative_count) / total
        
        logger.info(f"News sentiment: {sentiment:.3f} (pos={positive_count}, neg={negative_count}, total={total})")
        return sentiment
        
    def get_news_volume(self, articles: pd.DataFrame) -> int:
        """
        Get news volume (article count).
        
        Args:
            articles: DataFrame with news articles
            
        Returns:
            Number of articles
        """
        return len(articles)
        
    def _generate_mock_articles(self, query: str) -> List[Dict]:
        """Generate mock articles when API is unavailable."""
        return [
            {
                'title': f'Mock article about {query}',
                'description': f'This is a mock article for testing purposes.',
                'url': 'https://example.com',
                'publishedAt': datetime.now().isoformat(),
                'source': {'name': 'MockNews'}
            }
            for _ in range(5)
        ]
        
    def save_to_csv(self, df: pd.DataFrame, filename: str) -> None:
        """
        Save news data to CSV file.
        
        Args:
            df: DataFrame to save
            filename: Output filename
        """
        output_path = os.path.join('data', 'raw', filename)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        df.to_csv(output_path, index=False)
        logger.info(f"Saved news data to {output_path}")


def main():
    """Example usage of NewsScraper."""
    scraper = NewsScraper()
    
    # Fetch market news
    market_news = scraper.fetch_market_news(days_back=7)
    print("\nMarket News:")
    print(market_news.head())
    
    # Fetch ticker-specific news
    ticker_news = scraper.fetch_ticker_news('AAPL', days_back=7)
    print(f"\nNews for AAPL:")
    print(ticker_news.head())
    
    # Calculate sentiment
    sentiment = scraper.calculate_news_sentiment_score(ticker_news)
    print(f"\nNews Sentiment Score: {sentiment:.3f}")
    
    # Save to file
    scraper.save_to_csv(market_news, 'market_news.csv')


if __name__ == '__main__':
    main()
