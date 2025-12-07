"""
Reddit sentiment analysis module.
Analyzes sentiment from Reddit posts in finance-related subreddits.
"""

import os
import logging
from typing import Dict, List, Optional
from datetime import datetime, timedelta
import pandas as pd
import requests
from collections import Counter
from utils.logger import setup_logger

logger = setup_logger(__name__)


class RedditSentimentAnalyzer:
    """
    Analyze sentiment from Reddit finance communities.
    
    Subreddits tracked:
    - r/wallstreetbets
    - r/stocks
    - r/investing
    - r/options
    """
    
    def __init__(
        self,
        client_id: Optional[str] = None,
        client_secret: Optional[str] = None,
        user_agent: Optional[str] = None
    ):
        """
        Initialize Reddit sentiment analyzer.
        
        Args:
            client_id: Reddit API client ID
            client_secret: Reddit API client secret
            user_agent: Reddit API user agent
        """
        self.client_id = client_id or os.getenv('REDDIT_CLIENT_ID')
        self.client_secret = client_secret or os.getenv('REDDIT_CLIENT_SECRET')
        self.user_agent = user_agent or os.getenv('REDDIT_USER_AGENT', 'TradingAI/1.0')
        
        self.access_token = None
        
        # Subreddits to monitor
        self.subreddits = [
            'wallstreetbets',
            'stocks',
            'investing',
            'options',
            'StockMarket'
        ]
        
        # Sentiment keywords
        self.bullish_words = [
            'moon', 'rocket', 'calls', 'buy', 'bullish', 'long', 'pump',
            'gain', 'profit', 'yolo', 'tendies', 'squeeze', 'breakout'
        ]
        
        self.bearish_words = [
            'puts', 'short', 'bearish', 'sell', 'crash', 'dump', 'loss',
            'decline', 'bubble', 'overvalued', 'correction'
        ]
        
        logger.info("RedditSentimentAnalyzer initialized")
        
    def authenticate(self) -> bool:
        """
        Authenticate with Reddit API.
        
        Returns:
            True if authentication successful
        """
        if not self.client_id or not self.client_secret:
            logger.warning("Reddit API credentials not set, using mock data")
            return False
            
        auth_url = 'https://www.reddit.com/api/v1/access_token'
        
        auth = requests.auth.HTTPBasicAuth(self.client_id, self.client_secret)
        headers = {'User-Agent': self.user_agent}
        data = {'grant_type': 'client_credentials'}
        
        try:
            response = requests.post(auth_url, auth=auth, headers=headers, data=data, timeout=10)
            response.raise_for_status()
            
            self.access_token = response.json()['access_token']
            logger.info("Reddit API authentication successful")
            return True
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Reddit API authentication failed: {e}")
            return False
            
    def fetch_subreddit_posts(
        self,
        subreddit: str,
        limit: int = 100,
        time_filter: str = 'day'
    ) -> List[Dict]:
        """
        Fetch recent posts from a subreddit.
        
        Args:
            subreddit: Subreddit name
            limit: Maximum number of posts to fetch
            time_filter: Time filter ('hour', 'day', 'week', 'month')
            
        Returns:
            List of post dictionaries
        """
        if not self.access_token:
            if not self.authenticate():
                return self._generate_mock_posts(subreddit)
                
        url = f'https://oauth.reddit.com/r/{subreddit}/hot'
        headers = {
            'Authorization': f'bearer {self.access_token}',
            'User-Agent': self.user_agent
        }
        params = {'limit': limit, 't': time_filter}
        
        try:
            response = requests.get(url, headers=headers, params=params, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            posts = data['data']['children']
            
            logger.info(f"Fetched {len(posts)} posts from r/{subreddit}")
            return [post['data'] for post in posts]
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Error fetching posts from r/{subreddit}: {e}")
            return self._generate_mock_posts(subreddit)
            
    def extract_ticker_mentions(self, text: str) -> List[str]:
        """
        Extract stock ticker mentions from text.
        
        Args:
            text: Text to analyze
            
        Returns:
            List of ticker symbols found
        """
        import re
        
        # Match $TICKER or TICKER pattern
        pattern = r'\$?([A-Z]{1,5})\b'
        matches = re.findall(pattern, text.upper())
        
        # Filter out common false positives
        blacklist = {'A', 'I', 'SO', 'IT', 'ON', 'IN', 'AT', 'BY', 'OR', 'TO', 'CEO', 'IPO', 'DD'}
        tickers = [m for m in matches if m not in blacklist and len(m) <= 5]
        
        return tickers
        
    def analyze_post_sentiment(self, post: Dict) -> Dict:
        """
        Analyze sentiment of a single post.
        
        Args:
            post: Reddit post dictionary
            
        Returns:
            Dictionary with sentiment analysis
        """
        title = post.get('title', '').lower()
        text = post.get('selftext', '').lower()
        combined = f"{title} {text}"
        
        # Count sentiment words
        bullish_count = sum(combined.count(word) for word in self.bullish_words)
        bearish_count = sum(combined.count(word) for word in self.bearish_words)
        
        # Calculate sentiment score
        total = bullish_count + bearish_count
        if total == 0:
            sentiment_score = 0.0
        else:
            sentiment_score = (bullish_count - bearish_count) / total
            
        # Extract tickers
        tickers = self.extract_ticker_mentions(post.get('title', ''))
        
        return {
            'post_id': post.get('id'),
            'title': post.get('title'),
            'score': post.get('score', 0),
            'num_comments': post.get('num_comments', 0),
            'created_utc': post.get('created_utc'),
            'subreddit': post.get('subreddit'),
            'bullish_count': bullish_count,
            'bearish_count': bearish_count,
            'sentiment_score': sentiment_score,
            'tickers': tickers
        }
        
    def get_ticker_sentiment(
        self,
        ticker: str,
        days_back: int = 7
    ) -> Dict:
        """
        Get aggregated sentiment for a specific ticker.
        
        Args:
            ticker: Stock ticker symbol
            days_back: Number of days to look back
            
        Returns:
            Dictionary with sentiment metrics
        """
        all_posts = []
        
        # Fetch from all subreddits
        for subreddit in self.subreddits:
            posts = self.fetch_subreddit_posts(subreddit, limit=100)
            all_posts.extend(posts)
            
        # Analyze all posts
        analyses = [self.analyze_post_sentiment(post) for post in all_posts]
        
        # Filter for ticker mentions
        ticker_posts = [
            a for a in analyses 
            if ticker.upper() in [t.upper() for t in a['tickers']]
        ]
        
        if not ticker_posts:
            logger.warning(f"No Reddit posts found mentioning {ticker}")
            return {
                'ticker': ticker,
                'mention_count': 0,
                'sentiment_score': 0.0,
                'avg_score': 0.0,
                'bullish_ratio': 0.0
            }
            
        # Calculate aggregate metrics
        mention_count = len(ticker_posts)
        avg_sentiment = sum(p['sentiment_score'] for p in ticker_posts) / mention_count
        avg_score = sum(p['score'] for p in ticker_posts) / mention_count
        
        bullish_posts = sum(1 for p in ticker_posts if p['sentiment_score'] > 0)
        bullish_ratio = bullish_posts / mention_count if mention_count > 0 else 0
        
        result = {
            'ticker': ticker,
            'mention_count': mention_count,
            'sentiment_score': avg_sentiment,
            'avg_score': avg_score,
            'bullish_ratio': bullish_ratio,
            'top_posts': sorted(ticker_posts, key=lambda x: x['score'], reverse=True)[:5]
        }
        
        logger.info(
            f"Reddit sentiment for {ticker}: "
            f"mentions={mention_count}, sentiment={avg_sentiment:.3f}, "
            f"bullish_ratio={bullish_ratio:.2%}"
        )
        
        return result
        
    def get_trending_tickers(self, min_mentions: int = 10) -> pd.DataFrame:
        """
        Get trending tickers across all subreddits.
        
        Args:
            min_mentions: Minimum number of mentions to include
            
        Returns:
            DataFrame with trending tickers and their metrics
        """
        all_posts = []
        
        # Fetch from all subreddits
        for subreddit in self.subreddits:
            posts = self.fetch_subreddit_posts(subreddit, limit=100)
            all_posts.extend(posts)
            
        # Analyze all posts
        analyses = [self.analyze_post_sentiment(post) for post in all_posts]
        
        # Count ticker mentions
        ticker_counter = Counter()
        ticker_sentiments = {}
        ticker_scores = {}
        
        for analysis in analyses:
            for ticker in analysis['tickers']:
                ticker = ticker.upper()
                ticker_counter[ticker] += 1
                
                if ticker not in ticker_sentiments:
                    ticker_sentiments[ticker] = []
                    ticker_scores[ticker] = []
                    
                ticker_sentiments[ticker].append(analysis['sentiment_score'])
                ticker_scores[ticker].append(analysis['score'])
                
        # Build DataFrame
        trending = []
        for ticker, count in ticker_counter.most_common():
            if count < min_mentions:
                continue
                
            trending.append({
                'ticker': ticker,
                'mentions': count,
                'avg_sentiment': sum(ticker_sentiments[ticker]) / len(ticker_sentiments[ticker]),
                'avg_post_score': sum(ticker_scores[ticker]) / len(ticker_scores[ticker]),
                'bullish_ratio': sum(1 for s in ticker_sentiments[ticker] if s > 0) / len(ticker_sentiments[ticker])
            })
            
        df = pd.DataFrame(trending)
        
        if not df.empty:
            df = df.sort_values('mentions', ascending=False)
            logger.info(f"Found {len(df)} trending tickers with {min_mentions}+ mentions")
        else:
            logger.warning("No trending tickers found")
            
        return df
        
    def _generate_mock_posts(self, subreddit: str) -> List[Dict]:
        """Generate mock posts when API is unavailable."""
        tickers = ['AAPL', 'TSLA', 'GME', 'AMC', 'SPY']
        
        return [
            {
                'id': f'mock_{i}',
                'title': f'Mock post about ${tickers[i % len(tickers)]} - bullish!',
                'selftext': 'This is a mock post for testing.',
                'score': 100 + i * 10,
                'num_comments': 20 + i * 2,
                'created_utc': datetime.now().timestamp(),
                'subreddit': subreddit
            }
            for i in range(10)
        ]
        
    def save_to_csv(self, df: pd.DataFrame, filename: str) -> None:
        """
        Save Reddit data to CSV file.
        
        Args:
            df: DataFrame to save
            filename: Output filename
        """
        output_path = os.path.join('data', 'raw', filename)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        df.to_csv(output_path, index=False)
        logger.info(f"Saved Reddit data to {output_path}")


def main():
    """Example usage of RedditSentimentAnalyzer."""
    analyzer = RedditSentimentAnalyzer()
    
    # Get sentiment for specific ticker
    sentiment = analyzer.get_ticker_sentiment('AAPL', days_back=7)
    print("\nReddit Sentiment for AAPL:")
    print(f"Mentions: {sentiment['mention_count']}")
    print(f"Sentiment Score: {sentiment['sentiment_score']:.3f}")
    print(f"Bullish Ratio: {sentiment['bullish_ratio']:.2%}")
    
    # Get trending tickers
    trending = analyzer.get_trending_tickers(min_mentions=5)
    print("\nTrending Tickers on Reddit:")
    print(trending.head(10))
    
    # Save to file
    analyzer.save_to_csv(trending, 'reddit_trending.csv')


if __name__ == '__main__':
    main()
