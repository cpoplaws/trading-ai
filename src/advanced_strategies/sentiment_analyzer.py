"""
Sentiment analysis integration for trading signals from Twitter, Reddit, and news sources.
Now with REAL API integrations!
"""
import pandas as pd
import numpy as np
import requests
import json
import os
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Union
import logging
import re
import time
from urllib.parse import quote
import warnings
warnings.filterwarnings('ignore')

# Try to import real API clients
try:
    from newsapi import NewsApiClient
    NEWSAPI_AVAILABLE = True
except ImportError:
    NEWSAPI_AVAILABLE = False
    logging.warning("newsapi-python not installed. Install with: pip install newsapi-python")

try:
    import praw
    PRAW_AVAILABLE = True
except ImportError:
    PRAW_AVAILABLE = False
    logging.warning("praw not installed. Install with: pip install praw")

logger = logging.getLogger(__name__)

class SentimentAnalyzer:
    """
    Sentiment analysis for financial markets using multiple data sources.
    Supports: NewsAPI (real), Reddit (real via PRAW), Twitter (simulated - requires paid tier)
    """
    
    def __init__(self, api_keys: Optional[Dict[str, str]] = None):
        """
        Initialize sentiment analyzer.
        
        Args:
            api_keys: Dictionary of API keys for different services
                     Keys: 'news_api_key', 'reddit_client_id', 'reddit_client_secret', 'reddit_user_agent'
        """
        self.api_keys = api_keys or {}
        
        # Load from environment if not provided
        if not self.api_keys.get('news_api_key'):
            self.api_keys['news_api_key'] = os.getenv('NEWS_API_KEY')
        if not self.api_keys.get('reddit_client_id'):
            self.api_keys['reddit_client_id'] = os.getenv('REDDIT_CLIENT_ID')
        if not self.api_keys.get('reddit_client_secret'):
            self.api_keys['reddit_client_secret'] = os.getenv('REDDIT_CLIENT_SECRET')
        if not self.api_keys.get('reddit_user_agent'):
            self.api_keys['reddit_user_agent'] = os.getenv('REDDIT_USER_AGENT', 'trading-ai/1.0')
        
        self.sentiment_cache = {}
        self.rate_limits = {
            'twitter': {'calls': 0, 'reset_time': datetime.now()},
            'reddit': {'calls': 0, 'reset_time': datetime.now()},
            'news': {'calls': 0, 'reset_time': datetime.now()}
        }
        
        # Initialize API clients
        self._init_news_api()
        self._init_reddit_api()
    
    def _init_news_api(self):
        """Initialize NewsAPI client."""
        if NEWSAPI_AVAILABLE and self.api_keys.get('news_api_key'):
            try:
                self.news_client = NewsApiClient(api_key=self.api_keys['news_api_key'])
                logger.info("NewsAPI client initialized successfully")
            except Exception as e:
                logger.error(f"Failed to initialize NewsAPI: {e}")
                self.news_client = None
        else:
            self.news_client = None
            if not NEWSAPI_AVAILABLE:
                logger.warning("NewsAPI not available - using simulated data")
            else:
                logger.warning("NEWS_API_KEY not set - using simulated data")
    
    def _init_reddit_api(self):
        """Initialize Reddit API client."""
        if PRAW_AVAILABLE and all([
            self.api_keys.get('reddit_client_id'),
            self.api_keys.get('reddit_client_secret'),
            self.api_keys.get('reddit_user_agent')
        ]):
            try:
                self.reddit_client = praw.Reddit(
                    client_id=self.api_keys['reddit_client_id'],
                    client_secret=self.api_keys['reddit_client_secret'],
                    user_agent=self.api_keys['reddit_user_agent']
                )
                # Test connection
                user_info = self.reddit_client.user.me()  # Test connection by fetching user info
                logger.info("Reddit API client initialized successfully (read-only mode)")
            except Exception as e:
                logger.warning(f"Reddit API initialization failed (using read-only): {e}")
                # Try read-only mode without authentication
                try:
                    self.reddit_client = praw.Reddit(
                        client_id=self.api_keys['reddit_client_id'],
                        client_secret=self.api_keys['reddit_client_secret'],
                        user_agent=self.api_keys['reddit_user_agent']
                    )
                    logger.info("Reddit API in read-only mode")
                except Exception as e2:
                    logger.error(f"Failed to initialize Reddit API: {e2}")
                    self.reddit_client = None
        else:
            self.reddit_client = None
            if not PRAW_AVAILABLE:
                logger.warning("PRAW not available - using simulated data")
            else:
                logger.warning("Reddit API credentials not set - using simulated data")
        
    def clean_text(self, text: str) -> str:
        """Clean and preprocess text for sentiment analysis."""
        if not text:
            return ""
        
        # Remove URLs
        text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
        
        # Remove mentions and hashtags (keep the text)
        text = re.sub(r'[@#](\w+)', r'\1', text)
        
        # Remove extra whitespace
        text = ' '.join(text.split())
        
        return text.strip()
    
    def analyze_text_sentiment(self, text: str) -> Dict:
        """
        Analyze sentiment of text using lexicon-based approach.
        
        Args:
            text: Text to analyze
            
        Returns:
            Sentiment analysis results
        """
        try:
            # Simple lexicon-based sentiment analysis
            positive_words = {
                'bullish', 'buy', 'long', 'pump', 'moon', 'rocket', 'gain', 'profit',
                'up', 'rise', 'surge', 'rally', 'breakout', 'strong', 'good', 'great',
                'excellent', 'awesome', 'amazing', 'bullish', 'optimistic', 'positive'
            }
            
            negative_words = {
                'bearish', 'sell', 'short', 'dump', 'crash', 'drop', 'loss', 'lose',
                'down', 'fall', 'decline', 'correction', 'weak', 'bad', 'terrible',
                'awful', 'horrible', 'bearish', 'pessimistic', 'negative', 'fear'
            }
            
            # Financial sentiment modifiers
            strong_positive = {'moon', 'rocket', 'breakout', 'surge'}
            strong_negative = {'crash', 'dump', 'collapse', 'plunge'}
            
            words = text.lower().split()
            
            pos_score = 0
            neg_score = 0
            
            for word in words:
                if word in strong_positive:
                    pos_score += 2
                elif word in positive_words:
                    pos_score += 1
                
                if word in strong_negative:
                    neg_score += 2
                elif word in negative_words:
                    neg_score += 1
            
            # Calculate compound sentiment
            total_score = pos_score - neg_score
            total_words = len(words)
            
            if total_words == 0:
                return {'compound': 0, 'positive': 0.5, 'negative': 0.5, 'neutral': 0}
            
            # Normalize scores
            compound = total_score / max(total_words, 1)
            compound = max(-1, min(1, compound))  # Clamp to [-1, 1]
            
            if compound > 0.1:
                sentiment_label = 'positive'
                positive = 0.5 + (compound * 0.5)
                negative = 0.5 - (compound * 0.5)
            elif compound < -0.1:
                sentiment_label = 'negative'
                positive = 0.5 + (compound * 0.5)
                negative = 0.5 - (compound * 0.5)
            else:
                sentiment_label = 'neutral'
                positive = 0.5
                negative = 0.5
            
            return {
                'compound': compound,
                'positive': max(0, positive),
                'negative': max(0, negative),
                'neutral': 1 - abs(compound),
                'label': sentiment_label
            }
            
        except Exception as e:
            logger.error(f"Error analyzing text sentiment: {e}")
            return {'compound': 0, 'positive': 0.5, 'negative': 0.5, 'neutral': 0.5}
    
    def get_reddit_sentiment(self, symbol: str, limit: int = 100) -> Dict:
        """
        Get sentiment data from Reddit using REAL API or simulated data as fallback.
        
        Args:
            symbol: Stock symbol
            limit: Number of posts to analyze
            
        Returns:
            Reddit sentiment analysis
        """
        try:
            # Check rate limits
            if self._check_rate_limit('reddit'):
                time.sleep(1)
            
            logger.info(f"Analyzing Reddit sentiment for {symbol}")
            
            posts_texts = []
            
            # Try real Reddit API first
            if self.reddit_client:
                try:
                    # Search in wallstreetbets and stocks subreddits
                    subreddits = ['wallstreetbets', 'stocks', 'investing', 'StockMarket']
                    
                    for subreddit_name in subreddits:
                        subreddit = self.reddit_client.subreddit(subreddit_name)
                        
                        # Search for posts mentioning the symbol
                        for post in subreddit.search(symbol, limit=limit//len(subreddits), time_filter='week'):
                            post_text = f"{post.title} {post.selftext}"
                            posts_texts.append(self.clean_text(post_text))
                        
                        # Also check hot posts
                        for post in subreddit.hot(limit=10):
                            if symbol.upper() in post.title.upper() or symbol.upper() in post.selftext.upper():
                                post_text = f"{post.title} {post.selftext}"
                                posts_texts.append(self.clean_text(post_text))
                    
                    logger.info(f"Retrieved {len(posts_texts)} Reddit posts for {symbol}")
                    
                except Exception as e:
                    logger.warning(f"Reddit API error, falling back to simulation: {e}")
                    posts_texts = []
            
            # Fall back to simulated data if API fails or not available
            if not posts_texts:
                logger.info(f"Using simulated Reddit data for {symbol}")
                simulated_posts = [
                    f"{symbol} looking bullish, strong fundamentals",
                    f"Should I buy {symbol}? Seems undervalued",
                    f"{symbol} to the moon! 🚀",
                    f"Bearish on {symbol}, overpriced",
                    f"{symbol} great earnings, buying more",
                    f"Selling my {symbol} position, too risky",
                    f"{symbol} sideways movement, holding",
                    f"DCA into {symbol}, long term play"
                ]
                
                import random
                posts_texts = random.sample(simulated_posts, min(limit//20, len(simulated_posts)))
            
            # Analyze sentiment of collected posts
            sentiments = []
            for post in posts_texts[:limit]:  # Limit to avoid processing too many
                if post.strip():
                    sentiment = self.analyze_text_sentiment(post)
                    sentiments.append(sentiment)
            
            if not sentiments:
                return self._empty_sentiment_result('reddit')
            
            # Aggregate sentiment
            avg_compound = np.mean([s['compound'] for s in sentiments])
            avg_positive = np.mean([s['positive'] for s in sentiments])
            avg_negative = np.mean([s['negative'] for s in sentiments])
            
            # Calculate confidence based on volume and consistency
            volume_score = min(len(sentiments) / 50, 1.0)
            consistency = 1 - min(np.std([s['compound'] for s in sentiments]), 1.0)
            confidence = (volume_score + consistency) / 2
            
            return {
                'source': 'reddit',
                'symbol': symbol,
                'sentiment_score': avg_compound,
                'positive_ratio': avg_positive,
                'negative_ratio': avg_negative,
                'confidence': confidence,
                'post_count': len(sentiments),
                'timestamp': datetime.now(),
                'real_api': self.reddit_client is not None
            }
            
        except Exception as e:
            logger.error(f"Error getting Reddit sentiment: {e}")
            return self._empty_sentiment_result('reddit')
    
    def get_twitter_sentiment(self, symbol: str, limit: int = 100) -> Dict:
        """
        Get sentiment data from Twitter (simulated for demo).
        
        Args:
            symbol: Stock symbol
            limit: Number of tweets to analyze
            
        Returns:
            Twitter sentiment analysis
        """
        try:
            # Check rate limits
            if self._check_rate_limit('twitter'):
                time.sleep(1)
            
            logger.info(f"Analyzing Twitter sentiment for {symbol}")
            
            # Simulate Twitter tweets
            simulated_tweets = [
                f"$${symbol} breaking resistance levels 📈",
                f"Bullish on $${symbol} fundamentals are solid",
                f"$${symbol} chart looking weak, might sell",
                f"Great quarter for $${symbol}, buying the dip",
                f"$${symbol} overvalued, waiting for correction",
                f"$${symbol} long-term hold, accumulating",
                f"Profit taking on $${symbol} after good run",
                f"$${symbol} technical analysis shows uptrend"
            ]
            
            # Add randomness
            import random
            tweets_sample = random.sample(simulated_tweets, min(limit//15, len(simulated_tweets)))
            
            sentiments = []
            for tweet in tweets_sample:
                sentiment = self.analyze_text_sentiment(tweet)
                sentiments.append(sentiment)
            
            if not sentiments:
                return self._empty_sentiment_result('twitter')
            
            # Aggregate sentiment
            avg_compound = np.mean([s['compound'] for s in sentiments])
            avg_positive = np.mean([s['positive'] for s in sentiments])
            avg_negative = np.mean([s['negative'] for s in sentiments])
            
            # Twitter sentiment is often more volatile
            confidence = min(len(sentiments) / 30, 0.8)  # Cap confidence due to noise
            
            return {
                'source': 'twitter',
                'symbol': symbol,
                'sentiment_score': avg_compound,
                'positive_ratio': avg_positive,
                'negative_ratio': avg_negative,
                'confidence': confidence,
                'tweet_count': len(sentiments),
                'timestamp': datetime.now()
            }
            
        except Exception as e:
            logger.error(f"Error getting Twitter sentiment: {e}")
            return self._empty_sentiment_result('twitter')
    
    def get_news_sentiment(self, symbol: str, days_back: int = 7) -> Dict:
        """
        Get sentiment from financial news using REAL NewsAPI or simulated data as fallback.
        
        Args:
            symbol: Stock symbol
            days_back: Days of news to analyze
            
        Returns:
            News sentiment analysis
        """
        try:
            # Check rate limits
            if self._check_rate_limit('news'):
                time.sleep(1)
            
            logger.info(f"Analyzing news sentiment for {symbol}")
            
            headlines = []
            
            # Try real NewsAPI first
            if self.news_client:
                try:
                    # Get company name mapping for better search
                    company_names = {
                        'AAPL': 'Apple',
                        'MSFT': 'Microsoft',
                        'GOOGL': 'Google',
                        'AMZN': 'Amazon',
                        'TSLA': 'Tesla',
                        'META': 'Meta',
                        'NVDA': 'Nvidia',
                        'SPY': 'S&P 500',
                        'QQQ': 'Nasdaq'
                    }
                    
                    search_query = company_names.get(symbol.upper(), symbol)
                    
                    # Calculate date range
                    to_date = datetime.now()
                    from_date = to_date - timedelta(days=days_back)
                    
                    # Search for news articles
                    response = self.news_client.get_everything(
                        q=search_query,
                        from_param=from_date.strftime('%Y-%m-%d'),
                        to=to_date.strftime('%Y-%m-%d'),
                        language='en',
                        sort_by='relevancy',
                        page_size=min(100, days_back * 10)
                    )
                    
                    if response['status'] == 'ok' and response['articles']:
                        for article in response['articles']:
                            title = article.get('title', '')
                            description = article.get('description', '')
                            content = f"{title}. {description}"
                            headlines.append(self.clean_text(content))
                        
                        logger.info(f"Retrieved {len(headlines)} news articles for {symbol}")
                    else:
                        logger.warning(f"No news articles found for {symbol}")
                
                except Exception as e:
                    logger.warning(f"NewsAPI error, falling back to simulation: {e}")
                    headlines = []
            
            # Fall back to simulated data if API fails or not available
            if not headlines:
                logger.info(f"Using simulated news data for {symbol}")
                simulated_headlines = [
                    f"{symbol} Reports Strong Q3 Earnings, Beats Expectations",
                    f"Analysts Upgrade {symbol} Target Price Following Innovation Announcement",
                    f"{symbol} Faces Regulatory Challenges in Key Market",
                    f"Institutional Investors Increase Stakes in {symbol}",
                    f"{symbol} CEO Optimistic About Future Growth Prospects",
                    f"Market Volatility Impacts {symbol} Share Price",
                    f"{symbol} Announces Strategic Partnership Deal",
                    f"Economic Headwinds May Affect {symbol} Performance"
                ]
                
                import random
                headlines = random.sample(simulated_headlines, min(days_back, len(simulated_headlines)))
            
            # Analyze sentiment of collected headlines
            sentiments = []
            for headline in headlines:
                if headline.strip():
                    sentiment = self.analyze_text_sentiment(headline)
                    sentiments.append(sentiment)
            
            if not sentiments:
                return self._empty_sentiment_result('news')
            
            # News sentiment is typically more reliable
            avg_compound = np.mean([s['compound'] for s in sentiments])
            avg_positive = np.mean([s['positive'] for s in sentiments])
            avg_negative = np.mean([s['negative'] for s in sentiments])
            
            # News gets higher base confidence due to editorial standards
            base_confidence = 0.7
            volume_bonus = min(len(sentiments) / 20, 0.2)
            confidence = min(base_confidence + volume_bonus, 0.95)
            
            return {
                'source': 'news',
                'symbol': symbol,
                'sentiment_score': avg_compound,
                'positive_ratio': avg_positive,
                'negative_ratio': avg_negative,
                'confidence': confidence,
                'article_count': len(sentiments),
                'timestamp': datetime.now(),
                'real_api': self.news_client is not None
            }
            
        except Exception as e:
            logger.error(f"Error getting news sentiment: {e}")
            return self._empty_sentiment_result('news')
    
    def _check_rate_limit(self, source: str) -> bool:
        """Check if we're hitting rate limits."""
        now = datetime.now()
        rate_info = self.rate_limits[source]
        
        # Reset counter every hour
        if now - rate_info['reset_time'] > timedelta(hours=1):
            rate_info['calls'] = 0
            rate_info['reset_time'] = now
        
        rate_info['calls'] += 1
        return rate_info['calls'] > 100  # Limit to 100 calls per hour
    
    def _empty_sentiment_result(self, source: str) -> Dict:
        """Return empty sentiment result."""
        return {
            'source': source,
            'sentiment_score': 0,
            'positive_ratio': 0.5,
            'negative_ratio': 0.5,
            'confidence': 0,
            'timestamp': datetime.now()
        }
    
    def aggregate_sentiment_signals(self, symbol: str, 
                                  include_sources: List[str] = None) -> Dict:
        """
        Aggregate sentiment from multiple sources.
        
        Args:
            symbol: Stock symbol
            include_sources: Sources to include ('twitter', 'reddit', 'news')
            
        Returns:
            Aggregated sentiment analysis
        """
        try:
            if include_sources is None:
                include_sources = ['twitter', 'reddit', 'news']
            
            logger.info(f"Aggregating sentiment signals for {symbol}")
            
            # Collect sentiment from all sources
            sentiment_data = {}
            
            if 'reddit' in include_sources:
                sentiment_data['reddit'] = self.get_reddit_sentiment(symbol)
            
            if 'twitter' in include_sources:
                sentiment_data['twitter'] = self.get_twitter_sentiment(symbol)
            
            if 'news' in include_sources:
                sentiment_data['news'] = self.get_news_sentiment(symbol)
            
            # Remove failed sources
            valid_sources = {k: v for k, v in sentiment_data.items() 
                           if v.get('confidence', 0) > 0}
            
            if not valid_sources:
                return {
                    'symbol': symbol,
                    'overall_sentiment': 0,
                    'confidence': 0,
                    'signal': 'HOLD',
                    'sources': {},
                    'timestamp': datetime.now()
                }
            
            # Calculate weighted average sentiment
            total_weight = sum(data['confidence'] for data in valid_sources.values())
            if total_weight == 0:
                weighted_sentiment = 0
            else:
                weighted_sentiment = sum(
                    data['sentiment_score'] * data['confidence'] 
                    for data in valid_sources.values()
                ) / total_weight
            
            # Calculate overall confidence
            overall_confidence = np.mean([data['confidence'] for data in valid_sources.values()])
            
            # Generate trading signal
            sentiment_threshold = 0.1
            if weighted_sentiment > sentiment_threshold and overall_confidence > 0.5:
                signal = 'BUY'
                signal_strength = min(weighted_sentiment * overall_confidence * 2, 1.0)
            elif weighted_sentiment < -sentiment_threshold and overall_confidence > 0.5:
                signal = 'SELL'
                signal_strength = min(abs(weighted_sentiment) * overall_confidence * 2, 1.0)
            else:
                signal = 'HOLD'
                signal_strength = overall_confidence * 0.5
            
            # Additional metrics
            source_consensus = self._calculate_source_consensus(valid_sources)
            sentiment_trend = self._calculate_sentiment_trend(symbol, weighted_sentiment)
            
            return {
                'symbol': symbol,
                'overall_sentiment': weighted_sentiment,
                'confidence': overall_confidence,
                'signal': signal,
                'signal_strength': signal_strength,
                'source_consensus': source_consensus,
                'sentiment_trend': sentiment_trend,
                'sources': valid_sources,
                'source_count': len(valid_sources),
                'timestamp': datetime.now()
            }
            
        except Exception as e:
            logger.error(f"Error aggregating sentiment signals: {e}")
            return {
                'symbol': symbol,
                'overall_sentiment': 0,
                'confidence': 0,
                'signal': 'HOLD',
                'error': str(e)
            }
    
    def _calculate_source_consensus(self, sources: Dict) -> float:
        """Calculate how much sources agree on sentiment direction."""
        if len(sources) < 2:
            return 1.0
        
        sentiments = [data['sentiment_score'] for data in sources.values()]
        
        # Check if all sentiments have same sign (all positive or all negative)
        positive_count = sum(1 for s in sentiments if s > 0.05)
        negative_count = sum(1 for s in sentiments if s < -0.05)
        neutral_count = len(sentiments) - positive_count - negative_count
        
        # Calculate consensus
        total_sources = len(sentiments)
        max_agreement = max(positive_count, negative_count, neutral_count)
        consensus = max_agreement / total_sources
        
        return consensus
    
    def _calculate_sentiment_trend(self, symbol: str, current_sentiment: float) -> Dict:
        """Calculate sentiment trend (simplified version)."""
        # In a real implementation, you'd store historical sentiment
        # For demo, we'll simulate trend
        
        cache_key = f"{symbol}_sentiment_history"
        if cache_key not in self.sentiment_cache:
            self.sentiment_cache[cache_key] = []
        
        history = self.sentiment_cache[cache_key]
        history.append({'sentiment': current_sentiment, 'timestamp': datetime.now()})
        
        # Keep only last 10 readings
        history = history[-10:]
        self.sentiment_cache[cache_key] = history
        
        if len(history) < 3:
            return {'trend': 'insufficient_data', 'slope': 0}
        
        # Calculate simple trend
        recent_avg = np.mean([h['sentiment'] for h in history[-3:]])
        older_avg = np.mean([h['sentiment'] for h in history[:-3]])
        
        slope = recent_avg - older_avg
        
        if slope > 0.05:
            trend = 'improving'
        elif slope < -0.05:
            trend = 'deteriorating'
        else:
            trend = 'stable'
        
        return {'trend': trend, 'slope': slope}

# Example usage
if __name__ == "__main__":
    print("📊 Sentiment Analysis Demo")
    print("=" * 40)
    
    analyzer = SentimentAnalyzer()
    
    # Demo symbols
    symbols = ['AAPL', 'TSLA', 'MSFT']
    
    for symbol in symbols:
        print(f"\n🔍 Analyzing sentiment for {symbol}")
        
        # Get aggregated sentiment
        sentiment_result = analyzer.aggregate_sentiment_signals(
            symbol, 
            include_sources=['twitter', 'reddit', 'news']
        )
        
        if 'error' not in sentiment_result:
            print(f"Overall Sentiment: {sentiment_result['overall_sentiment']:.3f}")
            print(f"Confidence: {sentiment_result['confidence']:.3f}")
            print(f"Signal: {sentiment_result['signal']} (strength: {sentiment_result.get('signal_strength', 0):.3f})")
            print(f"Source Consensus: {sentiment_result['source_consensus']:.3f}")
            print(f"Sources Used: {sentiment_result['source_count']}")
            
            # Show individual source results
            for source, data in sentiment_result['sources'].items():
                print(f"  {source.title()}: {data['sentiment_score']:.3f} (confidence: {data['confidence']:.3f})")
        else:
            print(f"❌ Error: {sentiment_result['error']}")
    
    print("\n✅ Sentiment analysis demo completed!")