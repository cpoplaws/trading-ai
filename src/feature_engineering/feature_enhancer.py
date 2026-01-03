"""
feature_enhancer.py - Merges macro, news, and social sentiment signals into the technical feature set.
"""
from typing import Optional

import pandas as pd

from data_ingestion.macro_data import MacroDataFetcher
from data_ingestion.news_scraper import NewsScraper
from data_ingestion.reddit_sentiment import RedditSentimentAnalyzer
from feature_engineering.feature_generator import FeatureGenerator
from utils.logger import setup_logger

logger = setup_logger(__name__)


class FeatureEnhancer:
    """
    Orchestrates external data ingestion modules and enriches the base technical feature set.
    """

    def __init__(
        self,
        price_data: pd.DataFrame,
        macro_fetcher: Optional[MacroDataFetcher] = None,
        news_scraper: Optional[NewsScraper] = None,
        reddit_analyzer: Optional[RedditSentimentAnalyzer] = None,
    ):
        self.feature_generator = FeatureGenerator(price_data)
        self.macro_fetcher = macro_fetcher or MacroDataFetcher()
        self.news_scraper = news_scraper or NewsScraper()
        self.reddit_analyzer = reddit_analyzer or RedditSentimentAnalyzer()

    def enhance_with_external_signals(self, ticker: str, days_back: int = 7) -> pd.DataFrame:
        """
        Pull macro, news, and Reddit sentiment signals and merge them into the engineered feature set.

        Args:
            ticker: Symbol to collect news and social sentiment for.
            days_back: Lookback window for content collection.

        Returns:
            DataFrame of enhanced features aligned to the provided price data.
        """
        self._add_macro_context()
        self._add_news_context(ticker, days_back)
        self._add_reddit_context(ticker, days_back)

        return self.feature_generator.generate_features()

    def _add_macro_context(self) -> None:
        summary = {}
        try:
            summary = self.macro_fetcher.get_macro_summary()
        except Exception as exc:  # pragma: no cover - defensive path
            logger.warning(f"Macro data fetch failed, skipping macro features: {exc}")

        if not summary:
            return

        macro_features = {}

        regime_data = summary.get("regime")
        regime_dict = regime_data if isinstance(regime_data, dict) else {}
        regime_label = regime_dict.get("regime", "")
        if regime_dict:
            macro_features["regime_score"] = regime_dict.get("regime_score")
            macro_features["regime_confidence"] = regime_dict.get("confidence")

        yield_curve = summary.get("yield_curve", {})
        if isinstance(yield_curve, dict):
            macro_features["yield_curve_spread"] = yield_curve.get("spread")
            macro_features["yield_curve_confidence"] = yield_curve.get("confidence")

        consumer_sentiment = summary.get("consumer_sentiment")
        if consumer_sentiment is not None:
            macro_features["consumer_sentiment"] = consumer_sentiment

        filtered_macro = {k: v for k, v in macro_features.items() if v is not None}
        if filtered_macro:
            self.feature_generator.add_macro_features(filtered_macro)

        if regime_label:
            self.feature_generator.add_market_regime(regime_label.lower())

    def _add_news_context(self, ticker: str, days_back: int) -> None:
        try:
            articles = self.news_scraper.fetch_ticker_news(ticker, days_back)
            if articles is None:
                articles = pd.DataFrame()
            sentiment = (
                self.news_scraper.calculate_news_sentiment_score(articles) if not articles.empty else 0.0
            )
            volume = len(articles)
            self.feature_generator.add_news_sentiment(sentiment, volume)
        except Exception as exc:  # pragma: no cover - defensive path
            logger.warning(f"News sentiment ingestion failed, defaulting to neutral: {exc}")
            self.feature_generator.add_news_sentiment(0.0, 0)

    def _add_reddit_context(self, ticker: str, days_back: int) -> None:
        try:
            reddit_response = self.reddit_analyzer.get_ticker_sentiment(ticker, days_back)
            sentiment_dict = reddit_response if isinstance(reddit_response, dict) else {}
            score = sentiment_dict.get("sentiment_score", 0.0)
            mentions = sentiment_dict.get("mention_count", 0)
            self.feature_generator.add_social_sentiment(score, mentions)
        except Exception as exc:  # pragma: no cover - defensive path
            logger.warning(f"Reddit sentiment ingestion failed, defaulting to neutral: {exc}")
            self.feature_generator.add_social_sentiment(0.0, 0)
