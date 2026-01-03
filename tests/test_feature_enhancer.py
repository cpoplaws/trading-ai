import os
import sys

import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from feature_engineering.feature_enhancer import FeatureEnhancer


class DummyMacroFetcher:
    def get_macro_summary(self):
        return {
            "regime": {"regime": "EXPANSION", "regime_score": 2, "confidence": 0.8},
            "yield_curve": {"spread": 0.45, "confidence": 0.7},
            "consumer_sentiment": 80.0,
        }


class DummyNewsScraper:
    def fetch_ticker_news(self, ticker: str, days_back: int = 7):
        return pd.DataFrame(
            [
                {"title": "Stock surges on earnings beat", "publishedAt": "2024-01-01"},
                {"title": "Analysts stay bullish", "publishedAt": "2024-01-02"},
            ]
        )

    def calculate_news_sentiment_score(self, articles: pd.DataFrame) -> float:
        return 0.5


class DummyRedditAnalyzer:
    def get_ticker_sentiment(self, ticker: str, days_back: int = 7):
        return {"sentiment_score": 0.25, "mention_count": 12}


def build_price_frame(rows: int = 60) -> pd.DataFrame:
    dates = pd.date_range("2024-01-01", periods=rows, freq="D")
    close = pd.Series(range(rows), index=dates, dtype=float) + 100
    data = pd.DataFrame(
        {
            "close": close.values,
            "high": close.values + 1,
            "low": close.values - 1,
            "volume": 1_000_000,
        },
        index=dates,
    )
    return data


def test_enhancer_merges_external_signals():
    price_data = build_price_frame()

    enhancer = FeatureEnhancer(
        price_data=price_data,
        macro_fetcher=DummyMacroFetcher(),
        news_scraper=DummyNewsScraper(),
        reddit_analyzer=DummyRedditAnalyzer(),
    )

    features = enhancer.enhance_with_external_signals("AAPL", days_back=3)

    assert not features.empty
    last_row = features.iloc[-1]

    assert "macro_yield_curve_spread" in features.columns
    assert last_row["macro_yield_curve_spread"] == 0.45
    assert last_row["macro_regime_score"] == 2
    assert last_row["macro_consumer_sentiment"] == 80.0

    assert last_row["news_sentiment"] == 0.5
    assert last_row["news_volume"] == 2

    assert last_row["reddit_sentiment"] == 0.25
    assert last_row["reddit_mentions"] == 12
    assert last_row["regime_expansion"] == 1
