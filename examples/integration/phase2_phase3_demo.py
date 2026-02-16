"""
Phase 2 + Phase 3 Integration Demo
Shows broker integration + intelligence network working together.
"""

import os
import sys
from datetime import datetime, timedelta

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from data_ingestion.fetch_data import fetch_stock_data
from data_ingestion.macro_data import MacroDataFetcher
from data_ingestion.news_scraper import NewsScraper
from data_ingestion.reddit_sentiment import RedditSentimentAnalyzer
from feature_engineering.feature_generator import FeatureGenerator
from modeling.train_model import train_model
from execution.broker_interface import OrderType, OrderSide
from execution.alpaca_broker import AlpacaBroker
from execution.order_manager import OrderManager
from execution.portfolio_tracker import PortfolioTracker
from utils.logger import setup_logger

logger = setup_logger(__name__)


def main():
    """
    Comprehensive demo of Phase 2 + Phase 3 integration.
    """
    print("=" * 80)
    print("TRADING AI - PHASE 2 + 3 INTEGRATION DEMO")
    print("=" * 80)
    print()
    
    # Configuration
    ticker = 'AAPL'
    start_date = (datetime.now() - timedelta(days=365*2)).strftime('%Y-%m-%d')
    end_date = datetime.now().strftime('%Y-%m-%d')
    
    # =================================================================
    # PHASE 1: Data Ingestion + Feature Engineering (Already Complete)
    # =================================================================
    print("📊 PHASE 1: FETCHING PRICE DATA")
    print("-" * 80)
    
    df = fetch_stock_data(ticker, start_date, end_date)
    print(f"✅ Fetched {len(df)} days of price data for {ticker}")
    print(f"   Latest Close: ${df['Close'].iloc[-1]:.2f}")
    print()
    
    # =================================================================
    # PHASE 3: Intelligence Network (NEW!)
    # =================================================================
    print("🧠 PHASE 3: INTELLIGENCE NETWORK")
    print("-" * 80)
    
    # 3.1: Macro Economic Data
    print("\n1️⃣  Macro Economic Indicators:")
    macro_fetcher = MacroDataFetcher()
    macro_indicators = macro_fetcher.get_latest_indicators()
    
    for name, value in list(macro_indicators.items())[:5]:
        print(f"   • {name}: {value:.2f}")
    
    regime = macro_fetcher.calculate_macro_regime()
    print(f"   📈 Market Regime: {regime.upper()}")
    
    # 3.2: News Sentiment
    print("\n2️⃣  News Sentiment Analysis:")
    news_scraper = NewsScraper()
    ticker_news = news_scraper.fetch_ticker_news(ticker, days_back=7)
    news_sentiment = news_scraper.calculate_news_sentiment_score(ticker_news)
    news_volume = news_scraper.get_news_volume(ticker_news)
    
    sentiment_label = "BULLISH" if news_sentiment > 0 else "BEARISH" if news_sentiment < 0 else "NEUTRAL"
    print(f"   • Sentiment Score: {news_sentiment:.3f} ({sentiment_label})")
    print(f"   • News Volume: {news_volume} articles")
    print(f"   • Latest Headlines:")
    for i, row in ticker_news.head(3).iterrows():
        print(f"     - {row.get('title', 'N/A')[:70]}...")
    
    # 3.3: Social Media Sentiment
    print("\n3️⃣  Social Media Sentiment (Reddit):")
    reddit_analyzer = RedditSentimentAnalyzer()
    reddit_data = reddit_analyzer.get_ticker_sentiment(ticker, days_back=7)
    
    print(f"   • Reddit Mentions: {reddit_data['mention_count']}")
    print(f"   • Sentiment Score: {reddit_data['sentiment_score']:.3f}")
    print(f"   • Bullish Ratio: {reddit_data['bullish_ratio']:.2%}")
    
    # =================================================================
    # ENHANCED FEATURE ENGINEERING (Multimodal)
    # =================================================================
    print("\n🔧 ENHANCED FEATURE ENGINEERING")
    print("-" * 80)
    
    feature_gen = FeatureGenerator(df)
    features_df = feature_gen.generate_features()
    
    # Add Phase 3 features
    feature_gen.add_macro_features(macro_indicators)
    feature_gen.add_news_sentiment(news_sentiment, news_volume)
    feature_gen.add_social_sentiment(reddit_data['sentiment_score'], reddit_data['mention_count'])
    feature_gen.add_market_regime(regime)
    
    print(f"✅ Generated {len(feature_gen.data.columns)} total features")
    print(f"   • Technical Indicators: {len([c for c in feature_gen.data.columns if 'SMA' in c or 'RSI' in c or 'MACD' in c])}")
    print(f"   • Macro Features: {len([c for c in feature_gen.data.columns if 'macro_' in c])}")
    print(f"   • Sentiment Features: {len([c for c in feature_gen.data.columns if 'sentiment' in c or 'news_' in c or 'reddit_' in c])}")
    print(f"   • Regime Features: {len([c for c in feature_gen.data.columns if 'regime_' in c])}")
    
    # =================================================================
    # MODEL TRAINING (Enhanced with multimodal features)
    # =================================================================
    print("\n🤖 MODEL TRAINING")
    print("-" * 80)
    
    success, metrics = train_model(ticker, features_df)
    
    if success:
        print(f"✅ Model trained successfully")
        print(f"   • Accuracy: {metrics.get('accuracy', 0):.2%}")
        print(f"   • Precision: {metrics.get('precision', 0):.2%}")
        print(f"   • Recall: {metrics.get('recall', 0):.2%}")
        print(f"   • F1 Score: {metrics.get('f1_score', 0):.2%}")
    else:
        print("❌ Model training failed")
    
    # =================================================================
    # PHASE 2: BROKER INTEGRATION (NEW!)
    # =================================================================
    print("\n📡 PHASE 2: BROKER INTEGRATION")
    print("-" * 80)
    
    # Check if Alpaca credentials are set
    alpaca_key = os.getenv('ALPACA_API_KEY')
    alpaca_secret = os.getenv('ALPACA_SECRET_KEY')
    
    if alpaca_key and alpaca_secret and alpaca_key != 'your_alpaca_api_key_here':
        print("✅ Alpaca API credentials found")
        print("   Connecting to Alpaca Paper Trading...")
        
        try:
            # Initialize broker
            broker = AlpacaBroker(
                api_key=alpaca_key,
                secret_key=alpaca_secret,
                base_url=os.getenv('ALPACA_BASE_URL', 'https://paper-api.alpaca.markets')
            )
            
            # Initialize order manager and portfolio tracker
            order_manager = OrderManager(broker, max_order_value=10000.0)
            portfolio_tracker = PortfolioTracker(broker)
            
            # Get account info
            account = broker.get_account_info()
            print(f"\n   Account Status:")
            print(f"   • Cash: ${account.get('cash', 0):,.2f}")
            print(f"   • Portfolio Value: ${account.get('portfolio_value', 0):,.2f}")
            print(f"   • Buying Power: ${account.get('buying_power', 0):,.2f}")
            
            # Get current positions
            positions = broker.get_positions()
            print(f"\n   Current Positions: {len(positions)}")
            for symbol, position in list(positions.items())[:5]:
                print(f"   • {symbol}: {position.get('quantity', 0)} shares @ ${position.get('avg_price', 0):.2f}")
            
            # Demo: Place a paper trade order (commented out for safety)
            print(f"\n   📝 Demo Order (NOT EXECUTED):")
            print(f"   • Symbol: {ticker}")
            print(f"   • Side: BUY")
            print(f"   • Quantity: 1")
            print(f"   • Type: MARKET")
            print(f"   • Reason: AI Signal + Positive News Sentiment + Market Regime = {regime}")
            
            # Uncomment to actually place order:
            # order = order_manager.place_order(
            #     symbol=ticker,
            #     quantity=1,
            #     order_type=OrderType.MARKET,
            #     side=OrderSide.BUY
            # )
            # print(f"   ✅ Order placed: {order['order_id']}")
            
        except Exception as e:
            print(f"   ⚠️  Broker connection error: {e}")
            print("   This is expected if you haven't set up Alpaca API keys yet")
    else:
        print("⚠️  Alpaca API credentials not configured")
        print("   To enable live paper trading:")
        print("   1. Sign up at https://alpaca.markets")
        print("   2. Get your API keys from dashboard")
        print("   3. Copy .env.template to .env")
        print("   4. Add your API keys to .env")
    
    # =================================================================
    # SUMMARY
    # =================================================================
    print("\n" + "=" * 80)
    print("📊 INTEGRATION SUMMARY")
    print("=" * 80)
    
    print("\n✅ COMPLETED PHASES:")
    print("   ✓ Phase 1: Base System (Data + Features + ML)")
    print("   ✓ Phase 2: Broker Integration (Alpaca Paper Trading)")
    print("   ✓ Phase 3: Intelligence Network (Macro + News + Social)")
    
    print("\n🎯 TRADING SIGNALS:")
    
    # Calculate aggregate signal
    signals = []
    
    # Price signal (from model)
    if success and metrics.get('accuracy', 0) > 0.6:
        signals.append(("Price Model", "BULLISH", 0.7))
    
    # News signal
    if news_sentiment > 0.1:
        signals.append(("News Sentiment", "BULLISH", abs(news_sentiment)))
    elif news_sentiment < -0.1:
        signals.append(("News Sentiment", "BEARISH", abs(news_sentiment)))
    
    # Social signal
    if reddit_data['bullish_ratio'] > 0.6:
        signals.append(("Social Sentiment", "BULLISH", reddit_data['bullish_ratio']))
    elif reddit_data['bullish_ratio'] < 0.4:
        signals.append(("Social Sentiment", "BEARISH", 1 - reddit_data['bullish_ratio']))
    
    # Macro signal
    if regime in ['expansion', 'recovery']:
        signals.append(("Macro Regime", "BULLISH", 0.6))
    elif regime == 'recession':
        signals.append(("Macro Regime", "BEARISH", 0.7))
    
    print(f"\n   Signals for {ticker}:")
    for source, direction, strength in signals:
        print(f"   • {source}: {direction} (strength: {strength:.2f})")
    
    # Aggregate signal
    bullish_signals = sum(s[2] for s in signals if s[1] == "BULLISH")
    bearish_signals = sum(s[2] for s in signals if s[1] == "BEARISH")
    
    if bullish_signals > bearish_signals:
        final_signal = "🟢 BUY"
        confidence = bullish_signals / (bullish_signals + bearish_signals) if (bullish_signals + bearish_signals) > 0 else 0
    elif bearish_signals > bullish_signals:
        final_signal = "🔴 SELL"
        confidence = bearish_signals / (bullish_signals + bearish_signals) if (bullish_signals + bearish_signals) > 0 else 0
    else:
        final_signal = "⚪ HOLD"
        confidence = 0.5
    
    print(f"\n   🎯 FINAL SIGNAL: {final_signal}")
    print(f"   📊 Confidence: {confidence:.1%}")
    
    print("\n" + "=" * 80)
    print("🚀 NEXT STEPS:")
    print("=" * 80)
    print("   1. Set up Alpaca API keys for live paper trading")
    print("   2. Run `make test` to validate all components")
    print("   3. Monitor daily performance with `make pipeline`")
    print("   4. Review Phase 4 for advanced AI models (Transformers)")
    print("   5. Check Phase 5 for smart execution (RL agents)")
    print("\n")


if __name__ == '__main__':
    main()
