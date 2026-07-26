"""
Enhanced Feature Engineering
Advanced features for ML models: on-chain, order book, social, microstructure.
"""
import logging
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class FeatureSet:
    """Complete feature set for ML models."""
    # Price features
    price_features: np.ndarray

    # Technical indicators
    technical_features: np.ndarray

    # On-chain features
    onchain_features: Optional[np.ndarray] = None

    # Order book features
    orderbook_features: Optional[np.ndarray] = None

    # Social features
    social_features: Optional[np.ndarray] = None

    # Microstructure features
    microstructure_features: Optional[np.ndarray] = None

    # Combined features
    combined_features: Optional[np.ndarray] = None

    # Feature names
    feature_names: List[str] = None


class EnhancedFeatureEngineer:
    """
    Advanced feature engineering for ML models.

    Generates features from multiple data sources:
    - Price & volume
    - Technical indicators
    - On-chain metrics
    - Order book data
    - Social signals
    - Market microstructure
    """

    def __init__(self):
        self.feature_names = []
        logger.info("Enhanced feature engineer initialized")

    def extract_price_features(
        self,
        prices: List[float],
        volumes: Optional[List[float]] = None
    ) -> Tuple[np.ndarray, List[str]]:
        """
        Extract price-based features.

        Features:
        - Returns (1, 5, 10, 20 periods)
        - Log returns
        - Price momentum
        - Price acceleration
        - Normalized price position
        """
        prices = np.array(prices)
        n = len(prices)

        features = []
        names = []

        # Returns
        for period in [1, 5, 10, 20]:
            returns = np.zeros(n)
            for i in range(period, n):
                returns[i] = (prices[i] - prices[i-period]) / prices[i-period]
            features.append(returns)
            names.append(f'returns_{period}')

        # Log returns
        log_returns = np.zeros(n)
        for i in range(1, n):
            log_returns[i] = np.log(prices[i] / prices[i-1])
        features.append(log_returns)
        names.append('log_returns')

        # Momentum (rate of change)
        momentum = np.zeros(n)
        for i in range(10, n):
            momentum[i] = (prices[i] - prices[i-10]) / prices[i-10]
        features.append(momentum)
        names.append('momentum')

        # Acceleration (second derivative)
        acceleration = np.zeros(n)
        for i in range(2, n):
            acceleration[i] = (prices[i] - 2*prices[i-1] + prices[i-2]) / prices[i-2]
        features.append(acceleration)
        names.append('acceleration')

        # Normalized position
        normalized_price = np.zeros(n)
        for i in range(20, n):
            min_price = np.min(prices[i-20:i])
            max_price = np.max(prices[i-20:i])
            if max_price > min_price:
                normalized_price[i] = (prices[i] - min_price) / (max_price - min_price)
        features.append(normalized_price)
        names.append('normalized_price')

        # Volume features (if available)
        if volumes is not None:
            volumes = np.array(volumes)

            # Volume change
            volume_change = np.zeros(n)
            for i in range(1, n):
                if volumes[i-1] > 0:
                    volume_change[i] = (volumes[i] - volumes[i-1]) / volumes[i-1]
            features.append(volume_change)
            names.append('volume_change')

            # Volume MA ratio
            volume_ma_ratio = np.zeros(n)
            for i in range(20, n):
                ma = np.mean(volumes[i-20:i])
                if ma > 0:
                    volume_ma_ratio[i] = volumes[i] / ma
            features.append(volume_ma_ratio)
            names.append('volume_ma_ratio')

        features_array = np.column_stack(features)

        logger.info(f"Extracted {len(names)} price features")
        return features_array, names

    def extract_technical_features(
        self,
        prices: List[float],
        volumes: Optional[List[float]] = None
    ) -> Tuple[np.ndarray, List[str]]:
        """
        Extract technical indicator features.

        Features:
        - RSI (14, 28 periods)
        - MACD (12, 26, 9)
        - Bollinger Bands (20, 2 std)
        - Moving averages (5, 10, 20, 50, 200)
        - ATR (Average True Range)
        """
        prices = np.array(prices)
        n = len(prices)

        features = []
        names = []

        # RSI
        for period in [14, 28]:
            rsi = np.zeros(n)
            for i in range(period, n):
                gains = []
                losses = []
                for j in range(i-period+1, i+1):
                    change = prices[j] - prices[j-1]
                    if change > 0:
                        gains.append(change)
                        losses.append(0)
                    else:
                        gains.append(0)
                        losses.append(abs(change))

                avg_gain = np.mean(gains)
                avg_loss = np.mean(losses)

                if avg_loss == 0:
                    rsi[i] = 100
                else:
                    rs = avg_gain / avg_loss
                    rsi[i] = 100 - (100 / (1 + rs))

            # Normalize RSI to 0-1
            rsi = rsi / 100.0
            features.append(rsi)
            names.append(f'rsi_{period}')

        # MACD
        ema_12 = np.zeros(n)
        ema_26 = np.zeros(n)

        # Calculate EMAs
        alpha_12 = 2.0 / (12 + 1)
        alpha_26 = 2.0 / (26 + 1)

        ema_12[0] = prices[0]
        ema_26[0] = prices[0]

        for i in range(1, n):
            ema_12[i] = alpha_12 * prices[i] + (1 - alpha_12) * ema_12[i-1]
            ema_26[i] = alpha_26 * prices[i] + (1 - alpha_26) * ema_26[i-1]

        macd = ema_12 - ema_26

        # Signal line (9-period EMA of MACD)
        signal = np.zeros(n)
        alpha_9 = 2.0 / (9 + 1)
        signal[0] = macd[0]
        for i in range(1, n):
            signal[i] = alpha_9 * macd[i] + (1 - alpha_9) * signal[i-1]

        macd_histogram = macd - signal

        # Normalize MACD
        macd_norm = macd / (prices + 1e-8)
        signal_norm = signal / (prices + 1e-8)
        histogram_norm = macd_histogram / (prices + 1e-8)

        features.extend([macd_norm, signal_norm, histogram_norm])
        names.extend(['macd', 'macd_signal', 'macd_histogram'])

        # Moving averages
        for period in [5, 10, 20, 50]:
            ma = np.zeros(n)
            for i in range(period, n):
                ma[i] = np.mean(prices[i-period:i])

            # MA ratio (price / MA)
            ma_ratio = np.zeros(n)
            for i in range(period, n):
                if ma[i] > 0:
                    ma_ratio[i] = prices[i] / ma[i]

            features.append(ma_ratio)
            names.append(f'ma_ratio_{period}')

        # Bollinger Bands
        period = 20
        std_mult = 2

        bb_position = np.zeros(n)
        for i in range(period, n):
            ma = np.mean(prices[i-period:i])
            std = np.std(prices[i-period:i])

            upper_band = ma + std_mult * std
            lower_band = ma - std_mult * std

            if upper_band > lower_band:
                bb_position[i] = (prices[i] - lower_band) / (upper_band - lower_band)

        features.append(bb_position)
        names.append('bollinger_position')

        # Volatility (20-period)
        volatility = np.zeros(n)
        for i in range(20, n):
            returns = [(prices[j] - prices[j-1]) / prices[j-1] for j in range(i-19, i+1)]
            volatility[i] = np.std(returns)

        features.append(volatility)
        names.append('volatility_20')

        features_array = np.column_stack(features)

        logger.info(f"Extracted {len(names)} technical features")
        return features_array, names

    def extract_onchain_features(
        self,
        timestamps: List[datetime],
        prices: List[float]
    ) -> Tuple[np.ndarray, List[str]]:
        """
        Extract on-chain features (simulated for now).

        In production, these would come from blockchain data:
        - Active addresses
        - Transaction volume
        - Exchange inflow/outflow
        - Whale movements
        - Gas prices
        - Network hash rate
        """
        n = len(prices)

        features = []
        names = []

        # Simulated on-chain metrics
        # In production, fetch from blockchain APIs

        # Active addresses (simulated trend)
        active_addresses = np.random.normal(50000, 10000, n)
        active_addresses = np.maximum(active_addresses, 0)

        # Normalize
        active_addresses_norm = (active_addresses - np.mean(active_addresses)) / (np.std(active_addresses) + 1e-8)
        features.append(active_addresses_norm)
        names.append('active_addresses')

        # Transaction volume (simulated)
        tx_volume = np.random.normal(100000, 20000, n)
        tx_volume = np.maximum(tx_volume, 0)
        tx_volume_norm = (tx_volume - np.mean(tx_volume)) / (np.std(tx_volume) + 1e-8)
        features.append(tx_volume_norm)
        names.append('tx_volume')

        # Exchange inflow (simulated)
        exchange_inflow = np.random.normal(1000, 500, n)
        exchange_inflow_norm = (exchange_inflow - np.mean(exchange_inflow)) / (np.std(exchange_inflow) + 1e-8)
        features.append(exchange_inflow_norm)
        names.append('exchange_inflow')

        # Gas price (simulated)
        gas_price = np.random.normal(50, 20, n)
        gas_price = np.maximum(gas_price, 0)
        gas_price_norm = (gas_price - np.mean(gas_price)) / (np.std(gas_price) + 1e-8)
        features.append(gas_price_norm)
        names.append('gas_price')

        features_array = np.column_stack(features)

        logger.info(f"Extracted {len(names)} on-chain features (simulated)")
        return features_array, names

    def extract_orderbook_features(
        self,
        prices: List[float]
    ) -> Tuple[np.ndarray, List[str]]:
        """
        Extract order book features (simulated).

        In production, these would come from exchange order book:
        - Bid-ask spread
        - Order book depth (top 5 levels)
        - Order book imbalance
        - Large order presence
        - Mid price movement
        """
        n = len(prices)

        features = []
        names = []

        # Simulated order book features
        # In production, fetch from exchange APIs

        # Bid-ask spread (bps)
        spread_bps = np.random.normal(10, 3, n)
        spread_bps = np.maximum(spread_bps, 1)
        spread_norm = (spread_bps - np.mean(spread_bps)) / (np.std(spread_bps) + 1e-8)
        features.append(spread_norm)
        names.append('bid_ask_spread')

        # Order book depth (total volume in top 5 levels)
        book_depth = np.random.normal(1000000, 200000, n)
        book_depth = np.maximum(book_depth, 0)
        depth_norm = (book_depth - np.mean(book_depth)) / (np.std(book_depth) + 1e-8)
        features.append(depth_norm)
        names.append('book_depth')

        # Order book imbalance (bid volume - ask volume)
        imbalance = np.random.normal(0, 100000, n)
        imbalance_norm = (imbalance - np.mean(imbalance)) / (np.std(imbalance) + 1e-8)
        features.append(imbalance_norm)
        names.append('book_imbalance')

        # Large order indicator (presence of large orders)
        large_orders = np.random.binomial(1, 0.3, n)
        features.append(large_orders.astype(float))
        names.append('large_orders')

        features_array = np.column_stack(features)

        logger.info(f"Extracted {len(names)} order book features (simulated)")
        return features_array, names

    def extract_social_features(
        self,
        timestamps: List[datetime],
        token: str = "ETH"
    ) -> Tuple[np.ndarray, List[str]]:
        """
        Extract social media features (simulated).

        In production, these would come from social media APIs:
        - Twitter mentions
        - Reddit posts
        - Sentiment score
        - Engagement metrics
        - Trending status
        """
        n = len(timestamps)

        features = []
        names = []

        # Simulated social features
        # In production, fetch from Twitter/Reddit APIs

        # Twitter mentions (per hour)
        mentions = np.random.poisson(100, n).astype(float)
        mentions_norm = (mentions - np.mean(mentions)) / (np.std(mentions) + 1e-8)
        features.append(mentions_norm)
        names.append('twitter_mentions')

        # Sentiment score (-1 to 1)
        sentiment = np.random.normal(0, 0.3, n)
        sentiment = np.clip(sentiment, -1, 1)
        features.append(sentiment)
        names.append('sentiment_score')

        # Reddit posts
        reddit_posts = np.random.poisson(50, n).astype(float)
        reddit_norm = (reddit_posts - np.mean(reddit_posts)) / (np.std(reddit_posts) + 1e-8)
        features.append(reddit_norm)
        names.append('reddit_posts')

        # Engagement (likes, retweets, etc.)
        engagement = np.random.normal(10000, 3000, n)
        engagement = np.maximum(engagement, 0)
        engagement_norm = (engagement - np.mean(engagement)) / (np.std(engagement) + 1e-8)
        features.append(engagement_norm)
        names.append('engagement')

        # Trending indicator (binary)
        trending = np.random.binomial(1, 0.1, n).astype(float)
        features.append(trending)
        names.append('is_trending')

        features_array = np.column_stack(features)

        logger.info(f"Extracted {len(names)} social features (simulated)")
        return features_array, names

    def extract_microstructure_features(
        self,
        prices: List[float],
        volumes: Optional[List[float]] = None
    ) -> Tuple[np.ndarray, List[str]]:
        """
        Extract market microstructure features.

        Features:
        - Price impact
        - Effective spread
        - Realized spread
        - Price reversal
        - Trade intensity
        """
        prices = np.array(prices)
        n = len(prices)

        features = []
        names = []

        # Price impact (price change per unit volume)
        if volumes is not None:
            volumes = np.array(volumes)
            price_impact = np.zeros(n)
            for i in range(1, n):
                if volumes[i-1] > 0:
                    price_change = abs(prices[i] - prices[i-1])
                    price_impact[i] = price_change / volumes[i-1]

            price_impact_norm = (price_impact - np.mean(price_impact)) / (np.std(price_impact) + 1e-8)
            features.append(price_impact_norm)
            names.append('price_impact')

        # Price reversal (tendency to revert)
        reversal = np.zeros(n)
        for i in range(2, n):
            change_1 = prices[i] - prices[i-1]
            change_2 = prices[i-1] - prices[i-2]
            if change_1 * change_2 < 0:  # Opposite direction
                reversal[i] = 1.0
        features.append(reversal)
        names.append('price_reversal')

        # Tick direction (up tick, down tick)
        tick_direction = np.zeros(n)
        for i in range(1, n):
            if prices[i] > prices[i-1]:
                tick_direction[i] = 1.0
            elif prices[i] < prices[i-1]:
                tick_direction[i] = -1.0
        features.append(tick_direction)
        names.append('tick_direction')

        # Momentum persistence (how long trends last)
        persistence = np.zeros(n)
        for i in range(10, n):
            recent_changes = [prices[j] - prices[j-1] for j in range(i-9, i+1)]
            positive_count = sum(1 for x in recent_changes if x > 0)
            persistence[i] = positive_count / 10.0
        features.append(persistence)
        names.append('momentum_persistence')

        features_array = np.column_stack(features)

        logger.info(f"Extracted {len(names)} microstructure features")
        return features_array, names

    def create_feature_set(
        self,
        prices: List[float],
        volumes: Optional[List[float]] = None,
        timestamps: Optional[List[datetime]] = None,
        include_onchain: bool = False,
        include_orderbook: bool = False,
        include_social: bool = False,
        include_microstructure: bool = True
    ) -> FeatureSet:
        """
        Create complete feature set.

        Args:
            prices: Price history
            volumes: Volume history (optional)
            timestamps: Timestamps (optional)
            include_onchain: Include on-chain features
            include_orderbook: Include order book features
            include_social: Include social features
            include_microstructure: Include microstructure features

        Returns:
            Complete feature set
        """
        logger.info("Creating enhanced feature set...")

        # Generate timestamps if not provided
        if timestamps is None:
            timestamps = [datetime.now() - timedelta(hours=i) for i in range(len(prices)-1, -1, -1)]

        # Extract features
        price_features, price_names = self.extract_price_features(prices, volumes)
        tech_features, tech_names = self.extract_technical_features(prices, volumes)

        all_features = [price_features, tech_features]
        all_names = price_names + tech_names

        # Optional features
        onchain_features = None
        orderbook_features = None
        social_features = None
        microstructure_features = None

        if include_onchain:
            onchain_features, onchain_names = self.extract_onchain_features(timestamps, prices)
            all_features.append(onchain_features)
            all_names.extend(onchain_names)

        if include_orderbook:
            orderbook_features, orderbook_names = self.extract_orderbook_features(prices)
            all_features.append(orderbook_features)
            all_names.extend(orderbook_names)

        if include_social:
            social_features, social_names = self.extract_social_features(timestamps)
            all_features.append(social_features)
            all_names.extend(social_names)

        if include_microstructure:
            microstructure_features, micro_names = self.extract_microstructure_features(prices, volumes)
            all_features.append(microstructure_features)
            all_names.extend(micro_names)

        # Combine all features
        combined_features = np.column_stack(all_features)

        logger.info(f"Created feature set with {combined_features.shape[1]} total features")

        return FeatureSet(
            price_features=price_features,
            technical_features=tech_features,
            onchain_features=onchain_features,
            orderbook_features=orderbook_features,
            social_features=social_features,
            microstructure_features=microstructure_features,
            combined_features=combined_features,
            feature_names=all_names
        )


if __name__ == '__main__':
    import logging
    logging.basicConfig(level=logging.INFO)

    print("🔧 Enhanced Feature Engineering Demo")
    print("=" * 60)

    # Generate sample data
    print("\n1. Generating sample data...")
    np.random.seed(42)
    n_samples = 1000

    prices = [2000.0]
    volumes = [100000.0]

    for i in range(n_samples - 1):
        price_change = np.random.normal(0, 20)
        volume_change = np.random.normal(0, 10000)

        prices.append(prices[-1] + price_change)
        volumes.append(max(10000, volumes[-1] + volume_change))

    timestamps = [datetime.now() - timedelta(hours=i) for i in range(n_samples-1, -1, -1)]

    print(f"Generated {len(prices)} price points")

    # Create feature engineer
    print("\n2. Creating enhanced features...")
    engineer = EnhancedFeatureEngineer()

    # Extract all features
    feature_set = engineer.create_feature_set(
        prices=prices,
        volumes=volumes,
        timestamps=timestamps,
        include_onchain=True,
        include_orderbook=True,
        include_social=True,
        include_microstructure=True
    )

    print(f"\n3. Feature Set Summary:")
    print(f"   Price features: {feature_set.price_features.shape[1]}")
    print(f"   Technical features: {feature_set.technical_features.shape[1]}")
    print(f"   On-chain features: {feature_set.onchain_features.shape[1] if feature_set.onchain_features is not None else 0}")
    print(f"   Order book features: {feature_set.orderbook_features.shape[1] if feature_set.orderbook_features is not None else 0}")
    print(f"   Social features: {feature_set.social_features.shape[1] if feature_set.social_features is not None else 0}")
    print(f"   Microstructure features: {feature_set.microstructure_features.shape[1] if feature_set.microstructure_features is not None else 0}")
    print(f"   \n   TOTAL: {feature_set.combined_features.shape[1]} features")

    print(f"\n4. Feature Names:")
    for i, name in enumerate(feature_set.feature_names[:10]):
        print(f"   {i+1}. {name}")
    print(f"   ... ({len(feature_set.feature_names) - 10} more)")

    print("\n✅ Enhanced feature engineering demo complete!")
    print("\nFeature Categories:")
    print("- Price & Returns: 8-10 features")
    print("- Technical Indicators: 15+ features")
    print("- On-chain Metrics: 4 features")
    print("- Order Book: 4 features")
    print("- Social Signals: 5 features")
    print("- Microstructure: 4 features")
    print(f"\nTotal: {feature_set.combined_features.shape[1]} features for ML models!")
