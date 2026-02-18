"""
Crypto Trading Configuration
Defines supported chains and tokens for trading
"""
from typing import Dict, List
from enum import Enum


class Chain(Enum):
    """Supported blockchain networks (in priority order)"""
    BASE = "base"  # PRIMARY FOCUS - Coinbase's Layer 2
    SOLANA = "solana"
    OPTIMISM = "optimism"
    LINEA = "linea"
    ZKSYNC = "zksync"  # Zero network
    ARBITRUM = "arbitrum"
    BSC = "bsc"  # Binance Smart Chain
    POLYGON = "polygon"


class CryptoConfig:
    """
    Crypto trading configuration

    Priority:
    1. Base (Coinbase L2) - Primary focus
    2. Solana - High throughput
    3. Optimism, Linea, ZKsync, Arbitrum - L2s
    4. BSC, Polygon - Alternative chains
    """

    # Top tokens by chain
    TOKENS_BY_CHAIN = {
        Chain.BASE: [
            "WETH",  # Wrapped ETH
            "USDbC",  # USD Base Coin
            "USDC",  # USD Coin
            "DAI",   # Dai Stablecoin
            "cbETH", # Coinbase Wrapped Staked ETH
            "BRETT", # Base meme token
            "DEGEN", # Degen token
            "TOSHI", # Toshi token
        ],
        Chain.SOLANA: [
            "SOL",   # Solana native
            "USDC",  # USD Coin
            "USDT",  # Tether
            "JUP",   # Jupiter
            "BONK",  # Bonk
            "WIF",   # dogwifhat
            "PYTH",  # Pyth Network
        ],
        Chain.OPTIMISM: [
            "WETH",
            "USDC",
            "OP",    # Optimism token
            "VELO",  # Velodrome
        ],
        Chain.LINEA: [
            "WETH",
            "USDC",
            "USDT",
        ],
        Chain.ZKSYNC: [
            "WETH",
            "USDC",
            "ZK",    # ZKsync token
        ],
        Chain.ARBITRUM: [
            "WETH",
            "USDC",
            "ARB",   # Arbitrum token
            "GMX",   # GMX
        ],
        Chain.BSC: [
            "WBNB",  # Wrapped BNB
            "USDT",
            "USDC",
            "BUSD",  # Binance USD
            "CAKE",  # PancakeSwap
        ],
        Chain.POLYGON: [
            "WMATIC", # Wrapped MATIC
            "USDC",
            "USDT",
            "WETH",
        ],
    }

    # Primary trading pairs by chain
    PRIMARY_PAIRS = {
        Chain.BASE: [
            ("WETH", "USDbC"),  # Most liquid on Base
            ("WETH", "USDC"),
            ("cbETH", "WETH"),
            ("BRETT", "WETH"),
        ],
        Chain.SOLANA: [
            ("SOL", "USDC"),
            ("SOL", "USDT"),
            ("JUP", "SOL"),
            ("BONK", "SOL"),
        ],
        Chain.OPTIMISM: [
            ("WETH", "USDC"),
            ("OP", "WETH"),
        ],
        Chain.LINEA: [
            ("WETH", "USDC"),
        ],
        Chain.ZKSYNC: [
            ("WETH", "USDC"),
        ],
        Chain.ARBITRUM: [
            ("WETH", "USDC"),
            ("ARB", "WETH"),
            ("GMX", "WETH"),
        ],
        Chain.BSC: [
            ("WBNB", "USDT"),
            ("CAKE", "WBNB"),
        ],
        Chain.POLYGON: [
            ("WMATIC", "USDC"),
            ("WETH", "USDC"),
        ],
    }

    # DEX protocols by chain
    DEX_BY_CHAIN = {
        Chain.BASE: ["Uniswap V3", "Aerodrome", "BaseSwap"],
        Chain.SOLANA: ["Jupiter", "Orca", "Raydium"],
        Chain.OPTIMISM: ["Uniswap V3", "Velodrome"],
        Chain.LINEA: ["SyncSwap", "iZiSwap"],
        Chain.ZKSYNC: ["SyncSwap", "Mute.io"],
        Chain.ARBITRUM: ["Uniswap V3", "Camelot", "GMX"],
        Chain.BSC: ["PancakeSwap", "Biswap"],
        Chain.POLYGON: ["Uniswap V3", "QuickSwap"],
    }

    @staticmethod
    def get_default_trading_symbols() -> List[str]:
        """
        Get default trading symbols (Base network priority)

        Returns:
            List of symbol pairs in format "TOKEN1/TOKEN2"
        """
        symbols = []

        # Base (primary)
        for token1, token2 in CryptoConfig.PRIMARY_PAIRS[Chain.BASE]:
            symbols.append(f"{token1}/{token2}")

        # Solana (secondary)
        for token1, token2 in CryptoConfig.PRIMARY_PAIRS[Chain.SOLANA][:2]:
            symbols.append(f"{token1}/{token2}")

        return symbols

    @staticmethod
    def get_chain_tokens(chain: Chain) -> List[str]:
        """Get supported tokens for a chain"""
        return CryptoConfig.TOKENS_BY_CHAIN.get(chain, [])

    @staticmethod
    def get_chain_pairs(chain: Chain) -> List[tuple]:
        """Get primary trading pairs for a chain"""
        return CryptoConfig.PRIMARY_PAIRS.get(chain, [])

    @staticmethod
    def get_chain_dex(chain: Chain) -> List[str]:
        """Get DEX protocols for a chain"""
        return CryptoConfig.DEX_BY_CHAIN.get(chain, [])


# Strategy-to-Chain mapping
STRATEGY_CHAINS = {
    "mean_reversion": Chain.BASE,      # Base - most liquid
    "momentum": Chain.SOLANA,          # Solana - fast execution
    "rsi": Chain.BASE,                 # Base
    "ml_ensemble": Chain.BASE,         # Base - sophisticated
    "ppo_rl": Chain.SOLANA,           # Solana - high frequency
    "macd": Chain.OPTIMISM,           # Optimism
    "bollinger": Chain.BASE,          # Base
    "yield_optimizer": Chain.ARBITRUM, # Arbitrum - DeFi protocols
    "grid": Chain.BSC,                # BSC - lower fees
    "dca": Chain.BASE,                # Base
    "arbitrage": [Chain.BASE, Chain.OPTIMISM, Chain.ARBITRUM],  # Cross-chain
}


def get_strategy_symbols(strategy_id: str) -> List[str]:
    """
    Get appropriate symbols for a strategy

    Args:
        strategy_id: Strategy identifier

    Returns:
        List of symbol pairs for the strategy
    """
    chain = STRATEGY_CHAINS.get(strategy_id, Chain.BASE)

    if isinstance(chain, list):
        # Multi-chain strategy (e.g., arbitrage)
        symbols = []
        for c in chain:
            pairs = CryptoConfig.get_chain_pairs(c)
            if pairs:
                token1, token2 = pairs[0]
                symbols.append(f"{token1}/{token2}")
        return symbols
    else:
        # Single chain
        pairs = CryptoConfig.get_chain_pairs(chain)
        if not pairs:
            return ["WETH/USDC"]  # Fallback

        # Return first pair for the chain
        token1, token2 = pairs[0]
        return [f"{token1}/{token2}"]


if __name__ == "__main__":
    print("🔗 Crypto Trading Configuration")
    print("=" * 60)

    print("\n📊 Default Trading Symbols:")
    for symbol in CryptoConfig.get_default_trading_symbols():
        print(f"  • {symbol}")

    print("\n🔗 Supported Chains (Priority Order):")
    for i, chain in enumerate(Chain, 1):
        tokens = CryptoConfig.get_chain_tokens(chain)
        dex = CryptoConfig.get_chain_dex(chain)
        print(f"  {i}. {chain.value.upper()}")
        print(f"     Tokens: {', '.join(tokens[:4])}...")
        print(f"     DEX: {', '.join(dex)}")

    print("\n⚙️ Strategy Chain Assignments:")
    for strategy, chain in STRATEGY_CHAINS.items():
        if isinstance(chain, list):
            chain_names = ", ".join([c.value for c in chain])
            print(f"  • {strategy}: {chain_names}")
        else:
            print(f"  • {strategy}: {chain.value}")
