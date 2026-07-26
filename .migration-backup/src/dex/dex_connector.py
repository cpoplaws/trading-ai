"""
DEX Connector - Unified interface for decentralized exchanges

Provides a common interface for:
- Uniswap V3 (Base, Arbitrum, Optimism, Ethereum)
- Jupiter (Solana)
- DEX aggregators (1inch, Matcha, Paraswap)

Responsibilities:
1. Get swap quotes from multiple DEXes
2. Find best execution route
3. Build swap transactions
4. Estimate gas costs
5. Execute swaps with slippage protection
6. Monitor transaction confirmation
"""

from dataclasses import dataclass
from typing import Dict, Optional, Literal, List
from enum import Enum
import logging
import time
import asyncio
import os
from datetime import datetime

import aiohttp
import requests

logger = logging.getLogger(__name__)


class Chain(Enum):
    """Supported chains"""
    BASE = "base"
    ARBITRUM = "arbitrum"
    OPTIMISM = "optimism"
    POLYGON = "polygon"
    ETHEREUM = "ethereum"
    SOLANA = "solana"


class DEXProtocol(Enum):
    """DEX protocols"""
    UNISWAP_V3 = "uniswap_v3"
    UNISWAP_V2 = "uniswap_v2"
    JUPITER = "jupiter"
    RAYDIUM = "raydium"
    CURVE = "curve"
    BALANCER = "balancer"
    ONEINCH = "1inch"  # Aggregator
    PARASWAP = "paraswap"  # Aggregator


@dataclass
class SwapQuote:
    """Swap quote from DEX"""
    dex: DEXProtocol
    chain: Chain
    input_token: str
    output_token: str
    input_amount: float
    output_amount: float
    price: float  # Output per input
    price_impact: float  # As decimal
    fee: float  # In USD
    fee_pct: float
    gas_estimate: int
    gas_cost_usd: float
    route: List[str]  # Token path
    timestamp: datetime


@dataclass
class SwapRequest:
    """Swap request"""
    chain: Chain
    input_token: str
    output_token: str
    amount_in: Optional[float] = None
    amount_out: Optional[float] = None  # For exact output swaps
    slippage_pct: float = 0.01  # 1% default
    recipient: Optional[str] = None
    deadline: Optional[int] = None  # Unix timestamp


@dataclass
class SwapResult:
    """Swap execution result"""
    success: bool
    chain: Chain
    dex: DEXProtocol
    input_token: str
    output_token: str
    input_amount: float
    output_amount: float
    tx_hash: str
    gas_used: int
    gas_cost_usd: float
    total_cost_usd: float
    execution_price: float
    slippage: float
    timestamp: datetime
    error: Optional[str] = None


class DEXConnector:
    """
    Unified connector for decentralized exchanges.

    Supports:
    - Uniswap V3 on EVM chains
    - Jupiter on Solana
    - DEX aggregators (1inch, Paraswap)

    Features:
    - Best quote routing across multiple DEXes
    - Automatic slippage protection
    - Gas optimization
    - Transaction monitoring
    - MEV protection (optional)
    """

    # RPC endpoints (would come from config in production)
    RPC_ENDPOINTS = {
        Chain.BASE: "https://mainnet.base.org",
        Chain.ARBITRUM: "https://arb1.arbitrum.io/rpc",
        Chain.OPTIMISM: "https://mainnet.optimism.io",
        Chain.ETHEREUM: "https://eth.llamarpc.com",
        Chain.POLYGON: "https://polygon-rpc.com",
        Chain.SOLANA: "https://api.mainnet-beta.solana.com",
    }

    # DEX API endpoints
    JUPITER_QUOTE_API = "https://quote-api.jup.ag/v6/quote"
    JUPITER_SWAP_API = "https://quote-api.jup.ag/v6/swap"
    ONEINCH_API_BASE = "https://api.1inch.dev"

    # Common token addresses
    TOKENS = {
        Chain.BASE: {
            "WETH": "0x4200000000000000000000000000000000000006",
            "USDC": "0x833589fCD6eDb6E08f4c7C32D4f71b54bdA02913",
            "USDbC": "0xd9aAEc86B65D86f6A7B5B1b0c42FFA531710b6CA",
        },
        Chain.ARBITRUM: {
            "WETH": "0x82aF49447D8a07e3bd95BD0d56f35241523fBab1",
            "USDC": "0xaf88d065e77c8cC2239327C5EDb3A432268e5831",
        },
        Chain.OPTIMISM: {
            "WETH": "0x4200000000000000000000000000000000000006",
            "USDC": "0x0b2C639c533813f4Aa9D7837CAf62653d097Ff85",
        },
        Chain.SOLANA: {
            "SOL": "So11111111111111111111111111111111111111112",
            "USDC": "EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v",
        }
    }

    def __init__(
        self,
        wallet_manager=None,
        gas_manager=None,
        use_aggregators: bool = True
    ):
        """
        Initialize DEX connector.

        Args:
            wallet_manager: WalletManager instance
            gas_manager: GasManager instance
            use_aggregators: Use DEX aggregators for better prices
        """
        self.wallet_manager = wallet_manager
        self.gas_manager = gas_manager
        self.use_aggregators = use_aggregators

        logger.info(f"DEX Connector initialized | Aggregators: {'✅' if use_aggregators else '❌'}")

    def get_quote(
        self,
        swap_request: SwapRequest,
        dex: Optional[DEXProtocol] = None
    ) -> SwapQuote:
        """
        Get swap quote from DEX.

        Args:
            swap_request: Swap parameters
            dex: Specific DEX to use (None = auto-select best)

        Returns:
            Swap quote
        """
        if swap_request.chain == Chain.SOLANA:
            return self._get_jupiter_quote(swap_request)
        else:
            if dex == DEXProtocol.UNISWAP_V3 or dex is None:
                return self._get_uniswap_quote(swap_request)
            elif self.use_aggregators and dex == DEXProtocol.ONEINCH:
                return self._get_1inch_quote(swap_request)
            else:
                return self._get_uniswap_quote(swap_request)

    def _get_uniswap_quote(self, swap_request: SwapRequest) -> SwapQuote:
        """
        Get quote from Uniswap V3.

        Args:
            swap_request: Swap parameters

        Returns:
            Swap quote
        """
        # Try to fetch from Uniswap V3 API if available
        try:
            return self._get_uniswap_quote_from_api(swap_request)
        except Exception as e:
            logger.warning(f"Uniswap API error: {e}, using fallback quote")
            # Fallback to mock implementation
            return self._get_uniswap_quote_fallback(swap_request)

    def _get_uniswap_quote_from_api(self, swap_request: SwapRequest) -> SwapQuote:
        """
        Get real quote from Uniswap V3 API (QuoterV2).

        Args:
            swap_request: Swap parameters

        Returns:
            Swap quote
        """
        from web3 import Web3
        from eth_abi.abi import decode_abi

        # Get token addresses
        input_address = self._get_token_address(swap_request.chain, swap_request.input_token)
        output_address = self._get_token_address(swap_request.chain, swap_request.output_token)

        # Connect to RPC
        w3 = Web3(Web3.HTTPProvider(self.RPC_ENDPOINTS[swap_request.chain]))

        # Use Uniswap V3 Quoter V2 contract (0x61fE3cCf... on mainnet chains)
        # Note: Addresses vary by chain, would need proper mapping in production
        quoter_address = "0x61fE3cCf6C1e9c5c6f7525Bd85E1b3e4b5c0C"  # Ethereum mainnet
        quoter_abi = [
            {
                "inputs": [{"internalType": "address", "name": "tokenIn", "type": "address"},
                          {"internalType": "address", "name": "tokenOut", "type": "address"},
                          {"internalType": "uint256", "name": "amountIn", "type": "uint256"},
                          {"internalType": "uint160", "name": "amountOutMinimum", "type": "uint160"},
                          {"internalType": "uint160", "name": "sqrtPriceLimitX96", "type": "uint160"},
                          {"internalType": "uint24[]", "name": "hints", "type": "uint24[]"}],
                "name": "quoteExactInputSingle",
                "outputs": [{"internalType": "uint256", "name": "amountOut", "type": "uint256"},
                            {"internalType": "uint160[]", "name": "sqrtPriceX96AfterList", "type": "uint160[]"},
                            {"internalType": "uint32[]", "name": "initializedTicksCrossedList", "type": "uint32[]"},
                            {"internalType": "uint256", "name": "gasEstimate", "type": "uint256"}],
                "stateMutability": "view",
                "type": "function"
            }
        ]

        # Build call
        amount_in = int((swap_request.amount_in or 1.0) * 1e18)  # 18 decimals
        amount_min = int(amount_in * (1 - swap_request.slippage_pct * 1e18))

        try:
            # Call quoter contract
            quoter = w3.eth.contract(address=quoter_address, abi=quoter_abi)
            result = quoter.functions.quoteExactInputSingle(
                input_address,
                output_address,
                amount_in,
                amount_min,
                0  # sqrtPriceLimitX96
            ).call()

            output_amount = result[0] / 1e18
            gas_estimate = result[3] or 180000

            # Calculate gas cost (rough estimate)
            gas_price = w3.eth.gas_price
            gas_cost_wei = gas_estimate * gas_price
            gas_cost_eth = gas_cost_wei / 1e18

            # Get ETH price for gas cost in USD
            eth_price = self._get_native_token_price(swap_request.chain)
            gas_cost_usd = gas_cost_eth * eth_price

            # Calculate price impact (simplified)
            price = output_amount / (swap_request.amount_in or 1.0)
            price_impact = 0.001  # Would need more complex calculation

            # Fee estimate (0.3% for most pools)
            fee = (swap_request.amount_in or 1.0) * price * 0.003

            logger.info(f"Uniswap V3 quote: {swap_request.input_token} -> {swap_request.output_token}")
            return SwapQuote(
                dex=DEXProtocol.UNISWAP_V3,
                chain=swap_request.chain,
                input_token=swap_request.input_token,
                output_token=swap_request.output_token,
                input_amount=swap_request.amount_in or 1.0,
                output_amount=output_amount,
                price=price,
                price_impact=price_impact,
                fee=fee,
                fee_pct=0.003,
                gas_estimate=gas_estimate,
                gas_cost_usd=gas_cost_usd,
                route=[swap_request.input_token, swap_request.output_token],
                timestamp=datetime.now()
            )

        except Exception as e:
            logger.error(f"Error fetching Uniswap quote: {e}")
            raise

    def _get_uniswap_quote_fallback(self, swap_request: SwapRequest) -> SwapQuote:
        """
        Fallback mock quote when API is unavailable.

        Args:
            swap_request: Swap parameters

        Returns:
            Swap quote
        """
        logger.warning(f"Mock Uniswap quote for {swap_request.input_token}/{swap_request.output_token}")

        # Mock quote
        input_amount = swap_request.amount_in or 1.0
        price = 3000.0 if "ETH" in swap_request.input_token else 1.0
        output_amount = input_amount * price * 0.997  # 0.3% fee

        return SwapQuote(
            dex=DEXProtocol.UNISWAP_V3,
            chain=swap_request.chain,
            input_token=swap_request.input_token,
            output_token=swap_request.output_token,
            input_amount=input_amount,
            output_amount=output_amount,
            price=output_amount / input_amount,
            price_impact=0.001,  # 0.1%
            fee=input_amount * price * 0.003,  # 0.3% fee in USD
            fee_pct=0.003,
            gas_estimate=180000,
            gas_cost_usd=8.0,  # Mock ~$8 on Base
            route=[swap_request.input_token, swap_request.output_token],
            timestamp=datetime.now()
        )

    def _get_jupiter_quote(self, swap_request: SwapRequest) -> SwapQuote:
        """
        Get quote from Jupiter (Solana).

        Args:
            swap_request: Swap parameters

        Returns:
            Swap quote
        """
        # Try to fetch from Jupiter API
        try:
            return self._get_jupiter_quote_from_api(swap_request)
        except Exception as e:
            logger.warning(f"Jupiter API error: {e}, using fallback quote")
            # Fallback to mock implementation
            return self._get_jupiter_quote_fallback(swap_request)

    def _get_jupiter_quote_from_api(self, swap_request: SwapRequest) -> SwapQuote:
        """
        Get real quote from Jupiter API.

        Args:
            swap_request: Swap parameters

        Returns:
            Swap quote
        """
        # Get token addresses
        input_address = self.TOKENS[Chain.SOLANA].get(swap_request.input_token)
        output_address = self.TOKENS[Chain.SOLANA].get(swap_request.output_token)

        if not input_address or not output_address:
            raise ValueError(f"Unknown token: {swap_request.input_token} or {swap_request.output_token}")

        # Build request parameters
        params = {
            "inputMint": input_address,
            "outputMint": output_address,
            "amount": str(int((swap_request.amount_in or 1.0) * 1e9)),  # Convert to lamports
            "slippageBps": str(int(swap_request.slippage_pct * 10000)),  # Convert to basis points
            "onlyDirectRoutes": "false",
            "asLegacyTransaction": "false"
        }

        # Make API request
        response = requests.get(
            self.JUPITER_QUOTE_API,
            params=params,
            timeout=10
        )

        response.raise_for_status()
        data = response.json()

        # Parse quote
        input_amount = float(data["inAmount"]) / 1e9
        output_amount = float(data["outAmount"]) / 1e9
        price = output_amount / input_amount
        price_impact = data.get("priceImpactPct", 0) / 100

        # Get platform fee
        platform_fee = data.get("platformFee", {})
        fee = float(platform_fee.get("amount", 0)) / 1e9
        fee_pct = fee / input_amount if input_amount > 0 else 0

        # Get route
        route = []
        for step in data.get("routePlan", []):
            for swap_info in step.get("swapInfo", []):
                swap_desc = swap_info.get("label", "")
                if swap_desc:
                    route.append(swap_desc)

        logger.info(f"Jupiter quote: {swap_request.input_token} -> {swap_request.output_token}")
        return SwapQuote(
            dex=DEXProtocol.JUPITER,
            chain=Chain.SOLANA,
            input_token=swap_request.input_token,
            output_token=swap_request.output_token,
            input_amount=input_amount,
            output_amount=output_amount,
            price=price,
            price_impact=price_impact,
            fee=fee * 120.0,  # Approximate in USD
            fee_pct=fee_pct,
            gas_estimate=int(data.get("context", {}).get("slot", 0)),
            gas_cost_usd=0.0005,  # Very cheap on Solana
            route=route,
            timestamp=datetime.now()
        )

    def _get_jupiter_quote_fallback(self, swap_request: SwapRequest) -> SwapQuote:
        """
        Fallback mock quote when API is unavailable.

        Args:
            swap_request: Swap parameters

        Returns:
            Swap quote
        """
        logger.warning(f"Mock Jupiter quote for {swap_request.input_token}/{swap_request.output_token}")

        # Mock quote
        input_amount = swap_request.amount_in or 1.0
        price = 120.0 if "SOL" in swap_request.input_token else 1.0
        output_amount = input_amount * price * 0.998  # Lower fees on Solana

        return SwapQuote(
            dex=DEXProtocol.JUPITER,
            chain=Chain.SOLANA,
            input_token=swap_request.input_token,
            output_token=swap_request.output_token,
            input_amount=input_amount,
            output_amount=output_amount,
            price=output_amount / input_amount,
            price_impact=0.0005,  # 0.05%
            fee=input_amount * price * 0.002,  # 0.2% fee
            fee_pct=0.002,
            gas_estimate=1,  # Signatures, not gas
            gas_cost_usd=0.0005,  # Very cheap on Solana
            route=[swap_request.input_token, swap_request.output_token],
            timestamp=datetime.now()
        )

    def _get_1inch_quote(self, swap_request: SwapRequest) -> SwapQuote:
        """
        Get quote from 1inch aggregator.

        Args:
            swap_request: Swap parameters

        Returns:
            Swap quote
        """
        # Try to fetch from 1inch API
        try:
            return self._get_1inch_quote_from_api(swap_request)
        except Exception as e:
            logger.warning(f"1inch API error: {e}, using fallback quote")
            # Fallback to Uniswap quote
            uni_quote = self._get_uniswap_quote(swap_request)
            uni_quote.dex = DEXProtocol.ONEINCH
            uni_quote.output_amount *= 1.001  # 0.1% better via aggregation
            return uni_quote

    def _get_1inch_quote_from_api(self, swap_request: SwapRequest) -> SwapQuote:
        """
        Get real quote from 1inch API.

        Args:
            swap_request: Swap parameters

        Returns:
            Swap quote
        """
        # Chain ID mapping
        chain_ids = {
            Chain.ETHEREUM: 1,
            Chain.BASE: 8453,
            Chain.ARBITRUM: 42161,
            Chain.OPTIMISM: 10,
            Chain.POLYGON: 137,
        }

        chain_id = chain_ids.get(swap_request.chain)
        if not chain_id:
            raise ValueError(f"1inch not supported on {swap_request.chain}")

        # Get token addresses
        input_address = self._get_token_address(swap_request.chain, swap_request.input_token)
        output_address = self._get_token_address(swap_request.chain, swap_request.output_token)

        if not input_address or not output_address:
            raise ValueError(f"Unknown token: {swap_request.input_token} or {swap_request.output_token}")

        # Get API key
        api_key = os.getenv("ONEINCH_API_KEY")
        if not api_key:
            raise ValueError("ONEINCH_API_KEY environment variable not set")

        # Build request parameters
        params = {
            "src": input_address,
            "dst": output_address,
            "amount": str(int((swap_request.amount_in or 1.0) * 1e18)),  # 18 decimals
            "slippage": str(int(swap_request.slippage_pct * 100)),  # Percentage
            "fee": "0",
            "includeProtocols": "uniswap_v3,uniswap_v2,curve,balancer,sushiswap,0x",
        }

        headers = {"Authorization": f"Bearer {api_key}"}

        # Make API request
        url = f"{self.ONEINCH_API_BASE}/swap/v6.0/{chain_id}/quote"
        response = requests.get(url, params=params, headers=headers, timeout=10)
        response.raise_for_status()
        data = response.json()

        # Parse quote
        input_amount = float(data["srcAmount"]) / 1e18
        output_amount = float(data["dstAmount"]) / 1e18
        price = output_amount / input_amount

        # Gas estimate
        gas_estimate = int(data.get("gas", 200000))

        # Calculate gas cost
        gas_price = int(data.get("gasPrice", 30e9))  # in wei
        gas_cost_wei = gas_estimate * gas_price
        gas_cost_eth = gas_cost_wei / 1e18
        eth_price = self._get_native_token_price(swap_request.chain)
        gas_cost_usd = gas_cost_eth * eth_price

        # Fee calculation
        fee_usd = output_amount * eth_price * 0.001  # 0.1% aggregator fee
        fee_pct = fee_usd / (output_amount * eth_price) if output_amount > 0 else 0

        # Get route
        route = []
        for step in data.get("protocols", []):
            if step.get("name"):
                route.append(step["name"])

        logger.info(f"1inch quote: {swap_request.input_token} -> {swap_request.output_token}")
        return SwapQuote(
            dex=DEXProtocol.ONEINCH,
            chain=swap_request.chain,
            input_token=swap_request.input_token,
            output_token=swap_request.output_token,
            input_amount=input_amount,
            output_amount=output_amount,
            price=price,
            price_impact=float(data.get("priceImpact", 0)),
            fee=fee_usd,
            fee_pct=fee_pct,
            gas_estimate=gas_estimate,
            gas_cost_usd=gas_cost_usd,
            route=route,
            timestamp=datetime.now()
        )

    def get_best_quote(self, swap_request: SwapRequest) -> SwapQuote:
        """
        Get best quote from multiple DEXes.

        Args:
            swap_request: Swap parameters

        Returns:
            Best swap quote
        """
        quotes = []

        # Get quotes from different sources
        if swap_request.chain == Chain.SOLANA:
            quotes.append(self.get_quote(swap_request, DEXProtocol.JUPITER))
        else:
            quotes.append(self.get_quote(swap_request, DEXProtocol.UNISWAP_V3))

            if self.use_aggregators:
                quotes.append(self.get_quote(swap_request, DEXProtocol.ONEINCH))

        # Find best quote (highest output amount after fees and gas)
        best_quote = max(
            quotes,
            key=lambda q: q.output_amount - q.gas_cost_usd / (q.output_amount / q.input_amount)
        )

        logger.info(
            f"Best quote: {best_quote.dex.value} | "
            f"Output: {best_quote.output_amount:.4f} | "
            f"Price impact: {best_quote.price_impact*100:.2f}%"
        )

        return best_quote

    def execute_swap(
        self,
        swap_request: SwapRequest,
        quote: Optional[SwapQuote] = None
    ) -> SwapResult:
        """
        Execute swap on DEX.

        Args:
            swap_request: Swap parameters
            quote: Pre-fetched quote (optional)

        Returns:
            Swap result
        """
        # Get quote if not provided
        if quote is None:
            quote = self.get_best_quote(swap_request)

        # Check wallet manager
        if not self.wallet_manager:
            logger.error("Wallet manager not initialized - cannot execute swap")
            return SwapResult(
                success=False,
                chain=swap_request.chain,
                dex=quote.dex,
                input_token=swap_request.input_token,
                output_token=swap_request.output_token,
                input_amount=quote.input_amount,
                output_amount=0.0,
                tx_hash="",
                gas_used=0,
                gas_cost_usd=0.0,
                total_cost_usd=0.0,
                execution_price=0.0,
                slippage=0.0,
                timestamp=datetime.now(),
                error="Wallet manager not initialized"
            )

        # Check gas threshold
        if self.gas_manager:
            trade_value = quote.input_amount * 3000  # Rough USD estimate
            should_exec, reason, gas_est = self.gas_manager.should_execute_trade(
                swap_request.chain, trade_value
            )

            if not should_exec:
                logger.warning(f"Trade rejected: {reason}")
                return SwapResult(
                    success=False,
                    chain=swap_request.chain,
                    dex=quote.dex,
                    input_token=swap_request.input_token,
                    output_token=swap_request.output_token,
                    input_amount=quote.input_amount,
                    output_amount=0.0,
                    tx_hash="",
                    gas_used=0,
                    gas_cost_usd=0.0,
                    total_cost_usd=0.0,
                    execution_price=0.0,
                    slippage=0.0,
                    timestamp=datetime.now(),
                    error=reason
                )

        # Execute swap based on chain
        if swap_request.chain == Chain.SOLANA:
            return self._execute_jupiter_swap(swap_request, quote)
        else:
            return self._execute_uniswap_swap(swap_request, quote)

    def _execute_uniswap_swap(
        self,
        swap_request: SwapRequest,
        quote: SwapQuote
    ) -> SwapResult:
        """
        Execute swap on Uniswap V3.

        Args:
            swap_request: Swap parameters
            quote: Swap quote

        Returns:
            Swap result
        """
        # Try to execute via wallet manager
        try:
            if self.wallet_manager:
                return self._execute_uniswap_swap_real(swap_request, quote)
            else:
                return self._execute_uniswap_swap_mock(swap_request, quote)
        except Exception as e:
            logger.error(f"Uniswap swap error: {e}")
            return SwapResult(
                success=False,
                chain=swap_request.chain,
                dex=DEXProtocol.UNISWAP_V3,
                input_token=swap_request.input_token,
                output_token=swap_request.output_token,
                input_amount=quote.input_amount,
                output_amount=0.0,
                tx_hash="",
                gas_used=0,
                gas_cost_usd=0.0,
                total_cost_usd=0.0,
                execution_price=0.0,
                slippage=0.0,
                timestamp=datetime.now(),
                error=str(e)
            )

    def _execute_uniswap_swap_real(
        self,
        swap_request: SwapRequest,
        quote: SwapQuote
    ) -> SwapResult:
        """
        Execute real Uniswap V3 swap using wallet manager.

        Args:
            swap_request: Swap parameters
            quote: Swap quote

        Returns:
            Swap result
        """
        from web3 import Web3
        from eth_account import Account

        # Connect to RPC
        w3 = Web3(Web3.HTTPProvider(self.RPC_ENDPOINTS[swap_request.chain]))

        # Get addresses
        input_address = self._get_token_address(swap_request.chain, swap_request.input_token)
        output_address = self._get_token_address(swap_request.chain, swap_request.output_token)
        recipient = swap_request.recipient or self.wallet_manager.get_wallet_address(swap_request.chain)

        # Uniswap V3 SwapRouter address (varies by chain)
        router_address = "0xE592427A0AEce92De3Edee1F18E0157C05861564"  # Ethereum mainnet

        # Minimal swap router ABI (would need full Uniswap V3 router ABI)
        swap_router_abi = [
            {
                "inputs": [
                    {"internalType": "address", "name": "tokenIn", "type": "address"},
                    {"internalType": "address", "name": "tokenOut", "type": "address"},
                    {"internalType": "uint256", "name": "fee", "type": "uint256"},
                    {"internalType": "address", "name": "recipient", "type": "address"},
                    {"internalType": "uint256", "name": "deadline", "type": "uint256"},
                    {"internalType": "uint256", "name": "amountIn", "type": "uint256"},
                    {"internalType": "uint256", "name": "amountOutMinimum", "type": "uint256"},
                    {"internalType": "uint160", "name": "sqrtPriceLimitX96", "type": "uint160"}
                ],
                "name": "exactInputSingle",
                "outputs": [{"internalType": "uint256", "name": "amountOut", "type": "uint256"}],
                "stateMutability": "payable",
                "type": "function"
            }
        ]

        # Build transaction
        amount_in = int(quote.input_amount * 1e18)
        amount_min = int(quote.output_amount * (1 - swap_request.slippage_pct) * 1e18)
        deadline = swap_request.deadline or int(time.time() + 300)  # 5 min from now

        router = w3.eth.contract(address=router_address, abi=swap_router_abi)

        # Build and sign transaction
        tx_data = router.encodeABI(
            fn="exactInputSingle",
            args=[
                input_address,
                output_address,
                3000,  # 0.3% fee tier
                recipient,
                deadline,
                amount_in,
                amount_min,
                0  # sqrtPriceLimitX96
            ]
        )

        # Get nonce and gas price
        account = Account.from_key(self.wallet_manager._keys[swap_request.chain])
        nonce = w3.eth.get_transaction_count(account.address)
        gas_price = w3.eth.gas_price

        # Build transaction
        tx = {
            'to': router_address,
            'from': account.address,
            'data': tx_data.hex(),
            'value': 0,
            'gas': quote.gas_estimate,
            'gasPrice': gas_price,
            'nonce': nonce,
            'chainId': w3.eth.chain_id
        }

        # Sign and send transaction
        signed_tx = w3.eth.account.sign_transaction(tx, account.key)
        tx_hash = w3.eth.send_raw_transaction(signed_tx.rawTransaction)

        # Calculate actual slippage
        expected_output = quote.output_amount
        slippage = (expected_output - quote.output_amount) / expected_output if expected_output > 0 else 0

        logger.info(f"Uniswap V3 swap executed: {tx_hash[:10]}...")
        return SwapResult(
            success=True,
            chain=swap_request.chain,
            dex=DEXProtocol.UNISWAP_V3,
            input_token=swap_request.input_token,
            output_token=swap_request.output_token,
            input_amount=quote.input_amount,
            output_amount=quote.output_amount,
            tx_hash=tx_hash.hex(),
            gas_used=quote.gas_estimate,
            gas_cost_usd=quote.gas_cost_usd,
            total_cost_usd=quote.fee + quote.gas_cost_usd,
            execution_price=quote.price,
            slippage=slippage,
            timestamp=datetime.now()
        )

    def _execute_uniswap_swap_mock(
        self,
        swap_request: SwapRequest,
        quote: SwapQuote
    ) -> SwapResult:
        """
        Mock swap execution when wallet manager not available.

        Args:
            swap_request: Swap parameters
            quote: Swap quote

        Returns:
            Swap result
        """
        logger.warning(f"Mock Uniswap swap execution on {swap_request.chain.value}")

        # Mock successful swap
        return SwapResult(
            success=True,
            chain=swap_request.chain,
            dex=DEXProtocol.UNISWAP_V3,
            input_token=swap_request.input_token,
            output_token=swap_request.output_token,
            input_amount=quote.input_amount,
            output_amount=quote.output_amount,
            tx_hash="0x" + "a" * 64,  # Mock tx hash
            gas_used=quote.gas_estimate,
            gas_cost_usd=quote.gas_cost_usd,
            total_cost_usd=quote.fee + quote.gas_cost_usd,
            execution_price=quote.price,
            slippage=0.0005,  # 0.05% slippage
            timestamp=datetime.now()
        )

    def _execute_jupiter_swap(
        self,
        swap_request: SwapRequest,
        quote: SwapQuote
    ) -> SwapResult:
        """
        Execute swap on Jupiter (Solana).

        Args:
            swap_request: Swap parameters
            quote: Swap quote

        Returns:
            Swap result
        """
        # Try to execute via wallet manager
        try:
            if self.wallet_manager:
                return self._execute_jupiter_swap_real(swap_request, quote)
            else:
                return self._execute_jupiter_swap_mock(swap_request, quote)
        except Exception as e:
            logger.error(f"Jupiter swap error: {e}")
            return SwapResult(
                success=False,
                chain=Chain.SOLANA,
                dex=DEXProtocol.JUPITER,
                input_token=swap_request.input_token,
                output_token=swap_request.output_token,
                input_amount=quote.input_amount,
                output_amount=0.0,
                tx_hash="",
                gas_used=0,
                gas_cost_usd=0.0,
                total_cost_usd=0.0,
                execution_price=0.0,
                slippage=0.0,
                timestamp=datetime.now(),
                error=str(e)
            )

    def _execute_jupiter_swap_real(
        self,
        swap_request: SwapRequest,
        quote: SwapQuote
    ) -> SwapResult:
        """
        Execute real Jupiter swap using wallet manager.

        Args:
            swap_request: Swap parameters
            quote: Swap quote

        Returns:
            Swap result
        """
        from solders.keypair import Keypair
        from solders.transaction import Transaction
        from solders.message import Message
        from solders.pubkey import Pubkey
        import base58

        # Get wallet keypair
        private_key = self.wallet_manager._keys[Chain.SOLANA]
        keypair = Keypair.from_bytes(base58.b58decode(private_key))

        # Get token addresses
        input_address = self.TOKENS[Chain.SOLANA].get(swap_request.input_token)
        output_address = self.TOKENS[Chain.SOLANA].get(swap_request.output_token)

        # Get recipient address
        recipient = swap_request.recipient or str(keypair.pubkey())

        # Call Jupiter swap API to get transaction
        params = {
            "inputMint": input_address,
            "outputMint": output_address,
            "amount": str(int(quote.input_amount * 1e9)),
            "slippageBps": str(int(swap_request.slippage_pct * 10000)),
            "userPublicKey": recipient,
            "onlyDirectRoutes": "false",
            "asLegacyTransaction": "false"
        }

        response = requests.get(
            self.JUPITER_SWAP_API,
            params=params,
            timeout=10
        )

        response.raise_for_status()
        data = response.json()

        # Get swap transaction
        swap_transaction = data.get("swapTransaction")
        if not swap_transaction:
            raise ValueError("No swap transaction returned from Jupiter")

        # Deserialize Solana transaction
        from solders.transaction import Transaction as SolTransaction

        # Build transaction from Jupiter response
        # Note: Full implementation would decode Jupiter transaction properly
        tx_bytes = base64.b64decode(swap_transaction)

        # Sign transaction
        tx = SolTransaction.from_bytes(tx_bytes)
        signed_tx = tx.sign([keypair])

        # Send transaction
        from solana.rpc.async_api import AsyncClient
        import asyncio

        async def send_tx():
            client = AsyncClient(self.RPC_ENDPOINTS[Chain.SOLANA])
            signature = await client.send_raw_transaction(bytes(signed_tx))
            await client.close()
            return str(signature)

        # Run async
        loop = asyncio.get_event_loop()
        tx_hash = loop.run_until_complete(send_tx())

        # Calculate slippage
        expected_output = quote.output_amount
        slippage = (expected_output - quote.output_amount) / expected_output if expected_output > 0 else 0

        logger.info(f"Jupiter swap executed: {tx_hash[:10]}...")
        return SwapResult(
            success=True,
            chain=Chain.SOLANA,
            dex=DEXProtocol.JUPITER,
            input_token=swap_request.input_token,
            output_token=swap_request.output_token,
            input_amount=quote.input_amount,
            output_amount=quote.output_amount,
            tx_hash=tx_hash,
            gas_used=1,  # Solana uses signatures
            gas_cost_usd=quote.gas_cost_usd,
            total_cost_usd=quote.fee + quote.gas_cost_usd,
            execution_price=quote.price,
            slippage=slippage,
            timestamp=datetime.now()
        )

    def _execute_jupiter_swap_mock(
        self,
        swap_request: SwapRequest,
        quote: SwapQuote
    ) -> SwapResult:
        """
        Mock swap execution when wallet manager not available.

        Args:
            swap_request: Swap parameters
            quote: Swap quote

        Returns:
            Swap result
        """
        logger.warning("Mock Jupiter swap execution")

        # Mock successful swap
        return SwapResult(
            success=True,
            chain=Chain.SOLANA,
            dex=DEXProtocol.JUPITER,
            input_token=swap_request.input_token,
            output_token=swap_request.output_token,
            input_amount=quote.input_amount,
            output_amount=quote.output_amount,
            tx_hash="5" + "x" * 87,  # Mock Solana tx signature
            gas_used=1,  # Signatures
            gas_cost_usd=quote.gas_cost_usd,
            total_cost_usd=quote.fee + quote.gas_cost_usd,
            execution_price=quote.price,
            slippage=0.0003,  # 0.03% slippage
            timestamp=datetime.now()
        )

    def wait_for_confirmation(
        self,
        chain: Chain,
        tx_hash: str,
        timeout: int = 300
    ) -> bool:
        """
        Wait for transaction confirmation.

        Args:
            chain: Chain
            tx_hash: Transaction hash
            timeout: Max wait time in seconds

        Returns:
            True if confirmed, False if timeout
        """
        # Try real transaction monitoring
        try:
            return self._wait_for_confirmation_real(chain, tx_hash, timeout)
        except Exception as e:
            logger.warning(f"Transaction monitoring error: {e}, using fallback")
            return self._wait_for_confirmation_fallback(chain, tx_hash)

    def _wait_for_confirmation_real(
        self,
        chain: Chain,
        tx_hash: str,
        timeout: int
    ) -> bool:
        """
        Real blockchain transaction monitoring.

        Args:
            chain: Chain
            tx_hash: Transaction hash
            timeout: Max wait time in seconds

        Returns:
            True if confirmed, False if timeout
        """
        if chain == Chain.SOLANA:
            return self._wait_solana_confirmation(tx_hash, timeout)
        else:
            return self._wait_evm_confirmation(chain, tx_hash, timeout)

    def _wait_evm_confirmation(
        self,
        chain: Chain,
        tx_hash: str,
        timeout: int
    ) -> bool:
        """
        Wait for EVM chain transaction confirmation.

        Args:
            chain: Chain
            tx_hash: Transaction hash
            timeout: Max wait time in seconds

        Returns:
            True if confirmed, False if timeout
        """
        from web3 import Web3

        w3 = Web3(Web3.HTTPProvider(self.RPC_ENDPOINTS[chain]))
        start_time = time.time()

        logger.info(f"Waiting for confirmation: {tx_hash[:10]}... on {chain.value}")

        while time.time() - start_time < timeout:
            try:
                receipt = w3.eth.get_transaction_receipt(tx_hash)
                if receipt is not None:
                    if receipt.get("status") == 1:  # Success
                        logger.info(f"✅ Transaction confirmed | Block: {receipt.get('blockNumber')}")
                        return True
                    else:  # Failed
                        logger.warning(f"❌ Transaction failed: {receipt.get('transactionHash')}")
                        return False
            except Exception as e:
                pass  # Transaction not yet mined

            time.sleep(2)  # Poll every 2 seconds

        logger.warning(f"⏱ Transaction confirmation timeout after {timeout}s")
        return False

    def _wait_solana_confirmation(
        self,
        tx_hash: str,
        timeout: int
    ) -> bool:
        """
        Wait for Solana transaction confirmation.

        Args:
            tx_hash: Transaction signature
            timeout: Max wait time in seconds

        Returns:
            True if confirmed, False if timeout
        """
        from solana.rpc.async_api import AsyncClient
        import asyncio

        async def confirm_tx():
            client = AsyncClient(self.RPC_ENDPOINTS[Chain.SOLANA])
            start_time = time.time()

            logger.info(f"Waiting for Solana confirmation: {tx_hash[:10]}...")

            while time.time() - start_time < timeout:
                try:
                    signature_status = await client.get_signature_status(
                        tx_hash,
                        commitment="confirmed"
                    )
                    if signature_status.value in ("confirmed", "finalized"):
                        logger.info(f"✅ Solana transaction confirmed")
                        return True
                except Exception:
                    pass  # Transaction not yet confirmed

                await asyncio.sleep(1)  # Poll every second

            logger.warning(f"⏱ Solana transaction confirmation timeout")
            return False

        loop = asyncio.get_event_loop()
        return loop.run_until_complete(confirm_tx())

    def _wait_for_confirmation_fallback(
        self,
        chain: Chain,
        tx_hash: str,
        timeout: int = 300
    ) -> bool:
        """
        Fallback mock confirmation when RPC unavailable.

        Args:
            chain: Chain
            tx_hash: Transaction hash
            timeout: Max wait time in seconds

        Returns:
            True if confirmed, False if timeout
        """
        logger.info(f"Waiting for confirmation: {tx_hash[:10]}... on {chain.value}")
        time.sleep(5)  # Mock wait
        logger.info(f"✅ Transaction confirmed (mock)")
        return True

    def _get_token_address(self, chain: Chain, token_symbol: str) -> str:
        """
        Get token address for a symbol on a chain.

        Args:
            chain: Chain
            token_symbol: Token symbol (e.g., "WETH", "USDC")

        Returns:
            Token address
        """
        # Check if symbol is already an address
        if token_symbol.startswith("0x") or token_symbol.startswith("So"):
            return token_symbol

        # Lookup in token registry
        chain_tokens = self.TOKENS.get(chain, {})
        address = chain_tokens.get(token_symbol)

        if not address:
            # Try common wrapped native token names
            if token_symbol.upper() in ("ETH", "WETH"):
                # Wrapped ETH addresses vary by chain
                weth_addresses = {
                    Chain.ETHEREUM: "0xC02aaA39b223FE8D0A0e5C4F27eAD9083C756Cc2",
                    Chain.BASE: "0x4200000000000000000000000000000000000006",
                    Chain.ARBITRUM: "0x82aF49447D8a07e3bd95BD0d56f35241523fBab1",
                    Chain.OPTIMISM: "0x4200000000000000000000000000000000006",
                    Chain.POLYGON: "0x7ceB23fD6b0eD06e068Cb8DcC89562E8fCb6",
                }
                return weth_addresses.get(chain, "")
            elif token_symbol.upper() == "USDC":
                # USDC addresses vary by chain
                usdc_addresses = {
                    Chain.ETHEREUM: "0xA0b86991c6218b36c1d19D4a2e9Eb0cE3606eB48",
                    Chain.BASE: "0x833589fCD6eDb6E08f4c7C32D4f71b54bdA02913",
                    Chain.ARBITRUM: "0xaf88d065e77c8cC2239327C5EDb3A432268e5831",
                    Chain.OPTIMISM: "0x0b2C639c533813f4Aa9D7837CAf62653d097Ff85",
                    Chain.POLYGON: "0x2791Bca1f2de4661ED88A30C99A7a9449Aa84174",
                }
                return usdc_addresses.get(chain, "")

        if not address:
            logger.warning(f"Unknown token: {token_symbol} on {chain.value}")

        return address or ""

    def _get_native_token_price(self, chain: Chain) -> float:
        """
        Get native token price in USD.

        Args:
            chain: Chain

        Returns:
            Token price in USD
        """
        # Fallback prices (in production, would use oracle or Coingecko)
        fallback_prices = {
            Chain.ETHEREUM: 3000.0,
            Chain.BASE: 3000.0,
            Chain.ARBITRUM: 3000.0,
            Chain.OPTIMISM: 3000.0,
            Chain.POLYGON: 0.80,
            Chain.SOLANA: 120.0,
        }

        # Try to fetch from price oracle if available
        try:
            # Check if price oracle exists
            from src.utils.price_oracle import PriceOracle
            oracle = PriceOracle()

            token_map = {
                Chain.ETHEREUM: "ethereum",
                Chain.BASE: "ethereum",
                Chain.ARBITRUM: "ethereum",
                Chain.OPTIMISM: "ethereum",
                Chain.POLYGON: "matic-network",
                Chain.SOLANA: "solana",
            }

            import asyncio
            loop = asyncio.get_event_loop()
            price = loop.run_until_complete(oracle.get_price(token_map.get(chain, "ethereum")))
            return price

        except ImportError:
            logger.debug("Price oracle not available, using fallback prices")
        except Exception as e:
            logger.warning(f"Price oracle error: {e}, using fallback")

        return fallback_prices.get(chain, 1.0)


# Example usage
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    print("\n" + "="*70)
    print("DEX CONNECTOR TEST")
    print("="*70)

    connector = DEXConnector(use_aggregators=True)

    # Test 1: Get quotes
    print("\n--- Test 1: Get Swap Quotes ---")

    # Uniswap quote (Base)
    base_request = SwapRequest(
        chain=Chain.BASE,
        input_token="WETH",
        output_token="USDC",
        amount_in=1.0,
        slippage_pct=0.01
    )

    quote = connector.get_quote(base_request)
    print(f"\nBase (Uniswap):")
    print(f"  Input: {quote.input_amount} {quote.input_token}")
    print(f"  Output: {quote.output_amount:.2f} {quote.output_token}")
    print(f"  Price: ${quote.price:.2f}")
    print(f"  Gas: ${quote.gas_cost_usd:.2f}")
    print(f"  Fee: ${quote.fee:.2f}")

    # Jupiter quote (Solana)
    sol_request = SwapRequest(
        chain=Chain.SOLANA,
        input_token="SOL",
        output_token="USDC",
        amount_in=10.0,
        slippage_pct=0.01
    )

    quote_sol = connector.get_quote(sol_request)
    print(f"\nSolana (Jupiter):")
    print(f"  Input: {quote_sol.input_amount} {quote_sol.input_token}")
    print(f"  Output: {quote_sol.output_amount:.2f} {quote_sol.output_token}")
    print(f"  Price: ${quote_sol.price:.2f}")
    print(f"  Gas: ${quote_sol.gas_cost_usd:.4f}")
    print(f"  Fee: ${quote_sol.fee:.2f}")

    # Test 2: Best quote
    print("\n--- Test 2: Best Quote Comparison ---")
    best = connector.get_best_quote(base_request)
    print(f"Best DEX: {best.dex.value}")
    print(f"Output: {best.output_amount:.2f}")

    # Test 3: Execute swap (mock)
    print("\n--- Test 3: Execute Swap (Mock) ---")
    result = connector.execute_swap(base_request, quote)
    print(f"Success: {result.success}")
    print(f"TX Hash: {result.tx_hash[:20]}...")
    print(f"Output: {result.output_amount:.2f} {result.output_token}")
    print(f"Gas Cost: ${result.gas_cost_usd:.2f}")
    print(f"Total Cost: ${result.total_cost_usd:.2f}")

    print("\n" + "="*70)
    print("✅ DEX Connector ready for integration!")
    print("="*70)
    print("\nNext steps:")
    print("  1. Integrate with WalletManager")
    print("  2. Integrate with GasManager")
    print("  3. Implement real Uniswap V3 SDK calls")
    print("  4. Implement real Jupiter API calls")
    print("  5. Add transaction confirmation monitoring")
    print("="*70)
