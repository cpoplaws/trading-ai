"""
DEX Aggregator for finding best prices across multiple decentralized exchanges.
"""
import logging
from typing import Dict, List, Optional, Any, Tuple
from web3 import Web3
from enum import Enum

logger = logging.getLogger(__name__)


class DEX(Enum):
    """Supported DEXs."""
    UNISWAP_V2 = "uniswap_v2"
    UNISWAP_V3 = "uniswap_v3"
    SUSHISWAP = "sushiswap"
    PANCAKESWAP = "pancakeswap"
    CURVE = "curve"
    BALANCER = "balancer"
    ONEINCH = "1inch"


class DEXAggregator:
    """
    DEX aggregator for optimal trade routing across multiple exchanges.
    Integrates with 1inch, Paraswap, 0x APIs for best execution.
    """
    
    # API endpoints
    ONEINCH_API = "https://api.1inch.dev/swap/v5.2"
    PARASWAP_API = "https://apiv5.paraswap.io"
    ZEROX_API = "https://api.0x.org"
    
    # Common router addresses (Ethereum mainnet)
    ROUTERS = {
        DEX.UNISWAP_V2: "0x7a250d5630B4cF539739dF2C5dAcb4c659F2488D",
        DEX.UNISWAP_V3: "0xE592427A0AEce92De3Edee1F18E0157C05861564",
        DEX.SUSHISWAP: "0xd9e1cE17f2641f24aE83637ab66a2cca9C378B9F",
        DEX.CURVE: "0x8e764bE4288B842791989DB5b8ec067279829809",
        DEX.BALANCER: "0xBA12222222228d8Ba445958a75a0704d566BF2C8",
    }
    
    def __init__(self, chain_id: int = 1, web3_provider: Optional[Any] = None):
        """
        Initialize DEX aggregator.
        
        Args:
            chain_id: Blockchain network ID (1=Ethereum, 56=BSC, 137=Polygon)
            web3_provider: Web3 provider instance
        """
        self.chain_id = chain_id
        self.w3 = web3_provider
        
        logger.info(f"DEX Aggregator initialized for chain_id={chain_id}")
    
    def get_best_quote(self, token_in: str, token_out: str, amount_in: float,
                       excluded_dexs: Optional[List[DEX]] = None) -> Optional[Dict]:
        """
        Get best quote across all DEXs.
        
        Args:
            token_in: Input token address
            token_out: Output token address
            amount_in: Input amount
            excluded_dexs: DEXs to exclude from search
            
        Returns:
            Best quote with DEX, price, and route info
        """
        try:
            quotes = []
            excluded = excluded_dexs or []
            
            # Query 1inch
            if DEX.ONEINCH not in excluded:
                oneinch_quote = self._get_1inch_quote(token_in, token_out, amount_in)
                if oneinch_quote:
                    quotes.append(oneinch_quote)
            
            # Query direct DEXs
            for dex in DEX:
                if dex not in excluded and dex != DEX.ONEINCH:
                    quote = self._get_dex_quote(dex, token_in, token_out, amount_in)
                    if quote:
                        quotes.append(quote)
            
            if not quotes:
                logger.warning("No quotes available")
                return None
            
            # Find best quote (highest output)
            best_quote = max(quotes, key=lambda q: q['amount_out'])
            
            logger.info(f"Best quote: {best_quote['amount_out']} on {best_quote['dex']}")
            return best_quote
            
        except Exception as e:
            logger.error(f"Error getting best quote: {e}")
            return None
    
    def compare_dexs(self, token_in: str, token_out: str, amount_in: float) -> List[Dict]:
        """
        Compare prices across all DEXs.
        
        Args:
            token_in: Input token address
            token_out: Output token address
            amount_in: Input amount
            
        Returns:
            List of quotes from all DEXs, sorted by best price
        """
        try:
            quotes = []
            
            # Query all DEXs
            for dex in DEX:
                if dex == DEX.ONEINCH:
                    quote = self._get_1inch_quote(token_in, token_out, amount_in)
                else:
                    quote = self._get_dex_quote(dex, token_in, token_out, amount_in)
                
                if quote:
                    quotes.append(quote)
            
            # Sort by output amount (descending)
            quotes.sort(key=lambda q: q['amount_out'], reverse=True)
            
            return quotes
            
        except Exception as e:
            logger.error(f"Error comparing DEXs: {e}")
            return []
    
    def find_arbitrage_opportunities(self, token_pairs: List[Tuple[str, str]],
                                    min_profit_bps: int = 30) -> List[Dict]:
        """
        Find arbitrage opportunities across DEXs.
        
        Args:
            token_pairs: List of (token_in, token_out) pairs to check
            min_profit_bps: Minimum profit in basis points (0.3% = 30 bps)
            
        Returns:
            List of arbitrage opportunities
        """
        opportunities = []
        
        try:
            for token_in, token_out in token_pairs:
                # Get quotes from all DEXs
                amount_in = 1.0  # Use 1 token for comparison
                quotes = self.compare_dexs(token_in, token_out, amount_in)
                
                if len(quotes) < 2:
                    continue
                
                # Check for arbitrage
                best_buy = quotes[0]  # Best DEX to buy token_out
                best_sell = quotes[-1]  # Best DEX to sell token_out (buy back token_in)
                
                # Calculate potential profit
                # Buy on best_sell DEX, sell on best_buy DEX
                profit = best_buy['amount_out'] - best_sell['amount_out']
                profit_bps = (profit / best_sell['amount_out']) * 10000
                
                if profit_bps >= min_profit_bps:
                    opportunities.append({
                        'token_in': token_in,
                        'token_out': token_out,
                        'buy_dex': best_sell['dex'],
                        'sell_dex': best_buy['dex'],
                        'buy_price': best_sell['price'],
                        'sell_price': best_buy['price'],
                        'profit_bps': profit_bps,
                        'profit_percent': profit_bps / 100
                    })
            
            # Sort by profit
            opportunities.sort(key=lambda o: o['profit_bps'], reverse=True)
            
            logger.info(f"Found {len(opportunities)} arbitrage opportunities")
            return opportunities
            
        except Exception as e:
            logger.error(f"Error finding arbitrage: {e}")
            return []
    
    def calculate_price_impact(self, token_in: str, token_out: str,
                              amounts: List[float]) -> List[Dict]:
        """
        Calculate price impact for different trade sizes.
        
        Args:
            token_in: Input token address
            token_out: Output token address
            amounts: List of amounts to test
            
        Returns:
            List of price impact data for each amount
        """
        impacts = []
        
        try:
            base_quote = self.get_best_quote(token_in, token_out, amounts[0])
            if not base_quote:
                return []
            
            base_price = base_quote['price']
            
            for amount in amounts:
                quote = self.get_best_quote(token_in, token_out, amount)
                if quote:
                    price_impact = ((quote['price'] - base_price) / base_price) * 100
                    impacts.append({
                        'amount': amount,
                        'price': quote['price'],
                        'price_impact_percent': price_impact,
                        'dex': quote['dex']
                    })
            
            return impacts
            
        except Exception as e:
            logger.error(f"Error calculating price impact: {e}")
            return []
    
    def _get_1inch_quote(self, token_in: str, token_out: str, amount_in: float) -> Optional[Dict]:
        """Get quote from 1inch API."""
        try:
            # Simplified - would need actual API implementation
            # This is a placeholder showing the structure
            return {
                'dex': '1inch_aggregator',
                'amount_in': amount_in,
                'amount_out': amount_in * 3000,  # Placeholder
                'price': 3000,
                'gas_cost': 150000,
                'route': ['1inch']
            }
        except Exception as e:
            logger.error(f"Error getting 1inch quote: {e}")
            return None
    
    def _get_dex_quote(self, dex: DEX, token_in: str, token_out: str,
                       amount_in: float) -> Optional[Dict]:
        """Get quote from specific DEX."""
        try:
            # Simplified implementation
            # In production, would query actual DEX contracts
            router_address = self.ROUTERS.get(dex)
            if not router_address:
                return None
            
            # Placeholder quote
            return {
                'dex': dex.value,
                'amount_in': amount_in,
                'amount_out': amount_in * 2950,  # Placeholder
                'price': 2950,
                'gas_cost': 120000,
                'route': [dex.value]
            }
        except Exception as e:
            logger.error(f"Error getting {dex.value} quote: {e}")
            return None
    
    def estimate_optimal_route(self, token_in: str, token_out: str,
                              amount_in: float, max_splits: int = 3) -> Optional[Dict]:
        """
        Find optimal routing strategy including split trades.
        
        Args:
            token_in: Input token address
            token_out: Output token address
            amount_in: Input amount
            max_splits: Maximum number of routes to split across
            
        Returns:
            Optimal routing strategy
        """
        try:
            # Get all quotes
            quotes = self.compare_dexs(token_in, token_out, amount_in)
            
            if not quotes:
                return None
            
            # Simple strategy: use best single route
            # In production, would implement multi-path routing
            best = quotes[0]
            
            return {
                'strategy': 'single_route',
                'routes': [{
                    'dex': best['dex'],
                    'percentage': 100,
                    'amount_in': amount_in,
                    'amount_out': best['amount_out']
                }],
                'total_amount_out': best['amount_out'],
                'total_gas_cost': best['gas_cost']
            }
            
        except Exception as e:
            logger.error(f"Error estimating optimal route: {e}")
            return None


if __name__ == "__main__":
    # Test DEX aggregator
    aggregator = DEXAggregator(chain_id=1)
    
    print("=== DEX Aggregator Test ===")
    
    # Example token addresses (ETH/USDC)
    WETH = "0xC02aaA39b223FE8D0A0e5C4F27eAD9083C756Cc2"
    USDC = "0xA0b86991c6218b36c1d19D4a2e9Eb0cE3606eB48"
    
    # Get best quote
    quote = aggregator.get_best_quote(WETH, USDC, 1.0)
    if quote:
        print(f"Best Quote: {quote['amount_out']:.2f} USDC on {quote['dex']}")
    
    # Compare DEXs
    quotes = aggregator.compare_dexs(WETH, USDC, 1.0)
    print(f"\nFound {len(quotes)} quotes")
    for q in quotes[:3]:
        print(f"  {q['dex']}: {q['amount_out']:.2f} USDC")
    
    print("\n✅ DEX aggregator test completed!")
