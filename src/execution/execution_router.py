"""
ExecutionRouter - Smart routing between CEX and DEX

Responsibilities:
1. Get quotes from both CEX and DEX for a trade
2. Calculate net output after fees and gas
3. Route to the venue with best execution
4. Handle execution via appropriate connector

Decision logic:
- If DEX gas > 2% of trade value → use CEX
- Otherwise, use venue with best net output
"""

from dataclasses import dataclass
from typing import Dict, Optional, Literal
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class VenueType(Enum):
    """Type of trading venue"""
    CEX = "cex"
    DEX = "dex"


@dataclass
class Quote:
    """Quote from a trading venue"""
    venue_type: VenueType
    venue_name: str  # "Binance", "Base:Uniswap", etc.
    symbol: str
    side: Literal["buy", "sell"]
    quantity: float
    price: float
    total: float  # quantity * price
    fee: float  # in USD
    fee_pct: float
    gas: float = 0.0  # DEX only, in USD
    gas_pct: float = 0.0
    net_output: float = 0.0  # Total after all costs
    timestamp: str = ""


@dataclass
class RoutingDecision:
    """Result of routing decision"""
    selected_venue: VenueType
    selected_quote: Quote
    rejected_quote: Optional[Quote]
    reason: str
    savings: float  # How much better than alternative (USD)


class ExecutionRouter:
    """
    Smart router that selects best execution venue for each trade.

    Flow:
    1. Request quotes from CEX and DEX
    2. Calculate net output for each
    3. Apply rules (gas threshold, etc.)
    4. Select best venue
    5. Route to appropriate connector
    """

    def __init__(
        self,
        max_gas_pct: float = 0.02,  # 2% max gas as % of trade
        cex_priority: bool = False,  # If true, prefer CEX when equal
    ):
        self.max_gas_pct = max_gas_pct
        self.cex_priority = cex_priority

        # These will be set by integration with actual connectors
        self.cex_connector = None
        self.dex_connector = None

        logger.info(
            f"ExecutionRouter initialized | "
            f"Max gas: {max_gas_pct*100:.1f}% | "
            f"CEX priority: {cex_priority}"
        )

    def get_best_route(
        self,
        symbol: str,
        side: Literal["buy", "sell"],
        quantity: float,
        chain: str = "Base",
    ) -> RoutingDecision:
        """
        Get the best execution route for a trade.

        Args:
            symbol: Trading pair (e.g., "WETH/USDC")
            side: "buy" or "sell"
            quantity: Amount to trade
            chain: Chain for DEX quote (if routing to DEX)

        Returns:
            RoutingDecision with selected venue and rationale
        """
        # Get quotes from both venues
        cex_quote = self._get_cex_quote(symbol, side, quantity)
        dex_quote = self._get_dex_quote(symbol, side, quantity, chain)

        # Calculate net outputs
        if side == "buy":
            # For buys, we want maximum tokens received for our USD
            cex_quote.net_output = cex_quote.quantity - (cex_quote.fee / cex_quote.price)
            dex_quote.net_output = dex_quote.quantity - (dex_quote.fee / dex_quote.price) - (dex_quote.gas / dex_quote.price)
        else:  # sell
            # For sells, we want maximum USD received for our tokens
            cex_quote.net_output = cex_quote.total - cex_quote.fee
            dex_quote.net_output = dex_quote.total - dex_quote.fee - dex_quote.gas

        # Apply routing rules
        decision = self._apply_routing_rules(cex_quote, dex_quote)

        logger.info(
            f"Routing decision: {symbol} {side} {quantity} | "
            f"Selected: {decision.selected_venue.value} | "
            f"Reason: {decision.reason} | "
            f"Savings: ${decision.savings:.2f}"
        )

        return decision

    def _apply_routing_rules(
        self,
        cex_quote: Quote,
        dex_quote: Quote
    ) -> RoutingDecision:
        """
        Apply routing rules to select best venue.

        Rules:
        1. If DEX gas > max_gas_pct → use CEX
        2. Otherwise, use venue with best net output
        3. If equal, use CEX (if cex_priority=True)
        """
        # Rule 1: Check gas threshold
        if dex_quote.gas_pct > self.max_gas_pct:
            return RoutingDecision(
                selected_venue=VenueType.CEX,
                selected_quote=cex_quote,
                rejected_quote=dex_quote,
                reason=f"DEX gas too high ({dex_quote.gas_pct*100:.2f}% > {self.max_gas_pct*100:.1f}%)",
                savings=0.0,  # Not about savings, about practicality
            )

        # Rule 2: Compare net outputs
        if cex_quote.net_output > dex_quote.net_output:
            savings = cex_quote.net_output - dex_quote.net_output
            return RoutingDecision(
                selected_venue=VenueType.CEX,
                selected_quote=cex_quote,
                rejected_quote=dex_quote,
                reason=f"CEX better net output",
                savings=savings,
            )
        elif dex_quote.net_output > cex_quote.net_output:
            savings = dex_quote.net_output - cex_quote.net_output
            return RoutingDecision(
                selected_venue=VenueType.DEX,
                selected_quote=dex_quote,
                rejected_quote=cex_quote,
                reason=f"DEX better net output",
                savings=savings,
            )
        else:
            # Equal - use priority
            if self.cex_priority:
                return RoutingDecision(
                    selected_venue=VenueType.CEX,
                    selected_quote=cex_quote,
                    rejected_quote=dex_quote,
                    reason="Equal quotes, CEX priority enabled",
                    savings=0.0,
                )
            else:
                return RoutingDecision(
                    selected_venue=VenueType.DEX,
                    selected_quote=dex_quote,
                    rejected_quote=cex_quote,
                    reason="Equal quotes, DEX priority",
                    savings=0.0,
                )

    def _get_cex_quote(
        self,
        symbol: str,
        side: Literal["buy", "sell"],
        quantity: float
    ) -> Quote:
        """
        Get quote from centralized exchange.

        Currently returns mock data - will integrate with CEXConnector.
        """
        # Mock price data (will be replaced with real CEX API calls)
        mock_prices = {
            "BTC/USDT": 64000.0,
            "ETH/USDT": 3000.0,
            "SOL/USDT": 120.0,
            "WETH/USDC": 3000.0,
            "SOL/USDC": 120.0,
        }

        price = mock_prices.get(symbol, 100.0)
        total = quantity * price
        fee_pct = 0.001  # 0.1% typical for CEX
        fee = total * fee_pct

        return Quote(
            venue_type=VenueType.CEX,
            venue_name="Binance",  # TODO: Get from config
            symbol=symbol,
            side=side,
            quantity=quantity,
            price=price,
            total=total,
            fee=fee,
            fee_pct=fee_pct,
            gas=0.0,  # CEX has no gas
            gas_pct=0.0,
        )

    def _get_dex_quote(
        self,
        symbol: str,
        side: Literal["buy", "sell"],
        quantity: float,
        chain: str
    ) -> Quote:
        """
        Get quote from decentralized exchange.

        Currently returns mock data - will integrate with DEXConnector.
        """
        # Mock price data (slightly different from CEX to show routing logic)
        mock_prices = {
            "WETH/USDC": 2998.0,  # Slightly worse than CEX
            "WETH/USDbC": 2998.0,
            "SOL/USDC": 119.5,
            "cbETH/WETH": 1.02,
        }

        price = mock_prices.get(symbol, 100.0)
        total = quantity * price
        fee_pct = 0.003  # 0.3% typical for Uniswap
        fee = total * fee_pct

        # Mock gas estimation
        gas_cost_usd = 8.0  # ~$8 for a swap on Base
        gas_pct = gas_cost_usd / total if total > 0 else 0.0

        return Quote(
            venue_type=VenueType.DEX,
            venue_name=f"{chain}:Uniswap",
            symbol=symbol,
            side=side,
            quantity=quantity,
            price=price,
            total=total,
            fee=fee,
            fee_pct=fee_pct,
            gas=gas_cost_usd,
            gas_pct=gas_pct,
        )

    def execute_trade(
        self,
        decision: RoutingDecision,
        instance_id: str
    ) -> Dict:
        """
        Execute trade via the selected venue.

        Args:
            decision: Routing decision with selected venue
            instance_id: Strategy instance making the trade

        Returns:
            Trade result dict
        """
        quote = decision.selected_quote

        logger.info(
            f"Executing trade: {instance_id} | "
            f"{quote.symbol} {quote.side} {quote.quantity} | "
            f"Via: {quote.venue_name} | "
            f"Net: ${quote.net_output:.2f}"
        )

        # TODO: Route to actual connector
        if decision.selected_venue == VenueType.CEX:
            result = self._execute_via_cex(quote, instance_id)
        else:
            result = self._execute_via_dex(quote, instance_id)

        return result

    def _execute_via_cex(self, quote: Quote, instance_id: str) -> Dict:
        """Execute trade via CEX connector (mock for now)"""
        # TODO: Integrate with CEXConnector
        return {
            "success": True,
            "venue": quote.venue_name,
            "symbol": quote.symbol,
            "side": quote.side,
            "quantity": quote.quantity,
            "price": quote.price,
            "fee": quote.fee,
            "gas": 0.0,
            "net_output": quote.net_output,
            "instance_id": instance_id,
            "mock": True,  # Remove when real integration done
        }

    def _execute_via_dex(self, quote: Quote, instance_id: str) -> Dict:
        """Execute trade via DEX connector (mock for now)"""
        # TODO: Integrate with DEXConnector
        return {
            "success": True,
            "venue": quote.venue_name,
            "symbol": quote.symbol,
            "side": quote.side,
            "quantity": quote.quantity,
            "price": quote.price,
            "fee": quote.fee,
            "gas": quote.gas,
            "net_output": quote.net_output,
            "instance_id": instance_id,
            "mock": True,  # Remove when real integration done
        }


# Example usage
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    router = ExecutionRouter(max_gas_pct=0.02)

    # Test case 1: Small trade - high gas % - should use CEX
    print("\n=== Test 1: Small trade (high gas %) ===")
    decision = router.get_best_route(
        symbol="WETH/USDC",
        side="buy",
        quantity=0.01,  # Small amount
        chain="Base"
    )
    print(f"Selected: {decision.selected_venue.value}")
    print(f"Reason: {decision.reason}")

    # Test case 2: Large trade - low gas % - compare prices
    print("\n=== Test 2: Large trade (low gas %) ===")
    decision = router.get_best_route(
        symbol="WETH/USDC",
        side="buy",
        quantity=10.0,  # Larger amount
        chain="Base"
    )
    print(f"Selected: {decision.selected_venue.value}")
    print(f"Reason: {decision.reason}")
    print(f"Savings: ${decision.savings:.2f}")

    # Test case 3: Execute via selected venue
    print("\n=== Test 3: Execute trade ===")
    result = router.execute_trade(decision, instance_id="test_001")
    print(f"Execution result: {result}")
