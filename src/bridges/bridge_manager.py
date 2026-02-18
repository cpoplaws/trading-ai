"""
Bridge Manager - Cross-chain asset transfers

Supports:
- Multiple bridge protocols (Hop, Across, Stargate)
- Route optimization (best fees, fastest time)
- Transfer tracking and confirmation
- Balance rebalancing across chains
"""

import logging
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
from datetime import datetime

logger = logging.getLogger(__name__)


class BridgeProtocol(Enum):
    """Supported bridge protocols"""
    HOP = "hop"
    ACROSS = "across"
    STARGATE = "stargate"
    NATIVE = "native"  # Native L2 bridges


class TransferStatus(Enum):
    """Transfer status"""
    PENDING = "pending"
    SUBMITTED = "submitted"
    CONFIRMING = "confirming"
    CONFIRMED = "confirmed"
    FAILED = "failed"


@dataclass
class BridgeQuote:
    """Bridge transfer quote"""
    protocol: BridgeProtocol
    from_chain: str
    to_chain: str
    token: str
    amount_in: float
    amount_out: float  # After fees
    fee_usd: float
    gas_cost_usd: float
    estimated_time_minutes: int
    route: List[str]  # Chain path for multi-hop


@dataclass
class BridgeTransfer:
    """Bridge transfer record"""
    transfer_id: str
    protocol: BridgeProtocol
    from_chain: str
    to_chain: str
    token: str
    amount: float
    fee_usd: float
    status: TransferStatus
    tx_hash_source: Optional[str] = None
    tx_hash_destination: Optional[str] = None
    initiated_at: datetime = None
    confirmed_at: Optional[datetime] = None
    error_message: Optional[str] = None


class BridgeManager:
    """
    Manages cross-chain asset transfers.

    Features:
    - Multi-protocol support
    - Route optimization
    - Transfer tracking
    - Balance rebalancing
    """

    # Bridge protocol configurations
    BRIDGE_CONFIGS = {
        BridgeProtocol.HOP: {
            "supported_chains": ["base", "arbitrum", "optimism", "polygon", "ethereum"],
            "supported_tokens": ["ETH", "USDC", "USDT", "DAI"],
            "avg_time_minutes": 20,
            "base_fee_pct": 0.001,  # 0.1%
        },
        BridgeProtocol.ACROSS: {
            "supported_chains": ["base", "arbitrum", "optimism", "polygon", "ethereum"],
            "supported_tokens": ["ETH", "USDC", "USDT", "WBTC"],
            "avg_time_minutes": 15,
            "base_fee_pct": 0.0008,  # 0.08%
        },
        BridgeProtocol.STARGATE: {
            "supported_chains": ["arbitrum", "optimism", "polygon", "ethereum"],
            "supported_tokens": ["USDC", "USDT"],
            "avg_time_minutes": 25,
            "base_fee_pct": 0.0006,  # 0.06%
        },
    }

    def __init__(
        self,
        max_fee_pct: float = 0.005,  # 0.5% max fee
        max_time_minutes: int = 60,  # 1 hour max
    ):
        """
        Initialize bridge manager.

        Args:
            max_fee_pct: Maximum acceptable fee percentage
            max_time_minutes: Maximum acceptable transfer time
        """
        self.max_fee_pct = max_fee_pct
        self.max_time_minutes = max_time_minutes

        # Track active transfers
        self.transfers: Dict[str, BridgeTransfer] = {}
        self.transfer_counter = 0

        logger.info(
            f"Bridge Manager initialized | "
            f"Max fee: {max_fee_pct*100:.2f}% | "
            f"Max time: {max_time_minutes}min"
        )

    def get_quote(
        self,
        from_chain: str,
        to_chain: str,
        token: str,
        amount: float,
        protocol: Optional[BridgeProtocol] = None
    ) -> Optional[BridgeQuote]:
        """
        Get bridge quote for transfer.

        Args:
            from_chain: Source chain
            to_chain: Destination chain
            token: Token to bridge
            amount: Amount to bridge
            protocol: Specific protocol (optional)

        Returns:
            Bridge quote or None
        """
        # If protocol specified, get quote from that protocol only
        if protocol:
            return self._get_protocol_quote(protocol, from_chain, to_chain, token, amount)

        # Otherwise, find best quote across all protocols
        best_quote = None
        best_score = float('-inf')

        for proto in BridgeProtocol:
            if proto == BridgeProtocol.NATIVE:
                continue  # Skip native for now

            quote = self._get_protocol_quote(proto, from_chain, to_chain, token, amount)
            if quote is None:
                continue

            # Score: maximize output, minimize time
            score = quote.amount_out - (quote.estimated_time_minutes / 100.0)

            if score > best_score:
                best_score = score
                best_quote = quote

        if best_quote:
            logger.info(
                f"Best bridge quote: {best_quote.protocol.value} | "
                f"{from_chain} → {to_chain} | "
                f"Amount: {amount} {token} → {best_quote.amount_out:.4f} | "
                f"Fee: ${best_quote.fee_usd:.2f}"
            )

        return best_quote

    def _get_protocol_quote(
        self,
        protocol: BridgeProtocol,
        from_chain: str,
        to_chain: str,
        token: str,
        amount: float
    ) -> Optional[BridgeQuote]:
        """Get quote from specific protocol."""
        config = self.BRIDGE_CONFIGS.get(protocol)
        if not config:
            return None

        # Check if route is supported
        if from_chain not in config["supported_chains"]:
            return None
        if to_chain not in config["supported_chains"]:
            return None
        if token not in config["supported_tokens"]:
            return None

        # Calculate fees (mock for now - real implementation would call API)
        fee_pct = config["base_fee_pct"]
        fee_amount = amount * fee_pct

        # Mock token prices
        token_prices = {
            "ETH": 3000.0,
            "USDC": 1.0,
            "USDT": 1.0,
            "DAI": 1.0,
            "WBTC": 64000.0,
        }
        token_price = token_prices.get(token, 100.0)

        fee_usd = fee_amount * token_price

        # Estimate gas cost
        gas_cost_usd = self._estimate_gas_cost(from_chain, to_chain)

        # Calculate output amount
        amount_out = amount - fee_amount

        # Check if within limits
        total_fee_pct = (fee_amount + gas_cost_usd / token_price) / amount
        if total_fee_pct > self.max_fee_pct:
            logger.debug(f"Bridge fee too high: {total_fee_pct*100:.2f}%")
            return None

        estimated_time = config["avg_time_minutes"]
        if estimated_time > self.max_time_minutes:
            return None

        return BridgeQuote(
            protocol=protocol,
            from_chain=from_chain,
            to_chain=to_chain,
            token=token,
            amount_in=amount,
            amount_out=amount_out,
            fee_usd=fee_usd,
            gas_cost_usd=gas_cost_usd,
            estimated_time_minutes=estimated_time,
            route=[from_chain, to_chain]
        )

    def _estimate_gas_cost(self, from_chain: str, to_chain: str) -> float:
        """Estimate gas cost for bridge transfer (USD)."""
        # Mock gas costs
        gas_costs = {
            "base": 0.50,
            "arbitrum": 0.30,
            "optimism": 0.50,
            "polygon": 0.10,
            "ethereum": 15.0,
        }

        source_gas = gas_costs.get(from_chain, 1.0)
        # Destination gas is paid by relayer, not included
        return source_gas

    def initiate_transfer(
        self,
        quote: BridgeQuote,
        wallet_manager = None
    ) -> BridgeTransfer:
        """
        Initiate bridge transfer.

        Args:
            quote: Bridge quote
            wallet_manager: Wallet manager for signing

        Returns:
            Bridge transfer record
        """
        self.transfer_counter += 1
        transfer_id = f"bridge_{self.transfer_counter:06d}"

        transfer = BridgeTransfer(
            transfer_id=transfer_id,
            protocol=quote.protocol,
            from_chain=quote.from_chain,
            to_chain=quote.to_chain,
            token=quote.token,
            amount=quote.amount_in,
            fee_usd=quote.fee_usd,
            status=TransferStatus.PENDING,
            initiated_at=datetime.now()
        )

        # TODO: Real implementation would:
        # 1. Build bridge transaction
        # 2. Sign with wallet_manager
        # 3. Submit to bridge contract
        # 4. Monitor for confirmation

        # Mock execution
        transfer.status = TransferStatus.SUBMITTED
        transfer.tx_hash_source = f"0xmock_source_{transfer_id}"

        self.transfers[transfer_id] = transfer

        logger.info(
            f"Bridge transfer initiated: {transfer_id} | "
            f"{quote.from_chain} → {quote.to_chain} | "
            f"{quote.amount_in} {quote.token}"
        )

        return transfer

    def get_transfer_status(self, transfer_id: str) -> Optional[BridgeTransfer]:
        """Get status of bridge transfer."""
        return self.transfers.get(transfer_id)

    def wait_for_confirmation(
        self,
        transfer_id: str,
        timeout_minutes: int = 60
    ) -> bool:
        """
        Wait for transfer confirmation.

        Args:
            transfer_id: Transfer ID
            timeout_minutes: Max wait time

        Returns:
            True if confirmed, False if timeout/failed
        """
        transfer = self.transfers.get(transfer_id)
        if not transfer:
            return False

        # TODO: Real implementation would poll bridge API
        # Mock: immediately confirm
        transfer.status = TransferStatus.CONFIRMED
        transfer.confirmed_at = datetime.now()
        transfer.tx_hash_destination = f"0xmock_dest_{transfer_id}"

        logger.info(f"Bridge transfer confirmed: {transfer_id}")
        return True

    def suggest_rebalance(
        self,
        balances: Dict[str, float],  # chain -> balance
        target_distribution: Dict[str, float] = None  # chain -> target %
    ) -> List[BridgeQuote]:
        """
        Suggest rebalancing transfers.

        Args:
            balances: Current balances per chain
            target_distribution: Target % per chain (default: equal)

        Returns:
            List of suggested transfers
        """
        if not balances:
            return []

        total_balance = sum(balances.values())
        if total_balance == 0:
            return []

        # Default to equal distribution
        if target_distribution is None:
            num_chains = len(balances)
            target_distribution = {chain: 1.0 / num_chains for chain in balances.keys()}

        suggestions = []

        # Find imbalances
        for chain, balance in balances.items():
            current_pct = balance / total_balance
            target_pct = target_distribution.get(chain, 0)

            diff = current_pct - target_pct

            # If significantly over target, suggest transfer out
            if diff > 0.1:  # 10% threshold
                amount_to_move = total_balance * (diff - 0.05)  # Leave 5% buffer

                # Find chain that needs more
                for dest_chain, dest_balance in balances.items():
                    if dest_chain == chain:
                        continue

                    dest_current_pct = dest_balance / total_balance
                    dest_target_pct = target_distribution.get(dest_chain, 0)

                    if dest_current_pct < dest_target_pct:
                        # Get quote for transfer
                        quote = self.get_quote(
                            chain, dest_chain, "USDC", amount_to_move
                        )
                        if quote:
                            suggestions.append(quote)
                            break

        if suggestions:
            logger.info(f"Rebalance suggestions: {len(suggestions)} transfers")
            for quote in suggestions:
                logger.info(
                    f"   {quote.from_chain} → {quote.to_chain}: "
                    f"{quote.amount_in:.2f} USDC"
                )

        return suggestions

    def get_supported_routes(self) -> List[Tuple[str, str, List[str]]]:
        """
        Get all supported bridge routes.

        Returns:
            List of (from_chain, to_chain, [protocols])
        """
        routes = []

        # Collect all routes across protocols
        route_map = {}

        for protocol, config in self.BRIDGE_CONFIGS.items():
            chains = config["supported_chains"]

            for from_chain in chains:
                for to_chain in chains:
                    if from_chain != to_chain:
                        key = (from_chain, to_chain)
                        if key not in route_map:
                            route_map[key] = []
                        route_map[key].append(protocol.value)

        # Convert to list
        for (from_chain, to_chain), protocols in route_map.items():
            routes.append((from_chain, to_chain, protocols))

        return routes


# Example usage
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    print("\n" + "="*70)
    print("BRIDGE MANAGER TEST")
    print("="*70)

    manager = BridgeManager()

    # Test 1: Get quote
    print("\n--- Test 1: Get Bridge Quote ---")
    quote = manager.get_quote("base", "arbitrum", "USDC", 1000.0)

    if quote:
        print(f"Protocol: {quote.protocol.value}")
        print(f"Route: {' → '.join(quote.route)}")
        print(f"Input: {quote.amount_in} USDC")
        print(f"Output: {quote.amount_out:.4f} USDC")
        print(f"Fee: ${quote.fee_usd:.2f}")
        print(f"Gas: ${quote.gas_cost_usd:.2f}")
        print(f"Time: ~{quote.estimated_time_minutes} minutes")

    # Test 2: Compare protocols
    print("\n--- Test 2: Compare Bridge Protocols ---")
    protocols_to_test = [BridgeProtocol.HOP, BridgeProtocol.ACROSS, BridgeProtocol.STARGATE]

    for protocol in protocols_to_test:
        quote = manager.get_quote("base", "arbitrum", "USDC", 1000.0, protocol=protocol)
        if quote:
            net_output = quote.amount_out
            total_cost = quote.fee_usd + quote.gas_cost_usd
            print(f"{protocol.value:10s}: Output: {net_output:.4f} | Cost: ${total_cost:.2f} | Time: {quote.estimated_time_minutes}min")
        else:
            print(f"{protocol.value:10s}: Not supported")

    # Test 3: Initiate transfer
    print("\n--- Test 3: Initiate Transfer ---")
    if quote:
        transfer = manager.initiate_transfer(quote)
        print(f"Transfer ID: {transfer.transfer_id}")
        print(f"Status: {transfer.status.value}")
        print(f"TX Hash: {transfer.tx_hash_source}")

        # Wait for confirmation
        confirmed = manager.wait_for_confirmation(transfer.transfer_id)
        print(f"Confirmed: {confirmed}")

    # Test 4: Rebalancing suggestions
    print("\n--- Test 4: Rebalancing Suggestions ---")
    balances = {
        "base": 5000.0,      # 50%
        "arbitrum": 2000.0,  # 20%
        "optimism": 3000.0,  # 30%
    }

    print(f"Current balances:")
    for chain, balance in balances.items():
        print(f"   {chain}: ${balance:,.2f}")

    suggestions = manager.suggest_rebalance(balances)
    if suggestions:
        print(f"\nSuggested transfers:")
        for s in suggestions:
            print(f"   {s.from_chain} → {s.to_chain}: {s.amount_in:.2f} USDC")
    else:
        print("   No rebalancing needed")

    # Test 5: Supported routes
    print("\n--- Test 5: Supported Routes ---")
    routes = manager.get_supported_routes()
    print(f"Total routes: {len(routes)}")
    print("\nSample routes:")
    for from_chain, to_chain, protocols in routes[:5]:
        print(f"   {from_chain} → {to_chain}: {', '.join(protocols)}")

    print("\n" + "="*70)
    print("✅ Bridge Manager ready!")
    print("="*70)
