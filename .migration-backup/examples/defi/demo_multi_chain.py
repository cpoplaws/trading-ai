"""
Demo: Multi-Chain Portfolio Operations
Demonstrates connecting to multiple blockchains and managing a multi-chain portfolio.
"""
import os
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from blockchain.chain_manager import ChainManager, Chain
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def demo_multi_chain_connections():
    """Demonstrate connecting to multiple blockchains."""
    print("=" * 60)
    print("Multi-Chain Connection Demo")
    print("=" * 60)
    
    # Initialize chain manager
    manager = ChainManager()
    
    # List of chains to connect to
    chains_to_connect = [
        Chain.ETHEREUM,
        Chain.POLYGON,
        Chain.ARBITRUM,
        Chain.BSC,
        Chain.AVALANCHE,
        Chain.BASE
    ]
    
    print("\n1. Connecting to multiple chains...")
    results = {}
    for chain in chains_to_connect:
        try:
            success = manager.connect(chain)
            results[chain.value] = success
            status = "✅ Connected" if success else "❌ Failed"
            print(f"   {chain.value.upper()}: {status}")
        except Exception as e:
            print(f"   {chain.value.upper()}: ❌ Error - {e}")
            results[chain.value] = False
    
    # Show connected chains
    print(f"\n2. Connected chains: {', '.join(manager.list_connected_chains())}")
    
    # Get chain configurations
    print("\n3. Chain configurations:")
    for chain in chains_to_connect:
        if manager.is_connected(chain):
            config = manager.get_chain_config(chain)
            print(f"\n   {config.name}:")
            print(f"      Chain ID: {config.chain_id}")
            print(f"      Native Token: {config.native_token}")
            print(f"      Explorer: {config.explorer_url}")
    
    return manager, results


def demo_balance_queries(manager: ChainManager):
    """Demonstrate querying balances across chains."""
    print("\n" + "=" * 60)
    print("Multi-Chain Balance Queries")
    print("=" * 60)
    
    # Example address (Vitalik's address for demo - read-only)
    demo_address = "0xd8dA6BF26964aF9D7eEd9e03E53415D37aA96045"
    
    print(f"\nQuerying balances for address: {demo_address[:10]}...{demo_address[-8:]}")
    print("\nNote: Using a well-known address for demo purposes (read-only)\n")
    
    # Get balances across all connected chains
    try:
        balances = manager.get_all_balances(demo_address)
        
        print("Native token balances:")
        for chain_name, balance in balances.items():
            if balance > 0:
                print(f"   {chain_name}: {balance:.4f}")
            else:
                print(f"   {chain_name}: 0.0000")
                
    except Exception as e:
        logger.error(f"Error querying balances: {e}")
        print(f"   Error: {e}")


def demo_chain_info():
    """Demonstrate getting blockchain information."""
    print("\n" + "=" * 60)
    print("Chain Information")
    print("=" * 60)
    
    manager = ChainManager()
    
    # Connect to Ethereum
    if manager.connect(Chain.ETHEREUM):
        eth_connection = manager.get_connection(Chain.ETHEREUM)
        
        print("\nEthereum Mainnet:")
        print(f"   Block Number: {eth_connection.get_block_number():,}")
        print(f"   Chain ID: {eth_connection.chain_id}")
        print(f"   Connected: {eth_connection.w3.is_connected()}")
        
        # Get gas prices
        gas_prices = eth_connection.get_gas_price()
        print(f"\n   Gas Prices:")
        print(f"      Slow: {gas_prices['slow'] / 1e9:.2f} Gwei")
        print(f"      Standard: {gas_prices['standard'] / 1e9:.2f} Gwei")
        print(f"      Fast: {gas_prices['fast'] / 1e9:.2f} Gwei")
    
    # Connect to Polygon
    if manager.connect(Chain.POLYGON):
        poly_connection = manager.get_connection(Chain.POLYGON)
        
        print("\nPolygon:")
        print(f"   Block Number: {poly_connection.get_block_number():,}")
        print(f"   Chain ID: {poly_connection.chain_id}")
        
        gas_prices = poly_connection.get_gas_price()
        print(f"\n   Gas Prices:")
        print(f"      Slow: {gas_prices['slow'] / 1e9:.2f} Gwei")
        print(f"      Standard: {gas_prices['standard'] / 1e9:.2f} Gwei")
        print(f"      Fast: {gas_prices['fast'] / 1e9:.2f} Gwei")


def demo_token_info(manager: ChainManager):
    """Demonstrate querying token information."""
    print("\n" + "=" * 60)
    print("Token Information")
    print("=" * 60)
    
    # Connect to Ethereum
    if not manager.is_connected(Chain.ETHEREUM):
        manager.connect(Chain.ETHEREUM)
    
    eth_connection = manager.get_connection(Chain.ETHEREUM)
    
    if eth_connection:
        # Query USDC token info
        usdc_address = "0xA0b86991c6218b36c1d19D4a2e9Eb0cE3606eB48"
        
        print(f"\nQuerying USDC token info on Ethereum...")
        token_info = eth_connection.get_token_info(usdc_address)
        
        if token_info:
            print(f"\n   Name: {token_info['name']}")
            print(f"   Symbol: {token_info['symbol']}")
            print(f"   Decimals: {token_info['decimals']}")
            print(f"   Total Supply: {token_info['total_supply'] / 1e6:,.0f}")


def main():
    """Run all demos."""
    print("\n")
    print("╔" + "=" * 58 + "╗")
    print("║" + " " * 8 + "TRADING-AI: MULTI-CHAIN DEMO" + " " * 22 + "║")
    print("╚" + "=" * 58 + "╝")
    
    try:
        # Demo 1: Multi-chain connections
        manager, results = demo_multi_chain_connections()
        
        # Check if any chains connected successfully
        if not any(results.values()):
            print("\n⚠️  Warning: No chains connected successfully.")
            print("   This is expected if RPC endpoints are not configured.")
            print("   Update .env file with RPC URLs to enable connections.")
            return
        
        # Demo 2: Balance queries
        demo_balance_queries(manager)
        
        # Demo 3: Chain information
        demo_chain_info()
        
        # Demo 4: Token information
        demo_token_info(manager)
        
        print("\n" + "=" * 60)
        print("Demo Summary")
        print("=" * 60)
        print("\nThis demo showed:")
        print("   ✅ Multi-chain connection management")
        print("   ✅ Balance queries across chains")
        print("   ✅ Gas price estimation")
        print("   ✅ Token information retrieval")
        print("\nNext steps:")
        print("   • Configure RPC endpoints in .env")
        print("   • Add wallet private keys (for trading)")
        print("   • Explore DEX aggregation (demo_dex_aggregator.py)")
        print("   • Try funding rate arbitrage (demo_funding_arb.py)")
        
    except Exception as e:
        logger.error(f"Demo error: {e}")
        print(f"\n❌ Error: {e}")
        print("\nMake sure you have:")
        print("   1. Installed requirements: pip install -r requirements.txt")
        print("   2. Configured .env file with RPC URLs")
    
    print("\n" + "=" * 60)
    print("✅ Multi-chain demo completed!")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
