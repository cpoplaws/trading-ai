#!/usr/bin/env python3
"""
Quick Test: Connect to Base Blockchain
Run this to verify your Base connection works!
"""
import os
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from blockchain.chain_manager import ChainManager, Chain
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


def test_base_connection():
    """Test connection to Base blockchain."""
    print("=" * 60)
    print("🔵 BASE BLOCKCHAIN CONNECTION TEST")
    print("=" * 60)

    # Initialize chain manager
    print("\n1️⃣  Initializing Chain Manager...")
    manager = ChainManager()

    # Connect to Base
    print("\n2️⃣  Connecting to Base...")
    success = manager.connect(Chain.BASE)

    if not success:
        print("   ❌ Failed to connect to Base")
        print("\n   Troubleshooting:")
        print("   - Check internet connection")
        print("   - Try alternate RPC: export BASE_RPC_URL=https://base.llamarpc.com")
        print("   - Check firewall settings")
        return False

    print("   ✅ Connected to Base!")

    # Get Base configuration
    print("\n3️⃣  Base Configuration:")
    config = manager.get_chain_config(Chain.BASE)
    print(f"   Chain Name: {config.name}")
    print(f"   Chain ID: {config.chain_id}")
    print(f"   Native Token: {config.native_token}")
    print(f"   Explorer: {config.explorer_url}")

    # Get Base connection
    base_connection = manager.get_connection(Chain.BASE)

    # Get current block info
    print("\n4️⃣  Current Block Info:")
    try:
        if hasattr(base_connection, 'w3'):
            block_number = base_connection.w3.eth.block_number
            print(f"   Block Number: {block_number:,}")

            # Get gas price
            gas_price_wei = base_connection.w3.eth.gas_price
            gas_price_gwei = gas_price_wei / 1e9
            print(f"   Gas Price: {gas_price_gwei:.2f} Gwei")

            # Check if connected
            is_connected = base_connection.w3.is_connected()
            print(f"   Connection Status: {'✅ Active' if is_connected else '❌ Inactive'}")
    except Exception as e:
        print(f"   ⚠️  Could not fetch block info: {e}")

    # Test wallet balance (if private key provided)
    print("\n5️⃣  Wallet Check:")
    eth_private_key = os.getenv('ETH_PRIVATE_KEY')

    if eth_private_key and eth_private_key != 'your_ethereum_wallet_private_key_here':
        try:
            from eth_account import Account
            account = Account.from_key(eth_private_key)
            address = account.address

            print(f"   Wallet Address: {address}")

            # Get balance
            balance = manager.get_balance(Chain.BASE, address)
            print(f"   Base Balance: {balance:.6f} ETH")

            if balance == 0:
                print("\n   ⚠️  Your wallet has 0 ETH on Base")
                print("   To get ETH on Base:")
                print("   1. Visit https://bridge.base.org")
                print("   2. Bridge ETH from Ethereum to Base")
                print("   3. Or withdraw ETH from Coinbase directly to Base")
        except Exception as e:
            print(f"   ⚠️  Could not check wallet: {e}")
    else:
        print("   ⚠️  No private key configured")
        print("   Add ETH_PRIVATE_KEY to .env file to check your balance")

    # Summary
    print("\n" + "=" * 60)
    print("✅ BASE CONNECTION TEST COMPLETE!")
    print("=" * 60)
    print("\nYour system is ready to trade on Base blockchain!")
    print("\nNext steps:")
    print("   1. Fund your wallet: https://bridge.base.org")
    print("   2. Run multi-chain demo: python3 examples/defi/demo_multi_chain.py")
    print("   3. Read setup guide: BASE_SETUP_GUIDE.md")
    print("   4. Start trading on Base DEXs!")

    return True


def test_popular_base_tokens():
    """Test querying popular Base tokens."""
    print("\n" + "=" * 60)
    print("🪙  POPULAR BASE TOKENS")
    print("=" * 60)

    manager = ChainManager()
    if not manager.connect(Chain.BASE):
        return

    base_connection = manager.get_connection(Chain.BASE)

    # Popular Base tokens
    tokens = {
        "USDC": "0x833589fCD6eDb6E08f4c7C32D4f71b54bdA02913",
        "WETH": "0x4200000000000000000000000000000000000006",
    }

    print("\nPopular tokens on Base:")
    for name, address in tokens.items():
        print(f"   {name}: {address}")

    print("\nTo check token balances, add your wallet address to .env!")


def main():
    """Run all tests."""
    try:
        # Test 1: Base connection
        success = test_base_connection()

        if success:
            # Test 2: Popular tokens
            test_popular_base_tokens()

    except KeyboardInterrupt:
        print("\n\n⚠️  Test interrupted by user")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
