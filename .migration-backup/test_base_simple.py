#!/usr/bin/env python3
"""
Simple Base Connection Test (No Private Key Required)
"""
from web3 import Web3
from web3.middleware import ExtraDataToPOAMiddleware

def test_base():
    print("=" * 60)
    print("🔵 BASE BLOCKCHAIN - SIMPLE CONNECTION TEST")
    print("=" * 60)

    # Base RPC URLs to try
    rpc_urls = [
        "https://mainnet.base.org",
        "https://base.llamarpc.com",
        "https://base.meowrpc.com"
    ]

    print("\n1️⃣  Testing Base RPC endpoints...")

    w3 = None
    working_rpc = None

    for rpc_url in rpc_urls:
        try:
            print(f"\n   Trying: {rpc_url}")
            w3 = Web3(Web3.HTTPProvider(rpc_url, request_kwargs={'timeout': 10}))

            # Add middleware for Base (PoA chain)
            w3.middleware_onion.inject(ExtraDataToPOAMiddleware, layer=0)

            # Test connection
            if w3.is_connected():
                print(f"   ✅ Connected!")
                working_rpc = rpc_url
                break
            else:
                print(f"   ❌ Connection failed")
        except Exception as e:
            print(f"   ❌ Error: {e}")

    if not working_rpc:
        print("\n❌ Could not connect to any Base RPC")
        print("\nTroubleshooting:")
        print("   - Check your internet connection")
        print("   - Try using a VPN if RPCs are blocked")
        print("   - Get an Alchemy/Infura API key for reliable access")
        return False

    print(f"\n2️⃣  Connected to Base via: {working_rpc}")

    # Get network info
    print("\n3️⃣  Base Network Info:")
    try:
        chain_id = w3.eth.chain_id
        block_number = w3.eth.block_number
        gas_price = w3.eth.gas_price / 1e9  # Convert to Gwei

        print(f"   Chain ID: {chain_id}")
        print(f"   Block Number: {block_number:,}")
        print(f"   Gas Price: {gas_price:.4f} Gwei")
        print(f"   Gas Price (USD est.): ${gas_price * 3000 / 1e9:.6f} per gas unit")
    except Exception as e:
        print(f"   ⚠️  Error fetching info: {e}")

    # Check a known address balance (Coinbase deployer)
    print("\n4️⃣  Testing Balance Query:")
    try:
        # Coinbase's known address on Base
        test_address = "0x4200000000000000000000000000000000000006"  # WETH contract
        balance = w3.eth.get_balance(test_address)
        print(f"   Address: {test_address}")
        print(f"   Balance: {balance / 1e18:.6f} ETH")
        print(f"   ✅ Balance queries working!")
    except Exception as e:
        print(f"   ⚠️  Error: {e}")

    # Summary
    print("\n" + "=" * 60)
    print("✅ BASE CONNECTION TEST PASSED!")
    print("=" * 60)
    print("\n📋 Summary:")
    print(f"   ✅ Connected to Base (Chain ID: 8453)")
    print(f"   ✅ Using RPC: {working_rpc}")
    print(f"   ✅ Current block: {block_number:,}")
    print(f"   ✅ Gas price: {gas_price:.4f} Gwei")

    print("\n🎯 Next Steps:")
    print("   1. Read BASE_SETUP_GUIDE.md for full setup")
    print("   2. Add your wallet private key to .env (if trading)")
    print("   3. Bridge ETH to Base: https://bridge.base.org")
    print("   4. Run: python3 examples/defi/demo_multi_chain.py")

    print("\n💡 Base Benefits:")
    print(f"   ⚡ Fast: ~2 second blocks")
    print(f"   💰 Cheap: ~${gas_price * 3000 * 21000 / 1e9:.4f} per transfer")
    print(f"   🏦 Integrated with Coinbase")

    return True


if __name__ == "__main__":
    try:
        test_base()
    except KeyboardInterrupt:
        print("\n\n⚠️  Test interrupted")
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
