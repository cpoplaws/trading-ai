"""
DeFi Trading Demo - PancakeSwap integration with AI signals.

This demonstrates how to:
1. Connect to BSC and PancakeSwap
2. Fetch DeFi token price data
3. Apply AI models to DeFi tokens
4. Execute trades on PancakeSwap
"""
import sys
import os
from datetime import datetime, timedelta
import pandas as pd
import numpy as np

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.blockchain.bsc_interface import BSCInterface
from src.defi.pancakeswap_trader import PancakeSwapTrader
from web3 import Web3
from src.feature_engineering.feature_generator import FeatureGenerator
from src.modeling.train_model import train_model
from src.strategy.simple_strategy import generate_signals

def demo_bsc_connection():
    """Test BSC connection and basic functionality."""
    print("🔗 Testing BSC Connection...")
    
    try:
        # Connect to BSC testnet (safe for testing)
        bsc = BSCInterface(testnet=True)
        
        print(f"✅ Connected to BSC testnet")
        print(f"   Latest block: {bsc.w3.eth.block_number}")
        print(f"   Gas price: {bsc.estimate_gas_price() / 1e9:.2f} Gwei")
        
        # Test token info
        print(f"\n📊 Token Information:")
        for symbol, address in list(bsc.TOKENS.items())[:3]:
            if address != '0x0000000000000000000000000000000000000000':
                info = bsc.get_token_info(address)
                if info:
                    print(f"   {symbol}: {info['name']} ({info['symbol']}) - {info['decimals']} decimals")
        
        return bsc
        
    except Exception as e:
        print(f"❌ BSC connection failed: {e}")
        return None

def demo_pancakeswap_integration(bsc):
    """Test PancakeSwap integration."""
    print(f"\n🥞 Testing PancakeSwap Integration...")
    
    try:
        pancake = PancakeSwapTrader(bsc)
        
        # Get token prices
        print(f"\n💰 Current Token Prices (in BNB):")
        tokens_to_check = ['BUSD', 'CAKE', 'ETH']
        
        for symbol in tokens_to_check:
            if symbol in bsc.TOKENS:
                price = pancake.get_token_price(bsc.TOKENS[symbol])
                if price:
                    print(f"   {symbol}: {price:.6f} BNB")
        
        # Get trading opportunities
        print(f"\n🎯 Trading Opportunities:")
        opportunities = pancake.get_trading_opportunities(min_liquidity=1000)  # Lower for testnet
        
        if opportunities:
            for opp in opportunities[:5]:
                print(f"   {opp['symbol']}: {opp['price_bnb']:.6f} BNB, Liquidity: {opp['liquidity_bnb']:.2f} BNB")
        else:
            print(f"   No opportunities found (testnet has limited liquidity)")
        
        return pancake, opportunities
        
    except Exception as e:
        print(f"❌ PancakeSwap integration failed: {e}")
        return None, []

def simulate_defi_price_data(symbol: str, days: int = 100) -> pd.DataFrame:
    """
    Simulate DeFi token price data for AI model training.
    In production, this would fetch real price data from DEX APIs.
    """
    print(f"\n📈 Simulating {symbol} price data...")
    
    # Generate realistic price data
    np.random.seed(42)
    dates = pd.date_range(end=datetime.now(), periods=days, freq='1H')
    
    # Start with base price and add random walk
    base_price = 1.0  # Starting price in BNB
    price_changes = np.random.normal(0, 0.02, len(dates))  # 2% hourly volatility
    prices = [base_price]
    
    for change in price_changes[1:]:
        new_price = prices[-1] * (1 + change)
        prices.append(max(new_price, 0.01))  # Prevent negative prices
    
    # Create OHLCV data
    df = pd.DataFrame({
        'datetime': dates,
        'open': prices,
        'high': [p * (1 + abs(np.random.normal(0, 0.01))) for p in prices],
        'low': [p * (1 - abs(np.random.normal(0, 0.01))) for p in prices],
        'close': prices,
        'volume': np.random.randint(1000, 10000, len(dates))
    })
    
    df.set_index('datetime', inplace=True)
    print(f"   Generated {len(df)} data points from {df.index[0]} to {df.index[-1]}")
    
    return df

def demo_ai_defi_trading():
    """Demonstrate AI trading on DeFi tokens."""
    print(f"\n🤖 AI DeFi Trading Demo...")
    
    try:
        # Simulate CAKE token data
        cake_data = simulate_defi_price_data('CAKE', days=30)
        
        # Apply feature engineering
        print(f"   Generating technical features...")
        fg = FeatureGenerator(cake_data)
        features_df = fg.generate_features()
        
        # Train AI model on DeFi data
        print(f"   Training AI model on DeFi data...")
        model_path = './models/cake_defi_model.joblib'
        features_path = './models/cake_defi_features.joblib'
        
        success = train_model(
            features_df, 
            save_path=model_path
        )
        
        if success:
            print(f"   ✅ Model trained successfully")
            
            # Generate trading signals
            print(f"   Generating DeFi trading signals...")
            data_path = './data/processed/CAKE_defi.csv'
            features_df.to_csv(data_path)
            
            signals_generated = generate_signals(
                model_path=model_path,
                data_path=data_path,
                save_path='./signals/'
            )
            
            if signals_generated:
                # Load and display signals
                signals_df = pd.read_csv('./signals/CAKE_defi_signals.csv')
                recent_signals = signals_df.tail(10)
                
                print(f"   📊 Recent CAKE DeFi Trading Signals:")
                for _, signal in recent_signals.iterrows():
                    print(f"      {signal.get('Timestamp', 'N/A')}: {signal['Signal']} "
                          f"(Confidence: {signal['Confidence']:.3f})")
                
                return True
        
        return False
        
    except Exception as e:
        print(f"❌ AI DeFi trading demo failed: {e}")
        return False

def demo_trading_simulation(bsc, pancake):
    """Simulate DeFi trading execution (without real transactions)."""
    print(f"\n💼 DeFi Trading Simulation...")
    
    try:
        # Check if we have a wallet configured (we don't for this demo)
        if not bsc.address:
            print(f"   📝 Simulating trades (no wallet configured)")
            
            # Simulate trading scenario
            print(f"   \n🎯 Simulated Trade Execution:")
            print(f"      Action: BUY CAKE with 0.1 BNB")
            print(f"      Slippage: 0.5%")
            print(f"      Gas Estimate: ~0.003 BNB")
            
            # Calculate expected output
            cake_price = pancake.get_token_price(bsc.TOKENS.get('CAKE', ''))
            if cake_price:
                expected_cake = 0.1 / cake_price
                print(f"      Expected Output: {expected_cake:.4f} CAKE")
                
                # Simulate successful trade
                print(f"      Status: ✅ Trade Executed (Simulated)")
                print(f"      TX Hash: 0x1234567890abcdef... (Simulated)")
            
        else:
            print(f"   💰 Wallet Address: {bsc.address}")
            bnb_balance = bsc.get_balance()
            print(f"   💳 BNB Balance: {bnb_balance:.6f} BNB")
            
            # For safety, we won't execute real trades in demo
            print(f"   ⚠️  Real trading disabled in demo mode")
        
        return True
        
    except Exception as e:
        print(f"❌ Trading simulation failed: {e}")
        return False

def main():
    """Main demo function."""
    print("🚀 DeFi Trading AI Demo - PancakeSwap Integration")
    print("=" * 60)
    
    # Install dependencies reminder
    print("\n📦 Make sure to install DeFi dependencies:")
    print("   pip install web3 eth-account eth-keys eth-utils")
    
    # Test BSC connection
    bsc = demo_bsc_connection()
    if not bsc:
        print("\n❌ Cannot proceed without BSC connection")
        return
    
    # Test PancakeSwap integration
    pancake, opportunities = demo_pancakeswap_integration(bsc)
    if not pancake:
        print("\n❌ Cannot proceed without PancakeSwap integration")
        return
    
    # Demo AI trading on DeFi data
    ai_success = demo_ai_defi_trading()
    
    # Demo trading simulation
    trading_success = demo_trading_simulation(bsc, pancake)
    
    # Summary
    print(f"\n" + "=" * 60)
    print(f"🎉 DeFi Integration Demo Summary:")
    print(f"   BSC Connection: ✅")
    print(f"   PancakeSwap Integration: ✅") 
    print(f"   AI Model Training: {'✅' if ai_success else '❌'}")
    print(f"   Trading Simulation: {'✅' if trading_success else '❌'}")
    
    print(f"\n🔥 Next Steps:")
    print(f"   1. Add BSC_PRIVATE_KEY to .env for live trading")
    print(f"   2. Start with BSC testnet and test BNB")
    print(f"   3. Integrate real-time price feeds")
    print(f"   4. Implement advanced DeFi strategies")
    print(f"   5. Add yield farming automation")
    
    print(f"\n⚠️  IMPORTANT: This is testnet demo. Use real funds responsibly!")

if __name__ == "__main__":
    main()