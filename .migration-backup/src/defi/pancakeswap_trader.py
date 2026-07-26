"""
PancakeSwap DEX trading interface for automated DeFi trading.
"""
import os
import json
from web3 import Web3
from typing import Dict, List, Optional, Tuple, Union
import logging
from datetime import datetime, timedelta
from decimal import Decimal
import requests
import time

from ..blockchain.bsc_interface import BSCInterface

# Set up logging
logger = logging.getLogger(__name__)

class PancakeSwapTrader:
    """
    PancakeSwap DEX trading interface with AI integration.
    """
    
    # PancakeSwap V2 Contract Addresses (BSC)
    PANCAKE_ROUTER_V2 = "0x10ED43C718714eb63d5aA57B78B54704E256024E"  # Mainnet
    PANCAKE_ROUTER_V2_TESTNET = "0xD99D1c33F9fC3444f8101754aBC46c52416550D1"  # Testnet
    
    PANCAKE_FACTORY_V2 = "0xcA143Ce32Fe78f1f7019d7d551a6402fC5350c73"  # Mainnet
    PANCAKE_FACTORY_V2_TESTNET = "0x6725F303b657a9451d8BA641348b6761A6CC7a17"  # Testnet
    
    WBNB = "0xbb4CdB9CBd36B01bD1cBaEBF2De08d9173bc095c"  # Wrapped BNB
   
    def __init__(self, bsc_interface: BSCInterface):
        """
        Initialize PancakeSwap trader.
        
        Args:
            bsc_interface: BSC Web3 interface instance
        """
        self.bsc = bsc_interface
        self.testnet = bsc_interface.testnet
        
        # Set contract addresses based on network
        if self.testnet:
            self.router_address = self.PANCAKE_ROUTER_V2_TESTNET
            self.factory_address = self.PANCAKE_FACTORY_V2_TESTNET
        else:
            self.router_address = self.PANCAKE_ROUTER_V2
            self.factory_address = self.PANCAKE_FACTORY_V2
        
        # Initialize router contract
        self.router_contract = self._get_router_contract()
        
        logger.info(f"PancakeSwap trader initialized on {'testnet' if self.testnet else 'mainnet'}")
    
    def get_token_price(self, token_address: str, reference_token: str = None) -> Optional[float]:
        """
        Get token price from PancakeSwap liquidity pool.
        
        Args:
            token_address: Token contract address
            reference_token: Reference token address (WBNB if None)
            
        Returns:
            Token price in reference token units
        """
        try:
            if reference_token is None:
                reference_token = self.WBNB
            
            # Get token pair reserves
            amounts = self.router_contract.functions.getAmountsOut(
                int(1e18),  # 1 token with 18 decimals
                [self.bsc.w3.toChecksumAddress(token_address), self.bsc.w3.toChecksumAddress(reference_token)]
            ).call()
            
            # Calculate price
            price = amounts[1] / 1e18  # Assuming 18 decimals for reference token
            return price
            
        except Exception as e:
            logger.error(f"Error getting token price: {e}")
            return None
    
    def get_pair_info(self, token_a: str, token_b: str) -> Optional[Dict]:
        """
        Get liquidity pair information.
        
        Args:
            token_a: First token address
            token_b: Second token address
            
        Returns:
            Pair information including reserves and liquidity
        """
        try:
            factory_contract = self._get_factory_contract()
            pair_address = factory_contract.functions.getPair(
                Web3.toChecksumAddress(token_a),
                Web3.toChecksumAddress(token_b)
            ).call()
            
            if pair_address == "0x0000000000000000000000000000000000000000":
                return None  # Pair doesn't exist
            
            # Get pair contract
            pair_contract = self._get_pair_contract(pair_address)
            reserves = pair_contract.functions.getReserves().call()
            
            return {
                'pair_address': pair_address,
                'reserve0': reserves[0],
                'reserve1': reserves[1],
                'last_update': reserves[2],
                'token0': pair_contract.functions.token0().call(),
                'token1': pair_contract.functions.token1().call()
            }
            
        except Exception as e:
            logger.error(f"Error getting pair info: {e}")
            return None
    
    def calculate_swap_output(self, amount_in: float, token_in: str, token_out: str) -> Optional[float]:
        """
        Calculate expected output for a token swap.
        
        Args:
            amount_in: Input amount
            token_in: Input token address
            token_out: Output token address
            
        Returns:
            Expected output amount
        """
        try:
            # Get token decimals
            token_in_contract = self.bsc._get_token_contract(token_in)
            decimals_in = token_in_contract.functions.decimals().call()
            
            token_out_contract = self.bsc._get_token_contract(token_out)
            decimals_out = token_out_contract.functions.decimals().call()
            
            # Convert amount to wei
            amount_in_wei = int(amount_in * (10 ** decimals_in))
            
            # Get amounts out
            amounts = self.router_contract.functions.getAmountsOut(
                amount_in_wei,
                [Web3.toChecksumAddress(token_in), Web3.toChecksumAddress(token_out)]
            ).call()
            
            # Convert back to token units
            amount_out = amounts[1] / (10 ** decimals_out)
            return amount_out
            
        except Exception as e:
            logger.error(f"Error calculating swap output: {e}")
            return None
    
    def execute_swap(self, 
                    token_in: str, 
                    token_out: str, 
                    amount_in: float, 
                    min_amount_out: Optional[float] = None,
                    slippage_percent: float = 0.5,
                    deadline_minutes: int = 20) -> Optional[str]:
        """
        Execute a token swap on PancakeSwap.
        
        Args:
            token_in: Input token address
            token_out: Output token address  
            amount_in: Amount to swap
            min_amount_out: Minimum acceptable output (calculated if None)
            slippage_percent: Slippage tolerance percentage
            deadline_minutes: Transaction deadline in minutes
            
        Returns:
            Transaction hash if successful
        """
        if not self.bsc.account:
            raise ValueError("No wallet configured for trading")
        
        try:
            # Get token contracts and decimals
            token_in_contract = self.bsc._get_token_contract(token_in)
            token_out_contract = self.bsc._get_token_contract(token_out)
            
            decimals_in = token_in_contract.functions.decimals().call()
            decimals_out = token_out_contract.functions.decimals().call()
            
            # Convert amounts to wei
            amount_in_wei = int(amount_in * (10 ** decimals_in))
            
            # Calculate minimum output if not provided
            if min_amount_out is None:
                expected_out = self.calculate_swap_output(amount_in, token_in, token_out)
                if expected_out is None:
                    raise ValueError("Could not calculate expected output")
                min_amount_out = expected_out * (1 - slippage_percent / 100)
            
            min_amount_out_wei = int(min_amount_out * (10 ** decimals_out))
            
            # Set deadline
            deadline = int(time.time()) + (deadline_minutes * 60)
            
            # Build swap path
            path = [Web3.toChecksumAddress(token_in), Web3.toChecksumAddress(token_out)]
            
            # Check if we need to approve tokens first
            if token_in.lower() != self.bsc.TOKENS['BNB'].lower():
                allowance = token_in_contract.functions.allowance(
                    self.bsc.address, self.router_address
                ).call()
                
                if allowance < amount_in_wei:
                    logger.info("Approving token for swap...")
                    approve_tx = self.bsc.approve_token(token_in, self.router_address, amount_in * 2)
                    if approve_tx:
                        receipt = self.bsc.wait_for_transaction(approve_tx)
                        if not receipt:
                            raise ValueError("Token approval failed")
            
            # Build swap transaction
            nonce = self.bsc.w3.eth.get_transaction_count(self.bsc.address)
            gas_price = self.bsc.estimate_gas_price()
            
            if token_in.lower() == self.bsc.TOKENS['BNB'].lower():
                # Swapping BNB to token
                transaction = self.router_contract.functions.swapExactETHForTokens(
                    min_amount_out_wei,
                    path,
                    self.bsc.address,
                    deadline
                ).buildTransaction({
                    'chainId': 97 if self.testnet else 56,
                    'gas': 300000,
                    'gasPrice': gas_price,
                    'nonce': nonce,
                    'value': amount_in_wei
                })
            elif token_out.lower() == self.bsc.TOKENS['BNB'].lower():
                # Swapping token to BNB
                transaction = self.router_contract.functions.swapExactTokensForETH(
                    amount_in_wei,
                    min_amount_out_wei,
                    path,
                    self.bsc.address,
                    deadline
                ).buildTransaction({
                    'chainId': 97 if self.testnet else 56,
                    'gas': 300000,
                    'gasPrice': gas_price,
                    'nonce': nonce
                })
            else:
                # Token to token swap
                transaction = self.router_contract.functions.swapExactTokensForTokens(
                    amount_in_wei,
                    min_amount_out_wei,
                    path,
                    self.bsc.address,
                    deadline
                ).buildTransaction({
                    'chainId': 97 if self.testnet else 56,
                    'gas': 300000,
                    'gasPrice': gas_price,
                    'nonce': nonce
                })
            
            # Sign and send transaction
            signed_txn = self.bsc.w3.eth.account.sign_transaction(transaction, self.bsc.private_key)
            tx_hash = self.bsc.w3.eth.send_raw_transaction(signed_txn.rawTransaction)
            
            logger.info(f"Swap transaction sent: {tx_hash.hex()}")
            return tx_hash.hex()
            
        except Exception as e:
            logger.error(f"Error executing swap: {e}")
            return None
    
    def get_trading_opportunities(self, min_liquidity: float = 10000) -> List[Dict]:
        """
        Scan for trading opportunities based on AI signals.
        
        Args:
            min_liquidity: Minimum liquidity required for trading
            
        Returns:
            List of trading opportunities
        """
        opportunities = []
        
        try:
            # Popular BSC tokens to analyze
            tokens_to_check = [
                ('BUSD', self.bsc.TOKENS['BUSD']),
                ('USDT', self.bsc.TOKENS['USDT']),
                ('CAKE', self.bsc.TOKENS['CAKE']),
                ('ETH', self.bsc.TOKENS['ETH']),
                ('ADA', self.bsc.TOKENS['ADA']),
                ('DOT', self.bsc.TOKENS['DOT'])
            ]
            
            for symbol, address in tokens_to_check:
                try:
                    # Get pair info with WBNB
                    pair_info = self.get_pair_info(address, self.WBNB)
                    if not pair_info:
                        continue
                    
                    # Calculate liquidity in USD (approximate)
                    bnb_reserve = pair_info['reserve0'] if pair_info['token0'].lower() == self.WBNB.lower() else pair_info['reserve1']
                    liquidity_bnb = bnb_reserve / 1e18
                    
                    # Skip if liquidity too low
                    if liquidity_bnb * 300 < min_liquidity:  # Assume BNB ~$300
                        continue
                    
                    # Get current price
                    price = self.get_token_price(address)
                    if not price:
                        continue
                    
                    opportunities.append({
                        'symbol': symbol,
                        'address': address,
                        'price_bnb': price,
                        'liquidity_bnb': liquidity_bnb,
                        'pair_address': pair_info['pair_address'],
                        'last_update': datetime.fromtimestamp(pair_info['last_update'])
                    })
                    
                except Exception as e:
                    logger.warning(f"Error checking {symbol}: {e}")
                    continue
            
            return opportunities
            
        except Exception as e:
            logger.error(f"Error getting trading opportunities: {e}")
            return []
    
    def _get_router_contract(self):
        """Get PancakeSwap router contract instance."""
        router_abi = [
            {
                "inputs": [
                    {"internalType": "uint256", "name": "amountIn", "type": "uint256"},
                    {"internalType": "address[]", "name": "path", "type": "address[]"}
                ],
                "name": "getAmountsOut",
                "outputs": [{"internalType": "uint256[]", "name": "amounts", "type": "uint256[]"}],
                "stateMutability": "view",
                "type": "function"
            },
            {
                "inputs": [
                    {"internalType": "uint256", "name": "amountOutMin", "type": "uint256"},
                    {"internalType": "address[]", "name": "path", "type": "address[]"},
                    {"internalType": "address", "name": "to", "type": "address"},
                    {"internalType": "uint256", "name": "deadline", "type": "uint256"}
                ],
                "name": "swapExactETHForTokens",
                "outputs": [{"internalType": "uint256[]", "name": "amounts", "type": "uint256[]"}],
                "stateMutability": "payable",
                "type": "function"
            },
            {
                "inputs": [
                    {"internalType": "uint256", "name": "amountIn", "type": "uint256"},
                    {"internalType": "uint256", "name": "amountOutMin", "type": "uint256"},
                    {"internalType": "address[]", "name": "path", "type": "address[]"},
                    {"internalType": "address", "name": "to", "type": "address"},
                    {"internalType": "uint256", "name": "deadline", "type": "uint256"}
                ],
                "name": "swapExactTokensForETH",
                "outputs": [{"internalType": "uint256[]", "name": "amounts", "type": "uint256[]"}],
                "stateMutability": "nonpayable",
                "type": "function"
            },
            {
                "inputs": [
                    {"internalType": "uint256", "name": "amountIn", "type": "uint256"},
                    {"internalType": "uint256", "name": "amountOutMin", "type": "uint256"},
                    {"internalType": "address[]", "name": "path", "type": "address[]"},
                    {"internalType": "address", "name": "to", "type": "address"},
                    {"internalType": "uint256", "name": "deadline", "type": "uint256"}
                ],
                "name": "swapExactTokensForTokens",
                "outputs": [{"internalType": "uint256[]", "name": "amounts", "type": "uint256[]"}],
                "stateMutability": "nonpayable", 
                "type": "function"
            }
        ]
        
        return self.bsc.w3.eth.contract(
            address=Web3.toChecksumAddress(self.router_address),
            abi=router_abi
        )
    
    def _get_factory_contract(self):
        """Get PancakeSwap factory contract instance."""
        factory_abi = [
            {
                "inputs": [
                    {"internalType": "address", "name": "tokenA", "type": "address"},
                    {"internalType": "address", "name": "tokenB", "type": "address"}
                ],
                "name": "getPair",
                "outputs": [{"internalType": "address", "name": "pair", "type": "address"}],
                "stateMutability": "view",
                "type": "function"
            }
        ]
        
        return self.bsc.w3.eth.contract(
            address=Web3.toChecksumAddress(self.factory_address),
            abi=factory_abi
        )
    
    def _get_pair_contract(self, pair_address: str):
        """Get liquidity pair contract instance."""
        pair_abi = [
            {
                "inputs": [],
                "name": "getReserves",
                "outputs": [
                    {"internalType": "uint112", "name": "_reserve0", "type": "uint112"},
                    {"internalType": "uint112", "name": "_reserve1", "type": "uint112"},
                    {"internalType": "uint32", "name": "_blockTimestampLast", "type": "uint32"}
                ],
                "stateMutability": "view",
                "type": "function"
            },
            {
                "inputs": [],
                "name": "token0",
                "outputs": [{"internalType": "address", "name": "", "type": "address"}],
                "stateMutability": "view",
                "type": "function"
            },
            {
                "inputs": [],
                "name": "token1", 
                "outputs": [{"internalType": "address", "name": "", "type": "address"}],
                "stateMutability": "view",
                "type": "function"
            }
        ]
        
        return self.bsc.w3.eth.contract(
            address=Web3.toChecksumAddress(pair_address),
            abi=pair_abi
        )

# Example usage
if __name__ == "__main__":
    from ..blockchain.bsc_interface import BSCInterface
    
    # Test PancakeSwap integration (read-only)
    bsc = BSCInterface(testnet=True)
    pancake = PancakeSwapTrader(bsc)
    
    print("=== PancakeSwap Integration Test ===")
    
    # Get CAKE price in BNB
    cake_price = pancake.get_token_price(bsc.TOKENS['CAKE'])
    print(f"CAKE price: {cake_price} BNB")
    
    # Get trading opportunities
    opportunities = pancake.get_trading_opportunities()
    print(f"Found {len(opportunities)} trading opportunities")
    
    for opp in opportunities[:3]:  # Show first 3
        print(f"  {opp['symbol']}: {opp['price_bnb']:.6f} BNB, Liquidity: {opp['liquidity_bnb']:.2f} BNB")
    
    print("✅ PancakeSwap integration test completed!")