"""
Options trading strategies including spreads, straddles, and risk management.
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Union
import logging
from scipy.stats import norm
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

class OptionsStrategy:
    """
    Advanced options trading strategies with Greeks calculation and risk management.
    """
    
    def __init__(self, risk_free_rate: float = 0.05):
        """
        Initialize options strategy calculator.
        
        Args:
            risk_free_rate: Risk-free interest rate for options pricing
        """
        self.risk_free_rate = risk_free_rate
        
    def black_scholes_price(self, S: float, K: float, T: float, r: float, 
                           sigma: float, option_type: str = 'call') -> float:
        """
        Calculate Black-Scholes option price.
        
        Args:
            S: Current stock price
            K: Strike price
            T: Time to expiration (years)
            r: Risk-free rate
            sigma: Volatility
            option_type: 'call' or 'put'
            
        Returns:
            Option price
        """
        try:
            if T <= 0:
                # Option expired
                if option_type.lower() == 'call':
                    return max(S - K, 0)
                else:
                    return max(K - S, 0)
            
            d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
            d2 = d1 - sigma * np.sqrt(T)
            
            if option_type.lower() == 'call':
                price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
            else:  # put
                price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
            
            return max(price, 0)
            
        except Exception as e:
            logger.error(f"Error calculating Black-Scholes price: {e}")
            return 0
    
    def calculate_greeks(self, S: float, K: float, T: float, r: float, 
                        sigma: float, option_type: str = 'call') -> Dict:
        """
        Calculate option Greeks.
        
        Args:
            S: Current stock price
            K: Strike price
            T: Time to expiration (years)
            r: Risk-free rate
            sigma: Volatility
            option_type: 'call' or 'put'
            
        Returns:
            Dictionary of Greeks
        """
        try:
            if T <= 0:
                return {'delta': 0, 'gamma': 0, 'theta': 0, 'vega': 0, 'rho': 0}
            
            d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
            d2 = d1 - sigma * np.sqrt(T)
            
            # Delta
            if option_type.lower() == 'call':
                delta = norm.cdf(d1)
            else:
                delta = norm.cdf(d1) - 1
            
            # Gamma (same for calls and puts)
            gamma = norm.pdf(d1) / (S * sigma * np.sqrt(T))
            
            # Theta
            theta_part1 = -(S * norm.pdf(d1) * sigma) / (2 * np.sqrt(T))
            if option_type.lower() == 'call':
                theta_part2 = -r * K * np.exp(-r * T) * norm.cdf(d2)
                theta = (theta_part1 + theta_part2) / 365  # Daily theta
            else:
                theta_part2 = r * K * np.exp(-r * T) * norm.cdf(-d2)
                theta = (theta_part1 + theta_part2) / 365  # Daily theta
            
            # Vega (same for calls and puts)
            vega = S * norm.pdf(d1) * np.sqrt(T) / 100  # Per 1% volatility change
            
            # Rho
            if option_type.lower() == 'call':
                rho = K * T * np.exp(-r * T) * norm.cdf(d2) / 100
            else:
                rho = -K * T * np.exp(-r * T) * norm.cdf(-d2) / 100
            
            return {
                'delta': delta,
                'gamma': gamma,
                'theta': theta,
                'vega': vega,
                'rho': rho
            }
            
        except Exception as e:
            logger.error(f"Error calculating Greeks: {e}")
            return {'delta': 0, 'gamma': 0, 'theta': 0, 'vega': 0, 'rho': 0}
    
    def implied_volatility(self, market_price: float, S: float, K: float, 
                          T: float, r: float, option_type: str = 'call',
                          max_iterations: int = 100) -> float:
        """
        Calculate implied volatility using Newton-Raphson method.
        
        Args:
            market_price: Market price of the option
            S: Current stock price
            K: Strike price
            T: Time to expiration (years)
            r: Risk-free rate
            option_type: 'call' or 'put'
            max_iterations: Maximum iterations for convergence
            
        Returns:
            Implied volatility
        """
        try:
            if T <= 0 or market_price <= 0:
                return 0
            
            # Initial guess
            sigma = 0.2
            
            for i in range(max_iterations):
                # Calculate option price and vega with current sigma
                price = self.black_scholes_price(S, K, T, r, sigma, option_type)
                greeks = self.calculate_greeks(S, K, T, r, sigma, option_type)
                vega = greeks['vega'] * 100  # Convert back to per unit change
                
                # Newton-Raphson update
                if vega == 0:
                    break
                
                price_diff = price - market_price
                sigma_new = sigma - price_diff / vega
                
                # Ensure sigma stays positive
                sigma_new = max(sigma_new, 0.001)
                
                # Check convergence
                if abs(sigma_new - sigma) < 0.0001:
                    return sigma_new
                
                sigma = sigma_new
            
            return sigma
            
        except Exception as e:
            logger.error(f"Error calculating implied volatility: {e}")
            return 0.2  # Default volatility
    
    def bull_call_spread(
        self,
        S: float = None,
        K_long: float = None,
        K_short: float = None,
        T: float = None,
        sigma: float = None,
        quantity: int = 1,
        *,
        current_price: float = None,
        lower_strike: float = None,
        upper_strike: float = None,
        time_to_expiry: float = None,
        volatility: float = None,
    ) -> Dict:
        """
        Analyze bull call spread strategy.
        
        Args:
            S: Current stock price
            K_long: Strike price of long call
            K_short: Strike price of short call
            T: Time to expiration (years)
            sigma: Volatility
            quantity: Number of spreads
            
        Returns:
            Bull call spread analysis
        """
        try:
            # Support keyword arguments used in tests/demos
            if current_price is not None:
                S = current_price
            if lower_strike is not None:
                K_long = lower_strike
            if upper_strike is not None:
                K_short = upper_strike
            if time_to_expiry is not None:
                T = time_to_expiry
            if volatility is not None:
                sigma = volatility

            r = self.risk_free_rate
            
            # Calculate option prices
            long_call_price = self.black_scholes_price(S, K_long, T, r, sigma, 'call')
            short_call_price = self.black_scholes_price(S, K_short, T, r, sigma, 'call')
            
            # Net debit (cost to enter position)
            net_debit = (long_call_price - short_call_price) * quantity * 100
            
            # Maximum profit and loss
            max_profit = (K_short - K_long - (long_call_price - short_call_price)) * quantity * 100
            max_loss = net_debit
            
            # Breakeven point
            breakeven = K_long + (long_call_price - short_call_price)
            
            # Greeks
            long_greeks = self.calculate_greeks(S, K_long, T, r, sigma, 'call')
            short_greeks = self.calculate_greeks(S, K_short, T, r, sigma, 'call')
            
            net_delta = (long_greeks['delta'] - short_greeks['delta']) * quantity * 100
            net_gamma = (long_greeks['gamma'] - short_greeks['gamma']) * quantity * 100
            net_theta = (long_greeks['theta'] - short_greeks['theta']) * quantity * 100
            net_vega = (long_greeks['vega'] - short_greeks['vega']) * quantity * 100
            
            # Risk metrics
            return_on_risk = (max_profit / max_loss) if max_loss > 0 else 0
            prob_profit = self._estimate_probability_of_profit(S, breakeven, sigma, T)
            
            return {
                'strategy': 'bull_call_spread',
                'current_price': S,
                'long_strike': K_long,
                'short_strike': K_short,
                'net_debit': net_debit,
                'max_profit': max_profit,
                'max_loss': max_loss,
                'breakeven': breakeven,
                'return_on_risk': return_on_risk,
                'probability_of_profit': prob_profit,
                'greeks': {
                    'delta': net_delta,
                    'gamma': net_gamma,
                    'theta': net_theta,
                    'vega': net_vega
                },
                'days_to_expiration': T * 365,
                'quantity': quantity
            }
            
        except Exception as e:
            logger.error(f"Error analyzing bull call spread: {e}")
            return {'error': str(e)}
    
    def bear_put_spread(self, S: float, K_long: float, K_short: float, 
                       T: float, sigma: float, quantity: int = 1) -> Dict:
        """
        Analyze bear put spread strategy.
        
        Args:
            S: Current stock price
            K_long: Strike price of long put
            K_short: Strike price of short put
            T: Time to expiration (years)
            sigma: Volatility
            quantity: Number of spreads
            
        Returns:
            Bear put spread analysis
        """
        try:
            r = self.risk_free_rate
            
            # Calculate option prices
            long_put_price = self.black_scholes_price(S, K_long, T, r, sigma, 'put')
            short_put_price = self.black_scholes_price(S, K_short, T, r, sigma, 'put')
            
            # Net debit
            net_debit = (long_put_price - short_put_price) * quantity * 100
            
            # Maximum profit and loss
            max_profit = (K_long - K_short - (long_put_price - short_put_price)) * quantity * 100
            max_loss = net_debit
            
            # Breakeven point
            breakeven = K_long - (long_put_price - short_put_price)
            
            # Greeks
            long_greeks = self.calculate_greeks(S, K_long, T, r, sigma, 'put')
            short_greeks = self.calculate_greeks(S, K_short, T, r, sigma, 'put')
            
            net_delta = (long_greeks['delta'] - short_greeks['delta']) * quantity * 100
            net_gamma = (long_greeks['gamma'] - short_greeks['gamma']) * quantity * 100
            net_theta = (long_greeks['theta'] - short_greeks['theta']) * quantity * 100
            net_vega = (long_greeks['vega'] - short_greeks['vega']) * quantity * 100
            
            # Risk metrics
            return_on_risk = (max_profit / max_loss) if max_loss > 0 else 0
            prob_profit = self._estimate_probability_of_profit(S, breakeven, sigma, T, direction='down')
            
            return {
                'strategy': 'bear_put_spread',
                'current_price': S,
                'long_strike': K_long,
                'short_strike': K_short,
                'net_debit': net_debit,
                'max_profit': max_profit,
                'max_loss': max_loss,
                'breakeven': breakeven,
                'return_on_risk': return_on_risk,
                'probability_of_profit': prob_profit,
                'greeks': {
                    'delta': net_delta,
                    'gamma': net_gamma,
                    'theta': net_theta,
                    'vega': net_vega
                },
                'days_to_expiration': T * 365,
                'quantity': quantity
            }
            
        except Exception as e:
            logger.error(f"Error analyzing bear put spread: {e}")
            return {'error': str(e)}
    
    def long_straddle(
        self,
        S: float = None,
        K: float = None,
        T: float = None,
        sigma: float = None,
        quantity: int = 1,
        *,
        current_price: float = None,
        strike: float = None,
        time_to_expiry: float = None,
        volatility: float = None,
    ) -> Dict:
        """
        Analyze long straddle strategy.
        
        Args:
            S: Current stock price
            K: Strike price (same for call and put)
            T: Time to expiration (years)
            sigma: Volatility
            quantity: Number of straddles
            
        Returns:
            Long straddle analysis
        """
        try:
            if current_price is not None:
                S = current_price
            if strike is not None:
                K = strike
            if time_to_expiry is not None:
                T = time_to_expiry
            if volatility is not None:
                sigma = volatility

            r = self.risk_free_rate
            
            # Calculate option prices
            call_price = self.black_scholes_price(S, K, T, r, sigma, 'call')
            put_price = self.black_scholes_price(S, K, T, r, sigma, 'put')
            
            # Total premium paid
            total_premium = (call_price + put_price) * quantity * 100
            
            # Breakeven points
            upper_breakeven = K + call_price + put_price
            lower_breakeven = K - (call_price + put_price)
            
            # Greeks
            call_greeks = self.calculate_greeks(S, K, T, r, sigma, 'call')
            put_greeks = self.calculate_greeks(S, K, T, r, sigma, 'put')
            
            net_delta = (call_greeks['delta'] + put_greeks['delta']) * quantity * 100
            net_gamma = (call_greeks['gamma'] + put_greeks['gamma']) * quantity * 100
            net_theta = (call_greeks['theta'] + put_greeks['theta']) * quantity * 100
            net_vega = (call_greeks['vega'] + put_greeks['vega']) * quantity * 100
            
            # Profit/Loss at different price levels
            price_range = np.linspace(S * 0.7, S * 1.3, 20)
            pnl_profile = []
            
            for price in price_range:
                call_value = max(price - K, 0)
                put_value = max(K - price, 0)
                total_value = (call_value + put_value) * quantity * 100
                pnl = total_value - total_premium
                pnl_profile.append({'price': price, 'pnl': pnl})
            
            # Volatility impact
            vol_sensitivity = net_vega * 0.01  # Impact of 1% volatility change
            
            return {
                'strategy': 'long_straddle',
                'current_price': S,
                'strike_price': K,
                'call_price': call_price,
                'put_price': put_price,
                'total_premium': total_premium,
                'upper_breakeven': upper_breakeven,
                'lower_breakeven': lower_breakeven,
                'max_loss': total_premium,
                'greeks': {
                    'delta': net_delta,
                    'gamma': net_gamma,
                    'theta': net_theta,
                    'vega': net_vega
                },
                'volatility_sensitivity': vol_sensitivity,
                'pnl_profile': pnl_profile,
                'days_to_expiration': T * 365,
                'quantity': quantity
            }
            
        except Exception as e:
            logger.error(f"Error analyzing long straddle: {e}")
            return {'error': str(e)}
    
    def iron_condor(self, S: float, K1: float, K2: float, K3: float, K4: float,
                   T: float, sigma: float, quantity: int = 1) -> Dict:
        """
        Analyze iron condor strategy.
        
        Args:
            S: Current stock price
            K1: Short put strike (lowest)
            K2: Long put strike
            K3: Long call strike
            K4: Short call strike (highest)
            T: Time to expiration (years)
            sigma: Volatility
            quantity: Number of iron condors
            
        Returns:
            Iron condor analysis
        """
        try:
            r = self.risk_free_rate
            
            # Calculate option prices
            short_put_price = self.black_scholes_price(S, K1, T, r, sigma, 'put')
            long_put_price = self.black_scholes_price(S, K2, T, r, sigma, 'put')
            long_call_price = self.black_scholes_price(S, K3, T, r, sigma, 'call')
            short_call_price = self.black_scholes_price(S, K4, T, r, sigma, 'call')
            
            # Net credit received
            net_credit = (short_put_price - long_put_price + short_call_price - long_call_price) * quantity * 100
            
            # Maximum profit (credit received)
            max_profit = net_credit
            
            # Maximum loss
            put_spread_width = (K2 - K1) * quantity * 100
            call_spread_width = (K4 - K3) * quantity * 100
            max_loss = max(put_spread_width, call_spread_width) - net_credit
            
            # Breakeven points
            lower_breakeven = K2 - (net_credit / (quantity * 100))
            upper_breakeven = K3 + (net_credit / (quantity * 100))
            
            # Profit zone
            profit_range = upper_breakeven - lower_breakeven
            profit_range_pct = profit_range / S * 100
            
            # Greeks (net position)
            greeks_sum = {'delta': 0, 'gamma': 0, 'theta': 0, 'vega': 0}
            
            # Short put
            sp_greeks = self.calculate_greeks(S, K1, T, r, sigma, 'put')
            for greek in greeks_sum:
                greeks_sum[greek] -= sp_greeks[greek]
            
            # Long put
            lp_greeks = self.calculate_greeks(S, K2, T, r, sigma, 'put')
            for greek in greeks_sum:
                greeks_sum[greek] += lp_greeks[greek]
            
            # Long call
            lc_greeks = self.calculate_greeks(S, K3, T, r, sigma, 'call')
            for greek in greeks_sum:
                greeks_sum[greek] += lc_greeks[greek]
            
            # Short call
            sc_greeks = self.calculate_greeks(S, K4, T, r, sigma, 'call')
            for greek in greeks_sum:
                greeks_sum[greek] -= sc_greeks[greek]
            
            # Scale by quantity
            for greek in greeks_sum:
                greeks_sum[greek] *= quantity * 100
            
            # Probability of profit (price stays between breakevens)
            prob_profit = self._estimate_probability_between_levels(S, lower_breakeven, upper_breakeven, sigma, T)
            
            return {
                'strategy': 'iron_condor',
                'current_price': S,
                'strikes': {'put_short': K1, 'put_long': K2, 'call_long': K3, 'call_short': K4},
                'net_credit': net_credit,
                'max_profit': max_profit,
                'max_loss': max_loss,
                'lower_breakeven': lower_breakeven,
                'upper_breakeven': upper_breakeven,
                'profit_range': profit_range,
                'profit_range_percent': profit_range_pct,
                'probability_of_profit': prob_profit,
                'greeks': greeks_sum,
                'days_to_expiration': T * 365,
                'quantity': quantity
            }
            
        except Exception as e:
            logger.error(f"Error analyzing iron condor: {e}")
            return {'error': str(e)}
    
    def _estimate_probability_of_profit(self, S: float, target: float, sigma: float, 
                                      T: float, direction: str = 'up') -> float:
        """Estimate probability of reaching target price."""
        try:
            if T <= 0:
                return 1.0 if (direction == 'up' and S >= target) or (direction == 'down' and S <= target) else 0.0
            
            # Use lognormal distribution
            mu = np.log(S) + (self.risk_free_rate - 0.5 * sigma**2) * T
            std = sigma * np.sqrt(T)
            
            if direction == 'up':
                prob = 1 - norm.cdf(np.log(target), mu, std)
            else:  # down
                prob = norm.cdf(np.log(target), mu, std)
            
            return max(0, min(1, prob))
            
        except Exception:
            return 0.5  # Default probability
    
    def _estimate_probability_between_levels(self, S: float, lower: float, upper: float, 
                                           sigma: float, T: float) -> float:
        """Estimate probability of price staying between two levels."""
        try:
            if T <= 0:
                return 1.0 if lower <= S <= upper else 0.0
            
            mu = np.log(S) + (self.risk_free_rate - 0.5 * sigma**2) * T
            std = sigma * np.sqrt(T)
            
            prob_below_upper = norm.cdf(np.log(upper), mu, std)
            prob_below_lower = norm.cdf(np.log(lower), mu, std)
            
            prob_between = prob_below_upper - prob_below_lower
            return max(0, min(1, prob_between))
            
        except Exception:
            return 0.5  # Default probability
    
    def screen_options_opportunities(self, symbol: str, current_price: float, 
                                   volatility: float, market_outlook: str,
                                   days_to_expiration: int = 30) -> List[Dict]:
        """
        Screen for attractive options trading opportunities.
        
        Args:
            symbol: Stock symbol
            current_price: Current stock price
            volatility: Implied volatility
            market_outlook: 'bullish', 'bearish', 'neutral', or 'volatile'
            days_to_expiration: Days until expiration
            
        Returns:
            List of recommended options strategies
        """
        try:
            T = days_to_expiration / 365
            opportunities = []
            
            # Generate strike prices around current price
            strikes = []
            for pct in [-0.1, -0.05, 0, 0.05, 0.1]:
                strikes.append(current_price * (1 + pct))
            
            if market_outlook.lower() == 'bullish':
                # Bull call spreads
                for i in range(len(strikes) - 1):
                    if strikes[i] <= current_price <= strikes[i+1]:
                        strategy = self.bull_call_spread(
                            current_price, strikes[i], strikes[i+1], T, volatility
                        )
                        if 'error' not in strategy and strategy['return_on_risk'] > 0.3:
                            strategy['recommendation_score'] = (
                                strategy['return_on_risk'] * 0.4 + 
                                strategy['probability_of_profit'] * 0.6
                            )
                            opportunities.append(strategy)
            
            elif market_outlook.lower() == 'bearish':
                # Bear put spreads
                for i in range(len(strikes) - 1):
                    if strikes[i] <= current_price <= strikes[i+1]:
                        strategy = self.bear_put_spread(
                            current_price, strikes[i+1], strikes[i], T, volatility
                        )
                        if 'error' not in strategy and strategy['return_on_risk'] > 0.3:
                            strategy['recommendation_score'] = (
                                strategy['return_on_risk'] * 0.4 + 
                                strategy['probability_of_profit'] * 0.6
                            )
                            opportunities.append(strategy)
            
            elif market_outlook.lower() == 'volatile':
                # Long straddles
                atm_strike = min(strikes, key=lambda x: abs(x - current_price))
                strategy = self.long_straddle(current_price, atm_strike, T, volatility)
                if 'error' not in strategy:
                    # Score based on volatility sensitivity and time decay
                    vol_score = min(strategy['volatility_sensitivity'] / 100, 1.0)
                    time_score = max(0, 1 - abs(strategy['greeks']['theta']) / 50)
                    strategy['recommendation_score'] = vol_score * 0.7 + time_score * 0.3
                    opportunities.append(strategy)
            
            elif market_outlook.lower() == 'neutral':
                # Iron condors
                if len(strikes) >= 4:
                    strategy = self.iron_condor(
                        current_price, strikes[0], strikes[1], strikes[-2], strikes[-1], T, volatility
                    )
                    if 'error' not in strategy and strategy['probability_of_profit'] > 0.6:
                        strategy['recommendation_score'] = (
                            strategy['probability_of_profit'] * 0.8 + 
                            (strategy['max_profit'] / abs(strategy['max_loss'])) * 0.2
                        )
                        opportunities.append(strategy)
            
            # Sort by recommendation score
            opportunities.sort(key=lambda x: x.get('recommendation_score', 0), reverse=True)
            
            return opportunities[:5]  # Return top 5 opportunities
            
        except Exception as e:
            logger.error(f"Error screening options opportunities: {e}")
            return []

# Example usage
if __name__ == "__main__":
    print("📈 Options Strategies Demo")
    print("=" * 40)
    
    options_strategy = OptionsStrategy()
    
    # Example parameters
    current_price = 150.0
    strike_price = 150.0
    time_to_expiration = 30 / 365  # 30 days
    volatility = 0.25
    
    print(f"Current Price: ${current_price}")
    print(f"Volatility: {volatility:.1%}")
    print(f"Days to Expiration: 30")
    
    # Bull Call Spread
    print(f"\n🐂 Bull Call Spread (Buy $145 Call, Sell $155 Call):")
    bull_spread = options_strategy.bull_call_spread(
        current_price, 145, 155, time_to_expiration, volatility
    )
    if 'error' not in bull_spread:
        print(f"  Net Debit: ${bull_spread['net_debit']:.2f}")
        print(f"  Max Profit: ${bull_spread['max_profit']:.2f}")
        print(f"  Max Loss: ${bull_spread['max_loss']:.2f}")
        print(f"  Breakeven: ${bull_spread['breakeven']:.2f}")
        print(f"  Return on Risk: {bull_spread['return_on_risk']:.2%}")
        print(f"  Probability of Profit: {bull_spread['probability_of_profit']:.1%}")
    
    # Long Straddle
    print(f"\n🎭 Long Straddle (Buy $150 Call & Put):")
    straddle = options_strategy.long_straddle(
        current_price, strike_price, time_to_expiration, volatility
    )
    if 'error' not in straddle:
        print(f"  Total Premium: ${straddle['total_premium']:.2f}")
        print(f"  Upper Breakeven: ${straddle['upper_breakeven']:.2f}")
        print(f"  Lower Breakeven: ${straddle['lower_breakeven']:.2f}")
        print(f"  Max Loss: ${straddle['max_loss']:.2f}")
        print(f"  Volatility Sensitivity: ${straddle['volatility_sensitivity']:.2f}")
    
    # Iron Condor
    print(f"\n🦅 Iron Condor ($140/$145/$155/$160):")
    iron_condor = options_strategy.iron_condor(
        current_price, 140, 145, 155, 160, time_to_expiration, volatility
    )
    if 'error' not in iron_condor:
        print(f"  Net Credit: ${iron_condor['net_credit']:.2f}")
        print(f"  Max Profit: ${iron_condor['max_profit']:.2f}")
        print(f"  Max Loss: ${iron_condor['max_loss']:.2f}")
        print(f"  Profit Range: ${iron_condor['lower_breakeven']:.2f} - ${iron_condor['upper_breakeven']:.2f}")
        print(f"  Probability of Profit: {iron_condor['probability_of_profit']:.1%}")
    
    # Strategy Screening
    print(f"\n🔍 Options Opportunities (Bullish Outlook):")
    opportunities = options_strategy.screen_options_opportunities(
        'DEMO', current_price, volatility, 'bullish'
    )
    
    for i, opp in enumerate(opportunities[:3], 1):
        print(f"  {i}. {opp['strategy'].replace('_', ' ').title()}")
        print(f"     Score: {opp.get('recommendation_score', 0):.3f}")
        print(f"     Return/Risk: {opp.get('return_on_risk', 0):.2%}")
    
    print("\n✅ Options strategies demo completed!")
