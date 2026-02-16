#!/usr/bin/env python3
"""
Optimize strategy parameters using grid search.

This script tests multiple parameter combinations to find optimal settings
for trading strategies based on historical data.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
import pandas as pd
from itertools import product
from typing import Dict, List, Tuple
import json
from datetime import datetime

from crypto_strategies.dca_bot import DCABot, DCAConfig, DCAFrequency, DCAMode
from crypto_strategies.market_making import MarketMakingStrategy, MarketMakingConfig

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def generate_price_data(days: int = 365) -> np.ndarray:
    """Generate synthetic price data."""
    np.random.seed(42)
    base_price = 40000.0
    returns = np.random.normal(0.0001, 0.02, days * 24)
    prices = base_price * np.exp(np.cumsum(returns))
    trend = np.linspace(0, 0.5, len(prices))
    cycle = 0.1 * np.sin(np.linspace(0, 8*np.pi, len(prices)))
    return prices * (1 + trend + cycle)


def calculate_sharpe_ratio(returns: np.ndarray) -> float:
    """Calculate Sharpe ratio."""
    if len(returns) == 0 or np.std(returns) == 0:
        return 0.0
    return np.mean(returns) / np.std(returns) * np.sqrt(252)


def optimize_dca_parameters(prices: np.ndarray) -> Dict:
    """
    Optimize DCA Bot parameters using grid search.

    Parameters to optimize:
    - frequency: DAILY, WEEKLY, BIWEEKLY, MONTHLY
    - mode: FIXED, DYNAMIC, PORTFOLIO_PCT
    - base_amount: 50, 100, 200, 500
    """
    logger.info("Optimizing DCA Bot parameters...")

    param_grid = {
        'frequency': [DCAFrequency.DAILY, DCAFrequency.WEEKLY],
        'mode': [DCAMode.FIXED_AMOUNT, DCAMode.DYNAMIC],
        'base_amount': [50.0, 100.0, 200.0]
    }

    best_result = {
        'sharpe': -np.inf,
        'params': None,
        'return': 0.0,
        'pnl': 0.0
    }

    total_combinations = len(param_grid['frequency']) * len(param_grid['mode']) * len(param_grid['base_amount'])
    tested = 0

    for freq, mode, amount in product(
        param_grid['frequency'],
        param_grid['mode'],
        param_grid['base_amount']
    ):
        tested += 1
        logger.info(f"Testing {tested}/{total_combinations}: freq={freq.value}, mode={mode.value}, amount=${amount}")

        try:
            config = DCAConfig(
                symbol='BTC',
                frequency=freq,
                mode=mode,
                base_amount=amount
            )

            strategy = DCABot(config)

            # Simulate DCA
            total_coins = 0.0
            total_spent = 0.0
            equity_curve = [10000.0]

            # Frequency to intervals
            intervals = {
                DCAFrequency.DAILY: 24,
                DCAFrequency.WEEKLY: 168,
                DCAFrequency.BIWEEKLY: 336,
                DCAFrequency.MONTHLY: 720
            }

            interval = intervals[freq]

            for i in range(0, len(prices), interval):
                if total_spent >= 10000:
                    break

                price = prices[i]
                coins = amount / price
                total_coins += coins
                total_spent += amount

                current_value = total_coins * price
                equity_curve.append(current_value)

            final_value = total_coins * prices[-1]
            total_pnl = final_value - total_spent
            total_return = (total_pnl / total_spent) * 100 if total_spent > 0 else 0

            # Calculate Sharpe
            if len(equity_curve) > 1:
                returns = np.diff(equity_curve) / equity_curve[:-1]
                sharpe = calculate_sharpe_ratio(returns)
            else:
                sharpe = 0.0

            logger.info(f"  Result: Return={total_return:.2f}%, Sharpe={sharpe:.2f}, P&L=${total_pnl:.2f}")

            if sharpe > best_result['sharpe']:
                best_result = {
                    'sharpe': sharpe,
                    'params': {
                        'frequency': freq.value,
                        'mode': mode.value,
                        'base_amount': amount
                    },
                    'return': total_return,
                    'pnl': total_pnl,
                    'trades': len(equity_curve) - 1
                }

        except Exception as e:
            logger.error(f"  Error: {e}")
            continue

    return best_result


def optimize_market_making_parameters(prices: np.ndarray) -> Dict:
    """
    Optimize Market Making parameters using grid search.

    Parameters to optimize:
    - base_spread_bps: 5, 10, 20, 50
    - order_size_usd: 500, 1000, 2000
    - max_inventory_usd: 3000, 5000, 10000
    """
    logger.info("Optimizing Market Making parameters...")

    param_grid = {
        'base_spread_bps': [5.0, 10.0, 20.0],
        'order_size_usd': [500.0, 1000.0, 2000.0],
        'max_inventory_usd': [3000.0, 5000.0]
    }

    best_result = {
        'sharpe': -np.inf,
        'params': None,
        'return': 0.0,
        'pnl': 0.0
    }

    total_combinations = (
        len(param_grid['base_spread_bps']) *
        len(param_grid['order_size_usd']) *
        len(param_grid['max_inventory_usd'])
    )
    tested = 0

    for spread, size, inventory in product(
        param_grid['base_spread_bps'],
        param_grid['order_size_usd'],
        param_grid['max_inventory_usd']
    ):
        tested += 1
        logger.info(
            f"Testing {tested}/{total_combinations}: "
            f"spread={spread}bps, size=${size}, inventory=${inventory}"
        )

        try:
            config = MarketMakingConfig(
                symbol='BTC-USDC',
                base_spread_bps=spread,
                order_size_usd=size,
                max_inventory_usd=inventory
            )

            strategy = MarketMakingStrategy(config)
            results = strategy.simulate_market_making(prices, hit_probability=0.25)

            total_return = (results['total_pnl'] / results.get('initial_capital', 10000)) * 100

            # Calculate Sharpe
            if 'equity_curve' in results and len(results['equity_curve']) > 1:
                equity = np.array(results['equity_curve'])
                returns = np.diff(equity) / equity[:-1]
                sharpe = calculate_sharpe_ratio(returns)
            else:
                sharpe = 0.0

            logger.info(
                f"  Result: Return={total_return:.2f}%, Sharpe={sharpe:.2f}, "
                f"P&L=${results['total_pnl']:.2f}"
            )

            if sharpe > best_result['sharpe']:
                best_result = {
                    'sharpe': sharpe,
                    'params': {
                        'base_spread_bps': spread,
                        'order_size_usd': size,
                        'max_inventory_usd': inventory
                    },
                    'return': total_return,
                    'pnl': results['total_pnl'],
                    'trades': results.get('total_trades', 0)
                }

        except Exception as e:
            logger.error(f"  Error: {e}")
            continue

    return best_result


def generate_optimization_report(results: Dict) -> str:
    """Generate optimization report."""
    report = []
    report.append("=" * 80)
    report.append("PARAMETER OPTIMIZATION RESULTS")
    report.append("=" * 80)
    report.append(f"\nOptimization Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report.append(f"Data Period: 365 days")
    report.append(f"Optimization Method: Grid Search")
    report.append("\n" + "=" * 80)

    for strategy_name, result in results.items():
        report.append(f"\n{strategy_name.upper()}")
        report.append("-" * 80)
        report.append("\nOptimal Parameters:")
        for param, value in result['params'].items():
            report.append(f"  {param:<25} {value}")

        report.append("\nPerformance with Optimal Parameters:")
        report.append(f"  Sharpe Ratio:            {result['sharpe']:>10.2f}")
        report.append(f"  Total Return:            {result['return']:>9.2f}%")
        report.append(f"  Total P&L:               ${result['pnl']:>9,.2f}")
        report.append(f"  Number of Trades:        {result['trades']:>10}")
        report.append("")

    report.append("=" * 80)
    report.append("RECOMMENDATIONS")
    report.append("=" * 80)
    report.append("")
    report.append("Update your strategy configurations with the optimal parameters above.")
    report.append("These parameters were optimized for historical data.")
    report.append("Monitor performance in paper trading before live deployment.")
    report.append("")
    report.append("=" * 80)

    return "\n".join(report)


def main():
    """Main execution."""
    print("=" * 80)
    print("STRATEGY PARAMETER OPTIMIZATION")
    print("=" * 80)
    print("")

    # Generate price data
    logger.info("Generating synthetic price data...")
    prices = generate_price_data(days=365)
    print(f"Generated {len(prices)} hours of data")
    print(f"Price range: ${prices.min():,.2f} - ${prices.max():,.2f}")
    print("")

    # Optimize strategies
    results = {}

    print("\n" + "=" * 80)
    print("1. OPTIMIZING DCA BOT")
    print("=" * 80)
    results['DCA Bot'] = optimize_dca_parameters(prices)

    print("\n" + "=" * 80)
    print("2. OPTIMIZING MARKET MAKING")
    print("=" * 80)
    results['Market Making'] = optimize_market_making_parameters(prices)

    # Generate report
    print("\n")
    report = generate_optimization_report(results)
    print(report)

    # Save results
    output_dir = os.path.join(os.path.dirname(__file__), '..', 'backtests')
    os.makedirs(output_dir, exist_ok=True)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    report_file = os.path.join(output_dir, f'optimization_report_{timestamp}.txt')

    with open(report_file, 'w') as f:
        f.write(report)

    print(f"\nReport saved to: {report_file}")

    # Save JSON
    json_file = os.path.join(output_dir, f'optimization_results_{timestamp}.json')
    with open(json_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"Results saved to: {json_file}")
    print("\n✅ Optimization complete!")


if __name__ == '__main__':
    main()
