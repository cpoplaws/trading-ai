"""
Autonomous Trading Agent
AI agent that trades automatically without human intervention.

Features:
- Monitors markets 24/7
- Executes multiple strategies simultaneously
- Makes autonomous trading decisions
- Manages risk and portfolio allocation
- Sends alerts on important events
- Self-optimizes based on performance
"""
import logging
import time
import asyncio
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

logger = logging.getLogger(__name__)


class AgentState(Enum):
    """Agent operational state."""
    IDLE = "idle"
    MONITORING = "monitoring"
    ANALYZING = "analyzing"
    EXECUTING = "executing"
    RISK_CHECK = "risk_check"
    PAUSED = "paused"
    ERROR = "error"


class MarketCondition(Enum):
    """Market condition assessment."""
    STRONG_UPTREND = "strong_uptrend"
    UPTREND = "uptrend"
    RANGING = "ranging"
    DOWNTREND = "downtrend"
    STRONG_DOWNTREND = "strong_downtrend"
    HIGH_VOLATILITY = "high_volatility"


@dataclass
class AgentConfig:
    """Autonomous agent configuration."""
    # Portfolio
    initial_capital: float = 10000.0
    max_portfolio_risk: float = 0.10  # 10% max drawdown
    max_position_size: float = 0.20  # 20% per position
    
    # Strategy allocation
    strategies_enabled: List[str] = None
    
    # Risk management
    stop_loss_percent: float = 5.0
    take_profit_multiplier: float = 2.0
    max_daily_loss: float = 500.0
    
    # Monitoring
    check_interval_seconds: int = 60  # Check every minute
    price_update_interval: int = 10  # Update prices every 10s
    
    # Trading
    paper_trading: bool = True
    max_trades_per_day: int = 50
    
    # Alerts
    send_alerts: bool = True
    alert_on_trades: bool = True
    alert_on_errors: bool = True


@dataclass
class AgentMetrics:
    """Agent performance metrics."""
    uptime_seconds: int
    total_trades: int
    winning_trades: int
    losing_trades: int
    total_pnl: float
    daily_pnl: float
    win_rate: float
    sharpe_ratio: float
    max_drawdown: float
    current_positions: int
    strategies_active: int


class AutonomousTradingAgent:
    """
    Autonomous Trading Agent.
    
    Runs continuously, monitors markets, executes strategies,
    and manages portfolio automatically.
    """
    
    def __init__(self, config: AgentConfig):
        """
        Initialize autonomous agent.
        
        Args:
            config: Agent configuration
        """
        self.config = config
        self.state = AgentState.IDLE
        self.is_running = False
        
        # Portfolio tracking
        self.portfolio_value = config.initial_capital
        self.cash_balance = config.initial_capital
        self.positions: Dict[str, Any] = {}
        
        # Performance tracking
        self.total_pnl = 0.0
        self.daily_pnl = 0.0
        self.trades_today = 0
        self.trade_history: List[Dict] = []
        
        # Market analysis
        self.current_market_condition = MarketCondition.RANGING
        self.price_cache: Dict[str, float] = {}
        
        # Strategies
        self.active_strategies = {}
        self.strategy_performance: Dict[str, Dict] = {}
        
        # Timing
        self.start_time = None
        self.last_check = None
        self.last_price_update = None
        
        logger.info("Autonomous Trading Agent initialized")
    
    async def start(self):
        """Start the autonomous agent."""
        logger.info("🤖 Starting Autonomous Trading Agent...")
        
        self.is_running = True
        self.start_time = datetime.now()
        self.state = AgentState.MONITORING
        
        # Initialize strategies
        await self._initialize_strategies()
        
        # Print startup banner
        self._print_startup_banner()
        
        # Main control loop
        try:
            await self._main_loop()
        except KeyboardInterrupt:
            logger.info("Agent stopped by user")
            await self.stop()
        except Exception as e:
            logger.error(f"Agent error: {e}")
            self.state = AgentState.ERROR
            raise
    
    async def _main_loop(self):
        """Main agent control loop."""
        while self.is_running:
            try:
                cycle_start = time.time()
                
                # 1. Update market data
                await self._update_market_data()
                
                # 2. Analyze market conditions
                await self._analyze_market()
                
                # 3. Risk check
                if not await self._risk_check():
                    logger.warning("Risk check failed - pausing trading")
                    self.state = AgentState.PAUSED
                    await asyncio.sleep(60)
                    continue
                
                # 4. Generate signals from strategies
                signals = await self._generate_signals()
                
                # 5. Execute trades
                if signals:
                    await self._execute_signals(signals)
                
                # 6. Update portfolio
                await self._update_portfolio()
                
                # 7. Check stop conditions
                if self._should_stop_trading():
                    logger.info("Stop condition met - pausing")
                    self.state = AgentState.PAUSED
                    await asyncio.sleep(300)  # Wait 5 min
                    continue
                
                # 8. Log status
                if int(time.time()) % 60 == 0:  # Every minute
                    self._log_status()
                
                # Sleep until next cycle
                elapsed = time.time() - cycle_start
                sleep_time = max(0, self.config.check_interval_seconds - elapsed)
                await asyncio.sleep(sleep_time)
                
            except Exception as e:
                logger.error(f"Error in main loop: {e}")
                await asyncio.sleep(10)
    
    async def _initialize_strategies(self):
        """Initialize trading strategies."""
        logger.info("Initializing strategies...")
        
        # Import strategies
        try:
            from crypto_strategies.market_making import MarketMakingStrategy, MarketMakingConfig
            from crypto_strategies.dca_bot import DCABot, DCAConfig, DCAFrequency, DCAMode
            from crypto_strategies.momentum import MomentumStrategy, MomentumConfig
            
            # Market Making
            mm_config = MarketMakingConfig(
                symbol='BTC-USDC',
                base_spread_bps=10.0,
                order_size_usd=500.0,
                max_inventory_usd=2000.0
            )
            self.active_strategies['market_making'] = {
                'strategy': MarketMakingStrategy(mm_config),
                'config': mm_config,
                'weight': 0.3,
                'enabled': True
            }
            
            # DCA Bot
            dca_config = DCAConfig(
                symbol='BTC',
                frequency=DCAFrequency.DAILY,
                mode=DCAMode.DYNAMIC,
                base_amount=50.0
            )
            self.active_strategies['dca'] = {
                'strategy': DCABot(dca_config),
                'config': dca_config,
                'weight': 0.2,
                'enabled': True
            }
            
            # Momentum
            momentum_config = MomentumConfig(
                symbol='BTC',
                adx_threshold=25.0,
                use_trailing_stop=True
            )
            self.active_strategies['momentum'] = {
                'strategy': MomentumStrategy(momentum_config),
                'config': momentum_config,
                'weight': 0.3,
                'enabled': True
            }
            
            logger.info(f"✅ {len(self.active_strategies)} strategies initialized")
            
        except Exception as e:
            logger.error(f"Failed to initialize strategies: {e}")
    
    async def _update_market_data(self):
        """Update market data."""
        # Simulate price updates
        # In production, this would fetch from exchange API
        import numpy as np
        
        if 'BTC' not in self.price_cache:
            self.price_cache['BTC'] = 40000.0
        
        # Simulate price movement
        change = np.random.normal(0, 0.002)  # 0.2% volatility
        self.price_cache['BTC'] *= (1 + change)
        
        self.last_price_update = datetime.now()
    
    async def _analyze_market(self):
        """Analyze current market conditions."""
        self.state = AgentState.ANALYZING
        
        # Simple market condition detection
        # In production, use more sophisticated analysis
        if 'BTC' in self.price_cache:
            price = self.price_cache['BTC']
            
            # Placeholder logic
            if price > 45000:
                self.current_market_condition = MarketCondition.STRONG_UPTREND
            elif price > 42000:
                self.current_market_condition = MarketCondition.UPTREND
            elif price > 38000:
                self.current_market_condition = MarketCondition.RANGING
            else:
                self.current_market_condition = MarketCondition.DOWNTREND
    
    async def _risk_check(self) -> bool:
        """
        Check if trading should continue based on risk limits.
        
        Returns:
            True if risk is acceptable
        """
        self.state = AgentState.RISK_CHECK
        
        # Check daily loss limit
        if abs(self.daily_pnl) > self.config.max_daily_loss:
            logger.warning(f"Daily loss limit exceeded: ${self.daily_pnl:.2f}")
            return False
        
        # Check max trades per day
        if self.trades_today >= self.config.max_trades_per_day:
            logger.warning(f"Max trades per day reached: {self.trades_today}")
            return False
        
        # Check portfolio drawdown
        if self.portfolio_value > 0:
            drawdown = (self.config.initial_capital - self.portfolio_value) / self.config.initial_capital
            if drawdown > self.config.max_portfolio_risk:
                logger.warning(f"Max drawdown exceeded: {drawdown:.1%}")
                return False
        
        return True
    
    async def _generate_signals(self) -> List[Dict]:
        """
        Generate trading signals from all strategies.
        
        Returns:
            List of trading signals
        """
        signals = []
        
        for name, strategy_data in self.active_strategies.items():
            if not strategy_data['enabled']:
                continue
            
            try:
                strategy = strategy_data['strategy']
                
                # Generate signal based on strategy type
                if name == 'market_making':
                    quote = strategy.generate_quotes(
                        mid_price=self.price_cache.get('BTC', 40000)
                    )
                    if quote:
                        signals.append({
                            'strategy': name,
                            'type': 'market_making',
                            'data': quote
                        })
                
                elif name == 'dca':
                    # DCA runs on schedule
                    result = strategy.execute_purchase(
                        current_price=self.price_cache.get('BTC', 40000),
                        current_time=datetime.now()
                    )
                    if result.get('executed'):
                        signals.append({
                            'strategy': name,
                            'type': 'dca',
                            'data': result
                        })
                
                elif name == 'momentum':
                    signal = strategy.generate_signal(
                        current_price=self.price_cache.get('BTC', 40000)
                    )
                    if signal:
                        signals.append({
                            'strategy': name,
                            'type': 'momentum',
                            'data': signal
                        })
                
            except Exception as e:
                logger.error(f"Error generating signal for {name}: {e}")
        
        return signals
    
    async def _execute_signals(self, signals: List[Dict]):
        """Execute trading signals."""
        self.state = AgentState.EXECUTING
        
        for signal in signals:
            try:
                # Position sizing
                capital_allocation = self.cash_balance * self.config.max_position_size
                
                # Execute trade (simulated)
                trade = {
                    'timestamp': datetime.now(),
                    'strategy': signal['strategy'],
                    'type': signal['type'],
                    'price': self.price_cache.get('BTC', 40000),
                    'size': capital_allocation,
                    'pnl': 0.0
                }
                
                self.trade_history.append(trade)
                self.trades_today += 1
                
                logger.info(f"✅ Trade executed: {signal['strategy']} - ${capital_allocation:.2f}")
                
            except Exception as e:
                logger.error(f"Error executing signal: {e}")
    
    async def _update_portfolio(self):
        """Update portfolio value and P&L."""
        # Calculate current portfolio value
        # In production, this would fetch real positions and valuations
        
        # Simulate some profit/loss
        import numpy as np
        daily_return = np.random.normal(0.001, 0.01)  # 0.1% avg, 1% volatility
        pnl_today = self.portfolio_value * daily_return
        
        self.portfolio_value += pnl_today
        self.daily_pnl += pnl_today
        self.total_pnl += pnl_today
    
    def _should_stop_trading(self) -> bool:
        """Check if should stop trading."""
        # Stop if significant losses
        if self.daily_pnl < -self.config.max_daily_loss:
            return True
        
        # Stop if too many trades
        if self.trades_today >= self.config.max_trades_per_day:
            return True
        
        return False
    
    def _log_status(self):
        """Log agent status."""
        uptime = (datetime.now() - self.start_time).total_seconds() / 3600
        
        logger.info(
            f"📊 Agent Status | "
            f"State: {self.state.value} | "
            f"Portfolio: ${self.portfolio_value:.2f} | "
            f"P&L: ${self.total_pnl:+.2f} | "
            f"Trades: {len(self.trade_history)} | "
            f"Market: {self.current_market_condition.value} | "
            f"Uptime: {uptime:.1f}h"
        )
    
    def _print_startup_banner(self):
        """Print startup banner."""
        print("\n" + "="*60)
        print("🤖 AUTONOMOUS TRADING AGENT STARTED")
        print("="*60)
        print(f"\n💰 Initial Capital: ${self.config.initial_capital:,.2f}")
        print(f"📊 Strategies: {len(self.active_strategies)} active")
        print(f"⚙️  Mode: {'Paper Trading' if self.config.paper_trading else 'LIVE TRADING'}")
        print(f"🔄 Check Interval: {self.config.check_interval_seconds}s")
        print(f"🛡️  Max Daily Loss: ${self.config.max_daily_loss}")
        print(f"\n🚀 Agent is now monitoring markets and trading autonomously...")
        print(f"Press Ctrl+C to stop\n")
        print("="*60 + "\n")
    
    async def stop(self):
        """Stop the agent."""
        logger.info("Stopping agent...")
        self.is_running = False
        self.state = AgentState.IDLE
        
        # Print final report
        self._print_final_report()
    
    def _print_final_report(self):
        """Print final performance report."""
        uptime = (datetime.now() - self.start_time).total_seconds() / 3600
        win_rate = 0
        if self.trade_history:
            wins = sum(1 for t in self.trade_history if t.get('pnl', 0) > 0)
            win_rate = wins / len(self.trade_history) * 100
        
        print("\n" + "="*60)
        print("📊 AUTONOMOUS AGENT - FINAL REPORT")
        print("="*60)
        print(f"\n⏱️  Runtime: {uptime:.1f} hours")
        print(f"💰 Starting Capital: ${self.config.initial_capital:,.2f}")
        print(f"💵 Final Portfolio: ${self.portfolio_value:,.2f}")
        print(f"📈 Total P&L: ${self.total_pnl:+,.2f} ({(self.total_pnl/self.config.initial_capital)*100:+.2f}%)")
        print(f"📊 Total Trades: {len(self.trade_history)}")
        print(f"✅ Win Rate: {win_rate:.1f}%")
        print(f"🎯 Strategies Used: {len(self.active_strategies)}")
        print("\n" + "="*60 + "\n")
    
    def get_metrics(self) -> AgentMetrics:
        """Get current agent metrics."""
        uptime = 0
        if self.start_time:
            uptime = int((datetime.now() - self.start_time).total_seconds())
        
        wins = sum(1 for t in self.trade_history if t.get('pnl', 0) > 0)
        losses = len(self.trade_history) - wins
        win_rate = wins / len(self.trade_history) if self.trade_history else 0
        
        return AgentMetrics(
            uptime_seconds=uptime,
            total_trades=len(self.trade_history),
            winning_trades=wins,
            losing_trades=losses,
            total_pnl=self.total_pnl,
            daily_pnl=self.daily_pnl,
            win_rate=win_rate,
            sharpe_ratio=0.0,  # Calculate if needed
            max_drawdown=0.0,  # Calculate if needed
            current_positions=len(self.positions),
            strategies_active=len([s for s in self.active_strategies.values() if s['enabled']])
        )


async def main():
    """Main entry point."""
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    # Create agent config
    config = AgentConfig(
        initial_capital=10000.0,
        paper_trading=True,
        check_interval_seconds=5,  # Check every 5 seconds for demo
        max_daily_loss=500.0,
        send_alerts=True
    )
    
    # Create and start agent
    agent = AutonomousTradingAgent(config)
    await agent.start()


if __name__ == '__main__':
    asyncio.run(main())
