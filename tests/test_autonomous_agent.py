"""
Tests for the autonomous trading agent.
"""
import pytest
import os
import sys
import asyncio
from unittest.mock import Mock, patch, MagicMock, AsyncMock
from datetime import datetime

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from autonomous_agent.trading_agent import AutonomousTradingAgent, AgentConfig, AgentState


class TestAgentConfiguration:
    """Tests for agent configuration."""

    def test_default_config(self):
        """Test default configuration."""
        config = AgentConfig()
        assert config.initial_capital == 10000.0
        assert config.paper_trading is True
        assert config.check_interval_seconds > 0

    def test_custom_config(self):
        """Test custom configuration."""
        config = AgentConfig(
            initial_capital=50000.0,
            paper_trading=False,
            check_interval_seconds=10,
            max_daily_loss=1000.0
        )
        assert config.initial_capital == 50000.0
        assert config.paper_trading is False
        assert config.max_daily_loss == 1000.0

    def test_risk_limits(self):
        """Test risk limit configuration."""
        config = AgentConfig(
            max_daily_loss=500.0,
            max_position_size=0.1,
            max_drawdown=0.15
        )
        assert config.max_daily_loss == 500.0
        assert config.max_position_size == 0.1
        assert config.max_drawdown == 0.15


class TestAgentInitialization:
    """Tests for agent initialization."""

    def test_agent_creation(self):
        """Test creating an agent."""
        config = AgentConfig(initial_capital=10000.0)
        agent = AutonomousTradingAgent(config)

        assert agent.config == config
        assert agent.state == AgentState.IDLE
        assert agent.portfolio_value == 10000.0

    def test_agent_state_initialization(self):
        """Test agent state after initialization."""
        config = AgentConfig()
        agent = AutonomousTradingAgent(config)

        assert agent.state == AgentState.IDLE
        assert len(agent.positions) == 0
        assert agent.total_pnl == 0.0

    def test_strategy_initialization(self):
        """Test strategy initialization."""
        config = AgentConfig(
            enabled_strategies=['dca_bot', 'market_making']
        )
        agent = AutonomousTradingAgent(config)

        assert len(agent.strategies) > 0
        assert 'dca_bot' in agent.strategies or 'market_making' in agent.strategies


class TestAgentStateTransitions:
    """Tests for agent state transitions."""

    @pytest.mark.asyncio
    async def test_start_agent(self):
        """Test starting the agent."""
        config = AgentConfig(check_interval_seconds=1)
        agent = AutonomousTradingAgent(config)

        # Mock the main loop to prevent infinite running
        with patch.object(agent, '_run_main_loop', new_callable=AsyncMock) as mock_loop:
            mock_loop.return_value = None

            await agent.start()
            assert agent.state == AgentState.RUNNING
            mock_loop.assert_called_once()

    @pytest.mark.asyncio
    async def test_stop_agent(self):
        """Test stopping the agent."""
        config = AgentConfig()
        agent = AutonomousTradingAgent(config)
        agent.state = AgentState.RUNNING

        await agent.stop()
        assert agent.state == AgentState.STOPPED

    @pytest.mark.asyncio
    async def test_pause_agent(self):
        """Test pausing the agent."""
        config = AgentConfig()
        agent = AutonomousTradingAgent(config)
        agent.state = AgentState.RUNNING

        agent.pause()
        assert agent.state == AgentState.PAUSED

    @pytest.mark.asyncio
    async def test_resume_agent(self):
        """Test resuming the agent."""
        config = AgentConfig()
        agent = AutonomousTradingAgent(config)
        agent.state = AgentState.PAUSED

        agent.resume()
        assert agent.state == AgentState.RUNNING


class TestSignalGeneration:
    """Tests for trading signal generation."""

    @pytest.mark.asyncio
    async def test_generate_signals(self):
        """Test signal generation."""
        config = AgentConfig()
        agent = AutonomousTradingAgent(config)

        with patch.object(agent, '_get_market_data', return_value={'BTC-USD': 40000.0}):
            signals = await agent._generate_signals()
            assert isinstance(signals, list)

    @pytest.mark.asyncio
    async def test_signal_validation(self):
        """Test signal validation."""
        config = AgentConfig()
        agent = AutonomousTradingAgent(config)

        valid_signal = {
            'symbol': 'BTC-USD',
            'action': 'BUY',
            'size': 0.1,
            'price': 40000.0,
            'strategy': 'dca_bot'
        }

        assert agent._validate_signal(valid_signal) is True

    @pytest.mark.asyncio
    async def test_invalid_signal_rejection(self):
        """Test invalid signal rejection."""
        config = AgentConfig()
        agent = AutonomousTradingAgent(config)

        invalid_signal = {
            'symbol': 'BTC-USD',
            'action': 'INVALID',
            'size': -1.0
        }

        assert agent._validate_signal(invalid_signal) is False


class TestTradeExecution:
    """Tests for trade execution."""

    @pytest.mark.asyncio
    async def test_execute_trade(self):
        """Test trade execution."""
        config = AgentConfig()
        agent = AutonomousTradingAgent(config)

        signal = {
            'symbol': 'BTC-USD',
            'action': 'BUY',
            'size': 0.1,
            'price': 40000.0,
            'strategy': 'dca_bot'
        }

        with patch.object(agent, '_place_order', return_value={'success': True}):
            result = await agent._execute_trade(signal)
            assert result['success'] is True

    @pytest.mark.asyncio
    async def test_trade_execution_with_position_update(self):
        """Test position update after trade execution."""
        config = AgentConfig()
        agent = AutonomousTradingAgent(config)

        signal = {
            'symbol': 'BTC-USD',
            'action': 'BUY',
            'size': 0.1,
            'price': 40000.0,
            'strategy': 'dca_bot'
        }

        with patch.object(agent, '_place_order', return_value={'success': True}):
            await agent._execute_trade(signal)
            assert 'BTC-USD' in agent.positions

    @pytest.mark.asyncio
    async def test_failed_trade_execution(self):
        """Test failed trade execution."""
        config = AgentConfig()
        agent = AutonomousTradingAgent(config)

        signal = {
            'symbol': 'BTC-USD',
            'action': 'BUY',
            'size': 0.1,
            'price': 40000.0,
            'strategy': 'dca_bot'
        }

        with patch.object(agent, '_place_order', side_effect=Exception('Order failed')):
            result = await agent._execute_trade(signal)
            assert result is None or result.get('success') is False


class TestRiskManagement:
    """Tests for risk management."""

    def test_check_daily_loss_limit(self):
        """Test daily loss limit check."""
        config = AgentConfig(max_daily_loss=500.0)
        agent = AutonomousTradingAgent(config)

        agent.daily_pnl = -600.0
        assert agent._check_risk_limits() is False

        agent.daily_pnl = -300.0
        assert agent._check_risk_limits() is True

    def test_check_position_size_limit(self):
        """Test position size limit check."""
        config = AgentConfig(
            max_position_size=0.2,
            initial_capital=10000.0
        )
        agent = AutonomousTradingAgent(config)

        # Position value is 25% of portfolio (exceeds limit)
        large_position = {
            'quantity': 0.1,
            'current_price': 25000.0,
            'value': 2500.0
        }

        assert agent._check_position_size(large_position) is False

        # Position value is 10% of portfolio (within limit)
        small_position = {
            'quantity': 0.025,
            'current_price': 40000.0,
            'value': 1000.0
        }

        assert agent._check_position_size(small_position) is True

    def test_max_drawdown_check(self):
        """Test maximum drawdown check."""
        config = AgentConfig(
            max_drawdown=0.15,
            initial_capital=10000.0
        )
        agent = AutonomousTradingAgent(config)

        agent.portfolio_value = 8400.0  # 16% drawdown
        assert agent._check_drawdown() is False

        agent.portfolio_value = 8600.0  # 14% drawdown
        assert agent._check_drawdown() is True


class TestPortfolioManagement:
    """Tests for portfolio management."""

    def test_update_portfolio_value(self):
        """Test portfolio value update."""
        config = AgentConfig(initial_capital=10000.0)
        agent = AutonomousTradingAgent(config)

        agent.positions = {
            'BTC-USD': {
                'quantity': 0.1,
                'avg_entry_price': 40000.0,
                'current_price': 42000.0,
                'value': 4200.0
            }
        }

        agent._update_portfolio_value()
        assert agent.portfolio_value > 10000.0

    def test_calculate_pnl(self):
        """Test P&L calculation."""
        config = AgentConfig(initial_capital=10000.0)
        agent = AutonomousTradingAgent(config)

        agent.portfolio_value = 11000.0
        pnl = agent._calculate_total_pnl()

        assert pnl == 1000.0

    def test_position_tracking(self):
        """Test position tracking."""
        config = AgentConfig()
        agent = AutonomousTradingAgent(config)

        # Add position
        agent._add_position('BTC-USD', 0.1, 40000.0)
        assert 'BTC-USD' in agent.positions
        assert agent.positions['BTC-USD']['quantity'] == 0.1

        # Update position
        agent._update_position('BTC-USD', 42000.0)
        assert agent.positions['BTC-USD']['current_price'] == 42000.0

        # Remove position
        agent._remove_position('BTC-USD')
        assert 'BTC-USD' not in agent.positions


class TestMetricsAndReporting:
    """Tests for metrics and reporting."""

    def test_calculate_sharpe_ratio(self):
        """Test Sharpe ratio calculation."""
        config = AgentConfig()
        agent = AutonomousTradingAgent(config)

        agent.returns_history = [0.01, 0.02, -0.01, 0.015, 0.005]
        sharpe = agent._calculate_sharpe_ratio()

        assert isinstance(sharpe, float)
        assert sharpe >= 0

    def test_calculate_win_rate(self):
        """Test win rate calculation."""
        config = AgentConfig()
        agent = AutonomousTradingAgent(config)

        agent.trade_history = [
            {'pnl': 100},
            {'pnl': -50},
            {'pnl': 200},
            {'pnl': 150},
            {'pnl': -100}
        ]

        win_rate = agent._calculate_win_rate()
        assert win_rate == 0.6  # 3 wins out of 5 trades

    def test_generate_performance_report(self):
        """Test performance report generation."""
        config = AgentConfig()
        agent = AutonomousTradingAgent(config)

        agent.total_pnl = 1000.0
        agent.portfolio_value = 11000.0
        agent.trade_history = [
            {'pnl': 100},
            {'pnl': 200}
        ]

        report = agent.generate_performance_report()

        assert 'total_pnl' in report
        assert 'portfolio_value' in report
        assert 'total_trades' in report
        assert report['total_pnl'] == 1000.0


class TestErrorHandling:
    """Tests for error handling."""

    @pytest.mark.asyncio
    async def test_handle_market_data_error(self):
        """Test handling market data errors."""
        config = AgentConfig()
        agent = AutonomousTradingAgent(config)

        with patch.object(agent, '_get_market_data', side_effect=Exception('API Error')):
            # Agent should handle error gracefully
            try:
                await agent._generate_signals()
            except Exception:
                pytest.fail("Agent did not handle market data error gracefully")

    @pytest.mark.asyncio
    async def test_handle_trade_execution_error(self):
        """Test handling trade execution errors."""
        config = AgentConfig()
        agent = AutonomousTradingAgent(config)

        signal = {
            'symbol': 'BTC-USD',
            'action': 'BUY',
            'size': 0.1,
            'price': 40000.0,
            'strategy': 'dca_bot'
        }

        with patch.object(agent, '_place_order', side_effect=Exception('Execution Error')):
            result = await agent._execute_trade(signal)
            # Should return None or error result, not crash
            assert result is None or 'error' in result


class TestAlertSystem:
    """Tests for alert system."""

    @pytest.mark.asyncio
    async def test_send_alert(self):
        """Test sending alerts."""
        config = AgentConfig(send_alerts=True)
        agent = AutonomousTradingAgent(config)

        with patch.object(agent, '_send_alert', new_callable=AsyncMock) as mock_alert:
            await agent._send_trade_alert('BTC-USD', 'BUY', 0.1, 40000.0)
            mock_alert.assert_called_once()

    @pytest.mark.asyncio
    async def test_risk_limit_alert(self):
        """Test risk limit breach alert."""
        config = AgentConfig(
            max_daily_loss=500.0,
            send_alerts=True
        )
        agent = AutonomousTradingAgent(config)

        agent.daily_pnl = -600.0

        with patch.object(agent, '_send_alert', new_callable=AsyncMock) as mock_alert:
            await agent._check_and_alert_risk_breach()
            mock_alert.assert_called()


if __name__ == '__main__':
    pytest.main([__file__, '-v', '-s'])
