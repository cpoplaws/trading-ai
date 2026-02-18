"""
Agent Swarm Controller
Manages multiple specialized trading agents working together
"""
import logging
from typing import Dict, List, Optional, Tuple
from datetime import datetime
from enum import Enum
import numpy as np

logger = logging.getLogger(__name__)


class AgentAction(Enum):
    """Agent action types"""
    HOLD = 0
    BUY = 1
    SELL = 2
    REDUCE_POSITION = 3
    INCREASE_POSITION = 4


class AgentDecision:
    """Agent decision with confidence"""
    def __init__(self, action: AgentAction, confidence: float, reason: str = ""):
        self.action = action
        self.confidence = confidence
        self.reason = reason
        self.timestamp = datetime.now()


class BaseAgent:
    """Base class for specialized agents"""

    def __init__(self, name: str, enabled: bool = True):
        self.name = name
        self.enabled = enabled
        self.decisions = []
        self.performance = {
            "total_decisions": 0,
            "successful_decisions": 0,
            "failed_decisions": 0,
            "accuracy": 0.0
        }

    def decide(self, market_state: Dict) -> AgentDecision:
        """
        Make a decision based on market state

        Args:
            market_state: Current market data and portfolio state

        Returns:
            AgentDecision with action and confidence
        """
        raise NotImplementedError

    def update_performance(self, success: bool):
        """Update agent performance metrics"""
        self.performance["total_decisions"] += 1
        if success:
            self.performance["successful_decisions"] += 1
        else:
            self.performance["failed_decisions"] += 1

        if self.performance["total_decisions"] > 0:
            self.performance["accuracy"] = (
                self.performance["successful_decisions"] /
                self.performance["total_decisions"]
            )


class ExecutionAgent(BaseAgent):
    """
    Optimizes trade execution timing and sizing
    Minimizes slippage and market impact
    """

    def __init__(self):
        super().__init__("ExecutionAgent")

    def decide(self, market_state: Dict) -> AgentDecision:
        """
        Decide on execution timing and sizing

        Considers:
        - Market volatility
        - Spread
        - Volume
        - Time of day
        """
        # Extract market metrics
        volatility = market_state.get("volatility", 0.02)
        spread = market_state.get("spread", 0.001)
        volume = market_state.get("volume", 1000000)

        # Simple execution logic
        # Low volatility + tight spread = good time to execute
        if volatility < 0.015 and spread < 0.002:
            confidence = 0.8
            action = AgentAction.INCREASE_POSITION
            reason = "Low volatility, tight spread - good execution conditions"

        # High volatility = wait
        elif volatility > 0.03:
            confidence = 0.7
            action = AgentAction.HOLD
            reason = "High volatility - waiting for better conditions"

        # Normal conditions
        else:
            confidence = 0.6
            action = AgentAction.HOLD
            reason = "Normal market conditions - monitoring"

        decision = AgentDecision(action, confidence, reason)
        self.decisions.append(decision)
        return decision


class RiskAgent(BaseAgent):
    """
    Monitors and controls portfolio risk
    Enforces position limits and stops
    """

    def __init__(self):
        super().__init__("RiskAgent")
        self.max_position_pct = 0.20  # 20% max per position
        self.max_drawdown = 0.10  # 10% max drawdown
        self.max_leverage = 1.0  # No leverage

    def decide(self, market_state: Dict) -> AgentDecision:
        """
        Assess risk and recommend actions

        Considers:
        - Position size
        - Portfolio drawdown
        - Leverage
        - Correlation
        """
        # Extract risk metrics
        position_pct = market_state.get("position_pct", 0.0)
        drawdown = market_state.get("drawdown", 0.0)
        leverage = market_state.get("leverage", 1.0)
        pnl_pct = market_state.get("pnl_pct", 0.0)

        # Risk checks
        # 1. Excessive position size
        if position_pct > self.max_position_pct:
            confidence = 0.9
            action = AgentAction.REDUCE_POSITION
            reason = f"Position size {position_pct:.1%} exceeds limit {self.max_position_pct:.1%}"

        # 2. Drawdown limit
        elif drawdown > self.max_drawdown:
            confidence = 0.95
            action = AgentAction.SELL
            reason = f"Drawdown {drawdown:.1%} exceeds limit {self.max_drawdown:.1%}"

        # 3. Stop loss (5%)
        elif pnl_pct < -0.05:
            confidence = 0.85
            action = AgentAction.SELL
            reason = f"Stop loss triggered: {pnl_pct:.1%}"

        # 4. Take profit (15%)
        elif pnl_pct > 0.15:
            confidence = 0.75
            action = AgentAction.SELL
            reason = f"Take profit triggered: {pnl_pct:.1%}"

        # Normal risk levels
        else:
            confidence = 0.7
            action = AgentAction.HOLD
            reason = "Risk within acceptable levels"

        decision = AgentDecision(action, confidence, reason)
        self.decisions.append(decision)
        return decision


class ArbitrageAgent(BaseAgent):
    """
    Identifies and executes arbitrage opportunities
    Looks for price discrepancies across markets
    """

    def __init__(self):
        super().__init__("ArbitrageAgent")
        self.min_spread = 0.003  # 0.3% minimum spread

    def decide(self, market_state: Dict) -> AgentDecision:
        """
        Detect arbitrage opportunities

        Considers:
        - Price differences across exchanges
        - Funding rate arbitrage
        - Statistical arbitrage signals
        """
        # For now, simple mean reversion arbitrage
        price_zscore = market_state.get("price_zscore", 0.0)

        # Strong deviation = arbitrage opportunity
        if abs(price_zscore) > 2.0:
            if price_zscore > 2.0:
                # Price too high - sell
                confidence = min(0.8, abs(price_zscore) / 3.0)
                action = AgentAction.SELL
                reason = f"Price {price_zscore:.2f} std above mean - arbitrage sell"
            else:
                # Price too low - buy
                confidence = min(0.8, abs(price_zscore) / 3.0)
                action = AgentAction.BUY
                reason = f"Price {abs(price_zscore):.2f} std below mean - arbitrage buy"

        # Moderate deviation
        elif abs(price_zscore) > 1.5:
            confidence = 0.6
            action = AgentAction.HOLD
            reason = f"Moderate deviation ({price_zscore:.2f} std) - monitoring"

        # No arbitrage
        else:
            confidence = 0.5
            action = AgentAction.HOLD
            reason = "No arbitrage opportunity detected"

        decision = AgentDecision(action, confidence, reason)
        self.decisions.append(decision)
        return decision


class MarketMakingAgent(BaseAgent):
    """
    Provides liquidity and captures spread
    Places limit orders on both sides
    """

    def __init__(self):
        super().__init__("MarketMakingAgent")
        self.target_spread = 0.002  # 0.2% spread
        self.max_inventory = 1000  # Max inventory units

    def decide(self, market_state: Dict) -> AgentDecision:
        """
        Decide on market making strategy

        Considers:
        - Current spread
        - Inventory position
        - Market volatility
        - Order book depth
        """
        spread = market_state.get("spread", 0.001)
        inventory = market_state.get("inventory", 0)
        volatility = market_state.get("volatility", 0.02)

        # High spread = opportunity
        if spread > self.target_spread * 1.5:
            confidence = 0.7
            action = AgentAction.BUY if inventory < 0 else AgentAction.SELL
            reason = f"Wide spread {spread:.3%} - market making opportunity"

        # Inventory management
        elif abs(inventory) > self.max_inventory * 0.8:
            confidence = 0.75
            action = AgentAction.SELL if inventory > 0 else AgentAction.BUY
            reason = f"Inventory {inventory} near limit - rebalancing"

        # Normal market making
        elif spread > self.target_spread:
            confidence = 0.6
            action = AgentAction.HOLD
            reason = "Normal spread - maintaining quotes"

        else:
            confidence = 0.5
            action = AgentAction.HOLD
            reason = "Spread too tight for profitable market making"

        decision = AgentDecision(action, confidence, reason)
        self.decisions.append(decision)
        return decision


class SwarmController:
    """
    Coordinates multiple agents to make collective decisions
    """

    def __init__(self):
        self.agents: Dict[str, BaseAgent] = {
            "execution": ExecutionAgent(),
            "risk": RiskAgent(),
            "arbitrage": ArbitrageAgent(),
            "market_making": MarketMakingAgent()
        }

        self.coordination_mode = "weighted_voting"  # weighted_voting, hierarchical, consensus
        self.min_confidence = 0.6
        self.enabled = False

        # Agent weights (risk has highest priority)
        self.agent_weights = {
            "risk": 0.40,  # Risk management is priority
            "execution": 0.30,
            "arbitrage": 0.20,
            "market_making": 0.10
        }

        logger.info(f"SwarmController initialized with {len(self.agents)} agents")

    def get_swarm_decision(self, market_state: Dict) -> Tuple[AgentAction, float, Dict]:
        """
        Get collective decision from all agents

        Args:
            market_state: Current market and portfolio state

        Returns:
            action: Consensus action
            confidence: Overall confidence
            details: Decision details from each agent
        """
        if not self.enabled:
            return AgentAction.HOLD, 0.0, {"reason": "Swarm disabled"}

        # Get decisions from all enabled agents
        agent_decisions = {}
        for agent_name, agent in self.agents.items():
            if agent.enabled:
                try:
                    decision = agent.decide(market_state)
                    agent_decisions[agent_name] = decision
                except Exception as e:
                    logger.error(f"Error getting decision from {agent_name}: {e}")

        if not agent_decisions:
            return AgentAction.HOLD, 0.0, {"reason": "No agents available"}

        # Coordination strategy
        if self.coordination_mode == "weighted_voting":
            action, confidence = self._weighted_voting(agent_decisions)
        elif self.coordination_mode == "hierarchical":
            action, confidence = self._hierarchical_decision(agent_decisions)
        elif self.coordination_mode == "consensus":
            action, confidence = self._consensus_decision(agent_decisions)
        else:
            action, confidence = AgentAction.HOLD, 0.0

        # Compile details
        details = {
            agent_name: {
                "action": decision.action.name,
                "confidence": decision.confidence,
                "reason": decision.reason
            }
            for agent_name, decision in agent_decisions.items()
        }
        details["coordination_mode"] = self.coordination_mode
        details["final_confidence"] = confidence

        return action, confidence, details

    def _weighted_voting(self, decisions: Dict[str, AgentDecision]) -> Tuple[AgentAction, float]:
        """Weighted voting based on agent priorities"""
        # Risk agent has veto power for SELL signals
        if "risk" in decisions:
            risk_decision = decisions["risk"]
            if risk_decision.action in [AgentAction.SELL, AgentAction.REDUCE_POSITION]:
                if risk_decision.confidence > 0.7:
                    return risk_decision.action, risk_decision.confidence

        # Calculate weighted scores for each action
        action_scores = {}
        for agent_name, decision in decisions.items():
            weight = self.agent_weights.get(agent_name, 0.1)
            score = decision.confidence * weight

            if decision.action not in action_scores:
                action_scores[decision.action] = 0
            action_scores[decision.action] += score

        # Select action with highest score
        if action_scores:
            best_action = max(action_scores, key=action_scores.get)
            confidence = action_scores[best_action]
            return best_action, min(confidence, 1.0)

        return AgentAction.HOLD, 0.0

    def _hierarchical_decision(self, decisions: Dict[str, AgentDecision]) -> Tuple[AgentAction, float]:
        """Hierarchical decision: risk > execution > arbitrage > market_making"""
        priority_order = ["risk", "execution", "arbitrage", "market_making"]

        for agent_name in priority_order:
            if agent_name in decisions:
                decision = decisions[agent_name]
                if decision.confidence >= self.min_confidence:
                    return decision.action, decision.confidence

        return AgentAction.HOLD, 0.0

    def _consensus_decision(self, decisions: Dict[str, AgentDecision]) -> Tuple[AgentAction, float]:
        """All agents must agree (or at least 75%)"""
        if not decisions:
            return AgentAction.HOLD, 0.0

        # Count action votes
        action_votes = {}
        total_confidence = 0

        for decision in decisions.values():
            action_votes[decision.action] = action_votes.get(decision.action, 0) + 1
            total_confidence += decision.confidence

        # Check for majority (75% or more)
        total_agents = len(decisions)
        for action, votes in action_votes.items():
            if votes / total_agents >= 0.75:
                avg_confidence = total_confidence / total_agents
                return action, avg_confidence

        # No consensus - HOLD
        return AgentAction.HOLD, 0.0

    def enable_agent(self, agent_name: str):
        """Enable a specific agent"""
        if agent_name in self.agents:
            self.agents[agent_name].enabled = True
            logger.info(f"Enabled {agent_name}")

    def disable_agent(self, agent_name: str):
        """Disable a specific agent"""
        if agent_name in self.agents:
            self.agents[agent_name].enabled = False
            logger.info(f"Disabled {agent_name}")

    def get_status(self) -> Dict:
        """Get swarm status"""
        return {
            "enabled": self.enabled,
            "coordination_mode": self.coordination_mode,
            "agents": {
                agent_name: {
                    "enabled": agent.enabled,
                    "performance": agent.performance,
                    "recent_decisions": len(agent.decisions)
                }
                for agent_name, agent in self.agents.items()
            }
        }

    def get_recent_decisions(self, limit: int = 10) -> List[Dict]:
        """Get recent decisions from all agents"""
        all_decisions = []

        for agent_name, agent in self.agents.items():
            for decision in agent.decisions[-limit:]:
                all_decisions.append({
                    "agent": agent_name,
                    "action": decision.action.name,
                    "confidence": decision.confidence,
                    "reason": decision.reason,
                    "timestamp": decision.timestamp.isoformat()
                })

        # Sort by timestamp
        all_decisions.sort(key=lambda x: x["timestamp"], reverse=True)
        return all_decisions[:limit]


# Global swarm controller instance
swarm_controller: Optional[SwarmController] = None


def get_swarm_controller() -> SwarmController:
    """Get or create global swarm controller instance"""
    global swarm_controller
    if swarm_controller is None:
        swarm_controller = SwarmController()
    return swarm_controller
