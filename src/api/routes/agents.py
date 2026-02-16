"""
Agent Control Endpoints
"""
from fastapi import APIRouter, HTTPException, status
from typing import Dict, List, Optional
from datetime import datetime
from pydantic import BaseModel, Field

router = APIRouter()


# ============================================================
# Schemas
# ============================================================

class AgentConfig(BaseModel):
    """Agent configuration schema."""
    initial_capital: float = Field(10000.0, gt=0, description="Initial capital in USD")
    paper_trading: bool = Field(True, description="Paper trading mode")
    check_interval_seconds: int = Field(5, gt=0, le=60, description="Check interval")
    max_daily_loss: float = Field(500.0, gt=0, description="Max daily loss limit")
    max_position_size: float = Field(0.2, gt=0, le=1, description="Max position size as fraction")
    enabled_strategies: List[str] = Field(default_factory=list, description="Enabled strategy list")
    send_alerts: bool = Field(False, description="Enable alert notifications")


class AgentStatus(BaseModel):
    """Agent status response."""
    agent_id: str
    state: str
    uptime_seconds: float
    portfolio_value: float
    total_pnl: float
    daily_pnl: float
    total_trades: int
    active_positions: int
    enabled_strategies: List[str]
    last_update: str


class AgentMetrics(BaseModel):
    """Agent performance metrics."""
    sharpe_ratio: float
    max_drawdown: float
    win_rate: float
    total_return_percent: float
    total_trades: int
    winning_trades: int
    losing_trades: int
    avg_win: float
    avg_loss: float
    profit_factor: float


class StrategyControl(BaseModel):
    """Strategy enable/disable request."""
    strategy_name: str
    enabled: bool


class RiskLimitUpdate(BaseModel):
    """Risk limit update request."""
    max_daily_loss: Optional[float] = None
    max_position_size: Optional[float] = None
    max_drawdown: Optional[float] = None


# ============================================================
# In-memory agent registry (replace with database in production)
# ============================================================

_agents: Dict[str, Dict] = {}


# ============================================================
# Endpoints
# ============================================================

@router.post("/", status_code=status.HTTP_201_CREATED)
async def create_agent(config: AgentConfig) -> Dict:
    """
    Create a new trading agent.

    Creates and initializes a new autonomous trading agent with the specified configuration.
    """
    import uuid
    agent_id = str(uuid.uuid4())

    # Create agent (placeholder)
    _agents[agent_id] = {
        "id": agent_id,
        "config": config.dict(),
        "state": "stopped",
        "created_at": datetime.utcnow().isoformat(),
        "portfolio_value": config.initial_capital,
        "total_pnl": 0.0,
        "daily_pnl": 0.0,
        "trades": [],
        "positions": {},
        "metrics": {}
    }

    return {
        "agent_id": agent_id,
        "status": "created",
        "message": "Agent created successfully"
    }


@router.get("/")
async def list_agents() -> List[Dict]:
    """
    List all agents.

    Returns a list of all trading agents with their current status.
    """
    return [
        {
            "agent_id": agent_id,
            "state": agent["state"],
            "portfolio_value": agent["portfolio_value"],
            "total_pnl": agent["total_pnl"],
            "created_at": agent["created_at"]
        }
        for agent_id, agent in _agents.items()
    ]


@router.get("/{agent_id}")
async def get_agent(agent_id: str) -> AgentStatus:
    """
    Get agent status.

    Returns detailed status information for the specified agent.
    """
    if agent_id not in _agents:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Agent {agent_id} not found"
        )

    agent = _agents[agent_id]

    return AgentStatus(
        agent_id=agent_id,
        state=agent["state"],
        uptime_seconds=0.0,  # TODO: Calculate actual uptime
        portfolio_value=agent["portfolio_value"],
        total_pnl=agent["total_pnl"],
        daily_pnl=agent["daily_pnl"],
        total_trades=len(agent["trades"]),
        active_positions=len(agent["positions"]),
        enabled_strategies=agent["config"]["enabled_strategies"],
        last_update=datetime.utcnow().isoformat()
    )


@router.post("/{agent_id}/start")
async def start_agent(agent_id: str) -> Dict:
    """
    Start an agent.

    Starts the autonomous trading agent to begin trading.
    """
    if agent_id not in _agents:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Agent {agent_id} not found"
        )

    agent = _agents[agent_id]

    if agent["state"] == "running":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Agent is already running"
        )

    # Start agent (placeholder)
    agent["state"] = "running"
    agent["started_at"] = datetime.utcnow().isoformat()

    return {
        "agent_id": agent_id,
        "status": "started",
        "message": "Agent started successfully"
    }


@router.post("/{agent_id}/stop")
async def stop_agent(agent_id: str) -> Dict:
    """
    Stop an agent.

    Stops the autonomous trading agent gracefully.
    """
    if agent_id not in _agents:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Agent {agent_id} not found"
        )

    agent = _agents[agent_id]

    if agent["state"] != "running":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Agent is not running"
        )

    # Stop agent (placeholder)
    agent["state"] = "stopped"
    agent["stopped_at"] = datetime.utcnow().isoformat()

    return {
        "agent_id": agent_id,
        "status": "stopped",
        "message": "Agent stopped successfully"
    }


@router.post("/{agent_id}/pause")
async def pause_agent(agent_id: str) -> Dict:
    """
    Pause an agent.

    Temporarily pauses the agent without stopping it completely.
    """
    if agent_id not in _agents:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Agent {agent_id} not found"
        )

    agent = _agents[agent_id]

    if agent["state"] != "running":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Agent must be running to pause"
        )

    agent["state"] = "paused"

    return {
        "agent_id": agent_id,
        "status": "paused",
        "message": "Agent paused successfully"
    }


@router.post("/{agent_id}/resume")
async def resume_agent(agent_id: str) -> Dict:
    """
    Resume a paused agent.

    Resumes trading after a pause.
    """
    if agent_id not in _agents:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Agent {agent_id} not found"
        )

    agent = _agents[agent_id]

    if agent["state"] != "paused":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Agent must be paused to resume"
        )

    agent["state"] = "running"

    return {
        "agent_id": agent_id,
        "status": "resumed",
        "message": "Agent resumed successfully"
    }


@router.delete("/{agent_id}")
async def delete_agent(agent_id: str) -> Dict:
    """
    Delete an agent.

    Permanently deletes the agent and its data.
    """
    if agent_id not in _agents:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Agent {agent_id} not found"
        )

    agent = _agents[agent_id]

    if agent["state"] == "running":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Cannot delete running agent. Stop it first."
        )

    del _agents[agent_id]

    return {
        "agent_id": agent_id,
        "status": "deleted",
        "message": "Agent deleted successfully"
    }


@router.get("/{agent_id}/metrics")
async def get_agent_metrics(agent_id: str) -> AgentMetrics:
    """
    Get agent performance metrics.

    Returns detailed performance metrics for the agent.
    """
    if agent_id not in _agents:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Agent {agent_id} not found"
        )

    agent = _agents[agent_id]

    # Calculate metrics (placeholder)
    trades = agent["trades"]
    winning_trades = [t for t in trades if t.get("pnl", 0) > 0]
    losing_trades = [t for t in trades if t.get("pnl", 0) < 0]

    total_return = (agent["total_pnl"] / agent["config"]["initial_capital"]) * 100

    return AgentMetrics(
        sharpe_ratio=0.0,  # TODO: Calculate
        max_drawdown=0.0,  # TODO: Calculate
        win_rate=len(winning_trades) / len(trades) if trades else 0.0,
        total_return_percent=total_return,
        total_trades=len(trades),
        winning_trades=len(winning_trades),
        losing_trades=len(losing_trades),
        avg_win=sum(t.get("pnl", 0) for t in winning_trades) / len(winning_trades) if winning_trades else 0.0,
        avg_loss=sum(t.get("pnl", 0) for t in losing_trades) / len(losing_trades) if losing_trades else 0.0,
        profit_factor=1.0  # TODO: Calculate
    )


@router.post("/{agent_id}/strategies")
async def control_strategy(agent_id: str, control: StrategyControl) -> Dict:
    """
    Enable or disable a strategy.

    Controls which strategies the agent uses for trading.
    """
    if agent_id not in _agents:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Agent {agent_id} not found"
        )

    agent = _agents[agent_id]
    enabled_strategies = agent["config"]["enabled_strategies"]

    if control.enabled:
        if control.strategy_name not in enabled_strategies:
            enabled_strategies.append(control.strategy_name)
    else:
        if control.strategy_name in enabled_strategies:
            enabled_strategies.remove(control.strategy_name)

    return {
        "agent_id": agent_id,
        "strategy": control.strategy_name,
        "enabled": control.enabled,
        "message": f"Strategy {'enabled' if control.enabled else 'disabled'}"
    }


@router.put("/{agent_id}/risk-limits")
async def update_risk_limits(agent_id: str, limits: RiskLimitUpdate) -> Dict:
    """
    Update risk limits.

    Modifies the agent's risk management parameters.
    """
    if agent_id not in _agents:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Agent {agent_id} not found"
        )

    agent = _agents[agent_id]
    config = agent["config"]

    if limits.max_daily_loss is not None:
        config["max_daily_loss"] = limits.max_daily_loss

    if limits.max_position_size is not None:
        config["max_position_size"] = limits.max_position_size

    if limits.max_drawdown is not None:
        config["max_drawdown"] = limits.max_drawdown

    return {
        "agent_id": agent_id,
        "message": "Risk limits updated successfully",
        "updated_limits": {
            k: v for k, v in limits.dict().items() if v is not None
        }
    }


@router.get("/{agent_id}/strategies/performance")
async def get_strategy_performance(agent_id: str) -> List[Dict]:
    """
    Get performance by strategy.

    Returns performance metrics broken down by strategy.
    """
    if agent_id not in _agents:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Agent {agent_id} not found"
        )

    agent = _agents[agent_id]
    trades = agent["trades"]

    # Group trades by strategy
    strategy_trades = {}
    for trade in trades:
        strategy = trade.get("strategy", "unknown")
        if strategy not in strategy_trades:
            strategy_trades[strategy] = []
        strategy_trades[strategy].append(trade)

    # Calculate metrics per strategy
    performance = []
    for strategy, strat_trades in strategy_trades.items():
        total_pnl = sum(t.get("pnl", 0) for t in strat_trades)
        winning = [t for t in strat_trades if t.get("pnl", 0) > 0]

        performance.append({
            "strategy": strategy,
            "total_trades": len(strat_trades),
            "total_pnl": total_pnl,
            "win_rate": len(winning) / len(strat_trades) if strat_trades else 0.0,
            "avg_pnl_per_trade": total_pnl / len(strat_trades) if strat_trades else 0.0
        })

    return performance
