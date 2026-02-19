"""
Rollout Manager - Gradual mainnet deployment with safety checks

Features:
- Phased capital deployment ($100 → $1k → $10k)
- Automatic phase progression based on performance
- Safety checks and circuit breakers per phase
- Performance tracking and reporting
- Rollback capability if issues detected
"""

import logging
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum

logger = logging.getLogger(__name__)


class RolloutPhase(Enum):
    """Deployment phases"""
    PHASE_0 = "phase_0"  # Pre-launch ($0)
    PHASE_1 = "phase_1"  # Initial ($100)
    PHASE_2 = "phase_2"  # Scale up ($1,000)
    PHASE_3 = "phase_3"  # Production ($10,000)
    PAUSED = "paused"    # Temporarily stopped
    ROLLBACK = "rollback"  # Issues detected, rolling back


@dataclass
class PhaseConfig:
    """Configuration for a rollout phase"""
    phase: RolloutPhase
    total_capital: float
    min_runtime_hours: int  # Minimum time before next phase
    success_criteria: Dict  # Criteria to advance to next phase
    max_drawdown_pct: float  # Max drawdown before pause
    max_daily_loss_pct: float  # Max daily loss before pause
    required_win_rate: float  # Minimum win rate to continue
    required_trades: int  # Minimum trades before evaluation


@dataclass
class PhaseMetrics:
    """Performance metrics for a phase"""
    phase: RolloutPhase
    start_time: datetime
    end_time: Optional[datetime] = None
    total_trades: int = 0
    winning_trades: int = 0
    total_pnl: float = 0.0
    max_drawdown: float = 0.0
    peak_value: float = 0.0
    current_value: float = 0.0
    daily_returns: List[float] = field(default_factory=list)
    issues_detected: List[str] = field(default_factory=list)


class RolloutManager:
    """
    Manages gradual mainnet deployment.

    Deployment Strategy:
    1. Phase 1: Start with $100, monitor for 1 week
    2. Phase 2: Increase to $1,000 if Phase 1 successful
    3. Phase 3: Increase to $10,000 if Phase 2 successful

    Each phase has:
    - Minimum runtime before evaluation
    - Success criteria to advance
    - Safety checks and circuit breakers
    """

    # Phase configurations
    PHASE_CONFIGS = {
        RolloutPhase.PHASE_1: PhaseConfig(
            phase=RolloutPhase.PHASE_1,
            total_capital=100.0,
            min_runtime_hours=168,  # 1 week
            success_criteria={
                "min_trades": 20,
                "min_win_rate": 0.50,
                "max_drawdown": 0.15,
                "min_sharpe": 0.5,
                "no_critical_errors": True
            },
            max_drawdown_pct=0.20,  # 20% max drawdown
            max_daily_loss_pct=0.10,  # 10% max daily loss
            required_win_rate=0.45,
            required_trades=10
        ),
        RolloutPhase.PHASE_2: PhaseConfig(
            phase=RolloutPhase.PHASE_2,
            total_capital=1000.0,
            min_runtime_hours=168,  # 1 week
            success_criteria={
                "min_trades": 50,
                "min_win_rate": 0.52,
                "max_drawdown": 0.12,
                "min_sharpe": 0.8,
                "no_critical_errors": True
            },
            max_drawdown_pct=0.15,  # 15% max drawdown
            max_daily_loss_pct=0.08,  # 8% max daily loss
            required_win_rate=0.48,
            required_trades=20
        ),
        RolloutPhase.PHASE_3: PhaseConfig(
            phase=RolloutPhase.PHASE_3,
            total_capital=10000.0,
            min_runtime_hours=336,  # 2 weeks
            success_criteria={
                "min_trades": 100,
                "min_win_rate": 0.55,
                "max_drawdown": 0.10,
                "min_sharpe": 1.0,
                "no_critical_errors": True
            },
            max_drawdown_pct=0.12,  # 12% max drawdown
            max_daily_loss_pct=0.06,  # 6% max daily loss
            required_win_rate=0.50,
            required_trades=50
        )
    }

    def __init__(
        self,
        alert_manager=None,
        auto_advance: bool = False
    ):
        """
        Initialize rollout manager.

        Args:
            alert_manager: AlertManager for notifications
            auto_advance: Automatically advance phases (use with caution)
        """
        self.alert_manager = alert_manager
        self.auto_advance = auto_advance

        # Current state
        self.current_phase = RolloutPhase.PHASE_0
        self.phase_history: List[PhaseMetrics] = []
        self.current_metrics: Optional[PhaseMetrics] = None

        logger.info("Rollout Manager initialized")
        logger.info(f"   Auto-advance: {auto_advance}")

    def start_phase(self, phase: RolloutPhase) -> bool:
        """
        Start a deployment phase.

        Args:
            phase: Phase to start

        Returns:
            True if started successfully
        """
        if phase not in self.PHASE_CONFIGS:
            logger.error(f"Invalid phase: {phase}")
            return False

        config = self.PHASE_CONFIGS[phase]

        # Save previous phase metrics
        if self.current_metrics:
            self.current_metrics.end_time = datetime.now()
            self.phase_history.append(self.current_metrics)

        # Initialize new phase
        self.current_phase = phase
        self.current_metrics = PhaseMetrics(
            phase=phase,
            start_time=datetime.now(),
            current_value=config.total_capital,
            peak_value=config.total_capital
        )

        logger.info(f"Starting {phase.value}")
        logger.info(f"   Capital: ${config.total_capital:,.2f}")
        logger.info(f"   Min runtime: {config.min_runtime_hours} hours")
        logger.info(f"   Required trades: {config.required_trades}")
        logger.info(f"   Required win rate: {config.required_win_rate:.1%}")

        # Send alert
        if self.alert_manager:
            self.alert_manager.send_alert(
                level="INFO",
                title=f"Rollout {phase.value.upper()} Started",
                message=f"Deployed ${config.total_capital:,.2f} to production",
                metadata={"phase": phase.value, "capital": config.total_capital}
            )

        return True

    def record_trade(self, pnl: float, success: bool) -> None:
        """Record a trade result."""
        if not self.current_metrics:
            logger.warning("No active phase to record trade")
            return

        self.current_metrics.total_trades += 1
        if success:
            self.current_metrics.winning_trades += 1

        self.current_metrics.total_pnl += pnl
        self.current_metrics.current_value += pnl

        # Update peak and drawdown
        if self.current_metrics.current_value > self.current_metrics.peak_value:
            self.current_metrics.peak_value = self.current_metrics.current_value

        drawdown = (
            (self.current_metrics.peak_value - self.current_metrics.current_value)
            / self.current_metrics.peak_value
        )
        if drawdown > self.current_metrics.max_drawdown:
            self.current_metrics.max_drawdown = drawdown

        # Check safety limits
        self._check_safety_limits()

    def record_daily_return(self, return_pct: float) -> None:
        """Record daily return."""
        if not self.current_metrics:
            return

        self.current_metrics.daily_returns.append(return_pct)

        # Check for excessive loss
        config = self.PHASE_CONFIGS[self.current_phase]
        if return_pct < -config.max_daily_loss_pct:
            self._pause_phase(f"Daily loss exceeded: {return_pct*100:.2f}%")

    def _check_safety_limits(self) -> None:
        """Check if safety limits exceeded."""
        if not self.current_metrics:
            return

        config = self.PHASE_CONFIGS[self.current_phase]

        # Check drawdown
        if self.current_metrics.max_drawdown > config.max_drawdown_pct:
            self._pause_phase(
                f"Drawdown exceeded: {self.current_metrics.max_drawdown*100:.1f}% "
                f"> {config.max_drawdown_pct*100:.0f}%"
            )

        # Check win rate (if enough trades)
        if self.current_metrics.total_trades >= config.required_trades:
            win_rate = (
                self.current_metrics.winning_trades /
                self.current_metrics.total_trades
            )

            if win_rate < config.required_win_rate:
                self._pause_phase(
                    f"Win rate too low: {win_rate:.1%} < {config.required_win_rate:.1%}"
                )

    def _pause_phase(self, reason: str) -> None:
        """Pause current phase due to issues."""
        logger.warning(f"PAUSING {self.current_phase.value}: {reason}")

        if self.current_metrics:
            self.current_metrics.issues_detected.append(reason)

        self.current_phase = RolloutPhase.PAUSED

        # Send critical alert
        if self.alert_manager:
            self.alert_manager.send_alert(
                level="CRITICAL",
                title="Rollout Paused",
                message=f"Phase paused due to: {reason}",
                metadata={"reason": reason}
            )

    def can_advance_phase(self) -> Tuple[bool, str]:
        """
        Check if can advance to next phase.

        Returns:
            (can_advance, reason)
        """
        if not self.current_metrics:
            return False, "No active phase"

        if self.current_phase == RolloutPhase.PHASE_3:
            return False, "Already at final phase"

        if self.current_phase == RolloutPhase.PAUSED:
            return False, "Phase is paused"

        config = self.PHASE_CONFIGS[self.current_phase]

        # Check minimum runtime
        runtime = datetime.now() - self.current_metrics.start_time
        if runtime < timedelta(hours=config.min_runtime_hours):
            hours_left = config.min_runtime_hours - (runtime.total_seconds() / 3600)
            return False, f"Minimum runtime not met ({hours_left:.1f}h remaining)"

        # Check success criteria
        criteria = config.success_criteria

        # Min trades
        if self.current_metrics.total_trades < criteria["min_trades"]:
            return False, f"Not enough trades: {self.current_metrics.total_trades} < {criteria['min_trades']}"

        # Win rate
        win_rate = (
            self.current_metrics.winning_trades /
            self.current_metrics.total_trades
            if self.current_metrics.total_trades > 0 else 0
        )
        if win_rate < criteria["min_win_rate"]:
            return False, f"Win rate too low: {win_rate:.1%} < {criteria['min_win_rate']:.1%}"

        # Drawdown
        if self.current_metrics.max_drawdown > criteria["max_drawdown"]:
            return False, f"Drawdown too high: {self.current_metrics.max_drawdown:.1%}"

        # Sharpe ratio (mock calculation)
        if len(self.current_metrics.daily_returns) > 0:
            import numpy as np
            returns = np.array(self.current_metrics.daily_returns)
            if returns.std() > 0:
                sharpe = (returns.mean() / returns.std()) * np.sqrt(365)
            else:
                sharpe = 0
        else:
            sharpe = 0

        if sharpe < criteria["min_sharpe"]:
            return False, f"Sharpe ratio too low: {sharpe:.2f} < {criteria['min_sharpe']:.2f}"

        # All criteria met
        return True, "All success criteria met"

    def advance_phase(self) -> bool:
        """
        Advance to next phase.

        Returns:
            True if advanced successfully
        """
        can_advance, reason = self.can_advance_phase()

        if not can_advance:
            logger.warning(f"Cannot advance phase: {reason}")
            return False

        # Determine next phase
        phase_order = [
            RolloutPhase.PHASE_1,
            RolloutPhase.PHASE_2,
            RolloutPhase.PHASE_3
        ]

        current_idx = phase_order.index(self.current_phase)
        if current_idx >= len(phase_order) - 1:
            logger.info("Already at final phase")
            return False

        next_phase = phase_order[current_idx + 1]

        logger.info(f"Advancing from {self.current_phase.value} to {next_phase.value}")
        logger.info(f"   Reason: {reason}")

        return self.start_phase(next_phase)

    def get_phase_report(self) -> Dict:
        """Get current phase performance report."""
        if not self.current_metrics:
            return {"status": "No active phase"}

        metrics = self.current_metrics
        report_phase = metrics.phase
        config = self.PHASE_CONFIGS[report_phase]

        win_rate = (
            metrics.winning_trades / metrics.total_trades
            if metrics.total_trades > 0 else 0
        )

        runtime = datetime.now() - metrics.start_time
        runtime_hours = runtime.total_seconds() / 3600

        return {
            "phase": report_phase.value,
            "status": "active" if self.current_phase not in [RolloutPhase.PAUSED, RolloutPhase.ROLLBACK] else self.current_phase.value,
            "capital": config.total_capital,
            "current_value": metrics.current_value,
            "total_pnl": metrics.total_pnl,
            "return_pct": (metrics.current_value / config.total_capital - 1) * 100,
            "runtime_hours": runtime_hours,
            "min_runtime_hours": config.min_runtime_hours,
            "total_trades": metrics.total_trades,
            "win_rate": win_rate,
            "max_drawdown": metrics.max_drawdown,
            "issues": metrics.issues_detected,
            "can_advance": self.can_advance_phase()[0]
        }

    def print_status(self) -> None:
        """Print current rollout status."""
        report = self.get_phase_report()

        if "status" in report and report["status"] == "No active phase":
            print("\n⊘ No active rollout phase")
            return

        print("\n" + "="*70)
        print(f"ROLLOUT STATUS: {report['phase'].upper()}")
        print("="*70)

        status_emoji = {
            "active": "🟢",
            "paused": "⏸️",
            "rollback": "⏪"
        }
        print(f"Status: {status_emoji.get(report['status'], '⚫')} {report['status'].upper()}")

        print(f"\nCapital:")
        print(f"   Deployed: ${report['capital']:,.2f}")
        print(f"   Current:  ${report['current_value']:,.2f}")
        print(f"   P&L:      ${report['total_pnl']:+,.2f} ({report['return_pct']:+.2f}%)")

        print(f"\nPerformance:")
        print(f"   Trades:    {report['total_trades']}")
        print(f"   Win Rate:  {report['win_rate']:.1%}")
        print(f"   Drawdown:  {report['max_drawdown']:.1%}")

        print(f"\nProgress:")
        print(f"   Runtime:   {report['runtime_hours']:.1f}h / {report['min_runtime_hours']}h")
        progress_pct = min(100, (report['runtime_hours'] / report['min_runtime_hours']) * 100)
        bar_length = 40
        filled = int(bar_length * progress_pct / 100)
        bar = "█" * filled + "░" * (bar_length - filled)
        print(f"   Progress:  [{bar}] {progress_pct:.1f}%")

        if report['issues']:
            print(f"\n⚠️  Issues Detected:")
            for issue in report['issues']:
                print(f"   - {issue}")

        if report['can_advance']:
            print(f"\n✅ Ready to advance to next phase!")
        else:
            print(f"\n⏳ Continue monitoring...")

        print("="*70 + "\n")


# Example usage
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    print("\n" + "="*70)
    print("ROLLOUT MANAGER TEST")
    print("="*70)

    # Initialize rollout manager
    manager = RolloutManager(auto_advance=False)

    # Start Phase 1
    print("\n--- Starting Phase 1 ---")
    manager.start_phase(RolloutPhase.PHASE_1)

    # Simulate some trades
    print("\n--- Simulating Trades ---")
    import random
    random.seed(42)

    for i in range(25):
        pnl = random.gauss(2.0, 5.0)
        success = pnl > 0
        manager.record_trade(pnl, success)

    # Record daily returns
    for i in range(7):
        daily_return = random.gauss(0.01, 0.03)
        manager.record_daily_return(daily_return)

    # Print status
    manager.print_status()

    # Check if can advance
    can_advance, reason = manager.can_advance_phase()
    print(f"\nCan advance? {can_advance}")
    print(f"Reason: {reason}")

    print("\n✅ Rollout Manager ready!")
