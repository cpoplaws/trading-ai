"""
Alert Manager - Production monitoring and alerting

Features:
- Discord webhook integration
- Telegram bot integration
- Multi-level alerting (INFO, WARNING, ERROR, CRITICAL)
- Alert throttling and rate limiting
- Performance metrics tracking
- System health monitoring
"""

import os
import logging
import requests
import time
from typing import Dict, List, Optional, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import json

logger = logging.getLogger(__name__)


class AlertLevel(Enum):
    """Alert severity levels"""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class AlertChannel(Enum):
    """Alert delivery channels"""
    DISCORD = "discord"
    TELEGRAM = "telegram"
    LOG = "log"
    EMAIL = "email"


@dataclass
class Alert:
    """Alert message"""
    level: AlertLevel
    title: str
    message: str
    timestamp: datetime = field(default_factory=datetime.now)
    channel: AlertChannel = AlertChannel.LOG
    metadata: Dict = field(default_factory=dict)


@dataclass
class AlertThrottle:
    """Throttle configuration for alert types"""
    alert_type: str
    max_count: int  # Max alerts per window
    window_seconds: int  # Time window
    last_sent: datetime = field(default_factory=datetime.now)
    count_in_window: int = 0


class AlertManager:
    """
    Production alerting and monitoring system.

    Features:
    - Multi-channel alerts (Discord, Telegram, Logs)
    - Alert throttling to prevent spam
    - Severity-based routing
    - Performance metric tracking
    - Health check monitoring
    """

    def __init__(
        self,
        discord_webhook_url: Optional[str] = None,
        telegram_bot_token: Optional[str] = None,
        telegram_chat_id: Optional[str] = None,
        enable_console: bool = True,
        throttle_config: Dict[str, AlertThrottle] = None
    ):
        """
        Initialize alert manager.

        Args:
            discord_webhook_url: Discord webhook URL
            telegram_bot_token: Telegram bot token
            telegram_chat_id: Telegram chat ID
            enable_console: Whether to print to console
            throttle_config: Alert throttling configuration
        """
        # Get credentials from environment if not provided
        self.discord_webhook = discord_webhook_url or os.getenv("DISCORD_WEBHOOK_URL")
        self.telegram_token = telegram_bot_token or os.getenv("TELEGRAM_BOT_TOKEN")
        self.telegram_chat_id = telegram_chat_id or os.getenv("TELEGRAM_CHAT_ID")
        self.enable_console = enable_console

        # Throttle configuration
        self.throttle_config = throttle_config or {
            "trade_failed": AlertThrottle("trade_failed", 5, 300),  # 5 per 5min
            "api_error": AlertThrottle("api_error", 10, 300),
            "balance_low": AlertThrottle("balance_low", 1, 3600),  # 1 per hour
            "circuit_breaker": AlertThrottle("circuit_breaker", 1, 7200),  # 1 per 2h
        }

        # Alert history
        self.alert_history: List[Alert] = []
        self.max_history = 1000

        # Health metrics
        self.metrics = {
            "total_alerts": 0,
            "alerts_by_level": {level: 0 for level in AlertLevel},
            "alerts_by_channel": {channel: 0 for channel in AlertChannel},
            "last_alert_time": None,
            "throttled_alerts": 0,
        }

        logger.info("Alert Manager initialized")
        if self.discord_webhook:
            logger.info("   ✅ Discord webhook configured")
        if self.telegram_token and self.telegram_chat_id:
            logger.info("   ✅ Telegram bot configured")

    def send_alert(
        self,
        level: AlertLevel,
        title: str,
        message: str,
        channel: AlertChannel = AlertChannel.LOG,
        alert_type: str = "general",
        metadata: Dict = None
    ) -> bool:
        """
        Send an alert.

        Args:
            level: Alert severity level
            title: Alert title
            message: Alert message
            channel: Delivery channel
            alert_type: Alert type for throttling
            metadata: Additional metadata

        Returns:
            True if sent, False if throttled
        """
        # Check throttling
        if not self._check_throttle(alert_type):
            self.metrics["throttled_alerts"] += 1
            logger.debug(f"Alert throttled: {alert_type}")
            return False

        # Create alert
        alert = Alert(
            level=level,
            title=title,
            message=message,
            channel=channel,
            metadata=metadata or {}
        )

        # Update metrics
        self.metrics["total_alerts"] += 1
        self.metrics["alerts_by_level"][level] += 1
        self.metrics["alerts_by_channel"][channel] += 1
        self.metrics["last_alert_time"] = datetime.now()

        # Store in history
        self.alert_history.append(alert)
        if len(self.alert_history) > self.max_history:
            self.alert_history.pop(0)

        # Route to appropriate channel
        if channel == AlertChannel.DISCORD:
            self._send_discord(alert)
        elif channel == AlertChannel.TELEGRAM:
            self._send_telegram(alert)
        elif channel == AlertChannel.LOG:
            self._send_log(alert)

        # Always log
        self._send_log(alert)

        return True

    def _check_throttle(self, alert_type: str) -> bool:
        """Check if alert should be throttled."""
        if alert_type not in self.throttle_config:
            return True  # Not throttled

        throttle = self.throttle_config[alert_type]
        now = datetime.now()

        # Reset window if expired
        if (now - throttle.last_sent).total_seconds() > throttle.window_seconds:
            throttle.count_in_window = 0
            throttle.last_sent = now

        # Check if over limit
        if throttle.count_in_window >= throttle.max_count:
            return False

        # Increment count
        throttle.count_in_window += 1
        return True

    def _send_discord(self, alert: Alert) -> bool:
        """Send alert to Discord."""
        if not self.discord_webhook:
            return False

        try:
            # Color based on level
            colors = {
                AlertLevel.INFO: 3447003,      # Blue
                AlertLevel.WARNING: 16776960,  # Yellow
                AlertLevel.ERROR: 16711680,    # Red
                AlertLevel.CRITICAL: 10038562, # Dark red
            }

            embed = {
                "title": alert.title,
                "description": alert.message,
                "color": colors.get(alert.level, 0),
                "timestamp": alert.timestamp.isoformat(),
                "footer": {
                    "text": f"Trading AI System | {alert.level.value.upper()}"
                }
            }

            # Add metadata fields
            if alert.metadata:
                embed["fields"] = [
                    {"name": key, "value": str(value), "inline": True}
                    for key, value in alert.metadata.items()
                ]

            payload = {"embeds": [embed]}

            response = requests.post(
                self.discord_webhook,
                json=payload,
                timeout=5
            )

            if response.status_code != 204:
                logger.error(f"Discord webhook failed: {response.status_code}")
                return False

            return True

        except Exception as e:
            logger.error(f"Failed to send Discord alert: {e}")
            return False

    def _send_telegram(self, alert: Alert) -> bool:
        """Send alert to Telegram."""
        if not self.telegram_token or not self.telegram_chat_id:
            return False

        try:
            # Format message with markdown
            emoji = {
                AlertLevel.INFO: "ℹ️",
                AlertLevel.WARNING: "⚠️",
                AlertLevel.ERROR: "❌",
                AlertLevel.CRITICAL: "🚨",
            }

            text = f"{emoji.get(alert.level, '')} **{alert.title}**\n\n{alert.message}"

            # Add metadata
            if alert.metadata:
                text += "\n\n**Details:**"
                for key, value in alert.metadata.items():
                    text += f"\n• {key}: {value}"

            url = f"https://api.telegram.org/bot{self.telegram_token}/sendMessage"
            payload = {
                "chat_id": self.telegram_chat_id,
                "text": text,
                "parse_mode": "Markdown"
            }

            response = requests.post(url, json=payload, timeout=5)

            if response.status_code != 200:
                logger.error(f"Telegram API failed: {response.status_code}")
                return False

            return True

        except Exception as e:
            logger.error(f"Failed to send Telegram alert: {e}")
            return False

    def _send_log(self, alert: Alert) -> None:
        """Log alert to console/file."""
        log_methods = {
            AlertLevel.INFO: logger.info,
            AlertLevel.WARNING: logger.warning,
            AlertLevel.ERROR: logger.error,
            AlertLevel.CRITICAL: logger.critical,
        }

        log_method = log_methods.get(alert.level, logger.info)
        log_method(f"[{alert.level.value.upper()}] {alert.title}: {alert.message}")

        if self.enable_console:
            emoji = {
                AlertLevel.INFO: "ℹ️",
                AlertLevel.WARNING: "⚠️",
                AlertLevel.ERROR: "❌",
                AlertLevel.CRITICAL: "🚨",
            }
            print(f"\n{emoji.get(alert.level, '')} [{alert.level.value.upper()}] {alert.title}")
            print(f"   {alert.message}")

    def alert_trade_success(self, instance_id: str, symbol: str, pnl: float, channel: AlertChannel = AlertChannel.LOG) -> None:
        """Send trade success alert."""
        self.send_alert(
            AlertLevel.INFO,
            "Trade Executed",
            f"Instance: {instance_id}\nSymbol: {symbol}\nP&L: ${pnl:.2f}",
            channel=channel,
            alert_type="trade_success",
            metadata={"instance": instance_id, "symbol": symbol, "pnl": pnl}
        )

    def alert_trade_failed(self, instance_id: str, symbol: str, reason: str, channel: AlertChannel = AlertChannel.DISCORD) -> None:
        """Send trade failure alert."""
        self.send_alert(
            AlertLevel.ERROR,
            "Trade Failed",
            f"Instance: {instance_id}\nSymbol: {symbol}\nReason: {reason}",
            channel=channel,
            alert_type="trade_failed",
            metadata={"instance": instance_id, "symbol": symbol, "reason": reason}
        )

    def alert_balance_low(self, chain: str, balance: float, threshold: float, channel: AlertChannel = AlertChannel.TELEGRAM) -> None:
        """Send low balance alert."""
        self.send_alert(
            AlertLevel.WARNING,
            "Low Balance Warning",
            f"Chain: {chain}\nBalance: ${balance:.2f}\nThreshold: ${threshold:.2f}",
            channel=channel,
            alert_type="balance_low",
            metadata={"chain": chain, "balance": balance, "threshold": threshold}
        )

    def alert_circuit_breaker(self, reason: str, loss_pct: float, channel: AlertChannel = AlertChannel.TELEGRAM) -> None:
        """Send circuit breaker alert."""
        self.send_alert(
            AlertLevel.CRITICAL,
            "Circuit Breaker Triggered",
            f"Trading halted!\nReason: {reason}\nLoss: {loss_pct:.1f}%",
            channel=channel,
            alert_type="circuit_breaker",
            metadata={"reason": reason, "loss_pct": loss_pct}
        )

    def alert_api_error(self, api: str, error: str, channel: AlertChannel = AlertChannel.LOG) -> None:
        """Send API error alert."""
        self.send_alert(
            AlertLevel.ERROR,
            "API Error",
            f"API: {api}\nError: {error}",
            channel=channel,
            alert_type="api_error",
            metadata={"api": api, "error": error}
        )

    def alert_performance_milestone(self, milestone: str, value: float, channel: AlertChannel = AlertChannel.DISCORD) -> None:
        """Send performance milestone alert."""
        self.send_alert(
            AlertLevel.INFO,
            "Performance Milestone",
            f"Milestone: {milestone}\nValue: {value}",
            channel=channel,
            alert_type="milestone",
            metadata={"milestone": milestone, "value": value}
        )

    def get_metrics(self) -> Dict:
        """Get alerting metrics."""
        return {
            **self.metrics,
            "history_size": len(self.alert_history),
            "throttle_configs": {
                name: {
                    "max_count": throttle.max_count,
                    "window_seconds": throttle.window_seconds,
                    "current_count": throttle.count_in_window
                }
                for name, throttle in self.throttle_config.items()
            }
        }

    def get_recent_alerts(self, count: int = 10, level: AlertLevel = None) -> List[Alert]:
        """Get recent alerts."""
        alerts = self.alert_history

        if level:
            alerts = [a for a in alerts if a.level == level]

        return alerts[-count:]

    def clear_history(self) -> None:
        """Clear alert history."""
        self.alert_history.clear()
        logger.info("Alert history cleared")


# Singleton instance
_alert_manager: Optional[AlertManager] = None


def get_alert_manager(**kwargs) -> AlertManager:
    """Get alert manager singleton."""
    global _alert_manager

    if _alert_manager is None:
        _alert_manager = AlertManager(**kwargs)

    return _alert_manager


# Example usage
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    print("\n" + "="*70)
    print("ALERT MANAGER TEST")
    print("="*70)

    # Initialize alert manager
    manager = AlertManager(enable_console=True)

    # Test 1: Different alert levels
    print("\n--- Test 1: Alert Levels ---")
    manager.send_alert(AlertLevel.INFO, "System Started", "Trading system initialized")
    manager.send_alert(AlertLevel.WARNING, "High Gas", "Gas price above 50 Gwei")
    manager.send_alert(AlertLevel.ERROR, "Trade Failed", "Order rejected by exchange")
    manager.send_alert(AlertLevel.CRITICAL, "Circuit Breaker", "Daily loss limit exceeded")

    # Test 2: Specialized alerts
    print("\n--- Test 2: Specialized Alerts ---")
    manager.alert_trade_success("mean_reversion_base_001", "WETH/USDC", 45.50)
    manager.alert_trade_failed("breakout_arb_002", "BTC/USDC", "Insufficient balance")
    manager.alert_balance_low("base", 50.0, 100.0)

    # Test 3: Throttling
    print("\n--- Test 3: Alert Throttling ---")
    for i in range(10):
        sent = manager.send_alert(
            AlertLevel.ERROR,
            "Trade Failed",
            f"Trade #{i+1} failed",
            alert_type="trade_failed"
        )
        print(f"   Alert {i+1}: {'Sent' if sent else 'Throttled'}")

    # Test 4: Metrics
    print("\n--- Test 4: Alert Metrics ---")
    metrics = manager.get_metrics()
    print(f"   Total alerts: {metrics['total_alerts']}")
    print(f"   Throttled alerts: {metrics['throttled_alerts']}")
    print(f"   Alerts by level:")
    for level, count in metrics['alerts_by_level'].items():
        print(f"      {level.value}: {count}")

    # Test 5: Recent alerts
    print("\n--- Test 5: Recent Alerts ---")
    recent = manager.get_recent_alerts(5)
    print(f"   Last 5 alerts:")
    for alert in recent:
        print(f"      [{alert.level.value}] {alert.title}")

    print("\n" + "="*70)
    print("✅ Alert Manager ready!")
    print("="*70)
