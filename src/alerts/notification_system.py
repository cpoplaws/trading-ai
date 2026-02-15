"""
Multi-Channel Notification System
Sends alerts via Email, SMS, Telegram, Discord, and Slack.
"""
import logging
from typing import List, Optional, Dict, Any
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
import os

logger = logging.getLogger(__name__)


class NotificationChannel(Enum):
    """Notification channels."""
    EMAIL = "email"
    SMS = "sms"
    TELEGRAM = "telegram"
    DISCORD = "discord"
    SLACK = "slack"


class AlertPriority(Enum):
    """Alert priority levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class AlertConfig:
    """Alert configuration."""
    # Email
    email_enabled: bool = False
    email_addresses: List[str] = None
    smtp_server: str = "smtp.gmail.com"
    smtp_port: int = 587
    smtp_username: str = ""
    smtp_password: str = ""

    # SMS (Twilio)
    sms_enabled: bool = False
    phone_numbers: List[str] = None
    twilio_account_sid: str = ""
    twilio_auth_token: str = ""
    twilio_from_number: str = ""

    # Telegram
    telegram_enabled: bool = False
    telegram_bot_token: str = ""
    telegram_chat_ids: List[str] = None

    # Discord
    discord_enabled: bool = False
    discord_webhook_url: str = ""

    # Slack
    slack_enabled: bool = False
    slack_webhook_url: str = ""


@dataclass
class Alert:
    """Alert notification."""
    title: str
    message: str
    priority: AlertPriority
    channels: List[NotificationChannel]
    timestamp: datetime = None
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()
        if self.metadata is None:
            self.metadata = {}


class NotificationSystem:
    """
    Multi-channel notification system.

    Sends alerts via multiple channels: Email, SMS, Telegram, Discord, Slack.
    """

    def __init__(self, config: Optional[AlertConfig] = None):
        """
        Initialize notification system.

        Args:
            config: Alert configuration
        """
        self.config = config or AlertConfig()
        self.alert_history = []

        # Load from environment if not provided
        self._load_from_env()

        logger.info("Notification system initialized")

    def _load_from_env(self):
        """Load configuration from environment variables."""
        # Email
        if os.getenv("EMAIL_ENABLED"):
            self.config.email_enabled = True
            self.config.smtp_username = os.getenv("SMTP_USERNAME", "")
            self.config.smtp_password = os.getenv("SMTP_PASSWORD", "")

        # SMS
        if os.getenv("SMS_ENABLED"):
            self.config.sms_enabled = True
            self.config.twilio_account_sid = os.getenv("TWILIO_ACCOUNT_SID", "")
            self.config.twilio_auth_token = os.getenv("TWILIO_AUTH_TOKEN", "")
            self.config.twilio_from_number = os.getenv("TWILIO_FROM_NUMBER", "")

        # Telegram
        if os.getenv("TELEGRAM_ENABLED"):
            self.config.telegram_enabled = True
            self.config.telegram_bot_token = os.getenv("TELEGRAM_BOT_TOKEN", "")

        # Discord
        if os.getenv("DISCORD_ENABLED"):
            self.config.discord_enabled = True
            self.config.discord_webhook_url = os.getenv("DISCORD_WEBHOOK_URL", "")

        # Slack
        if os.getenv("SLACK_ENABLED"):
            self.config.slack_enabled = True
            self.config.slack_webhook_url = os.getenv("SLACK_WEBHOOK_URL", "")

    def send_email(self, alert: Alert) -> bool:
        """
        Send email notification.

        Args:
            alert: Alert to send

        Returns:
            Success status
        """
        if not self.config.email_enabled:
            logger.debug("Email notifications disabled")
            return False

        if not self.config.email_addresses:
            logger.warning("No email addresses configured")
            return False

        try:
            import smtplib
            from email.mime.text import MIMEText
            from email.mime.multipart import MIMEMultipart

            # Create message
            msg = MIMEMultipart()
            msg['From'] = self.config.smtp_username
            msg['Subject'] = f"[{alert.priority.value.upper()}] {alert.title}"

            # HTML body
            html = f"""
            <html>
                <body>
                    <h2>{alert.title}</h2>
                    <p><strong>Priority:</strong> {alert.priority.value.upper()}</p>
                    <p><strong>Time:</strong> {alert.timestamp.strftime('%Y-%m-%d %H:%M:%S')}</p>
                    <hr>
                    <p>{alert.message}</p>
                </body>
            </html>
            """
            msg.attach(MIMEText(html, 'html'))

            # Send to each recipient
            with smtplib.SMTP(self.config.smtp_server, self.config.smtp_port) as server:
                server.starttls()
                server.login(self.config.smtp_username, self.config.smtp_password)

                for email in self.config.email_addresses:
                    msg['To'] = email
                    server.send_message(msg)
                    logger.info(f"Email sent to {email}")

            return True

        except Exception as e:
            logger.error(f"Failed to send email: {e}")
            return False

    def send_sms(self, alert: Alert) -> bool:
        """
        Send SMS notification via Twilio.

        Args:
            alert: Alert to send

        Returns:
            Success status
        """
        if not self.config.sms_enabled:
            logger.debug("SMS notifications disabled")
            return False

        if not self.config.phone_numbers:
            logger.warning("No phone numbers configured")
            return False

        try:
            from twilio.rest import Client

            client = Client(
                self.config.twilio_account_sid,
                self.config.twilio_auth_token
            )

            # Create message text
            text = f"[{alert.priority.value.upper()}] {alert.title}\n\n{alert.message}"

            # Truncate if too long (SMS limit)
            if len(text) > 160:
                text = text[:157] + "..."

            # Send to each number
            for phone in self.config.phone_numbers:
                message = client.messages.create(
                    body=text,
                    from_=self.config.twilio_from_number,
                    to=phone
                )
                logger.info(f"SMS sent to {phone}: {message.sid}")

            return True

        except ImportError:
            logger.error("Twilio not installed. Install with: pip install twilio")
            return False
        except Exception as e:
            logger.error(f"Failed to send SMS: {e}")
            return False

    def send_telegram(self, alert: Alert) -> bool:
        """
        Send Telegram notification.

        Args:
            alert: Alert to send

        Returns:
            Success status
        """
        if not self.config.telegram_enabled:
            logger.debug("Telegram notifications disabled")
            return False

        if not self.config.telegram_chat_ids:
            logger.warning("No Telegram chat IDs configured")
            return False

        try:
            import requests

            # Format message with Markdown
            text = f"**[{alert.priority.value.upper()}] {alert.title}**\n\n{alert.message}\n\n_Time: {alert.timestamp.strftime('%H:%M:%S')}_"

            # Send to each chat
            for chat_id in self.config.telegram_chat_ids:
                url = f"https://api.telegram.org/bot{self.config.telegram_bot_token}/sendMessage"
                data = {
                    "chat_id": chat_id,
                    "text": text,
                    "parse_mode": "Markdown"
                }

                response = requests.post(url, json=data)
                response.raise_for_status()
                logger.info(f"Telegram message sent to chat {chat_id}")

            return True

        except ImportError:
            logger.error("requests not installed. Install with: pip install requests")
            return False
        except Exception as e:
            logger.error(f"Failed to send Telegram message: {e}")
            return False

    def send_discord(self, alert: Alert) -> bool:
        """
        Send Discord webhook notification.

        Args:
            alert: Alert to send

        Returns:
            Success status
        """
        if not self.config.discord_enabled:
            logger.debug("Discord notifications disabled")
            return False

        try:
            import requests

            # Color based on priority
            colors = {
                AlertPriority.LOW: 0x00FF00,  # Green
                AlertPriority.MEDIUM: 0xFFFF00,  # Yellow
                AlertPriority.HIGH: 0xFF9900,  # Orange
                AlertPriority.CRITICAL: 0xFF0000  # Red
            }

            # Create embed
            embed = {
                "title": alert.title,
                "description": alert.message,
                "color": colors.get(alert.priority, 0x00FF00),
                "timestamp": alert.timestamp.isoformat(),
                "fields": [
                    {
                        "name": "Priority",
                        "value": alert.priority.value.upper(),
                        "inline": True
                    }
                ]
            }

            data = {"embeds": [embed]}

            response = requests.post(self.config.discord_webhook_url, json=data)
            response.raise_for_status()
            logger.info("Discord webhook sent")

            return True

        except ImportError:
            logger.error("requests not installed")
            return False
        except Exception as e:
            logger.error(f"Failed to send Discord webhook: {e}")
            return False

    def send_slack(self, alert: Alert) -> bool:
        """
        Send Slack webhook notification.

        Args:
            alert: Alert to send

        Returns:
            Success status
        """
        if not self.config.slack_enabled:
            logger.debug("Slack notifications disabled")
            return False

        try:
            import requests

            # Create message
            data = {
                "text": f"*[{alert.priority.value.upper()}] {alert.title}*",
                "blocks": [
                    {
                        "type": "header",
                        "text": {
                            "type": "plain_text",
                            "text": alert.title
                        }
                    },
                    {
                        "type": "section",
                        "text": {
                            "type": "mrkdwn",
                            "text": alert.message
                        }
                    },
                    {
                        "type": "context",
                        "elements": [
                            {
                                "type": "mrkdwn",
                                "text": f"*Priority:* {alert.priority.value.upper()} | *Time:* {alert.timestamp.strftime('%H:%M:%S')}"
                            }
                        ]
                    }
                ]
            }

            response = requests.post(self.config.slack_webhook_url, json=data)
            response.raise_for_status()
            logger.info("Slack webhook sent")

            return True

        except ImportError:
            logger.error("requests not installed")
            return False
        except Exception as e:
            logger.error(f"Failed to send Slack webhook: {e}")
            return False

    def send_alert(self, alert: Alert) -> Dict[NotificationChannel, bool]:
        """
        Send alert to all configured channels.

        Args:
            alert: Alert to send

        Returns:
            Dict of channel -> success status
        """
        results = {}

        # Send to each requested channel
        for channel in alert.channels:
            if channel == NotificationChannel.EMAIL:
                results[channel] = self.send_email(alert)
            elif channel == NotificationChannel.SMS:
                results[channel] = self.send_sms(alert)
            elif channel == NotificationChannel.TELEGRAM:
                results[channel] = self.send_telegram(alert)
            elif channel == NotificationChannel.DISCORD:
                results[channel] = self.send_discord(alert)
            elif channel == NotificationChannel.SLACK:
                results[channel] = self.send_slack(alert)

        # Store in history
        self.alert_history.append(alert)

        # Log results
        successful = sum(1 for success in results.values() if success)
        logger.info(f"Alert sent: {successful}/{len(results)} channels successful")

        return results

    def create_price_alert(
        self,
        symbol: str,
        price: float,
        threshold: float,
        direction: str
    ) -> Alert:
        """Create price alert."""
        return Alert(
            title=f"Price Alert: {symbol}",
            message=f"{symbol} price {direction} ${threshold} (current: ${price})",
            priority=AlertPriority.MEDIUM,
            channels=[NotificationChannel.TELEGRAM, NotificationChannel.DISCORD]
        )

    def create_trade_alert(
        self,
        order_id: str,
        symbol: str,
        side: str,
        quantity: float,
        price: float
    ) -> Alert:
        """Create trade execution alert."""
        return Alert(
            title=f"Trade Executed: {symbol}",
            message=f"{side.upper()} {quantity} {symbol} @ ${price}\nOrder ID: {order_id}",
            priority=AlertPriority.HIGH,
            channels=[NotificationChannel.TELEGRAM, NotificationChannel.EMAIL]
        )

    def create_system_alert(
        self,
        message: str,
        priority: AlertPriority = AlertPriority.CRITICAL
    ) -> Alert:
        """Create system alert."""
        return Alert(
            title="System Alert",
            message=message,
            priority=priority,
            channels=[NotificationChannel.EMAIL, NotificationChannel.SMS, NotificationChannel.TELEGRAM]
        )


if __name__ == '__main__':
    import logging
    logging.basicConfig(level=logging.INFO)

    print("📢 Notification System Demo")
    print("=" * 60)

    # Create notification system
    config = AlertConfig()
    system = NotificationSystem(config)

    print("\n1. Testing notification channels...")
    print(f"   Email: {'✅ Enabled' if config.email_enabled else '❌ Disabled'}")
    print(f"   SMS: {'✅ Enabled' if config.sms_enabled else '❌ Disabled'}")
    print(f"   Telegram: {'✅ Enabled' if config.telegram_enabled else '❌ Enabled'}")
    print(f"   Discord: {'✅ Enabled' if config.discord_enabled else '❌ Disabled'}")
    print(f"   Slack: {'✅ Enabled' if config.slack_enabled else '❌ Disabled'}")

    # Create sample alerts
    print("\n2. Creating sample alerts...")

    # Price alert
    price_alert = system.create_price_alert(
        symbol="ETH",
        price=2100.50,
        threshold=2100.00,
        direction="above"
    )
    print(f"\n   Price Alert:")
    print(f"   Title: {price_alert.title}")
    print(f"   Message: {price_alert.message}")
    print(f"   Priority: {price_alert.priority.value}")
    print(f"   Channels: {[c.value for c in price_alert.channels]}")

    # Trade alert
    trade_alert = system.create_trade_alert(
        order_id="ORDER-12345",
        symbol="BTC",
        side="buy",
        quantity=0.5,
        price=45000.00
    )
    print(f"\n   Trade Alert:")
    print(f"   Title: {trade_alert.title}")
    print(f"   Message: {trade_alert.message}")

    # System alert
    system_alert = system.create_system_alert(
        message="System startup complete. All components operational.",
        priority=AlertPriority.LOW
    )
    print(f"\n   System Alert:")
    print(f"   Title: {system_alert.title}")
    print(f"   Priority: {system_alert.priority.value}")

    print("\n3. Alert History:")
    print(f"   Total alerts created: {len(system.alert_history)}")

    print("\n✅ Notification system demo complete!")
    print("\nTo enable notifications, set environment variables:")
    print("   EMAIL_ENABLED=true")
    print("   SMTP_USERNAME=your@email.com")
    print("   SMTP_PASSWORD=your_password")
    print("   TELEGRAM_ENABLED=true")
    print("   TELEGRAM_BOT_TOKEN=your_bot_token")
    print("   DISCORD_ENABLED=true")
    print("   DISCORD_WEBHOOK_URL=your_webhook_url")
    print("\nOr configure via AlertConfig in code")
