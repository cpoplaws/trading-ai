"""
Alerting system for Telegram, Discord, and Slack notifications.
"""
import os
import logging
import requests
from typing import Dict, List, Optional, Any
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)


class AlertingSystem:
    """
    Multi-channel alerting system for crypto trading notifications.
    """
    
    def __init__(self):
        """Initialize alerting system."""
        # Telegram
        self.telegram_bot_token = os.getenv('TELEGRAM_BOT_TOKEN')
        self.telegram_chat_id = os.getenv('TELEGRAM_CHAT_ID')
        self.telegram_enabled = bool(self.telegram_bot_token and self.telegram_chat_id)
        
        # Discord
        self.discord_webhook = os.getenv('DISCORD_WEBHOOK_URL')
        self.discord_enabled = bool(self.discord_webhook)
        
        # Slack
        self.slack_webhook = os.getenv('SLACK_WEBHOOK_URL')
        self.slack_enabled = bool(self.slack_webhook)
        
        # Rate limiting
        self.alert_count = 0
        self.max_alerts_per_hour = 20
        
        logger.info(f"Alerting system initialized (Telegram: {self.telegram_enabled}, Discord: {self.discord_enabled}, Slack: {self.slack_enabled})")
    
    def send_alert(self, message: str, level: str = 'info', channels: Optional[List[str]] = None) -> Dict[str, bool]:
        """
        Send alert to configured channels.
        
        Args:
            message: Alert message
            level: Alert level (info, warning, error, critical)
            channels: Specific channels to send to (all if None)
            
        Returns:
            Dictionary of channel -> success status
        """
        results = {}
        
        # Rate limiting check
        if self.alert_count >= self.max_alerts_per_hour:
            logger.warning("Alert rate limit exceeded")
            return results
        
        # Format message with level
        formatted_message = self._format_message(message, level)
        
        # Determine channels
        channels = channels or ['telegram', 'discord', 'slack']
        
        # Send to each channel
        if 'telegram' in channels and self.telegram_enabled:
            results['telegram'] = self._send_telegram(formatted_message)
        
        if 'discord' in channels and self.discord_enabled:
            results['discord'] = self._send_discord(formatted_message, level)
        
        if 'slack' in channels and self.slack_enabled:
            results['slack'] = self._send_slack(formatted_message, level)
        
        self.alert_count += 1
        return results
    
    def send_trade_alert(self, trade_info: Dict) -> Dict[str, bool]:
        """
        Send trade execution alert.
        
        Args:
            trade_info: Trade information dictionary
            
        Returns:
            Dictionary of channel -> success status
        """
        symbol = trade_info.get('symbol', 'Unknown')
        action = trade_info.get('action', 'Unknown')
        price = trade_info.get('price', 0)
        size = trade_info.get('size', 0)
        strategy = trade_info.get('strategy', 'Unknown')
        
        message = f"""
🔔 **Trade Executed**

Symbol: {symbol}
Action: {action}
Price: ${price:,.2f}
Size: {size}
Strategy: {strategy}
Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
"""
        
        return self.send_alert(message, level='info')
    
    def send_whale_alert(self, whale_info: Dict) -> Dict[str, bool]:
        """
        Send whale activity alert.
        
        Args:
            whale_info: Whale activity information
            
        Returns:
            Dictionary of channel -> success status
        """
        label = whale_info.get('wallet_label', 'Unknown Whale')
        direction = whale_info.get('direction', 'unknown')
        value = whale_info.get('value_usd', 0)
        
        emoji = '🐋' if direction == 'outflow' else '🐳'
        
        message = f"""
{emoji} **Whale Alert**

Wallet: {label}
Direction: {direction.upper()}
Value: ${value:,.2f}
Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
"""
        
        return self.send_alert(message, level='warning')
    
    def send_arbitrage_alert(self, arb_info: Dict) -> Dict[str, bool]:
        """
        Send arbitrage opportunity alert.
        
        Args:
            arb_info: Arbitrage opportunity information
            
        Returns:
            Dictionary of channel -> success status
        """
        pair = arb_info.get('token_pair', 'Unknown')
        profit_pct = arb_info.get('profit_percent', 0)
        buy_dex = arb_info.get('buy_dex', 'Unknown')
        sell_dex = arb_info.get('sell_dex', 'Unknown')
        
        message = f"""
💰 **Arbitrage Opportunity**

Pair: {pair}
Profit: {profit_pct:.2f}%
Buy: {buy_dex}
Sell: {sell_dex}
Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
"""
        
        return self.send_alert(message, level='info')
    
    def send_funding_rate_alert(self, funding_info: Dict) -> Dict[str, bool]:
        """
        Send funding rate alert.
        
        Args:
            funding_info: Funding rate information
            
        Returns:
            Dictionary of channel -> success status
        """
        symbol = funding_info.get('symbol', 'Unknown')
        rate = funding_info.get('funding_rate_percent', 0)
        annual_return = funding_info.get('expected_profit_annual', 0)
        
        message = f"""
📊 **High Funding Rate**

Symbol: {symbol}
Funding Rate: {rate:.3f}%
Annual Return: {annual_return:.2f}%
Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
"""
        
        return self.send_alert(message, level='info')
    
    def send_error_alert(self, error_message: str, context: Optional[Dict] = None) -> Dict[str, bool]:
        """
        Send error alert.
        
        Args:
            error_message: Error description
            context: Additional context information
            
        Returns:
            Dictionary of channel -> success status
        """
        message = f"""
❌ **System Error**

Error: {error_message}
Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
"""
        
        if context:
            message += f"\nContext: {context}"
        
        return self.send_alert(message, level='error')
    
    def _format_message(self, message: str, level: str) -> str:
        """Format message with level emoji."""
        emojis = {
            'info': 'ℹ️',
            'warning': '⚠️',
            'error': '❌',
            'critical': '🚨'
        }
        
        emoji = emojis.get(level, 'ℹ️')
        return f"{emoji} {message}"
    
    def _send_telegram(self, message: str) -> bool:
        """Send message via Telegram."""
        try:
            url = f"https://api.telegram.org/bot{self.telegram_bot_token}/sendMessage"
            payload = {
                'chat_id': self.telegram_chat_id,
                'text': message,
                'parse_mode': 'Markdown'
            }
            response = requests.post(url, json=payload, timeout=10)
            response.raise_for_status()
            logger.info("Telegram alert sent")
            return True
        except Exception as e:
            logger.error(f"Error sending Telegram alert: {e}")
            return False
    
    def _send_discord(self, message: str, level: str) -> bool:
        """Send message via Discord webhook."""
        try:
            # Discord color codes
            colors = {
                'info': 3447003,      # Blue
                'warning': 16776960,  # Yellow
                'error': 15158332,    # Red
                'critical': 10038562  # Dark red
            }
            
            payload = {
                'embeds': [{
                    'description': message,
                    'color': colors.get(level, 3447003),
                    'timestamp': datetime.utcnow().isoformat()
                }]
            }
            
            response = requests.post(self.discord_webhook, json=payload, timeout=10)
            response.raise_for_status()
            logger.info("Discord alert sent")
            return True
        except Exception as e:
            logger.error(f"Error sending Discord alert: {e}")
            return False
    
    def _send_slack(self, message: str, level: str) -> bool:
        """Send message via Slack webhook."""
        try:
            # Slack color codes
            colors = {
                'info': 'good',
                'warning': 'warning',
                'error': 'danger',
                'critical': 'danger'
            }
            
            payload = {
                'attachments': [{
                    'text': message,
                    'color': colors.get(level, 'good'),
                    'ts': int(datetime.now().timestamp())
                }]
            }
            
            response = requests.post(self.slack_webhook, json=payload, timeout=10)
            response.raise_for_status()
            logger.info("Slack alert sent")
            return True
        except Exception as e:
            logger.error(f"Error sending Slack alert: {e}")
            return False


if __name__ == "__main__":
    # Test alerting system
    alerting = AlertingSystem()
    
    print("=== Alerting System Test ===")
    print(f"Telegram: {'✅ Enabled' if alerting.telegram_enabled else '❌ Disabled'}")
    print(f"Discord: {'✅ Enabled' if alerting.discord_enabled else '❌ Disabled'}")
    print(f"Slack: {'✅ Enabled' if alerting.slack_enabled else '❌ Disabled'}")
    
    if alerting.telegram_enabled or alerting.discord_enabled or alerting.slack_enabled:
        print("\nSending test alert...")
        results = alerting.send_alert("Test alert from Trading-AI", level='info')
        print(f"Results: {results}")
    else:
        print("\n⚠️  No alert channels configured.")
        print("   Configure webhooks in .env to enable alerts.")
    
    print("\n✅ Alerting system test completed!")
