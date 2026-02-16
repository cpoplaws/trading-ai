# Multi-Channel Alerting Setup Guide

Complete guide for setting up Telegram, Discord, and Slack alerts for the Trading AI system.

## Overview

The alerting system sends notifications for:
- Trade executions
- Whale activity
- Arbitrage opportunities
- Funding rate alerts
- System errors and critical events

## Supported Channels

- **Telegram**: Bot-based messaging
- **Discord**: Webhook-based notifications
- **Slack**: Webhook-based notifications

## Setup Instructions

### 1. Telegram Bot Setup

#### Create a Telegram Bot

1. Open Telegram and search for `@BotFather`
2. Send `/newbot` command
3. Follow prompts to name your bot (e.g., "Trading AI Bot")
4. Copy the **Bot Token** (looks like `123456789:ABCdefGHIjklMNOpqrsTUVwxyz`)

#### Get Your Chat ID

1. Search for `@userinfobot` in Telegram
2. Start a conversation - it will send you your **Chat ID**
3. Alternatively, send a message to your bot, then visit:
   ```
   https://api.telegram.org/bot<YOUR_BOT_TOKEN>/getUpdates
   ```
   Look for `"chat":{"id":YOUR_CHAT_ID}` in the JSON response

#### Add to .env

```bash
TELEGRAM_BOT_TOKEN=123456789:ABCdefGHIjklMNOpqrsTUVwxyz
TELEGRAM_CHAT_ID=987654321
```

### 2. Discord Webhook Setup

#### Create a Discord Webhook

1. Open Discord and go to your server
2. Right-click on the channel where you want alerts
3. Select "Edit Channel" → "Integrations" → "Webhooks"
4. Click "New Webhook" or "Create Webhook"
5. Name it (e.g., "Trading AI Alerts")
6. Copy the **Webhook URL**

#### Add to .env

```bash
DISCORD_WEBHOOK_URL=https://discord.com/api/webhooks/123456789/abcdefghijklmnopqrstuvwxyz
```

### 3. Slack Webhook Setup

#### Create a Slack App

1. Go to https://api.slack.com/apps
2. Click "Create New App" → "From scratch"
3. Name your app (e.g., "Trading AI") and select your workspace
4. In the app settings, go to "Incoming Webhooks"
5. Toggle "Activate Incoming Webhooks" to ON
6. Click "Add New Webhook to Workspace"
7. Select the channel for alerts
8. Copy the **Webhook URL**

#### Add to .env

```bash
SLACK_WEBHOOK_URL=https://hooks.slack.com/services/T00000000/B00000000/XXXXXXXXXXXXXXXXXXXX
```

## Complete .env Configuration

Your `.env` file should contain:

```bash
# ===== ALERTS =====
TELEGRAM_BOT_TOKEN=your_telegram_bot_token
TELEGRAM_CHAT_ID=your_telegram_chat_id
DISCORD_WEBHOOK_URL=your_discord_webhook_url
SLACK_WEBHOOK_URL=your_slack_webhook_url
```

## Testing Alerts

### Test from Python

```python
from src.infrastructure.alerting import AlertingSystem

# Initialize alerting system
alerting = AlertingSystem()

# Test basic alert
alerting.send_alert("Test alert from Trading AI", level='info')

# Test trade alert
trade_info = {
    'symbol': 'BTC-USDC',
    'action': 'BUY',
    'price': 40000.00,
    'size': 0.1,
    'strategy': 'market_making'
}
alerting.send_trade_alert(trade_info)

# Test whale alert
whale_info = {
    'wallet_label': 'Whale #1234',
    'direction': 'outflow',
    'value_usd': 5000000.00
}
alerting.send_whale_alert(whale_info)

# Test error alert
alerting.send_error_alert("Test error message", context={'module': 'test'})
```

### Test from Command Line

```bash
# Test alerting system
python -m src.infrastructure.alerting

# Output will show which channels are enabled and send a test alert
```

## Alert Types

### Trade Execution Alert

```
🔔 Trade Executed

Symbol: BTC-USDC
Action: BUY
Price: $40,000.00
Size: 0.1
Strategy: market_making
Time: 2026-02-15 22:30:00
```

### Whale Activity Alert

```
🐋 Whale Alert

Wallet: Whale #1234
Direction: OUTFLOW
Value: $5,000,000.00
Time: 2026-02-15 22:30:00
```

### Arbitrage Opportunity Alert

```
💰 Arbitrage Opportunity

Pair: ETH-USDC
Profit: 1.25%
Buy: Uniswap
Sell: Sushiswap
Time: 2026-02-15 22:30:00
```

### Funding Rate Alert

```
📊 High Funding Rate

Symbol: BTC-PERP
Funding Rate: 0.125%
Annual Return: 45.63%
Time: 2026-02-15 22:30:00
```

### Error Alert

```
❌ System Error

Error: Connection timeout to exchange API
Time: 2026-02-15 22:30:00
Context: {'exchange': 'binance', 'retry': 3}
```

## Rate Limiting

The alerting system includes rate limiting to prevent spam:
- **Maximum**: 20 alerts per hour
- **Behavior**: Additional alerts are dropped and logged

To adjust the rate limit, modify `max_alerts_per_hour` in `AlertingSystem.__init__()`.

## Integration with Autonomous Agent

The autonomous trading agent can use the alerting system:

```python
from src.infrastructure.alerting import AlertingSystem

class AutonomousTradingAgent:
    def __init__(self, config):
        self.alerting = AlertingSystem()
        # ... rest of initialization

    async def _execute_signals(self, signals):
        for signal in signals:
            # Execute trade
            trade_result = await self._execute_trade(signal)

            # Send alert
            self.alerting.send_trade_alert({
                'symbol': signal['symbol'],
                'action': signal['action'],
                'price': trade_result['price'],
                'size': trade_result['size'],
                'strategy': signal['strategy']
            })
```

## Troubleshooting

### Telegram Not Receiving Messages

1. Verify bot token is correct
2. Ensure you've started a conversation with the bot (send `/start`)
3. Verify chat ID is correct
4. Check bot has permission to send messages

### Discord Not Receiving Messages

1. Verify webhook URL is correct and not expired
2. Check channel permissions
3. Ensure webhook hasn't been deleted
4. Test webhook with curl:
   ```bash
   curl -H "Content-Type: application/json" \
        -d '{"content": "Test message"}' \
        YOUR_DISCORD_WEBHOOK_URL
   ```

### Slack Not Receiving Messages

1. Verify webhook URL is correct
2. Check app has permission to post in channel
3. Ensure webhook hasn't been revoked
4. Test webhook with curl:
   ```bash
   curl -X POST -H "Content-Type: application/json" \
        -d '{"text": "Test message"}' \
        YOUR_SLACK_WEBHOOK_URL
   ```

### Rate Limiting Issues

If you're hitting the rate limit:
1. Review alert frequency in your application
2. Increase `max_alerts_per_hour` if needed
3. Consider batching similar alerts
4. Use log files for detailed information

## Security Considerations

1. **Never commit .env file** to version control
2. **Rotate webhooks periodically** for security
3. **Use separate bots/webhooks** for production and development
4. **Monitor webhook access logs** in platform settings
5. **Revoke compromised webhooks** immediately

## Advanced Configuration

### Custom Alert Levels

Modify `_format_message()` to add custom alert levels:

```python
emojis = {
    'info': 'ℹ️',
    'warning': '⚠️',
    'error': '❌',
    'critical': '🚨',
    'success': '✅',  # Add custom level
}
```

### Channel-Specific Routing

Send different alerts to different channels:

```python
# Trade alerts to Telegram only
alerting.send_trade_alert(trade_info, channels=['telegram'])

# Errors to all channels
alerting.send_error_alert(error_msg, channels=['telegram', 'discord', 'slack'])
```

### Custom Alert Types

Create custom alert methods:

```python
def send_liquidation_alert(self, liq_info: Dict) -> Dict[str, bool]:
    message = f"""
    ⚡ Liquidation Detected

    Symbol: {liq_info['symbol']}
    Size: ${liq_info['size_usd']:,.2f}
    Price: ${liq_info['price']:,.2f}
    """
    return self.send_alert(message, level='warning')
```

## Support

For issues with the alerting system:
1. Check logs in `logs/` directory
2. Verify environment variables are set
3. Test each channel individually
4. Review API documentation for platform-specific issues

## References

- [Telegram Bot API](https://core.telegram.org/bots/api)
- [Discord Webhooks](https://discord.com/developers/docs/resources/webhook)
- [Slack Incoming Webhooks](https://api.slack.com/messaging/webhooks)
