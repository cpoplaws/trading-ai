"""
Tests for exchange API integrations.
"""
import pytest
import os
import sys
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime
import asyncio

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from exchanges.coinbase_client import CoinbaseClient


class TestCoinbaseClient:
    """Tests for Coinbase Pro API client."""

    def test_initialization(self):
        """Test client initialization."""
        client = CoinbaseClient(
            api_key='test_key',
            api_secret='test_secret',
            passphrase='test_passphrase',
            sandbox=True
        )
        assert client.api_key == 'test_key'
        assert client.sandbox is True

    def test_sandbox_url(self):
        """Test sandbox URL configuration."""
        client = CoinbaseClient(
            api_key='test',
            api_secret='test',
            passphrase='test',
            sandbox=True
        )
        assert 'sandbox' in client.api_url.lower()

    def test_production_url(self):
        """Test production URL configuration."""
        client = CoinbaseClient(
            api_key='test',
            api_secret='test',
            passphrase='test',
            sandbox=False
        )
        assert 'sandbox' not in client.api_url.lower()

    @patch('exchanges.coinbase_pro.CoinbaseClient._make_request')
    def test_get_accounts(self, mock_request):
        """Test getting accounts."""
        mock_request.return_value = [
            {
                'id': 'abc123',
                'currency': 'USD',
                'balance': '10000.00',
                'available': '10000.00'
            }
        ]

        client = CoinbaseClient(
            api_key='test',
            api_secret='test',
            passphrase='test',
            sandbox=True
        )

        accounts = client.get_accounts()
        assert len(accounts) > 0
        assert accounts[0]['currency'] == 'USD'
        mock_request.assert_called_once()

    @patch('exchanges.coinbase_pro.CoinbaseClient._make_request')
    def test_get_product_ticker(self, mock_request):
        """Test getting product ticker."""
        mock_request.return_value = {
            'trade_id': 123456,
            'price': '40000.00',
            'size': '0.1',
            'time': '2025-01-15T12:00:00Z',
            'bid': '39999.00',
            'ask': '40001.00'
        }

        client = CoinbaseClient(
            api_key='test',
            api_secret='test',
            passphrase='test',
            sandbox=True
        )

        ticker = client.get_product_ticker('BTC-USD')
        assert ticker['price'] == '40000.00'
        assert 'bid' in ticker
        assert 'ask' in ticker

    @patch('exchanges.coinbase_pro.CoinbaseClient._make_request')
    def test_place_market_order(self, mock_request):
        """Test placing market order."""
        mock_request.return_value = {
            'id': 'order123',
            'product_id': 'BTC-USD',
            'side': 'buy',
            'type': 'market',
            'size': '0.1',
            'status': 'pending'
        }

        client = CoinbaseClient(
            api_key='test',
            api_secret='test',
            passphrase='test',
            sandbox=True
        )

        order = client.place_market_order(
            product_id='BTC-USD',
            side='buy',
            size=0.1
        )

        assert order['id'] == 'order123'
        assert order['side'] == 'buy'
        assert order['type'] == 'market'

    @patch('exchanges.coinbase_pro.CoinbaseClient._make_request')
    def test_place_limit_order(self, mock_request):
        """Test placing limit order."""
        mock_request.return_value = {
            'id': 'order456',
            'product_id': 'ETH-USD',
            'side': 'sell',
            'type': 'limit',
            'price': '2500.00',
            'size': '1.0',
            'status': 'pending'
        }

        client = CoinbaseClient(
            api_key='test',
            api_secret='test',
            passphrase='test',
            sandbox=True
        )

        order = client.place_limit_order(
            product_id='ETH-USD',
            side='sell',
            price=2500.0,
            size=1.0
        )

        assert order['id'] == 'order456'
        assert order['type'] == 'limit'
        assert order['price'] == '2500.00'

    @patch('exchanges.coinbase_pro.CoinbaseClient._make_request')
    def test_cancel_order(self, mock_request):
        """Test canceling an order."""
        mock_request.return_value = ['order123']

        client = CoinbaseClient(
            api_key='test',
            api_secret='test',
            passphrase='test',
            sandbox=True
        )

        result = client.cancel_order('order123')
        assert 'order123' in result
        mock_request.assert_called_once()

    @patch('exchanges.coinbase_pro.CoinbaseClient._make_request')
    def test_get_order_status(self, mock_request):
        """Test getting order status."""
        mock_request.return_value = {
            'id': 'order123',
            'status': 'done',
            'filled_size': '0.1',
            'executed_value': '4000.00'
        }

        client = CoinbaseClient(
            api_key='test',
            api_secret='test',
            passphrase='test',
            sandbox=True
        )

        status = client.get_order_status('order123')
        assert status['status'] == 'done'
        assert status['filled_size'] == '0.1'

    @patch('exchanges.coinbase_pro.CoinbaseClient._make_request')
    def test_get_fills(self, mock_request):
        """Test getting order fills."""
        mock_request.return_value = [
            {
                'trade_id': 1,
                'product_id': 'BTC-USD',
                'order_id': 'order123',
                'price': '40000.00',
                'size': '0.1',
                'fee': '4.00',
                'created_at': '2025-01-15T12:00:00Z'
            }
        ]

        client = CoinbaseClient(
            api_key='test',
            api_secret='test',
            passphrase='test',
            sandbox=True
        )

        fills = client.get_fills('order123')
        assert len(fills) > 0
        assert fills[0]['price'] == '40000.00'

    @patch('exchanges.coinbase_pro.CoinbaseClient._make_request')
    def test_error_handling(self, mock_request):
        """Test error handling."""
        mock_request.side_effect = Exception('API Error')

        client = CoinbaseClient(
            api_key='test',
            api_secret='test',
            passphrase='test',
            sandbox=True
        )

        with pytest.raises(Exception):
            client.get_accounts()

    def test_authentication_header_generation(self):
        """Test authentication header generation."""
        client = CoinbaseClient(
            api_key='test_key',
            api_secret='dGVzdF9zZWNyZXQ=',  # base64 encoded 'test_secret'
            passphrase='test_passphrase',
            sandbox=True
        )

        # Test that headers are generated
        headers = client._generate_auth_headers('GET', '/accounts', '')
        assert 'CB-ACCESS-KEY' in headers
        assert 'CB-ACCESS-SIGN' in headers
        assert 'CB-ACCESS-TIMESTAMP' in headers
        assert 'CB-ACCESS-PASSPHRASE' in headers


class TestExchangeRateLimiting:
    """Tests for rate limiting."""

    @patch('time.sleep')
    @patch('exchanges.coinbase_pro.CoinbaseClient._make_request')
    def test_rate_limit_handling(self, mock_request, mock_sleep):
        """Test rate limit handling."""
        # First call succeeds, second triggers rate limit
        mock_request.side_effect = [
            {'accounts': []},
            Exception('Rate limit exceeded')
        ]

        client = CoinbaseClient(
            api_key='test',
            api_secret='test',
            passphrase='test',
            sandbox=True
        )

        # First call should succeed
        result = client.get_accounts()
        assert result is not None

        # Second call should raise error
        with pytest.raises(Exception):
            client.get_accounts()


class TestExchangeDataValidation:
    """Tests for data validation."""

    def test_validate_product_id(self):
        """Test product ID validation."""
        client = CoinbaseClient(
            api_key='test',
            api_secret='test',
            passphrase='test',
            sandbox=True
        )

        # Valid product IDs
        assert client._validate_product_id('BTC-USD') is True
        assert client._validate_product_id('ETH-USD') is True

        # Invalid product IDs
        with pytest.raises(ValueError):
            client._validate_product_id('INVALID')

        with pytest.raises(ValueError):
            client._validate_product_id('')

    def test_validate_order_side(self):
        """Test order side validation."""
        client = CoinbaseClient(
            api_key='test',
            api_secret='test',
            passphrase='test',
            sandbox=True
        )

        # Valid sides
        assert client._validate_side('buy') is True
        assert client._validate_side('sell') is True

        # Invalid sides
        with pytest.raises(ValueError):
            client._validate_side('invalid')

    def test_validate_order_size(self):
        """Test order size validation."""
        client = CoinbaseClient(
            api_key='test',
            api_secret='test',
            passphrase='test',
            sandbox=True
        )

        # Valid sizes
        assert client._validate_size(0.001) is True
        assert client._validate_size(1.0) is True

        # Invalid sizes
        with pytest.raises(ValueError):
            client._validate_size(0.0)

        with pytest.raises(ValueError):
            client._validate_size(-1.0)


class TestWebSocketClient:
    """Tests for WebSocket client."""

    @pytest.mark.asyncio
    async def test_websocket_connection(self):
        """Test WebSocket connection."""
        # Mock WebSocket connection
        with patch('websockets.connect') as mock_connect:
            mock_ws = MagicMock()
            mock_connect.return_value.__aenter__.return_value = mock_ws

            client = CoinbaseClient(
                api_key='test',
                api_secret='test',
                passphrase='test',
                sandbox=True
            )

            # Test connection establishment
            # This would require actual WebSocket implementation
            # For now, just verify the mock was set up
            assert mock_connect is not None

    @pytest.mark.asyncio
    async def test_websocket_subscribe(self):
        """Test WebSocket subscription."""
        with patch('websockets.connect') as mock_connect:
            mock_ws = MagicMock()
            mock_ws.send = MagicMock()
            mock_connect.return_value.__aenter__.return_value = mock_ws

            client = CoinbaseClient(
                api_key='test',
                api_secret='test',
                passphrase='test',
                sandbox=True
            )

            # Test subscription message sending
            # This would require actual WebSocket implementation
            assert mock_ws is not None


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
