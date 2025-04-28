import alpaca_trade_api as tradeapi

class BrokerInterface:
    def __init__(self, api_key: str, secret_key: str, base_url: str):
        self.api_key = api_key
        self.secret_key = secret_key
        self.base_url = base_url
        self.api = None

    def connect(self):
        try:
            self.api = tradeapi.REST(self.api_key, self.secret_key, self.base_url)
            # Verify connection
            account = self.api.get_account()
            print(f"Connected to Alpaca Paper API. Account status: {account.status}")
        except Exception as e:
            print(f"Failed to connect to Alpaca Paper API: {e}")

# Example usage
if __name__ == "__main__":
    API_KEY = "your_api_key"
    SECRET_KEY = "your_secret_key"
    BASE_URL = "https://paper-api.alpaca.markets"

    broker = BrokerInterface(API_KEY, SECRET_KEY, BASE_URL)
    broker.connect()