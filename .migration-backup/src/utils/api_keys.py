"""
Utility helpers for loading API keys from environment variables or a .env file.

This centralizes secret loading to avoid hardcoding credentials in the codebase.
"""
import os
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Optional

from dotenv import load_dotenv

_DEFAULT_ENV_PATH = Path(__file__).resolve().parents[2] / ".env"


@dataclass
class APIKeys:
    """Container for API keys loaded from environment variables."""

    alpaca_api_key: Optional[str] = None
    alpaca_secret_key: Optional[str] = None
    alpaca_base_url: Optional[str] = None
    alpha_vantage_api_key: Optional[str] = None
    newsapi_api_key: Optional[str] = None
    reddit_client_id: Optional[str] = None
    reddit_client_secret: Optional[str] = None
    reddit_user_agent: Optional[str] = None
    binance_api_key: Optional[str] = None
    binance_secret_key: Optional[str] = None

    @classmethod
    def load(cls, dotenv_path: Optional[Path] = None, override: bool = False) -> "APIKeys":
        """
        Load API keys from the environment or a .env file.

        Args:
            dotenv_path: Optional path to a .env file. Defaults to repo root .env
            override: Whether values from the .env file should override existing env vars

        Returns:
            APIKeys: Populated APIKeys dataclass
        """
        env_path = Path(dotenv_path) if dotenv_path else _DEFAULT_ENV_PATH
        load_dotenv(env_path, override=override)

        return cls(
            alpaca_api_key=os.getenv("ALPACA_API_KEY"),
            alpaca_secret_key=os.getenv("ALPACA_SECRET_KEY"),
            alpaca_base_url=os.getenv("ALPACA_BASE_URL"),
            alpha_vantage_api_key=os.getenv("ALPHA_VANTAGE_API_KEY"),
            newsapi_api_key=os.getenv("NEWSAPI_API_KEY"),
            reddit_client_id=os.getenv("REDDIT_CLIENT_ID"),
            reddit_client_secret=os.getenv("REDDIT_CLIENT_SECRET"),
            reddit_user_agent=os.getenv("REDDIT_USER_AGENT"),
            binance_api_key=os.getenv("BINANCE_API_KEY"),
            binance_secret_key=os.getenv("BINANCE_SECRET_KEY"),
        )

    def as_dict(self) -> Dict[str, Optional[str]]:
        """Return API keys as a dictionary."""
        return asdict(self)
