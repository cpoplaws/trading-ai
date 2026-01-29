import os
import sys
from pathlib import Path

import pytest

# Ensure src is on path
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))

from utils.api_keys import APIKeys


def test_api_keys_loads_from_dotenv(tmp_path, monkeypatch):
    dotenv_file = tmp_path / "test.env"
    dotenv_file.write_text(
        "\n".join(
            [
                "ALPACA_API_KEY=from_file",
                "ALPACA_SECRET_KEY=from_file_secret",
                "ALPHA_VANTAGE_API_KEY=alpha_file",
            ]
        )
    )

    # Ensure environment is clean so values come from the file
    monkeypatch.delenv("ALPACA_API_KEY", raising=False)
    monkeypatch.delenv("ALPACA_SECRET_KEY", raising=False)
    monkeypatch.delenv("ALPHA_VANTAGE_API_KEY", raising=False)

    keys = APIKeys.load(dotenv_path=dotenv_file, override=True)

    assert keys.alpaca_api_key == "from_file"
    assert keys.alpaca_secret_key == "from_file_secret"
    assert keys.alpha_vantage_api_key == "alpha_file"


def test_api_keys_prefers_environment_when_override_disabled(tmp_path, monkeypatch):
    dotenv_file = tmp_path / "test.env"
    dotenv_file.write_text("BINANCE_API_KEY=file_value")

    monkeypatch.setenv("BINANCE_API_KEY", "env_value")

    keys = APIKeys.load(dotenv_path=dotenv_file, override=False)

    assert keys.binance_api_key == "env_value"
