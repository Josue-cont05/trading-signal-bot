import logging

import pandas as pd
import requests

from config.settings import BINANCE_BASE_URL, REQUEST_TIMEOUT_SECONDS


logger = logging.getLogger(__name__)


class BinanceService:
    def __init__(self, base_url: str = BINANCE_BASE_URL) -> None:
        self.base_url = base_url.rstrip("/")

    def get_klines(self, symbol: str, interval: str, limit: int = 200) -> pd.DataFrame:
        url = f"{self.base_url}/api/v3/klines"
        params = {"symbol": symbol, "interval": interval, "limit": limit}

        try:
            response = requests.get(url, params=params, timeout=REQUEST_TIMEOUT_SECONDS)
            response.raise_for_status()
            payload = response.json()
        except requests.RequestException as exc:
            logger.error("Binance request failed for %s %s: %s", symbol, interval, exc)
            raise

        columns = [
            "open_time",
            "open",
            "high",
            "low",
            "close",
            "volume",
            "close_time",
            "quote_asset_volume",
            "number_of_trades",
            "taker_buy_base_asset_volume",
            "taker_buy_quote_asset_volume",
            "ignore",
        ]
        df = pd.DataFrame(payload, columns=columns)

        numeric_columns = ["open", "high", "low", "close", "volume"]
        df[numeric_columns] = df[numeric_columns].astype(float)
        df["open_time"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
        df["close_time"] = pd.to_datetime(df["close_time"], unit="ms", utc=True)

        return df

