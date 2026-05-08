import logging
import time

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

        return self._klines_to_dataframe(payload)

    def get_historical_klines(self, symbol: str, interval: str, total_limit: int = 1000) -> pd.DataFrame:
        remaining = total_limit
        end_time = None
        chunks = []

        while remaining > 0:
            request_limit = min(1000, remaining)
            url = f"{self.base_url}/api/v3/klines"
            params = {"symbol": symbol, "interval": interval, "limit": request_limit}
            if end_time is not None:
                params["endTime"] = end_time

            try:
                response = requests.get(url, params=params, timeout=REQUEST_TIMEOUT_SECONDS)
                response.raise_for_status()
                payload = response.json()
            except requests.RequestException as exc:
                logger.error("Binance historical request failed for %s %s: %s", symbol, interval, exc)
                raise

            if not payload:
                break

            chunks = payload + chunks
            remaining -= len(payload)
            first_open_time = int(payload[0][0])
            next_end_time = first_open_time - 1
            if end_time == next_end_time:
                break
            end_time = next_end_time
            time.sleep(0.15)

            if len(payload) < request_limit:
                break

        return self._klines_to_dataframe(chunks).drop_duplicates(subset=["open_time"]).sort_values("open_time").reset_index(drop=True)

    @staticmethod
    def _klines_to_dataframe(payload: list) -> pd.DataFrame:
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
