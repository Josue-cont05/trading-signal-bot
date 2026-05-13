import logging
import os

import pandas as pd
import requests


logger = logging.getLogger(__name__)

INTERVAL_MAP = {
    "1m": "1min",
    "5m": "5min",
    "15m": "15min",
    "30m": "30min",
    "1h": "1h",
    "4h": "4h",
    "1d": "1day",
}

SYMBOL_MAP = {
    "EURUSD": "EUR/USD",
    "GBPUSD": "GBP/USD",
    "XAUUSD": "XAU/USD",
    "SPX500": "SPX",
    "BTCUSDT": "BTC/USD",
    "ETHUSDT": "ETH/USD",
    "SOLUSDT": "SOL/USD",
    "USDCAD": "USD/CAD",
}


class TwelveDataService:
    def __init__(self) -> None:
        self.api_key = os.getenv("TWELVEDATA_API_KEY", "")
        self.base_url = "https://api.twelvedata.com"
        self.timeout = 15

    def get_klines(self, symbol: str, interval: str, limit: int = 200) -> pd.DataFrame:
        payload = self._request_time_series(symbol, interval, limit)
        return self._time_series_to_dataframe(payload, interval)

    def get_historical_klines(self, symbol: str, interval: str, total_limit: int = 1000) -> pd.DataFrame:
        output_size = min(total_limit, 5000)
        payload = self._request_time_series(symbol, interval, output_size)
        return self._time_series_to_dataframe(payload, interval)

    def get_price(self, symbol: str) -> float:
        mapped_symbol = self._map_symbol(symbol)
        url = f"{self.base_url}/price"
        params = {"symbol": mapped_symbol, "apikey": self.api_key}

        try:
            response = requests.get(url, params=params, timeout=self.timeout)
            response.raise_for_status()
            payload = response.json()
        except requests.RequestException as exc:
            logger.error("Twelve Data price request failed for %s: %s", symbol, exc)
            raise

        self._raise_for_api_error(payload)

        try:
            return float(payload["price"])
        except (KeyError, TypeError, ValueError) as exc:
            logger.error("Invalid Twelve Data price response for %s: %s", symbol, exc)
            raise ValueError(f"Invalid Twelve Data price response for {symbol}") from exc

    def _request_time_series(self, symbol: str, interval: str, output_size: int) -> dict:
        mapped_symbol = self._map_symbol(symbol)
        mapped_interval = self._map_interval(interval)
        url = f"{self.base_url}/time_series"
        params = {
            "symbol": mapped_symbol,
            "interval": mapped_interval,
            "outputsize": output_size,
            "apikey": self.api_key,
        }

        try:
            response = requests.get(url, params=params, timeout=self.timeout)
            response.raise_for_status()
            payload = response.json()
        except requests.RequestException as exc:
            logger.error("Twelve Data request failed for %s %s: %s", symbol, interval, exc)
            raise

        self._raise_for_api_error(payload, require_ok=True)
        return payload

    def _time_series_to_dataframe(self, payload: dict, interval: str) -> pd.DataFrame:
        values = payload.get("values")
        if not values:
            message = payload.get("message", "Twelve Data response does not include values.")
            raise ValueError(message)

        df = pd.DataFrame(values)
        for column in ["open", "high", "low", "close"]:
            df[column] = df[column].astype(float)

        if "volume" not in df.columns:
            df["volume"] = 0.0
        df["volume"] = pd.to_numeric(df["volume"], errors="coerce").fillna(0.0).astype(float)
        df["open_time"] = pd.to_datetime(df["datetime"], utc=True)
        df["close_time"] = df["open_time"] + self._interval_delta(interval)

        columns = ["open", "high", "low", "close", "volume", "open_time", "close_time"]
        return df[columns].sort_values("open_time").reset_index(drop=True)

    @staticmethod
    def _map_interval(interval: str) -> str:
        try:
            return INTERVAL_MAP[interval]
        except KeyError as exc:
            raise ValueError(f"Unsupported Twelve Data interval: {interval}") from exc

    @staticmethod
    def _map_symbol(symbol: str) -> str:
        return SYMBOL_MAP.get(symbol, symbol)

    @staticmethod
    def _interval_delta(interval: str) -> pd.Timedelta:
        deltas = {
            "1m": pd.Timedelta(minutes=1),
            "5m": pd.Timedelta(minutes=5),
            "15m": pd.Timedelta(minutes=15),
            "30m": pd.Timedelta(minutes=30),
            "1h": pd.Timedelta(hours=1),
            "4h": pd.Timedelta(hours=4),
            "1d": pd.Timedelta(days=1),
        }
        try:
            return deltas[interval]
        except KeyError as exc:
            raise ValueError(f"Unsupported Twelve Data interval: {interval}") from exc

    @staticmethod
    def _raise_for_api_error(payload: dict, require_ok: bool = False) -> None:
        status = payload.get("status")
        if status != "ok" and (status is not None or require_ok):
            message = payload.get("message", "Twelve Data API error.")
            logger.error("Twelve Data API error: %s", message)
            raise ValueError(message)
