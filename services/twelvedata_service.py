import logging
import math
import os
import time
from datetime import datetime

import pandas as pd
import requests


logger = logging.getLogger(__name__)
HISTORICAL_REQUEST_DELAY_SECONDS = 8
MAX_HISTORICAL_CHUNK_SIZE = 200

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
    "SPX500": "SPY",
    "NAS100": "QQQ",
    "US30": "DIA",
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
        payload = self._request_time_series(symbol, interval, limit, timezone="UTC")
        return self._time_series_to_dataframe(payload, interval)

    def get_historical_klines(self, symbol: str, interval: str, total_limit: int = 1000) -> pd.DataFrame:
        remaining = total_limit
        end_date = None
        chunks = []
        previous_oldest_time = None
        max_requests = math.ceil(total_limit / MAX_HISTORICAL_CHUNK_SIZE) + 2

        for request_index in range(1, max_requests + 1):
            if remaining <= 0:
                break

            chunk_size = min(MAX_HISTORICAL_CHUNK_SIZE, remaining)
            df = self._get_historical_chunk(
                symbol=symbol,
                interval=interval,
                limit=chunk_size,
                end_date=end_date,
            )

            if df.empty:
                logger.warning(
                    "Historical download %s %s: chunk %s returned no candles.",
                    symbol,
                    interval,
                    request_index,
                )
                break

            oldest_time = df["open_time"].min()
            if previous_oldest_time is not None and oldest_time >= previous_oldest_time:
                logger.warning(
                    "Historical download %s %s stopped because pagination did not move backwards.",
                    symbol,
                    interval,
                )
                break

            logger.info(
                "Historical download %s %s: chunk %s returned %s candles.",
                symbol,
                interval,
                request_index,
                len(df),
            )
            chunks.append(df)

            previous_oldest_time = oldest_time
            end_date = oldest_time - pd.Timedelta(seconds=1)
            remaining -= len(df)

            if len(df) < chunk_size:
                logger.info(
                    "Historical download %s %s reached the available history after %s candles.",
                    symbol,
                    interval,
                    sum(len(chunk) for chunk in chunks),
                )
                break

            if remaining > 0 and request_index < max_requests:
                time.sleep(HISTORICAL_REQUEST_DELAY_SECONDS)

        if remaining > 0 and len(chunks) >= max_requests:
            logger.warning(
                "Historical download %s %s stopped after reaching max_requests=%s.",
                symbol,
                interval,
                max_requests,
            )

        if not chunks:
            return self._empty_klines_dataframe()

        result = (
            pd.concat(chunks, ignore_index=True)
            .drop_duplicates(subset=["open_time"])
            .sort_values("open_time")
            .reset_index(drop=True)
        )
        result = result.tail(total_limit).reset_index(drop=True)

        if not result.empty:
            logger.info(
                "Historical download completed for %s %s: %s candles. Range: %s -> %s",
                symbol,
                interval,
                len(result),
                result.iloc[0]["open_time"],
                result.iloc[-1]["open_time"],
            )

        return result

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

    def _get_historical_chunk(
        self,
        symbol: str,
        interval: str,
        limit: int = MAX_HISTORICAL_CHUNK_SIZE,
        end_date: datetime | None = None,
    ) -> pd.DataFrame:
        payload = self._request_time_series(
            symbol=symbol,
            interval=interval,
            output_size=min(limit, MAX_HISTORICAL_CHUNK_SIZE),
            end_date=end_date,
            timezone="UTC",
        )
        if not payload.get("values"):
            return self._empty_klines_dataframe()
        return self._time_series_to_dataframe(payload, interval)

    def _request_time_series(
        self,
        symbol: str,
        interval: str,
        output_size: int,
        end_date: datetime | None = None,
        timezone: str | None = None,
    ) -> dict:
        mapped_symbol = self._map_symbol(symbol)
        mapped_interval = self._map_interval(interval)
        url = f"{self.base_url}/time_series"
        params = {
            "symbol": mapped_symbol,
            "interval": mapped_interval,
            "outputsize": output_size,
            "apikey": self.api_key,
        }
        if end_date is not None:
            params["end_date"] = pd.Timestamp(end_date).strftime("%Y-%m-%d %H:%M:%S")
        if timezone is not None:
            params["timezone"] = timezone

        try:
            response = requests.get(url, params=params, timeout=self.timeout)
            response.raise_for_status()
            payload = response.json()
        except requests.RequestException as exc:
            logger.error("Twelve Data request failed for %s %s: %s", symbol, interval, exc)
            raise

        self._raise_for_api_error(payload, require_ok=True)
        return payload

    @staticmethod
    def _empty_klines_dataframe() -> pd.DataFrame:
        return pd.DataFrame(
            {
                "open": pd.Series(dtype="float64"),
                "high": pd.Series(dtype="float64"),
                "low": pd.Series(dtype="float64"),
                "close": pd.Series(dtype="float64"),
                "volume": pd.Series(dtype="float64"),
                "open_time": pd.Series(dtype="datetime64[ns, UTC]"),
                "close_time": pd.Series(dtype="datetime64[ns, UTC]"),
            }
        )

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
