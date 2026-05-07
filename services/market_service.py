import logging
import time
from datetime import datetime, timezone

import requests

from config.settings import BINANCE_BASE_URL, REQUEST_TIMEOUT_SECONDS, SYMBOLS
from services.log_service import add_log


logger = logging.getLogger(__name__)

_cache = {"expires_at": 0.0, "data": []}


def get_market_snapshot(symbols: list[str] | None = None, ttl_seconds: int = 45) -> list[dict]:
    now = time.time()
    if _cache["data"] and now < _cache["expires_at"]:
        return _cache["data"]

    symbols = symbols or SYMBOLS
    market = []

    for symbol in symbols:
        try:
            payload = _fetch_ticker(symbol)
            last_price = float(payload["lastPrice"])
            change_percent = float(payload["priceChangePercent"])
            market.append(
                {
                    "symbol": symbol,
                    "price": last_price,
                    "change_percent": change_percent,
                    "direction": "bullish" if change_percent >= 0 else "bearish",
                    "updated_at": datetime.now(timezone.utc).isoformat(),
                    "source": "binance",
                }
            )
        except Exception as exc:
            logger.warning("Market fallback for %s: %s", symbol, exc)
            add_log("market fallback", f"Binance data unavailable for {symbol}: {exc}", "warning", symbol)
            market.append(_fallback_market(symbol))

    _cache["data"] = market
    _cache["expires_at"] = now + ttl_seconds
    return market


def _fetch_ticker(symbol: str) -> dict:
    response = requests.get(
        f"{BINANCE_BASE_URL.rstrip('/')}/api/v3/ticker/24hr",
        params={"symbol": symbol},
        timeout=REQUEST_TIMEOUT_SECONDS,
    )
    response.raise_for_status()
    return response.json()


def _fallback_market(symbol: str) -> dict:
    fallback_prices = {
        "BTCUSDT": 0.0,
        "ETHUSDT": 0.0,
        "SOLUSDT": 0.0,
    }
    return {
        "symbol": symbol,
        "price": fallback_prices.get(symbol, 0.0),
        "change_percent": 0.0,
        "direction": "neutral",
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "source": "fallback",
    }

