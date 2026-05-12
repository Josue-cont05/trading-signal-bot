import logging
import time
from datetime import datetime, timezone

from config.settings import SYMBOLS
from services.log_service import add_log
from services.twelvedata_service import TwelveDataService


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
            payload = _fetch_price(symbol)
            last_price = float(payload["price"])
            change_percent = 0.0
            market.append(
                {
                    "symbol": symbol,
                    "price": last_price,
                    "change_percent": change_percent,
                    "direction": "bullish" if change_percent >= 0 else "bearish",
                    "updated_at": datetime.now(timezone.utc).isoformat(),
                    "source": "twelvedata",
                }
            )
        except Exception as exc:
            logger.warning("Market fallback for %s: %s", symbol, exc)
            add_log("market fallback", f"TwelveData unavailable for {symbol}: {exc}", "warning", symbol)
            market.append(_fallback_market(symbol))

    _cache["data"] = market
    _cache["expires_at"] = now + ttl_seconds
    return market


def _fetch_price(symbol: str) -> dict:
    service = TwelveDataService()
    price = service.get_price(symbol)
    return {"price": price}


def _fallback_market(symbol: str) -> dict:
    return {
        "symbol": symbol,
        "price": 0.0,
        "change_percent": 0.0,
        "direction": "neutral",
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "source": "fallback",
    }
