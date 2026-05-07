import logging

import numpy as np
import pandas as pd

from config.settings import EMA_PROXIMITY_PERCENT, ENTRY_INTERVAL


logger = logging.getLogger(__name__)


def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    data = df.copy()
    data["ema_50"] = data["close"].ewm(span=50, adjust=False).mean()
    data["ema_200"] = data["close"].ewm(span=200, adjust=False).mean()
    data["rsi_14"] = calculate_rsi(data["close"], period=14)
    data["volume_avg_20"] = data["volume"].rolling(window=20).mean()
    return data


def calculate_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)

    avg_gain = gain.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    relative_strength = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + relative_strength))

    return rsi.fillna(50)


def evaluate_swing_long_v1(symbol: str, daily_df: pd.DataFrame, entry_df: pd.DataFrame) -> dict | None:
    if len(daily_df) < 200 or len(entry_df) < 50:
        logger.warning("Not enough candles to evaluate %s.", symbol)
        return None

    daily = add_indicators(daily_df).dropna()
    entry = add_indicators(entry_df).dropna()

    if daily.empty or entry.empty:
        logger.warning("Indicator calculation produced empty data for %s.", symbol)
        return None

    daily_last = daily.iloc[-1]
    entry_last = entry.iloc[-1]
    entry_prev = entry.iloc[-2]

    daily_uptrend = daily_last["ema_50"] > daily_last["ema_200"]
    price = float(entry_last["close"])
    ema_50_4h = float(entry_last["ema_50"])
    ema_distance_percent = abs(price - ema_50_4h) / ema_50_4h * 100
    price_near_ema = ema_distance_percent <= EMA_PROXIMITY_PERCENT

    rsi_current = float(entry_last["rsi_14"])
    rsi_previous = float(entry_prev["rsi_14"])
    rsi_in_range = 40 <= rsi_current <= 60
    rsi_recovering = rsi_current > rsi_previous

    volume_current = float(entry_last["volume"])
    volume_average = float(entry_last["volume_avg_20"])
    volume_above_average = volume_current > volume_average

    conditions = [
        daily_uptrend,
        price_near_ema,
        rsi_in_range,
        rsi_recovering,
        volume_above_average,
    ]

    if not all(conditions):
        logger.info(
            "%s skipped. trend=%s near_ema=%s rsi_range=%s rsi_up=%s volume=%s",
            symbol,
            daily_uptrend,
            price_near_ema,
            rsi_in_range,
            rsi_recovering,
            volume_above_average,
        )
        return None

    recent_low = float(entry.tail(10)["low"].min())
    stop_loss = recent_low
    risk = price - stop_loss

    if risk <= 0:
        logger.info("%s skipped because calculated risk is not positive.", symbol)
        return None

    take_profit_1 = price + (risk * 2)
    take_profit_2 = price + (risk * 3)

    return {
        "symbol": symbol,
        "timeframe": ENTRY_INTERVAL,
        "entry_price": round(price, 8),
        "stop_loss": round(stop_loss, 8),
        "take_profit_1": round(take_profit_1, 8),
        "take_profit_2": round(take_profit_2, 8),
        "risk_reward": "1:2 / 1:3",
        "status": "active",
        "reasons": [
            "Tendencia diaria alcista: EMA 50 > EMA 200",
            f"Precio cerca de EMA50 4H: distancia {ema_distance_percent:.2f}%",
            f"RSI 14 recuperándose: {rsi_previous:.2f} -> {rsi_current:.2f}",
            "Volumen actual superior al promedio de las últimas 20 velas",
        ],
    }

